"""model.py

Decoder-only (GPT-style) transformer for autoregressive MIDI token prediction.

Architecture:
    token embedding  (vocab_size → d_model)
    + learned positional embedding  (context_len → d_model)
    → N × TransformerBlock
        LayerNorm → MultiHeadSelfAttention (causal mask) → residual
        LayerNorm → FFN (d_model → 4*d_model → d_model, GELU) → residual
    → final LayerNorm
    → linear projection (d_model → vocab_size)

Default config targets a mid-range GPU (≈4-8 GB VRAM):
    d_model=512, n_heads=8, n_layers=6, context_len=1024  →  ~40 M params
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv   = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj  = nn.Linear(d_model, d_model,     bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in qkv]

        # Scaled dot-product attention with causal mask (uses Flash Attention
        # when available via torch.nn.functional.scaled_dot_product_attention)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class FFN(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = FFN(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MidiTransformer(nn.Module):
    def __init__(
        self,
        vocab_size:  int = 388,
        context_len: int = 1024,
        d_model:     int = 512,
        n_heads:     int = 8,
        n_layers:    int = 6,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.context_len = context_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)
        self.drop    = nn.Dropout(dropout)

        self.blocks  = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share token embedding and output projection weights
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: (B, T) integer token IDs, T <= context_len
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = idx.shape
        assert T <= self.context_len, f"Sequence length {T} exceeds context_len {self.context_len}"

        positions = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x)

        return self.head(self.ln_f(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
