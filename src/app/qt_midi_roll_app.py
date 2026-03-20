"""Minimal Qt MIDI piano roll app.

Features:
- Import MIDI and render a piano-roll style view.
- Run model generation via src/model/generate.py.
- Load generated MIDI back into the roll.
- Display process output in an embedded console.
- Optional VST3 plugin playback via headless sidecar host.
"""

from __future__ import annotations

import sys
import threading
import time
import importlib.util
import importlib
from pathlib import Path
from typing import Callable

import mido
import numpy as np
from PySide6.QtCore import QObject, QProcess, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QSlider,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vst_host_client import VstHostClient


def _resolve_runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _resolve_resource_root(app_root: Path) -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        for name in ("_internal", "_INTERNAL"):
            candidate = app_root / name
            if candidate.exists():
                return candidate
    return app_root


APP_ROOT = _resolve_runtime_root()
RESOURCE_ROOT = _resolve_resource_root(APP_ROOT)
PROJECT_ROOT = APP_ROOT


def _first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


DEFAULT_CHECKPOINT = _first_existing(
    APP_ROOT / "checkpoints" / "best.pt",
    RESOURCE_ROOT / "checkpoints" / "best.pt",
)
GENERATE_SCRIPT = _first_existing(
    APP_ROOT / "src" / "model" / "generate.py",
    RESOURCE_ROOT / "src" / "model" / "generate.py",
)

MODEL_ROOT = _first_existing(
    APP_ROOT / "src" / "model",
    RESOURCE_ROOT / "src" / "model",
)


def _load_generate_from_midi_prompt():
    # In packaged builds, generate.py is typically bundled as a module, not a loose file.
    try:
        module = importlib.import_module("generate")
        fn = getattr(module, "generate_from_midi_prompt", None)
        if fn is not None:
            return fn
    except Exception:
        pass

    module_path = MODEL_ROOT / "generate.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing generation module: {module_path}")

    spec = importlib.util.spec_from_file_location("runtime_generate", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load generation module spec")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "generate_from_midi_prompt", None)
    if fn is None:
        raise RuntimeError("generate_from_midi_prompt not found in generation module")
    return fn


class PianoRollView(QGraphicsView):
    """Simple piano roll rendering using QGraphicsScene."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing, False)
        self.setRenderHint(QPainter.TextAntialiasing, True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(QColor("#1a1a1a"))
        self.px_per_tick = 0.08
        self.base_px_per_tick = 0.08
        self.row_h = 12
        self.key_w = 84
        self._first_load = True
        self._last_midi_path: Path | None = None
        self.on_zoom_changed: Callable[[int], None] | None = None
        self._playhead_item: QGraphicsRectItem | None = None
        self._last_max_tick = 0
        self._last_length_sec = 0.0

    @staticmethod
    def _is_black_key(pitch: int) -> bool:
        return (pitch % 12) in {1, 3, 6, 8, 10}

    @staticmethod
    def _note_name(pitch: int) -> str:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (pitch // 12) - 1
        return f"{names[pitch % 12]}{octave}"

    def clear_roll(self) -> None:
        self.scene.clear()
        self._playhead_item = None

    def set_horizontal_zoom(self, zoom_percent: int) -> None:
        zoom = max(10, min(400, int(zoom_percent)))
        self.px_per_tick = self.base_px_per_tick * (zoom / 100.0)
        if self._last_midi_path is not None:
            self.load_midi(self._last_midi_path)
        if self.on_zoom_changed is not None:
            self.on_zoom_changed(zoom)

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            current_zoom = int(round((self.px_per_tick / self.base_px_per_tick) * 100.0))
            step = 10 if delta > 0 else -10
            self.set_horizontal_zoom(current_zoom + step)
            event.accept()
            return
        super().wheelEvent(event)

    def load_midi(self, midi_path: Path) -> None:
        self._last_midi_path = midi_path
        self.clear_roll()
        mid = mido.MidiFile(str(midi_path))

        notes = []
        pitches = []
        max_tick = 0

        for track in mid.tracks:
            abs_tick = 0
            active: dict[tuple[int, int], int] = {}
            for msg in track:
                abs_tick += msg.time
                max_tick = max(max_tick, abs_tick)

                if msg.type == "note_on" and msg.velocity > 0:
                    key = (getattr(msg, "channel", 0), msg.note)
                    active[key] = abs_tick
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    key = (getattr(msg, "channel", 0), msg.note)
                    start = active.pop(key, None)
                    if start is not None and abs_tick >= start:
                        notes.append((start, abs_tick, msg.note))
                        pitches.append(msg.note)

        self._last_max_tick = int(max_tick)
        self._last_length_sec = float(mid.length)

        # Grid backdrop and piano keyboard strip.
        roll_w = max(1600, int(max_tick * self.px_per_tick) + 200)
        total_w = self.key_w + roll_w
        total_h = 128 * self.row_h

        no_pen = QPen(Qt.NoPen)
        label_font = QFont("Segoe UI", 7)

        for pitch in range(128):
            y = (127 - pitch) * self.row_h
            is_black = self._is_black_key(pitch)

            key_color = QColor("#141414") if is_black else QColor("#f0f0f0")
            lane_color = QColor("#202020") if is_black else QColor("#292929")

            self.scene.addRect(0, y, self.key_w, self.row_h, no_pen, key_color)
            self.scene.addRect(self.key_w, y, roll_w, self.row_h, no_pen, lane_color)

            # Label every key with uniform text size.
            text_color = QColor("#ececec") if is_black else QColor("#101010")
            label = self.scene.addSimpleText(self._note_name(pitch), label_font)
            label.setBrush(text_color)
            text_w = label.boundingRect().width()
            text_h = label.boundingRect().height()
            label_y = y + ((self.row_h - text_h) * 0.5)
            label_x = self.key_w - text_w - 4
            label.setPos(label_x, label_y)
            label.setZValue(20)

        # Stronger octave separators.
        for n in range(0, 128, 12):
            y = (127 - n) * self.row_h
            self.scene.addRect(0, y, total_w, 1, no_pen, QColor("#4a4a4a"))

        # Split line between keyboard and timeline.
        self.scene.addRect(self.key_w - 1, 0, 1, total_h, no_pen, QColor("#5a5a5a"))

        # Draw notes
        for start, end, pitch in notes:
            x = self.key_w + (start * self.px_per_tick)
            w = max(1.0, (end - start) * self.px_per_tick)
            y = (127 - pitch) * self.row_h
            item = QGraphicsRectItem(x, y, w, self.row_h - 1)
            item.setBrush(QColor("#5fb0ff"))
            item.setPen(QColor("#2f6fa6"))
            self.scene.addItem(item)

        # Playhead on top of everything.
        playhead = QGraphicsRectItem(self.key_w, 0, 2, total_h)
        playhead.setBrush(QColor("#ff5252"))
        playhead.setPen(QPen(Qt.NoPen))
        playhead.setZValue(100)
        self.scene.addItem(playhead)
        self._playhead_item = playhead

        self.scene.setSceneRect(0, 0, total_w, total_h)

        # Avoid fitInView scaling because it makes text on keys render poorly.
        self.resetTransform()

        # On first load, focus viewport around played pitch range.
        if self._first_load:
            if pitches:
                center_pitch = int(sum(pitches) / len(pitches))
            else:
                center_pitch = 60
            center_y = (127 - center_pitch) * self.row_h
            self.centerOn(self.key_w + 300, center_y)
            self._first_load = False

    def set_playhead_seconds(self, seconds: float) -> None:
        if self._playhead_item is None:
            return
        x = self.key_w
        if self._last_length_sec > 1e-6 and self._last_max_tick > 0:
            ticks_per_sec = self._last_max_tick / self._last_length_sec
            x = self.key_w + (seconds * ticks_per_sec * self.px_per_tick)
        right = self.scene.sceneRect().right()
        x = max(self.key_w, min(float(right), x))
        self._playhead_item.setRect(x, 0, 2, self.scene.sceneRect().height())

    def reset_playhead(self) -> None:
        self.set_playhead_seconds(0.0)

    def estimate_length_seconds(self) -> float:
        return self._last_length_sec


class SimpleMidiPlayer:
    """Very small internal MIDI player using additive sine synthesis."""

    def __init__(self, log_fn: Callable[[str], None]):
        self.log = log_fn
        self.sample_rate = 44100
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._play_obj = None
        self._sa = None
        self._load_backend()

    def _load_backend(self) -> None:
        try:
            import simpleaudio as sa
            self._sa = sa
        except Exception:
            self._sa = None

    def available(self) -> bool:
        return self._sa is not None

    def is_playing(self) -> bool:
        if self._play_obj is None:
            return False
        try:
            return bool(self._play_obj.is_playing())
        except Exception:
            return False

    def stop(self) -> None:
        self._stop_flag.set()
        if self._play_obj is not None:
            try:
                self._play_obj.stop()
            except Exception:
                pass

    def play(self, midi_path: Path) -> None:
        self.stop()
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._play_worker, args=(midi_path,), daemon=True)
        self._thread.start()

    def _play_worker(self, midi_path: Path) -> None:
        if self._sa is None:
            self.log("Playback unavailable: install simpleaudio in this environment.")
            return
        try:
            audio = self._synthesize(midi_path)
            if self._stop_flag.is_set() or audio.size == 0:
                return
            self._play_obj = self._sa.play_buffer(audio, 1, 2, self.sample_rate)
        except Exception as exc:
            self.log(f"Playback error: {exc}")

    def _synthesize(self, midi_path: Path) -> np.ndarray:
        mid = mido.MidiFile(str(midi_path))

        active: dict[tuple[int, int], tuple[float, int]] = {}
        notes: list[tuple[float, float, int, int]] = []
        t = 0.0
        for msg in mid:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[(getattr(msg, "channel", 0), msg.note)] = (t, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (getattr(msg, "channel", 0), msg.note)
                start_vel = active.pop(key, None)
                if start_vel is not None:
                    start, vel = start_vel
                    end = max(t, start + 0.01)
                    notes.append((start, end, msg.note, vel))

        if not notes:
            return np.zeros(1, dtype=np.int16)

        duration = max(end for _, end, _, _ in notes) + 0.2
        total_samples = int(duration * self.sample_rate)
        out = np.zeros(total_samples, dtype=np.float32)

        for start, end, pitch, vel in notes:
            if self._stop_flag.is_set():
                break
            i0 = max(0, int(start * self.sample_rate))
            i1 = min(total_samples, int(end * self.sample_rate))
            n = i1 - i0
            if n <= 1:
                continue

            sec = np.arange(n, dtype=np.float32) / self.sample_rate
            freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))

            # Very simple piano-like timbre and decay.
            wave = (
                np.sin(2.0 * np.pi * freq * sec)
                + 0.45 * np.sin(2.0 * np.pi * (freq * 2.0) * sec)
                + 0.2 * np.sin(2.0 * np.pi * (freq * 3.0) * sec)
            )
            attack_s = 0.004
            attack_n = max(1, int(attack_s * self.sample_rate))
            env = np.exp(-3.5 * sec / max(end - start, 1e-3))
            env[:attack_n] *= np.linspace(0.0, 1.0, attack_n, dtype=np.float32)

            amp = (vel / 127.0) * 0.28
            out[i0:i1] += (wave * env * amp).astype(np.float32)

        peak = float(np.max(np.abs(out))) if out.size > 0 else 0.0
        if peak > 0.99:
            out = out / peak

        return (out * 32767.0).astype(np.int16)


class GenerationWorker(QObject):
    log = Signal(str)
    finished = Signal(bool, str)

    def __init__(
        self,
        checkpoint: Path,
        vocab_path: Path,
        midi_prompt: Path,
        output: Path,
        min_note_ms: float,
        key_filter: bool,
        key_penalty: float,
        density_aware: bool,
        density_penalty: float,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.vocab_path = vocab_path
        self.midi_prompt = midi_prompt
        self.output = output
        self.min_note_ms = min_note_ms
        self.key_filter = key_filter
        self.key_penalty = key_penalty
        self.density_aware = density_aware
        self.density_penalty = density_penalty

    def run(self) -> None:
        try:
            generate_from_midi_prompt = _load_generate_from_midi_prompt()
            generate_from_midi_prompt(
                checkpoint=self.checkpoint,
                vocab_path=self.vocab_path,
                midi_prompt=self.midi_prompt,
                output=self.output,
                min_note_ms=self.min_note_ms,
                key_filter=self.key_filter,
                key_penalty=self.key_penalty,
                density_aware=self.density_aware,
                density_penalty=self.density_penalty,
                log_fn=self.log.emit,
            )
            self.finished.emit(True, "")
        except Exception as exc:
            self.finished.emit(False, str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MIDI Generator")
        self.resize(1400, 850)

        self.gen_thread: QThread | None = None
        self.gen_worker: GenerationWorker | None = None
        self.playback_process: QProcess | None = None
        self.current_midi_path: Path | None = None
        self.playhead_timer = QTimer(self)
        self.playhead_timer.setInterval(30)
        self.playhead_timer.timeout.connect(self._on_playhead_tick)
        self.playback_start_t = 0.0
        self.playback_length_sec = 0.0
        self.external_playing = False

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        controls = QWidget(self)
        form = QFormLayout(controls)

        self.input_edit = QLineEdit("")
        self.input_edit.setPlaceholderText("Choose an input MIDI file (.mid)")
        self.output_edit = QLineEdit("")
        self.output_edit.setPlaceholderText("Choose an output MIDI file (.mid)")
        self.ckpt_edit = QLineEdit(str(DEFAULT_CHECKPOINT))

        input_row = QWidget(self)
        input_row_layout = QHBoxLayout(input_row)
        input_row_layout.setContentsMargins(0, 0, 0, 0)
        self.browse_input_btn = QPushButton("...")
        self.browse_input_btn.setFixedWidth(32)
        input_row_layout.addWidget(self.input_edit)
        input_row_layout.addWidget(self.browse_input_btn)

        output_row = QWidget(self)
        output_row_layout = QHBoxLayout(output_row)
        output_row_layout.setContentsMargins(0, 0, 0, 0)
        self.browse_output_btn = QPushButton("...")
        self.browse_output_btn.setFixedWidth(32)
        output_row_layout.addWidget(self.output_edit)
        output_row_layout.addWidget(self.browse_output_btn)

        ckpt_row = QWidget(self)
        ckpt_row_layout = QHBoxLayout(ckpt_row)
        ckpt_row_layout.setContentsMargins(0, 0, 0, 0)
        self.browse_ckpt_btn = QPushButton("...")
        self.browse_ckpt_btn.setFixedWidth(32)
        ckpt_row_layout.addWidget(self.ckpt_edit)
        ckpt_row_layout.addWidget(self.browse_ckpt_btn)

        self.key_filter_cb = QCheckBox()
        self.key_filter_cb.setChecked(True)
        self.density_cb = QCheckBox()
        self.density_cb.setChecked(True)

        self.key_penalty = QDoubleSpinBox()
        self.key_penalty.setRange(0.0, 10.0)
        self.key_penalty.setValue(1.5)
        self.key_penalty.setSingleStep(0.1)

        self.density_penalty = QDoubleSpinBox()
        self.density_penalty.setRange(0.0, 10.0)
        self.density_penalty.setValue(2.0)
        self.density_penalty.setSingleStep(0.1)

        self.min_note_ms = QSpinBox()
        self.min_note_ms.setRange(0, 200)
        self.min_note_ms.setValue(25)

        self.player_cmd_edit = QLineEdit("")
        self.player_cmd_edit.setPlaceholderText("Optional external player/VST host command, use {midi} placeholder")

        # VST3 host integration
        self.use_vst_cb = QCheckBox()
        self.use_vst_cb.setChecked(False)
        
        self.vst_plugin_edit = QLineEdit("")
        self.vst_plugin_edit.setPlaceholderText("Path to VST3 plugin (.vst3)")
        
        vst_row = QWidget(self)
        vst_row_layout = QHBoxLayout(vst_row)
        vst_row_layout.setContentsMargins(0, 0, 0, 0)
        self.browse_vst_btn = QPushButton("...")
        self.browse_vst_btn.setFixedWidth(32)
        vst_row_layout.addWidget(self.vst_plugin_edit)
        vst_row_layout.addWidget(self.browse_vst_btn)

        form.addRow("Input MIDI", input_row)
        form.addRow("Output MIDI", output_row)
        form.addRow("Checkpoint", ckpt_row)
        form.addRow("Enable key filter", self.key_filter_cb)
        form.addRow("Enable density-aware", self.density_cb)
        form.addRow("Key penalty", self.key_penalty)
        form.addRow("Density penalty", self.density_penalty)
        form.addRow("Min note ms", self.min_note_ms)
        form.addRow("External player cmd", self.player_cmd_edit)
        form.addRow("Use VST3 Plugin", self.use_vst_cb)
        form.addRow("VST3 Plugin Path", vst_row)

        btn_row = QHBoxLayout()
        self.import_btn = QPushButton("Import To Roll")
        self.generate_btn = QPushButton("Generate")
        self.load_output_btn = QPushButton("Load Output To Roll")
        self.play_btn = QPushButton("Play")
        self.stop_btn = QPushButton("Stop")
        self.open_vst_editor_btn = QPushButton("Open VST Editor")

        for b in [self.import_btn, self.generate_btn, self.load_output_btn, self.play_btn, self.stop_btn, self.open_vst_editor_btn]:
            btn_row.addWidget(b)

        root.addWidget(controls)
        root.addLayout(btn_row)

        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Horizontal Zoom"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(25, 300)
        self.zoom_slider.setValue(100)
        self.zoom_value_label = QLabel("100%")
        self.zoom_value_label.setFixedWidth(48)
        zoom_row.addWidget(self.zoom_slider)
        zoom_row.addWidget(self.zoom_value_label)
        root.addLayout(zoom_row)

        self.roll = PianoRollView(self)
        root.addWidget(QLabel("Piano Roll"))
        root.addWidget(self.roll, stretch=3)

        self.console = QTextEdit(self)
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(200)
        root.addWidget(QLabel("Output Console"))
        root.addWidget(self.console, stretch=1)

        self.browse_input_btn.clicked.connect(self.browse_input)
        self.browse_output_btn.clicked.connect(self.browse_output)
        self.browse_ckpt_btn.clicked.connect(self.browse_checkpoint)
        self.browse_vst_btn.clicked.connect(self.browse_vst)
        self.import_btn.clicked.connect(self.load_input_to_roll)
        self.load_output_btn.clicked.connect(self.load_output_to_roll)
        self.generate_btn.clicked.connect(self.run_generation)
        self.play_btn.clicked.connect(self.play_current)
        self.stop_btn.clicked.connect(self.stop_playback)
        self.open_vst_editor_btn.clicked.connect(self.open_vst_editor)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        self.roll.on_zoom_changed = self.on_roll_zoom_changed

        self.midi_player = SimpleMidiPlayer(self.log)
        self.vst_client = VstHostClient()
        self.vst_loaded = False
        self.vst_load_in_progress = False
        self.vst_editor_open_in_progress = False
        self.vst_editor_retry_scheduled = False
        self.vst_play_in_progress = False
        self.vst_loaded_at = 0.0

        # Start sidecar in the background when the app launches.
        QTimer.singleShot(0, self._auto_start_sidecar)

    def log(self, msg: str) -> None:
        self.console.append(msg.rstrip("\n"))

    def browse_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Pick Input MIDI", str(PROJECT_ROOT), "MIDI Files (*.mid *.midi)")
        if path:
            self.input_edit.setText(path)

    def browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Pick Output MIDI", "generated.mid", "MIDI Files (*.mid)")
        if path:
            self.output_edit.setText(path)

    def browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Pick Checkpoint", str(PROJECT_ROOT), "PT Files (*.pt)")
        if path:
            self.ckpt_edit.setText(path)

    def browse_vst(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Pick VST3 Plugin", str(PROJECT_ROOT), "VST3 Files (*.vst3)")
        if path:
            self.vst_plugin_edit.setText(path)
            self._load_vst_plugin(path)

    def _load_roll(self, path_text: str) -> None:
        path = Path(path_text)
        if not path.exists():
            QMessageBox.warning(self, "Missing file", f"File not found:\n{path}")
            return
        try:
            self.roll.load_midi(path)
            self.current_midi_path = path
            self.log(f"Loaded into piano roll: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "MIDI load error", str(exc))

    def load_input_to_roll(self) -> None:
        self._load_roll(self.input_edit.text().strip())

    def load_output_to_roll(self) -> None:
        self._load_roll(self.output_edit.text().strip())

    def _auto_start_sidecar(self) -> None:
        """Launch the VST sidecar when the Qt app starts."""
        self.log("Starting VST host sidecar...")
        if self.vst_client.start_sidecar(force_clean=True):
            self.log("VST host sidecar is ready")
            if self.vst_client.last_launch_path is not None:
                self.log(f"VST host binary: {self.vst_client.last_launch_path}")
        else:
            # Keep startup non-blocking; user can still use internal/external playback.
            self.log(f"VST host sidecar not available: {self.vst_client.last_error}")

    def _load_vst_plugin(self, vst_path: str) -> None:
        """Load a VST3 plugin via the sidecar host."""
        if self.vst_load_in_progress:
            self.log("VST plugin load already in progress")
            return

        self.vst_load_in_progress = True
        try:
            # Start sidecar if not running
            if not self.vst_client.is_connected():
                self.log("Starting VST host sidecar...")
                if not self.vst_client.start_sidecar(force_clean=True):
                    QMessageBox.warning(self, "VST Error", f"Failed to start sidecar:\n{self.vst_client.last_error}")
                    return

            # Load plugin
            self.log(f"Loading VST plugin: {vst_path}")
            if self.vst_client.load_plugin(vst_path):
                self.log("VST plugin loaded successfully")
                self.vst_loaded = True
                self.vst_loaded_at = time.perf_counter()
                self.use_vst_cb.setChecked(True)
            else:
                err = self.vst_client.last_error.lower()
                # If host dropped the socket during load, restart once and retry.
                if "forcibly closed" in err or "connection reset" in err or "connection aborted" in err:
                    if self.vst_client.is_connected():
                        # Host can stay alive while plugin load resets this socket.
                        self.log("VST load socket reset, sidecar still running; retrying load once...")
                        if self.vst_client.load_plugin(vst_path):
                            self.log("VST plugin loaded successfully (after retry)")
                            self.vst_loaded = True
                            self.vst_loaded_at = time.perf_counter()
                            self.use_vst_cb.setChecked(True)
                            return
                    else:
                        self.log("VST host reset during load, restarting sidecar once...")
                        self.vst_client.shutdown()
                        if self.vst_client.start_sidecar(force_clean=True) and self.vst_client.load_plugin(vst_path):
                            self.log("VST plugin loaded successfully (after retry)")
                            self.vst_loaded = True
                            self.vst_loaded_at = time.perf_counter()
                            self.use_vst_cb.setChecked(True)
                            return

                QMessageBox.warning(self, "VST Error", f"Failed to load plugin:\n{self.vst_client.last_error}")
                self.vst_loaded = False
        except Exception as exc:
            QMessageBox.critical(self, "VST load error", str(exc))
            self.vst_loaded = False
        finally:
            self.vst_load_in_progress = False

    def open_vst_editor(self) -> None:
        """Open the loaded plugin editor window from sidecar."""
        if self.vst_editor_open_in_progress:
            self.log("VST editor open already in progress")
            return

        if not self.vst_client.is_connected():
            if not self.vst_client.start_sidecar(force_clean=True):
                QMessageBox.warning(self, "VST Error", f"Failed to start sidecar:\n{self.vst_client.last_error}")
                return

        vst_path = self.vst_plugin_edit.text().strip()
        if (not self.vst_loaded) and vst_path:
            self._load_vst_plugin(vst_path)

        if not self.vst_loaded:
            QMessageBox.warning(self, "VST Error", "No VST plugin loaded. Pick a .vst3 first.")
            return

        # Some plugins need a brief settle time after load before editor creation.
        elapsed_since_load = time.perf_counter() - self.vst_loaded_at
        if elapsed_since_load < 1.2:
            wait_ms = int((1.2 - elapsed_since_load) * 1000)
            self.log("Waiting briefly before opening VST editor...")
            self.vst_editor_open_in_progress = True
            QTimer.singleShot(max(50, wait_ms), lambda: self._attempt_open_vst_editor(vst_path, 1, 4))
            return

        self.vst_editor_open_in_progress = True
        self._attempt_open_vst_editor(vst_path, 1, 4)

    def _attempt_open_vst_editor(self, vst_path: str, attempt: int, max_attempts: int) -> None:
        if self.vst_client.show_editor():
            msg = "Opened VST editor window" if attempt == 1 else f"Opened VST editor window (attempt {attempt})"
            self.log(msg)
            self.vst_editor_open_in_progress = False
            self.vst_editor_retry_scheduled = False
            return

        if attempt >= max_attempts:
            self.vst_editor_open_in_progress = False
            self.vst_editor_retry_scheduled = False
            QMessageBox.warning(self, "VST Editor Error", f"Failed to open editor:\n{self.vst_client.last_error}")
            return

        delays_ms = [250, 500, 1000, 1500]
        delay_ms = delays_ms[min(attempt - 1, len(delays_ms) - 1)]
        self.log(f"VST editor open failed (attempt {attempt}), retrying...")
        self.vst_editor_retry_scheduled = True
        QTimer.singleShot(delay_ms, lambda: self._attempt_open_vst_editor(vst_path, attempt + 1, max_attempts))

    @staticmethod
    def _is_connection_reset_error(err_text: str) -> bool:
        e = err_text.lower()
        return ("forcibly closed" in e) or ("connection reset" in e) or ("connection aborted" in e) or ("10053" in e)

    def _play_vst_with_retry(self, midi_path: Path, vst_path: str) -> bool:
        if self.vst_client.load_midi(str(midi_path)) and self.vst_client.play():
            return True

        if not self._is_connection_reset_error(self.vst_client.last_error):
            return False

        self.log("VST host reset during play, restarting once...")
        self.vst_client.shutdown()
        self.vst_loaded = False

        if not self.vst_client.start_sidecar(force_clean=True):
            return False
        if not self.vst_client.load_plugin(vst_path):
            return False

        self.vst_loaded = True
        return self.vst_client.load_midi(str(midi_path)) and self.vst_client.play()

    def on_zoom_changed(self, value: int) -> None:
        self.zoom_value_label.setText(f"{value}%")
        self.roll.set_horizontal_zoom(value)

    def on_roll_zoom_changed(self, value: int) -> None:
        self.zoom_value_label.setText(f"{value}%")
        if self.zoom_slider.value() != value:
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(value)
            self.zoom_slider.blockSignals(False)

    def _play_with_external_cmd(self, midi_path: Path, cmd_template: str) -> bool:
        cmd = cmd_template.strip()
        if not cmd:
            return False

        rendered = cmd.replace("{midi}", str(midi_path))
        self.log(f"External playback: {rendered}")

        if self.playback_process is not None:
            self.playback_process.kill()
            self.playback_process.deleteLater()

        self.playback_process = QProcess(self)
        self.playback_process.setWorkingDirectory(str(PROJECT_ROOT))
        self.playback_process.setProgram("cmd.exe")
        self.playback_process.setArguments(["/C", rendered])
        self.playback_process.start()
        return True

    def play_current(self) -> None:
        if self.external_playing or self.vst_play_in_progress:
            self.log("Playback already running")
            return

        path = self.current_midi_path
        if path is None:
            in_text = self.input_edit.text().strip()
            out_text = self.output_edit.text().strip()
            in_path = Path(in_text) if in_text else None
            out_path = Path(out_text) if out_text else None
            if out_path is not None and out_path.is_file():
                path = out_path
            elif in_path is not None and in_path.is_file():
                path = in_path
            else:
                path = None

        if path is None or not path.exists():
            QMessageBox.warning(self, "No MIDI loaded", "Load or generate a MIDI file first.")
            return

        # VST playback (highest priority)
        if self.use_vst_cb.isChecked():
            self.vst_play_in_progress = True
            if not self.vst_client.is_connected():
                self.log("Starting VST host sidecar...")
                if not self.vst_client.start_sidecar(force_clean=True):
                    self.vst_play_in_progress = False
                    QMessageBox.warning(self, "VST Error", f"Failed to start sidecar:\n{self.vst_client.last_error}")
                    return

            vst_path = self.vst_plugin_edit.text().strip()
            if (not self.vst_loaded) and vst_path:
                self._load_vst_plugin(vst_path)
            if not self.vst_loaded:
                self.vst_play_in_progress = False
                QMessageBox.warning(self, "VST Error", "No VST plugin loaded. Pick a .vst3 first.")
                return

            self.log(f"Playing via VST: {path}")
            if self._play_vst_with_retry(path, vst_path):
                self._start_playhead(path)
                self.external_playing = True
            else:
                QMessageBox.warning(self, "VST Playback Error", f"Failed to play:\n{self.vst_client.last_error}")
            self.vst_play_in_progress = False
            return

        # External command playback
        cmd_template = self.player_cmd_edit.text().strip()
        if cmd_template:
            self._play_with_external_cmd(path, cmd_template)
            self._start_playhead(path)
            self.external_playing = True
            return

        # Internal synth playback
        if not self.midi_player.available():
            QMessageBox.warning(
                self,
                "Playback backend missing",
                "simpleaudio is not installed. Install it or provide an external player command.",
            )
            return

        self.log(f"Playing (internal synth): {path}")
        self.midi_player.play(path)
        self.external_playing = False
        self._start_playhead(path)

    def stop_playback(self) -> None:
        self.midi_player.stop()
        if self.vst_client.is_connected():
            self.vst_client.stop()
        if self.playback_process is not None:
            self.playback_process.kill()
        self.external_playing = False
        self.playhead_timer.stop()
        self.roll.reset_playhead()
        self.log("Playback stopped")

    def _start_playhead(self, midi_path: Path) -> None:
        # Ensure the loaded roll matches what is being played for accurate mapping.
        if self.current_midi_path != midi_path:
            self.roll.load_midi(midi_path)
            self.current_midi_path = midi_path

        self.playback_start_t = time.perf_counter()
        self.playback_length_sec = self.roll.estimate_length_seconds()
        self.roll.reset_playhead()
        self.playhead_timer.start()

    def _on_playhead_tick(self) -> None:
        elapsed = max(0.0, time.perf_counter() - self.playback_start_t)
        self.roll.set_playhead_seconds(elapsed)

        done_by_time = self.playback_length_sec > 0 and elapsed >= self.playback_length_sec
        done_by_player = (not self.external_playing) and (not self.midi_player.is_playing()) and elapsed > 0.2
        if done_by_time or done_by_player:
            self.playhead_timer.stop()
            self.external_playing = False

    def run_generation(self) -> None:
        in_text = self.input_edit.text().strip()
        out_text = self.output_edit.text().strip()
        ckpt_text = self.ckpt_edit.text().strip()

        if not in_text:
            QMessageBox.warning(self, "Missing input", "Pick an input MIDI file first.")
            return
        if not out_text:
            QMessageBox.warning(self, "Missing output", "Pick an output MIDI file first.")
            return
        if not ckpt_text:
            QMessageBox.warning(self, "Missing checkpoint", "Pick a checkpoint file first.")
            return

        in_path = Path(in_text)
        out_path = Path(out_text)
        ckpt_path = Path(ckpt_text)

        if not in_path.exists():
            QMessageBox.warning(self, "Missing input", f"Input MIDI not found:\n{in_path}")
            return
        if not ckpt_path.exists():
            QMessageBox.warning(self, "Missing checkpoint", f"Checkpoint not found:\n{ckpt_path}")
            return
        vocab_path = _first_existing(
            APP_ROOT / "data" / "processed" / "vocab.json",
            RESOURCE_ROOT / "data" / "processed" / "vocab.json",
        )
        if not vocab_path.exists():
            QMessageBox.warning(self, "Missing vocab", f"vocab.json not found:\n{vocab_path}")
            return

        self.log("\n=== Running generation ===")
        self.log(f"checkpoint={ckpt_path} midi_prompt={in_path} output={out_path}")

        if self.gen_thread is not None and self.gen_thread.isRunning():
            QMessageBox.information(self, "Generation running", "A generation job is already running.")
            return

        self.generate_btn.setEnabled(False)
        self.gen_thread = QThread(self)
        self.gen_worker = GenerationWorker(
            checkpoint=ckpt_path,
            vocab_path=vocab_path,
            midi_prompt=in_path,
            output=out_path,
            min_note_ms=float(self.min_note_ms.value()),
            key_filter=self.key_filter_cb.isChecked(),
            key_penalty=float(self.key_penalty.value()),
            density_aware=self.density_cb.isChecked(),
            density_penalty=float(self.density_penalty.value()),
        )
        self.gen_worker.moveToThread(self.gen_thread)
        self.gen_thread.started.connect(self.gen_worker.run)
        self.gen_worker.log.connect(self.log)
        self.gen_worker.finished.connect(self._on_generation_finished)
        self.gen_worker.finished.connect(self.gen_thread.quit)
        self.gen_worker.finished.connect(self.gen_worker.deleteLater)
        self.gen_thread.finished.connect(self.gen_thread.deleteLater)
        self.gen_thread.start()

    def _on_generation_finished(self, ok: bool, error: str) -> None:
        self.generate_btn.setEnabled(True)
        if ok:
            self.log("=== Generation finished, exit_code=0 ===")
            self.load_output_to_roll()
        else:
            self.log("=== Generation finished, exit_code=1 ===")
            QMessageBox.warning(self, "Generation error", error)

        self.gen_worker = None
        self.gen_thread = None

    def closeEvent(self, event) -> None:
        self.stop_playback()
        self.vst_client.shutdown()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
