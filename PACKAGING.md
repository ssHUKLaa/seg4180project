# Windows Packaging

This project can be packaged into a distributable zip that includes:
- Qt app executable
- model checkpoint (`checkpoints/best.pt`)
- vocab file (`data/processed/vocab.json`)
- sidecar host executable (`vst-host.exe`)

## Prerequisites

- Windows
- Sidecar already built in Visual Studio:
  - `sidecar/vst-host/Builds/VisualStudio2022/x64/Debug/ConsoleApp/vst-host.exe`
  - or use `-SidecarConfig Release`
- Python environment with project dependencies

## Build Package

From project root in PowerShell:

```powershell
./scripts/package_windows.ps1
```

Optional (use Release sidecar):

```powershell
./scripts/package_windows.ps1 -SidecarConfig Release
```

## Output

- App folder: `dist/MidiGenerator/`
- Zip file: `release/MidiGenerator-windows.zip`

Share the zip. End users can unzip and run `MidiGenerator.exe`.

## Size Optimization

- The generated zip size depends heavily on the PyTorch build in your packaging environment.
- For a smaller package, build/package from a dedicated CPU-only environment (avoid CUDA-enabled torch wheels).
- The packaging script uses a lean Torch include strategy; if runtime import errors appear, switch back to full collection as a fallback.


