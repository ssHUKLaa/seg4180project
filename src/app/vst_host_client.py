"""TCP JSON client for headless VST3 host sidecar."""

import json
import socket
import subprocess
import time
import os
import sys
from pathlib import Path
from typing import Optional


def _resolve_runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


class VstHostClient:
    """Connect to sidecar vst-host.exe on localhost:5057."""

    def __init__(self, sidecar_exe_path: Optional[Path] = None):
        self.sidecar_path = sidecar_exe_path
        self.last_launch_path: Optional[Path] = None
        self.sidecar_process = None
        self._sidecar_log_file = None
        self._sidecar_log_path: Optional[Path] = None
        self.host = "127.0.0.1"
        self.port = 5057
        self.last_error = ""
        self.request_id = 0
        self._auto_launched = False

    def _kill_existing_hosts(self) -> None:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/IM", "vst-host.exe", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                time.sleep(0.2)
        except Exception:
            pass

    def start_sidecar(self, force_clean: bool = False) -> bool:
        """Launch the sidecar exe if not already running."""
        if force_clean:
            self.shutdown()
            self._kill_existing_hosts()

        if self._auto_launched or self.is_connected():
            return True

        if self.sidecar_path is None:
            base = _resolve_runtime_root() / "sidecar" / "vst-host" / "Builds" / "VisualStudio2022" / "x64"
            release_path = base / "Release" / "ConsoleApp" / "vst-host.exe"
            debug_path = base / "Debug" / "ConsoleApp" / "vst-host.exe"
            candidates = [p for p in (debug_path, release_path) if p.exists()]
            if candidates:
                self.sidecar_path = max(candidates, key=lambda p: p.stat().st_mtime)
            else:
                self.last_error = f"Sidecar exe not found at {release_path} or {debug_path}"
                return False

        try:
            log_dir = _resolve_runtime_root() / "sidecar" / "vst-host" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._sidecar_log_path = log_dir / "sidecar.log"
            self._sidecar_log_file = open(self._sidecar_log_path, "ab", buffering=0)
            launch_marker = f"\n=== Launch {time.strftime('%Y-%m-%d %H:%M:%S')} | {self.sidecar_path} ===\n"
            self._sidecar_log_file.write(launch_marker.encode("utf-8", errors="replace"))

            startupinfo = None
            creationflags = 0
            if hasattr(subprocess, "STARTUPINFO"):
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0
            if hasattr(subprocess, "CREATE_NO_WINDOW"):
                creationflags |= subprocess.CREATE_NO_WINDOW

            self.sidecar_process = subprocess.Popen(
                [str(self.sidecar_path)],
                stdout=self._sidecar_log_file,
                stderr=self._sidecar_log_file,
                startupinfo=startupinfo,
                creationflags=creationflags
            )
            self._auto_launched = True
            self.last_launch_path = self.sidecar_path

            deadline = time.time() + 6.0
            while time.time() < deadline:
                if self.sidecar_process and self.sidecar_process.poll() is not None:
                    code = self.sidecar_process.returncode
                    self.last_error = f"Sidecar exited during startup (code {code}). {self._tail_sidecar_log()}"
                    return False
                if self.is_connected():
                    return True
                time.sleep(0.1)

            self.last_error = f"Sidecar did not open port 5057 in time. {self._tail_sidecar_log()}"
            return False
        except Exception as e:
            self.last_error = f"Failed to start sidecar: {e}"
            return False

    def _tail_sidecar_log(self, max_bytes: int = 2048) -> str:
        if self._sidecar_log_path is None or not self._sidecar_log_path.exists():
            return "No sidecar log available"
        try:
            with open(self._sidecar_log_path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - max_bytes), os.SEEK_SET)
                tail = fh.read().decode("utf-8", errors="replace").strip()
                if not tail:
                    return "Sidecar log is empty"
                return f"Recent sidecar log: {tail.splitlines()[-1]}"
        except Exception:
            return "Failed to read sidecar log"

    def is_connected(self) -> bool:
        """Check if sidecar is listening."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def send_command(self, cmd: str, timeout_sec: float = 2.0, **kwargs) -> dict:
        """Send a JSON command and get response."""
        self.request_id += 1
        payload = {"id": str(self.request_id), "cmd": cmd, **kwargs}

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout_sec)
            sock.connect((self.host, self.port))

            request = json.dumps(payload)
            sock.sendall(request.encode("utf-8"))
            try:
                sock.shutdown(socket.SHUT_WR)
            except Exception:
                pass

            response_data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response_data += chunk
                if response_data.endswith(b"}"):
                    break

            sock.close()

            if not response_data:
                self.last_error = "Empty response from sidecar"
                return {"ok": False, "error": "Empty response"}

            response = json.loads(response_data.decode("utf-8"))
            if not response.get("ok"):
                self.last_error = response.get("error", "Unknown error")
            return response

        except socket.timeout:
            self.last_error = "Sidecar connection timeout (host did not respond in time)"
            return {"ok": False, "error": "Timeout"}
        except ConnectionAbortedError as e:
            host_up = self.is_connected()
            self.last_error = f"Connection aborted by host process: {e}. sidecar_connected={host_up}. {self._tail_sidecar_log()}"
            return {"ok": False, "error": self.last_error}
        except ConnectionResetError as e:
            host_up = self.is_connected()
            self.last_error = f"Connection reset by host process: {e}. sidecar_connected={host_up}. {self._tail_sidecar_log()}"
            return {"ok": False, "error": self.last_error}
        except Exception as e:
            self.last_error = str(e)
            return {"ok": False, "error": str(e)}

    def load_plugin(self, plugin_path: str) -> bool:
        """Load a VST3 plugin."""
        resp = self.send_command("load_plugin", timeout_sec=12.0, path=plugin_path)
        return resp.get("ok", False)

    def load_midi(self, midi_path: str) -> bool:
        """Load a MIDI file."""
        resp = self.send_command("load_midi", timeout_sec=8.0, path=midi_path)
        return resp.get("ok", False)

    def play(self) -> bool:
        """Start playback."""
        resp = self.send_command("play", timeout_sec=8.0)
        return resp.get("ok", False)

    def stop(self) -> bool:
        """Stop playback."""
        resp = self.send_command("stop")
        return resp.get("ok", False)

    def get_status(self) -> dict:
        """Get current status."""
        resp = self.send_command("get_status")
        return resp

    def set_parameter(self, index: int, value: float) -> bool:
        """Set a plugin parameter."""
        resp = self.send_command("set_parameter", index=index, value=value)
        return resp.get("ok", False)

    def show_editor(self) -> bool:
        """Open plugin editor window."""
        resp = self.send_command("show_editor", timeout_sec=8.0)
        return resp.get("ok", False)

    def hide_editor(self) -> bool:
        """Close plugin editor window."""
        resp = self.send_command("hide_editor")
        return resp.get("ok", False)

    def shutdown(self) -> None:
        """Kill the sidecar process."""
        if self.sidecar_process:
            try:
                self.sidecar_process.terminate()
                self.sidecar_process.wait(timeout=2)
            except Exception:
                self.sidecar_process.kill()
            self.sidecar_process = None
            self._auto_launched = False
        if self._sidecar_log_file is not None:
            try:
                self._sidecar_log_file.close()
            except Exception:
                pass
            self._sidecar_log_file = None
