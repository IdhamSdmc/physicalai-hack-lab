"""Gestor de subprocesos para lanzar, parar y vigilar la tarea del robot SO-101."""

from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Deque, Dict, List, Optional


ESTADOS_VALIDOS = ("libre", "ejecutando", "detenido", "error")
EVENTOS_CLAVE = (
    "[MAPA INICIAL]",
    "[CICLO ",
    "[MAPA]",
    "[REMAP]",
    "[VERIFICACIÓN OK]",
    "[AUTO-RECOVERY]",
    "MISIÓN COMPLETADA",
    "[FALLO CRÍTICO]",
    "[DRY-RUN]",
    "[STOP]",
)


class TaskRunner:
    """Ejecuta una única tarea externa a la vez en su propio grupo de procesos."""

    def __init__(self, config: Dict[str, Any]) -> None:
        task_cfg = config.get("task", {})
        ui_cfg = config.get("ui", {})
        telemetry_cfg = config.get("telemetry", {})

        self.mock_mode: bool = bool(task_cfg.get("mock_mode", True))
        self.mock_duration_seconds: int = max(1, int(task_cfg.get("mock_duration_seconds", 10)))
        self.command: str = str(task_cfg.get("command", "")).strip()
        self.working_dir: Optional[str] = str(task_cfg.get("working_dir", "")).strip() or None
        self.stop_grace_seconds: float = max(0.5, float(task_cfg.get("stop_grace_seconds", 3.0)))
        self.poll_interval_ms: int = max(250, int(ui_cfg.get("poll_interval_ms", 750)))
        self.log_max_lines: int = max(5, int(ui_cfg.get("log_max_lines", 60)))

        # Rutas opcionales de telemetría de solo lectura generadas por el supervisor externo.
        self._hud_image_path: Optional[Path] = self._resolve_telemetry_path(
            telemetry_cfg.get("hud_image_path")
        )
        self._scene_map_path: Optional[Path] = self._resolve_telemetry_path(
            telemetry_cfg.get("scene_map_path")
        )

        # Lock para garantizar exclusión mutua: solo una tarea a la vez.
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen[str]] = None
        self._pgid: Optional[int] = None
        self._state: str = "libre"
        self._message: str = "Listo para iniciar la tarea."
        self._last_event: Optional[str] = None
        self._started_at: Optional[float] = None
        self._stop_requested: bool = False
        self._logs: Deque[str] = deque(maxlen=self.log_max_lines)

        self._append_log("Panel iniciado en modo simulación (mock)." if self.mock_mode else "Panel iniciado en modo real.")

    def _resolve_telemetry_path(self, raw_path: Any) -> Optional[Path]:
        """Resuelve una ruta de telemetría respecto a working_dir si está configurada."""
        text = str(raw_path or "").strip()
        if not text:
            return None
        candidate = Path(text)
        if candidate.is_absolute():
            return candidate
        if self.working_dir:
            return Path(self.working_dir) / candidate
        return candidate

    def _append_log(self, text: str) -> None:
        """Añade una línea con marca de tiempo al registro circular."""
        hora = datetime.now().strftime("%H:%M:%S")
        self._logs.append(f"[{hora}] {text}")

    def _build_command(self) -> List[str]:
        """Construye la lista de argumentos del comando desde la configuración."""
        if self.mock_mode:
            # En modo simulación lanzamos un subproceso Python real para probar
            # exactamente el mismo ciclo de vida, logs y os.killpg sin tocar el robot.
            script = (
                "import time, sys\n"
                f"total = {self.mock_duration_seconds}\n"
                "print('Simulación iniciada: ejecutando tarea de prueba...', flush=True)\n"
                "for i in range(1, total + 1):\n"
                "    print(f'Paso simulado {i}/{total}', flush=True)\n"
                "    time.sleep(1)\n"
                "print('Simulación completada correctamente.', flush=True)\n"
            )
            return [sys.executable, "-u", "-c", script]

        if not self.command:
            raise ValueError("No hay ningún comando configurado en config.yaml (task.command).")

        return shlex.split(self.command)

    def start(self) -> Dict[str, Any]:
        """Inicia la tarea si no hay otra en ejecución. Devuelve estado y resultado."""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return {
                    "ok": False,
                    "status_code": 409,
                    "message": "Ya hay una tarea en ejecución. Detenla antes de iniciar otra.",
                    "status": self._snapshot_locked(),
                }

            try:
                cmd = self._build_command()
                cwd: Optional[str] = None
                if not self.mock_mode and self.working_dir:
                    if not Path(self.working_dir).is_dir():
                        raise FileNotFoundError(
                            f"El directorio de trabajo no existe: {self.working_dir}"
                        )
                    cwd = self.working_dir

                # start_new_session=True crea un nuevo grupo de procesos (PGID = PID)
                # para poder detener el proceso y todos sus hijos con os.killpg.
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=cwd,
                    start_new_session=True,
                )
                self._pgid = os.getpgid(self._process.pid)
                self._state = "ejecutando"
                self._started_at = time.monotonic()
                self._stop_requested = False
                self._last_event = None
                self._message = (
                    "Tarea simulada en marcha." if self.mock_mode else "Tarea del robot en marcha."
                )
                self._append_log(self._message)

                # Hilos ligeros para leer salida y vigilar cuándo termina el proceso.
                threading.Thread(target=self._read_output, args=(self._process,), daemon=True).start()
                threading.Thread(target=self._watch_process, args=(self._process,), daemon=True).start()

                return {
                    "ok": True,
                    "status_code": 200,
                    "message": self._message,
                    "status": self._snapshot_locked(),
                }
            except Exception as exc:
                self._process = None
                self._pgid = None
                self._started_at = None
                self._state = "error"
                self._message = f"No se pudo iniciar la tarea: {exc}"
                self._append_log(f"ERROR: {self._message}")
                return {
                    "ok": False,
                    "status_code": 500,
                    "message": self._message,
                    "status": self._snapshot_locked(),
                }

    def stop(self) -> Dict[str, Any]:
        """Parada de emergencia: envía SIGTERM al grupo de procesos (liberación limpia) y SIGKILL si no responde."""
        with self._lock:
            proc = self._process
            pgid = self._pgid

            if proc is None or proc.poll() is not None:
                self._process = None
                self._pgid = None
                self._started_at = None
                if self._state == "ejecutando":
                    self._state = "detenido"
                self._message = "No había ninguna tarea en marcha."
                return {
                    "ok": True,
                    "status_code": 200,
                    "message": self._message,
                    "status": self._snapshot_locked(),
                }

            self._stop_requested = True
            self._append_log("Parada de emergencia solicitada (STOP / SIGTERM).")

            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGTERM)
                else:
                    proc.terminate()
            except ProcessLookupError:
                pass
            except Exception as exc:
                self._append_log(f"Aviso al enviar SIGTERM: {exc}")

        # Esperamos fuera del lock a que el supervisor pare rollout y libere torque limpiamente.
        try:
            proc.wait(timeout=self.stop_grace_seconds)
        except subprocess.TimeoutExpired:
            with self._lock:
                self._append_log("El proceso no cerró tras SIGTERM; enviando SIGKILL al grupo.")
                try:
                    if pgid is not None:
                        os.killpg(pgid, signal.SIGKILL)
                    else:
                        proc.kill()
                except ProcessLookupError:
                    pass
                except Exception as exc:
                    self._append_log(f"Aviso al enviar SIGKILL: {exc}")
            try:
                proc.wait(timeout=1.0)
            except Exception:
                pass

        with self._lock:
            self._process = None
            self._pgid = None
            self._started_at = None
            self._state = "detenido"
            self._message = "Tarea detenida por el usuario."
            self._append_log(self._message)
            return {
                "ok": True,
                "status_code": 200,
                "message": self._message,
                "status": self._snapshot_locked(),
            }

    def _read_output(self, proc: subprocess.Popen[str]) -> None:
        """Lee stdout/stderr del subproceso línea a línea sin bloquear el servidor."""
        if proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                clean = line.rstrip()
                if clean:
                    with self._lock:
                        self._append_log(clean)
                        if any(marker in clean for marker in EVENTOS_CLAVE):
                            self._last_event = clean
        except Exception:
            pass
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass

    def _watch_process(self, proc: subprocess.Popen[str]) -> None:
        """Espera a que termine el subproceso y actualiza el estado final."""
        returncode = proc.wait()
        with self._lock:
            # Si ya fue limpiado por stop(), no pisamos el estado 'detenido'.
            if self._process is not proc:
                return

            self._process = None
            self._pgid = None
            self._started_at = None

            if self._stop_requested:
                self._state = "detenido"
                self._message = "Tarea detenida por el usuario."
            elif returncode == 0:
                self._state = "libre"
                self._message = "Tarea completada correctamente."
                self._append_log(self._message)
            else:
                self._state = "error"
                self._message = f"La tarea terminó con error (código {returncode})."
                self._append_log(f"ERROR: {self._message}")

    def _read_telemetry(self) -> Optional[Dict[str, Any]]:
        """Lee de forma segura el JSON de mapa/conteo generado por el supervisor."""
        if self.mock_mode:
            return None

        scene_map: Optional[Dict[str, Any]] = None
        if self._scene_map_path and self._scene_map_path.is_file():
            try:
                raw = self._scene_map_path.read_text(encoding="utf-8")
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    scene_map = parsed
            except Exception:
                # Si el supervisor está reescribiendo el archivo, ignoramos este ciclo sin romper el polling.
                scene_map = None

        hud_mtime: Optional[float] = None
        if self._hud_image_path and self._hud_image_path.is_file():
            try:
                hud_mtime = round(self._hud_image_path.stat().st_mtime, 3)
            except Exception:
                hud_mtime = None

        if scene_map is None and hud_mtime is None:
            return None

        return {
            "scene_map": scene_map,
            "hud_available": hud_mtime is not None,
            "hud_mtime": hud_mtime,
        }

    def get_hud_image_path(self) -> Optional[Path]:
        """Devuelve la ruta al último snapshot HUD solo cuando no está en modo simulación."""
        if self.mock_mode or not self._hud_image_path:
            return None
        if not self._hud_image_path.is_file():
            return None
        return self._hud_image_path

    def _snapshot_locked(self) -> Dict[str, Any]:
        """Construye el diccionario de estado actual (debe llamarse con _lock)."""
        running = self._process is not None and self._process.poll() is None
        elapsed = 0
        if running and self._started_at is not None:
            elapsed = int(time.monotonic() - self._started_at)

        return {
            "state": self._state,
            "running": running,
            "mock_mode": self.mock_mode,
            "elapsed_seconds": elapsed,
            "message": self._message,
            "last_event": self._last_event,
            "logs": list(self._logs),
            "poll_interval_ms": self.poll_interval_ms,
        }

    def get_status(self) -> Dict[str, Any]:
        """Devuelve el estado actual de la tarea y su telemetría de forma segura."""
        with self._lock:
            snapshot = self._snapshot_locked()
        snapshot["telemetry"] = self._read_telemetry()
        return snapshot
