"""Servidor FastAPI para el panel de control remoto del brazo SO-101."""

from pathlib import Path
from typing import Any, Dict, Union

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import yaml

from runner import TaskRunner


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
EXAMPLE_CONFIG_PATH = BASE_DIR / "config.example.yaml"
DIST_DIR = BASE_DIR / "dist"
DIST_ASSETS_DIR = DIST_DIR / "assets"
INDEX_PATH = DIST_DIR / "index.html" if (DIST_DIR / "index.html").exists() else BASE_DIR / "index.html"
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> Dict[str, Any]:
    """Carga config.yaml; si no existe, usa config.example.yaml en modo seguro."""
    target = CONFIG_PATH if CONFIG_PATH.exists() else EXAMPLE_CONFIG_PATH
    if not CONFIG_PATH.exists():
        print(
            "[AVISO] No se encontró config.yaml. Usando valores seguros de config.example.yaml "
            "(mock_mode: true). Copia config.example.yaml a config.yaml para personalizar."
        )

    if not target.exists():
        return {
            "server": {"host": "0.0.0.0", "port": 8000},
            "task": {"mock_mode": True, "mock_duration_seconds": 10, "command": ""},
            "ui": {"poll_interval_ms": 750, "log_max_lines": 60},
        }

    with target.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


config = load_config()
runner = TaskRunner(config)

app = FastAPI(title="Panel SO-101 · Hackathon Bilbao II")
# Carpeta local para servir recursos estáticos (Three.js, HDRI, modelos GLTF) 100% offline.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if DIST_ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(DIST_ASSETS_DIR)), name="assets")


@app.get("/")
def serve_panel() -> FileResponse:
    """Devuelve el panel web único (index.html)."""
    return FileResponse(INDEX_PATH)


@app.get("/api/status")
def api_status() -> Dict[str, Any]:
    """Devuelve el estado actual de la tarea y los últimos registros."""
    return runner.get_status()


@app.get("/api/hud.jpg", response_model=None)
def api_hud_image() -> Union[FileResponse, JSONResponse]:
    """Devuelve la última imagen anotada del supervisor OpenCV si está disponible."""
    hud_path = runner.get_hud_image_path()
    if hud_path is None:
        return JSONResponse(
            status_code=404,
            content={"ok": False, "message": "Snapshot HUD no disponible."},
            headers={"Cache-Control": "no-store"},
        )
    return FileResponse(
        hud_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/start")
def api_start() -> JSONResponse:
    """Inicia la tarea configurada si no hay otra en ejecución."""
    result = runner.start()
    return JSONResponse(status_code=result["status_code"], content=result)


@app.post("/api/stop")
def api_stop() -> JSONResponse:
    """Ejecuta parada de emergencia sobre la tarea en curso."""
    result = runner.stop()
    return JSONResponse(status_code=result["status_code"], content=result)


if __name__ == "__main__":
    server_cfg = config.get("server", {})
    host = str(server_cfg.get("host", "0.0.0.0"))
    port = int(server_cfg.get("port", 8000))
    uvicorn.run(app, host=host, port=port)
