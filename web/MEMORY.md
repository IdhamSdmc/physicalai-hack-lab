# MEMORY.md — Bitácora del proyecto

Actualizar al terminar cada paso. Mantenerlo corto.

## Decisiones tomadas
- El panel es un mando: iniciar tarea, parar tarea, ver estado.
- La tarea asignada por defecto es la que el robot ya sabe hacer; el panel solo la lanza.
- El panel es opcional y no interfiere con LeLab. No hay modo monitor.
- Stack: FastAPI + Uvicorn + PyYAML y un `index.html` sin build.
- `mock_mode: true` por defecto.
- Los brazos ya están calibrados; el entrenamiento se hace en la Mac.

## Hecho
- [x] Definido el alcance y las reglas (`AGENTS.md`).
- [x] README y explicación corta preparados.
- [x] `config.example.yaml` (y `config.yaml` local en `mock_mode: true`).
- [x] `runner.py` (lanzar, parar con `os.killpg` y vigilar el subproceso con `threading.Lock`).
- [x] `main.py` (servidor FastAPI, montaje `/static` y endpoints `/`, `/api/status`, `/api/start`, `/api/stop`).
- [x] `static/` (`three@0.170.0` ESM en `static/vendor/three/` + `static/vendor/utils/BufferGeometryUtils.js` 100% offline, y `eroski-scene.js` con `RoomEnvironment`, `PMREMGenerator`, `ACESFilmicToneMapping`, `PCFSoftShadowMap`, `OrbitControls`, manzanas `LatheGeometry` PBR y brazo SO-101).
- [x] `index.html` (panel Eroski · Frutería Control de Calidad sin build, `importmap` + `<script type="module">`, responsive iPad A16 vertical/horizontal y móvil, visor 3D, Iniciar, STOP y registro plegable).
- [x] Pruebas QA en `mock_mode: true`: todos los módulos `/static/vendor/three/*`, `/static/vendor/utils/BufferGeometryUtils.js` y `/static/eroski-scene.js` servidos con HTTP 200, `/`, `/api/status`, `/api/start` (200 y 409 simultáneo), `/api/stop` (`detenido` + `os.killpg`), sintaxis ESM (`node --input-type=module --check`) y 0 URLs externas (`http/https`) en `index.html` y `static/eroski-scene.js`.

- [x] Migración a React + Vite + Tailwind CSS (`src/App.jsx`, `src/index.css`) con soporte Netlify + FastAPI.
- [x] Rediseño Liquid Glass iPadOS inspirado en Stitch con tokens CSS optimizados (Eroski Red, Fresh Green, Slate Ink, JetBrains Mono, Plus Jakarta Sans).
- [x] Implementadas pestañas: Triaje en Vivo (3D, control de ciclo, telemetría), Calibración (6-DOF, vacío 94 kPa, cámaras) y Lotes & Trazabilidad (balance de calidad, UV-C 254nm, exportación CSV).
- [x] Adaptación responsive verificada: 2 columnas en Desktop (60% visor 3D / 40% consola) y tarjeta vertical táctil en iPad/Mobile.
- [x] Verificación visual automatizada con subagente de navegador en `http://localhost:8000` (Desktop 1280x800 e iPad 820x1180).

- [x] Rediseño Web Dashboard real (eliminado el marco/chasis simulado de iPad, notch falso, batería y barra de home inferior; convertido en app web industrial full-width con navbar superior y layout responsive 2 columnas).
- [x] Renombrado el supermercado a "SUPERMARKET 5.12" en navbar, título, ficha de lote y footer.
- [x] Corregido el dock inferior según directrices Apple HIG: etiquetas concisas (`Triaje`, `Calibración`, `Lotes`) en 1 sola línea sin salto de renglón en iPhone (390px), iPad (820px) y Desktop (1440px).
- [x] Asegurado que el título "SUPERMARKET 5.12" se mantenga siempre en una sola línea en móviles estrechos (`whitespace-nowrap`).

- [x] Desplegado con éxito en Netlify: `https://eroski-so101-panel.netlify.app` (Deploy ID: `6b01bb751ad9f45f5989f0d1`).
- [x] Servidor local en puerto 8000 detenido y liberado a petición del usuario.
- [x] Conexión del panel (`/web`) con `physicalai-hack-lab`:
  - `config.example.yaml` y `config.yaml` actualizados con `working_dir`, comando real de `scripts/10_opencv_triage_supervisor.py` (`env PYTHONUNBUFFERED=1 PYTHONPATH=lerobot/src PYTORCH_ENABLE_MPS_FALLBACK=1 ...`) y rutas relativas de telemetría (`outputs/triage_hud_latest.jpg`, `outputs/triage_scene_map.json`).
  - `runner.py`: validación de `working_dir`, margen de espera (`stop_grace_seconds: 3.0`) tras `os.killpg(..., SIGTERM)` para respetar el cierre de `lerobot_rollout` y liberación limpia de torque en el supervisor antes de escalar a `SIGKILL`, lectura segura de `triage_scene_map.json` y detección de eventos clave en `stdout`.
  - `main.py`: añadido `GET /api/hud.jpg` con `Cache-Control: no-store` (Python 3.9 compatible con `typing.Union`).
  - `src/App.jsx` + `npm run build` (`dist/`): soporte para alternar entre `Cámara HUD` en vivo y `Visor 3D`, conteos reales por zona (`ZONA_A`, `B_SUP`, `B_INF`), `next_order` y último evento del supervisor.
  - Verificadas Fase 1 (`mock_mode: true`: `/api/status`, `/api/start` 200/409, `/api/stop` `detenido`) y Fase 2 (`mock_mode: false` con `--dry-run`: escaneo cámara top índice 1, mapa/HUD generados, `state: libre`, `/api/hud.jpg` HTTP 200).

## Pendiente
- [ ] Fase 3: quitar `--dry-run` de `config.yaml` cuando los checkpoints V2 (`act_sano`, `act_contaminado`) estén descargados en `outputs/train/` y probar con el robot físico SO-101.

## Notas
(Aquí apunta lo que vaya saliendo: errores, dudas, cambios de plan.)
