# Physical AI Hackathon Bilbao — Zero-Cross-Contamination Triage (LeRobot v0.7.0 / SO-101)

Kit de preparación, validación pre-vuelo, auditoría de datos, entrenamiento dual `ACT` en nube GPU y supervisor autónomo OpenCV en tiempo real para el proyecto **"Zero-Cross-Contamination Triage"** en el **Hackathon LeRobot Bilbao** (20–22 nov 2026, La Perrera Espazioa, Sabino Arana 50, Bilbao), alineado con la [Guía oficial del SO-101](https://hackathonbilbao.eus/guia-so-101).

---

## Repositorios y Recursos en Hugging Face

- **Políticas Dual ACT V4 (Producción):** [`coalva94/act_sano_v4`](https://huggingface.co/coalva94/act_sano_v4) · [`coalva94/act_contaminado_v4`](https://huggingface.co/coalva94/act_contaminado_v4)
- **Dataset Optimizado Contaminado V4 (`159` eps limpios, sin tiempos muertos):** [`coalva94/bilbao_triage_contaminado_v4`](https://huggingface.co/datasets/coalva94/bilbao_triage_contaminado_v4)
- **Dataset Completo Auditado V3 (`192` eps limpios / `200` grabados):** [`coalva94/bilbao_triage_complete_v3`](https://huggingface.co/datasets/coalva94/bilbao_triage_complete_v3)
- **Subconjuntos V3 (`96` + `96` eps):** [`coalva94/bilbao_triage_sano_v3`](https://huggingface.co/datasets/coalva94/bilbao_triage_sano_v3) · [`coalva94/bilbao_triage_contaminado_v3`](https://huggingface.co/datasets/coalva94/bilbao_triage_contaminado_v3)
- **Modelos V3 Previos:** [`coalva94/act_sano_v3`](https://huggingface.co/coalva94/act_sano_v3) · [`coalva94/act_contaminado_v3`](https://huggingface.co/coalva94/act_contaminado_v3)
- **Panel Web Dual-View (iPad + 3D + OpenCV HUD):** [`coalva94/web-io-robot`](https://github.com/coalva94/web-io-robot)
- **Supervisor Autónomo OpenCV + Sesión Persistente:** [`scripts/10_opencv_triage_supervisor.py`](scripts/10_opencv_triage_supervisor.py) (prioridad *Clean-First*, detección en vivo `~3.2 Hz`, `EarlyStopTracker`, re-mapeo automático ante golpes y retorno seguro `lift_first` a `HOME`).
- **Entrenamiento Dual ACT:** [`scripts/train_dual_act_triage.sh`](scripts/train_dual_act_triage.sh) y bitácora de captura en [`NOTAS_EPISODIOS_64.md`](NOTAS_EPISODIOS_64.md).

---

## Guía de Inicio Rápido (Clonado, Setup y Validación)

Este repositorio **no incluye archivos pesados, entornos virtuales, fotos privadas ni el código fuente de LeRobot**, ya que todo se construye con el script de *bootstrap*:

```bash
# 1. Ejecutar el instalador automático (clona huggingface/lerobot v0.7.0, crea .venv e instala extras + pytest)
bash scripts/00_setup_env.sh

# 2. Activar el entorno virtual
source .venv/bin/activate

# 3. Verificar portátil: Python 3.12/3.13, macOS 15+ (MPS), disco, ffmpeg/PyAV, HF auth y latencia ACT (<33.3 ms)
python scripts/01_check_m5_env.py

# 4. Cargar variables de hardware y ejecutar Preflight SO-101 (IDs, puertos, cámaras y contratos CLI v0.7.0)
source scripts/env_hackathon.sh
python scripts/09_preflight_so101.py

# 5. Ejecutar la suite de pruebas de validación del repositorio
python -m unittest discover -s tests -v
```

> **Alternativas oficiales recogidas en la guía de Bilbao:** Además de la terminal con `lerobot==0.7.0`, la organización contempla [**LeLab**](https://huggingface.co/docs/lerobot/lelab) (interfaz web local sobre LeRobot para SO-101) y **SenseCraft Robotics** (app de escritorio guiada para macOS/Windows).

---

## Documentación Principal del Repositorio

1. **[`PLAN_MAESTRO_HACKATHON.md`](PLAN_MAESTRO_HACKATHON.md):** Arquitectura híbrida (captura/smoke en Apple Silicon M5 16 GB + opción de entreno largo en M5 nocturno, Colab o HF Jobs) y reglas de hardware USB/alimentación.
2. **[`GUIA_PASO_A_PASO.md`](GUIA_PASO_A_PASO.md):** Bitácora técnica comentada con mediciones reales en M5 y notas de migración a LeRobot `v0.7.0`.
3. **[`CHEATSHEET_BILBAO_48H.md`](CHEATSHEET_BILBAO_48H.md):** Manual cronológico de supervivencia para las 48h en Bilbao (calibración, `lerobot-record`, auditoría, entrenamiento, `lerobot-rollout` `base`/`episodic`/`dagger` y rúbrica del jurado).

---

## Catálogo de Scripts del Laboratorio

| Script | Propósito | Comando de Ejecución |
| :--- | :--- | :--- |
| [`scripts/00_setup_env.sh`](scripts/00_setup_env.sh) | Clona `huggingface/lerobot` (`v0.7.0`), crea `.venv` (Python 3.12/3.13) e instala dependencias (`feetech`, `lekiwi`, `pusht`, `diffusion`, `pytest`). | `bash scripts/00_setup_env.sh` |
| [`scripts/01_check_m5_env.py`](scripts/01_check_m5_env.py) | Verifica Python, macOS >=15, disco libre (`>=20 GB`), `ffmpeg`/`PyAV`, LeRobot `v0.7.0` + CLIs, token HF, PyTorch `mps` y Hz de inferencia ACT. | `python scripts/01_check_m5_env.py` |
| [`scripts/02_audit_dataset.py`](scripts/02_audit_dataset.py) | **Auditor Parquet de Datasets:** Descarga solo metadatos/Parquet (~450 KB), detecta *frame drops*, *hesitations*, **tiempo muerto inicial**, *jerk* y exporta resumen para el jurado. | `python scripts/02_audit_dataset.py --repo-id lerobot/svla_so101_pickplace` |
| [`scripts/03_train_local_m5.sh`](scripts/03_train_local_m5.sh) | Entrenamiento local en GPU `mps` con 4 modos: `smoke`, `so101`, `sim` (`PushT`) y `resume`. | `bash scripts/03_train_local_m5.sh smoke 100` |
| [`scripts/04_camera_stress_test.py`](scripts/04_camera_stress_test.py) | Escanea puertos serie (`usbmodem`/`ttyACM`), valida índices únicos y mide FPS reales, jitter `P95/P99` y deriva de auto-exposición. | `python scripts/04_camera_stress_test.py --cameras 0 1 --duration 10` |
| [`scripts/05_vlm_orchestrator_template.py`](scripts/05_vlm_orchestrator_template.py) | Plantilla de orquestador multi-política (`pick`, `place`, `recover`) leyendo el servo #6 de la pinza en escala `RANGE_0_100` con límite de reintentos. | `python scripts/05_vlm_orchestrator_template.py` |
| [`scripts/06_inspect_and_infer_checkpoint.py`](scripts/06_inspect_and_infer_checkpoint.py) | Carga un checkpoint entrenado (`preprocessor` -> `ACT` -> `postprocessor`, compatible con `PolicyOutput` v0.7.0) y compara contra el ground truth humano. | `python scripts/06_inspect_and_infer_checkpoint.py` |
| [`scripts/07_visualize_episode_and_augmentations.py`](scripts/07_visualize_episode_and_augmentations.py) | Exporta grabación interactiva 3D Rerun (`.rrd`), ejecuta `lerobot-imgtransform-viz` y genera lámina visual en `outputs/viz/`. | `python scripts/07_visualize_episode_and_augmentations.py` |
| [`scripts/08_eval_simulation_rollout.py`](scripts/08_eval_simulation_rollout.py) | Entrena y evalúa **Diffusion Policy** en el simulador físico `PushT`, generando métricas `eval_info.json` y vídeos MP4. | `python scripts/08_eval_simulation_rollout.py` |
| [`scripts/09_preflight_so101.py`](scripts/09_preflight_so101.py) | **Validador Preflight SO-101:** Comprueba `env_hackathon.sh` (`ROBOT_ID != TELEOP_ID`, `rollout_*`), calibración de 6 motores, parseo real de `RecordConfig`/`RolloutConfig` y hardware USB. | `python scripts/09_preflight_so101.py` |
| [`scripts/env_hackathon.sh`](scripts/env_hackathon.sh) | Plantilla de variables de entorno (`FOLLOWER_PORT`, `LEADER_PORT`, `ROBOT_CAMERAS`, `DATASET_REPO`, `ROLLOUT_DATASET_REPO`, `DAGGER_DATASET_REPO`). | `source scripts/env_hackathon.sh` |

---

## Pruebas de Validación (`tests/`)

La carpeta [`tests/`](tests/) valida sin necesidad de brazo físico ni red:
- **Contratos CLI y documentación** ([`tests/test_hackathon_docs_and_cli_contract.py`](tests/test_hackathon_docs_and_cli_contract.py)): verifica que `env_hackathon.sh`, `03_train_local_m5.sh` y todos los comandos de `CHEATSHEET_BILBAO_48H.md` cumplen las reglas de LeRobot v0.7.0 (`rollout_` obligatorio en `lerobot-rollout`, prohibición de `eval_` en `lerobot-record`, `--dataset.no_stamp`, `ACTConfig` temporal ensembling).
- **Auditoría Parquet** ([`tests/test_audit_dataset.py`](tests/test_audit_dataset.py)): datasets sintéticos v3.0 para validar detección de caídas de FPS, *hesitation*, tiempo muerto inicial, *jerk* y reporte JSON para el jurado.
- **Preflight SO-101 y calibración** ([`tests/test_preflight_so101.py`](tests/test_preflight_so101.py)): validación de puertos, IDs distintos, cámaras, calibración de los 6 servos STS3215 y detección de calibraciones duplicadas.
- **Orquestador y utilidades de hardware/entorno** ([`tests/test_orchestrator_and_hardware_utils.py`](tests/test_orchestrator_and_hardware_utils.py)): máquina de estados en `RANGE_0_100`, detección de puertos serie macOS/Linux y métricas de stream de cámaras.

```bash
.venv/bin/python -m pytest -v
# o sin dependencias externas:
.venv/bin/python -m unittest discover -s tests -v
```

---

## Evaluación del Jurado y Entregas (Bilbao)

- **100 puntos base:** Demo en vivo — **5 intentos seguidos** sin tocar el robot (**30 pts**, 6/intento) · Datos y entrenamiento (**20 pts**) · Dificultad (**15 pts**) · Utilidad e impacto (**15 pts**) · Innovación (**10 pts**) · Presentación (**10 pts**, 3 min + vídeo de 1 min).
- **+5 puntos de bonus:** Dataset publicado en Hugging Face (**+3 pts**) y repositorio reproducible (**+2 pts**).

---

## Seguridad y Privacidad (`.gitignore`)

El archivo [`.gitignore`](.gitignore) bloquea automáticamente:
- La carpeta `lerobot/` y el entorno `.venv/`.
- Toda la carpeta `outputs/` y cualquier imagen (`.jpg`, `.png`), vídeo (`.mp4`), grabación (`.rrd`), tabla (`.parquet`) o peso de modelo (`.safetensors`), garantizando que **ninguna foto capturada por las cámaras durante las pruebas ni ningún dato privado se suba al repositorio**.
