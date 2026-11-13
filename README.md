# Physical AI Hackathon Bilbao — Laboratorio & Toolkit Local (LeRobot v0.6+)

Kit de preparación, auditoría de datos, entrenamiento local en Apple Silicon (`mps`) y despliegue para el **Physical AI Hackathon Bilbao**, diseñado para operar **100% en local** con brazos robóticos **SO-100 / SO-101** y robots **LeKiwi**.

---

## Guía de Inicio Rápido (Clonado e Instalación desde Cero)

Este repositorio **no incluye archivos pesados, entornos virtuales, fotos privadas ni el código fuente de LeRobot**, ya que todo se descarga y construye automáticamente con el script de *bootstrap*:

```bash
# 1. Ejecutar el instalador automático (clona huggingface/lerobot, crea .venv e instala extras)
bash scripts/00_setup_env.sh

# 2. Activar el entorno virtual
source .venv/bin/activate

# 3. Verificar aceleración GPU Metal (MPS) y latencia de inferencia (<33.3 ms para 30 Hz)
python scripts/01_check_m5_env.py
```

---

## Documentación Principal del Repositorio

1. **[`PLAN_MAESTRO_HACKATHON.md`](PLAN_MAESTRO_HACKATHON.md):** Arquitectura 100% local en Apple Silicon (16 GB RAM) y reglas de optimización de memoria y cámaras.
2. **[`GUIA_PASO_A_PASO.md`](GUIA_PASO_A_PASO.md):** Bitácora técnica comentada con la traducción de cada concepto de robótica (`observation.state`, `action`, `chunk_size`, `preprocessor/postprocessor`, `ACT` vs `Diffusion`) al mundo de Data & ML Engineering.
3. **[`CHEATSHEET_BILBAO_48H.md`](CHEATSHEET_BILBAO_48H.md):** Manual cronológico de supervivencia para las 48h en Bilbao (desde fijar cámaras con cinta americana y calibrar motores hasta desplegar con `lerobot-rollout` y `DAgger`).

---

## Catálogo de Scripts del Laboratorio

| Script | Propósito | Comando de Ejecución |
| :--- | :--- | :--- |
| [`scripts/00_setup_env.sh`](scripts/00_setup_env.sh) | Clona `huggingface/lerobot`, crea `.venv` (Python 3.12/3.13) e instala dependencias (`feetech`, `lekiwi`, `pusht`, `diffusion`). | `bash scripts/00_setup_env.sh` |
| [`scripts/01_check_m5_env.py`](scripts/01_check_m5_env.py) | Verifica PyTorch `mps`, memoria unificada y mide los Hz máximos de inferencia de una red tipo ACT. | `python scripts/01_check_m5_env.py` |
| [`scripts/02_audit_dataset.py`](scripts/02_audit_dataset.py) | **Auditor Parquet de Datasets:** Descarga solo metadatos/Parquet (~450 KB) y detecta *frame drops*, *hesitations* y *jerk*. | `python scripts/02_audit_dataset.py --repo-id lerobot/svla_so101_pickplace` |
| [`scripts/03_train_local_m5.sh`](scripts/03_train_local_m5.sh) | Entrenamiento local en GPU `mps` con 3 modos: `smoke` (5 episodios), `so101` (50 episodios + Data Augmentation) y `sim` (`PushT`). | `bash scripts/03_train_local_m5.sh smoke 100` |
| [`scripts/04_camera_stress_test.py`](scripts/04_camera_stress_test.py) | Escanea puertos serie (`/dev/tty.usbmodem*`) y mide FPS reales, jitter `P95/P99` y deriva de auto-exposición en cámaras USB. | `python scripts/04_camera_stress_test.py --cameras 0 --duration 5` |
| [`scripts/05_vlm_orchestrator_template.py`](scripts/05_vlm_orchestrator_template.py) | Plantilla de arquitectura ganadora: máquina de estados que conmuta políticas ACT (`pick`, `place`, `recover`) leyendo el ángulo de la pinza. | `python scripts/05_vlm_orchestrator_template.py` |
| [`scripts/06_inspect_and_infer_checkpoint.py`](scripts/06_inspect_and_infer_checkpoint.py) | Carga un checkpoint entrenado (`preprocessor` $\rightarrow$ `ACT` $\rightarrow$ `postprocessor`) y compara los grados predichos contra el humano. | `python scripts/06_inspect_and_infer_checkpoint.py` |
| [`scripts/07_visualize_episode_and_augmentations.py`](scripts/07_visualize_episode_and_augmentations.py) | Exporta grabación interactiva 3D Rerun (`.rrd`), ejecuta `lerobot-imgtransform-viz` y genera lámina visual en `outputs/viz/`. | `python scripts/07_visualize_episode_and_augmentations.py` |
| [`scripts/08_eval_simulation_rollout.py`](scripts/08_eval_simulation_rollout.py) | Entrena y evalúa **Diffusion Policy** en el simulador físico `PushT`, generando métricas `eval_info.json` y vídeos MP4. | `python scripts/08_eval_simulation_rollout.py` |
| [`scripts/env_hackathon.sh`](scripts/env_hackathon.sh) | Plantilla de variables de entorno (`FOLLOWER_PORT`, `LEADER_PORT`, `ROBOT_CAMERAS`) para el día del evento. | `source scripts/env_hackathon.sh` |

---

## Seguridad y Privacidad (`.gitignore`)

El archivo [`.gitignore`](.gitignore) bloquea automáticamente:
- La carpeta `lerobot/` y el entorno `.venv/`.
- Toda la carpeta `outputs/` y cualquier imagen (`.jpg`, `.png`), vídeo (`.mp4`), grabación (`.rrd`), tabla (`.parquet`) o peso de modelo (`.safetensors`), garantizando que **ninguna foto capturada por las cámaras durante las pruebas ni ningún dato privado se suba al repositorio**.
