# Handoff a MacBook Pro — Physical AI Hackathon Bilbao (SO-101)

## 1. Estado actual del proyecto
- **Rama actual:** `feat/so101-guide-validation`
- **Hardware:** Brazos `so101_follower` (`ROBOT_ID=bilbao_follower`) y `so101_leader` (`TELEOP_ID=bilbao_leader`), 2 cámaras USB (`top` cenital + `wrist` en la pinza).
- **Tarea objetivo:** `"Pick red cube and put in Almacen"` (dataset sugerido: `local_team/bilbao_cubes_v1`).
- **Calibración guardada:** Respaldada en `calibration_backup/` (`bilbao_follower.json` y `bilbao_leader.json`).
  - *Nota importante:* `wrist_roll` tiene calibración parcial intencional (`[1169, 3261]` y `[101, 3912]`) por un tope mecánico del brazo; los avisos de `09_preflight_so101.py` sobre `wrist_roll` son esperados y no impiden operar.

## 2. Problemas ya diagnosticados en el Mac anterior (no repetir)
1. **Ancho de banda USB con 2 cámaras:** Conectar ambas cámaras UVC (`top` y `wrist`) al mismo hub USB provoca `VIDEOIO(AVFOUNDATION): raised unknown C++ exception!` o `TimeoutError: latest frame is too old`.
   - **Regla física:** En el MacBook Pro, conecta **al menos una cámara (`top`) directa a un puerto USB-C del portátil** y la otra (`wrist`) en otro puerto distinto (o en el hub alimentado junto a los brazos).
2. **Rerun durante grabación (`--display_data=true`):**
   - Añade carga de CPU/gRPC que dispara timeouts de cámara.
   - Roba el foco del teclado en macOS si la Terminal no tiene permiso de Accesibilidad (`n` / `r` / `q` dejan de responder).
   - **Regla:** Grabar siempre con `--display_data=false --dataset.streaming_encoding=true --dataset.encoder_threads=2 --dataset.num_image_writer_threads_per_camera=2`.
3. **Expansión de `~` en zsh:** En `--dataset.root=...`, usar `"${HOME}/.cache/..."` en lugar de `~/...`.

## 3. Checklist de arranque rápido en el MacBook Pro
1. **Instalar entorno (si no está ya `.venv/`):**
   ```bash
   bash scripts/00_setup_env.sh
   # Opcional si vas a usar SmolVLA:
   uv pip install -e "./lerobot[smolvla]"
   ```
2. **Restaurar calibraciones sin recalibrar:**
   ```bash
   mkdir -p ~/.cache/huggingface/lerobot/calibration/robots/so_follower \
            ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader
   cp calibration_backup/robots/so_follower/bilbao_follower.json \
      ~/.cache/huggingface/lerobot/calibration/robots/so_follower/
   cp calibration_backup/teleoperators/so_leader/bilbao_leader.json \
      ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/
   ```
3. **Detectar puertos e índices de cámara en el nuevo Mac:**
   ```bash
   source scripts/env_hackathon.sh
   lerobot-find-port
   lerobot-find-cameras opencv
   ```
4. **Actualizar `scripts/env_hackathon.sh`** con los nuevos `FOLLOWER_PORT`, `LEADER_PORT`, `CAM_TOP_IDX` y `CAM_WRIST_IDX` (confirmando visualmente cuál foto en `outputs/captured_images/` es `top` y cuál es `wrist`).
5. **Validar todo antes de grabar:**
   ```bash
   unset FOLLOWER_PORT LEADER_PORT CAM_TOP_IDX CAM_WRIST_IDX
   source scripts/env_hackathon.sh
   python scripts/04_camera_stress_test.py --cameras ${CAM_TOP_IDX} ${CAM_WRIST_IDX} --duration 10
   python scripts/09_preflight_so101.py --require-calibration --check-hardware
   ```
6. **Grabar 20 episodios limpios:**
   ```bash
   export TASK_DESC="Pick red cube and put in Almacen"
   export DATASET_NAME="bilbao_cubes_v1"
   source scripts/env_hackathon.sh

   lerobot-record \
     --robot.type=${ROBOT_TYPE} \
     --robot.port=${FOLLOWER_PORT} \
     --robot.id=${ROBOT_ID} \
     --robot.cameras="${ROBOT_CAMERAS}" \
     --teleop.type=${TELEOP_TYPE} \
     --teleop.port=${LEADER_PORT} \
     --teleop.id=${TELEOP_ID} \
     --dataset.repo_id=${DATASET_REPO} \
     --dataset.single_task="${TASK_DESC}" \
     --dataset.num_episodes=20 \
     --dataset.episode_time_s=20 \
     --dataset.reset_time_s=8 \
     --dataset.no_stamp=true \
     --dataset.push_to_hub=false \
     --dataset.streaming_encoding=true \
     --dataset.encoder_threads=2 \
     --dataset.num_image_writer_threads_per_camera=2 \
     --display_data=false
   ```
7. **Auditar y entrenar ACT en `mps`:**
   ```bash
   python scripts/02_audit_dataset.py --repo-id ${DATASET_REPO} --local-dir "${HOME}/.cache/huggingface/lerobot/${DATASET_REPO}"
   ```
