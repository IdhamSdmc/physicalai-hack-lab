#!/usr/bin/env bash
# ==============================================================================
# ARCHIVO DE VARIABLES DE HARDWARE PARA EL DÍA DEL HACKATHON (BILBAO)
# ==============================================================================
# [INSTRUCCIONES EL VIERNES 20 DE NOVIEMBRE EN LA MESA]:
# 1. Ejecuta `.venv/bin/lerobot-find-port` y pega aquí los dos puertos detectados.
# 2. Ejecuta `.venv/bin/lerobot-find-cameras opencv` y pega los índices de las cámaras.
# 3. En tu terminal haz: `source scripts/env_hackathon.sh`
#    A partir de ese momento, todos los comandos del Cheat Sheet usarán tus puertos.
# ==============================================================================

# 1. Activar entorno virtual de Python + LeRobot v0.6.2 + Fallback Metal MPS
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/.venv/bin/activate"
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 2. Tipo de brazo (En Bilbao suele ser so101_follower/so101_leader o so100_follower/so100_leader)
export ROBOT_TYPE="so101_follower"
export TELEOP_TYPE="so101_leader"
export ROBOT_ID="bilbao_follower"
export TELEOP_ID="bilbao_leader"

# 3. Puertos Serie USB en macOS (Cámbialos tras correr `lerobot-find-port`)
export FOLLOWER_PORT="/dev/tty.usbmodem585A0076841"
export LEADER_PORT="/dev/tty.usbmodem58760431551"

# 4. Índices de Cámaras OpenCV (Cámbialos tras correr `lerobot-find-cameras opencv`)
#    IMPORTANTE: Los nombres `up` y `side` (o `front` y `wrist`) DEBEN ser idénticos
#    al grabar (`lerobot-record`), al entrenar (`lerobot-train`) y al desplegar (`lerobot-rollout`).
export CAM_UP_IDX=0
export CAM_SIDE_IDX=1
export ROBOT_CAMERAS="{ up: {type: opencv, index_or_path: ${CAM_UP_IDX}, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: ${CAM_SIDE_IDX}, width: 640, height: 480, fps: 30} }"

# 5. Identificador de tu Dataset y Tarea
export HF_USER="local_team"
export DATASET_REPO="${HF_USER}/bilbao_task1"
export TASK_DESC="Pick the target object and place it in the zone"

echo "✅ Variables cargadas: ROBOT=${ROBOT_TYPE} | FOLLOWER=${FOLLOWER_PORT} | LEADER=${LEADER_PORT}"
