#!/usr/bin/env bash
# ==============================================================================
# ARCHIVO DE VARIABLES DE HARDWARE PARA EL DÍA DEL HACKATHON (BILBAO)
# ==============================================================================
# [INSTRUCCIONES EL VIERNES 20 DE NOVIEMBRE EN LA MESA — La Perrera, 16:00]:
# 1. Conecta las fuentes etiquetadas (5V Leader / 12V Follower — NUNCA intercambiar).
# 2. Ejecuta `lerobot-find-port` y pega aquí los dos puertos detectados.
# 3. Ejecuta `lerobot-find-cameras opencv` y pega los índices de las cámaras.
# 4. Carga variables y valida antes de tocar el robot:
#      source scripts/env_hackathon.sh
#      python scripts/09_preflight_so101.py --check-hardware
# ==============================================================================

# 1. Activar entorno virtual local si existe (compatible también con conda `lerobot`)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${REPO_ROOT}/.venv/bin/activate" ] && [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ]; then
  source "${REPO_ROOT}/.venv/bin/activate"
fi
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HACK_DEVICE="${HACK_DEVICE:-mps}"

# 2. Tipo e IDs de los brazos (CRÍTICO: ROBOT_ID y TELEOP_ID DEBEN ser distintos;
#    si coinciden, LeRobot mezcla calibraciones sin lanzar error)
export ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
export TELEOP_TYPE="${TELEOP_TYPE:-so101_leader}"
export ROBOT_ID="${ROBOT_ID:-bilbao_follower}"
export TELEOP_ID="${TELEOP_ID:-bilbao_leader}"

# 3. Puertos Serie USB (macOS: /dev/tty.usbmodem* | Ubuntu: /dev/ttyACM*)
export FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/tty.usbmodem585A0076841}"
export LEADER_PORT="${LEADER_PORT:-/dev/tty.usbmodem58760431551}"

# 4. Índices y nombres de Cámaras OpenCV (verificados con `lerobot-find-cameras`)
#    IMPORTANTE: Los nombres (`top` y `wrist`, o `up` y `side`) DEBEN ser idénticos
#    al grabar (`lerobot-record`), entrenar (`lerobot-train`) y desplegar (`lerobot-rollout`).
export CAM_TOP_IDX="${CAM_TOP_IDX:-${CAM_UP_IDX:-1}}"
export CAM_WRIST_IDX="${CAM_WRIST_IDX:-${CAM_SIDE_IDX:-0}}"
export ROBOT_CAMERAS="{ top: {type: opencv, index_or_path: ${CAM_TOP_IDX}, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: ${CAM_WRIST_IDX}, width: 640, height: 480, fps: 30} }"

# 5. Usuario Hugging Face, Datasets y Política
#    - Detecta tu usuario automáticamente si ejecutaste `hf auth login`; si no, usa `local_team`.
#    - REGLA v0.7.0: Todo dataset grabado con `lerobot-rollout` (episodic/dagger/sentry/highlight)
#      DEBE empezar por `rollout_`.
if [ -z "${HF_USER:-}" ]; then
  if command -v hf >/dev/null 2>&1; then
    HF_USER="$(NO_COLOR=1 hf auth whoami 2>/dev/null | awk -F': *' 'NR==1 {print $2}' || true)"
  fi
  HF_USER="${HF_USER:-local_team}"
fi
export HF_USER
export DATASET_NAME="${DATASET_NAME:-bilbao_task1}"
export DATASET_REPO="${HF_USER}/${DATASET_NAME}"
export ROLLOUT_DATASET_REPO="${HF_USER}/rollout_${DATASET_NAME}"
export DAGGER_DATASET_REPO="${HF_USER}/rollout_${DATASET_NAME}_dagger"
export POLICY_REPO="${HF_USER}/act_${DATASET_NAME}"
export TASK_DESC="${TASK_DESC:-Pick the target object and place it in the box}"

if [ -z "${HACK_ENV_QUIET:-}" ]; then
  echo "✅ Variables cargadas: ROBOT=${ROBOT_TYPE} (${ROBOT_ID}) | TELEOP=${TELEOP_TYPE} (${TELEOP_ID})"
  echo "   Puertos: FOLLOWER=${FOLLOWER_PORT} | LEADER=${LEADER_PORT}"
  echo "   Dataset: ${DATASET_REPO} | Rollout: ${ROLLOUT_DATASET_REPO} | DAgger: ${DAGGER_DATASET_REPO}"
fi
