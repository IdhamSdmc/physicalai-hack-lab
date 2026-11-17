#!/usr/bin/env bash
# ==============================================================================
# SPRINT 2 — ENTRENAMIENTO LOCAL DE POLÍTICAS (ACT) EN APPLE M5 (16 GB RAM)
# ==============================================================================
# [¿QUÉ HACE ESTE SCRIPT?]
# Entrena una red neuronal "ACT" (Action Chunking with Transformers) usando la
# GPU Metal (`mps`) de tu MacBook M5 en LeRobot v0.6.2+ / v0.7.0.
#
# NOTA REALISTA DE TIEMPOS EN APPLE M5 (medido con 2 cámaras 640x480, batch=8):
#   ~1.65 s/paso -> 1.000 pasos (~28 min), 10.000 pasos (~4.5 h), 20.000 pasos (~9 h).
#   Para entrenamientos largos (20k–40k pasos), usa entrenamiento nocturno en M5,
#   Google Colab o Hugging Face Jobs (`--job.target=a10g-small`).
#
# [MODOS DISPONIBLES]
#   1. smoke  : Entrena 5 episodios (`--dataset.episodes="[0,1,2,3,4]"`) para validar flujo.
#   2. so101  : Entrena dataset completo de SO-101 + Data Augmentation.
#               Acepta 3er arg opcional para repo_id (o env DATASET_REPO).
#   3. sim    : Entrena y evalúa en vivo en el simulador físico 2D PushT.
#   4. resume : Retoma un entrenamiento cortado pasando el directorio output_dir.
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_TRAIN="${LEROBOT_TRAIN:-${REPO_ROOT}/.venv/bin/lerobot-train}"
if [ ! -x "${LEROBOT_TRAIN}" ]; then
  LEROBOT_TRAIN="$(command -v lerobot-train || true)"
fi
if [ -z "${LEROBOT_TRAIN}" ]; then
  echo "[ERROR] No se encontró lerobot-train en .venv/bin ni en PATH." >&2
  exit 1
fi

MODE="${1:-smoke}"
ARG2="${2:-100}"
TARGET_DATASET="${3:-${DATASET_REPO:-lerobot/svla_so101_pickplace}}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# [MARCA 1: FALLBACK TRANSPARENTE DE METAL (MPS)]
# ¿Por qué?: Si PyTorch encuentra alguna operación matemática menor aún no implementada
# en el chip Metal de Apple, la ejecuta en CPU automáticamente sin abortar el entrenamiento.
export PYTORCH_ENABLE_MPS_FALLBACK=1

if [ "$MODE" = "smoke" ]; then
  STEPS="${ARG2}"
  echo "=================================================================="
  echo " [MODO SMOKE TEST] Entrenando ACT con 5 episodios de SO-101       "
  echo " Dataset: ${TARGET_DATASET} | Device: mps | Batch: 8 | Steps: ${STEPS}"
  echo "=================================================================="
  # [MARCA 2: EL TRUCO DE --dataset.episodes]
  # En LeRobot v0.6+/v0.7.0, puedes filtrar qué episodios entrenar directamente por CLI
  # sin modificar el dataset. En el hackathon grabarás 5 episodios primero y
  # verificarás que `loss` y `l1_loss` bajan rápidamente antes de grabar 50.
  "${LEROBOT_TRAIN}" \
    --dataset.repo_id="${TARGET_DATASET}" \
    --dataset.episodes="[0, 1, 2, 3, 4]" \
    --dataset.video_backend=pyav \
    --policy.type=act \
    --policy.device=mps \
    --policy.push_to_hub=false \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --batch_size=8 \
    --num_workers=0 \
    --steps="${STEPS}" \
    --log_freq=20 \
    --env_eval_freq=0 \
    --save_checkpoint=true \
    --save_freq="${STEPS}" \
    --output_dir="outputs/train/act_so101_smoke_${TIMESTAMP}" \
    --trackio.enable=false

elif [ "$MODE" = "so101" ]; then
  STEPS="${ARG2}"
  echo "=================================================================="
  echo " [MODO SO-101 COMPLETO] Entrenando ACT + Data Augmentation        "
  echo " Dataset: ${TARGET_DATASET} | Device: mps | Chunk: 50 | Steps: ${STEPS}"
  echo "=================================================================="
  "${LEROBOT_TRAIN}" \
    --dataset.repo_id="${TARGET_DATASET}" \
    --dataset.video_backend=pyav \
    --policy.type=act \
    --policy.device=mps \
    --policy.push_to_hub=false \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --dataset.image_transforms.enable=true \
    --batch_size=8 \
    --num_workers=2 \
    --steps="${STEPS}" \
    --log_freq=50 \
    --env_eval_freq=0 \
    --save_checkpoint=true \
    --save_freq=500 \
    --output_dir="outputs/train/act_so101_${TIMESTAMP}" \
    --trackio.enable=false

elif [ "$MODE" = "sim" ]; then
  STEPS="${ARG2}"
  echo "=================================================================="
  echo " [MODO SIMULACIÓN] Entrenando ACT en PushT con evaluación en vivo "
  echo " Dispositivo: Apple M5 (mps) | Batch: 8 | Pasos: ${STEPS}         "
  echo "=================================================================="
  "${LEROBOT_TRAIN}" \
    --dataset.repo_id=lerobot/pusht \
    --dataset.video_backend=pyav \
    --env.type=pusht \
    --policy.type=act \
    --policy.device=mps \
    --policy.push_to_hub=false \
    --batch_size=8 \
    --num_workers=2 \
    --steps="${STEPS}" \
    --log_freq=50 \
    --env_eval_freq=500 \
    --eval.n_episodes=5 \
    --save_checkpoint=true \
    --save_freq=500 \
    --output_dir="outputs/train/act_pusht_${TIMESTAMP}" \
    --trackio.enable=false

elif [ "$MODE" = "resume" ]; then
  TRAIN_DIR="${ARG2}"
  CFG_PATH="${TRAIN_DIR}/checkpoints/last/pretrained_model/train_config.json"
  if [ ! -f "${CFG_PATH}" ]; then
    echo "[ERROR] No se encontró train_config.json en: ${CFG_PATH}" >&2
    echo "Uso: bash scripts/03_train_local_m5.sh resume outputs/train/<tu_run>" >&2
    exit 1
  fi
  echo "=================================================================="
  echo " [MODO RESUME] Retomando entrenamiento desde: ${CFG_PATH}"
  echo "=================================================================="
  "${LEROBOT_TRAIN}" \
    --config_path="${CFG_PATH}" \
    --resume=true

else
  echo "Modo desconocido: $MODE. Usa 'smoke', 'so101', 'sim' o 'resume'." >&2
  exit 1
fi
