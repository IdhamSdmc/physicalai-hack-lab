#!/usr/bin/env bash
# ==============================================================================
# SPRINT 2 — ENTRENAMIENTO LOCAL DE POLÍTICAS (ACT) EN APPLE M5 (16 GB RAM)
# ==============================================================================
# [¿QUÉ HACE ESTE SCRIPT?]
# Entrena una red neuronal "ACT" (Action Chunking with Transformers) usando la
# GPU Metal (`mps`) de tu MacBook M5 en LeRobot v0.6.2.
#
# [MODOS DISPONIBLES]
#   1. smoke : Entrena con SOLO 5 episodios del brazo SO-101 (`--dataset.episodes="[0,1,2,3,4]"`).
#              Ideal para el "Smoke Test" de la Hora 4 en el hackathon y para nuestro laboratorio.
#   2. so101 : Entrena con los 50 episodios completos del brazo SO-101 + Data Augmentation.
#   3. sim   : Entrena y evalúa en vivo en el simulador físico 2D PushT.
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_TRAIN="${REPO_ROOT}/.venv/bin/lerobot-train"
MODE="${1:-smoke}"
STEPS="${2:-100}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# [MARCA 1: FALLBACK TRANSPARENTE DE METAL (MPS)]
# ¿Por qué?: Si PyTorch encuentra alguna operación matemática menor aún no implementada
# en el chip Metal de Apple, la ejecuta en CPU automáticamente sin abortar el entrenamiento.
export PYTORCH_ENABLE_MPS_FALLBACK=1

if [ "$MODE" = "smoke" ]; then
  echo "=================================================================="
  echo " [MODO SMOKE TEST] Entrenando ACT con 5 episodios de SO-101       "
  echo " Dispositivo: Apple M5 (mps) | Batch: 8 | Pasos: ${STEPS}         "
  echo "=================================================================="
  # [MARCA 2: EL TRUCO DE --dataset.episodes]
  # En LeRobot v0.6+, puedes filtrar qué episodios entrenar directamente por CLI
  # sin modificar el dataset. En el hackathon grabarás 5 episodios primero y
  # verificarás que `loss` y `l1_loss` bajan rápidamente antes de grabar 50.
  "${LEROBOT_TRAIN}" \
    --dataset.repo_id=lerobot/svla_so101_pickplace \
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
  echo "=================================================================="
  echo " [MODO SO-101 COMPLETO] Entrenando ACT (50 episodios + Augment)   "
  echo " Dispositivo: Apple M5 (mps) | Chunk: 50 | Pasos: ${STEPS}        "
  echo "=================================================================="
  "${LEROBOT_TRAIN}" \
    --dataset.repo_id=lerobot/svla_so101_pickplace \
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
else
  echo "Modo desconocido: $MODE. Usa 'smoke', 'so101' o 'sim'."
  exit 1
fi
