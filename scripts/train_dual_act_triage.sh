#!/usr/bin/env bash
# ==============================================================================
# FASE 2 — ENTRENAMIENTO DUAL ACT EN APPLE SILICON (MPS)
# Entrena por separado `act_sano` (30 episodios Verdes limpios) y
# `act_contaminado` (30 episodios Rojos limpios) sobre el dataset de 64 episodios.
#
# Uso:
#   bash scripts/train_dual_act_triage.sh [sano|contaminado|both] [STEPS]
# Ejemplo:
#   bash scripts/train_dual_act_triage.sh both 15000
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TARGET="${1:-both}"
STEPS="${2:-15000}"
SAVE_FREQ="${SAVE_FREQ:-2500}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"

DATASET_REPO="${DATASET_REPO:-local/clasificador_objetos_20261121_171107}"
DATASET_ROOT="${DATASET_ROOT:-${HOME}/.cache/huggingface/lerobot/local/clasificador_objetos_20261121_171107}"

# 30 episodios Verdes limpios y 30 episodios Rojos limpios tras auditoría (excluidos 7, 11, 15, 18)
EPISODIOS_VERDES_CLEAN="[0, 1, 2, 3, 4, 5, 6, 16, 17, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55]"
EPISODIOS_ROJOS_CLEAN="[8, 9, 10, 12, 13, 14, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63]"

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONPATH="${REPO_ROOT}/lerobot/src:${PYTHONPATH:-}"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

run_train() {
  local job_name="$1"
  local episodes="$2"
  local out_dir="outputs/train/${job_name}"

  echo "=================================================================="
  echo "  ENTRENANDO ACT: ${job_name} | Steps=${STEPS} | Device=mps"
  echo "  Dataset : ${DATASET_REPO}"
  echo "  Root    : ${DATASET_ROOT}"
  echo "  Episodios: ${episodes}"
  echo "  Salida  : ${out_dir}"
  echo "=================================================================="

  "${PYTHON_BIN}" -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="${DATASET_REPO}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.episodes="${episodes}" \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=true \
    --policy.type=act \
    --policy.device=mps \
    --policy.push_to_hub=false \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --steps="${STEPS}" \
    --log_freq=50 \
    --env_eval_freq=0 \
    --save_checkpoint=true \
    --save_freq="${SAVE_FREQ}" \
    --output_dir="${out_dir}" \
    --job_name="${job_name}" \
    --trackio.enable=false
}

case "${TARGET}" in
  sano)
    run_train "act_sano" "${EPISODIOS_VERDES_CLEAN}"
    ;;
  contaminado)
    run_train "act_contaminado" "${EPISODIOS_ROJOS_CLEAN}"
    ;;
  both)
    run_train "act_sano" "${EPISODIOS_VERDES_CLEAN}"
    run_train "act_contaminado" "${EPISODIOS_ROJOS_CLEAN}"
    ;;
  *)
    echo "[ERROR] Objetivo desconocido: ${TARGET}. Usa 'sano', 'contaminado' o 'both'." >&2
    exit 1
    ;;
esac
