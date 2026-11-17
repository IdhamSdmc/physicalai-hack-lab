#!/usr/bin/env bash
# ==============================================================================
# PASO 0 — INSTALACIÓN AUTOMÁTICA DESDE CERO (BOOTSTRAP DEL REPOSITORIO)
# ==============================================================================
# [¿PARA QUÉ SIRVE ESTE SCRIPT?]
# Permite que cualquier miembro de tu equipo clone este repositorio ligero y,
# con un solo comando (`bash scripts/00_setup_env.sh`), descargue LeRobot oficial
# (por defecto tag v0.7.0 indicado en la guía oficial de Bilbao), cree el entorno
# virtual `.venv` con Python 3.12/3.13 e instale todas las dependencias de hardware
# (`feetech`, `lekiwi`), simulación (`pusht`), ML (`training`, `diffusion`) y tests.
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_REF="${LEROBOT_REF:-v0.7.0}"
cd "$REPO_ROOT"

echo "=================================================================="
echo "  PHYSICAL AI HACKATHON BILBAO — BOOTSTRAP DEL ENTORNO (${LEROBOT_REF})"
echo "=================================================================="

# 1. Clonar Hugging Face LeRobot si aún no existe en la carpeta local
if [ ! -d "${REPO_ROOT}/lerobot" ]; then
  echo "[1/3] Clonando huggingface/lerobot (${LEROBOT_REF})..."
  git clone --depth 1 --branch "${LEROBOT_REF}" https://github.com/huggingface/lerobot.git "${REPO_ROOT}/lerobot"
else
  echo "[1/3] Carpeta 'lerobot/' ya existe. Omitiendo clonado."
  echo "      (Para alinear a ${LEROBOT_REF}: pip install 'lerobot[core_scripts,training,feetech]==0.7.0')"
fi

# 2. Crear entorno virtual (.venv) prefiriendo Python 3.12 (recomendado en guía SO-101) o 3.13
if [ ! -d "${REPO_ROOT}/.venv" ]; then
  echo "[2/3] Creando entorno virtual .venv..."
  if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv "${REPO_ROOT}/.venv"
  elif command -v python3.13 >/dev/null 2>&1; then
    python3.13 -m venv "${REPO_ROOT}/.venv"
  elif command -v uv >/dev/null 2>&1; then
    uv venv "${REPO_ROOT}/.venv" --python 3.12
  else
    echo "[ERROR] Necesitas Python 3.12 o 3.13 (o 'uv') instalado."
    exit 1
  fi
else
  echo "[2/3] Entorno virtual '.venv/' ya existe."
fi

# 3. Instalar LeRobot en modo editable con extras de hackathon + OpenCV + pytest
echo "[3/3] Instalando LeRobot y dependencias (feetech, lekiwi, pusht, diffusion, opencv, pytest)..."
"${REPO_ROOT}/.venv/bin/pip" install --upgrade pip
"${REPO_ROOT}/.venv/bin/pip" install -e "${REPO_ROOT}/lerobot[core_scripts,training,feetech,lekiwi,pusht,diffusion]" opencv-python pytest

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[!] AVISO: 'ffmpeg' no encontrado en PATH (guía Bilbao recomienda 'brew install ffmpeg' o 'conda install ffmpeg -c conda-forge')."
fi

echo "=================================================================="
echo "  ✅ INSTALACIÓN COMPLETADA. Activa tu entorno y valida con:      "
echo "     source .venv/bin/activate                                    "
echo "     python scripts/01_check_m5_env.py                            "
echo "     python -m unittest discover -s tests -v                      "
echo "=================================================================="
