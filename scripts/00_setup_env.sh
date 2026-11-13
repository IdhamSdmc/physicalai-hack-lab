#!/usr/bin/env bash
# ==============================================================================
# PASO 0 — INSTALACIÓN AUTOMÁTICA DESDE CERO (BOOTSTRAP DEL REPOSITORIO)
# ==============================================================================
# [¿PARA QUÉ SIRVE ESTE SCRIPT?]
# Permite que cualquier miembro de tu equipo clone este repositorio ligero y,
# con un solo comando (`bash scripts/00_setup_env.sh`), descargue LeRobot oficial,
# cree el entorno virtual `.venv` con Python 3.12/3.13 e instale todas las
# dependencias de hardware (`feetech`, `lekiwi`), simulación (`pusht`) y ML.
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=================================================================="
echo "  PHYSICAL AI HACKATHON BILBAO — BOOTSTRAP DEL ENTORNO LOCAL      "
echo "=================================================================="

# 1. Clonar Hugging Face LeRobot si aún no existe en la carpeta local
if [ ! -d "${REPO_ROOT}/lerobot" ]; then
  echo "[1/3] Clonando repositorio oficial huggingface/lerobot..."
  git clone --depth 1 https://github.com/huggingface/lerobot.git "${REPO_ROOT}/lerobot"
else
  echo "[1/3] Carpeta 'lerobot/' ya existe. Omitiendo clonado."
fi

# 2. Crear entorno virtual (.venv) con Python 3.12 o 3.13 (requerido por LeRobot v0.6+)
if [ ! -d "${REPO_ROOT}/.venv" ]; then
  echo "[2/3] Creando entorno virtual .venv..."
  if command -v python3.13 >/dev/null 2>&1; then
    python3.13 -m venv "${REPO_ROOT}/.venv"
  elif command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv "${REPO_ROOT}/.venv"
  elif command -v uv >/dev/null 2>&1; then
    uv venv "${REPO_ROOT}/.venv" --python 3.12
  else
    echo "[ERROR] Necesitas Python 3.12 o 3.13 (o 'uv') instalado."
    exit 1
  fi
else
  echo "[2/3] Entorno virtual '.venv/' ya existe."
fi

# 3. Instalar LeRobot en modo editable con extras de hackathon + OpenCV
echo "[3/3] Instalando LeRobot y dependencias (feetech, lekiwi, pusht, diffusion, opencv)..."
"${REPO_ROOT}/.venv/bin/pip" install --upgrade pip
"${REPO_ROOT}/.venv/bin/pip" install -e "${REPO_ROOT}/lerobot[core_scripts,training,feetech,lekiwi,pusht,diffusion]" opencv-python

echo "=================================================================="
echo "  ✅ INSTALACIÓN COMPLETADA. Activa tu entorno con:               "
echo "     source .venv/bin/activate                                    "
echo "     python scripts/01_check_m5_env.py                            "
echo "=================================================================="
