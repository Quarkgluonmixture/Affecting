#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-.venv}

if ! $PYTHON_BIN -m venv --clear --system-site-packages "$VENV_DIR"; then
  BACKUP_DIR="${VENV_DIR}.bak.$(date +%s)"
  if [ -d "$VENV_DIR" ]; then
    mv "$VENV_DIR" "$BACKUP_DIR"
    echo "[setup] existing venv moved to $BACKUP_DIR"
  fi
  $PYTHON_BIN -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  "transformers>=4.51.0" \
  accelerate \
  "datasets<4" \
  bitsandbytes \
  pandas \
  tqdm \
  regex \
  huggingface_hub

echo "[setup] done. activate via: source $VENV_DIR/bin/activate"
