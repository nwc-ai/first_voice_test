#!/bin/bash
# One-time setup for the OmniVoice service (own venv, isolated deps).
# Usage: bash <path>/tts_omnivoice_service/setup.sh   (path-relative)
set -e
cd "$(cd "$(dirname "$0")" && pwd)"

echo "== creating venv (uv, python 3.12) =="
uv venv --python 3.12 .venv
PY=.venv/bin/python

echo "== installing torch/torchaudio for the RTX 5090 (CUDA 13) =="
uv pip install --python "$PY" \
    torch==2.11.0+cu130 torchaudio==2.11.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130

echo "== installing omnivoice + API server =="
uv pip install --python "$PY" -r requirements.txt

echo "== sanity: imports =="
"$PY" - <<'PYEOF'
import torch
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
from omnivoice import OmniVoice
print("omnivoice import OK")
PYEOF

echo
echo "Setup done. Start with:  bash $(pwd)/start.sh"
