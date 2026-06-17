#!/bin/bash
# One-time setup for the Qwen3-TTS-KSA service (own venv, isolated deps).
# Usage: bash <path>/tts_qwen_service/setup.sh   (path-relative)
set -e
cd "$(cd "$(dirname "$0")" && pwd)"

echo "== creating venv (uv, python 3.12) =="
uv venv --python 3.12 .venv
PY=.venv/bin/python

echo "== installing torch/torchaudio for the RTX 5090 (CUDA 13) =="
uv pip install --python "$PY" \
    torch==2.11.0+cu130 torchaudio==2.11.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130

echo "== installing qwen-tts + API server (transformers==4.57.3 pinned, isolated) =="
uv pip install --python "$PY" -r requirements.txt

echo "== sanity: imports =="
"$PY" - <<'PYEOF'
import torch
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
from qwen_tts import Qwen3TTSModel
print("qwen_tts import OK")
PYEOF

echo
echo "Setup done. Start with:  bash $(pwd)/start.sh"
