#!/bin/bash
# Start the Qwen3-TTS-KSA service on port 8772.
# Usage: bash <path>/tts_qwen_service/start.sh   (run setup.sh once first)
# Path-relative: works wherever this folder lives.

SVC="$(cd "$(dirname "$0")" && pwd)"
NV="$SVC/.venv/lib/python3.12/site-packages/nvidia"

export LD_LIBRARY_PATH=\
/usr/lib/x86_64-linux-gnu:\
$NV/cu13/lib:\
$NV/cublas/lib:\
$NV/cudnn/lib:\
$NV/cuda_nvrtc/lib

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$SVC/.venv/bin/python" "$SVC/service.py"
