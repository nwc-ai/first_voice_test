#!/bin/bash
# Run this to start the voice server.
# Usage: bash /home/taha/first_voice_test/start_server.sh

VENV=/home/taha/first_voice_test/.venv/lib/python3.12/site-packages/nvidia
OLLAMA_NEW=/home/taha/.ollama_bin_new

export LD_LIBRARY_PATH=\
/usr/lib/x86_64-linux-gnu:\
$OLLAMA_NEW/lib/ollama:\
$VENV/cu13/lib:\
$VENV/cublas/lib:\
$VENV/cudnn/lib:\
$VENV/cuda_nvrtc/lib

# Start Ollama if not already running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama 0.30.2 with CUDA 13 GPU support..."
    OLLAMA_HOME=/home/taha/.ollama \
    PATH=$OLLAMA_NEW/bin:/usr/local/bin:/usr/bin:/bin \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
    $OLLAMA_NEW/bin/ollama serve > /tmp/ollama.log 2>&1 &
    sleep 8
    grep -E "GeForce|RTX|GPU|cuda_v" /tmp/ollama.log | head -3
    echo "Ollama ready."
fi

# Reduce CUDA memory fragmentation on shared GPU servers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec /home/taha/first_voice_test/.venv/bin/python /home/taha/first_voice_test/server.py
