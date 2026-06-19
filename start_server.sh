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

# Start Ollama if not already running.
# OLLAMA_FLASH_ATTENTION + OLLAMA_KV_CACHE_TYPE=q8_0 roughly HALVE the KV-cache VRAM
# (near-lossless), so we can run a larger context (LLM_NUM_CTX) in the same memory.
# NOTE: these only apply to a FRESH `ollama serve`. If Ollama is already running it is
# left as-is — run `ollama stop`/kill the serve process first to pick up these flags.
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama 0.30.2 with CUDA 13 GPU support (flash-attn + q8_0 KV cache)..."
    OLLAMA_HOME=/home/taha/.ollama \
    OLLAMA_FLASH_ATTENTION=1 \
    OLLAMA_KV_CACHE_TYPE=q8_0 \
    PATH=$OLLAMA_NEW/bin:/usr/local/bin:/usr/bin:/bin \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
    $OLLAMA_NEW/bin/ollama serve > /tmp/ollama.log 2>&1 &
    echo "Waiting for Ollama to be ready..."
    for i in $(seq 1 30); do
        curl -s http://localhost:11434/api/tags > /dev/null 2>&1 && break
        sleep 1
    done
    grep -E "GeForce|RTX|GPU|cuda_v|flash|kv" /tmp/ollama.log | head -3
    echo "Ollama ready."
fi

# Reduce CUDA memory fragmentation on shared GPU servers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec /home/taha/first_voice_test/.venv/bin/python /home/taha/first_voice_test/server.py

#LLM_NUM_CTX=16384 bash /home/taha/first_voice_test/start_server.sh

