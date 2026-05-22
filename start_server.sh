#!/bin/bash
# Run this to start the voice server.
# Usage: bash /home/taha/first_voice_test/start_server.sh

VENV=/home/taha/first_voice_test/.venv/lib/python3.12/site-packages/nvidia

export LD_LIBRARY_PATH=\
$VENV/cu13/lib:\
$VENV/cublas/lib:\
$VENV/cudnn/lib:\
$VENV/cuda_nvrtc/lib

# Start Ollama if not already running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama..."
    OLLAMA_HOME=/home/taha/.ollama \
    PATH=/home/taha/.ollama_bin:/usr/local/bin:/usr/bin:/bin \
    /home/taha/.ollama_bin/ollama serve > /tmp/ollama.log 2>&1 &
    sleep 6
    echo "Ollama ready."
fi

exec /home/taha/first_voice_test/.venv/bin/python /home/taha/first_voice_test/server.py
