# first_voice_test

Local Arabic/English voice assistant pipeline: Silero VAD + faster-whisper large-v3 (STT) → qwen3.5:27b via Ollama (LLM) → OmniVoice zero-shot voice cloning with CATT tashkeel (TTS). Supports English, Fusha, Najdi, and Arabic-English code-switching. Runs fully on-GPU (RTX 5090).

Start: `bash scripts/start_server.sh` — then open `http://<server>:8765/`.
