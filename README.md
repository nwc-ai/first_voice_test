# first_voice_test

Local Arabic/English voice assistant pipeline: Silero VAD + faster-whisper large-v3 (STT) → dialect router → qwen3.5:27b via Ollama (LLM; opt-in Fanar-2 override via `LLM_MODEL`) → TTS split by dialect: OmniVoice zero-shot voice cloning with CATT tashkeel for Fusha/Najdi/English, VoiceTut (Egyptian-tuned OmniVoice) for Egyptian. Supports English, Fusha, Najdi, Egyptian, and Arabic-English code-switching. Runs fully on-GPU (RTX 5090).

Start: `bash scripts/start_server.sh` — then open `http://<server>:8765/`.
