# first_voice_test
Local Arabic/English voice assistant, **Saudi & Egyptian dialect-aware** (Najdi · Hijazi · Egyptian · Fusha; Egyptian is the default). Pipeline: Silero VAD → faster-whisper large-v3 STT → qwen3.5:27b LLM (Ollama) → OmniVoice TTS (per-dialect reference voice + `language=` dialect ID), 24 kHz.

See **CLAUDE.md** (project context), **ARCHITECTURE.md** (full code reference), and **SETUP.md** (how to run it).
