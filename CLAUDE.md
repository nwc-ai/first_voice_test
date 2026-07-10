# first_voice_test — Project Context

## What this project is

A standalone testing ground for a local Arabic/English conversational voice pipeline, running entirely on the server's RTX 5090 GPU. The TTS module built here (`tts_omnivoice_v1.py`) is a drop-in replacement for `tts_habibi_v3.py` (edge_tts, online) inside the main `nwc-copilot` voice assistant project at `/home/taha/devproject`.

**Language scope:** English, Fusha (MSA), Najdi Arabic, and Arabic-English code-switching. (Hijazi was removed; Gulf is honored only on explicit request.)

---

## The full voice pipeline

```
Browser (AudioWorklet, 512-sample Float32 @16 kHz)
  → Silero VAD (server-side onset/end, pre-roll, barge-in)
  → FRCRN denoise (short clips only; DENOISE_ENABLED gate)
  → faster-whisper large-v3 (int8_float16, lang detect + remap tables)
  → qwen3.5:27b via Ollama /api/chat (3-turn rolling history, streamed)
  → tts_omnivoice_v1: sentence flushing → CATT tashkeel (Fusha only)
    → OmniVoice zero-shot clone (Saudi ref voice) → one MP3 per sentence
  → Browser: ordered decode, gapless playback, barge-in pause/resume
```

Barge-in: playback pauses instantly on speech onset; the turn is cancelled only when STT **accepts** the utterance — a rejected false trigger resumes playback (`speech_rejected` event).

---

## Server details

- Machine: `devserver`, user `taha`, GPU **RTX 5090 32 GB**, CUDA 13.0
- GitHub org: `nwc-ai` — **repo is public: never commit logs/transcripts**
- Run with: `bash /home/taha/first_voice_test/start_server.sh` (starts Ollama if needed)

## Tech stack

| Component | Choice | Notes |
|---|---|---|
| VAD | Silero VAD | server-side, per 512-sample chunk |
| Denoise | ClearVoice FRCRN_SE_16K | ≤4 s clips only; under evaluation for removal |
| STT | faster-whisper large-v3 | int8_float16 on CUDA |
| LLM | qwen3.5:27b via Ollama | **locked** — a second model would OOM the GPU |
| Diacritization | CATT tashkeel | Fusha replies only; `CATT_ENABLED=0` to disable |
| TTS | k2-fsa/OmniVoice | zero-shot voice clone, 24 kHz, fp16 |
| Audio encoding | lameenc (PCM → MP3) | complete MP3 per sentence (browser `decodeAudioData`) |
| Python isolation | venv (no Docker — no sudo) | GPU works out of the box |

Env knobs: `LLM_NUM_CTX` (default 8192), `CATT_ENABLED` (default 1), `OMNIVOICE_MODEL`, `OMNIVOICE_DEVICE`.

---

## TTS module contract (`tts_omnivoice_v1.py`)

1. Public API:
   ```python
   async def stream_tts_to_ws(token_gen, ws, cancel_event, on_first_audio=None, language=None) -> None
   ```
2. Every `ws.send_bytes()` sends a **complete self-contained MP3** (one per sentence)
3. JSON events: `{"event":"token","text":...}` per token, `{"event":"tts_end"}` at end
4. Sentence boundaries: `HARD_BREAK`, `SOFT_BREAK`, `SOFT_BREAK_MIN`, `FIRST_SOFT_MIN`, `_should_flush`
5. Cancellation checked at 3 points: token-loop top, before synthesis, after synthesis
6. `on_first_audio` fires exactly once before the first `ws.send_bytes`
7. GPU inference wrapped in `asyncio.to_thread()` — never block the event loop
8. Models loaded lazily via module-level cache + `threading.Lock`
9. `language` gates CATT tashkeel only (values like `"standard arabic"`/`"najdi arabic"`); it is not passed to OmniVoice

## Project files

```
first_voice_test/
├── CLAUDE.md              ← you are here
├── README.md
├── requirements.txt
├── server.py              ← FastAPI app: VAD, STT, routing, LLM, WS orchestration
├── tts_omnivoice_v1.py    ← TTS module (OmniVoice + CATT)
├── static/index.html      ← browser client
├── start_server.sh        ← starts Ollama (flash-attn, q8_0 KV) + the server
├── test_local.py          ← no-mic pipeline test (LLM → TTS → MP3 files)
├── voices/                ← Saudi reference clips for voice cloning
└── logs/                  ← interactions.jsonl (gitignored — private)
```

## Key decisions

- **LLM locked to qwen3.5:27b** — model selector removed; two LLMs don't fit VRAM alongside the in-process stack.
- **`num_predict: 300` stays** — very long answers may truncate mid-sentence; accepted tradeoff to keep voice replies bounded.
- **CATT gated to Fusha and applied per-sentence on the reply text** — MSA-trained; it mis-vocalizes Najdi words.
- **Najdi vs Fusha routing** is lexicon-based on normalized text (see `_NAJDI_MARKERS`/`_looks_najdi` and the MSA→Najdi glossary in the Najdi turn instruction).
- **MP3 format** — browser `decodeAudioData` needs complete containers, not raw PCM.
- **Sentence-level synthesis** — balances first-audio latency vs audio completeness.
- **No Docker** — no sudo; venv only.
- Dashboards: `/review` (latency + transcripts table), `/logs` (raw JSON).
