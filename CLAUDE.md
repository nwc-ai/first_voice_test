# first_voice_test — Project Context

## What this project is

A standalone testing ground to build and validate `tts_silma_v1.py` — a local Arabic/English TTS module powered by Silma TTS running on the server's RTX 5090 GPU.

This module is a drop-in replacement for `tts_habibi_v3.py` (which uses Microsoft's edge_tts online service) inside the main `nwc-copilot` voice assistant project at `/home/taha/devproject`.

---

## The full voice pipeline (what we are testing)

```
User speaks → STT (faster-whisper large-v3) → LLM (Qwen2.5:7b via Ollama) → TTS (Silma) → Audio plays back
```

Barge-in is supported: if the user speaks while the AI is talking, the AI stops immediately.

---

## Server details

- Machine: `devserver`
- User: `taha`
- GPU: **NVIDIA RTX 5090, 32GB VRAM**
- CUDA: 13.0
- GitHub org: `nwc-ai`
- Main project: `/home/taha/devproject` (repo: `nwc-ai/nwc-copilot`)

---

## Tech stack for this project

| Component | Choice | Reason |
|---|---|---|
| Speech-to-Text | faster-whisper large-v3 + Silero VAD | Open source, runs locally |
| Language Model | Qwen2.5:7b via Ollama | Fast, good Arabic+English, fits in VRAM |
| Text-to-Speech | Silma TTS (what we are building) | Local, Arabic-native, uses GPU |
| Audio encoding | lameenc (PCM → MP3) | No ffmpeg dependency |
| Python isolation | venv (no Docker — no sudo access) | Simple, GPU works out of the box |

---

## The main file to build: `tts_silma_v1.py`

### Hard requirements (do not violate)
1. Public API must exactly match `tts_habibi_v3.py`:
   ```python
   async def stream_tts_to_ws(
       token_gen,         # AsyncIterator[str]
       ws,                # FastAPI WebSocket
       cancel_event,      # asyncio.Event
       on_first_audio=None
   ) -> None
   ```
2. Every `ws.send_bytes()` must send a **complete self-contained MP3** (one per sentence)
3. JSON events: send `{"event":"token","text":token}` per token, `{"event":"tts_end"}` at end
4. Sentence boundary detection uses `HARD_BREAK`, `SOFT_BREAK`, `SOFT_BREAK_MIN`, `_should_flush` — copied verbatim from `tts_habibi_v3.py`
5. Cancellation checked at 3 points: (a) top of token loop, (b) before synthesis, (c) after synthesis
6. `on_first_audio` fires exactly once before the first `ws.send_bytes`
7. Silma inference wrapped in `asyncio.to_thread()` — never block the event loop
8. Model loaded lazily (not at import) using a module-level cache + `threading.Lock`

---

## Project files plan

```
first_voice_test/
├── CLAUDE.md              ← you are here
├── README.md
├── requirements.txt       ← Python dependencies
├── tts_silma_v1.py        ← main file to build
├── tts_habibi_v3.py       ← reference implementation (edge_tts based)
└── test_local.py          ← standalone test script
```

---

## Steps remaining

- [x] Phase 1: GitHub repo created (nwc-ai/first_voice_test), cloned to /home/taha/first_voice_test, opened in VSCode
- [ ] Phase 2: Check what's already installed (Python, pip, CUDA packages, faster-whisper, etc.)
- [ ] Phase 3: Install everything needed (Ollama + Qwen2.5:7b, venv, faster-whisper, Silero VAD, lameenc, Silma TTS)
- [ ] Phase 4: Build tts_silma_v1.py
- [ ] Phase 5: Test full pipeline

---

## Key decisions made

- **No Docker** — no sudo access on the server; GPU passthrough needs root. Using venv instead.
- **Qwen2.5:7b for now** — prototyping only; swap to 32b for production
- **Public repo** — user's choice
- **MP3 format** — browser's `decodeAudioData` requires complete MP3 containers, not raw PCM/WAV
- **Sentence-level synthesis** — not word-level or stream-level; balances latency vs audio completeness

---

## Reference: existing TTS (tts_habibi_v3.py)

Located at: `/home/taha/devproject/` (inside nwc-copilot project)
Uses: `edge_tts` with voice `ar-SA-ZariyahNeural` (Saudi Arabic female, Microsoft online service)
Why replacing: depends on internet; Silma runs locally on GPU with potentially better Arabic quality
