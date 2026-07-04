# first_voice_test — Project Context

## What this project is

A standalone local Arabic/English voice assistant powered by:
- **STT**: faster-whisper large-v3 (int8_float16) + Silero VAD + FRCRN denoiser
- **LLM**: qwen3.5:27b via Ollama (`/api/chat`, streaming, keep_alive:-1)
- **TTS**: OmniVoice (k2-fsa/OmniVoice, zero-shot voice cloning, 24kHz) — in-process, runs on GPU

It is a **general assistant** that recognizes and replies in Saudi & Egyptian Arabic dialects + English,
matching the speaker's dialect. **Fusha (MSA) is the default** Arabic dialect when the dialect is unclear.

> Full code reference: see **ARCHITECTURE.md** (file-by-file, function-by-function).

---

## Full voice pipeline

```
User speaks → Silero VAD → FRCRN denoiser → faster-whisper STT
  → dialect detect → qwen3.5:27b LLM (reply in the user's dialect)
  → OmniVoice TTS (per-dialect voice clip + language= dialect ID) → MP3 → audio plays back
```

Barge-in supported: user speaking while AI talks cancels the current turn immediately.

---

## Dialects

Recognized + replied-in: **Najdi, Hijazi, Egyptian, Fusha (MSA)**. **Fusha (MSA) is the default** when the
spoken dialect is unclear, or an Arabic reply is requested without naming a dialect. English input → English.
(Fusha is routed through the Saudi voice clip with `language="standard arabic"`; there is no separate spoken-Fusha
classifier — any Arabic that doesn't match a Najdi/Hijazi/Egyptian marker falls through to Fusha.)

Each turn the server decides three things from the user's words:
- **reply dialect** — a committed instruction to the LLM (`_detect_dialect` for spoken dialect,
  `_requested_dialect` for explicit requests).
- **voice clip** — Egyptian-routed turns use the Egyptian clip; Najdi/Hijazi/Fusha/English use the Saudi clip
  (`_VOICES` registry + `_resolve_voice` in tts_omnivoice_v1.py).
- **`language=` ID** — passed to `OmniVoice.generate()` to pin pronunciation
  (`egyptian arabic` / `najdi arabic` / `hijazi arabic` / `standard arabic`); this fixed the
  Saudi/Egyptian pronunciation mixing where some words came out in the wrong dialect.

STT is also biased toward dialect spelling via Whisper `hotwords` on the forced-Arabic re-pass.

**Tashkeel/diacritization was evaluated and dropped** — the CATT diacritizer is MSA-trained and mangles
Egyptian/dialect words; pronunciation is handled by `language=` + the reference voice instead.

---

## Server details

- Machine: `devserver`, user: `taha`
- GPU: **NVIDIA RTX 5090, 32GB VRAM**, CUDA 13.0
- Server port **8765**; Ollama **11434**. Browser reaches the server via an **SSH port-forward tunnel**
  (idle `1006/1005` WS closes are the tunnel dropping, not an app bug — the browser auto-reconnects).

---

## Tech stack

| Component | Choice |
|---|---|
| Speech-to-Text | faster-whisper large-v3 (int8_float16) + dialect `hotwords` on the forced-Arabic pass |
| VAD | Silero VAD |
| Denoiser | ClearVoice FRCRN (skipped for clips > 4s to avoid VRAM OOM) |
| Language Model | qwen3.5:27b via Ollama (locked) |
| Text-to-Speech | OmniVoice (k2-fsa, zero-shot) — per-dialect reference voice + `language=` dialect ID |
| Audio encoding | lameenc (PCM → MP3, 64kbps) |
| Python isolation | uv-managed venv (no Docker — no sudo access) |

---

## Project files

```
first_voice_test/
├── CLAUDE.md              ← you are here
├── ARCHITECTURE.md        ← full file-by-file / function-by-function code reference
├── SETUP.md               ← how to stand the system up on a machine (env, models, run)
├── README.md
├── requirements.txt       ← Python deps (torch installed separately for CUDA 13)
├── server.py              ← FastAPI WebSocket server (full pipeline + dialect routing)
├── tts_omnivoice_v1.py    ← OmniVoice TTS module (voice registry + voice/language params)
├── test_local.py          ← standalone TTS/LLM smoke test (no mic)
├── start_server.sh        ← launch script (Ollama + uvicorn, CUDA libs, SSH-tunnel note)
├── static/index.html      ← browser UI (AudioWorklet mic, WebAudio playback)
└── voices/
    ├── silma-tts-saudi-24k.wav            ← Saudi male — default voice (Najdi/Hijazi/Fusha/English)
    ├── omnivoice-tts-egyptian-24k-v3.wav  ← Egyptian voice (ACTIVE)
    ├── omnivoice-tts-egyptian-24k-v2.wav  ← Egyptian v2 (unused, superseded by v3)
    └── omnivoice-tts-egyptian-24k.wav     ← Egyptian v1 (unused, superseded by v2)
```

---

## Key decisions

- **No Docker** — no sudo access; GPU passthrough needs root. uv-managed venv instead (the `.venv` has no
  `pip`; use `uv pip`). torch/torchaudio `2.11.0+cu130` installed separately.
- **qwen3.5:27b (locked)** — good Arabic+English, fits in VRAM alongside OmniVoice. Fixed in both the UI and
  the server (a second LLM would OOM the GPU). Prior switchable-model version is on the `multi-engine-snapshot` branch.
- **Dialect = wording + voice + language**, decided per turn: committed LLM dialect instruction +
  per-dialect voice clip + OmniVoice `language=` ID. **Fusha (MSA) is the default** (Saudi clip + `standard arabic`).
- **`language=` param** pins OmniVoice pronunciation per dialect — fixed the Saudi/Egyptian word-mixing.
- **Per-dialect voice registry** (`_VOICES`): Egyptian→Egyptian clip, everything else→Saudi clip; extensible
  by dropping a WAV + one registry entry.
- **Tashkeel dropped** — CATT (MSA-trained) mangles dialect words; `language=` + the reference voice handle pronunciation.
- **MP3 per sentence** — browser `decodeAudioData` needs complete MP3 containers; one per WS message.
- **Sentence-level synthesis** — balances latency vs audio completeness.
- **FRCRN max 4s** — FRCRN VRAM scales with clip length; longer clips OOM. Whisper handles long clips fine without it.
- **Single-connection enforcement** — a new browser tab sends close code 4001 to the old tab, which does NOT reconnect.

---

## VRAM (measured)

| Process | VRAM |
|---|---|
| Ollama / qwen3.5:27b | ~15.7 GB (separate process; leaves ~10 GB free on the card) |
| OmniVoice TTS | **~2.4 GB** (NOT the 6–7 GB previously assumed) |
| faster-whisper large-v3 (int8_float16) | ~3 GB |
| FRCRN + Silero VAD | ~0.5 GB |

There is comfortably more headroom than the earlier "~31.3/32 GB" estimate implied.
