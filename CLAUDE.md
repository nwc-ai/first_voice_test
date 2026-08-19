# first_voice_test — Project Context

## What this project is

A standalone testing ground for a local Arabic/English conversational voice pipeline, running entirely on the server's RTX 5090 GPU. The TTS module built here (`pipeline/tts_omnivoice_v1.py`) is a drop-in replacement for the SILMA-based TTS backend (`voice/app/pipeline/tts.py`) inside the main `nwc-copilot` voice assistant project at `/home/taha/devproject`.

**Language scope:** English, Fusha (MSA), Najdi Arabic, Egyptian Arabic, and Arabic-English code-switching. Hijazi and Gulf/Khaleeji remain unsupported — a request for either falls through to Fusha. **Fusha and Najdi are a protected baseline**: their prompting, generation and OmniVoice TTS configuration must not change as part of Egyptian work (a previous Egyptian attempt degraded Najdi — the current design isolates Egyptian per-turn instead).

---

## The full voice pipeline

```
Browser (AudioWorklet, 512-sample Float32 @16 kHz)
  → Silero VAD (server-side onset/end, pre-roll, barge-in)
  → FRCRN denoise (short clips only; DENOISE_ENABLED gate)
  → faster-whisper large-v3 (int8_float16, lang detect + remap tables)
  → dialect router (routing.route_arabic: Najdi-exclusive > Egyptian-exclusive
    > shared markers→Najdi > Fusha; shared اللي/عشان/لسه/يلا never decide)
  → qwen3.5:27b via Ollama /api/chat (3-turn rolling history, streamed;
    Arabic↔Arabic dialect switches withhold other-dialect history — see below;
    LLM_MODEL env = opt-in Fanar-2 override for A/B)
  → TTS split by route:
      Fusha/Najdi/English → tts_omnivoice_v1: sentence flushing → CATT tashkeel
        (Fusha only) → OmniVoice zero-shot clone (Saudi ref) → one MP3/sentence
      Egyptian → tts_voicetut_v1: same scaffold, VoiceTut checkpoint (Egyptian-
        tuned OmniVoice), Egyptian ref voice, NO CATT, lexical repairs (EGY_REPAIRS)
  → Browser: ordered decode, gapless playback, barge-in pause/resume
```

Barge-in: playback pauses instantly on speech onset; the turn is cancelled only when STT **accepts** the utterance — a rejected false trigger resumes playback (`speech_rejected` event).

---

## Server details

- Machine: `devserver`, user `taha`, GPU **RTX 5090 32 GB**, CUDA 13.0
- GitHub org: `nwc-ai` — **repo is public: never commit logs/transcripts**
- Run with: `bash /home/taha/first_voice_test/scripts/start_server.sh` (starts Ollama if needed)

## Tech stack

| Component | Choice | Notes |
|---|---|---|
| VAD | Silero VAD | server-side, per 512-sample chunk |
| Denoise | ClearVoice FRCRN_SE_16K | ≤4 s clips only; under evaluation for removal |
| STT | faster-whisper large-v3 | int8_float16 on CUDA |
| LLM | qwen3.5:27b via Ollama | **locked default** — a second model would OOM the GPU. `LLM_MODEL` env = opt-in override (Fanar-2 A/B; restart Ollama + server to switch) |
| Diacritization | CATT tashkeel | Fusha replies only; `CATT_ENABLED=0` to disable; never on Egyptian |
| TTS (Fusha/Najdi/En) | k2-fsa/OmniVoice | zero-shot voice clone, 24 kHz, fp16 — **protected config** |
| TTS (Egyptian) | mohammedaly22/VoiceTut-TTS | Egyptian-tuned OmniVoice fine-tune, same API, ~3 GB fp16, lazy-loaded on first Egyptian turn |
| Audio encoding | lameenc (PCM → MP3) | complete MP3 per sentence (browser `decodeAudioData`) |
| Python isolation | venv (no Docker — no sudo) | GPU works out of the box |

Env knobs: `LLM_NUM_CTX` (default 8192), `LLM_MODEL` (default qwen3.5:27b — opt-in Fanar-2 override), `CATT_ENABLED` (default 1), `OMNIVOICE_MODEL`, `OMNIVOICE_DEVICE`, `VOICETUT_MODEL`, `VOICETUT_DEVICE`, `VOICETUT_PRELOAD` (default 0 = lazy load), `EGY_REPAIRS` (default 1 — Egyptian lexical repairs).

---

## TTS module contract (`pipeline/tts_omnivoice_v1.py`)

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
├── CLAUDE.md                       ← you are here
├── README.md
├── requirements.txt
├── server.py                       ← entry point. FastAPI app: WS orchestration only
├── pipeline/                       ← the pipeline package (models do the work)
│   ├── stt.py                      ← Silero VAD, FRCRN denoiser, faster-whisper
│   ├── routing.py                  ← language/dialect detection (incl. Egyptian tiers), text-acceptance policy
│   ├── llm.py                      ← Ollama client, model config (qwen + fanar), prompt construction
│   ├── tts_omnivoice_v1.py         ← TTS: Fusha/Najdi/English (OmniVoice + CATT) — PROTECTED
│   └── tts_voicetut_v1.py          ← TTS: Egyptian only (VoiceTut, no CATT, EGY_REPAIRS filter)
├── scripts/
│   ├── start_server.sh             ← starts Ollama (flash-attn, q8_0 KV) + the server
│   ├── test_local.py               ← no-mic pipeline test (LLM → TTS → MP3 files)
│   ├── test_routing.py             ← no-GPU dialect-routing tests + baseline snapshot proof
│   └── fixtures_routing_baseline.json ← pre-Egyptian build_turn/looks_najdi snapshot (60 rows)
├── static/index.html               ← browser client
├── static/review.html              ← /review dashboard (latency + transcripts table)
├── voices/                         ← Saudi reference clips for voice cloning
└── logs/                           ← interactions.jsonl (gitignored — private)
```

## Key decisions

- **LLM locked to qwen3.5:27b** — model selector removed; two LLMs don't fit VRAM alongside the in-process stack.
- **`num_predict: 300` stays** — very long answers may truncate mid-sentence; accepted tradeoff to keep voice replies bounded.
- **CATT gated to Fusha and applied per-sentence on the reply text** — MSA-trained; it mis-vocalizes Najdi words.
- **Najdi vs Fusha routing** is lexicon-based on normalized text (see `_NAJDI_MARKERS`/`looks_najdi` in `pipeline/routing.py` and the MSA→Najdi glossary in the Najdi turn instruction).
- **Egyptian routing is tiered** (`route_arabic`): Najdi-exclusive markers win, then Egyptian-exclusive (مش/ده/دي/عايز/دلوقتي/النهارده/كده/فين/بتاع/curated م…ش forms), then shared markers (اللي/عشان/لسه/يلا — pan-dialectal, NEVER decisive) keep the pre-Egyptian Najdi behavior, else Fusha. `_NAJDI_MARKERS`/`looks_najdi` are byte-identical to pre-Egyptian (they also back the CATT gate). ليه is glossary-only, not a marker; دول and جداً are markers/forbidden NOWHERE (user constraints).
- **EGYPTIAN_CARD is per-turn, Egyptian turns only** — the shared SYSTEM_PROMPT is untouched. The card names only Egyptian's own correct forms (pink-elephant lesson: naming forbidden other-dialect tokens measurably increases leaks — see NAJDI_NO_OTHER_DIALECTS_RULE).
- **Arabic dialect-history isolation** (`_visible_history` in server.py, ALL Arabic pairs incl. Fusha↔Najdi — owner's decision 2026-08-18): each history pair is tagged with its reply route; an Arabic turn's prompt sees only same-dialect + English/mixed pairs. Withheld, never deleted; explicit requests ("قلها بالمصري") see full history; English↔Arabic behavior unchanged.
- **Egyptian TTS is a separate module** (`tts_voicetut_v1.py`, scaffold copied from the OmniVoice module — same precedent as Silma→OmniVoice). Lazy-loaded; on load failure falls back to OmniVoice for the turn. Never modify `tts_omnivoice_v1.py` for Egyptian needs.
- **Fanar-2 A/B**: `LLM_MODEL` env override; `MODEL_CONFIGS["fanar"]` (think:False + `strip_think_tokens` safety net, fanar-only). Local model: `hf.co/mradermacher/Fanar-2-27B-Instruct-i1-GGUF:i1-Q4_K_M`.
- **MP3 format** — browser `decodeAudioData` needs complete containers, not raw PCM.
- **Sentence-level synthesis** — balances first-audio latency vs audio completeness.
- **No Docker** — no sudo; venv only.
- Dashboards: `/review` (latency + transcripts table), `/logs` (raw JSON).
