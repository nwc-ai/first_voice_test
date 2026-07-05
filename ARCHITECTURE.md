# first_voice_test — Architecture & Code Reference

A complete, file-by-file, function-by-function reference for the local Arabic/English
voice assistant. Nothing in the codebase is intentionally omitted. Line references use
`file:line` and reflect the current working tree (branch `omnivoice-tts`).

> **State note:** This document describes the system **as it currently runs** on branch `omnivoice-tts`.
> The dialect routing, **Fusha-as-default**, the Egyptian **v3** voice, and the STT language-pick
> refinement are all committed (HEAD = `ba71bbc`); only the docs may carry an uncommitted refresh (see §14).

---

## Table of contents
1. [Overview](#1-overview)
2. [End-to-end architecture & data flow](#2-end-to-end-architecture--data-flow)
3. [Tech stack & versions](#3-tech-stack--versions)
4. [Repository map](#4-repository-map)
5. [Runtime & deployment (`start_server.sh`, venv, VRAM, access)](#5-runtime--deployment)
6. [`server.py` — full walkthrough](#6-serverpy--full-walkthrough)
7. [`tts_omnivoice_v1.py` — full walkthrough](#7-tts_omnivoice_v1py--full-walkthrough)
8. [`static/index.html` — browser client](#8-staticindexhtml--browser-client)
9. [`test_local.py` — offline smoke test](#9-test_localpy--offline-smoke-test)
10. [Data & assets (logs, voices, checkpoints)](#10-data--assets)
11. [Dialect engine deep-dive](#11-dialect-engine-deep-dive)
12. [Concurrency, lifecycle & barge-in](#12-concurrency-lifecycle--barge-in)
13. [Known issues & caveats](#13-known-issues--caveats)
14. [Current working-tree state](#14-current-working-tree-state)

---

## 1. Overview

`first_voice_test` is a **single-user, local, real-time voice assistant** for **Arabic (Saudi
+ Egyptian dialects) and English**. A browser captures microphone audio, streams it over a
WebSocket to a Python server that runs the entire pipeline **in one process on one GPU**
(except the LLM, which runs in a separate Ollama process), and streams synthesized speech back.

- **General-purpose assistant** (not domain-specific), tuned to **reply in the user's dialect**.
- **Dialects:** Najdi, Hijazi, Egyptian, Fusha (MSA). **Fusha (MSA) is the default** when the
  dialect is unclear (see §11).
- **Latency target:** time-to-first-audio ≈ 1–1.5 s (full turn typically 3–5 s).
- **Barge-in:** speaking over the AI cancels its turn immediately.
- **Single active connection:** a new browser tab supersedes the old one (close code 4001).

Machine: `devserver`, user `taha`, **NVIDIA RTX 5090 (32 GB)**, CUDA 13.0. Server port **8765**;
Ollama on **11434**.

---

## 2. End-to-end architecture & data flow

```
┌─────────── Browser (static/index.html) ────────────┐
│  mic → AudioWorklet "mic-sender"                    │
│  → 512-sample Float32 frames @ 16 kHz               │
└───────────────┬─────────────────────────────────────┘
                │  binary frames over WebSocket  ws://host/ws
                ▼
┌──────────────────────── server.py (FastAPI + uvicorn :8765) ─────────────────────┐
│  receive_loop:                                                                    │
│    Silero VAD (per 512-sample chunk) → utterance segmentation (pre-roll + tail)   │
│      → FRCRN denoise (CPU/GPU, ≤4 s clips only)                                   │
│      → faster-whisper large-v3 (int8_float16) transcription + language detect     │
│      → filters (lang, confidence, length, repetition, code-switch)                │
│      → utterance_queue.put((text, lang, stt_ms, denoise_ms))                      │
│                                                                                   │
│  respond_loop:                                                                    │
│    dialect detection (_detect_dialect) + per-turn language instruction            │
│      → messages = [system] + history + [wrapped user turn]                        │
│      → ollama_chat_token_gen → /api/chat (qwen3.5:27b, streaming)                 │
│      → _filter_cjk (strip CJK/Cyrillic tokens)                                    │
│      → stream_tts_to_ws (tts_omnivoice_v1):                                       │
│           sentence-buffer tokens → background synth_worker                        │
│           → OmniVoice.generate (per-dialect ref clip + language= ID, 24 kHz)      │
│           → lameenc MP3 (64 kbps, one per sentence)                               │
└───────────────┬───────────────────────────────────────────────────────────────────┘
                │  JSON events {ping,loading,ready,speech_start,transcript,token,tts_end}
                │  + binary MP3 frames (one complete MP3 per sentence)
                ▼
┌─────────── Browser playback ───────────────────────┐
│  decode chain (ordered) → WebAudio gapless schedule │
│  control events back to server: playback_start/done │
└─────────────────────────────────────────────────────┘
```

**Two-loop concurrency:** one WebSocket session runs `receive_loop` (VAD+STT, producer) and
`respond_loop` (LLM+TTS, consumer) concurrently via `asyncio.gather`, joined by an
`utterance_queue` and a single `cancel_event`. (See §12.)

---

## 3. Tech stack & versions

| Component | Choice | Pin / note |
|---|---|---|
| Web server | FastAPI + uvicorn | `fastapi==0.136.1`, `uvicorn==0.47.0` |
| HTTP client | httpx | `0.28.1` (Ollama calls) |
| Arrays | numpy | `1.26.4` |
| Audio I/O | soundfile | `0.12.1` (also the `torchaudio.load` replacement) |
| MP3 encode | lameenc | `1.8.2` (PCM→MP3 64 kbps) |
| STT | faster-whisper large-v3 | `1.1.1`, `int8_float16` on CUDA |
| VAD | Silero VAD | via `torch.hub` (`snakers4/silero-vad`) |
| Denoiser | ClearVoice FRCRN_SE_16K | `clearvoice==0.1.2` |
| TTS | OmniVoice (`k2-fsa/OmniVoice`) | `omnivoice==0.1.5`, pulls `transformers>=5.3` |
| LLM | qwen3.5:27b via Ollama | Ollama 0.30.2, `/api/chat`, `keep_alive:-1` |
| PyTorch | torch / torchaudio `2.11.0+cu130` | **installed separately** (not in requirements) |
| Python env | **uv-managed `.venv`** (Python 3.12) | has **no `pip`** — use `uv pip` |

`requirements.txt` ([requirements.txt:1-16](requirements.txt)) lists the pip deps; torch is a
comment-only note (installed out of band to match CUDA 13). The trailing note points to the
`multi-engine-snapshot` branch which preserves the older switchable-LLM version.

---

## 4. Repository map

```
first_voice_test/
├── server.py              # 1298 lines — FastAPI WS server, the whole pipeline
├── tts_omnivoice_v1.py    #  259 lines — in-process OmniVoice TTS module (drop-in API)
├── static/index.html      #  450 lines — browser UI (AudioWorklet mic + WebAudio playback)
├── test_local.py          #  149 lines — offline TTS/LLM smoke test (no mic)
├── start_server.sh        #   53 lines — launch script (Ollama + uvicorn, env, CUDA libs)
├── requirements.txt       #   16 lines — pip deps (+ torch note)
├── README.md              #    2 lines — one-line description
├── CLAUDE.md              # project context for AI assistants
├── ARCHITECTURE.md        # this document
├── .gitignore             # ignores __pycache__, .venv, checkpoints/, *.log
├── voices/
│   ├── silma-tts-saudi-24k.wav             # Saudi male default clip (24 kHz mono, 7.64 s)
│   └── omnivoice-tts-egyptian-24k-v3.wav   # active Egyptian clip (24 kHz mono, 8.0 s); v2/v1 unused
├── checkpoints/
│   └── FRCRN_SE_16K/      # ClearVoice denoiser weights (last_best_checkpoint.pt, ~153 MB; gitignored)
├── logs/
│   ├── interactions.jsonl # per-turn perf + transcript/response log (258 lines)
│   └── barge_diag.log     # TEMP diagnostic: barge-in path + WS disconnect codes
├── .venv/                 # uv-created virtualenv (gitignored)
└── __pycache__/
```

Not in git: `.venv/`, `checkpoints/`, `*.log` (so `barge_diag.log` is ignored; `interactions.jsonl`
is tracked). `test_output/` is produced by `test_local.py` (not committed).

---

## 5. Runtime & deployment

### `start_server.sh` ([start_server.sh:1-53](start_server.sh))
1. **Header comment ([:1-12](start_server.sh#L1))** — usage + the SSH-tunnel access note: a
   `1006/1005` WebSocket close almost always means the **tunnel** dropped (laptop sleep, wifi
   change, sshd idle timeout), not an app bug; the browser auto-reconnects. Recommends launching
   the tunnel with keepalives:
   `ssh -L 8765:localhost:8765 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ExitOnForwardFailure=yes taha@devserver`.
2. **`LD_LIBRARY_PATH` ([:17-23](start_server.sh#L17))** — prepends the venv's bundled NVIDIA libs
   (`cu13`, `cublas`, `cudnn`, `cuda_nvrtc`) and Ollama's lib dir so the native `.so`s resolve.
3. **Ollama bootstrap ([:30-45](start_server.sh#L30))** — if `:11434` isn't answering, start
   `ollama serve` with **`OLLAMA_FLASH_ATTENTION=1`** and **`OLLAMA_KV_CACHE_TYPE=q8_0`** (these
   roughly halve KV-cache VRAM, enabling a larger `LLM_NUM_CTX` in the same memory). Waits up to
   30 s for readiness. **Flags only apply to a fresh `ollama serve`** — if Ollama is already up it's
   left as-is.
4. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ([:48](start_server.sh#L48))** — reduces
   CUDA fragmentation on the shared GPU.
5. **`exec .venv/bin/python server.py` ([:50](start_server.sh#L50))** — launches the server.
6. Trailing comment ([:52](start_server.sh#L52)) shows raising context: `LLM_NUM_CTX=16384 bash start_server.sh`.

### Python environment
The `.venv` is **uv-created and has no `pip`** (use `uv pip install --python .venv/bin/python …`).
`torch/torchaudio 2.11.0+cu130` are installed separately to match CUDA 13.

### VRAM (measured, not estimated)
- OmniVoice peak ≈ **2.4 GB** (the "6–7 GB" figure in older docs is wrong).
- qwen3.5:27b loads in ≈ **15.7 GB** via Ollama, leaving ≈ **10 GB free** on the card.
- faster-whisper large-v3 (`int8_float16`) ≈ 3 GB; FRCRN + Silero ≈ 0.5 GB.
- The pipeline calls `torch.cuda.empty_cache()` after each turn ([server.py:1268](server.py#L1268))
  and before the FRCRN free-memory check ([server.py:645](server.py#L645)).

### Access
Browser → server is via an **SSH port-forward tunnel** to `localhost:8765`. Idle `1006/1005`
disconnects come from the tunnel, not the app (see §13).

---

## 6. `server.py` — full walkthrough

The 1298-line core. Sections below follow the file top-to-bottom.

### 6.1 CUDA preload + torchaudio monkeypatch ([:25-58](server.py#L25))
- **`_NVIDIA` loop ([:27-39](server.py#L27))** — `ctypes.CDLL(... , RTLD_GLOBAL)` pre-loads the
  venv's CUDA `.so`s **before** torch imports, so `torchcodec`/cublas symbols resolve globally.
  Failures are swallowed (best-effort).
- **`_sf_load` ([:51-55](server.py#L51))** — replaces `torchaudio.load` with a `soundfile`-based
  loader. torchaudio 2.11 routes `load()` through `torchcodec` (needs CUDA NPP libs absent here);
  soundfile reads WAV/FLAC with no GPU dependency. This patch is **why OmniVoice can load its
  reference WAV** without torchcodec.

### 6.2 Module constants & config ([:65-157](server.py#L65))
- `STATIC_DIR`, `OLLAMA_URL` (`/api/generate`, **warm-up only**), `OLLAMA_CHAT_URL` (`/api/chat`,
  conversation) ([:65-67](server.py#L65)).
- **`MAX_HISTORY_TURNS = 3`** ([:68](server.py#L68)) — rolling memory of last 3 user+assistant pairs.
- **`LLM_NUM_CTX`** ([:76](server.py#L76)) — env-overridable KV-cache size, default **8192**. Used by
  both the warm-up and chat so the model loads once at this size.
- `LOG_DIR`, `PERF_LOG` (`logs/interactions.jsonl`) ([:77-79](server.py#L77)).
- **`MODEL = "qwen3.5:27b"`** ([:107](server.py#L107)) — the ONLY model, hard-locked in both UI and
  server (a second LLM would OOM the GPU).
- **`SYSTEM_PROMPT`** ([:112-144](server.py#L112)) — the full system message. 13 numbered rules
  (0–12): rule 0 language-override; rule 1 English→English; **rule 2 dialect map (Najdi/Hijazi/
  Egyptian/Gulf with example words)**; rule 3 code-switch mirroring; **rule 4 unclear → DEFAULT
  Fusha/MSA (not a regional dialect)**; rule 5 never mix two dialects; rule 6 no CJK/Cyrillic scripts; rule 7
  full spoken sentences; rule 8 punctuation; rule 9 no markdown; rule 10 no filler openers; rule 11
  never ask for clarification; rule 12 spell out abbreviations (voice).
- **VAD tuning ([:147-157](server.py#L147)):** `MIN_SPEECH_CHUNKS=4` (~128 ms onset),
  `MIN_SPEECH_CHUNKS_BARGE=3` (~96 ms onset while AI audible — **assumes headphones**),
  `MAX_SILENCE_CHUNKS=25` (~0.8 s tail to end an utterance), `PREROLL_CHUNKS=10` (~320 ms kept
  before onset to guard the first syllable), `SAMPLE_RATE=16000`.
- **Model singletons ([:160-162](server.py#L160)):** `_vad_model`, `_whisper_model`, `_denoiser`
  (None if FRCRN failed to load).

### 6.3 Logging & diagnostics
- **`_write_log(entry)` ([:82-87](server.py#L82))** — append one JSON line to `interactions.jsonl`.
- **`_diag(msg)` ([:94-99](server.py#L94))** — append a timestamped line to `logs/barge_diag.log` (path
  constant **`_BARGE_DIAG`** [:93](server.py#L93)).
  Explicitly a **TEMP DIAGNOSTIC** ([:90-93](server.py#L90)) to trace what stops AI playback;
  remove once the disconnect cause is fixed.
- **`_active_ws_task` / `_active_ws_ref` ([:104-105](server.py#L104))** — globals for single-connection
  enforcement.

### 6.4 Model loading & lifecycle
- **`_load_all_blocking()` ([:167-195](server.py#L167))** — loads, in order: OmniVoice TTS
  (`tts_omnivoice_v1.load_models()`), Silero VAD (`torch.hub.load`), faster-whisper large-v3
  (`WhisperModel("large-v3", device="cuda", compute_type="int8_float16")`), ClearVoice FRCRN
  (`ClearVoice(task="speech_enhancement", model_names=["FRCRN_SE_16K"])`). FRCRN failure is
  non-fatal (denoising is then skipped).
- **`_models_ready` ([:198](server.py#L198))** — `asyncio.Event` gating the WS "ready" event.
- **`_warm_llm(model)` ([:201-223](server.py#L201))** — fires one throwaway `/api/generate`
  (`num_predict:1`, `num_ctx:LLM_NUM_CTX`, `keep_alive:-1`) so the 27B cold-load happens behind the
  loading screen, not on the first user turn. `num_ctx` MUST match the chat path to avoid a reload.
- **`lifespan(app)` ([:226-235](server.py#L226))** — background-loads everything then `_warm_llm`,
  sets `_models_ready`; `yield`s immediately so the page can load while models warm.

### 6.5 FastAPI app & HTTP endpoints
- **`app` + static mount ([:238-239](server.py#L238))** — `/static` serves the `static/` dir.
- **`GET /` → `index()` ([:242-250](server.py#L242))** — serves `index.html` with
  `Cache-Control: no-store` (forces re-fetch so edits/restarts aren't served stale).
- **`GET /logs` → `get_logs()` ([:253-266](server.py#L253))** — returns the last 200
  `interactions.jsonl` entries + total count, as JSON.
- **`GET /review` → `review_page()` ([:269-420](server.py#L269))** — a self-contained dark-theme
  HTML dashboard (inline CSS+JS) over `/logs`: sortable/filterable table of Time, Model, Lang,
  Transcript, Response, and latencies (STT, TTFT, TTS-1st, LLM-total, E2E) with color buckets
  (`msClass`: <800 ms fast, <2000 ms med, else slow). JS fns: `msClass`, `fmt`, `shortModel`,
  `sortBy`, `getVal`, `render`, `loadData`.
- **`WS /ws`** — the main endpoint (§6.9).

### 6.6 VAD + utterance segmentation
- **`_reset_vad_states()` ([:425-431](server.py#L425))** — clears Silero's LSTM state per
  connection/utterance.
- **`make_stt_processor(on_speech_start, is_ai_audible)` ([:434-498](server.py#L434))** — returns an
  async `process_chunk(bytes)` closure with per-connection state (`preroll` deque, `speech_buffer`,
  `in_speech`, `silence_chunks`, `speech_chunks_count`). Per 512-sample frame:
  - Runs Silero VAD → `speech_prob`; `is_speech = prob >= 0.5`.
  - **Onset:** needs `MIN_SPEECH_CHUNKS_BARGE` consecutive speech chunks if `is_ai_audible()` else
    `MIN_SPEECH_CHUNKS`; on confirm, prepends the pre-roll and calls `on_speech_start()`.
  - **End:** after onset, `MAX_SILENCE_CHUNKS` of silence concatenates `speech_buffer` → returns the
    full utterance `np.ndarray` and resets state.
  - **Idle:** recycles dropped chunks into the pre-roll deque (first-syllable guard).

### 6.7 STT pipeline
- **Thresholds ([:501-504](server.py#L501)):** `LANG_PROB_THRESHOLD=0.25`,
  `LANG_PROB_THRESHOLD_AR=0.10` (Arabic misfires as Urdu/Farsi/Punjabi), `WORD_CONF_THRESHOLD=0.3`,
  `ALLOWED_LANGS={"ar","en"}`.
- **`_denoise_blocking(audio)` ([:636-661](server.py#L636))** — FRCRN denoise, **gated**: returns
  audio unchanged if denoiser is None, clip > `_FRCRN_MAX_SAMPLES` (4 s), or < `_FRCRN_MIN_FREE_MB`
  (150 MB) free after `empty_cache()`. OOM is caught and the audio passed through.
- **`_TRANSCRIBE_KWARGS` ([:664-678](server.py#L664))** — `beam_size=5`,
  `condition_on_previous_text=False`, `vad_filter=True`, `vad_parameters={min_silence_duration_ms:300}`,
  `word_timestamps=True`, and decoder anti-hallucination **`no_repeat_ngram_size=3`**,
  **`repetition_penalty=1.1`**.
- **`_AR_HOTWORDS` ([:686](server.py#L686))** — dialect marker vocabulary passed as Whisper
  `hotwords` **only on the forced-Arabic re-pass** (biases dialect spelling once Arabic is committed).
- **`_transcribe_blocking(audio) → (text, lang)` ([:701-738](server.py#L701)):**
  1. Pass-1 auto-detect (`_whisper_model.transcribe`); also capture the full LID distribution
     `info.all_language_probs`.
  2. If detected lang ∈ `_ARABIC_SCRIPT_REMAP` (`fa/ps/ug/prs/ckb/sd/pa` — **`ur` was removed**), force a
     **dialect-biased re-pass** with `language="ar"` + `hotwords=_AR_HOTWORDS`.
  3. **Else if the guess is neither ar nor en** (`ur`, `nn`, `hi`, …): pick between our two real languages
     from the distribution — whichever of **P(en)/P(ar)** is higher → re-decode forced to it. This stops
     accented English (mislabeled as Urdu) being force-transcribed into phonetic Arabic; genuinely foreign
     speech leaves both near zero → dropped by the gate below.
  4. Gate on `lang_prob` (AR threshold lower) → drop if below.
  5. Compute mean per-word confidence → drop if `< WORD_CONF_THRESHOLD`.
  6. Join segment text; if lang still isn't ar/en but text is pure Latin → remap to `en`.

### 6.8 Dialect engine (constants/functions)
Covered in depth in §11. Symbols: `_ARABIC_CHARS_RE`/`_LATIN_WORDS_RE` ([:507-508](server.py#L507)) which
feed `_is_mixed` ([:510](server.py#L510)), `_WANTS_ARABIC_RE`/
`_WANTS_ENGLISH_RE` ([:515-526](server.py#L515)), `_DIALECT_PATTERNS` ([:534-546](server.py#L534)),
`_requested_dialect` ([:548-553](server.py#L548)), `_NAJDI_MARKERS`/`_HIJAZI_MARKERS`/
`_EGYPTIAN_MARKERS` ([:560-571](server.py#L560)), `_AR_WORD_SPLIT_RE` ([:572](server.py#L572)),
`_detect_dialect` ([:574-587](server.py#L574)), `_ARABIC_SCRIPT_REMAP` ([:595](server.py#L595)),
`MIN_TEXT_CHARS=3`/`MAX_TEXT_CHARS=500` ([:596-597](server.py#L596)).

### 6.9 LLM token streaming + filters
- **`_UNWANTED_SCRIPT_RE` ([:587-599](server.py#L587))** — strips CJK (ideographs, ext-A, compat,
  symbols), katakana, hiragana, hangul, fullwidth forms (incl. `？！`), Cyrillic.
- **`_filter_cjk(token_gen)` ([:601-612](server.py#L601))** — async generator wrapper that removes
  unwanted-script chars per token and **propagates `aclose()`** down so a barge-in tears down the
  Ollama HTTP stream.
- **`_REPETITION_RE` ([:615](server.py#L615))** — detects ASR stuck-loops (`ا ا ا ا`, `هل هل هل هل`).
- **`_INJECTION_RE` ([:618-627](server.py#L618))** — prompt-injection patterns (AR/EN/UR).
- **`_STOP_SEQUENCES` ([:736](server.py#L736))** — stop strings (`User:`, `\nالمستخدم:`, `Human:`, …).
- **`MODEL_CONFIGS` ([:741-774](server.py#L741))** — only `qwen3.5` (`think:False`, temp 0.5, top_p 0.8,
  top_k 20, presence_penalty 1.5, num_predict 300, num_ctx LLM_NUM_CTX) and `default` remain; the old
  per-model configs were removed (they live on `multi-engine-snapshot`).
- **`_get_model_config(name)` ([:777-785](server.py#L777))** — substring match, falls back to `default`.
- **`ollama_chat_token_gen(messages, model, on_first_token)` ([:790-821](server.py#L790))** — streams
  `/api/chat` (`stream:True`, `keep_alive:-1`, merged options/extra), yields `message.content` tokens,
  fires `on_first_token` once.
- **`_single_token(text)` ([:824-825](server.py#L824))** — one-shot async generator (used for the
  empty-response fallback).

### 6.10 WebSocket endpoint ([:830-1285](server.py#L830))
- **`_LockedWS` ([:830-856](server.py#L830))** — wraps the socket; serializes `send_json`/`send_bytes`
  through one `asyncio.Lock` (Starlette forbids concurrent sends). `__getattr__` passes through
  everything else.
- **`websocket_endpoint(ws, model)` ([:859-1285](server.py#L859)):**
  - **Single-connection ([:861-881](server.py#L861))** — accepts, records itself as
    `_active_ws_task/_ref`, and if a prior session exists, closes it with **code 4001** ("superseded")
    then cancels it. Then wraps `ws` in `_LockedWS`. `active_model = MODEL` (browser param ignored).
  - **`keepalive_loop()` ([:890-903](server.py#L890))** — `{"event":"ping"}` every 3 s (starts before
    the model wait so the browser watchdog doesn't reconnect during cold start).
  - **Ready handshake ([:905-913](server.py#L905))** — sends `loading` then waits `_models_ready`,
    then `ready`.
  - **Per-connection state ([:915-924](server.py#L915))** — `cancel_event`, `utterance_queue`,
    `ai_active`, `ai_speaking`, `client_playing`, `history`.
  - **`on_speech_start()` ([:926-940](server.py#L926))** — sets `cancel_event` **only if `ai_speaking`**
    (true barge-in), always sends `speech_start`; diag-logs when AI is audible.
  - **`receive_loop()` ([:948-1047](server.py#L948))** — reads frames: handles
    `websocket.disconnect` (logs close code), control text events (`playback_start`/`playback_done`/
    `barge_in` — the **`barge_in` branch is a kept-but-dead hook**, see §12), and binary mic chunks via
    `process_chunk`. On a completed utterance: cancels if `ai_speaking`, drains stale queue items if
    `ai_active`, denoises + transcribes (off-thread), applies filters (`_is_mixed`→`lang="mixed"`,
    allowed-langs, length, repetition), and enqueues `(text, lang, stt_ms, denoise_ms)`. STT OOM is
    caught and the utterance skipped.
  - **`respond_loop()` ([:1050-1272](server.py#L1050))** — pulls an utterance, clears `cancel_event`,
    sets `ai_active`; blocks injection attempts (`_INJECTION_RE`) with a transcript+`tts_end`; sends
    `transcript`; builds the **per-turn `lang_instruction`** (the dialect router — §11); wraps it in
    `turn_content` ([:1187-1200](server.py#L1187)) with style + anti-hallucination + dialect-aware
    "say you're unsure" phrasing; assembles `messages = [system] + history + [turn]`; streams
    LLM→TTS via `_collecting_token_gen` (collects tokens, wraps `_filter_cjk`+`ollama_chat_token_gen`,
    propagates `aclose`) into `stream_tts_to_ws`. Then: empty-response fallback
    ([:1200-1214](server.py#L1200)); commits the **clean** user text + reply to `history` (trimmed to
    `MAX_HISTORY_TURNS*2`) unless cancelled; computes latencies (`llm_ttft_ms`, `tts_first_ms`,
    `llm_total_ms`, and the **reconstructed `e2e_ms` = MAX_SILENCE_CHUNKS*32 + denoise + stt + turn**)
    and `_write_log`s them; `finally` resets flags + `empty_cache()`.
  - **`asyncio.gather(receive_loop(), respond_loop())` ([:1274-1285](server.py#L1274))** — runs both;
    `finally` cancels keepalive, sets `cancel_event`, clears the globals if still owner.
- **`__main__` ([:1288-1298](server.py#L1288))** — `uvicorn.run(app, host=0.0.0.0, port=8765,
  ws_ping_interval=30.0, ws_ping_timeout=120.0)` (long protocol-level leash; the app does its own 3 s ping).

---

## 7. `tts_omnivoice_v1.py` — full walkthrough

In-process OmniVoice TTS (zero-shot voice clone, 24 kHz). Public API `stream_tts_to_ws` is a drop-in
for the earlier Silma module.

- **Sentence-break constants ([:28-32](tts_omnivoice_v1.py#L28)):** `HARD_BREAK={! ? ؟}`,
  `SOFT_BREAK={. , ، ; :}`, `SOFT_BREAK_MIN=40`, `FIRST_SOFT_MIN=20` (earlier first flush → faster
  first audio), `_HEAD_PROBE_CHARS=30`.
- **`_ABBREV_RULES` + `_expand_abbreviations(text)` ([:36-52](tts_omnivoice_v1.py#L36))** — spell-out
  rules for spoken Arabic: `1هـ→1 هجري`, `1م→1 ميلادي`, `ق.م→قبل الميلاد`, `1%→1 بالمئة`, `د.→دكتور`,
  `أ.→أستاذ`, `إلخ→وما إلى ذلك`, and digit/letter separation. Runs per sentence before synthesis.
- **`SAMPLE_RATE=24000`; voice registry** — `_REF_AUDIO`/`_REF_TEXT` (Saudi **default** clip + transcript)
  and `_EGY_REF_AUDIO`/`_EGY_REF_TEXT` (the **active Egyptian v3** clip + transcript) feed **`_VOICES`**
  (`{"saudi": …, "egyptian": …}`) with `DEFAULT_VOICE="saudi"`. **`_resolve_voice(key)`** maps a voice key
  → `(ref_audio, ref_text)`, falling back to Saudi on an unknown key or a missing file. OmniVoice CLONES the
  chosen clip, so the clip IS the spoken voice. `server.py` picks the key per turn from the routed dialect.
- **`_MODEL_ID`/`_DEVICE`** — env-overridable (`k2-fsa/OmniVoice`, `cuda:0`).
- **`load_models()` / `_get_model()`** — lazy singleton behind a `threading.Lock`; `load_models` asserts
  **every `_VOICES` clip exists** (fails loudly on a typo). **Note:** the lock guards *load*, not
  `generate()` (latent multi-user concern; not an issue single-user).
- **`_should_flush(buffer, char, first)` ([:90-96](tts_omnivoice_v1.py#L90))** — flush on hard break, or
  soft break once buffer ≥ `FIRST_SOFT_MIN`/`SOFT_BREAK_MIN`.
- **`_OPENER_RE` / `_strip_openers(text)` ([:100-115](tts_omnivoice_v1.py#L100))** — removes leading
  filler openers (sure/of course/طبعاً/أكيد…) from the first chunk.
- **`_synthesize_mp3_blocking(text, ref_audio, ref_text, language=None) → bytes`** —
  `OmniVoice.generate(text, ref_audio, ref_text, language=language)` → int16 PCM → **lameenc 64 kbps
  mono, quality 7** → one complete MP3 container. `ref_audio/ref_text` select the cloned voice;
  **`language`** is a dialect ID (`egyptian arabic`/`najdi arabic`/…) that pins pronunciation (`None` =
  language-agnostic). Defaults keep it backward-compatible (Saudi clip).
- **`_synthesize_mp3(text, ref_audio, ref_text, language=None)`** — `asyncio.to_thread` wrapper.
- **`stream_tts_to_ws(token_gen, ws, cancel_event, on_first_audio=None, voice=None, language=None)`:**
  resolves the voice once at entry (`_resolve_voice(voice)`) and passes the chosen `ref_audio/ref_text`
  + `language` to every sentence's synth.
  - `synth_worker()` — background task: pulls sentences,
    **3-point cancellation** ((b) before synth, (c) after synth), expands abbreviations, synthesizes,
    fires `on_first_audio` once, `ws.send_bytes(mp3)`.
  - `_emit(text_chunk)` ([:194-208](tts_omnivoice_v1.py#L194)) — sends `{"event":"token"}` to the
    browser **and** feeds the sentence buffer so display text and spoken text are identical; flushes
    sentences via `_should_flush`.
  - Token loop ([:213-235](tts_omnivoice_v1.py#L213)) — **(a)** cancellation check at top; buffers a
    `head` to strip an opener across the leading edge; flushes remaining buffer at the end.
  - `finally` ([:236-256](tts_omnivoice_v1.py#L236)) — `aclose()` the token gen (stops Ollama), sentinel
    the queue, cancel the worker if barge-in, await worker. Sends `{"event":"tts_end"}` only if not
    cancelled.

---

## 8. `static/index.html` — browser client

A single self-contained page (no external assets). RTL Arabic UI.

### 8.1 Markup & CSS ([:1-97](static/index.html#L1))
- A locked **LLM selector** (only `qwen3.5:27b`), a `#status` pill, a transcript box (`أنت:`), a
  response box (`المساعد:`), and the circular **🎙️/⏹️ connect button**.
- CSS state classes: `#status.listening/.speaking/.processing`; `#connect-btn.connected` (breathing
  animation), `.ai-speaking`, `.user-speaking`.

### 8.2 Constants & state ([:100-153](static/index.html#L100))
`CHUNK_SIZE=512`, `SAMPLE_RATE=16000`; `ws`, `audioCtx` (16 kHz capture), **`playCtx` (native-rate
playback — decoding 24 kHz audio in a 16 kHz context would muffle it)**, `workletNode`, `micStream`,
`connected`, `serverReady`, `responseBuffer`; playback state (`activeSources`, `nextStartTime`,
`playGen`, `decodeChain`, `isPlaying`); watchdog/reconnect state (`lastMsgTime`, `watchdogTimer`,
`reconnectDelay`, `wantReconnect`, `reconnectTimer`). A long comment ([:124-136](static/index.html#L124))
documents that **client-side RMS barge-in was REMOVED**; barge-in is now server-VAD-driven.

### 8.3 Functions
- **`setStatus(text, cls)` ([:155-158](static/index.html#L155))**.
- **`loadModels()` ([:163-169](static/index.html#L163))** — hardcodes the single `qwen3.5:27b` option;
  hides the reload button (model is fixed).
- **`WORKLET_CODE` ([:172-193](static/index.html#L172))** — inline `AudioWorkletProcessor` "mic-sender":
  buffers mic samples into 512-frame `Float32Array`s and `postMessage`s each to the main thread.
- **`connect()` ([:202-297](static/index.html#L202))** — getUserMedia (echo-cancel + noise-suppress +
  AGC), creates the two AudioContexts, loads the worklet, opens `ws://host/ws?model=…`. **Socket-scoped
  handlers** (`sock` captured locally so a stale socket's late events can't kill a new session):
  `onopen` (starts mic forwarding gated on `serverReady` + the 20 s watchdog), `onclose` (handles **4001 =
  superseded → don't reconnect**, else backoff reconnect), `onerror`, `onmessage → handleMessage`.
- **`scheduleReconnect()` ([:299-303](static/index.html#L299))** — exponential backoff (0.5→1→2…→10 s).
- **`disconnect(stopServer)` ([:305-331](static/index.html#L305))** — tears down timers, worklet, mic,
  both audio contexts, and detaches/closes the socket so late events can't fire into a future session.
- **`handleMessage(event)` ([:334-386](static/index.html#L334))** — feeds the watchdog; binary →
  `queueMp3`; JSON events: **`ping`** (ignored), **`loading`**, **`ready`** (mic starts, status
  listening), **`speech_start`** (`clearAudioQueue` → barge-in stop; user-speaking style), **`transcript`**
  (clear audio + boxes, processing), **`token`** (append to response box), **`tts_end`** (reset to
  listening if nothing is playing).
- **Playback ([:388-447](static/index.html#L388)):** `SEAM_OVERLAP=0.05` (swallow LAME padding);
  `sendControl(event, extra)` (reports `playback_start`/`playback_done` to the server); **`queueMp3`** —
  serialized decode chain (`decodeChain`) + sample-accurate gapless scheduling on `playCtx`, `playGen`
  invalidates in-flight decodes; **`clearAudioQueue`** — bumps `playGen`, stops all sources, resets.

---

## 9. `test_local.py` — offline smoke test

Standalone test ([test_local.py:1-149](test_local.py)) that exercises **LLM→TTS only** (no mic, no VAD,
no STT, no server). `MockWebSocket` ([:39-59](test_local.py#L39)) prints tokens and saves each MP3 to
`./test_output/sentence_NNN.mp3`. `ollama_token_gen` ([:64-86](test_local.py#L64)) streams `/api/generate`
(note: **not** `/api/chat`, no system prompt / dialect routing / history) with `think:False`. `run_test`
([:91-145](test_local.py#L91)) warms OmniVoice, runs `stream_tts_to_ws`, prints time-to-first-audio,
total time, MP3 count/bytes, and an `scp` hint. Despite its "Full pipeline" docstring it is a **TTS-module
smoke test**, not a full end-to-end test.

---

## 10. Data & assets

- **`logs/interactions.jsonl`** — one JSON object per turn. Keys: `ts`, `model`, `lang`, `transcript`,
  `response`, and **`latency`** = `{denoise_ms, stt_ms, llm_ttft_ms, llm_total_ms, tts_first_ms, e2e_ms}`
  (written at [server.py:1238](server.py#L1238)). Tracked in git (258 lines). Consumed by `/logs` and
  `/review`.
- **`logs/barge_diag.log`** — timestamped lines from `_diag`: `[SERVER-VAD] speech_start …`,
  `[SERVER-UTTERANCE] …`, `[CLIENT-BARGE] …`, `[WS-DISCONNECT] code=…`. Gitignored (`*.log`). Temporary.
- **`voices/`** — OmniVoice reference clips (24 kHz mono PCM-16): **`silma-tts-saudi-24k.wav`** (Saudi male,
  7.64 s — the **default** voice for Najdi/Hijazi/Fusha/English, transcript `_REF_TEXT`);
  **`omnivoice-tts-egyptian-24k-v3.wav`** (Egyptian, 8.0 s — the **active** Egyptian voice, transcript
  `_EGY_REF_TEXT`); **`omnivoice-tts-egyptian-24k-v2.wav`** and **`omnivoice-tts-egyptian-24k.wav`**
  (Egyptian v2/v1, **unused**, superseded by v3).
- **`checkpoints/FRCRN_SE_16K/`** — ClearVoice denoiser weights (`last_best_checkpoint.pt`, ~153 MB).
  Gitignored.

---

## 11. Dialect engine deep-dive

The system understands and replies in **Najdi, Hijazi, Egyptian, Fusha**; **Fusha (MSA) is the default**.

**Explicit requests — `_DIALECT_PATTERNS` + `_requested_dialect` ([server.py:534-553](server.py#L534)):**
regex-matches a *named* dialect or language request (e.g. "in Najdi", "بالمصري", "in Fusha"). The Arabic
tokens require a **request prefix** (`بال…` / `لهجة`/`لغة …`) so a bare adjective or proper noun
(«المتحف المصري»، «الثورة المصرية») is NOT mistaken for a request; the English names (`\bgulf\b`,
`\begyptian\b`, …) are left permissive by design (a known, accepted gap). First match wins; returns a
descriptive phrase or `None`.

**Spoken-dialect detection — `_detect_dialect(text)` ([server.py:574-587](server.py#L574)):** splits the
transcript into Arabic-letter words (`_AR_WORD_SPLIT_RE`) and counts **distinguishing marker** hits:
- `_NAJDI_MARKERS` = وش، أبغى/ابغى، الحين، زين، ماله، يبيلك، صج، عاد، هيه، أدري/ادري
- `_HIJAZI_MARKERS` = إيش/ايش، أبي/ابي، دحين، هلا، تمام، إيوه/أيوه/ايوه، مشكور، كيفك
- `_EGYPTIAN_MARKERS` = إزاي/ازاي، إزيك/ازيك، عايز/عاوز/عايزة، دلوقتي/دلوقت، مش، كده/كدا، علشان، ده، دي، دول، النهاردة، إمبارح/امبارح، أهو، **إيه/ايه، كام، فين** (the Egyptian interrogatives, added to catch markerless-looking Egyptian; lifted Egyptian discrimination 58%→92% on the 100-Q test)

Returns the strict-max dialect, or **`None` on no-marker or tie** (the common case for short
utterances). Shared words are deliberately excluded — notably bare **`عشان`** (used across
Najdi/Hijazi/Gulf/Egyptian; it was tying real Najdi utterances to Egyptian, so only `علشان` is kept), plus
وين، ليش، بعدين، خلاص، يلا، بس، مرة.

**Routing — the per-turn `lang_instruction` in `respond_loop` ([server.py:1111-1182](server.py#L1111)),
priority order:**
1. **Explicit Arabic request** (`_requested_dialect` or `_WANTS_ARABIC_RE`) → reply in the named dialect,
   or **Fusha (MSA)** if no dialect named ([:1111-1124](server.py#L1111)).
2. **Explicit English request** (`_WANTS_ENGLISH_RE`) → English ([:1125-1132](server.py#L1125)).
3. **`lang=="mixed"`** → mirror the AR/EN mix; Arabic parts in the detected dialect, else **Fusha**
   ([:1133-1144](server.py#L1133)).
4. **`lang=="ar"`** → committed instruction for detected **Najdi / Hijazi / Egyptian**, else **Fusha
   default** ([:1145-1181](server.py#L1145)).
5. **else (English)** → English ([:1182](server.py#L1182)).

So: clearly-detected Najdi/Hijazi/Egyptian win; English stays English; **everything else Arabic
defaults to Fusha/MSA** (SYSTEM_PROMPT rule 4 reinforces this). Fusha is routed through the Saudi voice
clip with `language="standard arabic"` — there is no separate spoken-Fusha classifier.
STT recognition is biased toward dialect spelling via `_AR_HOTWORDS` on the forced-Arabic re-pass.

**Per-dialect voice + pronunciation (in `respond_loop`, alongside `lang_instruction`):** each turn also
computes `tts_voice` and `tts_language`, both passed to `stream_tts_to_ws`:

| Routed dialect | `tts_voice` (clip) | `tts_language` (OmniVoice `language=`) |
|---|---|---|
| Egyptian (detected/requested) | `egyptian` (Egyptian **v3** clip) | `egyptian arabic` |
| Najdi | `saudi` | `najdi arabic` |
| Hijazi | `saudi` | `hijazi arabic` |
| Gulf (explicit) | `saudi` | `gulf arabic` |
| **Fusha (default: unclear / no-marker / requested-no-dialect)** | `saudi` | `standard arabic` |
| English | `saudi` | `None` |
| Mixed AR+EN | per detected dialect (else Saudi/Fusha) | `None` (avoids mispronouncing the English) |

The `language=` ID pins OmniVoice's pronunciation to one dialect — this **fixed the Saudi/Egyptian
word-mixing** where, with only a reference clip, some words came out in the wrong dialect.

> **Tashkeel/diacritization is intentionally NOT used** — CATT (the diacritizer) is MSA-trained and
> mis-vocalizes Egyptian/dialect words, which would undo the dialect pronunciation. See §13.

---

## 12. Concurrency, lifecycle & barge-in

- **Two loops, one queue, one cancel flag.** `receive_loop` (producer) and `respond_loop` (consumer) run
  under `asyncio.gather`; `utterance_queue` carries utterances; `cancel_event` signals barge-in/teardown.
  `None` on the queue is the shutdown sentinel.
- **Blocking work off the event loop.** VAD runs inline (fast), but `_denoise_blocking`,
  `_transcribe_blocking`, and OmniVoice synth run via `asyncio.to_thread`.
- **`_LockedWS`** serializes all sends (Starlette forbids concurrent sends).
- **Single connection.** A new tab supersedes the old via close **code 4001**; the old browser sees 4001
  and does **not** reconnect (kills the reconnect ping-pong).
- **Barge-in (the real path):** server Silero VAD detects speech onset while `ai_speaking` →
  `on_speech_start` sets `cancel_event` **and** sends `speech_start`; the browser calls
  `clearAudioQueue()` to stop playback. `cancel_event` propagates through `_filter_cjk.aclose()` →
  `ollama_chat_token_gen.aclose()` (stops the 27B) and cancels the TTS `synth_worker`.
- **`barge_in` WS message handler ([server.py:973-990](server.py#L973)) is a kept-but-dead hook** — the
  current browser never sends it (client-side RMS detection was removed); retained for a future
  client-side detector. Real barge-in is entirely server-VAD-driven.
- **`ai_active` vs `ai_speaking`:** barge-in cancels only once audio is actually playing
  (`ai_speaking`), so talking while the AI is still *thinking* doesn't kill the turn.

---

## 13. Known issues & caveats

- **Idle WebSocket disconnects (`1006/1005/1012`)** — almost always the **SSH tunnel** dropping
  (laptop sleep, wifi change, sshd timeout), not an app bug; the browser auto-reconnects. Mitigation =
  SSH keepalives (documented in `start_server.sh`). `barge_diag.log` exists to characterize this.
- **Single-user only.** All GPU components are shared singletons and only one WebSocket session is
  allowed; true multi-user would need a GPU-serialization lane + a 2nd GPU for Ollama. (Designed but
  deferred.)
- **Barge-in assumes headphones.** `MIN_SPEECH_CHUNKS_BARGE=3` is aggressive; on open speakers the AI's
  own voice can bleed into the mic.
- **Hallucination control is prompt + sampling only** (no RAG/tools). The model can still refuse or
  guess on real-time/unknowable facts.
- **`e2e_ms` is reconstructed**, not measured end-to-end (excludes network + actual playback time).
- **Tashkeel (diacritization) is intentionally NOT used.** CATT was evaluated twice; it is MSA-trained and
  mangles Egyptian/dialect words (e.g. `علطول`→`عُلْطُولُ`), which would re-MSA-ify dialect pronunciation.
  Production has zero tashkeel code; pronunciation is handled by the `language=` param + the reference voice.
  (The `catt_tashkeel` package may still sit unused in the venv.)
- **Per-dialect accent IS implemented** — per-dialect reference clips (Egyptian v3 + Saudi) chosen by the
  voice registry, plus the OmniVoice `language=` ID per dialect (§11). Caveat: it depends on a good reference
  clip + the `language=` param; OmniVoice's zero-shot clone can still imperfectly capture an accent, so
  swapping a cleaner clip (one-line registry change) is the lever if a voice sounds off. (An earlier
  *synthetic* NAMAA clip was tried and discarded in favor of the user-provided real Egyptian clip.)
- **`think:False` is required** for qwen3.5 here — with thinking on, Ollama 0.30.2 spends the
  `num_predict` budget reasoning and emits no spoken text.

---

## 14. Current working-tree state

Branch **`omnivoice-tts`**. **HEAD = `ba71bbc`** ("Refine language detection and transcription handling for
Arabic-script languages"). All **code** (`server.py`, `tts_omnivoice_v1.py`, `static/index.html`,
`start_server.sh`) is **committed**; only the docs (`CLAUDE.md`, `README.md`, `SETUP.md`, this file) may
carry an uncommitted refresh.

**Dialect / voice / STT work now in git (commits `2562aca` → `f3b1758` → `ba71bbc`):**
- Dialect engine: `_detect_dialect` (3-way Najdi/Hijazi/Egyptian markers) + `_requested_dialect` (explicit
  requests, Arabic tokens gated behind a `بال…`/`لهجة` request prefix so proper nouns don't false-trigger)
  + committed per-turn routing; dialect-aware abstention phrasing.
- **Fusha (MSA) is the DEFAULT** (unclear / no-marker / no-named-dialect / mixed-Arabic-part → Fusha),
  reversing the earlier Egyptian-default; SYSTEM_PROMPT rule 4 reinforces it.
- Egyptian markers refined: bare `عشان` removed (shared word); interrogatives `إيه/ايه/كام/فين` added
  (Egyptian discrimination 58%→92% on a 100-question test).
- **Per-dialect voice registry** (`_VOICES`/`_resolve_voice`) + `voice=` param; active Egyptian clip is
  **`omnivoice-tts-egyptian-24k-v3.wav`** (v2/v1 unused). Per-dialect `language=` threaded through
  `stream_tts_to_ws` → `OmniVoice.generate()` (fixes Saudi/Egyptian word-mixing).
- **STT language pick**: `ur` removed from `_ARABIC_SCRIPT_REMAP`; non-en/ar guesses (ur/nn/hi/…) resolved
  from the `info.all_language_probs` distribution (higher of P(en)/P(ar) wins) instead of blindly forcing
  Arabic — stops accented English being transcribed as phonetic Arabic.
- Tashkeel evaluated and **dropped** (CATT mangles dialect words).

The older switchable-LLM version lives on the `multi-engine-snapshot` branch.

> **Line-ref note:** `file:line` refs in §6.7–§6.8 and §11 (dialect/STT) were re-synced to the current
> tree. A few refs in §6.9–§6.10/§12 may be off by ~30 lines after the STT/marker insertions — symbol
> names are authoritative; grep the symbol if a line number looks off.
