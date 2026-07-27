# first_voice_test — Project Context

## What this project is

A standalone testing ground for a local Arabic/English conversational voice pipeline, running entirely on the server's RTX 5090 GPU. The TTS module built here (`tts_omnivoice_v1.py`) is a drop-in replacement for the SILMA-based TTS backend (`voice/app/pipeline/tts.py`) inside the main `nwc-copilot` voice assistant project at `/home/taha/devproject`.

**Language scope:** English, Fusha (MSA), Najdi Arabic, Egyptian Arabic (Masri, reintroduced 2026-07-20), and Arabic-English code-switching. Hijazi and Gulf/Khaleeji were removed; a request for either falls through to Fusha. Egyptian was added under a hard invariant: **every byte sent on Najdi/Fusha/English/mixed turns is identical to the pre-Egyptian baseline** (enforced by `eval/golden_prompts.py`); Najdi detection always wins first (`looks_najdi` short-circuits before `looks_egyptian`) for genuinely Najdi-exclusive markers. اللي/عشان/لسه/يلا were **removed from the Najdi marker set on 2026-07-24** for being pan-dialect rather than Najdi-exclusive (they're common in real Egyptian speech too) — they no longer force Najdi routing; the short-circuit precedence itself is unchanged, only the marker set shrank. This fixed most of the Egyptian-misroutes-as-Najdi collision (`eval/dialect_id_cases.jsonl`'s Egyptian recall went 64%→80%) at the cost of one accepted regression: a genuinely-Najdi utterance whose only marker was لسه now goes undetected. Mixed/code-switch turns deliberately ignore Egyptian in v1.

---

## The full voice pipeline

```
Browser (AudioWorklet, 512-sample Float32 @16 kHz)
  → Silero VAD (server-side onset/end, pre-roll, barge-in)
  → FRCRN denoise (short clips only; DENOISE_ENABLED gate)
  → faster-whisper large-v3 (int8_float16, lang detect + remap tables)
  → qwen3.5:27b via Ollama /api/chat (3-turn rolling history, streamed)
  → tts_omnivoice_v1: sentence flushing → CATT tashkeel (Fusha only)
    → OmniVoice zero-shot clone (Saudi ref voice; Egyptian ref voice + language id
      on Egyptian-routed turns only) → one MP3 per sentence
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
9. `language` gates CATT tashkeel (values `"standard arabic"`/`"najdi arabic"`/`"egyptian arabic"`) AND selects the voice: `"egyptian arabic"` turns use the Egyptian clone prompt with `language="egyptian arabic"` passed to `OmniVoice.generate()` (clip pins timbre, language id pins pronunciation); **every other value keeps the exact legacy call** — Saudi clone prompt, no `language` kwarg (pinned by `eval/test_tts_args.py`). Missing Egyptian clip → warn + Saudi fallback, never a startup failure. `generate()` is serialized behind `_gen_lock` (barge-in orphan-synthesis race)

## Project files

```
first_voice_test/
├── CLAUDE.md              ← you are here
├── README.md
├── requirements.txt
├── server.py              ← FastAPI app: WS orchestration only (models below do the work)
├── stt.py                 ← Silero VAD, FRCRN denoiser, faster-whisper
├── routing.py             ← language/dialect detection, text-acceptance policy
├── llm.py                 ← Ollama client, model config, prompt construction
├── tts_omnivoice_v1.py    ← TTS module (OmniVoice + CATT; Saudi + Egyptian clone prompts)
├── static/index.html      ← browser client
├── static/review.html     ← /review dashboard (latency + transcripts table)
├── start_server.sh        ← starts Ollama (flash-attn, q8_0 KV) + the server
├── test_local.py          ← no-mic pipeline test (LLM → TTS → MP3 files)
├── eval/                  ← non-regression harness (see eval/README.md): golden prompt-byte
│                            gate, routing pins, dialect-ID recall, TTS-arg pins, A/B
│                            harness, purity lint, committed baselines (BASELINES.md)
├── voices/                ← reference clips: Saudi (default) + Egyptian v4 (v3 kept on disk, superseded)
└── logs/                  ← interactions.jsonl (gitignored — private)
```

## Key decisions

- **LLM locked to qwen3.5:27b** — model selector removed; two LLMs don't fit VRAM alongside the in-process stack.
- **`num_predict: 300` stays** — very long answers may truncate mid-sentence; accepted tradeoff to keep voice replies bounded.
- **CATT gated to Fusha and applied per-sentence on the reply text** — MSA-trained; it mis-vocalizes Najdi words.
- **Najdi vs Fusha routing** is lexicon-based on normalized text (see `_NAJDI_MARKERS`/`looks_najdi` in `routing.py` and the MSA→Najdi glossary in the Najdi turn instruction).
- **Egyptian (2026-07-20, staged reintroduction under the byte-invariant):** `looks_egyptian`/`requested_egyptian` in `routing.py` (guarded request pattern — "Egyptian Museum" and negated «لا ترد بالمصري» never fire; دول demoted to weak, MSA collision); Najdi-first precedence in `build_turn`; `EGYPTIAN_CARD` appended per-turn on Egyptian-routed turns only — **never mention Egyptian material on Najdi turns** (measured pink-elephant regression, see `NAJDI_NO_OTHER_DIALECTS_RULE`); rolling history is CLEARED when a turn crosses the Egyptian boundary (`llm.crosses_egyptian_boundary` — never fires in Egyptian-free conversations); SYSTEM_PROMPT and `stt.py`'s `_AR_INITIAL_PROMPT` are byte-frozen (any future STT-prompt change needs a per-dialect WER A/B first).
- **Eval gates are mandatory:** run `eval/test_routing.py`, `eval/golden_prompts.py`, `eval/dialect_id_eval.py`, `eval/test_tts_args.py` after ANY routing/prompt/TTS change (seconds, no GPU). Never re-capture `eval/golden_fixtures.jsonl` to make a red gate green — a red gate means a frozen surface moved. Baselines live in `eval/BASELINES.md`.
- **MP3 format** — browser `decodeAudioData` needs complete containers, not raw PCM.
- **Sentence-level synthesis** — balances first-audio latency vs audio completeness.
- **No Docker** — no sudo; venv only.
- Dashboards: `/review` (latency + transcripts table), `/logs` (raw JSON).
