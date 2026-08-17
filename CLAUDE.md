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
    → Egyptian-routed turns: VoiceTut-TTS (default) → OmniVoice+Egyptian-clip →
      OmniVoice+Saudi-clip, each a fallback for the one before
      Najdi/Fusha/English/mixed: OmniVoice zero-shot clone (Saudi ref voice)
      → one MP3 per sentence
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
| LLM | qwen3.5:27b via Ollama | **locked default** — a second model resident at once would OOM the GPU. `LLM_MODEL_OVERRIDE` env var (unset = today's exact behavior) swaps the live pipeline to a candidate model for deliberate, one-off local A/B tests — never `export` it, always pass inline on the invocation line. Live-test candidates so far: Fanar-2-27B-Instruct (`MODEL_CONFIGS["fanar-2"]`), prior generation Fanar-1-9B-Instruct (removed 2026-08-07 — instruction-following too weak, per owner's own live test). See `eval/BASELINES.md`. |
| Diacritization | CATT tashkeel | Fusha replies only; `CATT_ENABLED=0` to disable |
| TTS (Najdi/Fusha/English/mixed) | k2-fsa/OmniVoice | zero-shot voice clone, 24 kHz, fp16 |
| TTS (Egyptian, default) | VoiceTut-TTS (`mohammedaly22/VoiceTut-TTS`) | 2026-08-13, replaces OmniVoice for Egyptian — a fine-tune of THIS project's own OmniVoice base (confirmed via HF's `base_model:k2-fsa/OmniVoice` tag), zero new deps. Root cause this exists at all: OmniVoice's own training data is only ~23h Egyptian vs ~204h Najdi/~1484h MSA. Promoted to default after live-testing beat the two prior Egyptian candidates — Habibi-TTS and Lahgtna-OmniVoice, both fully removed (packages uninstalled, checkpoints deleted). Cleanest license/data story of any Egyptian candidate tried: Apache-2.0 confirmed with no discrepancy, ~380h disclosed Egyptian training audio. `VOICETUT_TTS_ENABLED=0` reverts to OmniVoice+Egyptian-clip. See `eval/BASELINES.md`. |
| Audio encoding | lameenc (PCM → MP3) | complete MP3 per sentence (browser `decodeAudioData`) |
| Python isolation | venv (no Docker — no sudo) | GPU works out of the box; managed with `uv`, not `pip` (pip is intentionally absent — see `requirements.txt`) |

Env knobs: `LLM_NUM_CTX` (default 8192), `CATT_ENABLED` (default 1), `OMNIVOICE_MODEL`, `OMNIVOICE_DEVICE`, `VOICETUT_TTS_ENABLED` (default 1).

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
9. `language` gates CATT tashkeel (Fusha only) AND selects the voice/engine: `"egyptian arabic"` turns try, in order, VoiceTut-TTS (only if `VOICETUT_TTS_ENABLED=1` — the default Egyptian engine since 2026-08-13), then OmniVoice's Egyptian clone prompt + `language="egyptian arabic"` if VoiceTut is disabled/unavailable/fails for that sentence, then the Saudi clone prompt if the Egyptian clip is also missing; **every other value keeps the exact legacy call** — Saudi clone prompt via OmniVoice, no `language` kwarg (pinned by `eval/test_tts_args.py`). `generate()`/VoiceTut's `generate()` are both serialized behind the same `_gen_lock` (barge-in orphan-synthesis race; neither is documented thread-safe). VoiceTut-TTS is a fine-tune of THIS project's own OmniVoice base (confirmed via HF's `base_model:k2-fsa/OmniVoice` tag + `config.json`'s `model_type == "omnivoice"`) — Apache-2.0 confirmed in both the model card AND HF's structured license field (no discrepancy), ~380h disclosed Egyptian training audio.
   **History** (full detail in `eval/BASELINES.md`): three prior Egyptian engines were tried and removed before VoiceTut-TTS became the default. Habibi-TTS (SWivid/Habibi-TTS, F5-TTS architecture) was the original default from 2026-07-30 — its EGY checkpoint license was stated Apache-2.0 in README prose but tagged `cc-by-nc-sa-4.0` at the HF-repo level, an unresolved discrepancy. Lahgtna-OmniVoice (`oddadmix/lahgtna-omnivoice-v2`) was an opt-in A/B candidate from the same date — training data entirely undisclosed, no license tag at all. EGTTS-V0.1 (`OmarSamir/EGTTS-V0.1`) was trialed 2026-08-11 and removed 2026-08-12 — explicitly non-commercial (CPML). Owner live-tested all of them by ear; none was judged satisfactory until VoiceTut-TTS. Habibi and Lahgtna were both **fully removed 2026-08-13** (code, packages, downloaded checkpoints) once VoiceTut-TTS was confirmed better — not left dormant.
   **Egyptian tashkeel — tried twice, removed twice** (full detail in `eval/BASELINES.md`): CAMeL Tools' BERT-based Egyptian disambiguator was wired as an opt-in diacritizer (mirroring CATT's Fusha role) on 2026-08-10, found "not working perfectly" by ear, and fully removed the same day. Re-added 2026-08-13 for a second live test at the owner's explicit request (independent corroboration had emerged in the meantime — Lahgtna-OmniVoice's own v1→v3 model card documents the identical finding from a different angle, that diacritized Egyptian text "loses coherence and babbles"). The second live test also came back bad ("completely bad" per the owner), and it was fully removed again 2026-08-13. Current state: no diacritization step for Egyptian at all — the research conclusion behind both attempts (Egyptian colloquial mostly lacks the i'rab/case-ending ambiguity CATT exists to resolve for Fusha) has now been confirmed by ear twice; not planned to be tried a third time absent a genuinely different candidate.

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
├── tts_omnivoice_v1.py    ← TTS module (OmniVoice + CATT + VoiceTut-TTS; Saudi + Egyptian voices)
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
- **Egyptian (2026-07-20, staged reintroduction under the byte-invariant):** `looks_egyptian`/`requested_egyptian` in `routing.py` (guarded request pattern — "Egyptian Museum" and negated «لا ترد بالمصري» never fire; دول demoted to weak, MSA collision); Najdi-first precedence in `build_turn`; `EGYPTIAN_CARD` appended per-turn on Egyptian-routed turns only — **never mention Egyptian material on Najdi turns** (measured pink-elephant regression, see `NAJDI_NO_OTHER_DIALECTS_RULE`); rolling history is CLEARED whenever a turn crosses from one Arabic dialect to a different one (`llm.crosses_dialect_boundary` — generalized 2026-07-27 from Egyptian-only to all Najdi/Fusha/Egyptian pairs; English/mixed turns never trigger it; accepted cost is a higher spurious-clear rate for Najdi↔Fusha than Egyptian ever had, since Fusha is the default fallback and Najdi recall is imperfect — see `eval/BASELINES.md`); SYSTEM_PROMPT and `stt.py`'s `_AR_INITIAL_PROMPT` are byte-frozen (any future STT-prompt change needs a per-dialect WER A/B first).
- **Eval gates are mandatory:** run `eval/test_routing.py`, `eval/golden_prompts.py`, `eval/dialect_id_eval.py`, `eval/test_tts_args.py` after ANY routing/prompt/TTS change (seconds, no GPU). Never re-capture `eval/golden_fixtures.jsonl` to make a red gate green — a red gate means a frozen surface moved. Baselines live in `eval/BASELINES.md`.
- **MP3 format** — browser `decodeAudioData` needs complete containers, not raw PCM.
- **Sentence-level synthesis** — balances first-audio latency vs audio completeness.
- **No Docker** — no sudo; venv only.
- Dashboards: `/review` (latency + transcripts table), `/logs` (raw JSON).
