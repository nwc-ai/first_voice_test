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
| TTS (Egyptian) | mohammedaly22/VoiceTut-TTS | Egyptian-tuned OmniVoice fine-tune, same API, ~3 GB fp16, preloaded at startup (`VOICETUT_PRELOAD=0` for lazy) |
| Audio encoding | lameenc (PCM → MP3) | complete MP3 per sentence (browser `decodeAudioData`) |
| Python isolation | venv (no Docker — no sudo) | GPU works out of the box |

Env knobs: `LLM_NUM_CTX` (default 8192), `LLM_MODEL` (default qwen3.5:27b — opt-in Fanar-2 override), `CATT_ENABLED` (default 1), `OMNIVOICE_MODEL`, `OMNIVOICE_DEVICE`, `VOICETUT_MODEL`, `VOICETUT_DEVICE`, `VOICETUT_PRELOAD` (default 1 — loads at startup like OmniVoice; 0 = lazy-load on first Egyptian turn), `EGY_REPAIRS` (default 1 — Egyptian lexical repairs).

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
- **Segment-wise CATT for mixed MSA+English sentences** (owner-approved 2026-08-20): CATT DELETES Latin words from its output (verified — English app names silently vanished from Fusha audio). `_add_tashkeel` now diacritizes Arabic runs and preserves Latin runs verbatim; pure-Arabic sentences keep the original single-call path byte-identical. Known non-fixable at text level: حج → "Hagg" on the MSA voice is OmniVoice phonology (CATT already outputs الْحَجِّ correctly).
- **Numbers are verbalized to Arabic words before synthesis, on all Arabic routes** (owner-approved 2026-08-24, layered fix): CATT DELETES raw digits outright (verified: 911/60/123 vanished — every number in every Fusha reply was silently dropped from audio), and the voice models improvise unreliable readings elsewhere. Layer 1: digit runs are protected from CATT like Latin runs (`_PROTECTED_RUN_RE`). Layer 2: standalone INTEGERS → Arabic words via num2words (`_digits_to_arabic_words`, both TTS modules; synthesis-time only — display keeps digits). Decimals/times stay digits (num2words' Arabic decimals are broken: "3.5"→"ثلاثة، خمسون") and survive via layer 1. 911 is digit-by-digit (تسعة واحد واحد) by an earlier rule that wins over the cardinal. Known quirk: num2words gives nominative forms (ستون even after كل).
- **Latent Silma-era bug found & fixed 2026-08-24**: the ق.م→قبل الميلاد abbreviation rule had an optional dot, so it matched the قم inside رقم/الرقم — the audio said "الرقبل الميلاد" for "الرقم" (synthesis-only, display never showed it). Dot now mandatory + preceding-Arabic-letter forbidden, both TTS modules, regression-tested.
- **CATT deletes EVERYTHING outside the Arabic alphabet** (exhaustive sweep 2026-08-27): Latin, all digit systems, ALL punctuation (mid-sentence too), all symbols (٪ ﷼ + = emoji), tatweel — it regenerates text char-by-char from an Arabic-only alphabet. It preserves all Arabic letterforms perfectly, handles pre-diacritized input, no tail-truncation to ~600 chars, but **crashes (ONNX, 1024-position encoder limit) on very long sentences** — `_add_tashkeel`'s exception fallback covers that (plain undiacritized text). Guards now in place: Latin/digit/punctuation runs protected via `_PROTECTED_RUN_RE` (owner-approved 2026-08-27 — mid-sentence ، pause cues now survive; CATT works clause-by-clause), symbol expansions ٪→بالمئة، ﷼→ريال، +→زائد، =→يساوي in both TTS modules, and `_restore_pausal_form` peels surviving trailing punctuation before the waqf strip. Second latent glue-rule bug fixed same day: the digit-separation rules used the raw ؀-ۿ block whose "letter" side included eastern digits — ٥٠ was split into "٥ ٠".
- **Pausal-form (waqf) restoration on Fusha sentences** (owner-approved 2026-08-22): CATT writes the full case ending on the sentence-final word (الْإِسْلَامِ، شَهْرِيٍّ) and deletes the final ./؟ — audibly an "extra vowel" at sentence ends. `_restore_pausal_form` strips the final word's vowel/tanwīn (shadda/sukūn kept, works in either Unicode mark order) and re-appends the original trailing punctuation. Mid-sentence tashkeel untouched; Najdi/Egyptian/English routes never enter this path.
- **Egyptian pronunciation fixes are per-word, synthesis-time only** (`_EGY_PRONUNCIATION_FIXES` in tts_voicetut_v1: المشي/مشاهدة/التنبؤ hand-diacritized) — no CATT on Egyptian ever; display text stays undiacritized. وايد (Gulf/Najdi "very") is repaired to كتير in Egyptian output via EGY_REPAIRS.
- **Najdi vs Fusha routing** is lexicon-based on normalized text (see `_NAJDI_MARKERS`/`looks_najdi` in `pipeline/routing.py` and the MSA→Najdi glossary in the Najdi turn instruction).
- **Egyptian routing is tiered** (`route_arabic`): Najdi-exclusive markers win, then Egyptian-exclusive (مش/ده/دي/عايز/دلوقتي/النهارده/كده/فين/بتاع/curated م…ش forms), then shared markers (اللي/عشان/لسه/يلا — pan-dialectal, NEVER decisive) keep the pre-Egyptian Najdi behavior, else Fusha. `_NAJDI_MARKERS`/`looks_najdi` are byte-identical to pre-Egyptian (they also back the CATT gate). ليه is glossary-only, not a marker; دول and جداً are markers/forbidden NOWHERE (user constraints).
- **EGYPTIAN_CARD is per-turn, Egyptian turns only** — the shared SYSTEM_PROMPT is untouched. The card names only Egyptian's own correct forms (pink-elephant lesson: naming forbidden other-dialect tokens measurably increases leaks — see NAJDI_NO_OTHER_DIALECTS_RULE).
- **Arabic dialect-history isolation** (`_visible_history` in server.py, ALL Arabic pairs incl. Fusha↔Najdi — owner's decision 2026-08-18): each history pair is tagged with its reply route; an Arabic turn's prompt sees only same-dialect + English/mixed pairs. Withheld, never deleted; English↔Arabic behavior unchanged. **NO exceptions (owner decision 2026-08-21)**: the original explicit-request bypass ("قلها بالمصري" saw full history) was removed after live Fanar testing proved it was the cross-dialect contamination vector — the model copied the visible other-dialect answer instead of generating fresh. Known accepted cost: a bare "say that in X" right after an answer in a different Arabic dialect loses its referent (re-ask the full question); after an ENGLISH answer it still works (English pairs are always visible).
- **Egyptian TTS is a separate module** (`tts_voicetut_v1.py`, scaffold copied from the OmniVoice module — same precedent as Silma→OmniVoice). Preloaded at startup by default (owner decision 2026-08-20, live-proven to fit; `VOICETUT_PRELOAD=0` reverts to lazy) via non-throwing `ensure_loaded()` — a VoiceTut failure never blocks startup; on load failure Egyptian falls back to OmniVoice. Never modify `tts_omnivoice_v1.py` for Egyptian needs.
- **Fanar-2 A/B**: `LLM_MODEL` env override; `MODEL_CONFIGS["fanar"]` (think:False + `strip_think_tokens` safety net, fanar-only). Local model: `hf.co/mradermacher/Fanar-2-27B-Instruct-i1-GGUF:i1-Q4_K_M`.
- **Fanar-only output guards** (QA 2026-08-21, owner directive: fix Fanar without touching qwen): everything is gated on "fanar" in the model name — `FANAR_ARABIC_NOTE` (per-turn, Arabic routes: answer directly, translate cross-dialect restatements fully, keep brands in Latin script), `FANAR_ARABIC_REPAIRS` (الملكة→المملكة dropped-letter fix, any route) and `FANAR_NAJDI_REPAIRS` (Egyptian tokens leaking into Najdi replies: دلوقتي/ده/دي/كده/عايز/observed بـ-verbs → Najdi forms) applied to the token stream BEFORE display/TTS/history. Homographs deliberately NOT repaired (قوي، بقى، دول، جداً — would corrupt valid text); no generic بـ-prefix regex (بيتنا/بيانات collisions). Named-banned-word prompts remain forbidden (measured pink-elephant backfire). `scripts/test_routing.py` refuses to run with `LLM_MODEL` set (it validates the qwen default path).
- **MP3 format** — browser `decodeAudioData` needs complete containers, not raw PCM.
- **Sentence-level synthesis** — balances first-audio latency vs audio completeness.
- **No Docker** — no sudo; venv only.
- Dashboards: `/review` (latency + transcripts table), `/logs` (raw JSON).
