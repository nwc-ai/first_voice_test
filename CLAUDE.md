# first_voice_test — Project Context

## What this project is

A standalone local Arabic/English voice assistant powered by:
- **STT**: faster-whisper large-v3 (int8_float16) + Silero VAD (FRCRN denoiser exists but is OFF by default)
- **LLM**: qwen3.5:27b via Ollama (`/api/chat`, streaming, keep_alive:-1)
- **TTS**: OmniVoice (k2-fsa/OmniVoice, zero-shot voice cloning, 24kHz) — in-process, runs on GPU

It is a **general assistant** that recognizes and replies in the Saudi (Najdi) Arabic dialect + English,
matching the speaker's dialect. **Fusha (MSA) is the default** Arabic dialect when the dialect is unclear.

> Full code reference: see **ARCHITECTURE.md** (file-by-file, function-by-function).

> ⚠️ **LICENSE (commercial use planned):** the OmniVoice *code* is Apache-2.0 but the **model weights are
> CC-BY-NC (non-commercial)** — stated on the k2-fsa/OmniVoice HF README (Emilia training-data constraint).
> Decision (2026-07-06): keep OmniVoice for now; before any commercial deployment either obtain license
> clearance for the weights or switch TTS to **Chatterbox Multilingual v3 (MIT, incl. weights)** — accepting
> weaker dialect pinning (clip-only accent, no `language=` dialect IDs) — or Habibi-TTS's Apache-2.0
> EGY/MSA checkpoints for those dialects. Deferred, not forgotten.

---

## Full voice pipeline

```
User speaks → Silero VAD → [FRCRN denoiser — OFF by default] → faster-whisper STT
  → dialect detect → qwen3.5:27b LLM (reply in the user's dialect)
  → OmniVoice TTS (per-dialect voice clip + language= dialect ID) → MP3 → audio plays back
```

Barge-in supported — **pause-then-confirm (2026-07-06)**: speech onset while the AI is audible
PAUSES playback (`playCtx.suspend()`); the turn is cancelled and the audio cleared only after STT
ACCEPTS the utterance as real speech. Rejected noise → `resume_playback` and the reply continues
(bystander voices used to destroy replies mid-sentence — 10 false kills logged 2026-07-06).
Barge-context utterances must also clear `BARGE_CONF_THRESHOLD=0.55` (vs 0.3 normal) so distant
bystander speech can't hijack the turn — TUNABLE from the `barge rejected: seg_conf` prints.
`barge_diag.log` now logs `[FALSE-BARGE-RECOVERED]` (benign, noise-environment metric) + WS closes.

**Startup warm-up** covers all three engines: Ollama warm generation (`_warm_llm`), one throwaway Whisper
transcribe, and one OmniVoice synthesis (`tts_omnivoice_v1.warm_up()` — which also precomputes the
per-voice clone prompts, so reference clips are never re-encoded per sentence).

---

## Dialects

Recognized + replied-in: **Najdi, Fusha (MSA)**. **Fusha (MSA) is the default** when the
spoken dialect is unclear, or an Arabic reply is requested without naming a dialect. English input → English.
(Fusha is routed through the Saudi voice clip with `language="standard arabic"`; there is no separate spoken-Fusha
classifier — any Arabic that doesn't match a Najdi marker falls through to Fusha.)

Each turn the server decides three things from the user's words:
- **reply dialect** — a committed instruction to the LLM (`_detect_dialect` for spoken dialect,
  `_requested_dialect` for explicit requests).
- **voice clip** — every routed dialect (Najdi/Fusha/English) uses the one Saudi clip
  (`_VOICES` registry + `_resolve_voice` in tts_omnivoice_v1.py; the Egyptian clip/registry key were
  removed 2026-07-09 — see "Key decisions").
- **`language=` ID** — passed to `OmniVoice.generate()` to pin pronunciation
  (`najdi arabic` / `standard arabic`); this fixed the
  Saudi/Egyptian pronunciation mixing where some words came out in the wrong dialect.

STT is also biased toward dialect spelling via Whisper `hotwords` on the forced-Arabic re-pass.

**Dialect-detection details (revised 2026-07-06):**
- Spoken detection (`_detect_dialect`) is marker-based; distinctive Najdi words only — Egyptian's
  marker set (and the whole strong/weak scoring + tie-break mechanism it needed when there were two
  dialects to race against each other) was deleted 2026-07-09 along with Egyptian support itself, so
  this is now a plain "does the text contain a Najdi marker word" check. Shared/pan-dialect words are
  excluded so they don't false-trigger: `عشان`/`وين`/`ليش`/`أيوه`/`تمام`/`هلا`. No marker → Fusha default.
- Explicit dialect requests (`_requested_dialect`) require request context on BOTH sides now: Arabic needs
  the `بال…` / `لهجة …` prefix, and **English names need a dialect noun ("Najdi dialect/arabic/accent")
  or a speak-verb ("reply/speak/say it in Najdi")** — so «المتحف المصري» / "the Egyptian Museum" /
  "Gulf region" can't trigger (still guarded even though Egyptian/Gulf aren't routable dialects any more —
  these are proper nouns that must never be mistaken for a dialect request, full stop). **Negated
  requests are skipped** («لا ترد بالنجدي، رد بالفصحى» → Fusha).
- **Translation questions are not language requests** (`_TRANSLATION_Q_RE`): "How do you say good morning
  in Arabic?" no longer forces an all-Arabic reply at an English learner.
- All of the above is pinned by `eval/test_routing.py` — run it after ANY marker/pattern change.

**Dialect quality (revised 2026-07-06 after the live-conversation eval):**
- The per-turn instruction embeds a **dialect card** (`_DIALECT_CARDS` in server.py) built from the
  owner's cross-dialect glossary: ~15 function-word mappings ("use X, NEVER Y") + morphology rules
  (Najdi future = بـ/راح NEVER the Egyptian هـ prefix; Fusha gender/number agreement). Cards are function words
  + morphology ONLY — no topic phrases — and explicitly say "not a checklist, write naturally", because
  the eval showed bare word lists cause keyword-stuffing (وش dropped into Egyptian grammar). Cards are
  also **meaning-conditioned (2026-07-07)**: راح/بـ = FUTURE ONLY (the model was stuffing راح onto
  18th-century past events), دلوقتي = the present moment only (it was landing inside historical narration).
- **No-prompt-leak — now DETERMINISTIC (2026-07-07)**: SYSTEM_PROMPT rule 13 + the turn wrapper forbid
  mentioning rules/instructions, but the model still opened generic-"in Arabic dialect" turns with
  «بما أنك لم تحدد لهجة معينة، سألتزم بالقاعدة الرابعة…» (twice on 2026-07-06). Every flushed
  sentence-chunk now passes a per-turn **chunk filter** (built in respond_loop, applied inside
  `stream_tts_to_ws`) BEFORE display/TTS: `_META_LEAK_RE` drops rules-narrating sentences (tight,
  first-person meta only — grammar answers about «القاعدة الأولى» and the unknown-dialect note pass),
  and `_DIALECT_FIXUPS` swaps single wrong-dialect words whose replacement fills the IDENTICAL syntax
  slot (جداً→مرة، أوي→مرة، دلوقتي→الحين، كتير→كثير; since 2026-07-08 also the Saudi-dialect
  demonstrative/negation swaps ده/دا→هذا، دي→هذي، كده/كدا→كذا، مش→مو، هيك→كذا — postposed
  Egyptian-style demonstratives map 1:1 onto Najdi's own ones, so the old "needs restructuring"
  exclusion was too conservative). كمان stays un-fixed
  (placement varies). Fixups are skipped on translation questions and Fusha. (Hijazi's and
  Egyptian's own fixup tables were removed with those dialects, 2026-07-09 — Najdi's table
  survives unchanged, since it defends against Egyptian/Hijazi-flavored words leaking into
  Najdi regardless of whether those dialects are routable.)
  Text display is therefore **per sentence-chunk, not per token** — box and voice always carry
  identical, filtered text. Abstention phrases («مش متأكد بصراحة») are ONLY for genuine uncertainty,
  never openers; unknown proper names are omitted, not guessed.
- **Anti-recycling (2026-07-07)**: the model recycled same-topic answers across dialect switches (the
  20:56 "Najdi" purpose-of-life reply was the 20:54 Egyptian one, كمان/دي/مش included). The wrapper now
  gets a contrast note whenever the rolling history contains assistant turns in OTHER Arabic dialects
  («Your earlier answers are in Egyptian — compose every sentence fresh in Najdi»), via the per-connection
  `history_dialects` list. History also stores the DELIVERED (post-fixup) text, so clean exemplars stop
  re-seeding جداً in-context.
- **Prompt layering (2026-07-07 audit):** general behavior rules live ONLY in SYSTEM_PROMPT; the
  per-turn wrapper (`_build_turn_content`, testable) carries only turn-specific content — the dialect
  card + the ROUTED dialect's abstention phrase (never all four — cross-dialect exemplars in-context
  seeded leakage) + fresh-answer/no-meta/name-omission. English turns get an all-English wrapper.
  SYSTEM_PROMPT examples are dialect-neutral (no أيوه). Rule 11: answerable questions are answered
  directly, but unintelligible/cut-off input gets ONE short clarifying question («ما فهمت عليك، ممكن
  تعيد؟») — the old absolute never-clarify ban forced rambling non-answers on garbled STT. Owner chose
  to keep long answers (no length cap rule). Do NOT re-add duplicated rules to the wrapper, and do NOT
  add topic examples/few-shot replies to the cards — that is the overfitting direction.
- **`eval/dialect_purity_lint.py`** measures the cross-dialect leak rate per dialect from the logs.
  Rules were TIGHTENED 2026-07-07 (ده/دي/مش/فين + Najdi حاجة added; **جداً promoted to hard leak** in
  all dialect replies; كيف = Egyptian-only drift; auto-fixed column reads `llm.fixups`), so baselines are
  not comparable across that date: old rules pre-cards Najdi 67% / Egyptian 5%; new rules over all of
  2026-07-06 → Najdi 63% / Egyptian 58% (mostly جداً, now auto-fixed) / Fusha 0%. Run it after every
  prompt/model change; target = 0% leaks in DELIVERED text, and a falling auto-fixed count.
- **Spoken register + domain lexicon (2026-07-07, owner decisions):** git archaeology of the "Najdi
  was better in June" claim showed the June replies the owner liked were CONVERSATIONAL (and, honestly,
  keyword-stuffed Gulf-style mix that fails his own glossary — إيش/شنو/شلون/جداً); today's failure mode
  is lecture-register answers. The dialect cards (not Fusha) now end with `_SPOKEN_REGISTER` ("answer
  the way a knowledgeable local TALKS… never the tone of a written article") — tone only, no mandated
  closing questions. And since the real deployment is a **water-utility field assistant**, Najdi's
  card (the only one left — see the removal bullets below) carries the glossary's FIELD/STATUS words
  (broken=خربان، full=ممتلي); identical technical nouns (خزان/عداد/ضغط/تدفق) need no
  card space. Najdi stays STRICT glossary (وش-only — no شلون/شنو loosening; owner decision).
- **Gulf/Khaleeji REMOVED entirely (2026-07-07, owner decision).** No Gulf card, request pattern,
  fixups, or linter target remain. "in Gulf/Khaleeji dialect" now routes through the unknown_dialect
  branch (Fusha + the supported-dialects note); Arabic «بالخليجي» falls to the Fusha default.
  Gulf-adjacent TOKENS (وايد/شنو/يبيلك…) stay in the linter as forbidden-inside-Fusha words.
- **Hijazi REMOVED entirely (2026-07-09, owner decision).** No Hijazi card, marker set, fixup table,
  or per-dialect linter target remain — Hijazi never had its own TTS voice clip (it always shared
  the Saudi clip with Najdi/Fusha, distinguished only by the `language="hijazi arabic"` pronunciation
  ID, which is also gone). "in Hijazi dialect" now routes through the unknown_dialect branch (Fusha +
  the supported-dialects note), same as Gulf; Arabic «بالحجازي» falls to the Fusha default.
  Hijazi-only TOKENS (إيش/دحين) stay in the linter as forbidden-inside-Najdi/Fusha words.
- **Egyptian REMOVED entirely (2026-07-09, owner decision, same day as Hijazi).** Supported set is
  now: **Najdi, Fusha, English, mixed AR+EN** — that's it. No Egyptian card, marker/weak-marker set,
  fixup table, or per-dialect linter target remain; `_detect_dialect` was simplified from a
  multi-dialect scoring race down to a single "does this contain a Najdi marker" check (the
  strong/weak-marker + tie-break machinery it needed is gone, not just Egyptian's entries in it).
  Unlike Hijazi, Egyptian DID have its own TTS voice clip (`voices/omnivoice-tts-egyptian-24k-v3.wav`)
  — the `_VOICES` registry key + its `_EGY_REF_AUDIO`/`_EGY_REF_TEXT` constants were deleted from
  tts_omnivoice_v1.py, but the WAV file itself was deliberately left on disk, unreferenced (owner
  decision — not deleted like the superseded v1/v2 clips). "in Egyptian dialect" now routes through
  the unknown_dialect branch (Fusha + the supported-dialects note); Arabic «بالمصري» falls to the
  Fusha default. Egyptian's word set (`_EGY` in the linter) stays forbidden-inside-Najdi/Fusha words,
  same reuse pattern as Gulf/Hijazi's. A former Najdi/Hijazi or Najdi/Egyptian 1-1 tie in
  `_detect_dialect` now resolves cleanly to Najdi — intended, not a bug; pinned in `eval/test_routing.py`.
- **`eval/dialect_ab.py`** — like-for-like A/B harness: a FIXED ~15-question set (conversational +
  field + informational) through the exact production prompt surface per dialect, reporting linter
  results + would-be fixups, full replies archived in `eval/ab_runs/` (gitignored). **Run a
  before/after pair around every prompt change**; register quality is judged by ear on the reports.
- Known residual weakness: qwen3.5's **Najdi generation is fundamentally weak** (it may still drift to
  plain MSA without leaking, which the linter can't flag lexically); if cards don't fix it, the
  evidence-backed paths are the Fanar-2-27B A/B or a SADA LoRA (both currently deferred by the owner).

**STT language pick (en vs ar):** Whisper auto-detects the spoken language. Arabic-script confusions
(`fa`/`ps`/… — but NOT `ur`) still force Arabic, but any other non-en/ar guess (Urdu, Norwegian, …) is
resolved by the language-probability distribution — the higher of P(en)/P(ar) wins — instead of blindly
forcing Arabic. This stopped English being transcribed as phonetic Arabic. After a FORCED re-decode the
first-pass lang-prob gate is skipped (it says nothing about the re-decode); the segment-confidence gate
(`exp(avg_logprob)`, replaced the per-word gate) judges the transcript that's actually used.

**STT rejects give feedback now:** utterances that fail the language/repetition gates send an
`stt_rejected` event (browser shows «لم أفهم — حاول مرة أخرى» briefly) instead of silent dead air;
over-long utterances are **truncated at 500 chars, not discarded**; `MIN_TEXT_CHARS=2` so «لا» works;
the repetition filter needs 6+ identical words (emphatic «لا لا لا لا» passes).

**Anti-phantom gates (2026-07-06):** noise clips were becoming phantom turns (Whisper's canonical
"Thank you." hallucination — the model answered a question nobody asked). Two gates in
`_transcribe_blocking`: (1) `no_speech_prob > 0.6` → drop (Whisper's own not-speech signal, was
unused); (2) `_is_hallucination` — canonical outro phrases ("thank you", «شكراً للمشاهدة»…) matched
as the FULL utterance are dropped when the decode carries independent doubt (forced re-decode /
lang_prob < 0.6 / no_speech > 0.4 / conf < 0.7). A clearly spoken genuine "thank you" passes. The
FRCRN denoiser was NOT the missing piece — 2026-07-04 logs show noise phantoms with FRCRN on.

**Tashkeel/diacritization (CATT) — RE-ADDED 2026-07-09, owner decision, `CATT_ENABLED=1` by
default.** Originally evaluated and dropped because CATT is MSA-trained and mis-vocalizes
colloquial words (`علطول`→`عُلْطُولُ`) — that reasoning was never Egyptian-specific, and re-verifying
it live (now that only Najdi + Fusha remain) confirms the same class of problem for Najdi: Fusha
diacritizes cleanly (it IS MSA), but Najdi words can be semantically misread — `مرة` ("very") comes
back as `مَرّةً` (the unrelated MSA noun "a time/once"), `صج` ("really") comes back as `صَجَّ` (an
unrelated real MSA verb root, "he shouted"). Owner chose to enable it for BOTH dialects anyway,
accepting that risk for Najdi. `tts_omnivoice_v1.py` diacritizes text just before OmniVoice
synthesis (`_add_tashkeel`, gated on `language` being Najdi/Fusha) — the browser text box is
untouched (undiacritized), only the audio's pronunciation input changes. Falls back to plain text
on any CATT error. `CATT_ENABLED=0` is the fast revert (same on/off shape as `FRCRN_ENABLED`) if
Najdi mispronunciations turn out to sound bad in practice.

**Per-turn logging (`logs/interactions.jsonl`, expanded 2026-07-06):** each turn records, besides
transcript/response/latency: **`route`** (routing branch, requested + detected dialect, voice,
`tts_language`, translation-question flag), **`stt`** (lang_prob, forced re-decode, seg_conf,
no_speech, which gate dropped it), **`llm`** (`done_reason` — "length" = hit token cap;
history_turns; **`fixups`** — wrong-dialect words auto-swapped before delivery, e.g. "جداً→أوي";
**`meta_leak_filtered`** — a rules-narrating sentence was dropped), and **`cancelled`** (barge-in cut
the turn → response is partial, exclude from quality eval). `response` is the DELIVERED text (post
filter/fixups). The `/review` dashboard shows a Dialect column (hover for the full route + STT
signals; ✂ = cancelled, …cap = token-capped, ✎ = auto-fixed, ⛔ = meta-leak filtered).
`_route_turn()` returns this dict; `_transcribe_blocking()` returns `(text, lang, meta)`.

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
| Denoiser | ClearVoice FRCRN — **OFF by default** (`FRCRN_ENABLED=1` to A/B; evidence: enhancement hurts Whisper) |
| Language Model | qwen3.5:27b via Ollama (locked); num_predict 400 + graceful `done_reason=length` handling |
| Text-to-Speech | OmniVoice (k2-fsa, zero-shot) — per-dialect reference voice + `language=` dialect ID; weights **CC-BY-NC** |
| Audio encoding | lameenc (PCM → MP3, 64kbps) |
| Evaluation | `eval/` — routing regression suite, dialect-ID eval, purity linter, dialect A/B harness, per-dialect WER (see eval/README.md) |
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
├── tts_omnivoice_v1.py    ← OmniVoice TTS module (voice registry + clone-prompt cache + gen lock)
├── test_local.py          ← standalone TTS/LLM smoke test (no mic)
├── start_server.sh        ← launch script (Ollama + uvicorn, CUDA libs, SSH-tunnel note)
├── static/
│   ├── index.html         ← browser UI (AudioWorklet mic, WebAudio playback)
│   └── review.html        ← /review latency dashboard (was inline in server.py)
├── eval/
│   ├── test_routing.py    ← routing regression suite (run after ANY marker/pattern change)
│   ├── dialect_id_eval.py + dialect_id_cases.jsonl  ← _detect_dialect measurement (seed set)
│   ├── stt_eval.py        ← per-dialect WER through the production STT path
│   └── README.md          ← how to build the ground-truth set (SAVE_UTTERANCES=1, SADA, MGB-3, ArzEn)
└── voices/
    ├── silma-tts-saudi-24k.wav            ← Saudi male — default voice (Najdi/Fusha/English)
    └── omnivoice-tts-egyptian-24k-v3.wav  ← UNREFERENCED since Egyptian's removal (2026-07-09);
                                              left on disk on purpose, unlike v1/v2 (deleted, in git history)
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
- **Voice registry** (`_VOICES`): every routed dialect → the one Saudi clip (Egyptian's clip and
  registry key were removed 2026-07-09, the WAV left unreferenced on disk); extensible again later
  by dropping a WAV + one registry entry.
- **Tashkeel RE-ADDED 2026-07-09** — CATT diacritizes Najdi/Fusha text just before synthesis
  (`CATT_ENABLED=1` default, `_add_tashkeel` in tts_omnivoice_v1.py); `language=` + the reference
  voice still handle the dialect-identity side of pronunciation. Known accepted risk: CATT can
  still misread Najdi function words (`مرة`/`صج`) as unrelated MSA words — `CATT_ENABLED=0` reverts.
- **MP3 per sentence** — browser `decodeAudioData` needs complete MP3 containers; one per WS message.
  (Measured: encode ~6 ms/sentence, synthesis RTF ~0.09 — the worker outruns playback >10×; do NOT
  bother switching to PCM/Opus or parallel synthesis.)
- **Sentence-level synthesis** — balances latency vs audio completeness. OmniVoice cannot stream
  (architecturally — open issue #77); sentence-chunking IS the correct usage.
- **Sentence-chunk text display (2026-07-07)** — the browser gets text at the same flush points that
  feed TTS (not per token), because every chunk passes the meta-leak filter + dialect fixups first;
  a sentence can only be rewritten/dropped BEFORE any of it is shown. Audio timing unchanged (TTS
  always waited for full sentences). `stream_tts_to_ws(chunk_filter=…)` is the hook.
- **FRCRN OFF by default (2026-07-06)** — three 2022–2026 studies show single-channel enhancement
  before Whisper-class ASR is neutral-to-harmful (worst for large models), the browser already applies
  noiseSuppression, and it only ever ran on ≤4 s clips (9/49 logged turns). `FRCRN_ENABLED=1` re-enables
  for an A/B; delete the code path once the A/B agrees.
- **Truncation guard** — replies cut by `num_predict` (`done_reason=="length"`) show the dangling tail
  in the text box but never SPEAK it (sounded broken mid-word).
- **`generate()` serialized** (`_gen_lock` in tts module) — a barge-in orphans the in-flight synthesis
  thread; the lock stops the next turn's synthesis from running concurrently on the same model object.
- **Single-connection enforcement** — a new browser tab sends close code 4001 to the old tab, which does NOT reconnect.
- **Eval before tuning** — `eval/test_routing.py` pins every routing false-positive fixed in the 2026-07
  review; `SAVE_UTTERANCES=1` collects real audio for the WER ground-truth set (eval/README.md).

---

## VRAM (measured)

| Process | VRAM |
|---|---|
| Ollama / qwen3.5:27b | ~15.7 GB (separate process; leaves ~10 GB free on the card) |
| OmniVoice TTS | **~2.4 GB** (NOT the 6–7 GB previously assumed) |
| faster-whisper large-v3 (int8_float16) | ~3 GB |
| FRCRN + Silero VAD | ~0.5 GB |
| CATT tashkeel (ONNX) | **0 GB — CPU-only**, measured via `nvidia-smi` before/after load |

There is comfortably more headroom than the earlier "~31.3/32 GB" estimate implied.
