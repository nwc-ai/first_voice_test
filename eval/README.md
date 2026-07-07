# eval/ — measurement harness for the voice assistant

Until 2026-07 the repo had **no eval at all** — every dialect/model decision was by ear.
This directory makes the three quality claims measurable. Run everything with the project
venv: `.venv/bin/python eval/<script>`.

## 1. `test_routing.py` — routing regression suite (fast, no GPU)

Pass/fail tests over the pure-logic layer (`_detect_dialect`, `_requested_dialect`,
`_route_turn`, injection/repetition/translation-question regexes). Every false positive
confirmed in the 2026-07 technical review is pinned here. **Run after ANY change to
markers, patterns, or routing** — it takes seconds:

```bash
.venv/bin/python eval/test_routing.py
```

## 2. `dialect_id_eval.py` — spoken-dialect classifier measurement (fast, no GPU)

Reports recall + cross-dialect confusion of `_detect_dialect` over labeled transcripts.
Ships with `dialect_id_cases.jsonl`, a **synthetic seed set** (marked per-row in `source`)
— useful for catching regressions, not for absolute numbers. What must stay ~0 is
**cross-dialect confusion** (an Egyptian utterance routed as Hijazi switches the voice
and pronunciation). Missing a dialect (→ Fusha default) is the designed, safe failure.

```bash
.venv/bin/python eval/dialect_id_eval.py                # seed set
.venv/bin/python eval/dialect_id_eval.py my_cases.jsonl # your labeled data
```

## 3. `dialect_purity_lint.py` — cross-dialect leak rate (fast, no GPU)

Scans the Arabic replies in `interactions.jsonl` and flags tokens that must not appear in the
routed dialect (Egyptian هـ-future or ده/دي/مش/دلوقتي in a Najdi reply, الحين/وش in an Egyptian
reply, any dialect word in Fusha), plus a softer MSA-drift count (حيث/مليء; كيف for Egyptian
only), plus an **auto-fixed** column: turns where the server's `_DIALECT_FIXUPS` swapped a
wrong word before delivery (`llm.fixups` in the log). Since 2026-07-07 the delivered text is
clean for جداً-class words *by construction* — auto-fixed is the honest model-quality signal.
Built from the owner's cross-dialect glossary. **Run after every prompt or model change**:

```bash
.venv/bin/python eval/dialect_purity_lint.py --since 2026-07-07
```

Baselines — numbers are NOT comparable across the 2026-07-07 rule change (ده/دي/مش/فين added,
حاجة for Najdi, and **جداً promoted from soft drift to hard leak** in all dialect replies — every
card says NEVER جداً and it was the top violation, 20/33 Egyptian replies on 2026-07-06):
- Old rules, 2026-07-06 pre-cards: Najdi 67%, Egyptian 5% + heavy جداً-drift, Fusha 0%.
- New rules, all of 2026-07-06 rescored: Najdi 63%, Egyptian 58% (almost entirely جداً — now
  auto-fixed server-side), Fusha 0%. Target for delivered text from 2026-07-07 on: **0% leaks**;
  watch the auto-fixed column to see whether the *model* is actually improving.

Known limitations: it catches wrong-dialect *words*, not (a) a reply that drifts to plain MSA
without using any forbidden token (turn 45's failure mode — watch MSA-drift + judge by ear),
(b) grammar errors (التاريخ دي), (c) factual fabrication, or (d) the Egyptian بـ-present
(بيفتخروا) inside a Najdi reply — lexically identical to the legitimate Najdi بـ-future, hence
unlintable (see the module docstring). Every flagged token is printed so a human can veto false
positives (e.g. زين as the name "Zain").

## 4. `dialect_ab.py` — like-for-like dialect A/B harness (GPU via Ollama)

Runs a FIXED set of ~15 questions (conversational + water-utility field scenarios +
informational register stress tests) through the EXACT production prompt surface
(`_route_turn` → `_build_turn_content` → SYSTEM_PROMPT → qwen3.5 options) for each dialect,
fresh context per turn. Reports linter leaks/drift AND what the server's auto-fixups would
change, per reply; writes a full markdown report to `eval/ab_runs/` (gitignored).

```bash
.venv/bin/python eval/dialect_ab.py --tag before-mychange   # requires Ollama up
# ...edit prompts...
.venv/bin/python eval/dialect_ab.py --tag after-mychange
```

**Every prompt change should get a before/after pair.** The linter numbers are the floor;
register/naturalness judgment stays with the owner's ear on the saved reports — the script
just guarantees both runs answered the SAME questions (2026-07-07: comparing memories across
different live sessions is how the "Najdi was better in June" confusion happened; the June
replies, recovered from git, failed the owner's own glossary). Do not casually edit the
question set — edits break comparability with all earlier reports.

Baseline (2026-07-07, pre-register-note cards): Najdi 4/15 leaky, Hijazi 2/15, Egyptian 4/15,
Fusha 0/15.

## 5. `stt_eval.py` — per-dialect WER through the production STT path (GPU)

Runs labeled audio through `server._transcribe_blocking` (exact production config:
two-pass LID, hotwords, gates) and reports WER per dialect + en/ar LID accuracy + how many
clips the confidence gates silently dropped.

```bash
.venv/bin/python eval/stt_eval.py logs/utterances/manifest.jsonl
```

### Building the ground-truth set (target: 1–2 h of real audio)

1. **Own usage (best data — your mics, your dialects):** run the server with
   `SAVE_UTTERANCES=1` — every accepted utterance's raw audio lands in `logs/utterances/`
   with a `manifest.jsonl` row `{"audio", "text", "lang", "dialect": null}`. The `text` is
   the *hypothesis* — correct it by hand and fill `dialect` before evaluating (otherwise
   you are measuring the model against itself).
2. **Public dialect test sets** (fill the dialects you can't record enough of — convert to
   the same manifest format, 16 kHz mono WAV):
   - **SADA** (~600 h Saudi multi-dialect incl. Najdi/Hijazi labels, SDAIA) — use test-clean slices.
   - **MGB-3** (Egyptian broadcast, ~16 h).
   - **ArzEn / ArzEn-ST** (Egyptian Arabic–English code-switching).
   - **Casablanca** (multi-dialect incl. Egyptian + code-switch segments).
   Each has its own license/registration — check before downloading; keep the audio out
   of git.
3. Published baseline for context (Open Universal Arabic ASR Leaderboard, whisper-large-v3
   zero-shot on SADA): MSA ≈ 28% WER, Najdi ≈ 49%, Hijazi ≈ 50%, Egyptian ≈ 59%. If your
   measured numbers on close-mic short utterances are far better, that's expected — these
   references are broadcast/spontaneous speech.

### What to use the numbers for

- Before/after any STT change (beam size, gate thresholds, FRCRN on/off via
  `FRCRN_ENABLED=1`, a Whisper dialect LoRA, an engine A/B like Nemotron streaming ASR).
- `dropped-by-gates` is the "assistant silently ignored me" rate — watch it alongside WER.
- LLM dialect-generation quality is NOT covered here — that stays an A/B-by-ear judgment
  (or a rubric over `logs/interactions.jsonl` responses) until a dialect-faithfulness
  judge is added.

### interactions.jsonl is now eval-grade (2026-07-06)

Every live turn logs the full routing decision (`route`: branch, requested/detected dialect,
voice, tts_language), STT quality signals (`stt`: lang_prob, forced, seg_conf, no_speech), LLM
outcome (`llm.done_reason`, `llm.history_turns`) and `cancelled`. Useful queries:

```bash
# dialect-routing distribution over live usage
jq -r '.route | "\(.route) \(.requested_dialect // .detected_dialect // "-")"' logs/interactions.jsonl | sort | uniq -c
# turns that hit the token cap (answers too long for num_predict)
jq 'select(.llm.done_reason == "length") | .transcript' logs/interactions.jsonl
# exclude barge-cancelled (partial) turns from any response-quality scoring
jq 'select(.cancelled != true)' logs/interactions.jsonl
```
Rows written before 2026-07-06 lack these fields — the /review dashboard shows them as
`(pre-route-log)`.

`SAVE_UTTERANCES=1` manifests now also carry `dialect_pred` (the classifier's guess) next to
the human `dialect` label — label faster by correcting instead of writing from scratch, but
never copy `dialect_pred` blindly (that would just measure the classifier against itself).
