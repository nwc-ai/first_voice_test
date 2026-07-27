# eval/ — the non-regression harness for `najdi-test`

Built 2026-07-20 as **Step 0** of the Egyptian-reintroduction plan, BEFORE any Egyptian code.
Purpose: make "Najdi/Fusha did not regress" a machine-checkable fact instead of an assertion.
Case ideas and pinned regressions are ported from `chatterbox-tts:eval/` (the old harness
targeted the retired `server.py` monolith API); all plumbing here is rewritten against this
branch's modules (`routing.py`, `llm.py`, `stt.py`, `tts_omnivoice_v1.py`).

The invariant these gates enforce: **on every turn routed Najdi/Fusha/English/mixed, the
request bytes (SYSTEM_PROMPT + build_turn output + model options), the STT config, and the
TTS args are byte-identical to the frozen baseline** (HEAD `e0faf6c`). Egyptian additions may
only (a) move the explicitly enumerated `may_move` cases to `"egyptian arabic"`, and (b) add
bytes on Egyptian-routed turns.

## Scripts (run after ANY routing/prompt change — the first three take seconds, no GPU)

| script | what it gates | run |
|---|---|---|
| `test_routing.py` | pinned routing behavior: `looks_najdi`, `requested_dialect`, `build_turn` matrix, acceptance regexes. Pin classes: FROZEN (never change), KNOWN-PERMISSIVE (frozen quirks), PRE-EGYPTIAN (flip exactly once at plan Step 2/4, updating the pin in the same commit) | `.venv/bin/python eval/test_routing.py` |
| `golden_prompts.py` | **G1** routing replay (decisions == fixtures except enumerated `may_move` set), **G2** byte-identical `turn_content` on every non-Egyptian route, **G3** hash-freeze of SYSTEM_PROMPT / MODEL_CONFIGS / Whisper Arabic initial_prompt + kwargs / CATT language set | `.venv/bin/python eval/golden_prompts.py` (`--capture` only on an owner-approved baseline) |
| `dialect_id_eval.py` | **G5** labeled-transcript routing: truth vs designed route, incl. the owner-accepted v1 recall cap (collision rows) and accepted-FP watch rows | `.venv/bin/python eval/dialect_id_eval.py` |
| `dialect_ab.py` | **G6** LLM-output quality: fixed question set through the exact production prompt surface, leak-linted; compare summary tables across runs | `.venv/bin/python eval/dialect_ab.py --tag <name>` (needs Ollama) |
| `dialect_ab_heldout.py` | Same production prompt surface as `dialect_ab.py`, but a genuinely SEPARATE, never-tuned-against question set (`dialect_ab_heldout_cases.jsonl`) — the anti-overfitting check. Run ONCE per completed change, after dev-set iteration is believed done; never used to choose between candidate wordings (see the script's own docstring for the full discipline) | `.venv/bin/python eval/dialect_ab_heldout.py --tag <name>` (needs Ollama) |
| `test_tts_args.py` | **G4** TTS argument freeze: fake-model pins on exactly what reaches `OmniVoice.generate()` per language (Saudi call byte-frozen, Egyptian arm exact, clip-missing fallback, CATT truth table) | `.venv/bin/python eval/test_tts_args.py` |
| `dialect_purity_lint.py` | leak/drift rates over live `logs/interactions.jsonl` (needs the `route` block, present from Step 1 on; older rows skipped) | `.venv/bin/python eval/dialect_purity_lint.py --since 2026-07-20` |
| `leak_lint.py` | shared token sets + `find_leaks()` (imported by dialect_ab + purity lint) | library, no CLI |
| `test_leak_lint.py` | regression pins for `leak_lint.py` detection gaps found by manual review (شو stray-Levantine leak, `_TSAWWA_RE` suffix-aware Najdi-verb-family leak) — real-transcript-sourced, not synthetic | `.venv/bin/python eval/test_leak_lint.py` |
| `quality_lint.py` | **DEPRECATED — DO NOT USE, see Rules below.** LLM-judge (qwen3:32b) findings, kept in the tree for reference/history only. Was soft/informational by design, but a manual re-read of its own judge-sweep report (2026-07-21) found it both missed real defects a human catches immediately AND fabricated a verbatim quote that never appeared in the source text. Do not add new `--judge` runs; do not treat existing judge findings in old reports as reliable. | superseded by direct manual transcript review |

Reports from `dialect_ab.py`/`dialect_ab_heldout.py` land in `logs/ab_runs/` (gitignored — the
repo is public). Record summary numbers in `BASELINES.md` (committed). Judge runs over live
production traffic (`dialect_purity_lint.py --judge`) land in `logs/judge_runs/` and get
tracked as a trend line in `PRODUCTION_JUDGE_LOG.md` (both gitignored/committed the same way).

Marker-vetting decisions (which candidate Najdi/Egyptian words got added or rejected, and why)
are logged in `dialect_eval_holdout_report.md` — read it before adding a new marker to
`routing.py`, and add a worked-example row there (even for a rejection) when you do.

## Known limitations (owner-decided, not fixed — do not re-attempt without revisiting the decision)

- **Egyptian subject-verb gender agreement** (e.g. a reply's verb staying masculine — بيبقى —
  when its subject is a feminine noun like البطارية, instead of agreeing — بتبقى). Researched
  directly (2026-07-22, `najdi-q2-wrong-elegant-papert.md`): no regex/lookup shortcut is safe
  here — Arabic's free word order, pro-drop, and coordinated subjects break any "nearest noun"
  heuristic, so a wrong "fix" would produce confident-looking wrong corrections. The only two
  things that would actually solve it are a live self-check pass (ruled out — the live request
  path's latency stays untouched) or a reliable Egyptian-dialect dependency parser (none exists
  in production-quality, dialect-tolerant form — checked `.venv/lib/python3.12/site-packages/`,
  no Arabic NLP library is installed, and CAMeL Tools, the most credible option, is a
  morphological analyzer, not an agreement-checker). Accepted as a documented limitation;
  tracked via ongoing manual review of `logs/interactions.jsonl`, not fixed. Revisit only if it
  proves frequent enough in production to justify reopening the live-self-check latency
  question.
- **ج loanword pronunciation** (transliterated English/French loanwords like اوكسجين
  pronounced with Egyptian's standard hard "g" ج when the source word needs a soft "j," e.g.
  "oxygen"). Researched directly against the installed `omnivoice` package source
  (2026-07-22): OmniVoice has **no word/phoneme-level pronunciation override for Arabic at
  all** — `language=` is whole-utterance only, `instruct` is a closed style-attribute
  vocabulary that rejects free-form instructions, and the pinyin/CMU-dict pronunciation
  override documented in OmniVoice's own README is a Chinese/English-only trained convention
  with no Arabic equivalent anywhere in the package. This is a hard TTS-layer wall, not a
  text/prompt problem — dropped from scope entirely. Revisit only if OmniVoice adds
  pronunciation-control features upstream, or if a different TTS engine is considered for
  loanword-heavy content (a materially bigger change).
- **مرة leaking into Egyptian as "very"** (found via `eval/dialect_eval_full.py`, 2026-07-22:
  4 independent instances, clears the recurrence bar). NOT promoted to `DIALECT_REPAIR_MAP`
  like جداً was — مرة is a genuine, high-frequency homograph (also plain "one time" — أول
  مرة، مرة واحدة، مرة تانية — and "wife" in some registers), and `_repair_pattern`'s
  word-boundary-only substitution has no concept of position; a blind swap would corrupt
  legitimate "one time" sentences. Would need a purpose-built positional regex (مرة
  immediately after an adjective, not preceded by أول/تاني/واحدة) — real design work, not
  attempted. See `BASELINES.md`'s 2026-07-22 full-eval entries for the specific instances.
- **Najdi's سوى-verb family (تسويها/تسويه/etc.) leaking into Egyptian** (same eval, 2026-07-22:
  4 independent instances). Detection already catches every instance (`leak_lint`'s
  `_TSAWWA_RE`); no repair exists because Egyptian's equivalent (يعمل/عمل family) would need
  a conjugation-aware regex mapping each Najdi person/gender/number form to its Egyptian
  counterpart — meaningfully more complex and error-prone than any existing
  `DIALECT_REPAIR_MAP` entry. Documented as a tracked pattern, not fixed.

## Rules

- **Never re-capture `golden_fixtures.jsonl` to make a red gate green.** A red gate means the
  change moved a frozen surface; either the change is wrong, or the owner explicitly approves
  a new baseline (then re-capture in its own commit with the approval noted).
- The `dialect_ab.py` QUESTIONS list is verbatim from the old branch and must not be edited
  (comparability contract). New scenarios (e.g. dialect-switch history probes, plan Step 5/6)
  go in NEW files — `dialect_ab_heldout.py`/`dialect_ab_heldout_cases.jsonl` is one such file,
  added 2026-07-21 as the genuinely-separate held-out check.
- `stt.py`'s `_AR_INITIAL_PROMPT` is a shared surface frozen by G3. Any future change there
  requires a per-dialect WER A/B on labeled audio first (plan §G3).
- Documented negative result (do not re-attempt): naming banned cross-dialect tokens inside a
  Najdi turn instruction RAISED the targeted leak 1.7%→8.3% (`routing.py`
  `NAJDI_NO_OTHER_DIALECTS_RULE`, reverted). Egyptian material must never appear on
  Najdi-routed turns.
- **Do not use `quality_lint.py`'s LLM-judge for quality evaluation — read transcripts
  directly instead (added 2026-07-21).** A manual re-read of its own first full judge sweep
  (`logs/ab_runs/2026-07-21_1713-stepC2-full-judge-sweep.md`) found real defects it missed
  entirely (a stray Levantine leak, a Najdi-verb leak into Egyptian, a `NAJDI_GRAMMAR_RULE`
  بـ-prefix violation, a flat wrong-word error, two factual hallucinations) *and* caught it
  fabricating a verbatim quote (`«مشي»`, flagged as the wrong word for "blue" — that string
  never appears in the actual reply, which says `«ماسي»`) plus several findings whose own
  reasoning text says "not a defect" while still surfacing as a flagged finding. Confirms the
  concern the tool's own docstring already raised (probabilistic, both false-positive- and
  false-negative-prone) — direct manual review is now the standard practice; `quality_lint.py`
  stays in the tree for reference only, not for active use.
- **Turning a production-discovered pattern into a fix, without overfitting** (added
  2026-07-21, `najdi-q2-wrong-elegant-papert.md` Part A.5):
  1. Confirm recurrence — the pattern must appear in ≥3 different real utterances/turns (cited
     by timestamp from `dialect_purity_lint.py`'s offending-turns list or `--judge` findings),
     not the same row re-read.
  2. Classify the fix shape **before** writing any wording (the pink-elephant gate):
     - Generalizable grammar/morphology pattern → phrase as a rule about the pattern class,
       never naming the literal observed token (the `NAJDI_GRAMMAR_RULE`/`EGYPTIAN_GRAMMAR_RULE`
       shape).
     - Single closed-form, extremely-high-base-rate token (the جداً shape, see
       `routing.DIALECT_REPAIR_MAP`) → prefer a deterministic, non-prompt fix, not a prompt
       rule at all.
     - Would require naming a set of specific forbidden tokens/dialect names → rejected shape
       (`NAJDI_NO_OTHER_DIALECTS_RULE`'s reverted precedent above); document as an open issue
       rather than force a fix likely to backfire.
  3. Validate on: (a) the dev set (`dialect_ab.py`), all affected dialects, no regression
     within the ±2-noise rule; (b) a fresh `dialect_purity_lint.py --judge`/`leak_lint` sample
     over production logs taken *after* new traffic accumulates post-fix, confirming the
     pattern actually dropped on genuinely unseen phrasing; (c) a `dialect_ab_heldout.py` run
     as a third, independent confirmation — especially for `SYSTEM_PROMPT`-level changes.
  4. Record the outcome either way (success or documented negative result) in `BASELINES.md`
     and, if a production pattern motivated it, `PRODUCTION_JUDGE_LOG.md`.
