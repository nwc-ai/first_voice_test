# Committed baselines — `najdi-test`

Numbers only; full A/B reports stay in `logs/ab_runs/` (gitignored — repo is public).
These are the "before" side of every future before/after. Do not edit past rows; append.

## 2026-07-20 — Step 0 of the Egyptian-reintroduction plan (HEAD `e0faf6c`, unmodified)

### Routing / detection (deterministic, `eval/test_routing.py` + `eval/dialect_id_eval.py`)

- `test_routing.py`: ALL PASS (110 pins: FROZEN / KNOWN-PERMISSIVE / PRE-EGYPTIAN classes).
- `golden_prompts.py`: 100 cases captured — 75 frozen, 25 enumerated `may_move` (the ONLY
  cases allowed to change, all to `"egyptian arabic"`). G1/G2/G3 green on unmodified code.
- `dialect_id_eval.py` (65 labeled rows): truth-Najdi routed-as-Najdi **15/17 (88%)**
  (the 2 misses are designed: a no-marker row + the bare-ايه watch row); truth-null false
  dialect firings **2/16, both owner-accepted frozen behaviors** (عشان، يلا as Najdi markers);
  zero unexpected classifications.

### LLM output quality (`eval/dialect_ab.py --tag step0-baseline`, qwen3.5:27b, fresh
context per turn, 15 turns/dialect: 1 spoken-Arabic greeting + 14 English+suffix)

| dialect | turns | leaky | drifty | avg sec/turn | leak tokens seen |
|---|---|---|---|---|---|
| Najdi | 15 | **7** | 0 | 1.7 | جدا ×5، دي ×1، هيك ×1 |
| Fusha | 15 | **1** | 0 | 1.2 | ده ×1 |

Report file: `logs/ab_runs/2026-07-20_1453-step0-baseline.md` (local only).
(The harness gained an `invalid` column — empty/non-Arabic reply on a dialect-routed turn —
after this run; all 30 baseline replies were verified valid Arabic, so the row above is
directly comparable to future runs: treat its invalid count as 0.)

Interpretation notes:
- جداً is the dominant Najdi leak, same as the old branch's finding (its cards said NEVER
  جداً; this branch's `NAJDI_GLOSSARY` maps جداً→مرة). A future Najdi-side fixup/prompt tweak
  is OUT OF SCOPE for the Egyptian plan — this table exists so the Egyptian change can be
  shown to leave these numbers unmoved (within noise), not to improve them.
- The بـ-present leak is not lintable (identical surface to the Najdi بـ-future) — by-ear
  only; `NAJDI_GRAMMAR_RULE` targets it prompt-side.

## 2026-07-20 — Step 4 go-live run (`--tag step4-golive`, Egyptian routing live)

| dialect | turns | leaky | drifty | invalid | avg sec/turn | leak tokens seen |
|---|---|---|---|---|---|---|
| Najdi | 15 | **3** | 2 | 0 | 1.5 | جدا ×2، دي ×1 |
| Fusha | 15 | **1**(*) | 0 | 0 | 1.3 | هندستها (lint FP, whitelisted after this run) |
| Egyptian | 15 | **5** | 0 | 0 | 2.0 | جدا ×4، مشكور ×1 |

G6 verdict: Najdi 3/15 vs baseline 7/15 and Fusha 1/15 vs 1/15 — **no regression** (both
within the ±2 noise rule; Najdi actually better this run). Egyptian first measurement 5/15,
جداً-dominated — the same profile the old branch measured (its reference: 4/15); the
deterministic جداً→أوي fixup is the v1.1 candidate this data supports.
(*) الـFusha "leak" was a linter false positive (هندستها matching the هـ-future regex);
fixed in leak_lint after the run — score it as Fusha 0/15 real leaks.
Report file: `logs/ab_runs/2026-07-20_1610-step4-golive.md` (local only).

## 2026-07-21 — A.1+A.2: Egyptian domain-vocabulary top-up + EGYPTIAN_GRAMMAR_RULE (negation)

Manual-QA-driven fixes (see plan `najdi-q2-wrong-elegant-papert.md`): `EGYPTIAN_CARD` gained a
`settle/sediment=يترسب` word choice + a "technical utility nouns are the same in Egyptian and
MSA" sentence (routing.py); a new `EGYPTIAN_GRAMMAR_RULE` constant targets the ما...ش-onto-a-noun
invalid-negation pattern (e.g. "ما يجفافش"). Both wired into `build_turn()`'s Egyptian branch.

| dialect | turns | leaky | drifty | invalid | avg sec/turn | leak tokens seen |
|---|---|---|---|---|---|---|
| Egyptian | 15 | 7 | 1 | 0 | 1.8 | جدا ×6، وش ×1 |

Report file: `logs/ab_runs/2026-07-21_1313-step7-a1a2-egyptian-vocab-grammar.md` (local only).

Interpretation notes:
- **Targeted defects confirmed fixed by manual read** (leak_lint has no detector for either):
  `short-safe` now says "ترسبات" (settle/sediment root) instead of the previous wrong
  "يترسخ" (entrenched); `smalltalk-heat`'s negation is now "ما يجفش" (valid — يجف is a real
  verb) instead of the previous invalid "ما يجفافش" (جفاف is a noun, can't take ش-negation).
- Leaky count 7/15 vs step4-golive's 5/15 is within/at the edge of the ±2-noise rule
  (`BASELINES.md` below) — not attributable to these changes: neither edit touches جداً or
  وش, and the new وش leak (ar-greeting) is an unrelated single-run variance, not a pattern.
  جداً itself is addressed separately by the deterministic code-level fixup (next section).

## 2026-07-21 — A.3: جداً→أوي deterministic fixup (tts_omnivoice_v1.py, code-level)

Adds `fix_egyptian_leaks()` (tts_omnivoice_v1.py), a deterministic regex substitution
(جداً/جدا → أوي) applied to each fully-materialized sentence in `synth_worker()` right before
synthesis, when `language == "egyptian arabic"`. Also applied to the assistant turn stored in
rolling `history` (server.py) so a follow-up turn doesn't see the model's own raw جداً
reinforced in its own context — the logged `response` field and printed terminal line
intentionally stay RAW (unfixed), so `eval/dialect_purity_lint.py`'s leak measurement on
`logs/interactions.jsonl` isn't blinded to the true LLM-side leak rate.

**Important: `eval/dialect_ab.py` CANNOT validate this fix** — it calls Ollama directly and
inspects raw LLM output only ("LLM + prompt layer only" per its own docstring); it never
invokes `stream_tts_to_ws`/`synth_worker`, so this fixup is entirely outside that harness's
code path. (A `dialect_ab.py --dialects Egyptian` run immediately after this change showed
leaky 2/15 vs the previous 7/15 — this is ordinary LLM sampling variance, NOT evidence of the
fixup, and is not a valid before/after comparison for it; not recorded as a real data point.)

**Actual verification** (direct test of the deterministic function against real جداً
occurrences pulled from prior transcripts):
```
'الجو حار جداً وبيضايق الناس'   -> 'الجو حار أوي وبيضايق الناس'
'بطارية تدوم معاك جدا. شكراً'   -> 'بطارية تدوم معاك أوي. شكراً'
'هتلاقي جدول أعمال اليوم'      -> unchanged (جدول correctly NOT matched)
'في جدال كبير حول الموضوع'     -> unchanged (جدال correctly NOT matched)
'قديم جداً!'                   -> 'قديم أوي!'
'متشكر جداً، دي آخر حاجة'      -> 'متشكر أوي، دي آخر حاجة'
```
All substitutions correct; no false positives on جدول/جدال (words that start with the same
letters but aren't جداً). A true end-to-end check would require running the live server and
listening to synthesized Egyptian audio containing جداً — not done here; the deterministic,
narrow scope of a plain regex substitution makes the isolated function test sufficient
confidence for this change.

## 2026-07-21 — A.4: SYSTEM_PROMPT rules 14 (person) + 15 (gender) + hallucination sentence

Deliberate golden-baseline recapture (see plan `najdi-q2-wrong-elegant-papert.md`): added rules
14 (advice must address the listener, not be framed as the assistant's own first-person want)
and 15 (one consistent grammatical gender per reply) to `SYSTEM_PROMPT`; extended the per-turn
anti-hallucination line in `build_turn()` to specifically caution against confidently-stated
but unverified street/building/historical specifics. `eval/golden_prompts.py --capture` run
immediately after confirming the failure shape was exactly as predicted (G3 SYSTEM_PROMPT hash
+ uniform G2 diff on all 76 non-Egyptian, non-may-move cases) — `golden_fixtures.jsonl`
recaptured, `GOLDEN GATES G1/G2/G3 GREEN` on the new baseline. `test_routing.py` and
`dialect_id_eval.py` both stayed green untouched (as expected — neither surface is affected).

| dialect | turns | leaky | drifty | invalid | avg sec/turn |
|---|---|---|---|---|---|
| Najdi | 15 | 3 | 0 | 0 | 1.6 |
| Fusha | 15 | 0 | 0 | 0 | 1.5 |
| Egyptian | 15 | 5 | 1 | 0 | 1.7 |

Report file: `logs/ab_runs/2026-07-21_1325-stepA4-system-prompt-rules14-15.md` (local only).

**G6 verdict: no regression.** Najdi 3/15 and Egyptian 5/15 match the step4-golive numbers
exactly; Fusha improved to 0/15 (from 1/15, within noise either way).

**Manual read against the specific defects that motivated these rules** (leak_lint has no
detector for any of these three — this is the real acceptance test, per the plan):
- Rule 15 (gender consistency): the Egyptian `advice-phone` turn — the exact turn that
  originally produced the بتاعك/ليكي masculine→feminine mid-reply switch — now stays
  consistently masculine throughout. One data point, but the fix is directly implicated.
- Hallucination sentence: Egyptian `info-jeddah` now correctly says **الرواشين** (rowshan —
  the real architectural term) instead of the previously fabricated "الشراغيف". A genuine,
  specific improvement.
- Rule 14 (person framing): partial effect, not full elimination — Najdi `field-meter` now
  leads directly with "لازم" (the desired pattern), but `greet-city`/`field-tank`/`garbled`
  still open advice with "أبغى + 2nd-person-verb" framing (better than a pure 1st-person
  statement, but not the clean 2nd-person imperative rule 14 asks for). Directionally
  better, matches the plan's expectation of a real-but-partial effect, not a guarantee.
- Hallucination sentence, still failing elsewhere: Fusha `info-jeddah` invents a bizarre,
  confidently-stated claim — "تُعرف محلياً بالقرنبيط" ("known locally as cauliflower" —
  nonsensical as an architectural descriptor). Confirms the plan's own caveat: this is a
  modest, partial mitigation, not a fix for confident domain-knowledge gaps. If this
  persists, retrieval-augmentation (injecting a verified fact snippet for recurring topics
  like "old Jeddah") is the real fix, out of scope here.

## 2026-07-21 — B: eval/quality_lint.py — LLM-judge findings (soft, informational — read the report, don't gate on the count)

New tooling (see plan `najdi-q2-wrong-elegant-papert.md`), not a lettered gate: `quality_lint.py`
feeds each reply to a second local model (qwen3:32b, JSON-schema-constrained via Ollama's
`format` field) as a judge for the six defect classes leak_lint cannot see. Integrated into
`dialect_ab.py --judge` (fixed 15-question set) and `dialect_purity_lint.py --judge
--judge-sample N` (seeded random sample over live `logs/interactions.jsonl`, default N=25).

**VRAM reality, discovered empirically (not assumed):** qwen3.5:27b (17GB) + qwen3:32b (20GB)
do not both fit in this box's 32GB VRAM. `dialect_ab.py --judge` needs BOTH in the same process
(qwen3.5:27b to generate the reply, qwen3:32b to judge it) — confirmed Ollama auto-evicts one
to load the other on every single turn rather than erroring (`ollama ps` verified the swap
completes cleanly), so it works, just slower than a plain `dialect_ab.py` run (tens of seconds
per turn, not ~1-3s). `dialect_purity_lint.py --judge` has no such cost — it only reads
already-recorded log text and never needs qwen3.5:27b at all.

**Validation (targeted, not a full 45-turn sweep — see rationale above):** ran the judge
standalone (`eval/quality_lint.py --dialect ... --question ... --reply ...`) against excerpts
from the original manual-QA transcripts that motivated this tooling:

| defect (original finding) | judge output |
|---|---|
| Egyptian info-jeddah hallucination (invented street/house names) | `factual_hallucination[high]` correctly flagged "الجزيرة القديمة في جدة" and the UNESCO-first-Arab-site claim as needing human verification |
| Najdi/Egyptian advice-phone person-framing ("أبغى تنصحك", "يحتاجك تشوف") | `person_perspective[high]` correctly flagged both, with a corrected suggestion in each case |
| Najdi advice-phone invented word ("تغل") | `invented_word[high]` correctly flagged «تغل» → suggested تقفل — exact match to the original manual finding |

Also ran the full `dialect_purity_lint.py --judge --judge-sample` path against a synthetic
2-row log (not committed — real `logs/interactions.jsonl` was empty at the time) to confirm the
sampling + report-rendering glue code works end-to-end; both synthetic rows produced correctly-
shaped `person_perspective` findings.

**Known limitation, observed directly, not just theorized:** the judge does not catch every
instance of every category in a single pass — e.g. it flagged `person_perspective` on the
Egyptian advice-phone excerpt but did NOT flag the accompanying gender-agreement inconsistency
(بتاعك...ليكي) in the same text on that pass. This is exactly why these findings are soft/
informational, never a gate — expect real signal, not perfect recall.

**Not yet run:** a full `--dialects Najdi,Fusha,Egyptian --judge` sweep (45 judge calls,
non-trivial wall-clock given the per-turn model swap above) and a `--judge` pass over real
production `logs/interactions.jsonl` — do this once real traffic has accumulated post-deploy.

## 2026-07-21 — Part B: generalized `DIALECT_REPAIR_MAP` (replaces Egyptian-only `fix_egyptian_leaks`)

Per the plan `najdi-q2-wrong-elegant-papert.md`: `routing.DIALECT_REPAIR_MAP` +
`apply_dialect_repairs()` replace the old Egyptian-only `fix_egyptian_leaks()` — now covers
**both** Egyptian (جداً/جدا→أوي) and Najdi (جداً/جدا→مرة), since step0-baseline already showed
جداً is Najdi's dominant leak too (جدا×5) and `NAJDI_GLOSSARY` already has the fix pair. Also
fixed real drift: `leak_lint.FORBIDDEN["Egyptian"]` was missing `مرة`/`كثير` despite
`EGYPTIAN_CARD` explicitly forbidding both — added. Deduped `TTS_LANG_TO_DIALECT` (was defined
independently in both `routing.py` and `leak_lint.py`; `leak_lint.py` now imports the one in
`routing.py`).

**Same measurement-scope caveat as the original A.3 fixup applies here, unchanged:**
`eval/dialect_ab.py` calls Ollama directly and inspects raw LLM output only — it never invokes
`tts_omnivoice_v1.synth_worker`/`apply_dialect_repairs`, so it structurally cannot validate a
TTS-side fix. A `dialect_ab.py` before/after around this change is not a valid data point for
it, same reasoning as before — not run/recorded as one here.

**Actual verification:** `eval/test_dialect_repair.py` (new file), all green — 15 checks:
true-positive substitutions sourced from real transcript text (`logs/ab_runs/` — Egyptian
info-coffee "...دور مهم جداً..." → "...دور مهم أوي...", Najdi info-jeddah "...شي مميز جداً..."
→ "...شي مميز مرة..."), the جدول/جدال/جدة (Jeddah) non-collision safety carried forward,
Fusha and `None`-dialect no-ops, plus the two anti-drift invariants: every
`DIALECT_REPAIR_MAP` entry's wrong-word appears in its dialect's prompt card (prose-sync), and
is already in `leak_lint.FORBIDDEN` for that dialect (forbidden-sync). `test_routing.py` and
`dialect_id_eval.py` both re-run and confirmed green (untouched surfaces, as expected).

## 2026-07-21 — C.1: rule 14 bad-example rewrite (remove literal أبغى) — plateaued, kept anyway

Per `najdi-q2-wrong-elegant-papert.md` Part C.1: rewrote `SYSTEM_PROMPT` rule 14's bad-example
to describe the grammatical pattern (first-person intention-verb, not addressing the listener)
without naming أبغى literally — the previous wording repeated the exact rhetorical shape
`NAJDI_NO_OTHER_DIALECTS_RULE` already backfired on. `SYSTEM_PROMPT`-only edit (no `build_turn()`
change this time) — golden-baseline recapture confirmed the cleanest possible diff shape: only
the G3 `SYSTEM_PROMPT` hash changed, **zero** G2 failures (unlike the original rules 14/15
addition, which also touched `build_turn()`'s wrapper and tripped G2 on every non-Egyptian
case). `--capture` run, `GOLDEN GATES G1/G2/G3 GREEN` confirmed after.

| dialect | turns | leaky | drifty | invalid | avg sec/turn |
|---|---|---|---|---|---|
| Najdi | 15 | 4 | 0 | 0 | 1.6 |
| Fusha | 15 | 0 | 0 | 0 | 1.3 |
| Egyptian | 15 | 1 | 1 | 0 | 1.8 |

Report file: `logs/ab_runs/2026-07-21_1708-stepC1-rule14-rewrite.md` (local only). **No
regression** — all three dialects within/better than the established noise range.

**Manual read against the exact four turns flagged in the original A.4 review (the real
acceptance test — `leak_lint` has no detector for person-framing):**
- `greet-city` (Najdi): STILL opens "أبغى تشوف أول شي..." — unchanged hybrid pattern.
- `field-tank` (Najdi): STILL opens "أبغى تشوف أول شي..." — unchanged.
- `advice-phone` (Najdi): STILL "أبغى تنصحك إن تسوي..." — unchanged.
- `garbled` (Najdi): IMPROVED — now "أبغى أشرح لك" (a well-formed first-person statement of
  the assistant's own intended action, "I'll explain to you") instead of the previous broken
  "أبغى تشرح لي" (a mis-formed أبغى + 2nd-person-verb hybrid). Genuine fix on this one turn.

**Verdict: 1/4 improved, 3/4 unchanged — this category has plateaued, per the plan's own
stop-criterion.** The model's association between Najdi advice-giving and an أبغى-led opening
appears to be a strong pattern from its general training distribution, not something a
system-prompt instruction reliably overrides regardless of how the instruction is worded.
**Decision: keep the rewritten wording** (it removes a real, independently-valid pink-elephant
risk and is not worse on any measured number — the Najdi/Egyptian/Fusha leak/drift counts
above are within/better than the existing noise range) **but stop iterating rule 14's prompt
wording further.** All further person-framing effort redirects to `quality_lint`'s
`person_perspective` category (already independently confirmed to catch this exact pattern
reliably, per the original B-section validation) — consistent with how C.2/C.3 already treat
rule 15 and the hallucination sentence.

## 2026-07-21 — C.2: first full `dialect_ab.py --dialects Najdi,Fusha,Egyptian --judge` sweep

Per `najdi-q2-wrong-elegant-papert.md` Part C.2 — this exact sweep was flagged as "not yet run"
in the previous entry; running it now provides the real baseline that was missing before any
further rule-15 wording decision.

| dialect | turns | leaky | drifty | invalid | avg sec/turn | judged |
|---|---|---|---|---|---|---|
| Najdi | 15 | 1 | 1 | 0 | 5.3 | 8 |
| Fusha | 15 | 0 | 0 | 0 | 5.6 | 1 |
| Egyptian | 15 | 4 | 1 | 0 | 5.7 | 9 |

Report file: `logs/ab_runs/2026-07-21_1713-stepC2-full-judge-sweep.md` (local only). leak/drift
numbers are consistent with/better than the established noise range for all three dialects —
no regression. `avg sec/turn` is inflated vs. non-`--judge` runs (~1-2s elsewhere) because
`ask_ollama`'s timer captures the qwen3.5:27b **reload** cost each turn now pays from the
model-swap thrashing against qwen3:32b (documented, expected — see `dialect_ab.py --judge`'s
own help text). One judge call failed cleanly (`JSONDecodeError`, malformed JSON from the
judge model) and correctly degraded to an empty finding list + printed warning, exactly per
`quality_lint.judge_reply()`'s "never raises" contract.

**Important calibration finding from manually spot-checking the actual judge reasoning text
(not just the raw counts) — do not read the `judged` column at face value:**
- **`person_perspective` findings look well-calibrated** — every instance spot-checked (Najdi
  `ar-greeting`, `greet-city`, `field-tank` ×2, `field-meter`) is a genuine, correctly-quoted
  instance of the exact أبغى-framing pattern the manual C.1 read already found — independent
  confirmation, from a completely different signal (LLM judge vs. human read), that person-
  framing is still a widespread issue in Najdi post-rewrite. This corroborates, rather than
  contradicts, the C.1 "plateaued" conclusion.
- **`gender_agreement` has a real false-positive problem in this run.** Spot-checking the
  Egyptian findings (7 of 9 judged Egyptian turns carried a `gender_agreement[high]` flag):
  only ~1 was a genuine mid-reply gender switch (`field-meter`-area: "تخليها مغلقة" (feminine)
  vs. "فلابد تشوف"/"جربت" (masculine) — a real catch). Several others are the judge flagging
  **self-consistent masculine usage** as a "finding" while its OWN note text says "this is
  fine if the default is to address a male listener... this is not a defect" or "no gender
  agreement issue found" — i.e. the judge contradicts itself in the same output, despite
  `quality_lint.py`'s own prompt already explicitly instructing "Do NOT flag a reply that
  consistently uses one gender throughout." One flagged case (`بتعتبر`/موقع) is also a
  different grammatical category entirely (3rd-person subject-verb gender agreement, a real
  but separate Arabic grammar concern) rather than the 2nd-person listener-address-consistency
  rule 15 targets — the category's judge prompt may be catching a broader class of gender
  grammar than intended. All of these were marked `[high]` confidence, so confidence level
  does not reliably separate real catches from these false positives in this run.

**Conclusion for the C.2 question ("does rule 15 need another wording pass?"): inconclusive,
and deliberately not acted on this round.** The raw `judged` counts substantially overstate the
real gender-agreement defect rate in Egyptian because of the judge miscalibration just
described — treating "7/15 Egyptian turns flagged" as "rule 15 is failing 47% of the time"
would be exactly the kind of misleading data point this project's culture already guards
against. The real, disciplined next step (not done here — out of scope for this round, which
was pre-approved as "run the sweep and record it," not "iterate rule 15 wording again") is
tightening `quality_lint.py`'s `gender_agreement` judge prompt/instructions first (a v1.1
candidate — e.g. explicitly re-stating the "consistent-is-fine" exemption more forcefully, or
adding a second-pass self-check step before surfacing this category), so a *future* judge run
produces a trustworthy count before any wording decision is made from it.

## 2026-07-21 — Manual re-evaluation of the C.2 judge sweep (judge LLM discontinued)

Per owner instruction: stop using `quality_lint.py`'s LLM-judge for quality evaluation; check
quality by direct manual transcript review instead (see `eval/README.md`'s Rules — this policy
is now recorded there, not just here). A full manual read of every one of the 45 turns in
`logs/ab_runs/2026-07-21_1713-stepC2-full-judge-sweep.md` (not just the judge-flagged ones)
follows.

**Real defects found that the judge missed entirely:**
- Najdi `garbled`: **"شو"** (Levantine "what," never Najdi وش/إيش) — leaking, undetected.
- Egyptian `greet-city`: **"تسويها"** (Najdi "to do/make," already a vetted `_NAJDI_MARKERS`
  entry) — leaking, undetected.
- Najdi `info-sea`: **"بيقوم بامتصاص"** — a real `NAJDI_GRAMMAR_RULE` بـ-prefix violation
  (the exact pattern that rule targets), on a turn marked entirely clean.
- Najdi `field-meter`: **"ترفع البلاغة"** (rhetoric/eloquence) where **"بلاغ"** (a report/
  notice) was needed — a flat wrong-word error.
- Najdi `garbled`: **"هي الشيء"** — gender mismatch (fem. pronoun, masc. noun شيء).
- Najdi `short-bye`: **"لا يبيلك"** — Najdi negates with ما, not لا; likely should be "ما يبيلك".
- **Two concrete factual hallucinations about the same claim, both wrong, appearing across
  all three dialects' `info-jeddah` turns**: Najdi claimed Jeddah is "أول مدينة عربية" on the
  UNESCO list (false — Historic Cairo 1979, Petra 1985 predate it); Fusha and Egyptian both
  claimed "أول موقع سعودي" (also false — Al-Hijr/Madain Saleh, 2008, predates Jeddah's 2014
  inscription by six years). Same underlying wrong "belief," surfacing on all three routes —
  strengthens (does not reverse) the existing C.3 conclusion that hallucination-mitigation
  wording has plateaued; the actual fix is retrieval-augmentation, still out of scope.

**Fixed** (`eval/leak_lint.py`, `eval/test_leak_lint.py` — new regression tests, all passing,
real-transcript-sourced): `شو` added to `_STRAY` (forbidden everywhere); a new suffix-aware
`_TSAWWA_RE` regex added for the تسوي/يسوي/نسوي/سويت family (forbidden outside Najdi) — a
plain token-set entry was tried first and found NOT to work (Arabic attaches object pronouns
directly, e.g. تسويها, so the whole token never exact-matches a bare root; verified empirically
before committing to the regex approach, same discipline as the جداً fixup). Verified no
false-positive on تسويق ("marketing," an unrelated root). No `SYSTEM_PROMPT`/routing.py changes
needed — detection-only, since these markers were already vetted (شو as a known stray-dialect
word, تسوي already a curated `_NAJDI_MARKERS` entry) rather than new candidates.

**Deliberately left as "watched," not fixed this round** (each is a single occurrence in one
15-turn sample — does not yet clear the A.5 checklist's "≥3 occurrences across ≥2 runs" bar
for promoting a new rule/pattern; watch for recurrence in future runs before acting):
- "ترجع البلاغة" wrong-word, "هي الشيء" gender mismatch, "لا يبيلك" negation choice — each
  looks like ordinary open-class LLM sampling noise (the same category تغل/يترسخ were already
  assessed as), not yet confirmed as a recurring pattern.
- The `NAJDI_GRAMMAR_RULE` بـ-prefix gap in `info-sea` — `leak_lint.py`'s own docstring
  already documents this as structurally unlintable (identical surface form to the legitimate
  Najdi بـ-future) — no safe deterministic detection exists without real tense disambiguation;
  the prompt-side rule already targets it and just isn't fully effective (same ceiling as
  جداً, but harder to backstop deterministically since there's genuine ambiguity here).

**Judge false positives found** (independent confirmation the LLM-judge itself is unreliable,
beyond the fabricated-quote issue that triggered discontinuing it — see `eval/README.md`):
flagged a turn with zero أبغى/want-framing at all (Najdi `greet-city`); flagged a *correct*
rule-13-compliant self-limitation statement as if it were wrong (Najdi `field-leak`); compared
a gendered word against a gender-*neutral* one and called it inconsistent (Fusha `short-bye`);
wrote "not a defect" in its own reasoning while still surfacing the finding (Egyptian
`anchor-purpose`); misidentified the grammatical subject of a sentence, flagging correct
agreement as an error (Egyptian `info-jeddah`); flagged a plain "thanks" as a person-framing
violation (Egyptian `short-bye`).

## 2026-07-22 — Egyptian grammar/leak fixes from real production testing, incl. the جداً/دول correction

Owner ran live production testing (real microphone interactions — `logs/interactions.jsonl`,
34 routed replies, the first real production data this project has had) and shared the log for
review. Full plan: `najdi-q2-wrong-elegant-papert.md`.

### CORRECTION: جداً/جدا is genuinely correct Egyptian Arabic — it was wrongly treated as a leak

Owner-confirmed, 2026-07-22. `EGYPTIAN_CARD`'s "NEVER جداً/مرة" line,
`routing.DIALECT_REPAIR_MAP["Egyptian"]`'s جداً→أوي auto-substitution, and
`eval/leak_lint.FORBIDDEN["Egyptian"]`'s `_JIDDAN` inclusion were all removed — apparently
inherited from an old branch's design choice never actually validated against genuine Egyptian
usage. **Every prior Egyptian leak-rate number in this file that included جداً/جدا in its "leak
tokens seen" column overstated the real defect rate** — do not read those historical rows as an
accurate leak rate without mentally subtracting the جداً/جدا instances. This does NOT extend to
Najdi (جداً→مرة stays correct there, unchallenged, independently confirmed across many runs).

Recomputed precisely on `logs/interactions.jsonl` (`dialect_purity_lint.py`, read-only, before
any code change): Egyptian's real leak rate with جداً correctly excluded was **1/19 (5%)**, not
the 47% (9/19) an unreviewed reading of the raw run would have suggested.

**Second instance of the same mistake, found via a critical self-audit** (owner-requested):
`eval/leak_lint._EGY` also hard-forbade `دول` in Najdi/Fusha — but `routing.py`'s own comment
already documented دول as a genuine MSA homograph ("countries," e.g. "دول الخليج") demoted to a
*weak* marker on the detection side for exactly this reason; that fix was never propagated to
leak-measurement. Removed `دول` from `_EGY`. Regression-pinned both corrections in
`eval/test_leak_lint.py` (دول) and `eval/test_dialect_repair.py` (جداً no-op for Egyptian).

**Unplanned but necessary**: `EGYPTIAN_CARD`/`EGYPTIAN_GRAMMAR_RULE` text changes (this
correction, plus the Part C/D additions below) trip `golden_prompts.py`'s **G2** gate for every
already-Egyptian-routed fixture case — not exempt the way the plan assumed ("Egyptian-routed
turn_content is exempt from G2" is only true for cases whose *route* is moving in that specific
comparison, not for content changes on cases already routed there). Confirmed the failure shape
was uniform (25 G2 failures, identical diff, zero G1/G3) before running `--capture` — golden
gates confirmed GREEN after.

### Part C — new Egyptian lexical fixes (`DIALECT_REPAIR_MAP["Egyptian"]`)

Added: الذي→اللي، تمشى→تمشي، تأكل→تاكل. **`مش فيه`→`مفيش` was considered and REJECTED** for
the deterministic map (unlike the plan's assumption it was a "pure lexical swap") — فيه is
itself a homograph (existential "there is" vs. locative "in it," e.g. "الفلوس مش فيه" can mean
"not inside it"), so it fails the admission bar's "unambiguous in ANY sentence" test. Left as a
prompt-only nudge — `EGYPTIAN_CARD` already taught "there isn't=مفيش" before this session, no
change needed there.

**Live validation, same log**: after landing, `دول`/`الذي`/`تمشى`/`تأكل` correctly fire on real
production text — تأكل+تمشى caught in a *second*, previously-unreviewed turn (14:34:01,
Jeddah-visit question) distinct from the one that motivated the fix (15:37:23), confirming
these are recurring patterns, not one-off overfitting.

### Part D — new `EGYPTIAN_GRAMMAR_RULE` patterns (prompt-side, no deterministic detection possible)

Added three pattern rules (not word lists): weak-final-verb ش-negation allomorph (تنسى→تنساش
vowel shift), بي-habitual-prefix drop after subjunctive/purpose triggers (عشان/لازم/إذا/لو —
يكون not بيكون), Form-V/VI تت-/اتـ prefix requirement (تتجنب not تجنب). Plus جف→نشف
("dry out") added to `EGYPTIAN_CARD`'s word list. Same honest ceiling as every prior
`EGYPTIAN_GRAMMAR_RULE`/`NAJDI_GRAMMAR_RULE` addition — reduces the rate, doesn't guarantee
elimination; no automated way to verify beyond the golden-gate/dev-set regression check
(subject-verb agreement and morphology patterns aren't leak_lint-detectable).

### Part E — two documented limitations (owner-decided, not fixed)

- **Gender agreement** (بيبقى→بتبقى class): confirmed no regex/lookup shortcut is safe
  (Arabic's free word order/pro-drop/coordination breaks "nearest noun" heuristics); the only
  fixes (live self-check, or a reliable Egyptian-dialect parser) are either ruled out (latency)
  or don't exist (checked: no Arabic NLP library installed; CAMeL Tools is the closest option
  but is a morphological analyzer, not an agreement-checker). Tracked via manual log review, not
  fixed. See `eval/README.md`'s new "Known limitations" section.
- **ج loanword pronunciation** (اوكسجين hard-g vs. soft-j): confirmed by reading the installed
  `omnivoice` package source — no word/phoneme-level pronunciation override exists for Arabic at
  all (`language=` is whole-utterance only; `instruct` is a closed style vocabulary; the
  pinyin/CMU-dict override in OmniVoice's own docs is Chinese/English-only). Hard TTS-layer
  wall, dropped from scope entirely.

### Najdi/Fusha observations (flagged per instruction, not silently fixed beyond what's listed)

- **A.1**: the 15:07:01 Fusha turn's heavy Najdi leak (تسوي، تسويه، يبيلك، الحين) is now
  correctly caught by last session's `_TSAWWA_RE` fix — built from a single eval-transcript
  instance, confirmed generalizing to real, independent production data.
- **A.2**: Najdi `info-coffee` (15:14:28) repeated the "منطقة الكاف" coffee-origin
  hallucination — third independent confirmation (2 prior eval runs + this). Clears the
  recurrence bar but stays a documented, deferred hallucination issue (needs
  retrieval-augmentation, materially bigger effort, per the prior session's C.3 conclusion) —
  not new prompt work.
- **A.3**: Najdi `smalltalk`-ish turn (15:15:48) leaked شنو (Gulf/Iraqi "what," already
  correctly forbidden in `FORBIDDEN["Najdi"]`) — single occurrence, watched not fixed.

### Final state, this log, after all fixes

| dialect | replies | with-leaks | leak-rate | msa-drift |
|---|---|---|---|---|
| Najdi | 3 | 1 | 33% | 1 |
| Egyptian | 19 | 3 | 16% | 2 |
| Fusha | 12 | 1 | 8% | 0 |

(Egyptian's rate rose from the جداً-corrected 5% to 16% because Part C's new detection is now
catching real leaks — الذي, تأكل+تمشى — that were previously invisible; this is improved
detection accuracy, not a regression. Najdi/Fusha numbers on this tiny n=3/n=12 sample are not
statistically meaningful on their own, just recorded for the trend line.)

### Dev-set confirmation (`dialect_ab.py --dialects Egyptian`, fixed 15-question set)

| dialect | turns | leaky | drifty | invalid | avg-sec | leak tokens seen |
|---|---|---|---|---|---|---|
| Egyptian | 15 | 0 | 3 | 0 | 1.6s | (none) |

Report: `logs/ab_runs/2026-07-22_2149-2026-07-22-jiddan-correction.md` (local only). Leak count
dropped to 0/15 (from the last non-judge dev-set baseline of 1/15, `stepC1-rule14-rewrite`) —
consistent with جداً no longer being wrongly counted, and no new leak introduced by Part C/D.
Drift ticked up to 3/15 (from 1/15) — all three are pre-existing MSA-connective drift markers
(كيف×2، حيث×1), unrelated to this session's changes and within the established noise range;
not a regression. Full gate suite (`test_routing.py`, `golden_prompts.py`, `test_tts_args.py`,
`test_dialect_repair.py`, `test_leak_lint.py`, `dialect_id_eval.py`) reconfirmed all-green
immediately before this run.

### Gate for later steps (G6)

After each behavior-affecting step (plan Steps 4-6), re-run
`.venv/bin/python eval/dialect_ab.py --tag step<N>` — Najdi and Fusha rows must stay within
noise of this table (rule of thumb on n=15: ±2 turns leaky). Egyptian gets its own row from
Step 4 on (working target: ≤10-15% hard-leak rate before fixups, per the old branch's 4/15
reference).

## 2026-07-22 — Full 245-question eval (`dialect_eval_full.py`), pre-fix baseline

Owner asked to expand the old branch's original Najdi/Fusha question banks (100 questions
across `general`/`tank_level`/`pressure`/`filling_percentage`/`mina_complaint`/
`muzdalifah_complaint`/`arafat_complaint`, decoded from a corrupted paste — see
`eval/dialect_eval_questions.json`'s header history) plus a 60-question routing/holdout set
(`eval/dialect_eval_holdout_questions.json`) with matching Egyptian additions (50 + 35 new
questions), then run and evaluate the whole thing (245 cases total) against the live
production prompt surface. New harness: `eval/dialect_eval_full.py` — unlike `dialect_ab.py`
(fixed English questions + explicit-dialect-request wrapping), this sends each question's
native dialect-phrased Arabic text as-is and lets real routing (`looks_najdi`/
`looks_egyptian`/`requested_dialect`) decide the dialect, so it measures ROUTING accuracy and
REPLY quality together. Deterministic checks only (`leak_lint` + invalid-response detection) —
no LLM judge, per standing policy; qualitative grammar review below is from reading the report
directly (`logs/ab_runs/2026-07-22_2253-2026-07-22-full-245-full.md`, local only).

| dialect | n | leaky | drifty | routing-bad |
|---|---|---|---|---|
| Fusha | 65 | 0 | 0 | 0 |
| Najdi | 65 | 12 | 2 | 3 |
| Egyptian | 115 | 9 | 0 | ~30 |

**Routing — two new, confirmed detection gaps in `looks_egyptian`** (found by tracing why
natural Egyptian test questions failed to route Egyptian; verified directly against
`routing.py`, not inferred from the run alone):
1. **`ليه` (why) was missing from `_EGYPTIAN_MARKERS`** — one of Egyptian's most common
   question words (already taught in `EGYPTIAN_CARD`'s own word list: "why=ليه"), sitting
   right next to إيه/فين/كام which *are* markers. No code comment documents it as a
   deliberate exclusion (unlike عشان/إمتى, which are) — a plain gap. Checked for collisions
   first: zero occurrences of `ليه` in any Najdi/Fusha-expected row across
   `dialect_id_cases.jsonl`/`test_routing.py`; its one existing appearance
   (`dialect_id_cases.jsonl:33`) is already Egyptian-expected. Safe to add.
2. **و/ف/ب-prefixed markers didn't match** — `وإزاي`/`وعايز`-style tokens normalize to one
   glued token (`وازاي`) that never equals the bare marker (`ازاي`) in the set, since Arabic
   attaches these conjunctions/prepositions with no space. `looks_najdi` already strips a
   leading `ال` for the same reason (its own docstring); `looks_egyptian`'s docstring
   explicitly says it does NOT do this ("no definite-article stripping... stripping would
   manufacture false hits from ordinary MSA") — true for `ال`, but و/ف/ب is a different,
   safe-to-strip prefix class since none of the curated Egyptian markers collide with MSA to
   begin with (that's the whole point of a "distinctly-Egyptian vocabulary" set).

Most of the remaining ~30 Egyptian routing misses are the **already-documented, owner-accepted
Najdi-first collision** (اللي/عشان/لسه firing first — CLAUDE.md's "Egyptian speech carrying
pan-dialect Najdi markers still routes Najdi" invariant) — my natural-sounding Egyptian test
questions used these words often (they're genuinely common in real Egyptian speech too), which
just reconfirms the known recall cap at larger scale. Not a new problem, not touched.

**Leaks that now clear the project's own ≥3-occurrence recurrence bar** (`eval/README.md`'s
"Turning a production-discovered pattern into a fix" checklist):
- **مرة leaking into Egyptian, 3× in this run** (E020, EG07, EG15) — combined with the single
  instance found in the 2026-07-22 production-log review earlier this session, this clears the
  bar. NOT promoted to `DIALECT_REPAIR_MAP` this round: unlike جداً, مرة is a genuine, high-
  frequency homograph (also plain "one time" — أول مرة، مرة واحدة، مرة تانية — and "wife" in
  some registers), and `_repair_pattern`'s word-boundary-only substitution has no concept of
  position; a blind swap would corrupt legitimate "one time" sentences. Needs a purpose-built
  positional regex (مرة immediately after an adjective, not preceded by أول/تاني/واحدة) — real
  design work, not a "comfortable" same-shape addition. Documented, not fixed this round.
- **Najdi's سوى-verb family (تسويها/تسويه) leaking into Egyptian, 4×** (E009, E013, EG04, ED4)
  — detection already catches every instance (that's what surfaced them); the finding here is
  that the underlying LLM habit is frequent enough now to be a real candidate for a future
  repair. Not attempted this round — Egyptian's equivalent (يعمل/عمل family) would need a
  conjugation-aware regex mapping each Najdi person/gender/number form to its Egyptian
  equivalent, meaningfully more complex and error-prone than any existing `DIALECT_REPAIR_MAP`
  entry. Documented as a tracked pattern, not fixed.
- هيك (Levantine "like this") leaking into both Najdi and Egyptian, 2× (ND3, E002) — a
  third-dialect leak, correctly caught by existing detection (not a gap); too few instances to
  act on.
- مزيان (Moroccan "good"), 1× (EG19) — a curiosity, watched not fixed.

**Genuine grammar/typo defects found by direct reading** (leak_lint structurally cannot catch
these — not cross-dialect tokens, just wrong/garbled text, each a single occurrence so none
individually clears the recurrence bar, but recorded per the project's manual-review practice):
- Najdi: `فلابس عليك الاتصال` (N036, garbled — probably intended لازم), `بخرانعرفات`/`الما`
  (N046, missing letters — بخزان عرفات / الماي), `بسب ما` for `بسبب` (N038), `الحين اللي فاتت`
  (N043 — contradictory, mixes "now" with "that has passed").
- `منطقة منا` instead of `منى` (E039, Fusha-routed but a spelling error regardless of dialect).
- E003 (Najdi-routed via the عشان collision) stacked "مرة جداً" — both the correct Najdi word
  AND the leaking MSA one together. Worth noting for whenever `مرة`'s repair is designed: if a
  plain substitution fires on this shape it would produce "مرة مرة" (a stutter), a failure mode
  not seen before now.

**Positive findings, worth naming explicitly:** Fusha was flawless across all 65 fresh
questions read (zero leaks, zero grammar issues, appropriately declines to fabricate real-time
tank/pressure/ticket data throughout the 15 new mina/muzdalifah/arafat complaint questions).
Egyptian's grammar rules landed earlier this session (تت-/اتـ prefix, ما...ش vowel shift,
بي-drop after subjunctive) are visibly holding up on entirely fresh, previously-unseen content
(`ما تنسىش`، `اتسد`، `بتتحدى`، `اتعملت` all appear correctly formed). All three dialects
correctly refuse to fabricate live operational data they cannot know — safe behavior, not a
defect.

**Fixes landing this round** (see the next dated entry below for the post-fix numbers): adding
`ليه` to `_EGYPTIAN_MARKERS`, and و/ف/ب-prefix stripping in `looks_egyptian` — both scoped
strictly to Egyptian's own detection function, verified against the full gate suite before and
after. مرة's positional repair and the سوى-family Egyptian mapping are deliberately NOT
attempted this round (design risk outweighs "comfortable, without disturbing other things");
the one-off typos are watched, not actionable (single occurrences, not a repeating pattern).

## 2026-07-22 — Same 245-question eval, after the ليه/prefix-stripping fix

Full gate suite (`test_routing.py`, `golden_prompts.py`, `dialect_id_eval.py`,
`test_tts_args.py`, `test_dialect_repair.py`, `test_leak_lint.py`) reconfirmed all-green
immediately before this run — zero G1/G2/G3 movement, `dialect_id_cases.jsonl`'s Egyptian
recall unchanged at 16/25 (a different, smaller, previously-curated set that doesn't happen to
contain a ليه/و-prefixed row — no contradiction).

| dialect | n | leaky | drifty | routing-bad |
|---|---|---|---|---|
| Fusha | 65 | 0 | 0 | 0 |
| Najdi | 65 | 13 | 1 | 3 |
| Egyptian | 115 | 8 | 1 | ~26 |

Routing-bad total: **47 → 43** (line-by-line diff against the pre-fix report, not just the
summary table). Confirmed by diffing the two full reports directly: exactly **4 cases flipped
from bad routing to correct Egyptian routing** — `E005` (`وإزاي`, the prefix fix), `E028` and
`E043` (`ليه`), `EG02` (`ليه`) — and, critically, **zero other case's routing outcome changed**.
Every other line-level diff between the two runs is a LEAK/drift difference from ordinary LLM
sampling variance (same question, fresh non-deterministic generation each run) — routing itself
is a pure function of the fixed question text, so it cannot vary run-to-run; only these 4 cases
moved, exactly the ones the fix targeted. Leaky/drifty counts are flat within noise (Najdi
12→13, Egyptian 9→8) — no regression.

Remaining ~26 Egyptian routing misses are overwhelmingly the pre-existing, owner-accepted
Najdi-first collision (اللي/عشان/لسه) — not touched, not a new problem.

## 2026-07-24 — Removed اللي/عشان/لسه/يلا from `_NAJDI_MARKERS` (pan-dialect re-partition)

Owner questioned the "Najdi-first collision" tradeoff itself: اللي/عشان/لسه/يلا are genuinely
common in **both** Najdi and Egyptian colloquial speech, not Najdi-exclusive, so using them to
route to Najdi specifically was never really justified — it just happened to be the original
(pre-Egyptian) design. Decided to remove all 4 from `routing._NAJDI_MARKERS` entirely (not
added to `_EGYPTIAN_MARKERS` either — they simply stop being a dialect-discriminating signal).

**Impact quantified by running the real `routing.looks_najdi`/`looks_egyptian` code against
every row in `eval/dialect_id_cases.jsonl` before/after** (not estimated): 8 rows changed —

| text | old expect | new expect | outcome |
|---|---|---|---|
| لسه ما وصلت الفاتورة | Najdi | None | **accepted regression** — genuine Najdi speech, لسه was its only marker |
| إيه اللي حصل النهاردة؟ | Najdi | Egyptian | win — إيه+النهاردة carry it |
| لسه مش عارف أعمل إيه | Najdi | Egyptian | win — إيه carries it |
| عايز أروح البيت عشان تعبت | Najdi | Egyptian | win — عايز carries it |
| يلا بينا نمشي دلوقتي | Najdi | Egyptian | win — دلوقتي carries it |
| هي دي الحاجة اللي كنت عايزها | Najdi | None | partial — stops being wrongly Najdi, but عايزها (suffixed) doesn't token-match bare عايز; a separate, real, out-of-scope gap (same class as `_TSAWWA_RE`/the و-ف-ب-prefix fix), not attempted here |
| عشان كذا قلت لك من البداية | Najdi (accepted FP) | None | win — no longer a false positive on ambiguous text |
| يلا بينا نمشي | Najdi (accepted FP) | None | win — same |

Two more cases surfaced only by running the full golden-fixture gate (not in
`dialect_id_cases.jsonl`): `najdi-elli` ("هذا اللي قلت لك عنه أمس") and
`najdi-collision-elli-haga` ("هي دي الحاجة اللي كنت أدور عليها", no عايز this time) — both
also fall to None, both the same accepted-regression/unresolved-gap shape as above.

**`eval/dialect_id_eval.py` recall, before → after**: Najdi 15/17 (88%) → 14/17 (82%) — the one
accepted regression. Egyptian 16/25 (64%) → **20/25 (80%)** — the net win.

**Files touched**: `routing.py` (`_NAJDI_MARKERS` + two comment blocks), `eval/test_routing.py`
(~10 assertions flipped: lines testing اللي/لسه/عشان/يلا in isolation now expect `False`/
`"standard arabic"`; the three former "FROZEN collision" `build_turn` checks now expect
`"egyptian arabic"`), `eval/dialect_id_cases.jsonl` (8 rows), `eval/golden_prompt_cases.jsonl`
(9 fixture IDs — 4 given `may_move`/`expected_v1: "egyptian arabic"` per the established
`egy-dol-nas` precedent, since `golden_prompts.py`'s `may_move` mechanism can only tolerate a
move to Egyptian, never to Fusha; the other 5 needed a direct, deliberate `--capture`, which is
exactly the "owner explicitly approves a new baseline" case `eval/README.md`'s rule permits),
followed by `.venv/bin/python eval/golden_prompts.py --capture` (confirmed scoped to exactly
these 9 IDs by exhaustively grepping `golden_prompt_cases.jsonl` for all 4 target words first —
the file isn't git-tracked so a `git diff` safety check wasn't available; the exhaustive-grep
+ direct-execution approach was used instead), `CLAUDE.md` (the language-scope paragraph).

Full 6-script gate suite (`test_routing.py`, `dialect_id_eval.py`, `golden_prompts.py`,
`test_tts_args.py`, `test_dialect_repair.py`, `test_leak_lint.py`) green after every step.

**Side effect worth knowing, not gated**: `tts_omnivoice_v1.py`'s per-sentence CATT gate shares
`looks_najdi()`. A Fusha-routed reply sentence containing only one of these 4 words (no other
Najdi marker) now gets CATT tashkeel applied where it previously didn't — no test currently
pins this either way, `eval/test_tts_args.py`'s fixed Arabic strings don't contain any of the
4 words so G4 stayed green untouched, but flagging it so it isn't "discovered" as a surprise.

## 2026-07-27 — Generalized dialect-boundary history clearing (Najdi↔Fusha now clears too)

Owner questioned the Egyptian-only scope of `llm.crosses_egyptian_boundary` (history clearing
on dialect switch): traced the original rationale to the 2026-07-20 Egyptian-reintroduction
planning doc — Najdi↔Fusha was excluded not because those dialects are "close enough" to share
context safely, but purely as a byte-invariance constraint for that specific rollout ("Najdi/
Fusha/English behavior, including history mechanics, must stay byte-identical to the
pre-Egyptian baseline"). That rollout landed 3 days ago; the scoping reason no longer applies.
The actual bug the clearing exists to prevent (in-context stylistic imitation of a 27B model's
own recent replies — documented as the mechanism behind the pre-cards 67% Najdi leak rate) is
not Egyptian-specific, so there's no principled reason Najdi↔Fusha switches would be immune.

**Decision**: generalize to a fully symmetric rule — clear on any transition between two
different Arabic dialects among {Najdi, Fusha, Egyptian}; English/mixed unchanged. Considered
and rejected: a debounce/hysteresis variant for Najdi↔Fusha specifically (cheaper long-term
fix for the noise concern below, but adds complexity without measured need yet); a prompt-side
"disregard the prior dialect's wording" reminder (same shape as `NAJDI_NO_OTHER_DIALECTS_RULE`,
which measurably backfired 1.7%→8.3%); summarizing instead of clearing (extra LLM call,
violates the "keep live latency untouched" constraint); clearing only assistant turns (breaks
the user/assistant pairing invariant `server.py`'s trim logic assumes, and produces a message
shape — consecutive user turns, no assistant reply between them — never sent to the model
before, an unmeasured risk trade for a measured one).

**Known, accepted tradeoff**: Najdi↔Fusha is not a clean discrete boundary the way Egyptian is
— Fusha is the default *fallback* bucket, not a positively-detected register, and Najdi recall
is currently 14/17 (82%, `dialect_id_eval.py`). Over the 3-turn rolling window
(`MAX_HISTORY_TURNS`), that's roughly a 1-(0.82³) ≈ 45% back-of-envelope chance at least one
turn in an otherwise-consistent Najdi conversation gets silently mislabeled Fusha and triggers
a clear nobody asked for — a meaningfully higher spurious-clear rate than Egyptian's boundary
ever had (Egyptian requires distinctive markers or an explicit request, making its crossings
comparatively rare and deliberate). Accepted as the cost of the simpler design; the debounce
variant above is the natural fallback if production data later shows this hurts conversational
continuity in practice.

**Files changed**: `llm.py` (`crosses_egyptian_boundary` → `crosses_dialect_boundary`, logic
generalized from the two special-cased branches to `any(d in _ARABIC_DIALECTS and d !=
turn_label for d in history_dialects)` — actually simpler than the code it replaced),
`server.py` (4 reference points: declaration comment, pre-call comment, call site, log
message — rename/reword only, zero mechanics change), `eval/test_routing.py` (renamed the `B`
assignment; flipped exactly 2 pins — `"najdi after fusha"` and `"fusha after najdi"`, both now
`CLEAR` instead of `keep`; reworded the stale "THE INVARIANT" comment; added 1 new coverage pin
confirming an interspersed English turn doesn't mask an Arabic-dialect difference elsewhere in
history), `CLAUDE.md` (the Egyptian bullet under "Key decisions").

**Verification**: ran the full 6-script gate suite (`test_routing.py`, `dialect_id_eval.py`,
`golden_prompts.py`, `test_tts_args.py`, `test_dialect_repair.py`, `test_leak_lint.py`) both
immediately BEFORE this change (to cleanly isolate its effect from the still-fresh 2026-07-24
marker-set change) and immediately AFTER. Identical numbers both times — Najdi 14/17 (82%),
Egyptian 20/25 (80%), all 6 scripts green — confirming this change touches only history
mechanics, with zero effect on routing/detection/repair (expected: dialect detection itself,
`routing.py`, was not touched by this change at all).

No live multi-turn integration test exists for this behavior in either direction (Egyptian's
original clearing or this generalization) — every eval harness in this repo (`dialect_ab.py`,
`dialect_eval_full.py`) deliberately uses fresh context per turn. The 13 unit pins on the pure
`crosses_dialect_boundary` function are the only automated coverage; a live sanity check (speak
Najdi, then Fusha, then Najdi again; confirm the `[history] cleared at dialect boundary` log
line fires on both switches and replies are composed fresh) is a manual follow-up, not gated.

## 2026-07-27 — Full 245-question re-run, current state (owner-requested check)

Re-ran `eval/dialect_eval_full.py` against the same 245 questions as the 2026-07-22 runs, to
confirm current state after this session's marker-removal (اللي/عشان/لسه/يلا) and
history-clearing generalization. Report: `logs/ab_runs/2026-07-27_1302-2026-07-27-post-history-fix-full.md`.

| | routing-bad, before marker removal | routing-bad, now | delta |
|---|---|---|---|
| Egyptian (n=115) | 32 | 22 | **−10, win** |
| Najdi (n=65) | 11 | 15 | **+4, accepted tradeoff** |
| Fusha (n=65) | 0 | 0 | unchanged |

Egyptian improvement matches the 2026-07-24 marker-removal's intent directly. The Najdi
increase is almost entirely (11/11 of the prior baseline, +4 new) the same two already-known,
already-accepted causes: (a) 11 of these were ALREADY routing-bad before any of this session's
changes — confirmed by testing each string directly against `looks_najdi()` — genuinely
marker-less Najdi phrasing (no lexicon hit at all, e.g. relying on verb-form choice like
يقدر rather than a listed marker word), the same "precision-first design: marker-less Najdi is
an EXPECTED miss" limitation `dialect_id_cases.jsonl` has documented from the start, unrelated
to any code changed this session; (b) exactly 4 NEW misses (`N035`, `O03`, `G18` via عشان;
`O05` via اللي) are the direct, already-accepted cost of removing those words as Najdi markers
(both confirmed to now return `False` for `looks_najdi()` where they returned `True` before).
No unexplained or newly-surprising routing regressions found.

**Leak count worth flagging**: مرة leaking into Egyptian appeared **6 times in this run alone**
(`E003`, `E015`, `E030`, `EG05`, `EG15`, `EG19`) — on top of the 4 prior instances already
recorded in the 2026-07-22 entries. This is now a strongly, repeatedly confirmed pattern (10+
independent instances across three separate runs) and the strongest remaining candidate for a
future fix — still not attempted, for the same reason as before (مرة is a genuine homograph
also meaning "one time"/"wife" in some registers; needs a positional regex, not a plain
`DIALECT_REPAIR_MAP` word-swap). سوى-verb-family leaking into Egyptian also recurred (2-3
instances) — same status, same reasoning, not re-litigated here.

No new leak/grammar pattern found beyond what's already documented in the 2026-07-22 entries.

## 2026-07-27 (later) — CORRECTION: مرة was never a confirmed leak — leak_lint itself was broken

Owner asked to fix مرة "without disturbing other things." Before designing the positional
repair flagged above, checked all 10 real occurrences flagged across the 2026-07-22/27 runs
individually (not just counted) — **every single one was مرة meaning "time/occurrence"**
(`من أول مرة`, `كل مرة`, `مرة واحدة`, `مرة ثانية`), never the Gulf/Levantine "very" sense that
`EGYPTIAN_CARD` actually forbids. Zero genuine intensifier leaks found in any real data. The
"10+ confirmed instances, clears the recurrence bar" conclusion recorded in this file's
2026-07-22 and 2026-07-27 entries above was itself wrong — driven entirely by
`eval/leak_lint.py`'s `FORBIDDEN["Egyptian"]` treating bare `مرة` as an unconditional
word-match, unable to distinguish the two senses of the homograph. **Same shape of mistake as
جداً/دول** (this file's earlier entries) — a detector flagging a legitimate word because it
never accounted for a second meaning. This time it was the *measurement* that was wrong, not
the prompt guidance (`EGYPTIAN_CARD`'s "NEVER مرة" for the intensifier sense is still correct
advice — it just had no working detector behind it).

**Fix**: removed bare `مرة` from `FORBIDDEN["Egyptian"]`'s plain word-set; added
`_marra_leaks()`/`_MARRA_TIME_RE` to `eval/leak_lint.py` — a context-aware regex (same shape as
the existing `_TSAWWA_RE`/`_HA_FUTURE_RE` special-case detectors) that only flags `مرة` when it
appears OUTSIDE a known time-of-occurrence construction (`أول/كل/آخر/تاني/ثاني/كام/من مرة`, or
`مرة واحدة/تانية/ثانية/أخرى/كمان`), including the glued و/ف/ل-conjunction forms (`فكل مرة`,
`لكل مرة`). Verified against all 10 real false-positive sentences (all now clean) and 3
constructed genuine-intensifier sentences (all still correctly flagged, e.g. `الأكل ده حلو
مرة`). No change to `routing.py`/`DIALECT_REPAIR_MAP`/`EGYPTIAN_CARD` — this was purely a
detection-side fix, no repair needed since no genuine leak has actually been observed yet.
Regression-pinned in `eval/test_leak_lint.py` (7 new cases: 5 false-positive-fix, 1
true-positive, 1 confirming مرة stays allowed in Najdi where it's the correct word for "very").

Full 6-script gate suite green after the change, zero effect on anything else (`مرة` was never
a `DIALECT_REPAIR_MAP` key, so `test_dialect_repair.py`'s prose-sync/forbidden-sync checks were
never coupled to it).

**Going forward**: if a genuine مرة-as-"very" leak is ever observed in real data (now that the
detector can actually catch one), it clears the recurrence bar from zero, not from a false
starting count — treat any future report as the first real instance, not a continuation of the
"10+" figure in the entries above, which is now known to be entirely false positives.

## 2026-07-27 (later still) — Full manual read of the current 245-response report

Owner asked directly whether responses were being evaluated manually, not just via leak_lint —
did a complete read of every response in `logs/ab_runs/2026-07-27_1302-2026-07-27-post-history-fix-full.md`
(not just the flagged ones). Findings:

- **Fixed** (see below): `routing.requested_egyptian()` failed to match English explicit-dialect
  requests with an object pronoun between the verb and "in X" — `"Answer me in Masri please"`
  and `"Reply to me in Egyptian please"` both returned `False` (only `"Answer in Masri please"`,
  no pronoun, matched). Live consequence, confirmed in the eval report (`ED2`): the model, never
  told it was an explicit Egyptian request, hallucinated a fake self-imposed rule ("my
  instructions require me to speak only English or Najdi") and replied in English — a real,
  user-facing failure, not a subtle grammar nuance. Checked `WANTS_ARABIC_RE`/`WANTS_ENGLISH_RE`
  for the same gap — confirmed NOT affected, they have a generic `\b(in|into|to)\s+arabic\b`
  catch-all arm with no verb requirement, so a pronoun in between never breaks them (verified
  directly, not assumed — avoided an unnecessary edit there).
- **Confirmed already mitigated, no action needed**: two occurrences of literal CJK characters
  embedded mid-word in Fusha replies (`F005`: "لعب书法家ون عظام", `FD3`: "随着年龄") — checked
  `routing.filter_cjk`/`_UNWANTED_SCRIPT_RE` and confirmed it IS wired into the live pipeline
  (`server.py:446`), so these never reach the user's ears via TTS in production, only visible
  in this raw-generation eval log.
- **Logged, not acted on this round**: a logical self-contradiction in a Najdi reply (`G01`:
  claims phone use before sleep is "very necessary for your health" then advises stopping
  screen use before bed in the same reply); a garbled non-word in Egyptian (`E038`:
  "بيتركونسوا", likely intended "بيتركزوا"); a confused connector in Najdi (`G05`: "وشنو
  يرتبط بأمور أخرى" doesn't parse); a code-switching instruction violation (`CS10`: replied
  100% English on a mixed-language turn, invisible to leak_lint since mixed/English turns skip
  it entirely); a third recurrence of the منى→منا place-name misspelling (`N038`, `E039`,
  clearing the recurrence bar, candidate for a future dialect-agnostic post-processing fix, not
  attempted). Each either single-occurrence (doesn't clear the recurrence bar alone) or noted
  as a candidate for later, not urgent.

**Fix implemented**: `routing._en_dialect_req()` (feeds `_EGYPTIAN_REQUEST_RE`) now tolerates
an optional object pronoun (`me`/`us`/`him`/`her`/`them`/`to me`/`to us`) between the request
verb and "in/into/to <dialect>", via a new shared `_OBJ_PRONOUN_RE` fragment. Verified against
the exact failing sentences plus the existing FROZEN proper-noun and negation guards (all still
pass — `"The Egyptian Museum"`, `"Don't answer me in Masri"` both still correctly `False`). 3
new regression pins added to `eval/test_routing.py`. Full 6-script gate suite green, zero
routing-outcome change on any of the 66 `dialect_id_cases.jsonl` rows or 101 golden fixtures
(this only affects the *explicit English request* pattern, a surface none of those exercise
with a pronoun in this exact position).

## 2026-07-27 (later still) — منى→منا place-name typo fix

Owner asked to resolve the recurring منى (Mina) → منا misspelling (3 independent instances:
2026-07-22's E039, this session's N038 and E039). Unlike the other repairs this session,
this one is **dialect-agnostic** — seen on both Najdi- and Fusha-routed replies — so it can't
live in `DIALECT_REPAIR_MAP` (keyed per-dialect). Also, same homograph risk as مرة: bare منا
is ALSO the very common word "from us" (من+نا — واحد منا، طلب منا، قريب منا), so a blind
substitution was never an option.

All 3 confirmed real instances shared one shape: a domain noun immediately followed by منا
("مخيمات منا", "منطقة منا"). Added `routing._MINA_MISSPELL_RE` — `\b(خزان|محطات?|مخيمات?|
منطقة)\s+منا\b` — gated on that exact adjacency (these domain nouns don't parse as "from us"
in natural Arabic, so the reading is unambiguous), wired into `apply_dialect_repairs()` as a
new unconditional step applied regardless of dialect (including `None`/English-mixed turns,
since a place name can appear in any turn). Verified against both real sentences (both now
correctly fixed) and 3 false-positive safety checks (واحد منا، طلب منا، قريب منا all
untouched). 6 new regression tests added to `eval/test_dialect_repair.py`. Full 6-script gate
suite green — no other file needed a change (the existing two call sites of
`apply_dialect_repairs` in `server.py`/`tts_omnivoice_v1.py` pick this up automatically since
it lives inside the shared function, not a new call site).

## 2026-07-27 (later still) — Fanar-2 wired for a live mic/browser test (owner-requested)

Per the offline 245-question A/B above: recommendation was to keep qwen3.5:27b in production.
**Owner explicitly asked to test Fanar-2 live anyway** — this entry is the live-test wiring,
not a production-default change. `llm.MODEL` is now `os.environ.get("LLM_MODEL_OVERRIDE",
"qwen3.5:27b")` — unset behaves byte-identically to before; `LLM_MODEL_OVERRIDE` must always be
passed inline on the invocation line (`LLM_MODEL_OVERRIDE=... bash start_server.sh`), never
`export`ed into a persistent shell. Added `MODEL_CONFIGS["fanar"]`, identical to `"default"`
(temp 0.7/top_p 0.9/top_k 40/num_predict 300/stop=`_STOP_SEQUENCES`, `extra: {}`, no
`"think": False`) plus one addition: an explicit `num_ctx: LLM_NUM_CTX`. This matters live in a
way it didn't in the offline eval (fresh context per turn there, a few thousand tokens either
way) — `llm.py`'s own `num_ctx` comment already documents that a 27B model's *native* 32768
context OOMs with OmniVoice in-process on this GPU; confirmed live before this change that
Fanar-2 was in fact still pinned at exactly that 32768 context from the offline eval run
(`ollama ps`, `keep_alive: Forever`) — freed with `ollama stop <tag>` before proceeding.

**Gate suite**: predicted and confirmed exactly one failure before `--capture` — `G3
MODEL_CONFIGS: shared surface changed`, zero G1/G2 (confirms only the gated dict changed,
nothing about prompt content or routing did). `test_routing.py`, `dialect_id_eval.py`
(Najdi 14/17, Egyptian 20/25 — unchanged), `test_tts_args.py`, `test_dialect_repair.py`,
`test_leak_lint.py` all green untouched. `golden_prompts.py --capture` run, `GOLDEN GATES
G1/G2/G3 GREEN` reconfirmed after.

**Smoke test**: `test_local.py` rewritten to call the real `llm.py` functions
(`llm.build_turn`/`llm.ollama_chat_token_gen`) instead of a hand-rolled duplicate `/api/generate`
payload — this makes it inherit `LLM_MODEL_OVERRIDE` for free and, unlike the offline eval
(`stream: False`, single call), exercises the actual live streaming path (`stream: True`,
token-by-token `chunk["message"]["content"]`). Added an explicit check that no literal `<think`
substring ever appears in the streamed text — genuinely untested by the offline eval, which
only ever inspected the final joined non-streamed response.

**Revert**: don't pass `LLM_MODEL_OVERRIDE` on the next run — no file edit needed. The
`"fanar"` `MODEL_CONFIGS` entry sitting unused in the checked-in file is harmless (never
selected unless the override is set) and left in place pending an owner decision on whether to
keep it.

**Deferred, not part of this pass**: a new `SYSTEM_PROMPT` rule against model-identity
disclosure (the audit's `F036` "trained by QCRI" persona break has no existing rule guarding
it, unlike Fanar's other issues, which are all *existing* rules — 9, 10, 13, the anti-
hallucination line — that it simply follows less reliably than qwen3.5). Left out because it
touches the qwen3.5-shared prompt surface (its own separate G3 recapture) and isn't a blocker
for the live test itself.

## 2026-07-30 — Habibi-TTS replaces OmniVoice for Egyptian TTS only (Najdi/Fusha unchanged)

Owner reported OmniVoice sounds good for Najdi/Fusha but noticeably bad for Egyptian, and
asked for research into (1) an Egyptian-specific diacritizer alongside CATT, (2) a better
Egyptian TTS engine.

**Diacritization — researched, not adopted.** No production-ready Egyptian-specific
diacritizer exists: CAMeL Tools' CALIMA-EGY gets only an 11% relative improvement on Egyptian
vs. 20% on MSA in its own paper (Zalmout & Habash, ACL 2020); Abjad AI's own dedicated 2025
dialectal attempt (CATT-Whisper, NADI 2025) needed audio input, not just text, and still
scored 55% WER. More fundamentally: Egyptian colloquial writing lacks MSA's i'rab (case
endings), which is what tashkeel actually disambiguates — this project's own architecture
already reflects that by giving Egyptian a dedicated voice-clone clip + language id
(TTS-side), while Fusha's only pronunciation lever is text diacritics. CATT stays exactly as
it was (Fusha-only, `_TASHKEEL_LANGUAGES = {"standard arabic"}`, untouched).

**TTS root cause, found directly in OmniVoice's own docs**: `k2-fsa/OmniVoice`'s
`docs/languages.md` lists per-dialect training hours — Standard Arabic 1,483.5h, Najdi Arabic
203.5h, **Egyptian Arabic only 23.2h**. That gap, not the reference clip or the
`language="egyptian arabic"` kwarg (both already in place), is why Egyptian lagged despite
already being tuned the same way Najdi was. Zero-shot cloning transfers speaker *timbre* from
the reference clip; the dialect's actual phonetics/prosody come from training data the model
never had much of for Egyptian.

**Fix**: **Habibi-TTS** (`SWivid/Habibi-TTS`, an F5-TTS-architecture model with a checkpoint
trained specifically and only on Egyptian speech, 37-103h — several times OmniVoice's 23h
slice) is now the PRIMARY engine for Egyptian-routed turns only. Najdi/Fusha/English/mixed
are completely untouched — they never import or call any Habibi-TTS code.

**Integration** (`tts_omnivoice_v1.py`): a lazy singleton (`_get_habibi_model()`, mirroring the
existing `_get_model()`/`_get_tashkeel_model()` pattern) loads the EGY-Specialized checkpoint
(`hf://SWivid/Habibi-TTS/Specialized/EGY/model_100000.safetensors`, ~1.35GB) + its Vocos
vocoder + a preprocessed reference clip at startup (soft-fail: any error → sticky `None`,
prints a warning, never crashes startup — same philosophy as the missing-Egyptian-clip
fallback). Reuses the SAME reference clip/transcript as OmniVoice's Egyptian voice
(`_EGY_REF_AUDIO`/`_EGY_REF_TEXT`) as a starting point — not yet re-auditioned by ear
specifically for Habibi, same "judge by ear" discipline that produced OmniVoice's v1→v4
Egyptian clip should be applied here too before treating this as final. `_synthesize_egyptian_habibi()`
calls Habibi's `infer_process()` (`nfe_step` tunable via `HABIBI_NFE_STEP`, default 16 — the
library default is 32; measured on this GPU both are comfortably faster than real-time, RTF
0.12-0.25). Three-tier fallback in `_synthesize_mp3_blocking`, each tier independently already
sensible: Habibi-TTS → OmniVoice Egyptian clip+language-id (today's pre-existing path) →
OmniVoice Saudi voice (pre-existing missing-clip fallback). Both engines' calls serialized
behind the SAME `_gen_lock` (neither is documented thread-safe). `HABIBI_TTS_ENABLED=0`
reverts to OmniVoice-only for Egyptian in one env var, no code change.

**Dependency install — real blockers hit and resolved, worth recording**:
- This venv is managed with `uv` (confirmed via `.venv/pyvenv.cfg`), not `pip` (`pip` is
  intentionally absent — `python -m pip` → "No module named pip", same as encountered earlier
  this project with the `gguf` package). `uv pip install --python .venv/bin/python ...` is the
  working equivalent.
- `habibi-tts`'s own `pyproject.toml` pins `torch>=2.4.0,<2.9.0` — installing normally would
  have tried to DOWNGRADE this box's `torch==2.11.0+cu130` (required for the RTX 5090/CUDA
  13.0) to satisfy that range. Installed `habibi-tts`/`f5-tts` with `--no-deps` instead;
  verified by direct import afterward that torch stayed at `2.11.0+cu130`. Five additional
  pure-Python runtime deps its inference path actually needs but `--no-deps` skips
  (`cached-path`, `rjieba`, `pypinyin`, `wandb`, `datasets` — found by iterating real
  `ImportError`s, not by guessing from the package's stated dependency list) installed the
  same safe way — see `requirements.txt`'s comment for the exact commands.
- This box's `torchaudio` build (matching the same CUDA 13 torch) has dropped its old
  loading backends in favor of requiring the separate `torchcodec` package, which itself
  needs system FFmpeg shared libraries not installed here (no apt/sudo). Habibi's
  `infer_process()` calls `torchaudio.load()` internally just to re-read the reference clip —
  shimmed via a `soundfile`-backed monkeypatch (`soundfile` already a proven dependency
  throughout this project) instead of adding a new system-library requirement for one call.
- A separate, unrelated CUDA issue (`nvrtc: error: failed to open libnvrtc-builtins.so.13.0`)
  during first real inference was an `LD_LIBRARY_PATH` gap in my own ad-hoc test invocations,
  not a real project bug — `start_server.sh` already sets this correctly for the live server
  (`$VENV/cuda_nvrtc/lib` is already in its `LD_LIBRARY_PATH` construction); only matters for
  anyone invoking Habibi-TTS-touching scripts directly outside `start_server.sh`.

**License note** (owner-acknowledged 2026-07-30, "proceed anyway"): the EGY-Specialized
checkpoint is stated Apache-2.0 in the GitHub README's prose ("the rest specialized models
(ALG, EGY, IRQ, MAR, MSA) are released under Apache 2.0 license"), but HuggingFace's own
machine-readable repo-level license tag reads `cc-by-nc-sa-4.0` for the whole repo, and there
is no separate `LICENSE` file scoped to `Specialized/EGY/` alone to settle the discrepancy
independently (confirmed: `.../Specialized/EGY/LICENSE` 404s). Accepted for this local
testing-ground use; revisit before any commercial/production deployment.

**Verification**: `eval/test_tts_args.py` (G4) rewritten to pin the new 3-tier contract via a
stubbed `_synthesize_egyptian_habibi` (same "stub the third-party model" pattern already used
for CATT) — confirms Habibi is called ONLY on `"egyptian arabic"` (never Najdi/Fusha/None),
confirms OmniVoice's `generate()` is NEVER called when Habibi succeeds, confirms the existing
OmniVoice-fallback and Saudi-fallback shapes are byte-identical to before. Full gate suite
green with ZERO golden-fixture changes (`llm.py`/`SYSTEM_PROMPT`/`MODEL_CONFIGS` untouched —
this is a TTS-module-only change, G1/G2/G3 never at risk). Live end-to-end smoke test (real
`llm.build_turn`/`ollama_chat_token_gen`/`stream_tts_to_ws`, not just the fake-model gate): an
Egyptian-phrased question correctly routed, loaded Habibi-TTS cleanly, produced 6 valid MP3s
with no fallback triggered; a Fusha `test_local.py` run immediately after confirmed the
OmniVoice/Saudi path is untouched (same 2-MP3 output shape as before this change). Not yet
done: an actual by-ear quality comparison of Habibi vs. OmniVoice on Egyptian (the owner's own
next step, same as every prior voice-quality judgment in this project).

## 2026-07-30 (later) — Owner's live test of Habibi-TTS: pronunciation quality unsatisfactory

Owner actually live-tested Habibi-TTS (real listening, not just the automated smoke test
above) and reported Egyptian pronunciation quality is NOT as good as OmniVoice's own
Najdi/Fusha bar. A friend suggested `NAMAA-Space/NAMAA-Egyptian-TTS` — researched rigorously
(fetched the actual model card, diffed checkpoint hashes against base Chatterbox, checked HF
discussions/GitHub issues) and **rejected**: undisclosed training data (unlike NAMAA's own
well-documented Saudi TTS model), authors' own card admits ق-mispronunciation + number-reading
bugs, near-zero independent usage (99 downloads/7 likes, empty discussions tab), architecturally
LESS dialect-aware (flat `language_id="ar"`, no Egyptian-specific code, unlike OmniVoice's own
`language=` kwarg or Habibi's dedicated EGY checkpoint), and a worse expected install
(`torch==2.6.0` hard pin vs. this box's `torch==2.11.0+cu130` — a 5-minor-version gap, vs.
Habibi's more forgiving `<2.9.0`).

Two follow-up research passes (wide survey, then an institutional/academic-focused pass —
QCRI, MBZUAI, Meta MMS, NVIDIA MagpieTTS, LDC CALLHOME Egyptian, Mozilla Common Voice, Egyptian
university labs) found nothing both rigorously-trained AND locally-deployable. Notable finding:
QCRI's Fanar 2.0 "Aura-TTS" claims genuine Egyptian zero-shot cloning from a credible,
well-resourced lab, but is API-only with zero downloadable weights — disqualified for this
local/self-hosted architecture. Everything else surveyed (NileTTS/synthetic ground truth,
`MAdel121`'s XTTS/F5-TTS/VibeVoice Egyptian fine-tunes, Chatterbox-based hobbyist fine-tunes,
`Qwen3-TTS-Egy`/no cloning, Masry/no released weights, Higgs Audio/no Egyptian signal) was
either disqualified outright or not worth an install cycle. Full detail in the session
transcript; headline conclusion both passes converged on: **the field is genuinely
under-resourced, not a case of an obvious better option being missed.**

## 2026-07-30 (later still) — Lahgtna-OmniVoice wired as an opt-in Egyptian A/B candidate

One candidate stood out from the survey specifically because it's a fine-tune of THIS
project's own OmniVoice base: `oddadmix/lahgtna-omnivoice-v2`. Verified hands-on (not just
model-card reading) before wiring it in:

- **Real evidence of an actual training run**: the HF repo has genuine HuggingFace Trainer
  checkpoint artifacts (`scheduler.bin`, `random_states_0.pkl`) and a `train_config.json`
  showing 20,000 steps of continued training from `k2-fsa/OmniVoice`'s own base checkpoint,
  via OmniVoice's own official fine-tuning recipe — more concrete technical evidence than
  most other candidates surveyed.
- **Training data is still undisclosed**: checked the model card, GitHub README,
  `train_config.json` (`hf_dataset_repo_id` is literally `null`), and the referenced
  data-config file (turned out to be OmniVoice's own generic, never-filled-in template, not
  Lahgtna's real manifest). Same category of "we don't actually know how well-trained this
  is" as everything else surveyed — a live A/B candidate to judge by ear, not a verified-better
  replacement.
- **License**: `k2-fsa/OmniVoice` and the Lahgtna GitHub fork are both confirmed Apache-2.0
  (checked directly via GitHub's API). The fine-tuned WEIGHTS repo on HuggingFace itself has
  no license tag declared at all — same category of gap as Habibi's messier situation, a
  different flavor of it.
- **Zero new install cost, confirmed empirically**: same `OmniVoice` class/loader already
  running in production — `OmniVoice.from_pretrained("oddadmix/lahgtna-omnivoice-v2", ...)`
  loaded and generated cleanly with no new pip package, no torch/CUDA compatibility risk, no
  torchaudio monkeypatch (unlike Habibi's whole install saga). This is what made it worth
  wiring in at all, unlike every other surveyed candidate.
- **Measured, non-subjective difference from the hands-on test**: raw output runs noticeably
  quieter than Habibi's/OmniVoice's own on identical text (peak ~0.24–0.62 vs ~0.47–0.89 on
  the same 6 comparison sentences) — not a quality defect, just a loudness mismatch.

**Integration** (`tts_omnivoice_v1.py`): a lazy singleton (`_get_lahgtna_model()`, same
soft-fail pattern as Habibi/CATT) loads the checkpoint and builds its voice-clone prompt from
the SAME Egyptian reference clip/transcript already in use (`_EGY_REF_AUDIO`/`_EGY_REF_TEXT`).
`_synthesize_egyptian_lahgtna()` peak-normalizes its output to `_LAHGTNA_TARGET_PEAK = 0.75`
to correct the measured loudness gap (verified: output peak lands exactly at 0.75 after
normalization) — without this, an A/B listening test would conflate "sounds different" with
"sounds quieter," and Egyptian replies would sound quieter than Najdi/Fusha ones if ever
adopted for real. `LAHGTNA_TTS_ENABLED=1` (env var, default OFF — this is the newer,
less-vetted candidate) makes it the PRIMARY Egyptian engine for that run, tried before Habibi;
falls back to Habibi → OmniVoice-Egyptian-clip → Saudi exactly as if disabled, on any load or
synthesis failure. `LAHGTNA_ENABLED` takes precedence over `HABIBI_ENABLED` when both would
otherwise apply — testing one candidate at a time avoids ambiguity about which engine actually
produced a given reply. A real bug was caught and fixed before it shipped: an early draft of
`_synthesize_egyptian_lahgtna` acquired `_gen_lock` internally, which would have deadlocked
since the caller (`_synthesize_mp3_blocking`) already holds that same non-reentrant lock for
the whole Egyptian branch.

**Verification**: `eval/test_tts_args.py` (G4) extended with the same stubbed-third-party-model
pattern used for Habibi/CATT — pins that Lahgtna is tried first when enabled, that its success
skips both Habibi AND `OmniVoice.generate()` entirely, that Habibi still works correctly when
Lahgtna is disabled/fails (the pre-existing contract, unbroken), and that neither new engine is
ever touched for Najdi/Fusha/English/mixed. Full gate suite green, zero golden-fixture changes
(`llm.py` untouched, as with the Habibi change). Live end-to-end smoke test with
`LAHGTNA_TTS_ENABLED=1`: both Habibi and Lahgtna loaded at startup (Habibi stays loaded as the
fallback), log confirmed "Lahgtna-OmniVoice ready — this is now the PRIMARY Egyptian engine,"
an Egyptian question produced 8 valid MP3s with no fallback triggered, and a direct call to
`_synthesize_egyptian_lahgtna()` confirmed the peak-normalization lands exactly on target.

**Not yet done**: the actual by-ear comparison against Habibi-TTS — the real point of this
work, and the owner's own next step, same discipline as every prior voice decision in this
project. No new pip packages/`requirements.txt` entries were needed for this change.

## 2026-08-07 — Fanar-1-9B-Instruct wired as a second LLM live-test candidate

Owner asked to test a smaller Fanar alongside Fanar-2-27B. Researched first (didn't assume
"smaller Fanar-2" exists): confirmed via QCRI's official HF collections and the Fanar/Fanar-2
papers that there is **no smaller Fanar-2 text model** — 27B is the only text LLM in the 2.0
generation (companion non-text models exist: Oryx image models, Diwan poetry model, not
relevant here). The real ~9B option is `QCRI/Fanar-1-9B-Instruct`, the **prior generation**:
built on `google/gemma-2-9b` (not Gemma-3 like Fanar-2), no documented Arabic-reasoning/
tool-use specialization the 27B's card specifically claims, identical (equally generic, no
Najdi mention either way) dialect-coverage wording to the 27B. The often-assumed "Fanar 7B"
("Fanar Star," trained from scratch) was never released as open weights — doesn't exist to
pull, don't chase it.

**Pulled and smoke-tested**: `hf.co/mradermacher/Fanar-1-9B-Instruct-i1-GGUF:i1-Q4_K_M`
(Apache-2.0, confirmed at every layer — HF API tags on both the QCRI weights and the GGUF
mirror, no discrepancy this time, unlike Habibi's Apache/CC-BY-NC mismatch). Loaded and
generated cleanly on the first raw `/api/chat` call — **no crash, no GGUF patch needed this
time**: checked the model's `tokenizer_config.json` directly before pulling, confirmed no
`{% raw %}`/tool-calling section (the exact construct that crashed Fanar-2's GGUF and needed
a binary patch) — absence confirmed structurally, then confirmed empirically with a real load.
Also checked for native thinking-mode leakage (a real risk with Fanar-2, needed `strip_think()`
handling in the eval harness) — a 400-token raw completion came back as plain, direct text
with zero `<think>` tags, `done_reason: "length"` (used its whole budget on the actual answer,
not reasoning) — no thinking-mode handling needed for this model either.

**One real, concrete constraint found and handled**: Gemma-2-9B's own `config.json` states
`max_position_embeddings=4096`/`sliding_window=4096` — smaller than this project's shared
`LLM_NUM_CTX` default of 8192. Rather than reuse the shared knob (the same class of mistake
the qwen3.5/Fanar-2 32768-vs-8192 OOM lesson already warns against, just a quality/behavior
risk here instead of VRAM), pinned `MODEL_CONFIGS["fanar-1"]`'s `num_ctx` explicitly to 4096,
the model's real native window.

**Integration** (`llm.py`): added `MODEL_CONFIGS["fanar-1"]`. This required **renaming the
existing bare `"fanar"` key to `"fanar-2"`** — `get_model_config()`'s substring match means
`"fanar-1-9b-instruct..."` would otherwise have also matched the old generic `"fanar"` key
first, silently giving the 9B model the 27B's config (including its unset/32768-native
`num_ctx`, exactly the OOM combination just described). `"fanar-1"` and `"fanar-2"` are
mutually exclusive substrings of their respective model tags — no collision either direction.
Same neutral temp/top_p/top_k/num_predict/stop as `"default"`/Fanar-2 (no basis yet to tune
differently — this is a fresh, unvalidated candidate). Reuses the exact same
`LLM_MODEL_OVERRIDE` env-var mechanism already in place for Fanar-2 — no new plumbing.

**Verification**: predicted and confirmed the exact same single-variable gate failure as
every prior `MODEL_CONFIGS` change — `G3 MODEL_CONFIGS: shared surface changed`, zero G1/G2
— confirming the key rename + addition changed nothing about prompt content or routing.
`golden_prompts.py --capture` run, `GOLDEN GATES G1/G2/G3 GREEN` reconfirmed. Full gate suite
(`test_routing.py`, `dialect_id_eval.py`, `test_tts_args.py`, `test_dialect_repair.py`,
`test_leak_lint.py`) green, untouched by this change.

**To test live**: `LLM_MODEL_OVERRIDE=hf.co/mradermacher/Fanar-1-9B-Instruct-i1-GGUF:i1-Q4_K_M
bash start_server.sh` — same pattern as Fanar-2, combinable with `LAHGTNA_TTS_ENABLED=1`/
`HABIBI_TTS_ENABLED` for a full independent-axis A/B (LLM choice × Egyptian TTS engine).
**Set expectations honestly going in**: this is an older, smaller, less-specialized model
than Fanar-2 with a meaningfully shorter context window — being 9B rather than 27B is very
likely to cost real capability, not just speed; the point of testing it is to see where that
tradeoff actually lands for this project's voice-assistant use case, not to find a free win.

## 2026-08-07 (later) — Fanar-1-9B-Instruct removed after live testing

Owner live-tested Fanar-1-9B-Instruct and found instruction-following too weak to be worth
pursuing further — the expectation set in the entry above ("likely to cost real capability,
not just speed") held. **Removed completely**: `MODEL_CONFIGS["fanar-1"]` deleted from
`llm.py` (the `"fanar-2"` key stands alone again, no longer needs the "-2" suffix for
disambiguation but keeping it anyway — no reason to introduce a second rename); `CLAUDE.md`'s
LLM row updated to drop it as an active option; the GGUF removed from Ollama
(`ollama rm hf.co/mradermacher/Fanar-1-9B-Instruct-i1-GGUF:i1-Q4_K_M`, freed ~5.4GB disk).
This BASELINES.md entry itself is kept (append-only history, per this file's own header) —
only the live-code traces are removed, not the record of why. Gate suite re-run and green
after the removal (see immediately below).

## 2026-08-10 — Egyptian tashkeel wired as an opt-in diacritizer candidate (GPL v2 isolated)

Owner asked again whether an Egyptian-specific diacritizer (mirroring CATT's Fusha role)
could be added, after the earlier research conclusion (revisit here, not re-litigated:
Egyptian colloquial text likely doesn't need full tashkeel the way MSA does — CATT exists to
resolve i'rab/case-ending ambiguity colloquial Egyptian mostly lacks). Owner wanted it wired
up and tested anyway despite that caveat — proceeded on that explicit instruction.

**Candidate**: CAMeL Tools (`camel-tools`, MIT) — specifically its BERT-based Egyptian
disambiguator, `BERTUnfactoredDisambiguator.pretrained(model_name='egy')`. A real, working,
downloadable model — verified hands-on, not just from docs: loads in ~1.4s, diacritizes a
sentence in 5-266ms (real-time-safe for live per-sentence synthesis), and correctly
vocalizes Egyptian-specific colloquial words (e.g. دِلْوَقْتِي "now", كُوَيِّس "good") that an
MSA-only model would likely mis-vocalize.

**Real license discrepancy found and resolved by testing, not by trusting docs**: the
readthedocs "Available Packages" page lists only an older MLE-based Egyptian package
(`disambig-mle-calima-egy-r13`, GPL v2) with no BERT-based entry at all — appears to be
stale/incomplete documentation. Running `camel_data -l` directly against the actual installed
package catalogue revealed the real, current list: `disambig-bert-unfactored-egy` (445.5MB,
**MIT**) is the real, modern Egyptian disambiguator. However, installing it pulls in
`morphology-db-egy-r13` (67.3MB) as a genuine runtime dependency of the BERT disambiguator's
own analysis pipeline, and that companion data package **is GPL v2** — confirmed directly via
`camel_data -l`'s own license column, not inferred. So the honest picture: the model itself is
MIT, but it cannot function without a GPL v2 data file at runtime.

**Owner decision on the GPL v2 dependency** (a different license CATEGORY — copyleft — from
every other dependency in this project so far, including Habibi/Lahgtna's merely-ambiguous
gaps): "Proceed, but isolate the GPL dependency clearly." Implemented as:
- `EGY_TASHKEEL_ENABLED` defaults to `"0"`. With it unset, `camel_tools`/`camel_kenlm` are
  pip-installed but their code is **never imported** — the import lives inside
  `_get_egy_tashkeel_model()`, reached only from `_add_egyptian_tashkeel()`, which itself
  early-returns before either is touched when the flag is off. The GPL-licensed
  `morphology-db-egy-r13` data is never downloaded unless a real diacritization call happens
  with the flag on.
- A dedicated, prominent module-level comment block in `tts_omnivoice_v1.py` (right above
  `EGY_TASHKEEL_ENABLED`) spells out the isolation boundary explicitly so a future edit
  doesn't accidentally move the import earlier or start the model eagerly without the guard.

**Integration** (`tts_omnivoice_v1.py`): `_get_egy_tashkeel_model()`/`_add_egyptian_tashkeel()`
mirror `_get_tashkeel_model()`/`_add_tashkeel()` (CATT) exactly — same lazy-singleton pattern,
same "fall back to plain text on ANY error" philosophy. Tokenizes via CAMeL's own
`simple_word_tokenize` (verified it splits punctuation as separate tokens correctly, e.g.
`؟`/`.`/`،`), diacritizes each token via `.disambiguate()`, reconstructs with natural spacing
(no space before closing punctuation — `_EGY_NO_SPACE_BEFORE`). Wired into
`_synthesize_mp3_blocking` as an `elif` sibling to CATT's existing `if` — mutually exclusive
by construction since `_TASHKEEL_LANGUAGES`/`_EGY_TASHKEEL_LANGUAGES` are disjoint sets. Same
Najdi-drift guard CATT already has (`not looks_najdi(text)`) applied here too, for the
analogous reason: an Egyptian-routed reply that drifts into Najdi-flavored text would likely
get mis-vocalized by an Egyptian-specific model the same way Najdi text mis-vocalizes under
CATT's MSA model. Eagerly warmed at startup when enabled (`load_models()`), same latency
rationale as every other lazy model in this file — still fully gated behind the same flag.

**Install** (see `requirements.txt` for the exact commands): precompiled wheels exist for
both `camel-tools` and its `camel-kenlm` dependency (cp312/manylinux) — no Rust/C++ compiler
toolchain needed, contrary to an earlier, apparently-outdated note from a lighter research
pass. Installed with `--no-deps` to protect `numpy==1.26.4` (camel-tools states
`numpy>=2.0.0`, but this was verified empirically NOT required at actual import/runtime —
same over-conservative-pin pattern already seen with Habibi-TTS's stated torch range); the
7 genuinely-missing pure-Python deps (`cachetools`, `docopt`, `emoji`, `future`, `muddler`,
`pyrsistent`, `tabulate`) installed normally with no numpy/torch impact, confirmed via
dry-run first.

**Verification**: `eval/test_tts_args.py` (G4) extended with the same stubbed-third-party-model
pattern already used for CATT/Habibi/Lahgtna — pins that Egyptian tashkeel is OFF by default,
fires only for `"egyptian arabic"` + non-Najdi-drift text when enabled, stays mutually
exclusive with CATT in both directions (Fusha never gets the Egyptian marker even with the
flag on; Egyptian never gets CATT's marker), and that enabling it doesn't touch Najdi/Fusha
at all. Full gate suite green, zero golden-fixture changes (`llm.py` untouched). Live
end-to-end smoke test (`EGY_TASHKEEL_ENABLED=1`, real `llm.build_turn`/`ollama_chat_token_gen`/
`stream_tts_to_ws`): loaded cleanly at startup, an Egyptian question produced 3 valid MP3s
with no errors, and a direct call to `_add_egyptian_tashkeel()` confirmed real diacritics
applied with correct punctuation spacing (`"الخزان دلوقتي شغال كويس والضغط طبيعي."` →
`"الخَزّان دِلْوَقْتِي شَغّال كُوَيِّس وَالضَّغْط طَبِيعِي."`).

**Not yet done, and the actual open question**: whether this measurably improves Egyptian TTS
output quality at all — the research conclusion going in was skeptical (Egyptian likely
doesn't have the i'rab-driven ambiguity tashkeel exists to resolve, and pronunciation quality
issues found so far have tracked back to engine/training-data gaps, not text preprocessing).
This is wired for the owner to judge by ear against Habibi/Lahgtna output with and without
the flag, same discipline as every other voice-quality decision in this project — not a
confirmed win.

## 2026-08-10 (later) — Egyptian tashkeel removed after live testing

Owner live-tested with `EGY_TASHKEEL_ENABLED=1` and found it "not working perfectly" — the
skeptical research conclusion going into this feature (Egyptian likely doesn't have the
i'rab-driven ambiguity CATT exists to resolve for MSA) held up against a real listen, same
outcome pattern as Fanar-1-9B's removal above. Owner asked for a complete removal, "even its
remains," given the extra weight this one carried: a GPL v2 runtime dependency, not just an
ambiguous license like Habibi/Lahgtna's.

**Removed completely**:
- `tts_omnivoice_v1.py`: the whole `EGY_TASHKEEL_ENABLED` block (constants, module-level GPL
  isolation comment, `_get_egy_tashkeel_model()`, `_add_egyptian_tashkeel()`), its
  `load_models()` eager-load branch, and the `elif` branch in `_synthesize_mp3_blocking` that
  called it. Verified the module still imports and the full pipeline still runs clean after
  (re-ran `test_local.py` end-to-end).
- `eval/test_tts_args.py` (G4): the stub, the two dedicated test sections, and the module-
  surface checks. Full suite re-run, still all green with no gaps.
- `requirements.txt`: the entire CAMeL Tools block (`camel-tools`, `camel-kenlm`, and its 7
  pure-Python deps).
- `CLAUDE.md`: the tech-stack row, the `EGY_TASHKEEL_ENABLED` env-knob mention, and contract
  point 10.
- **Actually uninstalled**, not just delisted: `uv pip uninstall` removed all 9 packages
  (`camel-tools`, `camel-kenlm`, `cachetools`, `docopt`, `emoji`, `future`, `muddler`,
  `pyrsistent`, `tabulate`) from the venv.
- **Deleted the downloaded model/GPL data**: `rm -rf ~/.camel_tools` — freed 490MB, and more
  importantly means the GPL v2-licensed `morphology-db-egy-r13` data no longer exists
  anywhere on this box, not just unused-but-present. This fully resolves the GPL v2 concern
  by removing the dependency outright rather than leaving it dormant.

Gate suite green after every step (`test_routing.py`, `dialect_id_eval.py`,
`test_tts_args.py`, `test_dialect_repair.py`, `test_leak_lint.py`, `golden_prompts.py` — the
latter needed no recapture at all, since this touched only `tts_omnivoice_v1.py`/its own
test, never `llm.py`). This entry and the one above it are kept (append-only history, per
this file's own header) — only the live-code and on-disk traces are gone, not the record of
what was tried and why it didn't work.

**Current state, for clarity**: Egyptian TTS is back to exactly the two-engine setup from the
2026-07-30 entries — Habibi-TTS (default) and Lahgtna-OmniVoice (opt-in A/B via
`LAHGTNA_TTS_ENABLED=1`) — with no diacritization step for Egyptian at all, same as before
this whole diacritizer detour began.

## 2026-08-11 — EGTTS-V0.1 wired in as a third Egyptian TTS candidate (opt-in, off by default)

Owner live-tested BOTH Lahgtna-OmniVoice and Habibi-TTS by ear and found neither's Egyptian
pronunciation quality satisfactory. Rather than remove Egyptian TTS entirely, owner asked for
a third candidate to be added to the same tiered fallback chain, tried FIRST (ahead of
Lahgtna): OmarSamir/EGTTS-V0.1, an XTTS-v2 fine-tune trained specifically on Egyptian Arabic.

**Architecture/API, confirmed by reading the model card + the actively-working community Space
(`MohamedRashad/Egyptian-Arabic-TTS`, built explicitly against this checkpoint) directly, not
assumed**: cloning is reference-AUDIO-ONLY — `model.get_conditioning_latents(audio_path=[...])`
has no ref-text parameter at all, unlike Habibi/OmniVoice, which both need a transcript.
Synthesis is `model.inference(text, "ar", gpt_cond_latent, speaker_embedding, temperature=...)`,
returning `out["wav"]`. Output confirmed 24000 Hz (matches this project's `SAMPLE_RATE`,
`config.json`'s own `output_sample_rate` field, and the Space's own return value) — no
resampling needed, same as Habibi/Lahgtna. `deepspeed` is confirmed NOT required: `use_deepspeed
=False` is the library default and is exactly what the model's own author-credited live Space
uses.

**Install blocker found and resolved**: the model card's literal instruction, `pip install
git+https://github.com/coqui-ai/TTS`, does not work on this box — that repo's own `setup.py`
hard-pins `python_requires=">=3.9.0, <3.12"`, and this venv is Python 3.12. Used the
actively-maintained fork's PyPI package instead: `coqui-tts` (`idiap/coqui-ai-TTS`, latest
0.27.5), which supports Python <3.15/>=3.10 and torch 2.2+ and carries the same MPL-2.0 code
license. This is an unofficial substitute for OmarSamir's checkpoint (not referenced anywhere
in EGTTS-V0.1's own docs) but was verified with a real, hands-on load + inference test before
trusting it — produced valid, playable 24kHz mono audio (peak 0.82–0.87, RMS 0.09–0.10, no
NaNs) from three test sentences using the project's existing Egyptian reference clip
(`voices/omnivoice-tts-egyptian-24k-v4.wav`).

**Three runtime workarounds needed** (installed/applied on top of the install-blocker fix
above, each confirmed necessary by hitting the actual failure, not anticipated from docs):
1. `transformers.pytorch_utils.isin_mps_friendly` — this project's pinned `transformers`
   build removed this function, but `TTS/tts/layers/tortoise/autoregressive.py` imports it
   BY NAME at module-import time (blocks even `import TTS`, not scoped to any one code path).
   Patched back with a plain `torch.isin()` call (CUDA-only, so faithful — the removed
   function only mattered for MPS/Apple Silicon).
2. `transformers.utils.import_utils.is_torchcodec_available` — `TTS/__init__.py` itself
   hard-raises `ImportError` on `torch>=2.9` unless `torchcodec` is importable (needs system
   FFmpeg, absent on this box, no apt/sudo). Forced to return `True`; the actual audio load
   goes through workaround 3, not torchcodec.
3. A soundfile-backed `torchaudio.load()` shim — identical class of fix to Habibi-TTS's
   existing monkeypatch, needed because `get_conditioning_latents()`'s `load_audio()` helper
   calls `torchaudio.load()` to read the reference clip.

**Packages actually installed** (`--no-deps` on `coqui-tts` itself, to protect
`torch==2.11.0+cu130`/`numpy==1.26.4`; 5 genuinely-missing pure-Python deps installed first,
found by iterating real `ImportError`s one at a time): `coqui-tts==0.27.5`,
`coqpit-config==0.2.5`, `coqui-tts-trainer==0.4.0`, `ko-speech-tools==0.1.0`,
`num2words==0.5.14`, `anyascii==0.3.3`. One install gotcha recorded for future readers: plain
`coqpit` (an outdated package name some READMEs still reference) is the WRONG package —
`coqui-tts` needs the forked `coqpit-config` instead; installing both into the same import
path corrupts it. `torch`/`numpy`/`torchaudio`/`transformers` all verified unchanged after
every install step. `deepspeed` was never installed — confirmed unneeded (see above).

**License — materially more restrictive than either existing Egyptian candidate**: Coqui
Public Model License 1.0.0 (CPML), confirmed via the HF API's structured `license` field
("other" / license_name "coqui-public-model-license") AND the repo's actual `LICENSE.txt`
prose. Unlike Habibi's Apache-2.0-vs-cc-by-nc-sa-4.0 ambiguity or Lahgtna's missing license
tag, this one is unambiguous: EXPLICITLY non-commercial ("non-commercial purpose" = any use
that does not generate direct or indirect payment), with an attribution-propagation clause and
a patent-defense termination clause. Accepted for this local testing-ground use only, same
"proceed to evaluate, revisit before production" latitude as Habibi/Lahgtna — but this is the
first of the three Egyptian candidates where the license question has no ambiguity to resolve
in the owner's favor later; a genuinely non-commercial grant, full stop.

**Integration** (`tts_omnivoice_v1.py`): `EGTTS_ENABLED` (env `EGTTS_V01_ENABLED`, default
`"0"`) → `_get_egtts_model()`/`_synthesize_egyptian_egtts()` mirror
`_get_lahgtna_model()`/`_synthesize_egyptian_lahgtna()`'s lazy-singleton-plus-sticky-failure
shape exactly. Wired into `_synthesize_mp3_blocking`'s Egyptian branch as the FIRST tier tried
(ahead of Lahgtna): EGTTS → Lahgtna → Habibi → OmniVoice's Egyptian clip+language-id → Saudi
voice. `Xtts.inference()` is NOT called under its own lock — it runs inside the `_gen_lock`
`with` block `_synthesize_mp3_blocking` already holds for the whole Egyptian branch (a plain,
non-reentrant `threading.Lock`; acquiring it again inside `_synthesize_egyptian_egtts` would
deadlock — same no-internal-lock discipline already documented in
`_synthesize_egyptian_lahgtna`/`_synthesize_egyptian_habibi`). Eagerly warmed at startup in
`load_models()`, gated behind the same flag, soft-fail (never blocks other startup). The whole
block is wrapped in a distinctive comment banner and every symbol is prefixed
`_egtts`/`EGTTS_ENABLED`, by explicit owner request, so it can be found and removed in one step
later without touching OmniVoice/Habibi/Lahgtna/CATT.

**Verification**: `eval/test_tts_args.py` (G4) extended with the same stubbed-third-party-model
pattern already used for Habibi/Lahgtna — pins that EGTTS is never called for Najdi/Fusha/
English/mixed, IS called (unconditionally, at the call site) for Egyptian regardless of
whether it succeeds, and that when it succeeds neither Lahgtna nor Habibi-TTS nor
`OmniVoice.generate()` is touched. All the pre-existing fallback-chain pins (Lahgtna succeeds,
Habibi succeeds, clip-missing fallback) still hold unchanged, just extended with the new
EGTTS-first tier. Full gate suite green; `golden_prompts.py` needed ZERO recapture (`llm.py`
untouched by this change, as expected — this integration only touches
`tts_omnivoice_v1.py`/`eval/test_tts_args.py`/`requirements.txt`/`CLAUDE.md`).

**Removal path**: `scripts/remove_egtts.sh` uninstalls exactly the 6 pip packages above and
deletes exactly the downloaded checkpoint directory
(`~/.cache/huggingface/hub/models--OmarSamir--EGTTS-V0.1`, ~5.3GB) — it does not touch
OmniVoice/Habibi/Lahgtna/CATT. `EGTTS_V01_ENABLED=0` (the default) already makes the feature
fully inert even before that script is run or the code is removed.

**Not yet done, and the actual open question**: whether EGTTS-V0.1's Egyptian pronunciation
quality is any better than Lahgtna's or Habibi's to the owner's ear — this entry covers only
that the integration is installed, wired correctly, and passes the same non-regression gates
as every other engine in this chain. Judge by ear against the other two, same discipline as
every other voice-quality decision in this project — not a confirmed win.

## 2026-08-12 — EGTTS-V0.1 removed after live testing

Owner live-tested EGTTS-V0.1 and found the Egyptian pronunciation quality "very very very
bad" — the open question left at the end of the entry above is now settled by ear, same
outcome pattern as Fanar-1-9B's and the Egyptian-tashkeel diacritizer's removals above (a
live-tested candidate that didn't hold up, removed completely rather than left dormant).
Owner asked for a full removal, "it and its remains as well."

**Removed completely**:
- `tts_omnivoice_v1.py`: the whole `EGTTS_ENABLED` constants/singleton-state block, its
  `load_models()` eager-warm branch, and the `_get_egtts_model()`/`_synthesize_egyptian_egtts()`
  function pair. The tier call in `_synthesize_mp3_blocking`'s Egyptian branch is gone too —
  the chain is back to Lahgtna → Habibi-TTS → OmniVoice's Egyptian clip+language-id → Saudi
  voice (4 tiers, down from 5), and its docstring was reworded to match. Verified no `EGTTS`/
  `egtts` string remains anywhere in the file.
- `eval/test_tts_args.py` (G4): the `_fake_egtts` stub, the `_egtts_calls`/`_egtts_result`
  tracking state, the `expect_egtts_call` parameter (dropped from every `synth()` call site,
  not just left unused), the dedicated "EGTTS succeeds" test section, and the `_egtts_lock`/
  `EGTTS_ENABLED` module-surface checks. Full suite re-run, still all green with no gaps.
- `requirements.txt`: the entire EGTTS-V0.1 block (`coqui-tts`, `coqpit-config`,
  `coqui-tts-trainer`, `ko-speech-tools`, `num2words`, `anyascii`, and the explanatory comment
  block).
- `CLAUDE.md`: the pipeline diagram (back to a 4-tier Egyptian chain), the tech-stack row, the
  `EGTTS_V01_ENABLED` env-knob mention, and contract point 9's wording — reverted to describe
  Lahgtna → Habibi → OmniVoice-clip → Saudi, with a one-line historical pointer back to this
  entry for anyone wondering what happened to the third candidate.
- **Actually uninstalled**, not just delisted: `bash scripts/remove_egtts.sh` uninstalled all
  6 pip packages (`anyascii`, `coqpit-config`, `coqui-tts`, `coqui-tts-trainer`,
  `ko-speech-tools`, `num2words`) from the venv.
- **Deleted the downloaded checkpoint**: the same script removed
  `~/.cache/huggingface/hub/models--OmarSamir--EGTTS-V0.1/` — freed ~5.3GB, and the CPML-
  licensed checkpoint no longer exists anywhere on this box, not just unused-but-present.
- `scripts/remove_egtts.sh` itself was then deleted (and the now-empty `scripts/` directory
  with it) — its one job was done, and keeping a removal script for a feature no longer in
  the codebase would just be confusing dead weight. Unlike the CAMeL Tools/Fanar-1-9B removals
  above, this candidate had a dedicated script rather than ad-hoc commands; the script is gone,
  but the exact commands it ran are preserved in this entry and the one above it.

Gate suite green after every step (`test_routing.py`, `dialect_id_eval.py`,
`test_tts_args.py`, `test_dialect_repair.py`, `test_leak_lint.py`, `golden_prompts.py` — the
latter needed no recapture at all, since this touched only `tts_omnivoice_v1.py`/its own test/
`requirements.txt`/`CLAUDE.md`, never `llm.py`). This entry and the one above it are kept
(append-only history, per this file's own header) — only the live-code, test, doc, package,
and downloaded-data traces are gone, not the record of what was tried and why it didn't work.

**Current state, for clarity**: Egyptian TTS is back to exactly the two-engine setup from the
2026-07-30 entries — Habibi-TTS (default) and Lahgtna-OmniVoice (opt-in A/B via
`LAHGTNA_TTS_ENABLED=1`) — the same state the pipeline was in before EGTTS-V0.1 was ever
tried. Three Egyptian TTS candidates have now been live-tested end-to-end by the owner
(Habibi, Lahgtna, EGTTS-V0.1) and none has been judged fully satisfactory; Habibi remains the
shipped default only because it was the original replacement for OmniVoice's own weak
Egyptian training data, not because its quality was ever confirmed as good.

## 2026-08-13 — VoiceTut-TTS wired in as a fourth Egyptian TTS candidate (opt-in, off by default)

Owner asked about `mohammedaly22/VoiceTut-TTS` after finding it independently. Researched
first (model card, HF's structured API metadata, the GitHub repo, `config.json` directly),
THEN verified every non-obvious claim hands-on before wiring anything in — same discipline as
every other candidate this project has evaluated.

**What it actually is (confirmed, not just model-card prose)**: `config.json`'s
`model_type` is literally `"omnivoice"`, and HF's own structured tags carry
`base_model:k2-fsa/OmniVoice` + `base_model:finetune:k2-fsa/OmniVoice` — a genuine fine-tune
of the same OmniVoice architecture already in production for Najdi/Fusha/Lahgtna, not a
different stack like Habibi/F5-TTS or the now-removed EGTTS/XTTS. Text backbone: Qwen3-0.6B.
24kHz output — matches this pipeline's existing sample rate exactly.

**Best license/training-data profile of any Egyptian candidate tried here so far**:
- License: **Apache-2.0, confirmed in BOTH the model card prose AND HF's structured
  `license` API field** — no discrepancy. Habibi has a stated-Apache-vs-tagged-cc-by-nc-
  sa-4.0 mismatch; Lahgtna's repo carries no license tag at all; EGTTS was explicitly
  non-commercial (CPML). This is the first of the four with no ambiguity to resolve.
- Training data: **~380h of disclosed, dialect-tagged (`arz`) Egyptian YouTube podcast
  audio** — every other Egyptian candidate here had undisclosed or thin training data
  (Habibi/Lahgtna both undisclosed; OmniVoice's own base is only ~23h Egyptian).

**Hands-on verification, in order** (the actual reason this was worth wiring in, not just
the license/data story):
1. **Loads via the ALREADY-INSTALLED `omnivoice==0.1.5` PyPI package** — confirmed by direct
   test: `OmniVoice.from_pretrained("mohammedaly22/VoiceTut-TTS", ...)` succeeds, zero new pip
   packages (`pip list` diff confirmed empty). The model's own GitHub repo instructions claim
   OmniVoice must be installed from source (`pip install git+https://github.com/k2-fsa/
   OmniVoice.git`), not PyPI — that claim did NOT hold up under real testing; the checkpoint's
   architecture class is generic enough to load through the pinned PyPI version already
   serving the rest of the pipeline. This was the single biggest open risk going in (a
   version conflict with the omnivoice package Najdi/Fusha/Lahgtna all depend on) and it
   resolved cleanly.
2. **No monkeypatches, no torchaudio shims, no dependency conflicts** — confirmed
   `torch`/`numpy`/`torchaudio`/`transformers` all unchanged after load. The cleanest install
   of any Egyptian candidate tried in this project (EGTTS alone needed three runtime patches).
3. **Resolved a real model-card-vs-GitHub-example contradiction**: the HF card claims
   zero-shot cloning needs no reference transcript; the GitHub usage sample passes `ref_text`
   anyway. Inspecting `create_voice_clone_prompt`'s actual signature directly settled it:
   `ref_text: Optional[str] = None` — genuinely optional. Passed it anyway (same as Habibi/
   Lahgtna) since this project already has a transcript for its Egyptian reference clip and
   it's the likely-better-quality path.
4. Synthesized 6 free-form Egyptian sentences + 6 category-matched clips (short_greeting/
   short_status/medium_question/medium_informational/numbers_heavy/long_explanatory — same
   taxonomy as the earlier Habibi re-audition set) using this project's own Egyptian
   reference clip, same clone-prompt path as Lahgtna. All objectively sane: no NaNs, peak
   0.33–0.60, RMS 0.045–0.075, durations scaling correctly with sentence length. Files
   generated in `test_output_voicetut/` for the owner's own listen.
5. One honest caveat found and reported, not hidden: the downloaded checkpoint is **6.9GB**
   on disk — larger than its card-stated 0.6B-parameter backbone would suggest. Not a
   blocker (3.1TB free on this box), just a claim-vs-reality gap worth noting for the record
   (cleaned up ~5GB of incomplete-download debris from an unrelated duplicate-process mistake
   during testing before taking this measurement).

**Integration** (`tts_omnivoice_v1.py`): `VOICETUT_ENABLED` (env `VOICETUT_TTS_ENABLED`,
default `"0"`) → `_get_voicetut_model()`/`_synthesize_egyptian_voicetut()` mirror
`_get_lahgtna_model()`/`_synthesize_egyptian_lahgtna()`'s lazy-singleton-plus-sticky-failure
shape exactly, including the same peak-normalization treatment (`_VOICETUT_TARGET_PEAK =
0.75` — its raw output runs on the quiet side too, same class of loudness difference already
seen with Lahgtna). Wired into `_synthesize_mp3_blocking`'s Egyptian branch as the FIRST tier
tried when enabled: VoiceTut → Lahgtna → Habibi → OmniVoice's Egyptian clip+language-id →
Saudi voice (5 tiers total). `generate()` is NOT called under its own lock — same
no-internal-lock discipline as every other engine in this chain (`_gen_lock` is a plain,
non-reentrant `threading.Lock` already held by the caller for the whole Egyptian branch).
Eagerly warmed at startup in `load_models()`, gated behind the same flag, soft-fail.

Named "opt-in A/B candidate #3" rather than reusing EGTTS's now-freed "#2" slot — matches
this project's own no-renumbering precedent (see the `fanar-2` naming note in `llm.py`: a
freed numeric suffix is left alone rather than reused, to avoid ambiguity against historical
entries that already reference the old numbering).

**Verification**: `eval/test_tts_args.py` (G4) extended with the same stubbed-third-party-
model pattern already used for Habibi/Lahgtna — pins that VoiceTut is never called for Najdi/
Fusha/English/mixed, IS called (unconditionally, at the call site) for Egyptian regardless of
whether it succeeds, and that when it succeeds neither Lahgtna nor Habibi-TTS nor
`OmniVoice.generate()` is touched. All pre-existing fallback-chain pins still hold unchanged,
just extended with the new VoiceTut-first tier. Full gate suite green; `golden_prompts.py`
needed ZERO recapture (`llm.py` untouched — this integration only touches
`tts_omnivoice_v1.py`/`eval/test_tts_args.py`/`requirements.txt`/`CLAUDE.md`).

**Not yet done, and the actual open question**: whether VoiceTut-TTS's Egyptian pronunciation
quality is any better than Lahgtna's/Habibi's/EGTTS's to the owner's ear — this entry covers
only that the integration is installed, wired correctly, and passes the same non-regression
gates as every other engine in this chain. Best license/data story so far is not the same
claim as best-sounding; judge by ear, same discipline as every other voice decision here.

## 2026-08-13 (later) — VoiceTut-TTS promoted to default; Habibi-TTS AND Lahgtna-OmniVoice fully removed

Owner live-tested VoiceTut-TTS (via `VOICETUT_TTS_ENABLED=1` alongside the Fanar-2 LLM
override) and reported: **"VoiceTut-TTS is the best TTS model for Egyptian so far. so if we
have any other for the egyptian dailect like Habibi TTS completely remove it even its
reamins as well."** Asked explicitly whether that also covered Lahgtna-OmniVoice (the other
still-present Egyptian candidate) — owner confirmed: remove Lahgtna too. Net effect: VoiceTut-
TTS is promoted from "opt-in A/B candidate #3" to THE Egyptian engine, and both remaining
prior candidates are fully removed, leaving only VoiceTut-TTS + base OmniVoice's Egyptian
clip (the pre-2026-07-30 fallback) in the Egyptian chain — three engines down to one.

**Promotion** (`tts_omnivoice_v1.py`): `VOICETUT_ENABLED` default flipped from
`os.environ.get("VOICETUT_TTS_ENABLED", "0")` to `..., "1")` — same on-by-default shape
Habibi previously had via `HABIBI_TTS_ENABLED`. `VOICETUT_TTS_ENABLED=0` still reverts
Egyptian-routed turns to OmniVoice's own Egyptian clone prompt in one env var, no code change
needed — same reversibility guarantee every engine in this chain has always had.

**Removed completely** (mirroring the exact discipline used for every prior removal in this
project — Fanar-1-9B, the CAMeL-Tools Egyptian diacritizer, EGTTS-V0.1):
- `tts_omnivoice_v1.py`: the entire Habibi-TTS block (constants, `_torchaudio_load_patched`
  state, `_get_habibi_model()`, `_synthesize_egyptian_habibi()`) and the entire
  Lahgtna-OmniVoice block (constants, `_get_lahgtna_model()`, `_synthesize_egyptian_lahgtna()`).
  `load_models()`'s eager-warm branches for both removed. The Egyptian tier chain in
  `_synthesize_mp3_blocking` collapsed from 5 tiers to 3: VoiceTut-TTS (default) →
  OmniVoice's Egyptian clip+language-id → Saudi voice. Module header docstring and every
  function docstring that referenced Habibi/Lahgtna as still-existing engines (VoiceTut's own
  loader/synth docstrings, the no-internal-lock comment) reworded. Verified no `habibi`/
  `Habibi`/`HABIBI`/`lahgtna`/`Lahgtna`/`LAHGTNA` string remains anywhere in the file except
  the intentional historical-context paragraph at the top explaining what replaced what.
- `eval/test_tts_args.py` (G4): full rewrite — dropped the `_fake_habibi`/`_fake_lahgtna`
  stubs, `_habibi_calls`/`_lahgtna_calls` tracking, `expect_habibi_call`/`expect_lahgtna_call`
  params (removed from every `synth()` call site, not left unused), and the dedicated Habibi-
  succeeds/Lahgtna-succeeds test sections. VoiceTut is now tested as the always-active
  default (no `expect_*_call=False` cases for it on non-Egyptian text needed changing —
  those assertions already existed and still hold). Added explicit module-surface checks
  that `_habibi_lock`/`_lahgtna_lock` do NOT exist anymore (`hasattr(...) == False`), so a
  future accidental reintroduction would fail this gate. Full suite re-run, all green.
- `requirements.txt`: Habibi's entire block (`habibi-tts`, `f5-tts`, `cached-path`, `rjieba`,
  `pypinyin`, `wandb`, `datasets`, plus the explanatory comment) and Lahgtna's comment block
  (Lahgtna itself never added any packages) both removed. VoiceTut's own comment updated to
  describe it as the default, not an opt-in candidate, and to record what it replaced.
- `CLAUDE.md`: pipeline diagram (back to 2 Egyptian tiers: VoiceTut-TTS default →
  OmniVoice+Egyptian-clip → OmniVoice+Saudi-clip), tech-stack table (Habibi's and Lahgtna's
  rows removed; VoiceTut's row rewritten as the default with a summary of what it replaced),
  env-knobs line (`HABIBI_TTS_ENABLED`/`HABIBI_NFE_STEP`/`LAHGTNA_TTS_ENABLED` all removed,
  `VOICETUT_TTS_ENABLED` now documented as default 1), contract point 9 rewritten for the
  3-tier chain with a condensed **History** paragraph pointing back to this file for the full
  Habibi/Lahgtna/EGTTS story, and the project-files tree's one-line description of
  `tts_omnivoice_v1.py` updated.
- **Actually uninstalled**, not just delisted: `uv pip uninstall` removed all 7 Habibi/f5-tts
  packages (`habibi-tts`, `f5-tts`, `cached-path`, `rjieba`, `pypinyin`, `wandb`, `datasets`)
  from the venv — confirmed via `pip show` returning nothing for each afterward. Verified
  first that none of the 7 were imported anywhere else in the codebase (only
  `tts_omnivoice_v1.py`'s own now-deleted Habibi code imported `cached_path`/`f5_tts`) and
  that `omnivoice`/`catt_tashkeel`/`clearvoice`/`faster_whisper` all still import cleanly
  after the uninstall. Lahgtna never needed its own packages (it reused the already-installed
  `omnivoice` package), so there was nothing to uninstall for it beyond the checkpoint below.
- **Deleted the downloaded checkpoints**: `~/.cache/huggingface/hub/models--SWivid--Habibi-TTS`
  (1.3GB), `~/.cache/huggingface/hub/models--oddadmix--lahgtna-omnivoice-v2` (2.3GB), and
  `~/.cache/huggingface/hub/models--charactr--vocos-mel-24khz` (52MB — F5-TTS's own vocoder,
  downloaded by Habibi's `load_vocoder()` call, not exclusive to the Habibi repo itself but
  orphaned now that f5-tts is uninstalled and nothing else in the project uses it) — freed
  ~3.65GB total.
- **Deleted the Habibi/Lahgtna-only test-output artifact directories**: `test_output_egy/`,
  `test_output_egy_reaudition/` (both from the 2026-07-30 Habibi re-audition work), and
  `test_output_lahgtna/` — untracked generated audio with no ongoing purpose once both
  engines were gone. `test_output_voicetut/` and the unrelated `test_output/` were left alone
  (not attributable to either removed engine).

Gate suite green after every step (`test_routing.py`, `dialect_id_eval.py`,
`test_tts_args.py`, `test_dialect_repair.py`, `test_leak_lint.py`, `golden_prompts.py` — the
latter needed no recapture at all, since this touched only `tts_omnivoice_v1.py`/its own
test/`requirements.txt`/`CLAUDE.md`, never `llm.py`). This entry and every entry above it are
kept (append-only history, per this file's own header) — only the live-code, test, doc,
package, downloaded-data, and test-output traces are gone, not the record of what was tried,
in what order, and why each one didn't stick.

**Current state, for clarity**: Egyptian TTS is now VoiceTut-TTS (default, promoted after
beating all three prior candidates by ear) with OmniVoice's own Egyptian clip+language-id as
the sole remaining fallback — the simplest the Egyptian TTS chain has been since Habibi-TTS
first replaced plain OmniVoice on 2026-07-30. Four Egyptian candidates have now been tried in
total (Habibi, Lahgtna, EGTTS-V0.1, VoiceTut-TTS); three were tried, judged, and fully
removed; VoiceTut-TTS is the first to actually win a live A/B in this project's history.

## 2026-08-13 (later still) — CAMeL Tools Egyptian tashkeel re-added for a second live test

Owner asked "what is the bestest diacritizer for Egyptian dialect" — flagged that this exact
question was already answered 3 days ago (CAMeL Tools tried and removed 2026-08-10, see both
dated entries above) before doing fresh research. Nothing new had emerged in the field
(CAMeL Tools' `disambig-bert-unfactored-egy` is still the only dedicated Egyptian-dialect
diacritizer that exists), but the research turned up independent corroboration of the
original skeptical conclusion: Lahgtna-OmniVoice's own v1→v3 model card documents that
diacritized Egyptian text (v1/v2) "loses coherence and babbles," and v3 deliberately switched
to raw, non-diacritized text "to fix pronunciation problems caused by MSA diacritics on
Egyptian colloquial." Recommendation given: don't re-try it, no diacritizer is likely correct
for Egyptian. Owner's decision: "re-trying CAMeL Tool I want to test it one more time" —
explicit, informed choice to test anyway, not re-litigated further.

**Re-added, mirroring the first integration exactly** (no original code survived to recover
from — the 2026-08-10 add-then-remove both happened as uncommitted working-tree edits in the
same session, confirmed via `git log`; reconstructed from this file's own detailed 2026-08-10
entry rather than any git history):
- **Packages reinstalled**: the 7 pure-Python deps first (`cachetools==7.1.7`,
  `docopt==0.6.2`, `emoji==2.15.0`, `future==1.0.0`, `muddler==0.1.3`, `pyrsistent==0.20.0`,
  `tabulate==0.10.0`), then `camel-tools==1.6.0`/`camel-kenlm==2026.2.7` with `--no-deps` —
  re-verified numpy==1.26.4/torch==2.11.0+cu130 untouched after each step, same as the first
  time. Package versions differ slightly from 3 days ago (e.g. `camel-kenlm` is now a
  date-versioned `2026.2.7` release) — expected, not a concern, verified working regardless.
- **Model data re-downloaded**: `camel_data -i disambig-bert-unfactored-egy` — confirmed via
  `camel_data -l` that the license split is unchanged (`disambig-bert-unfactored-egy` MIT,
  `morphology-db-egy-r13` GPL v2 pulled in as its runtime dependency) — same GPL v2 caveat as
  before, isolated the same way.
- **`tts_omnivoice_v1.py`**: `EGY_TASHKEEL_ENABLED` block, `_get_egy_tashkeel_model()`,
  `_add_egyptian_tashkeel()` reconstructed to mirror CATT's own `_get_tashkeel_model()`/
  `_add_tashkeel()` shape exactly (same lazy-singleton pattern, same "fall back to plain text
  on ANY error" philosophy, same Najdi-drift guard). Wired into `_synthesize_mp3_blocking` as
  an `elif` sibling to CATT's `if` (mutually exclusive via disjoint `_TASHKEEL_LANGUAGES`/
  `_EGY_TASHKEEL_LANGUAGES` sets). Eagerly warmed at startup in `load_models()` when enabled,
  same soft-fail discipline as every other model in this file.
- **`eval/test_tts_args.py`** (G4): re-added the stub (`_add_egyptian_tashkeel` → a visible
  `"EGY_TASHKEEL::"` marker, same technique already used for CATT), plus tests pinning: off
  by default (captured BEFORE the stub override, since it defaults OFF unlike CATT); fires
  only for `"egyptian arabic"` + non-Najdi-drift text when enabled; mutually exclusive with
  CATT in both directions (Fusha never gets the Egyptian marker even with the flag on;
  Egyptian never gets CATT's marker); Najdi-drift text on an Egyptian-routed turn skips it
  too, mirroring CATT's own drift-guard test. Full suite green.
- **`requirements.txt`**/**`CLAUDE.md`**: both re-added — tech-stack row, env-knob mention,
  contract-point addendum, pipeline diagram — same content shape as the 2026-08-10 entry,
  updated with the current package versions and this file's own two prior entries as history.

**Verification, hands-on, not just re-running the old test**: direct call to
`_add_egyptian_tashkeel()` reproduced the EXACT same output as the first integration's own
verification — `"الخزان دلوقتي شغال كويس والضغط طبيعي."` → `"الخَزّان دِلْوَقْتِي شَغّال
كُوَيِّس وَالضَّغْط طَبِيعِي."` — confirming the model/package resurrection is faithful, not
a subtly different reconstruction. A full live `_synthesize_mp3_blocking()` call (Egyptian-
routed, `EGY_TASHKEEL_ENABLED=1`, VoiceTut-TTS default-on) produced a valid 17,664-byte MP3
with the diacritized text flowing through to VoiceTut-TTS correctly. Full gate suite green
(`test_routing.py`, `dialect_id_eval.py`, `test_tts_args.py`, `test_dialect_repair.py`,
`test_leak_lint.py`, `golden_prompts.py` — zero recapture needed, `llm.py` untouched).

**Not yet done, and the actual open question — unchanged from the first attempt**: whether
this measurably improves Egyptian TTS output quality at all, now specifically against
VoiceTut-TTS (the engine in place this time, not Habibi/Lahgtna as before). The research
argument against it is now stronger than the first attempt (Lahgtna's own model-card
corroboration, found this round), but the owner's call to test again stands — wired for a
second live judge-by-ear, same discipline as every other voice decision in this project, not
a confirmed win either way yet.

## 2026-08-13 (later still) — CAMeL Tools Egyptian tashkeel removed a second time

Owner live-tested the re-added diacritizer against VoiceTut-TTS and reported: "again the
dicretization is not working good for egyptian dialect. its completely bad. remove it
completely and it remains as well." Second live test, second rejection — the outcome the
research argument predicted going in both times (Egyptian colloquial mostly lacks the i'rab/
case-ending ambiguity CATT resolves for Fusha) now holds against TWO different Egyptian TTS
engines (Habibi/Lahgtna the first time, VoiceTut-TTS this time), which is stronger evidence
than either single test alone. Combined with Lahgtna's own independent model-card
corroboration (found during this round's research, see the entry above), there are now three
separate data points all pointing the same way. Removed completely again, same discipline as
every other removal in this project.

**Removed completely** (mirrors the 2026-08-10 removal exactly, second time through the same
motions):
- `tts_omnivoice_v1.py`: the whole `EGY_TASHKEEL_ENABLED` block (constants, GPL isolation
  comment, `_egy_tashkeel_model`/`_egy_tashkeel_lock` state), `_get_egy_tashkeel_model()`,
  `_add_egyptian_tashkeel()`, the `load_models()` eager-warm branch, and the `elif` branch
  in `_synthesize_mp3_blocking` — reverted its docstring back to CATT-only wording. Verified
  no `camel_tools`/`egy_tashkeel`/`EGY_TASHKEEL` string remains anywhere in the file.
- `eval/test_tts_args.py` (G4): the stub, the default-capture variable, the mutual-exclusivity
  and Najdi-drift test cases, and the module-surface checks — replaced with a single
  negative-assertion check (`hasattr(tts, "_egy_tashkeel_lock") == False`) so a future
  accidental reintroduction would fail this gate, same pattern already used for Habibi/
  Lahgtna's own removal. Full suite re-run, all green.
- `requirements.txt`: the entire CAMeL Tools block (`camel-tools`, `camel-kenlm`, and its 7
  pure-Python deps) removed again.
- `CLAUDE.md`: the tech-stack row, the `EGY_TASHKEEL_ENABLED` env-knob mention, the pipeline
  diagram's diacritizer stage, and contract point 9's addendum — the addendum was rewritten
  (not just deleted) to record "tried twice, removed twice" as the actual historical outcome,
  pointing back to this file for detail, rather than silently reverting to as if it never
  happened a second time.
- **Actually uninstalled**, not just delisted: `uv pip uninstall` removed all 9 packages
  (`camel-tools`, `camel-kenlm`, `cachetools`, `docopt`, `emoji`, `future`, `muddler`,
  `pyrsistent`, `tabulate`) from the venv — confirmed via re-verifying `numpy==1.26.4`/
  `torch==2.11.0+cu130`/`omnivoice`/`catt_tashkeel` all still import cleanly afterward.
- **Deleted the downloaded model/GPL data**: `rm -rf ~/.camel_tools` — freed 490MB, and the
  GPL v2-licensed `morphology-db-egy-r13` data no longer exists anywhere on this box again.

Gate suite green after every step (`test_routing.py`, `dialect_id_eval.py`,
`test_tts_args.py`, `test_dialect_repair.py`, `test_leak_lint.py`, `golden_prompts.py` — zero
recapture needed, `llm.py` untouched). Every entry above this one is kept (append-only
history, per this file's own header) — only the live-code, test, doc, package, and
downloaded-data traces are gone, not the record of what was tried, twice, and why it didn't
work either time.

**Current state, for clarity**: no diacritization step for Egyptian at all — Egyptian TTS is
VoiceTut-TTS (default) with OmniVoice's own Egyptian clip+language-id as the sole fallback,
identical to the state right after VoiceTut-TTS was promoted, before this whole second
diacritizer detour began. **Recommendation for any future request to try an Egyptian
diacritizer a third time**: the evidence against it is now genuinely strong — two independent
live tests against two different TTS engines, plus one independent third-party team's own
model-development history, all reaching the same conclusion. A third attempt would need a
materially different candidate or a materially different argument to be worth the integration
cost again, not just a retry of the same idea.
