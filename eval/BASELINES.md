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

## 2026-07-27 — Offline A/B: Fanar-2-27B-Instruct vs. qwen3.5:27b (evaluation-only, not a swap)

A 2026-07-10 finding (this file's earlier entries) plus this session's own reactive fixes both
pointed at the same ceiling: qwen3.5's Najdi output has a genuine **verb-morphology** leak —
the بـ+imperfective prefix (بيوصلوا، بيحفظ، بيخونك — Levantine/Egyptian grammar, not Najdi) and
a هـ-future variant (هتلاقي، هيساعدك) — that no glossary/prompt fix can reach, since it's
grammatical, not lexical. Fanar-2-27B-Instruct (QCRI/HBKU, Apache 2.0, `google/gemma-3-27b-pt`
base, 32k ctx) explicitly claims Gulf/Levantine/Egyptian dialect training and was the deferred
candidate for testing this. Ran per the approved plan (`najdi-q2-wrong-elegant-papert.md`):
offline only, via `dialect_eval_full.py --model` against Ollama's `/api/chat` directly — `llm.py`,
`server.py`, `routing.py`, `tts_omnivoice_v1.py` untouched, zero production risk.

**Blocker fixed first, not a config problem**: the GGUF pulled from
`hf.co/mradermacher/Fanar-2-27B-Instruct-i1-GGUF:i1-Q4_K_M` crashed Ollama/llama-server's
Jinja-subset parser on load (`Unknown statement: raw`) — its embedded `tokenizer.chat_template`
contains a `{% raw %}...{% endraw %}` tool-calling block the parser doesn't support. An Ollama
Modelfile `TEMPLATE` override does NOT reach this (confirmed: `llama-server` reads the GGUF's
own embedded Jinja template directly, regardless of any Modelfile-level Go-template override —
two separate template systems). Fixed at the source: patched the GGUF blob's
`tokenizer.chat_template` KV string in place (pure-Python `struct`, no `gguf` package available
without pip/sudo) — removed the unused tool-calling section and made the `no_thinking`
empty-`<think></think>` injection unconditional, same byte length so no other GGUF offset
moved. Verified byte-for-byte post-write. This lives in the blob itself, so it applies to any
Ollama tag pointing at it.

**Harness change** (`eval/dialect_eval_full.py`, diff-reviewable, additive only): `--model` flag
(omitted = today's exact unchanged behavior, confirmed via a no-flag run); a neutral
`_ALT_MODEL_CONFIG` used only when `--model` is passed (temp 0.7/top_p 0.9/top_k 40 — from
`MODEL_CONFIGS["default"]`, deliberately NOT qwen3.5's reverse-engineered tuning, to avoid
confounding "better dialect training" with "likes qwen's own sampling knobs" — `num_predict`
raised to 4000, a functional precondition for a thinking-by-default model, not a quality
choice); `strip_think()` removes a closed `<think>...</think>` block and flags an **unclosed**
one `invalid` rather than scoring the truncated reasoning fragment as a reply (leak_lint's
Arabic-word check can't tell a reasoning trace from a real reply — both are well-formed
Arabic). No `llm.py`/`MODEL_CONFIGS` change — keeps `golden_prompts.py`'s G3 gate (which hashes
that surface) untouched; reconfirmed `GOLDEN GATES G1/G2/G3 GREEN`, same hash as before this
work.

**Full 245-question run** (`eval/dialect_eval_questions.json` + `..._holdout_questions.json`,
same set as the 2026-07-27 qwen3.5 run this compares against — `logs/ab_runs/2026-07-27_1302-
2026-07-27-post-history-fix-full.md`), `keep_alive:-1`, sequential (qwen3.5 stopped first, no
interleaving):

| model | leaky | drifty | invalid | routing-bad | avg sec/turn (general/general) |
|---|---|---|---|---|---|
| qwen3.5:27b | 38 | 1 | 0 | 37 | 2.2 |
| Fanar-2-27B | 31 | 1 | 0 | 37 | 2.2 |

Report: `logs/ab_runs/2026-07-27_1930-fanar2-27b-vs-qwen35-full.md` (local only). **Routing-bad
is line-for-line identical (37/37, same cases both runs)** — confirms `looks_najdi`/
`looks_egyptian` are a pure function of input text, model-independent, exactly as the plan
assumed. `invalid` stayed 0 for Fanar across all 245 cases — the GGUF-level `no_thinking` patch
fully suppressed `<think>` output; no truncated-reasoning artifacts anywhere in the run.

**The actual question — does Fanar reduce the verb-morphology leak class? Manual read of every
Najdi general+holdout reply (both reports, same 100+ turns), grep-verified, not eyeballed:**

- **بـ-imperfective verb forms** (excluding ordinary nouns like بيئة/بيانات/بناء that just start
  with the same letters): qwen3.5 ≈20 real instances (بيصير، بيجيب، بتقلل، بتعلم، بتعطي، بتصير،
  بتسبب، بتسافر، بتخدم، بتجف، بتبقى، بيقبل، بيعمل، بيضرب، بيشد، بيزيد، بيجي, etc.) vs. Fanar-2
  **6** (بيوصلوا، بيحفظ، بيخونك، بيؤثر ×2، بتتعلمين) — roughly a **70% reduction**.
- **هـ-future verb forms**: qwen3.5 has real instances (هتلاقي ×4، هيساعدك، هتقدر); Fanar-2 has
  **zero** real instances (every ه-prefixed hit on manual check was a false-positive noun,
  اهتمام/اهتمامات — not a verb).
- **Verdict: confirmed.** Fanar-2 genuinely reduces the specific grammatical leak class that
  motivated this test — this is a real, structural improvement `leak_lint`'s word-list checks
  never measured (both runs show 0 detector hits for this class; the raw 38→31 leak count
  actually understates the improvement, since it's dominated by vocabulary hits, not morphology).

**New leak class, not present in qwen3.5 at all — Egyptian vocabulary bleeding into Najdi/Fusha:**
`كويس` (Egyptian "good/well") appears **14 times** in Fanar's Najdi output (general + holdout) vs.
**0** times in qwen3.5's — same question set. This is a real, substantial regression, distinct
from (and larger than) `leak_lint`'s own per-turn LEAK-count for `كويس` (~7 flagged turns; it
fires once per turn even when a reply uses the word twice). Gulf/Khaleeji `وايد` also appears
(4× Fanar vs. 1× qwen3.5 on the same set) — smaller numbers, same direction, out-of-scope
vocabulary per CLAUDE.md's dialect list either way.

**Known `leak_lint` false positive, not new, already documented (2026-07-22 entry above)**:
`مرة` fired on Fanar Egyptian turn `EG15` for "من أول مرة"/"المرة الجاية" (ordinary "time/occurrence"
usage, not the forbidden Najdi-intensifier sense) — the same unfixed positional-ambiguity gap
recorded on 2026-07-22 ("needs a purpose-built positional regex... documented, not fixed"),
recurring here on a different model, not a new bug.

**Fluency/general quality, by direct reading**: both models produce grammatically sound,
fluent Najdi and Egyptian text; no invented words or broken negation noticed in either report on
this pass. One Fanar-specific instruction-following miss, seen in the earlier 5-question smoke
test (`logs/ab_runs/2026-07-27_1928-fanar2-smoketest-full.md`, F001): opens with "بالتأكيد،
إليك..." ("Certainly, here are...") — the exact AI-boilerplate opener `SYSTEM_PROMPT` explicitly
tells the model not to use; qwen3.5 follows this instruction reliably. Latency is comparable
between the two models across every group in the summary table (both ~1-2s/turn; Fanar is
faster on `short_utterance`, 0.5s vs 1.0s) — the `num_predict:4000`/thinking-mode risk this test
was designed around did not materialize as a latency cost, because the GGUF patch suppressed
`<think>` entirely.

**Overall conclusion**: Fanar-2 is a real, measurable improvement specifically on the
grammatical-morphology axis that motivated this test (بـ-prefix/هـ-future), at the cost of a new
vocabulary-level leak (Egyptian words in Najdi/Fusha output) that this project's own tooling
(`DIALECT_REPAIR_MAP`, `leak_lint.FORBIDDEN`) is already structurally built to catch and patch —
unlike morphology, which the project's own `EGYPTIAN_GRAMMAR_RULE`/`NAJDI_GRAMMAR_RULE` prompt
patterns can only nudge, not fix. **Not acted on this round** — this was an evaluation-only
comparison per the approved plan, not a production swap decision; a swap would additionally need
a real production-swap plan (Modelfile/`MODEL_CONFIGS` entry, `golden_prompts.py` G3 recapture,
the CATT/TTS-voice-prompt interaction re-checked, `LLM_NUM_CTX` behavior with a real 32k-context
model) — none of that is in scope here.
