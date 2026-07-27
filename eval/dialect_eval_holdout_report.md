# Dialect marker vetting methodology

This file resolves a dead citation: `routing.py`'s `_NAJDI_MARKERS` comment has referenced
`dialect_eval_holdout_report.md` as the source of its vetting decisions since the marker set
was expanded, but the file was never actually committed anywhere in this repo's history. This
is that file — created 2026-07-21 as part of `najdi-q2-wrong-elegant-papert.md`'s Part A.

**What this is, and isn't:** despite the legacy name, this is a **marker-vetting methodology
and worked-examples log** for the deterministic lexicon classifiers (`looks_najdi`/
`looks_egyptian` in `routing.py`), not a statistical train/holdout split. See the "why no
split" conclusion below for the reasoning — a closed, enumerable lexicon has a different
overfitting shape than an LLM's stochastic output, and doesn't need one.

## Purpose

`looks_najdi`/`looks_egyptian` are precision-first lexicon matchers: a candidate word/phrase
only becomes a marker if it can't plausibly appear in ordinary Fusha/MSA speech. Every marker
addition or rejection should be traceable to a decision recorded here, so future contributors
don't have to reverse-engineer the reasoning from a code comment or guess whether a similar
candidate was already considered and rejected.

## Acceptance criteria

A candidate marker is:

- **Accepted (strong)** if it is either (a) a grammatical conjugation of an already-vetted
  word, or (b) a standalone word that essentially never occurs in real Fusha text.
- **Rejected** if it collides with a plausible real MSA sentence, even if a genuine dialect
  sense also exists — precision over recall, the same philosophy `dialect_id_eval.py`'s own
  docstring states for the whole classifier.
- **Weak (0.5 weight)** for common-but-not-exclusive words that can support a strong marker or
  pair up with another weak marker, but never decide alone (the Egyptian detector's `_EGYPTIAN_WEAK`
  mechanism — see `routing.py`).

## Worked examples (decisions already made, transcribed from code comments)

### Rejected

- **`قدر`/`يقدر` family** — collides with real MSA usage: "لا يقدر على" (unable to), "بقدر
  الإمكان" (as much as possible). A dialect sense exists, but the MSA collision risk is too
  high for a strong marker.
- **`راح` alone** — collides with plain MSA "went" (the past tense of راح، يروح). Also
  independently excluded from `eval/leak_lint.py`'s forbidden sets for the same reason: "راح
  is deliberately NOT forbidden in Egyptian — it is also valid Egyptian past 'went'; راح-future
  vs راح-went can't be separated lexically."
- **`وسايل`/`يقرا`** — just hamza-dropped spellings of ordinary MSA words (وسائل، يقرأ), not a
  dialect signal at all.

### Retired (2026-07-24: the accepted cost stopped being accepted)

- **`عشان`، `اللي`، `لسه`/`لسا`، `يلا`** — all four were carried in the "Accepted, with caveats"
  tier below (2026-07-20 decision) on the reasoning that the collision cost was worth the Najdi
  recall they bought. Owner revisited this: since these words are genuinely pan-dialect (not
  Najdi-exclusive), using them to route to Najdi specifically was never well-justified — it was
  the original, pre-Egyptian design carried over by default, not a considered tradeoff. Removed
  from `_NAJDI_MARKERS` entirely (not added to `_EGYPTIAN_MARKERS` either — they simply stop
  discriminating dialect). Quantified impact: `dialect_id_cases.jsonl`'s Egyptian recall
  64%→80%, Najdi recall 88%→82% (one accepted regression — see `eval/BASELINES.md`'s
  2026-07-24 entry for the full row-by-row before/after).

### Watched / deferred (flagged, not yet acted on)

- **Bare `ايه`** — normalized ايه is both the Najdi colloquial "yes" and an Egyptian
  interrogative "what." Known accepted false-positive, pinned in `dialect_id_cases.jsonl`'s
  `najdi-yes-watch` row. v1.1 candidate: demote to a weak marker instead of a strong one.
- **`دول`** — demoted to weak in this branch (vs. a stronger treatment on an earlier branch):
  bare دول is the everyday MSA plural of دولة in construct state ("دول الخليج", "دول العالم") —
  as a strong marker it flips real Fusha questions to Egyptian. Pinned in
  `dialect_id_cases.jsonl`'s `msa-duwal-guard` row.

## Why no train/holdout split (conclusion, not a deliverable)

An LLM's output is open-ended — a fixed question set can never enumerate every phrasing a real
user might say, which is why `eval/dialect_ab_heldout_cases.jsonl` exists as a genuinely
separate, paraphrased set (see `eval/README.md`). A lexicon classifier is different: the
question "does this candidate word collide with plausible ordinary MSA" is a closed,
enumerable check a human can reason through directly, and `eval/dialect_id_cases.jsonl`'s
existing provenance tags (`seed-no-marker`, `collision`, `weak-pair-watch`,
`msa-duwal-guard`, `stt-hallucination-watch`, ...) already function as permanent adversarial
regression pins baked into the one file — a held-out partition would just be a second copy of
the same closed universe, not a check against something genuinely unseen.

## Vetting checklist for any new candidate marker

1. **Propose** the candidate with the specific real utterance(s) that motivated it — ideally
   sourced from production logs once traffic exists (`eval/dialect_purity_lint.py --judge`),
   not invented from memory.
2. **Check** it against:
   - (i) existing rows in `eval/dialect_id_cases.jsonl` — does a similar candidate already
     have a recorded decision?
   - (ii) a written-out adversarial set of plausible ordinary-MSA sentences containing the
     candidate — if you can construct 2-3 natural, non-dialectal MSA sentences using it, it's
     a rejection candidate.
   - (iii) a real production-log grep once traffic exists — does it actually show up in
     genuine MSA usage in this project's domain (water-utility/field-ops conversations)?
3. **Decide** accept (strong) / reject / weak, and write the reasoning as a new row in this
   document's worked-examples section — **even for rejections**, mirroring `eval/BASELINES.md`'s
   documented-negative-results ethos (a rejected candidate with its reasoning is as valuable a
   record as an accepted one).
4. **If accepted:** add it to the relevant marker set in `routing.py` with a comment pointing
   at this document; add both a positive row and a fresh collision-guard row to
   `eval/dialect_id_cases.jsonl`.
5. **Re-run** `eval/test_routing.py` and `eval/dialect_id_eval.py` — both must stay green
   (or, if the change is an intentional PRE-EGYPTIAN-style flip, the pin updates in the same
   commit per `test_routing.py`'s own pin-class convention).
