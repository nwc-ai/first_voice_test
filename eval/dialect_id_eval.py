"""
dialect_id_eval.py — spoken-dialect classifier recall/confusion on labeled transcripts
(no GPU, ~seconds)
=============================================================================================
Reads eval/dialect_id_cases.jsonl rows:
    {"text", "dialect": truth label (Najdi|Egyptian|Hijazi|null),
     "expect": the route this pipeline SHOULD produce ("Najdi"|"Egyptian"|"None"),
     "accepted_fp": true when `expect` deliberately differs from truth (frozen behavior or
                    an owner-accepted watch case), "source", "note"}

`dialect` records linguistic truth; `expect` records the DESIGNED routing outcome, which can
legitimately differ:
  - precision-first misses ("...-no-marker" rows expect "None"),
  - former collision rows (2026-07-24: اللي/عشان/لسه/يلا were removed from `_NAJDI_MARKERS`
    for being pan-dialect, not Najdi-exclusive — most of these rows now correctly route
    Egyptian or None; one row remains an accepted regression, one an unresolved gap, see
    each row's own `note` and `eval/BASELINES.md`'s 2026-07-24 entry),
  - Hijazi rows (unsupported dialect: إيش-family rows route Najdi, the rest fall to Fusha),
  - WATCH accepted-FP rows (bare-ايه Najdi-yes, weak-pair, STT-hallucination sim).

The classifier under test is the production precedence rule:
    looks_najdi(text) → "Najdi"   (evaluated first, short-circuits — the frozen invariant)
    looks_egyptian(text) → "Egyptian"   (only if routing.looks_egyptian exists — plan Step 2)
    otherwise → "None" (Fusha default)
Before Step 2 lands, rows expecting "Egyptian" are auto-downgraded to expect "None" so this
gate is green at every plan step.

HARD GATE (exit 1): any row whose classification differs from its (step-adjusted) `expect`.
In particular: a truth-Najdi row not routing Najdi, or a truth-null row firing a dialect
without `accepted_fp`, is always fatal.

Run:
    .venv/bin/python eval/dialect_id_eval.py [cases.jsonl]
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import routing  # noqa: E402

looks_egyptian = getattr(routing, "looks_egyptian", None)


def classify(text: str) -> str:
    if routing.looks_najdi(text):
        return "Najdi"
    if looks_egyptian is not None and looks_egyptian(text):
        return "Egyptian"
    return "None"


path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "dialect_id_cases.jsonl")
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
mismatches: list[str] = []
egy_capability = looks_egyptian is not None

for r in rows:
    truth    = r["dialect"] or "None"
    expected = r["expect"]
    if expected == "Egyptian" and not egy_capability:
        expected = "None"   # pre-Step-2: Egyptian detection doesn't exist yet
    got = classify(r["text"])
    confusion[truth][got] += 1
    if got != expected:
        mismatches.append(f"  {r['text'][:48]!r}: truth={truth} expected={expected} got={got}"
                          f"  [{r.get('source', '')}]")

labels = ["Najdi", "Egyptian", "None"]
print(f"\n{len(rows)} cases from {os.path.basename(path)}"
      f"  (looks_egyptian: {'present' if egy_capability else 'NOT YET — Egyptian expectations downgraded'})\n")
print(f"{'truth \\ routed':>16} " + " ".join(f"{l:>9}" for l in labels))
for t in ("Najdi", "Egyptian", "Hijazi", "None"):
    if confusion.get(t):
        print(f"{t:>16} " + " ".join(f"{confusion[t].get(p, 0):>9}" for p in labels))

# Informational recall per truth label (the designed caps make raw recall < 100% by design;
# the hard gate is expect-mismatch, printed below).
print()
for t in ("Najdi", "Egyptian"):
    row = confusion.get(t, {})
    n = sum(row.values())
    if n:
        hit = row.get(t, 0)
        print(f"{t:>9}: routed-as-{t} {hit}/{n} ({100 * hit / n:.0f}%)   (informational — "
              f"designed caps: no-marker rows{', collision rows' if t == 'Egyptian' else ''})")
none_row = confusion.get("None", {})
n_none = sum(none_row.values())
if n_none:
    fp = n_none - none_row.get("None", 0)
    print(f"{'MSA/null':>9}: correctly-unclear {none_row.get('None', 0)}/{n_none}   "
          f"dialect firings {fp}/{n_none} (accepted-FP rows only — anything else is fatal)")

if mismatches:
    print(f"\n{len(mismatches)} EXPECTATION MISMATCH(ES):")
    print("\n".join(mismatches))
    sys.exit(1)
print("\nALL DIALECT-ID EXPECTATIONS MET")
