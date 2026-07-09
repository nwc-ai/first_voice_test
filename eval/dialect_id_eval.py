"""
dialect_id_eval.py — measure _detect_dialect accuracy on labeled transcripts (no GPU)
======================================================================================
Reads a JSONL of {"text", "dialect"} rows (dialect: "Najdi"|"Egyptian"|null,
null = MSA/unclear → the expected outcome is the Fusha default) and reports per-dialect
recall, false-switch rate, and a confusion table.

The default input is eval/dialect_id_cases.jsonl — a SYNTHETIC seed set written for
bootstrap only. Replace/extend it with real labeled transcripts as they accumulate
(SAVE_UTTERANCES=1 collects them — see eval/README.md); the numbers only start to mean
something once the set is real speech.

Run:
    .venv/bin/python eval/dialect_id_eval.py [cases.jsonl]

Interpreting: recall on marker-less rows ("source": "...-no-marker") is EXPECTED to be 0 —
the classifier is precision-first by design; those rows exist to quantify how much a future
ML dialect-ID model could add. What must stay near-zero is CROSS-dialect confusion
(e.g. Egyptian rows classified Najdi), which misroutes voice + pronunciation.
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "dialect_id_cases.jsonl")
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
for r in rows:
    expected = r["dialect"] or "None"
    got = server._detect_dialect(r["text"]) or "None"
    confusion[expected][got] += 1

labels = ["Najdi", "Egyptian", "None"]
print(f"\n{len(rows)} cases from {os.path.basename(path)}\n")
print(f"{'true \\ pred':>12} " + " ".join(f"{l:>9}" for l in labels))
for t in labels:
    if not confusion.get(t):
        continue
    print(f"{t:>12} " + " ".join(f"{confusion[t].get(p, 0):>9}" for p in labels))

print()
total_cross = 0
for t in labels[:3]:
    row = confusion.get(t, {})
    n = sum(row.values())
    if not n:
        continue
    hit   = row.get(t, 0)
    cross = sum(v for k, v in row.items() if k not in (t, "None"))
    total_cross += cross
    print(f"{t:>9}: recall {hit}/{n} ({100*hit/n:.0f}%)   cross-dialect confusion {cross}/{n}"
          f"{'   ← MUST be ~0' if cross else ''}")
none_row = confusion.get("None", {})
n_none = sum(none_row.values())
if n_none:
    fp = n_none - none_row.get("None", 0)
    print(f"{'None/MSA':>9}: correctly-unclear {none_row.get('None',0)}/{n_none}   false dialect firing {fp}/{n_none}")

sys.exit(1 if total_cross else 0)
