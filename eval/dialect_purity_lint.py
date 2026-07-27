"""
dialect_purity_lint.py — cross-dialect leak rates over live logs (no GPU, ~seconds)
=============================================================================================
Scans the Arabic assistant replies in logs/interactions.jsonl and reports, per ROUTED
dialect, tokens that must not appear there (leak_lint.find_leaks: e.g. دلوقتي inside a
Najdi reply, وش inside an Egyptian reply, any dialect word inside Fusha). This turns "the
dialects are mixing" into a NUMBER to watch after every prompt/routing change.

The routed dialect comes from the row's `route` block (added to _write_log at plan Step 1,
2026-07-20). Rows without it (older logs) are counted and SKIPPED — they predate Egyptian
on this branch and there is no monolith _route_turn to reconstruct them with. English and
mixed turns are skipped (mixed replies are deliberately un-pinned).

Severity, same doctrine as the old branch: LEAK = hard (cross-dialect token, hijacked
identity); msa-drift = soft (educated register creeping in). Every offending token is
printed so a human can veto false positives (e.g. زين as the name "Zain"). The Egyptian
بـ-present inside a Najdi reply is NOT lintable (identical surface to the Najdi بـ-future)
— NAJDI_GRAMMAR_RULE handles it prompt-side; judging it stays by ear.

Usage:
    .venv/bin/python eval/dialect_purity_lint.py                     # whole log
    .venv/bin/python eval/dialect_purity_lint.py --since 2026-07-20  # from a date
    .venv/bin/python eval/dialect_purity_lint.py path/to/log.jsonl --since 2026-07-21
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leak_lint     # noqa: E402
import quality_lint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logfile", nargs="?",
                    default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "logs", "interactions.jsonl"))
    ap.add_argument("--since", default="", help="only rows whose ts >= this prefix (e.g. 2026-07-20)")
    ap.add_argument("--judge", action="store_true",
                    help="also run the LLM-judge quality pass (soft/informational — see "
                         "quality_lint.py) on a bounded random sample of routed rows. Stop the "
                         "live server first. Unlike dialect_ab.py, this script never calls the "
                         "live-serving model itself (it only reads already-recorded text from "
                         "the log), so no VRAM swapping happens here.")
    ap.add_argument("--judge-sample", type=int, default=25,
                    help="max routed rows to judge (seeded/repeatable) — production logs are "
                         "unbounded and grow over time, unlike dialect_ab.py's fixed 15-question "
                         "set, so judge-pass cost must stay constant regardless of log size.")
    ap.add_argument("--judge-model", default=quality_lint.JUDGE_MODEL)
    args = ap.parse_args()
    if args.judge:
        quality_lint.check_judge_model_available(args.judge_model)

    if not os.path.exists(args.logfile):
        print(f"No log file at {args.logfile} — nothing to lint yet.")
        return 0
    rows = [json.loads(l) for l in open(args.logfile, encoding="utf-8") if l.strip()]
    if args.since:
        rows = [r for r in rows if r.get("ts", "") >= args.since]

    stats: dict[str, dict] = {}
    no_route = 0
    judge_candidates: list[tuple[str, str, str, str]] = []   # (ts, dialect, transcript, reply)
    for row in rows:
        reply = row.get("response", "")
        if not leak_lint._AR_WORD_RE.search(reply):
            continue                      # empty / English reply — nothing to lint
        route = row.get("route")
        if route is None:
            no_route += 1                 # pre-Step-1 row — skip (see docstring)
            continue
        dialect = leak_lint.TTS_LANG_TO_DIALECT.get(route.get("tts_language") or "")
        if dialect is None:
            continue                      # English/mixed turn
        leaks, drift = leak_lint.find_leaks(reply, dialect)
        s = stats.setdefault(dialect, {"n": 0, "leaky": 0, "drifty": 0, "rows": []})
        s["n"] += 1
        if leaks:
            s["leaky"] += 1
        if drift:
            s["drifty"] += 1
        if leaks or drift:
            s["rows"].append((row.get("ts", "?"), leaks, drift, row.get("transcript", "")[:60]))
        judge_candidates.append((row.get("ts", "?"), dialect, row.get("transcript", ""), reply))

    if not stats:
        print(f"No routed Arabic replies found in range."
              f"{f' ({no_route} pre-route-block rows skipped.)' if no_route else ''}")
        return 0

    print(f"\n== Dialect-purity lint: {os.path.basename(args.logfile)}"
          f"{' since ' + args.since if args.since else ''} =="
          f"{f'  ({no_route} pre-route-block rows skipped)' if no_route else ''}\n")
    print(f"{'dialect':>9}  {'replies':>7}  {'with-leaks':>10}  {'leak-rate':>9}  {'msa-drift':>9}")
    total_n = total_leaky = 0
    for d in ("Najdi", "Egyptian", "Fusha"):
        s = stats.get(d)
        if not s:
            continue
        total_n += s["n"]; total_leaky += s["leaky"]
        print(f"{d:>9}  {s['n']:>7}  {s['leaky']:>10}  {100 * s['leaky'] / s['n']:>8.0f}%  {s['drifty']:>9}")
    print(f"{'TOTAL':>9}  {total_n:>7}  {total_leaky:>10}  {100 * total_leaky / max(total_n, 1):>8.0f}%")

    print("\n-- offending turns --")
    for d in ("Najdi", "Egyptian", "Fusha"):
        for ts, leaks, drift, q in stats.get(d, {}).get("rows", []):
            parts = []
            if leaks:
                parts.append("LEAK: " + "، ".join(leaks))
            if drift:
                parts.append("msa-drift: " + "، ".join(drift))
            print(f"  [{ts}] {d:<8} {' | '.join(parts)}   (Q: {q})")

    if args.judge and judge_candidates:
        import httpx
        sample = random.Random(0).sample(judge_candidates, min(args.judge_sample, len(judge_candidates)))
        print(f"\n-- judge findings (sampled {len(sample)}/{len(judge_candidates)} routed rows, "
              f"seed=0, soft/informational) --")
        with httpx.Client() as client:
            for ts, dialect, transcript, reply in sample:
                findings = quality_lint.judge_reply(transcript, dialect, reply, client,
                                                    model=args.judge_model)
                for f in findings:
                    arrow = f" → suggested: {f['suggested_fix']}" if f["suggested_fix"] else ""
                    print(f"  [{ts}] {dialect:<8} {f['category']}[{f['confidence']}] "
                          f"«{f['span']}»{arrow} — {f['note']}   (Q: {transcript[:60]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
