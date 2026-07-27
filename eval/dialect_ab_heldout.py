"""
dialect_ab_heldout.py — held-out A/B check (paraphrased, NEVER-tuned-against question set)
=============================================================================================
Same production prompt surface as dialect_ab.py (llm.build_turn -> llm.SYSTEM_PROMPT ->
qwen3.5 options; fresh context per turn), but reads eval/dialect_ab_heldout_cases.jsonl
instead of the frozen QUESTIONS list — a genuinely separate set that stress-tests whether a
fix generalizes to new wording/topics, not just the specific dev-set examples it was built
and tuned against. See the plan najdi-q2-wrong-elegant-papert.md and eval/README.md's
discipline note for when this is allowed to run.

DISCIPLINE (procedural, same footing as "never re-capture golden_fixtures.jsonl"): this set
runs ONCE per completed change, after dev-set (dialect_ab.py) iteration is believed done —
never used to choose between candidate wordings. Reading only the aggregate summary table
never "spends" a case; making a wording change because of something read in an INDIVIDUAL
held-out transcript does — that case gets reworded/rotated before it's trusted again (log the
rotation in eval/BASELINES.md).

Reuses dialect_ab.py's run_turn/ask_ollama/_REQUEST_SUFFIX/_AR_GREETING rather than
duplicating them — the underlying call machinery is identical, only the question source
differs.

Usage:
    .venv/bin/python eval/dialect_ab_heldout.py --tag <name>
    .venv/bin/python eval/dialect_ab_heldout.py --dialects Najdi,Egyptian --judge
Requires Ollama on localhost:11434 with qwen3.5:27b loaded (start_server.sh starts it).
Reports land in logs/ab_runs/<timestamp>-heldout[-tag].md (gitignored — data, not code).
"""

import argparse
import datetime
import json
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dialect_ab    # noqa: E402  reuse run_turn/ask_ollama/_REQUEST_SUFFIX/_AR_GREETING
import quality_lint  # noqa: E402

_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_PATH = os.path.join(_DIR, "dialect_ab_heldout_cases.jsonl")


def load_cases() -> list[dict]:
    with open(CASES_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dialects", default="Najdi,Fusha")
    ap.add_argument("--tag", default="", help="suffix for the report filename (e.g. after-ruleC1)")
    ap.add_argument("--judge", action="store_true",
                    help="also run the LLM-judge quality pass (see quality_lint.py). Stop the "
                         "live server + `ollama stop qwen3.5:27b` first.")
    ap.add_argument("--judge-model", default=quality_lint.JUDGE_MODEL)
    args = ap.parse_args()
    dialects = [d.strip() for d in args.dialects.split(",") if d.strip()]
    if args.judge:
        quality_lint.check_judge_model_available(args.judge_model)

    cases = load_cases()
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "logs", "ab_runs")
    os.makedirs(run_dir, exist_ok=True)
    report_path = os.path.join(run_dir, f"{ts}-heldout{'-' + args.tag if args.tag else ''}.md")

    stats = {d: {"n": 0, "leaky": 0, "drifty": 0, "invalid": 0, "judged": 0, "secs": 0.0}
            for d in dialects}
    lines = [f"# HELD-OUT dialect A/B run — {ts}{' (' + args.tag + ')' if args.tag else ''}",
             f"Model: {dialect_ab.llm.MODEL}. Fresh context per turn (no history). "
             f"Question set: eval/dialect_ab_heldout_cases.jsonl (never tuned against — "
             f"see this script's own discipline note).\n"]

    with httpx.Client() as client:
        for d in dialects:
            lines.append(f"\n## {d}\n")
            print(f"\n=== {d} (HELD-OUT) ===")
            turns = [("ho-ar-greeting", "conversational", dialect_ab._AR_GREETING[d], "ar")] + [
                (c["id"], c["kind"], c["text"] + dialect_ab._REQUEST_SUFFIX[d], "en")
                for c in cases]
            for qid, kind, text, lang in turns:
                try:
                    r = dialect_ab.run_turn(client, text, lang, judge=args.judge,
                                            judge_model=args.judge_model)
                except Exception as e:
                    print(f"  {qid}: ERROR {type(e).__name__}: {e}")
                    lines.append(f"### {qid} — ERROR: {e}\n")
                    continue
                s = stats[d]
                s["n"] += 1
                s["secs"] += r["elapsed_s"]
                flags = []
                if r["invalid"]:
                    s["invalid"] += 1
                    flags.append("INVALID: empty/non-Arabic reply on a dialect-routed turn")
                if r["leaks"]:
                    s["leaky"] += 1
                    flags.append("LEAK: " + "، ".join(r["leaks"]))
                if r["drift"]:
                    s["drifty"] += 1
                    flags.append("drift: " + "، ".join(r["drift"]))
                if r["findings"]:
                    s["judged"] += 1
                    flags.append("JUDGE: " + "; ".join(
                        f"{f['category']}[{f['confidence']}]" for f in r["findings"]))
                print(f"  {qid:<18} [{kind}] {r['elapsed_s']:.1f}s → "
                      f"{r['target'] or r['tts_language'] or 'EN'}"
                      f"  {' | '.join(flags) if flags else 'clean'}")
                lines.append(f"### {qid} [{kind}] — routed {r['target'] or 'None'} — "
                             f"{r['elapsed_s']:.1f}s — {' | '.join(flags) if flags else 'clean'}")
                lines.append(f"**Q:** {text}\n\n**A:** {r['raw']}\n")
                if r["findings"]:
                    lines.append("**Judge findings:**")
                    for f in r["findings"]:
                        arrow = f" → suggested: {f['suggested_fix']}" if f["suggested_fix"] else ""
                        lines.append(f"- {f['category']} [{f['confidence']}] «{f['span']}»{arrow}"
                                     f" — {f['note']}")
                    lines.append("")

    print(f"\n{'dialect':>9}  {'turns':>5}  {'leaky':>5}  {'drifty':>6}  {'invalid':>7}  {'avg-sec':>7}"
          + (f"  {'judged':>6}" if args.judge else ""))
    header = ("| dialect | turns | leaky | drifty | invalid | avg sec/turn |"
             + (" judged |" if args.judge else ""))
    sep    = "|---|---|---|---|---|---|" + ("---|" if args.judge else "")
    lines.append(f"\n## Summary\n\n{header}\n{sep}")
    for d in dialects:
        s = stats[d]
        avg = s["secs"] / max(s["n"], 1)
        print(f"{d:>9}  {s['n']:>5}  {s['leaky']:>5}  {s['drifty']:>6}  {s['invalid']:>7}  {avg:>6.1f}s"
              + (f"  {s['judged']:>6}" if args.judge else ""))
        row = f"| {d} | {s['n']} | {s['leaky']} | {s['drifty']} | {s['invalid']} | {avg:.1f} |"
        if args.judge:
            row += f" {s['judged']} |"
        lines.append(row)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport: {report_path}")
    print("HELD-OUT set — per this script's discipline note, read the individual transcripts "
          "at most ONCE per completed change. Record the summary table in eval/BASELINES.md "
          "under a 'Held-out' subsection, separate from the dev-set numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
