"""
dialect_eval_full.py — full-scale dialect quality run over dialect_eval_questions.json +
dialect_eval_holdout_questions.json (245 questions total: Najdi/Fusha/Egyptian x 7 operational
categories, plus the routing/holdout buckets with Egyptian additions).

Unlike dialect_ab.py (14 fixed English questions, explicit-dialect-request wrapping), this
harness sends each question's text AS-IS with its recorded `lang` (or "ar" for the
dialect_eval_questions.json set, which is natural dialect-phrased Arabic) and lets
llm.build_turn's real routing (looks_najdi/looks_egyptian/requested_dialect) decide the
dialect — so this measures ROUTING accuracy and REPLY quality together, not reply quality
alone. Deterministic checks only (leak_lint + invalid-response detection) — no LLM judge,
per project policy (eval/README.md Rules). Qualitative grammar review is manual, by the
owner/assistant reading the saved report.

Usage:
    .venv/bin/python eval/dialect_eval_full.py --tag <name>
Requires Ollama on localhost:11434 with qwen3.5:27b loaded.
Report: logs/ab_runs/<timestamp>-<tag>.md (gitignored). Record summary numbers in BASELINES.md.
"""

import argparse
import datetime
import json
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import leak_lint  # noqa: E402
import llm        # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/chat"
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

_DIALECT_TO_TTS = {"fusha": "standard arabic", "najdi": "najdi arabic", "egyptian": "egyptian arabic"}


def load_cases():
    general = json.load(open(os.path.join(EVAL_DIR, "dialect_eval_questions.json"), encoding="utf-8"))
    holdout = json.load(open(os.path.join(EVAL_DIR, "dialect_eval_holdout_questions.json"), encoding="utf-8"))
    cases = []
    for row in general:
        cases.append({
            "id": row["id"], "group": f"general/{row['category']}", "text": row["text"],
            "lang": "ar", "expected_tts_language": _DIALECT_TO_TTS[row["dialect"]],
        })
    for row in holdout:
        cases.append({
            "id": row["id"], "group": f"holdout/{row['bucket']}", "text": row["text"],
            "lang": row["lang"], "expected_tts_language": row["expected_tts_language"],
        })
    return cases


def ask_ollama(client: httpx.Client, messages: list[dict]) -> str:
    cfg = llm.get_model_config(llm.MODEL)
    payload = {"model": llm.MODEL, "messages": messages, "stream": False,
               "keep_alive": -1, "options": cfg["options"], **cfg["extra"]}
    r = client.post(OLLAMA_URL, json=payload, timeout=300.0)
    r.raise_for_status()
    msg = r.json().get("message") or {}
    return (msg.get("content", "") or "").strip()


def run_case(client: httpx.Client, case: dict) -> dict:
    turn_content, tts_language, route_meta = llm.build_turn(case["text"], case["lang"])
    messages = [
        {"role": "system", "content": llm.SYSTEM_PROMPT},
        {"role": "user", "content": turn_content},
    ]
    t0 = time.time()
    raw = ask_ollama(client, messages)
    elapsed = time.time() - t0
    actual_dialect = leak_lint.TTS_LANG_TO_DIALECT.get(tts_language or "")
    invalid = bool(actual_dialect) and not leak_lint._AR_WORD_RE.search(raw)
    leaks, drift = leak_lint.find_leaks(raw, actual_dialect) if actual_dialect and not invalid else ([], [])
    expected_dialect = leak_lint.TTS_LANG_TO_DIALECT.get(case["expected_tts_language"] or "")
    routing_ok = (case["expected_tts_language"] is None) or (tts_language == case["expected_tts_language"])
    return {"tts_language": tts_language, "actual_dialect": actual_dialect,
            "expected_dialect": expected_dialect, "routing_ok": routing_ok,
            "raw": raw, "invalid": invalid, "leaks": leaks, "drift": drift,
            "elapsed_s": elapsed, "route_meta": route_meta}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="", help="suffix for the report filename")
    ap.add_argument("--limit", type=int, default=0, help="only run the first N cases (0 = all)")
    args = ap.parse_args()

    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = os.path.join(os.path.dirname(EVAL_DIR), "logs", "ab_runs")
    os.makedirs(run_dir, exist_ok=True)
    report_path = os.path.join(run_dir, f"{ts}{'-' + args.tag if args.tag else ''}-full.md")

    stats = {}  # group -> counters
    lines = [f"# Full dialect eval run — {ts}{' (' + args.tag + ')' if args.tag else ''}",
             f"Model: {llm.MODEL}. {len(cases)} cases. Fresh context per turn (no history).\n"]

    with httpx.Client() as client:
        for case in cases:
            group = case["group"]
            s = stats.setdefault(group, {"n": 0, "leaky": 0, "drifty": 0, "invalid": 0,
                                          "routing_bad": 0, "secs": 0.0})
            try:
                r = run_case(client, case)
            except Exception as e:
                print(f"  {case['id']}: ERROR {type(e).__name__}: {e}")
                lines.append(f"### [{group}] {case['id']} — ERROR: {e}\n")
                continue
            s["n"] += 1
            s["secs"] += r["elapsed_s"]
            flags = []
            if not r["routing_ok"]:
                s["routing_bad"] += 1
                flags.append(f"ROUTING: expected {case['expected_tts_language']!r} got {r['tts_language']!r}")
            if r["invalid"]:
                s["invalid"] += 1
                flags.append("INVALID: empty/non-Arabic reply on a dialect-routed turn")
            if r["leaks"]:
                s["leaky"] += 1
                flags.append("LEAK: " + "، ".join(r["leaks"]))
            if r["drift"]:
                s["drifty"] += 1
                flags.append("drift: " + "، ".join(r["drift"]))
            tag = " | ".join(flags) if flags else "clean"
            print(f"  [{group:<28}] {case['id']:<8} {r['elapsed_s']:.1f}s → {r['actual_dialect'] or r['tts_language'] or 'EN/mixed'}  {tag}")
            lines.append(f"### [{group}] {case['id']} — routed {r['actual_dialect'] or r['tts_language'] or 'None'} — "
                         f"{r['elapsed_s']:.1f}s — {tag}")
            lines.append(f"**Q:** {case['text']}\n\n**A:** {r['raw']}\n")

    print(f"\n{'group':<32} {'n':>4} {'leaky':>6} {'drifty':>7} {'invalid':>8} {'route-bad':>10} {'avg-sec':>8}")
    lines.append("\n## Summary\n\n| group | n | leaky | drifty | invalid | routing-bad | avg sec/turn |\n"
                  "|---|---|---|---|---|---|---|")
    for group, s in stats.items():
        avg = s["secs"] / max(s["n"], 1)
        print(f"{group:<32} {s['n']:>4} {s['leaky']:>6} {s['drifty']:>7} {s['invalid']:>8} {s['routing_bad']:>10} {avg:>7.1f}s")
        lines.append(f"| {group} | {s['n']} | {s['leaky']} | {s['drifty']} | {s['invalid']} | {s['routing_bad']} | {avg:.1f} |")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
