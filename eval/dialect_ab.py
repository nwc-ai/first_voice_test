"""
dialect_ab.py — like-for-like dialect A/B harness (LLM + prompt layer only; GPU via Ollama)
=============================================================================================
Runs a FIXED question set through the EXACT production prompt surface (llm.build_turn →
llm.SYSTEM_PROMPT → qwen3.5 options; fresh context per turn, no history) and reports, per
reply, the leak_lint findings. Because the questions never change, two runs are directly
comparable — judge any prompt/routing change by running this BEFORE and AFTER.

Ported from chatterbox-tts:eval/dialect_ab.py; rewritten against this branch's llm.py API
(the old harness targeted the server.py monolith). The QUESTIONS list is carried over
VERBATIM from the old branch — do not casually edit it; every edit breaks comparability
with all earlier reports. Dialect-switch/history scenarios deliberately live in a separate
future eval, not here (this harness isolates prompt quality; fresh context per turn).

What it does NOT do: mic/VAD/STT/TTS, history effects, or register judgment (spoken vs
lecture tone stays a by-ear call on the saved report). Linter numbers are the floor, the
owner's ear is the ceiling.

Usage:
    .venv/bin/python eval/dialect_ab.py --tag step0-baseline     # names the report file
    .venv/bin/python eval/dialect_ab.py --dialects Najdi,Fusha   # (default)
Requires Ollama on localhost:11434 with qwen3.5:27b loaded (start_server.sh starts it).
Reports land in logs/ab_runs/<timestamp>[-tag].md (gitignored — data, not code; the repo is
public and reports may echo model output). Record the summary numbers in eval/BASELINES.md.
"""

import argparse
import datetime
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import leak_lint     # noqa: E402
import llm           # noqa: E402
import quality_lint  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/chat"

# How each dialect is requested for the English questions — mirrors the owner's real usage.
# Egyptian entries are present but inert until the Egyptian routing lands (plan Step 4);
# before that an "in Egyptian dialect" request simply routes English/Fusha and the run
# reports it under whatever it routed to.
_REQUEST_SUFFIX = {
    "Najdi":    " Tell me that in Najdi dialect.",
    "Egyptian": " Tell me that in Egyptian dialect.",
    "Fusha":    " Tell me that in Arabic.",
}
# Spoken-Arabic opener per dialect (exercises spoken detection the way live speech does).
_AR_GREETING = {
    "Najdi":    "هلا والله، وش أخبارك اليوم؟",
    "Egyptian": "إزيك؟ عامل إيه النهاردة؟",
    "Fusha":    "كيف حالك اليوم؟",
}

# The FIXED question set (VERBATIM from the old branch — comparability contract).
QUESTIONS = [
    # id, kind, English text (suffix appended per dialect)
    ("greet-city",    "conversational", "I'm visiting your city tomorrow. What should I do first?"),
    ("opinion-tea",   "conversational", "Do you think coffee or tea is better with breakfast?"),
    ("advice-phone",  "conversational", "My phone battery dies really quickly. Any advice?"),
    ("smalltalk-heat","conversational", "It's really hot these days. How do you deal with it?"),
    ("field-tank",    "field",          "The main tank level looks low today. What should I check first?"),
    ("field-meter",   "field",          "A customer says his water meter reading is wrong. What do I do?"),
    ("field-leak",    "field",          "There is water leaking near the station pump. How do I report it?"),
    ("info-coffee",   "informational",  "Tell me about the history of coffee."),
    ("info-sea",      "informational",  "Why does the sea look blue?"),
    ("info-jeddah",   "informational",  "What is special about the old city of Jeddah?"),
    ("anchor-purpose","informational",  "What is the purpose of life according to the Islamic scriptures?"),
    ("short-safe",    "short",          "Is drinking water straight from the tank safe?"),
    ("short-bye",     "short",          "Okay, thank you. That's all for today."),
    ("garbled",       "short",          "Can you explain the, uh, the thing about the..."),
]


def ask_ollama(client: httpx.Client, messages: list[dict]) -> str:
    cfg = llm.get_model_config(llm.MODEL)
    payload = {"model": llm.MODEL, "messages": messages, "stream": False,
               "keep_alive": -1, "options": cfg["options"], **cfg["extra"]}
    r = client.post(OLLAMA_URL, json=payload, timeout=300.0)
    r.raise_for_status()
    msg = r.json().get("message") or {}
    return (msg.get("content", "") or "").strip()


def run_turn(client, text: str, lang: str, judge: bool = False,
             judge_model: str = quality_lint.JUDGE_MODEL) -> dict:
    result = llm.build_turn(text, lang)   # shape-agnostic: 2-tuple pre-Step-1, 3-tuple after
    turn_content, tts_language = result[0], result[1]
    messages = [
        {"role": "system", "content": llm.SYSTEM_PROMPT},
        {"role": "user",   "content": turn_content},
    ]
    t0 = time.time()
    raw = ask_ollama(client, messages)
    elapsed = time.time() - t0
    target = leak_lint.TTS_LANG_TO_DIALECT.get(tts_language or "")
    # A dialect-routed turn whose reply is empty or carries no Arabic at all is INVALID, not
    # "clean" — the documented qwen3.5 empty-response failure mode must not read as an
    # improvement in the leak columns.
    invalid = bool(target) and not leak_lint._AR_WORD_RE.search(raw)
    leaks, drift = leak_lint.find_leaks(raw, target) if target and not invalid else ([], [])
    findings: list[dict] = []
    if judge and target and not invalid:
        findings = quality_lint.judge_reply(text, target, raw, client, model=judge_model)
    return {"tts_language": tts_language, "target": target, "raw": raw, "invalid": invalid,
            "leaks": leaks, "drift": drift, "elapsed_s": elapsed, "findings": findings}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dialects", default="Najdi,Fusha")
    ap.add_argument("--tag", default="", help="suffix for the report filename (e.g. step0-baseline)")
    ap.add_argument("--judge", action="store_true",
                    help="also run the LLM-judge quality pass (soft/informational — see "
                         "quality_lint.py). Stop the live server first. NOTE: qwen3.5:27b (17GB) "
                         "+ the judge model (qwen3:32b, 20GB) do not both fit in 32GB VRAM — "
                         "Ollama will evict/reload one for the other on EVERY turn (confirmed "
                         "working, just slow: expect tens of seconds per turn, not the ~1-3s "
                         "generation-only runs get). Fine for an occasional offline tool, but "
                         "budget real wall-clock time for a full --dialects sweep.")
    ap.add_argument("--judge-model", default=quality_lint.JUDGE_MODEL)
    args = ap.parse_args()
    dialects = [d.strip() for d in args.dialects.split(",") if d.strip()]
    if args.judge:
        quality_lint.check_judge_model_available(args.judge_model)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "logs", "ab_runs")
    os.makedirs(run_dir, exist_ok=True)
    report_path = os.path.join(run_dir, f"{ts}{'-' + args.tag if args.tag else ''}.md")

    stats = {d: {"n": 0, "leaky": 0, "drifty": 0, "invalid": 0, "judged": 0, "secs": 0.0}
            for d in dialects}
    lines = [f"# Dialect A/B run — {ts}{' (' + args.tag + ')' if args.tag else ''}",
             f"Model: {llm.MODEL}. Fresh context per turn (no history).\n"]

    with httpx.Client() as client:
        for d in dialects:
            lines.append(f"\n## {d}\n")
            print(f"\n=== {d} ===")
            turns = [("ar-greeting", "conversational", _AR_GREETING[d], "ar")] + [
                (qid, kind, q + _REQUEST_SUFFIX[d], "en") for qid, kind, q in QUESTIONS]
            for qid, kind, text, lang in turns:
                try:
                    r = run_turn(client, text, lang, judge=args.judge, judge_model=args.judge_model)
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
                print(f"  {qid:<14} [{kind}] {r['elapsed_s']:.1f}s → {r['target'] or r['tts_language'] or 'EN'}"
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
    header = "| dialect | turns | leaky | drifty | invalid | avg sec/turn |" + (" judged |" if args.judge else "")
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
    print("Record the summary table in eval/BASELINES.md (numbers only — reports stay gitignored).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
