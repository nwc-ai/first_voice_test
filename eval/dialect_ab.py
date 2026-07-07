"""
dialect_ab.py — like-for-like dialect A/B harness (LLM + prompt layer only; GPU via Ollama)
=============================================================================================
Runs a FIXED set of questions through the EXACT production prompt surface (server._route_turn
→ server._build_turn_content → SYSTEM_PROMPT → qwen3.5 options) for each dialect, and reports
per reply: purity-linter leaks/drift AND what the server's auto-fixups would change. Because
the questions never change, two runs are directly comparable — judge any prompt tweak by
running this BEFORE and AFTER instead of comparing memories across live sessions.

What it does NOT do: mic/VAD/STT/TTS (those have their own evals), history effects (every turn
is sent fresh, no rolling context — isolates prompt quality), or *register* judgment — whether
a reply sounds spoken vs lecture-like stays a by-ear call on the saved report. The linter
numbers are the floor, your ear is the ceiling.

Usage:
    .venv/bin/python eval/dialect_ab.py                          # full run (~60 turns, 3-5 min)
    .venv/bin/python eval/dialect_ab.py --tag before-register    # names the report file
    .venv/bin/python eval/dialect_ab.py --dialects Najdi,Egyptian
Requires Ollama running on localhost:11434 with the pinned model (start_server.sh starts it).
Reports land in eval/ab_runs/<timestamp>[-tag].md (gitignored — they are data, not code).
"""

import argparse
import datetime
import json
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dialect_purity_lint as lint  # noqa: E402
import server                       # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "qwen3.5:27b"

# How each dialect is requested for the English questions — mirrors the owner's real usage
# ("… in Najdi dialect"). Fusha uses the generic-Arabic request (routes explicit_arabic→Fusha).
_REQUEST_SUFFIX = {
    "Najdi":    " Tell me that in Najdi dialect.",
    "Hijazi":   " Tell me that in Hijazi dialect.",
    "Egyptian": " Tell me that in Egyptian dialect.",
    "Fusha":    " Tell me that in Arabic.",
}

# Spoken-Arabic opener per dialect (exercises _detect_dialect the way live speech does).
_AR_GREETING = {
    "Najdi":    "هلا والله، وش أخبارك اليوم؟",
    "Hijazi":   "إيش أخبارك اليوم؟ أبي أسولف معك شوية",
    "Egyptian": "إزيك؟ عامل إيه النهاردة؟",
    "Fusha":    "كيف حالك اليوم؟",
}

# The FIXED question set. Do not casually edit — every edit breaks comparability with all
# earlier reports. Conversational + field questions are where dialect should shine;
# informational ones are the register stress test (models drift to MSA on lecture content).
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
    cfg = server._get_model_config(MODEL)
    payload = {"model": MODEL, "messages": messages, "stream": False,
               "options": cfg["options"], **cfg["extra"]}
    r = client.post(OLLAMA_URL, json=payload, timeout=180.0)
    r.raise_for_status()
    return (r.json().get("message") or {}).get("content", "").strip()


def run_turn(client, text: str, lang: str) -> dict:
    route = server._route_turn(text, lang)
    messages = [
        {"role": "system", "content": server.SYSTEM_PROMPT},
        {"role": "user",   "content": server._build_turn_content(text, route)},
    ]
    raw = ask_ollama(client, messages)
    target = lint._TTS_LANG_TO_DIALECT.get(route["tts_language"] or "")
    leaks, drift = lint.find_leaks(raw, target) if target else ([], [])
    fixups: list[str] = []
    delivered = server._apply_fixups(raw, route["tts_language"] if not route["translation_q"] else None, fixups)
    return {"route": route["route"], "target": target, "raw": raw, "delivered": delivered,
            "leaks": leaks, "drift": drift, "fixups": fixups}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dialects", default="Najdi,Hijazi,Egyptian,Fusha")
    ap.add_argument("--tag", default="", help="suffix for the report filename (e.g. before-register)")
    args = ap.parse_args()
    dialects = [d.strip() for d in args.dialects.split(",") if d.strip()]

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ab_runs")
    os.makedirs(run_dir, exist_ok=True)
    report_path = os.path.join(run_dir, f"{ts}{'-' + args.tag if args.tag else ''}.md")

    stats = {d: {"n": 0, "leaky": 0, "drifty": 0, "fixed": 0} for d in dialects}
    lines = [f"# Dialect A/B run — {ts}{' (' + args.tag + ')' if args.tag else ''}",
             f"Model: {MODEL}. Fresh context per turn (no history). "
             f"'delivered' = after server auto-fixups.\n"]

    with httpx.Client() as client:
        for d in dialects:
            lines.append(f"\n## {d}\n")
            print(f"\n=== {d} ===")
            turns = [("ar-greeting", "conversational", _AR_GREETING[d], "ar")] + [
                (qid, kind, q + _REQUEST_SUFFIX[d], "en") for qid, kind, q in QUESTIONS]
            for qid, kind, text, lang in turns:
                try:
                    r = run_turn(client, text, lang)
                except Exception as e:
                    print(f"  {qid}: ERROR {type(e).__name__}: {e}")
                    lines.append(f"### {qid} — ERROR: {e}\n")
                    continue
                s = stats[d]
                s["n"] += 1
                flags = []
                if r["leaks"]:  s["leaky"]  += 1; flags.append("LEAK: "  + "، ".join(r["leaks"]))
                if r["drift"]:  s["drifty"] += 1; flags.append("drift: " + "، ".join(r["drift"]))
                if r["fixups"]: s["fixed"]  += 1; flags.append("would-fix: " + "، ".join(r["fixups"]))
                print(f"  {qid:<14} [{kind}] {' | '.join(flags) if flags else 'clean'}")
                lines.append(f"### {qid} [{kind}] — {' | '.join(flags) if flags else 'clean'}")
                lines.append(f"**Q:** {text}\n\n**A (raw):** {r['raw']}\n")
                if r["fixups"]:
                    lines.append(f"**A (delivered):** {r['delivered']}\n")

    print(f"\n{'dialect':>9}  {'turns':>5}  {'leaky':>5}  {'drifty':>6}  {'would-fix':>9}")
    lines.append("\n## Summary\n\n| dialect | turns | leaky | drifty | would-fix |\n|---|---|---|---|---|")
    for d in dialects:
        s = stats[d]
        print(f"{d:>9}  {s['n']:>5}  {s['leaky']:>5}  {s['drifty']:>6}  {s['fixed']:>9}")
        lines.append(f"| {d} | {s['n']} | {s['leaky']} | {s['drifty']} | {s['fixed']} |")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
