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
    .venv/bin/python eval/dialect_eval_full.py --tag <name> --model <ollama-tag>
Requires Ollama on localhost:11434 with the target model already pulled/loaded.
Report: logs/ab_runs/<timestamp>-<tag>.md (gitignored). Record summary numbers in BASELINES.md.

--model support (2026-07-27, added for an offline Fanar-2-27B vs. qwen3.5:27b A/B comparison —
see eval/BASELINES.md): omitted = today's exact unchanged behavior (llm.MODEL, llm.MODEL_CONFIGS
lookup). When passed, uses a neutral, model-agnostic sampling config local to THIS script (never
a new llm.MODEL_CONFIGS key — that dict is hashed by golden_prompts.py's G3 gate, and adding a
key there would require an owner-approved fixture recapture for a change that's purely an
eval-time experiment, not a production one). The alt config also raises num_predict well above
the qwen3.5-tuned default — a candidate model with native <think>...</think> reasoning (Fanar-2
does, on by default, with no clean way to disable it through Ollama's generic hf.co-pull chat
template) would otherwise exhaust its whole budget mid-reasoning before ever answering. Any
<think>...</think> block is stripped from scoring; an OPENED but never CLOSED think tag (reasoning
truncated by num_predict) marks that turn invalid rather than scoring the raw reasoning fragment
as if it were the reply — leak_lint's _AR_WORD_RE would not otherwise catch this, since a native
Arabic reasoning trace is itself well-formed Arabic.
"""

import argparse
import datetime
import json
import os
import re
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

# Used only when --model overrides llm.MODEL — deliberately NOT reusing llm.MODEL_CONFIGS's
# qwen3.5 entry (temp 0.5/presence_penalty 1.5/etc. are reverse-engineered fixes for QWEN's
# specific repetition/fabrication behavior; transplanting them onto a different base model would
# confound "is the candidate model's dialect training better" with "does it happen to like
# qwen's own tuning"). Starts from MODEL_CONFIGS["default"]'s neutral values (never tuned
# against anything) with num_predict raised an order of magnitude — a functional precondition
# for a thinking-by-default model to get past its own reasoning trace, not a quality choice.
_ALT_MODEL_CONFIG = {
    "extra": {},
    "options": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "num_predict": 4000,
        "stop": llm._STOP_SEQUENCES,
    },
}

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)


def strip_think(raw: str) -> tuple[str, bool]:
    """Strip closed <think>...</think> block(s) from `raw`. Returns (cleaned, unclosed) —
    unclosed=True means an opening <think> remains after stripping closed blocks (the
    reasoning trace ran out of num_predict before closing) — caller should treat that turn
    as invalid, not score the truncated fragment as a reply. No-op for text with no <think>
    tag at all (e.g. every qwen3.5 response, which never emits one)."""
    cleaned = _THINK_BLOCK_RE.sub("", raw).strip()
    return cleaned, bool(_THINK_OPEN_RE.search(cleaned))


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


def ask_ollama(client: httpx.Client, messages: list[dict],
               model: str = None, config: dict = None) -> str:
    model = model or llm.MODEL
    cfg = config or llm.get_model_config(model)
    payload = {"model": model, "messages": messages, "stream": False,
               "keep_alive": -1, "options": cfg["options"], **cfg["extra"]}
    r = client.post(OLLAMA_URL, json=payload, timeout=300.0)
    r.raise_for_status()
    msg = r.json().get("message") or {}
    return (msg.get("content", "") or "").strip()


def run_case(client: httpx.Client, case: dict, model: str = None, config: dict = None) -> dict:
    turn_content, tts_language, route_meta = llm.build_turn(case["text"], case["lang"])
    messages = [
        {"role": "system", "content": llm.SYSTEM_PROMPT},
        {"role": "user", "content": turn_content},
    ]
    t0 = time.time()
    raw_response = ask_ollama(client, messages, model=model, config=config)
    elapsed = time.time() - t0
    raw, think_unclosed = strip_think(raw_response)
    actual_dialect = leak_lint.TTS_LANG_TO_DIALECT.get(tts_language or "")
    invalid = think_unclosed or (bool(actual_dialect) and not leak_lint._AR_WORD_RE.search(raw))
    leaks, drift = leak_lint.find_leaks(raw, actual_dialect) if actual_dialect and not invalid else ([], [])
    expected_dialect = leak_lint.TTS_LANG_TO_DIALECT.get(case["expected_tts_language"] or "")
    routing_ok = (case["expected_tts_language"] is None) or (tts_language == case["expected_tts_language"])
    return {"tts_language": tts_language, "actual_dialect": actual_dialect,
            "expected_dialect": expected_dialect, "routing_ok": routing_ok,
            "raw": raw, "invalid": invalid, "think_unclosed": think_unclosed,
            "leaks": leaks, "drift": drift,
            "elapsed_s": elapsed, "route_meta": route_meta}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="", help="suffix for the report filename")
    ap.add_argument("--limit", type=int, default=0, help="only run the first N cases (0 = all)")
    ap.add_argument("--model", default="", help="override llm.MODEL for this run (e.g. an "
                    "hf.co/... Ollama tag) — uses a neutral _ALT_MODEL_CONFIG instead of "
                    "llm.MODEL_CONFIGS's qwen3.5-tuned entry; omit for today's exact behavior")
    args = ap.parse_args()

    model = args.model or None
    config = _ALT_MODEL_CONFIG if args.model else None
    display_model = model or llm.MODEL

    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = os.path.join(os.path.dirname(EVAL_DIR), "logs", "ab_runs")
    os.makedirs(run_dir, exist_ok=True)
    report_path = os.path.join(run_dir, f"{ts}{'-' + args.tag if args.tag else ''}-full.md")

    stats = {}  # group -> counters
    lines = [f"# Full dialect eval run — {ts}{' (' + args.tag + ')' if args.tag else ''}",
             f"Model: {display_model}. {len(cases)} cases. Fresh context per turn (no history).\n"]

    with httpx.Client() as client:
        # Throwaway warm-up call — qwen3.5 runs have always inherited an already-warm model
        # from start_server.sh or a prior eval run; a --model override (or a cold-started
        # qwen3.5) has no such luck, and its first timed question would otherwise silently
        # include several seconds of model-load latency in elapsed_s.
        print(f"  [warmup] {display_model} ...")
        try:
            ask_ollama(client, [{"role": "user", "content": "مرحبا"}], model=model, config=config)
        except Exception as e:
            print(f"  [warmup] failed: {type(e).__name__}: {e} (continuing anyway)")

        for case in cases:
            group = case["group"]
            s = stats.setdefault(group, {"n": 0, "leaky": 0, "drifty": 0, "invalid": 0,
                                          "routing_bad": 0, "secs": 0.0})
            try:
                r = run_case(client, case, model=model, config=config)
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
                if r["think_unclosed"]:
                    flags.append("INVALID: <think> opened but never closed (num_predict exhausted mid-reasoning)")
                else:
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