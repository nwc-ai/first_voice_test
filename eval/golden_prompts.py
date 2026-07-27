"""
golden_prompts.py — the byte-identity gate (G1 routing replay + G2 golden prompt bytes +
G3 STT-config freeze; no GPU, no model loads, ~seconds)
=============================================================================================
THE load-bearing non-regression gate for the Egyptian reintroduction (2026-07-20 plan).

Design axiom it enforces: the LLM is stateless per request, so Najdi/Fusha behavior cannot
regress while the request bytes on Najdi/Fusha-routed turns are byte-identical to the frozen
baseline. This script freezes those bytes.

  capture mode   .venv/bin/python eval/golden_prompts.py --capture
                 Runs every case in golden_prompt_cases.jsonl through llm.build_turn and
                 writes fixtures (tts_language + full turn_content + config hashes) to
                 golden_fixtures.jsonl. Run ONCE on the frozen baseline (HEAD e0faf6c) and
                 commit the result. Re-capturing on modified code defeats the gate — only
                 re-capture when the OWNER approves a deliberate baseline change.

  compare mode   .venv/bin/python eval/golden_prompts.py
                 Recomputes every case and enforces:
                   G1 — routing decision (tts_language) identical to the fixture for every
                        case, EXCEPT cases marked "may_move": true, which are allowed exactly
                        one change: to their "expected_v1" value ("egyptian arabic"). Any
                        other change — especially any change on a Najdi-routed case — fails.
                   G2 — for every case whose (new) route is NOT "egyptian arabic", the full
                        turn_content must be BYTE-IDENTICAL to the fixture. Moved cases get
                        new (Egyptian) bytes by design and are exempted from byte equality
                        but listed in the MOVED report.
                   G3 — SYSTEM_PROMPT, MODEL_CONFIGS, MAX_HISTORY_TURNS, MODEL, the Whisper
                        Arabic initial_prompt + transcribe kwargs, and the TTS CATT-language
                        set hash-match the fixture (the shared surfaces frozen by policy).
                 Exit 0 = all gates green. Non-zero = regression, details on stdout.

Environment note: run with default env (no LLM_NUM_CTX / CATT_ENABLED overrides) — those
change the captured config surface and will be reported as a G3 mismatch.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm      # noqa: E402
import routing  # noqa: E402
import stt      # noqa: E402  (imports torch — slow but loads no models)
import tts_omnivoice_v1 as tts  # noqa: E402

_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_PATH    = os.path.join(_DIR, "golden_prompt_cases.jsonl")
FIXTURES_PATH = os.path.join(_DIR, "golden_fixtures.jsonl")

ALLOWED_MOVE_TARGET = "egyptian arabic"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def config_surface() -> dict[str, str]:
    """Hashes of every shared surface frozen by the plan (G3)."""
    return {
        "SYSTEM_PROMPT":        _sha(llm.SYSTEM_PROMPT),
        "MODEL":                llm.MODEL,
        "MODEL_CONFIGS":        _sha(json.dumps(llm.MODEL_CONFIGS, sort_keys=True, ensure_ascii=False)),
        "MAX_HISTORY_TURNS":    str(llm.MAX_HISTORY_TURNS),
        "AR_INITIAL_PROMPT":    _sha(stt._AR_INITIAL_PROMPT),
        "TRANSCRIBE_KWARGS":    _sha(json.dumps(stt._TRANSCRIBE_KWARGS, sort_keys=True, ensure_ascii=False)),
        "TASHKEEL_LANGUAGES":   _sha(json.dumps(sorted(tts._TASHKEEL_LANGUAGES))),
        "TTS_SENTENCE_CONSTS":  _sha(json.dumps([sorted(tts.HARD_BREAK), sorted(tts.SOFT_BREAK),
                                                 tts.SOFT_BREAK_MIN, tts.FIRST_SOFT_MIN])),
    }


def run_case(text: str, lang: str) -> tuple[str, str | None]:
    with contextlib.redirect_stdout(io.StringIO()):   # silence build_turn debug prints
        result = llm.build_turn(text, lang)
    # Shape-agnostic: (turn_content, tts_language) at Step 0; +route_meta from Step 1 on.
    return result[0], result[1]


def load_cases() -> list[dict]:
    with open(CASES_PATH, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def capture() -> int:
    cases = load_cases()
    with open(FIXTURES_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": config_surface()}, ensure_ascii=False) + "\n")
        for c in cases:
            turn_content, tts_language = run_case(c["text"], c["lang"])
            f.write(json.dumps({"id": c["id"], "tts_language": tts_language,
                                "sha256": _sha(turn_content), "turn_content": turn_content},
                               ensure_ascii=False) + "\n")
    print(f"Captured {len(cases)} fixtures → {FIXTURES_PATH}")
    print("Commit this file. Do NOT re-capture without owner approval.")
    return 0


def compare() -> int:
    cases = {c["id"]: c for c in load_cases()}
    meta: dict = {}
    fixtures: dict[str, dict] = {}
    with open(FIXTURES_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "_meta" in row:
                meta = row["_meta"]
            else:
                fixtures[row["id"]] = row

    failures: list[str] = []
    case_fail_ids: set[str] = set()
    moved:    list[str] = []
    pending:  list[str] = []   # may_move cases still on their old route (pre-Step-4 state)

    # G3 — shared-surface freeze
    now = config_surface()
    for key, want in meta.items():
        if now.get(key) != want:
            failures.append(f"G3 {key}: shared surface changed (hash {want[:12]}… → {str(now.get(key))[:12]}…)")

    # G1 + G2 — per-case routing + bytes
    if set(cases) != set(fixtures):
        only_cases = set(cases) - set(fixtures)
        only_fix   = set(fixtures) - set(cases)
        if only_cases:
            failures.append(f"cases without fixtures (re-run --capture on the BASELINE): {sorted(only_cases)}")
        if only_fix:
            failures.append(f"fixtures without cases: {sorted(only_fix)}")

    def _first_diff(a: str, b: str) -> str:
        i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        lo = max(0, i - 30)
        return f"@{i}: fixture …{a[lo:i + 50]!r} vs now …{b[lo:i + 50]!r}"

    for cid in sorted(set(cases) & set(fixtures)):
        c, fx = cases[cid], fixtures[cid]
        turn_content, tts_language = run_case(c["text"], c["lang"])
        if tts_language == fx["tts_language"]:
            if turn_content != fx["turn_content"]:
                failures.append(f"G2 {cid}: same route ({tts_language}) but turn_content bytes CHANGED "
                                f"{_first_diff(fx['turn_content'], turn_content)}")
                case_fail_ids.add(cid)
            elif c.get("may_move"):
                pending.append(cid)
        elif (c.get("may_move") and tts_language == c.get("expected_v1") == ALLOWED_MOVE_TARGET):
            moved.append(cid)   # allowed move — Egyptian bytes are new by design
        else:
            failures.append(f"G1 {cid}: route changed {fx['tts_language']!r} → {tts_language!r}"
                            f"{'' if c.get('may_move') else ' (NOT in the allowed moved set)'}")
            case_fail_ids.add(cid)

    print(f"\n{len(cases)} cases: {len(cases) - len(moved) - len(pending) - len(case_fail_ids)} frozen-ok, "
          f"{len(moved)} moved→egyptian (allowed), {len(pending)} may-move-pending, "
          f"{len(case_fail_ids)} failing")
    if moved:
        print("  MOVED (allowed, enumerated):", ", ".join(moved))
    if pending:
        print("  may-move pending (still on old route):", ", ".join(pending))
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f_ in failures:
            print(" ", f_)
        return 1
    print("GOLDEN GATES G1/G2/G3 GREEN")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", action="store_true", help="write fixtures from the current code (BASELINE ONLY)")
    args = ap.parse_args()
    sys.exit(capture() if args.capture else compare())
