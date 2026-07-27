"""
test_dialect_repair.py — deterministic dialect-repair regression suite (no GPU, ~seconds)
=============================================================================================
Pins routing.apply_dialect_repairs()/DIALECT_REPAIR_MAP — the generalized replacement for the
old Egyptian-only fix_egyptian_leaks(). Test cases are sourced from REAL transcript text pulled
from logs/ab_runs/ (not synthetic examples), matching the same discipline used to validate the
original جداً→أوي fixup.

Also enforces two anti-drift invariants (see routing.py's DIALECT_REPAIR_MAP docstring and
eval/README.md's feedback-loop checklist):
  - prose-sync:      every repair-map wrong-word must appear in the matching dialect's prompt
                     card (NAJDI_GLOSSARY/EGYPTIAN_CARD) — the dictionary can never drift ahead
                     of what the prompt actually teaches.
  - forbidden-sync:  every repair-map wrong-word must already be in leak_lint.FORBIDDEN for
                     that dialect — never auto-correct a token leak_lint doesn't even flag.

Run:
    .venv/bin/python eval/test_dialect_repair.py     # exit 0 = all pass
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import leak_lint  # noqa: E402
import routing    # noqa: E402

FAILURES: list[str] = []


def check(name: str, actual, expected) -> None:
    if actual != expected:
        FAILURES.append(f"FAIL {name}: expected {expected!r}, got {actual!r}")
        print(FAILURES[-1])
    else:
        print(f"  ok {name}")


# ── True-positive substitutions — real leaks pulled from logs/ab_runs/ ────────────────────

check(
    "najdi-tp-info-jeddah",
    routing.apply_dialect_repairs(
        "مدينة جدة القديمة فيها شي مميز جداً لأنها تعتبر أول موقع تراثي في المملكة العربية "
        "السعودية تم تسجيله من قبل اليونسكو",
        "Najdi"),
    "مدينة جدة القديمة فيها شي مميز مرة لأنها تعتبر أول موقع تراثي في المملكة العربية "
    "السعودية تم تسجيله من قبل اليونسكو",
)

check(
    "egyptian-tp-alathi",
    routing.apply_dialect_repairs(
        "إن الفول هو حب البقول الذي بيقشر ويغلى ويتبل بالزيت والثوم والبصل", "Egyptian"),
    "إن الفول هو حب البقول اللي بيقشر ويغلى ويتبل بالزيت والثوم والبصل",
)
check("egyptian-tp-tamsha", routing.apply_dialect_repairs("لازم تمشى بالخطة", "Egyptian"),
      "لازم تمشي بالخطة")
check("egyptian-tp-takul", routing.apply_dialect_repairs("تأكل الأكل ده كله؟", "Egyptian"),
      "تاكل الأكل ده كله؟")

# ── الذي/اللذين collision safety (dual-form must never get truncated to اللي) ─────────────

check(
    "no-match-alladhayn",
    routing.apply_dialect_repairs("الطالبان اللذين نجحا اتكرموا", "Egyptian"),
    "الطالبان اللذين نجحا اتكرموا",
)

# ── Proper-noun / prefix-collision safety (جدول/جدال/جدة never confused with جداً) ─────────

check("no-match-jadwal", routing.apply_dialect_repairs("هتلاقي جدول أعمال اليوم", "Egyptian"),
      "هتلاقي جدول أعمال اليوم")
check("no-match-jidal", routing.apply_dialect_repairs("في جدال كبير حول الموضوع", "Egyptian"),
      "في جدال كبير حول الموضوع")
check(
    "najdi-jeddah-not-jiddan",
    "جدة" in routing.apply_dialect_repairs("مدينة جدة القديمة فيها شي مميز جداً", "Najdi"),
    True,
)

# ── No-op cases: Fusha and Egyptian (جداً is correct in both — see the 2026-07-22 ─────────
# ── correction: Egyptian was wrongly forcing جداً→أوي for two prior sessions) and
# ── unrecognized/None dialect ─────────────────────────────────────────────────────────────

check("fusha-noop", routing.apply_dialect_repairs("الجو حار جداً اليوم", "Fusha"),
      "الجو حار جداً اليوم")
check("egyptian-jiddan-noop",
      routing.apply_dialect_repairs(
          "القهوة دي لعبت دور مهم جداً في الحياة الاجتماعية والثقافية المصرية لحد دلوقتي",
          "Egyptian"),
      "القهوة دي لعبت دور مهم جداً في الحياة الاجتماعية والثقافية المصرية لحد دلوقتي")
check("none-dialect-noop", routing.apply_dialect_repairs("الجو حار جداً اليوم", None),
      "الجو حار جداً اليوم")

# ── Anti-drift invariants ───────────────────────────────────────────────────────────────

_CARDS = {"Najdi": routing.NAJDI_GLOSSARY, "Egyptian": routing.EGYPTIAN_CARD}
for _dialect, _pairs in routing.DIALECT_REPAIR_MAP.items():
    for _wrong in _pairs:
        check(f"prose-sync-{_dialect}-{_wrong}", _wrong in _CARDS[_dialect], True)
        check(f"forbidden-sync-{_dialect}-{_wrong}", _wrong in leak_lint.FORBIDDEN[_dialect], True)

# ── summary ───────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(" ", f)
    sys.exit(1)
print("ALL DIALECT-REPAIR TESTS PASSED")
