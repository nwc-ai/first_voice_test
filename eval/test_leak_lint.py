"""
test_leak_lint.py — regression tests for leak_lint.py detection gaps found by manual review
=============================================================================================
Real leaks/false-positives a full manual re-read of production transcripts found that NO
automated tooling (including the LLM judge) caught:
  - شو (Levantine "what") leaking into a Najdi-routed reply — added to _STRAY.
  - تسوي/يسوي/نسوي/سويت (Najdi "to do/make," already a vetted _NAJDI_MARKERS entry) leaking
    into an Egyptian-routed reply — added as _TSAWWA_RE, a suffix-aware regex (a plain token
    set misses attached-object-pronoun forms like تسويها; see leak_lint.py's comment).
  - دول (2026-07-22): the OPPOSITE problem — a legitimate MSA word ("countries," e.g. "دول
    الخليج") that was wrongly hard-forbidden in Najdi/Fusha, mis-flagging correct Fusha/Najdi
    replies as Egyptian leaks. Removed from `_EGY`. Same shape of mistake as جداً being
    wrongly forbidden in Egyptian — see routing.py's `DIALECT_REPAIR_MAP` correction comment.

Test cases are sourced from REAL transcript text (not synthetic), matching the discipline used
in test_dialect_repair.py.

Run:
    .venv/bin/python eval/test_leak_lint.py     # exit 0 = all pass
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leak_lint  # noqa: E402

FAILURES: list[str] = []


def check(name: str, actual, expected) -> None:
    if actual != expected:
        FAILURES.append(f"FAIL {name}: expected {expected!r}, got {actual!r}")
        print(FAILURES[-1])
    else:
        print(f"  ok {name}")


# ── شو (Levantine "what") — real leak, Najdi info-jeddah/garbled-style reply ──────────────

_shu_leaks, _ = leak_lint.find_leaks(
    "ما أدري شو هي الشيء اللي تقصده بالضبط لأنك ما ذكرته", "Najdi")
check("shu-leaks-into-najdi", "شو" in _shu_leaks, True)

_shu_egy_leaks, _ = leak_lint.find_leaks("شو رأيك في الموضوع ده", "Egyptian")
check("shu-leaks-into-egyptian", "شو" in _shu_egy_leaks, True)

# ── تسوي family — real leak, Egyptian greet-city reply ("لازم تسويها دلوقتي") ─────────────

_tsawwa_leaks, _ = leak_lint.find_leaks(
    "أول حاجة لازم تسويها دلوقتي إنك تخطط لجدول زياراتك", "Egyptian")
check("tsawwa-suffixed-leaks-into-egyptian",
      any(l.startswith("سوى-verb:") for l in _tsawwa_leaks), True)

_tsawwa_bare_leaks, _ = leak_lint.find_leaks("يجب أن تسوي وضعك القانوني", "Fusha")
check("tsawwa-bare-leaks-into-fusha",
      any(l.startswith("سوى-verb:") for l in _tsawwa_bare_leaks), True)

_tsawwa_conjugations, _ = leak_lint.find_leaks(
    "عايز يسويها بنفسه وإحنا نسويها مع بعض وأنا سويتها امبارح", "Egyptian")
check("tsawwa-all-conjugations-caught", len(_tsawwa_conjugations), 3)

# ── False-positive safety: تسويق (marketing, root سوّق) must never match ──────────────────

_tsawwiq_leaks, _ = leak_lint.find_leaks("الشركة دي بتعمل تسويق للمنتجات الجديدة", "Egyptian")
check("no-match-taswiq", any(l.startswith("سوى-verb:") for l in _tsawwiq_leaks), False)

_tsawwiq_fusha_leaks, _ = leak_lint.find_leaks("التسويق الرقمي مهم جداً للشركات", "Fusha")
check("no-match-taswiq-fusha", any(l.startswith("سوى-verb:") for l in _tsawwiq_fusha_leaks), False)

# ── سوى family stays allowed in its home dialect (Najdi) ──────────────────────────────────

_najdi_ok_leaks, _ = leak_lint.find_leaks("لازم تسوي هالشي بنفسك وتشوف النتيجة", "Najdi")
check("tsawwa-allowed-in-najdi",
      any(l.startswith("سوى-verb:") for l in _najdi_ok_leaks), False)

# ── دول false-positive fix (2026-07-22): legitimate MSA "countries" must not leak-flag ─────

_dowal_najdi_leaks, _ = leak_lint.find_leaks("لازم نتعاون مع دول الخليج في هذا الموضوع", "Najdi")
check("dowal-khaleej-not-a-leak-najdi", "دول" in _dowal_najdi_leaks, False)

_dowal_fusha_leaks, _ = leak_lint.find_leaks("العالم اليوم فيه دول كثيرة تواجه هذا التحدي", "Fusha")
check("dowal-aalam-not-a-leak-fusha", "دول" in _dowal_fusha_leaks, False)

# ── summary ───────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(" ", f)
    sys.exit(1)
print("ALL LEAK-LINT REGRESSION TESTS PASSED")
