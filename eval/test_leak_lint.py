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

# ── مرة false-positive fix (2026-07-27): another جداً/دول-shaped homograph mistake ─────────
# Checked all 10 real occurrences flagged across the 2026-07-22/27 full-eval runs
# (eval/BASELINES.md) — every single one was the "time/occurrence" sense (أول مرة، كل مرة،
# مرة واحدة، مرة ثانية), never the genuine Gulf/Levantine "very" leak. Real sentences below.

_marra_awwal, _ = leak_lint.find_leaks(
    "لما يشوفوها يقفزوا عليها ويمسكوها من أول مرة", "Egyptian")
check("marra-awwal-marra-not-a-leak", "مرة" in _marra_awwal, False)

_marra_kul, _ = leak_lint.find_leaks(
    "لازم تركز على مهمة واحدة كل مرة بدل ما تتشتت", "Egyptian")
check("marra-kul-marra-not-a-leak", "مرة" in _marra_kul, False)

_marra_kul_glued, _ = leak_lint.find_leaks(
    "فكل مرة تقابل فيها جاراك وتسلم عليه بإخاء", "Egyptian")
check("marra-glued-prefix-kul-marra-not-a-leak", "مرة" in _marra_kul_glued, False)

_marra_wahda, _ = leak_lint.find_leaks(
    "بدل ما تحط هدف كبير مرة واحدة", "Egyptian")
check("marra-wahda-not-a-leak", "مرة" in _marra_wahda, False)

_marra_tania, _ = leak_lint.find_leaks(
    "جرب تعمل فحص عشان يترتب الضغط مرة ثانية ويوصل للمستوى الطبيعي", "Egyptian")
check("marra-tania-not-a-leak", "مرة" in _marra_tania, False)

# ── مرة true-positive: the genuine Gulf/Levantine "very" leak must still be caught ─────────

_marra_intensifier, _ = leak_lint.find_leaks("الأكل ده حلو مرة والزحمة كانت مرة النهاردة", "Egyptian")
check("marra-intensifier-still-a-leak", "مرة" in _marra_intensifier, True)

# مرة is Najdi's CORRECT word for "very" (DIALECT_REPAIR_MAP: جداً→مرة) — never forbidden there.
_marra_najdi_ok, _ = leak_lint.find_leaks("الجو حار مرة اليوم", "Najdi")
check("marra-intensifier-allowed-in-najdi", "مرة" in _marra_najdi_ok, False)

# ── summary ───────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(" ", f)
    sys.exit(1)
print("ALL LEAK-LINT REGRESSION TESTS PASSED")
