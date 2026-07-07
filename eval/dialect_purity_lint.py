"""
dialect_purity_lint.py — glossary-based cross-dialect leak detector (no GPU, runs in seconds)
==============================================================================================
Scans the Arabic assistant replies in logs/interactions.jsonl and reports, per routed dialect,
tokens that must NOT appear there — e.g. the Egyptian هـ-future (هخبرك/هتكون) or دلوقتي inside
a Najdi reply, or الحين/وش inside an Egyptian reply. Built from the user's cross-dialect
glossary (2026-07-06). This turns "the dialects are mixing" into a NUMBER to watch after every
prompt/model change.

Usage:
    .venv/bin/python eval/dialect_purity_lint.py                     # whole log
    .venv/bin/python eval/dialect_purity_lint.py --since 2026-07-06  # from a date
    .venv/bin/python eval/dialect_purity_lint.py path/to/log.jsonl --since 2026-07-07

Three severity levels:
  LEAK       — a token from ANOTHER dialect (misroutes voice/pronunciation identity), OR جداً
               inside a dialect reply (promoted from soft drift 2026-07-07: the owner's cards
               say NEVER جداً — Najdi/Hijazi=مرة, Egyptian=أوي). Hard error.
  MSA-drift  — MSA function words (حيث، مليء، لذا…; كيف for Egyptian only) inside a dialect
               reply. Soft: an educated register is acceptable, but rising drift = the model
               is slipping back to Fusha.
  auto-fixed — the server's _DIALECT_FIXUPS swapped a wrong word BEFORE delivery (llm.fixups
               log field, 2026-07-07). Delivered text is clean, but the count measures what
               the model ATTEMPTED — the honest model-quality signal now that fixups exist.

The target dialect comes from the logged `route` block; rows logged before route data existed
are reconstructed through the live server._route_turn(). English/mixed turns are skipped
(mixed replies are deliberately un-pinned).

NOTE on precision: sets contain only HIGH-PRECISION tokens (a word shared across dialects is
never flagged). The report prints every offending token so a human can veto false positives —
e.g. زين appearing as the name "Zain". The هـ-future regex has a whitelist for lookalikes
(هناك، هيئة…); anything it flags is shown with the matched word.

KNOWN BLIND SPOT (not lintable): the Egyptian بـ-present (بيفتخروا، بتنقل) inside a Najdi
reply cannot be flagged lexically — بـ + imperfect is also the legitimate Najdi FUTURE
(بيكون = "it will be"): the same surface form is correct in one meaning, wrong in the other.
The dialect card forbids it; judging it stays by ear.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server  # noqa: E402  (for _route_turn reconstruction on old rows)

# ── Token groups (from the user's glossary; high-precision distinctive words only) ─────────
# ده/دي/مش/فين added 2026-07-07: the live «والعبادة دي مش بس الصلاة» Najdi reply passed the
# lint unflagged. They are weak markers for DETECTION (shared with urban Hijazi speech), but
# the owner's cards forbid them outright in Najdi AND Hijazi replies — the linter measures
# card compliance, so they are hard leaks here. Veto by eye if one shows up in a quote.
_EGY = {"دلوقتي", "دلوقت", "كده", "كدا", "النهاردة", "إمبارح", "امبارح", "إزاي", "ازاي",
        "إزيك", "ازيك", "عايز", "عاوز", "عايزة", "مفيش", "معنديش", "متشكر", "أوي",
        "بتاع", "بتاعة", "دول", "كتير", "كويس", "ده", "دي", "مش", "فين",
        # Water-utility domain rows (owner's glossary, 2026-07-07). Only the high-precision
        # Egyptian-only forms; deliberately NOT added: مليان/خربان (cross-dialect colloquial
        # use), عاطل (Egyptian homograph "unemployed"), رشح (MSA homograph "nominate/filter/
        # a cold"), قطع (ubiquitous MSA root) — low precision, human-ear territory.
        "بايظ", "قراية",
        # Glued ـش-negation verbs (Egyptian-native, leak anywhere else — the bare-regex case
        # is handled by _SHIN_NEG_RE below):
        "معرفش", "مكانش", "ماكانش", "منفعش"}
# Najdi + Gulf-adjacent words. Gulf is NO LONGER a routed dialect (removed 2026-07-07,
# owner decision) but these tokens stay forbidden inside Egyptian/Fusha replies.
_NAJDI_GULF = {"وش", "أبغى", "ابغى", "الحين", "يبيلك", "صج", "وايد", "شنو", "دحين"}
_HIJAZI_ONLY = {"إيش", "ايش", "دحين"}          # per user's directive: Najdi says وش, never إيش
# جداً was PROMOTED out of soft drift (2026-07-07): every card says NEVER جداً, and it was the
# single most frequent violation (20/33 Egyptian replies on 2026-07-06). Tokenizer note: the
# [ء-ي]+ tokenizer strips the tanween, so جداً surfaces as جدا.
_JIDDAN = {"جدا", "جداً"}
_MSA_DRIFT = {"حيث", "مليء", "مليئة", "بعيدا", "بعيداً", "لذا", "كذلك"}
_EGY_DRIFT_EXTRA = {"كيف"}   # Egyptian wants إزاي; كيف is native in Najdi/Hijazi/Gulf

# Egyptian هـ-future — forbidden outside Egyptian replies. Two shapes:
#  2nd/3rd person: ه + ي/ت/ن + stem (هيكون، هتكون، هنروح) — regex + lookalike whitelist.
#  1st person: ه attaches straight to the verb (هقولك، هخبرك، هعمل) — regex would swallow
#  ordinary nouns (هجوم، هدوء), so these are a curated list of common forms.
_HA_FUTURE_RE = re.compile(r"\bه[يتن][ء-ي]{2,}\b")
_HA_FUTURE_OK = {"هناك", "هنالك", "هيئة", "هيئات", "هيبة", "هيكل", "هياكل", "هيمنة",
                 "هيه", "هيا", "هتاف", "هتف", "هند", "هندي", "هندية", "هنود", "هندسة",
                 "هنيئا", "هنيئاً",
                 "هينزلا"}   # Whisper's garble of "Hunza" (وادي هينزلا) — live FP 2026-07-06
_HA_FUTURE_1P = {"هقول", "هقولك", "هقوللك", "هقولكم", "هعمل", "هعمله", "هعملها", "هروح",
                 "هاروح", "هشوف", "هشوفك", "هاخد", "هاخده", "هخبرك", "هخبركم", "هبقى",
                 "هكون", "هقدر", "هلاقي", "هعرف", "هعرفك", "هجيب", "هجيبلك", "هحاول",
                 "هكلمك", "هبعت", "هحكي", "هحكيلك", "هفكر", "هبدأ", "هحط", "هديك", "هساعدك"}

# What is forbidden INSIDE a reply routed to each dialect. حاجة is Najdi-only forbidden
# (Najdi=شي; Hijazi uses حاجة natively). _JIDDAN is forbidden in every dialect but NOT Fusha,
# where جداً is correct MSA.
FORBIDDEN: dict[str, set] = {
    "Najdi":    _EGY | _HIJAZI_ONLY | _JIDDAN | {"وايد", "شنو", "كمان", "بدي", "حاجة"},
    "Hijazi":   _EGY | _JIDDAN | {"شنو", "يبيلك", "صج"},
    # وين added 2026-07-07 (live: an Egyptian joke recycled from a Hijazi one kept «وين الفول»;
    # Egyptian says فين). راح is deliberately NOT here — it is also valid Egyptian past "went"
    # (راح البيت), and راح-future vs راح-went can't be separated lexically; the Egyptian card
    # forbids راح-future prompt-side instead.
    "Egyptian": _NAJDI_GULF | _HIJAZI_ONLY | _JIDDAN | {"مشكور", "زين", "وين"},
    # Fusha must contain no dialect function words at all:
    "Fusha":    (_EGY | _NAJDI_GULF | _HIJAZI_ONLY
                 | {"مش", "ده", "دي", "كمان", "بدي", "يلا", "معليش"}),
}
_HA_FORBIDDEN_IN = {"Najdi", "Hijazi", "Fusha"}   # هـ-future allowed only in Egyptian

# Egyptian ـش-negation as a PATTERN (ما + verb + ش: «ما نعرفش») — caught live in a Najdi reply
# 2026-07-07 where the glued list missed it. Same dialect set as the هـ-future gate; the rare
# MSA lookalike (a noun ending in ش after ما) is human-veto territory like everything else.
_SHIN_NEG_RE = re.compile(r"\bما\s+[ء-ي]{2,}ش\b")

_AR_WORD_RE = re.compile(r"[ء-ي]+")
# NOTE: "gulf arabic" is intentionally absent (Gulf removed 2026-07-07) — any old log rows
# routed to it are skipped rather than scored against a dialect that no longer exists.
_TTS_LANG_TO_DIALECT = {"najdi arabic": "Najdi", "hijazi arabic": "Hijazi",
                        "egyptian arabic": "Egyptian", "standard arabic": "Fusha"}


def find_leaks(text: str, dialect: str) -> tuple[list[str], list[str]]:
    """Return (leaks, msa_drift) for a reply routed to `dialect`.
    leaks = cross-dialect tokens + هـ-future markers; msa_drift = soft MSA function words."""
    words = set(_AR_WORD_RE.findall(text))
    # Arabic glues the conjunctions و/ف onto the next word («وشنو», «ودلوقتي») — also test
    # the prefix-stripped form so glued leaks are still caught.
    words |= {w[1:] for w in words if len(w) > 3 and w[0] in "وف"}
    leaks = sorted(words & FORBIDDEN.get(dialect, set()))
    if dialect in _HA_FORBIDDEN_IN:
        for m in sorted(words & _HA_FUTURE_1P):
            leaks.append(f"هـ-future:{m}")
        for m in _HA_FUTURE_RE.findall(text):
            if m not in _HA_FUTURE_OK:
                leaks.append(f"هـ-future:{m}")
        for m in _SHIN_NEG_RE.findall(text):
            leaks.append(f"ش-negation:{m}")
    drift_set = _MSA_DRIFT | (_EGY_DRIFT_EXTRA if dialect == "Egyptian" else set())
    drift = sorted(words & drift_set) if dialect in ("Najdi", "Hijazi", "Egyptian") else []
    return leaks, drift


def target_dialect(row: dict) -> str | None:
    """Routed dialect of a logged turn; None = not a pinned-dialect Arabic reply (skip)."""
    r = row.get("route")
    if r is None:                              # row predates route logging — reconstruct
        try:
            r = server._route_turn(row.get("transcript", ""), row.get("lang", "en"))
        except Exception:
            return None
    return _TTS_LANG_TO_DIALECT.get(r.get("tts_language") or "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logfile", nargs="?",
                    default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "logs", "interactions.jsonl"))
    ap.add_argument("--since", default="", help="only rows whose ts starts >= this prefix (e.g. 2026-07-06)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.logfile, encoding="utf-8") if l.strip()]
    if args.since:
        rows = [r for r in rows if r.get("ts", "") >= args.since]

    stats: dict[str, dict] = {}
    for row in rows:
        reply = row.get("response", "")
        if row.get("cancelled") or not _AR_WORD_RE.search(reply):
            continue
        d = target_dialect(row)
        if d is None:
            continue
        leaks, drift = find_leaks(reply, d)
        # Server-side auto-fixups (2026-07-07): the delivered reply is clean, but the model
        # WROTE the wrong word — count it so fixups don't hide model regressions.
        fixups = (row.get("llm") or {}).get("fixups") or []
        s = stats.setdefault(d, {"n": 0, "leaky": 0, "leak_tokens": 0, "drifty": 0,
                                 "fixed": 0, "rows": []})
        s["n"] += 1
        if leaks:
            s["leaky"] += 1
            s["leak_tokens"] += len(leaks)
        if drift:
            s["drifty"] += 1
        if fixups:
            s["fixed"] += 1
        if leaks or drift or fixups:
            s["rows"].append((row.get("ts", "?"), leaks, drift, fixups,
                              row.get("transcript", "")[:60]))

    if not stats:
        print("No pinned-dialect Arabic replies found in range.")
        return 0

    print(f"\n== Dialect-purity lint: {os.path.basename(args.logfile)}"
          f"{' since ' + args.since if args.since else ''} ==\n")
    print(f"{'dialect':>9}  {'replies':>7}  {'with-leaks':>10}  {'leak-rate':>9}  {'msa-drift':>9}  {'auto-fixed':>10}")
    total_n = total_leaky = 0
    for d in ("Najdi", "Hijazi", "Egyptian", "Fusha"):
        s = stats.get(d)
        if not s:
            continue
        total_n += s["n"]; total_leaky += s["leaky"]
        print(f"{d:>9}  {s['n']:>7}  {s['leaky']:>10}  {100*s['leaky']/s['n']:>8.0f}%  "
              f"{s['drifty']:>9}  {s['fixed']:>10}")
    print(f"{'TOTAL':>9}  {total_n:>7}  {total_leaky:>10}  {100*total_leaky/max(total_n,1):>8.0f}%")

    print("\n-- offending turns --")
    for d in ("Najdi", "Hijazi", "Egyptian", "Fusha"):
        for ts, leaks, drift, fixups, q in stats.get(d, {}).get("rows", []):
            parts = []
            if leaks:
                parts.append("LEAK: " + "، ".join(leaks))
            if drift:
                parts.append("msa-drift: " + "، ".join(drift))
            if fixups:
                parts.append("auto-fixed: " + "، ".join(fixups))
            print(f"  [{ts}] {d:<8} {' | '.join(parts)}   (Q: {q})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
