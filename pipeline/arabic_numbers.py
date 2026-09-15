"""arabic_numbers — dialect-aware number verbalization for the TTS layer.

Owner-approved 2026-08-31, replacing num2words: three in-house tables (MSA /
Najdi / Egyptian) built from the owner's grammar instruction block. The dialect
number systems are INVARIABLE (no case endings, no gender polarity) so they are
fully deterministic; the MSA table is nominative-base (case/polarity need
sentence context, which lives in the fanar prompt note — deterministic code
cannot see the counted noun).

Colloquial forms are hand-diacritized (synthesis-only — display keeps digits).
The Najdi table's 3-10 short invariable forms are the owner-chosen default,
whole table pending Leen's native sign-off. Numbers above 999,999 are returned
as None — the caller leaves them as digits (the CATT digit-guard keeps them
alive and the voice improvises; meter readings never get that large).
"""

import re

_TABLES = {
    "msa": {
        "zero": "صفر",
        # 1..10 (default ة-forms for 3-9 — noun-blind masculine-count default)
        "units": ["", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة",
                  "ستة", "سبعة", "ثمانية", "تسعة", "عشرة"],
        # bare stems used for attached hundreds (ثلاثمئة)
        "unit_stems": ["", "", "", "ثلاث", "أربع", "خمس", "ست", "سبع", "ثمان", "تسع"],
        "teens": ["عشرة", "أحد عشر", "اثنا عشر", "ثلاثة عشر", "أربعة عشر",
                  "خمسة عشر", "ستة عشر", "سبعة عشر", "ثمانية عشر", "تسعة عشر"],
        "tens": ["", "", "عشرون", "ثلاثون", "أربعون", "خمسون",
                 "ستون", "سبعون", "ثمانون", "تسعون"],
        "tens_time": ["", "", "عشرين", "ثلاثين", "أربعين", "خمسين",
                      "ستين", "سبعين", "ثمانين", "تسعين"],
        # مئة (no silent alif) — updated 2026-09-09 per Leen's corrected block
        # (the alif in مائة was a TTS mis-read). 200 keeps the alif (مائتان) by
        # owner decision. Cascades to attached 300-900 (ثلاثمئة، ثمانمئة).
        "hundred": "مئة", "two_hundred": "مائتان",
        "hundreds_attached": True,          # ثلاثمئة
        "thousand": "ألف", "two_thousand": "ألفان", "thousands_plural": "آلاف",
        "million": "مليون", "two_million": "مليونان", "millions_plural": "ملايين",
        "milliard": "مليار", "two_milliard": "ملياران", "milliards_plural": "مليارات",
        "join": " و",                        # "مائة وخمسة" — و attached to next word
        "fasla": "فاصلة",
    },
    "najdi": {
        "zero": "صِفْر",
        # owner decision: short invariable 3-10 forms (pending Leen's ear)
        "units": ["", "وَاحِد", "اِثْنَيْن", "ثَلَاث", "أَرْبَع", "خَمْس",
                  "سِتّ", "سَبْع", "ثِمَان", "تِسْع", "عَشْر"],
        "unit_stems": ["", "", "", "ثَلَاث", "أَرْبَع", "خَمْس", "سِتّ", "سَبْع", "ثِمَان", "تِسْع"],
        "teens": ["عَشْر", "حِدَاعَش", "اِثْنَعَش", "ثَلَطْعَش", "أَرْبَعْطَعَش",
                  "خَمْسْطَعَش", "سِتَّعَش", "سَبْعَطَعَش", "ثَمَنْطَعَش", "تِسَعْطَعَش"],
        "tens": ["", "", "عِشْرِين", "ثَلَاثِين", "أَرْبَعِين", "خَمْسِين",
                 "سِتِّين", "سَبْعِين", "ثَمَانِين", "تِسْعِين"],
        "tens_time": ["", "", "عِشْرِين", "ثَلَاثِين", "أَرْبَعِين", "خَمْسِين",
                      "سِتِّين", "سَبْعِين", "ثَمَانِين", "تِسْعِين"],
        "hundred": "مِيَّة", "two_hundred": "مِيَّتَين",
        "hundreds_attached": False,         # ثَلَاث مِيَّة (spaced, per the block)
        "thousand": "أَلْف", "two_thousand": "أَلْفَين", "thousands_plural": "آلَاف",
        "million": "مِلْيُون", "two_million": "مِلْيُونَين", "millions_plural": "مَلَايِين",
        "milliard": "مِلْيَار", "two_milliard": "مِلْيَارَين", "milliards_plural": "مِلْيَارَات",
        "join": " وْ",                       # وَاحِد وْعِشْرِين
        "fasla": "فَاصِلَة",
    },
    "egyptian": {
        "zero": "صِفْر",
        "units": ["", "وَاحِد", "اِتْنِين", "تَلَاتَة", "أَرْبَعَة", "خَمْسَة",
                  "سِتَّة", "سَبْعَة", "تَمَانْيَة", "تِسْعَة", "عَشَرَة"],
        # 300 = تَلْتُمِيَّة (تلت, alif dropped — updated 2026-09-09 per Leen's block);
        # 3000 uses thousand_stems below and stays تَلَات آلَاف.
        "unit_stems": ["", "", "", "تَلْتُ", "أَرْبَعُ", "خَمْسُ", "سِتُّ", "سَبْعُ", "تَمَنُ", "تِسْعُ"],
        "teens": ["عَشَرَة", "حِدَاشَر", "اِتْنَاشَر", "تَلَتَّاشَر", "أَرْبَعْتَاشَر",
                  "خَمَسْتَاشَر", "سِتَّاشَر", "سَبَعْتَاشَر", "تَمَنْتَاشَر", "تِسَعْتَاشَر"],
        "tens": ["", "", "عِشْرِين", "تَلَاتِين", "أَرْبَعِين", "خَمْسِين",
                 "سِتِّين", "سَبْعِين", "تَمَانِين", "تِسْعِين"],
        "tens_time": ["", "", "عِشْرِين", "تَلَاتِين", "أَرْبَعِين", "خَمْسِين",
                      "سِتِّين", "سَبْعِين", "تَمَانِين", "تِسْعِين"],
        "hundred": "مِيَّة", "two_hundred": "مِيَّتِين",
        "hundreds_attached": True,          # تَلْتُمِيَّة (attached, per the block)
        # bare stems for the SPACED thousands (تَلَات آلَاف — the damma in
        # unit_stems belongs only to the attached-hundred form):
        "thousand_stems": ["", "", "", "تَلَات", "أَرْبَع", "خَمَس", "سِتّ",
                           "سَبَع", "تَمَن", "تِسْع", "عَشَر"],
        "thousand": "أَلْف", "two_thousand": "أَلْفِين", "thousands_plural": "آلَاف",
        "million": "مِلْيُون", "two_million": "مِلْيُونِين", "millions_plural": "مَلَايِين",
        "milliard": "مِلْيَار", "two_milliard": "مِلْيَارِين", "milliards_plural": "مِلْيَارَات",
        "join": " وِ",                       # تَلَاتَة وِعِشْرِين
        "fasla": "فَاصْلَة",
    },
}

MAX_VERBALIZED = 999_999_999   # up to ~1e9 (millions); 10+-digit runs are read
                               # digit-by-digit by the phone rule before reaching here


# ── Redundant spelled-out parenthetical (Issue 1, owner-approved 2026-09-09) ──
# Some models (observed on Fanar) write a number as a digit AND immediately spell
# it out: "168 (مائة وثمانية وستين) ساعة". Since we verbalize the digit too, the
# voice would say the number twice. Drop the parenthetical when it sits right
# after a run of digits and contains ONLY number-words + the و connector. This is
# a text-normalization (model-agnostic), not a table change; a parenthetical like
# "(ساعة)" or "(NWC)" is never number-words, so it is never touched.
_HARAKAT_RE = re.compile(r"[ً-ْٰ]")


def _bare(w: str) -> str:
    return _HARAKAT_RE.sub("", w)


def _build_number_words() -> set:
    words = set()
    for t in _TABLES.values():
        for key in ("units", "unit_stems", "teens", "tens", "tens_time"):
            for entry in t[key]:
                for tok in entry.split():
                    if tok:
                        words.add(_bare(tok))
        for key in ("zero", "hundred", "two_hundred", "thousand",
                    "two_thousand", "thousands_plural"):
            words.add(_bare(t[key]))
        for entry in t.get("thousand_stems", []):
            if entry:
                words.add(_bare(entry))
        # composed attached-hundreds (ثلاثمئة، تلتمية…) are built at runtime, not
        # stored as tokens — add them so a spelled parenthetical is recognized:
        if t.get("hundreds_attached"):
            h = _bare(t["hundred"])
            for stem in t["unit_stems"][3:10]:
                if stem:
                    words.add(_bare(stem) + h)
                    if h == "مئة":                       # MSA: also the alif spelling
                        words.add(_bare(stem) + "مائة")
    # common MSA spellings / case variants a model emits that may differ from the
    # current table cells (e.g. مائة with the silent alif; oblique تين-duals):
    words |= {"مائة", "مئة", "مائتان", "مئتان", "مائتين", "مئتين",
              "احد", "إحدى", "احدى", "اثنتان", "اثنتا", "اثنا", "اثني", "اثنين",
              "عشر", "عشرة", "الف", "الاف", "آلاف"}
    words.discard("")
    return words


_NUMBER_WORDS = _build_number_words()
_PAREN_AFTER_DIGITS_RE = re.compile(r"(?P<digits>\d+)\s*\((?P<inner>[^)]*)\)")
# Reverse case (2026-09-10): number-WORD immediately followed by a pure-digits
# parenthetical, e.g. "مية (١٠٠)". pre is the token before "(".
_PAREN_AFTER_WORD_RE = re.compile(r"(?P<pre>[^\s()]+)\s*\((?P<inner>[^)]*)\)")
_DIGITS_ONLY_RE = re.compile(r"[\d٠-٩]+([.,:/][\d٠-٩]+)*")


def _is_number_word(tok: str) -> bool:
    b = _bare(tok).strip("،؛.:!؟,")
    return b in _NUMBER_WORDS or (b.startswith("و") and b[1:] in _NUMBER_WORDS)


def strip_redundant_number_parenthetical(text: str) -> str:
    """Drop a parenthetical that merely re-states a number already present next to
    it — in EITHER direction (a model may write "168 (مائة وثمانية وستين)" or the
    mirror "مية (١٠٠)"). A "(NWC)"/"(ساعة)"/"(100°C)" paren is never pure number
    on both sides, so it is never touched."""
    def _forward(m):   # DIGITS ( spelled-out number ) → keep the digit, drop the words
        inner = m.group("inner").strip()
        if inner and all(_is_number_word(tok) for tok in inner.split()):
            return m.group("digits")
        return m.group(0)

    def _reverse(m):   # number-WORD ( pure digits ) → keep the word, drop the digits
        inner = m.group("inner").strip()
        if inner and _DIGITS_ONLY_RE.fullmatch(inner) and _is_number_word(m.group("pre")):
            return m.group("pre")
        return m.group(0)

    text = _PAREN_AFTER_DIGITS_RE.sub(_forward, text)
    text = _PAREN_AFTER_WORD_RE.sub(_reverse, text)
    return text


def _under_hundred(n: int, t: dict, time_form: bool = False) -> str:
    if n == 0:
        return ""
    if n <= 10:
        return t["units"][n]
    if n < 20:
        return t["teens"][n - 10]
    tens_table = t["tens_time"] if time_form else t["tens"]
    tens, unit = divmod(n, 10)
    if unit == 0:
        return tens_table[tens]
    return t["units"][unit] + t["join"] + tens_table[tens]


def _hundreds_word(h: int, t: dict) -> str:
    if h == 1:
        return t["hundred"]
    if h == 2:
        return t["two_hundred"]
    if t["hundreds_attached"]:
        return t["unit_stems"][h] + t["hundred"]
    return t["unit_stems"][h] + " " + t["hundred"]


def _group_below_1000(g: int, t: dict) -> str:
    """Words for a 1-999 group (hundreds + tail), joined by the dialect's و."""
    parts = []
    hundreds, tail = divmod(g, 100)
    if hundreds:
        parts.append(_hundreds_word(hundreds, t))
    if tail:
        parts.append(_under_hundred(tail, t))
    return t["join"].join(parts)


def _scale_word(count: int, sing: str, dual: str, plural: str, t: dict) -> str:
    """`count` groups of a scale word (thousand/million/milliard), count 1-999.
    1→singular, 2→dual, 3-10→unit+plural (ثلاثة آلاف/تلات ملايين), 11+→group+singular."""
    if count == 1:
        return sing
    if count == 2:
        return dual
    if 3 <= count <= 10:
        if t is _TABLES["msa"]:
            unit = t["units"][count]
        else:
            stems = t.get("thousand_stems", t["unit_stems"])
            unit = stems[count] if count < len(stems) else t["units"][count]
        return unit + " " + plural
    return _group_below_1000(count, t) + " " + sing


# high → low; each scale takes (value, singular-key, dual-key, plural-key)
_SCALES = [
    (1_000_000_000, "milliard", "two_milliard", "milliards_plural"),
    (1_000_000,     "million",  "two_million",  "millions_plural"),
    (1_000,         "thousand", "two_thousand", "thousands_plural"),
]


def int_to_words(n: int, dialect: str) -> str | None:
    """0..MAX_VERBALIZED → spoken words for the dialect; None if out of range."""
    t = _TABLES[dialect]
    if n < 0 or n > MAX_VERBALIZED:
        return None
    if n == 0:
        return t["zero"]
    parts = []
    for value, sing_k, dual_k, plural_k in _SCALES:
        c, n = divmod(n, value)
        if c:
            parts.append(_scale_word(c, t[sing_k], t[dual_k], t[plural_k], t))
    if n:                                    # the remaining 0-999 units group
        parts.append(_group_below_1000(n, t))
    return t["join"].join(parts)


def decimal_to_words(int_part: str, frac_part: str, dialect: str) -> str | None:
    """N.M → '<N> فاصلة <digit by digit M>' per dialect (block rule: never a bare digit)."""
    t = _TABLES[dialect]
    left = int_to_words(int(int_part), dialect)
    if left is None:
        return None
    return f"{left} {t['fasla']} {digit_string_to_words(frac_part, dialect)}"


def digit_string_to_words(s: str, dialect: str) -> str:
    """Read a digit string one digit at a time (phone/account/reference numbers,
    fractional parts) — e.g. '0501' → 'صفر خمسة صفر واحد'."""
    t = _TABLES[dialect]
    return " ".join(t["units"][int(d)] if d != "0" else t["zero"] for d in s)


# Gregorian month names as used in KSA + Egypt (يناير-style, not the Levantine
# كانون/شباط set). Shared across dialects — the month names don't vary here.
_GREGORIAN_MONTHS = ["", "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
                     "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]


def date_to_words(day: str, month: str, year: str, dialect: str) -> str | None:
    """D/M/YYYY → '<day> <month-name> <year>' so a date is spoken as a date, not
    three bare numbers in a row. Returns None on an implausible date (caller then
    leaves the digits for the number rules)."""
    d, m, y = int(day), int(month), int(year)
    if not (1 <= d <= 31 and 1 <= m <= 12 and 1000 <= y <= 2999):
        return None
    dw, yw = int_to_words(d, dialect), int_to_words(y, dialect)
    if dw is None or yw is None:
        return None
    return f"{dw} {_GREGORIAN_MONTHS[m]} {yw}"


def time_to_words(hours: str, minutes: str, dialect: str,
                  seconds: str | None = None) -> str | None:
    """H:MM[:SS] → '<H> و<MM>[ و<SS>]' colloquial clock reading (owner decision
    2026-08-31, +seconds 2026-09-09); minutes/seconds use the oblique ين-forms."""
    t = _TABLES[dialect]
    h, m = int(hours), int(minutes)
    s = int(seconds) if seconds is not None else None
    if h > 24 or m > 59 or (s is not None and s > 59):
        return None
    parts = [int_to_words(h, dialect)]
    if m or s is not None:
        parts.append(_under_hundred(m, t, time_form=True) if m else t["zero"])
    if s is not None:
        parts.append(_under_hundred(s, t, time_form=True) if s else t["zero"])
    return t["join"].join(parts)
