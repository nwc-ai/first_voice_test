"""
leak_lint.py — glossary-based cross-dialect leak detector (pure functions, no GPU; imports only
`routing` for the shared TTS_LANG_TO_DIALECT map — no imports from the GPU-touching modules
tts_omnivoice_v1/llm/server). Ported from chatterbox-tts:eval/dialect_purity_lint.py
(2026-07-06/07/08 token decisions) and adapted to this branch's supported set: Najdi, Fusha,
Egyptian(*).

(*) Egyptian is being reintroduced on this branch (2026-07-20 plan). The Egyptian target set
is already here so the A/B harness and the Step-6 log linter score Egyptian-routed replies the
moment they exist; until then it is simply never selected.

Three kinds of finding:
  LEAK       — a token from ANOTHER dialect inside a routed reply (misroutes the reply's
               dialect identity), the Egyptian هـ-future / ما...ش-negation patterns outside
               Egyptian, or جداً inside a dialect reply (promoted from soft drift 2026-07-07:
               the cards say NEVER جداً — Najdi=مرة, Egyptian=أوي). Hard error.
  MSA-drift  — MSA function words (حيث، مليء، لذا…) inside a dialect reply. Soft signal:
               educated register is acceptable, but rising drift = slipping back to Fusha.
  (this file only DETECTS — the deterministic auto-repair for a subset of these findings,
  e.g. جداً, lives in routing.DIALECT_REPAIR_MAP/apply_dialect_repairs, applied at synthesis
  time; every DIALECT_REPAIR_MAP key must be a member of FORBIDDEN for its dialect —
  enforced by eval/test_dialect_repair.py, not duplicated here.)

Precision notes carried over from the old branch (documented negative results — keep):
  - Sets contain only HIGH-PRECISION tokens; shared/pan-dialect words are never flagged.
  - راح is deliberately NOT forbidden in Egyptian — it is also valid Egyptian past "went";
    راح-future vs راح-went can't be separated lexically.
  - The Egyptian بـ-present inside a Najdi reply is NOT lintable — the same surface form is
    the legitimate Najdi FUTURE (بيكون). NAJDI_GRAMMAR_RULE handles it prompt-side; judging
    it stays by ear.
  - _HA_FUTURE_OK whitelists هـ-lookalikes incl. هينزلا (Whisper's garble of "Hunza" — live
    false positive 2026-07-06).
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from routing import TTS_LANG_TO_DIALECT  # noqa: E402  single source of truth (was a local copy)

# ── Token groups (from the owner's cross-dialect glossary) ────────────────────────────────
_EGY = {"دلوقتي", "دلوقت", "كده", "كدا", "النهاردة", "إمبارح", "امبارح", "إزاي", "ازاي",
        "إزيك", "ازيك", "عايز", "عاوز", "عايزة", "مفيش", "معنديش", "متشكر", "أوي",
        "بتاع", "بتاعة", "كتير", "كويس", "ده", "دي", "مش", "فين",
        # Water-utility domain rows — only the high-precision Egyptian-only forms
        # (مليان/خربان/عاطل/رشح/قطع deliberately excluded: cross-dialect or MSA homographs):
        "بايظ", "قراية",
        # Glued ـش-negation verbs (the bare pattern is _SHIN_NEG_RE below):
        "معرفش", "مكانش", "ماكانش", "منفعش"}
# دول deliberately EXCLUDED from the set above (removed 2026-07-22): bare دول is the
# everyday MSA plural of دولة in construct state («دول الخليج، دول العالم») —
# routing.py already demoted it to a weak marker for dialect-DETECTION for exactly this
# collision, but that fix was never propagated here, so a correct Najdi/Fusha "دول
# الخليج" was being mis-flagged as an Egyptian leak. Same shape of mistake as جداً (see
# DIALECT_REPAIR_MAP's correction comment in routing.py). دول is a genuine homograph
# (MSA "countries" vs. Egyptian's demonstrative "الناس دول") word-set matching can't
# disambiguate — same tradeoff already accepted on the detection side: stop
# false-flagging the common MSA usage, accept missing the rarer genuine leak.
# Najdi + Gulf-adjacent words — forbidden inside Egyptian/Fusha replies.
_NAJDI_GULF = {"وش", "أبغى", "ابغى", "الحين", "يبيلك", "صج", "وايد", "شنو", "دحين"}
# Hijazi is no longer a routed dialect on this branch, but its distinctive tokens stay
# forbidden in Egyptian/Fusha (إيش is Najdi-acceptable per the owner's Najdi rule 2 exemplars
# وش/إيش, so it only counts against Egyptian/Fusha; مشكور is native Najdi/Gulf "thank you" —
# old-branch semantics forbid it in EGYPTIAN only, never in Najdi/Fusha).
_HIJAZI_ONLY = {"دحين"}
# جداً — forbidden in dialect replies, correct in Fusha. Tokenizer strips tanween → جدا.
_JIDDAN = {"جدا", "جداً"}
_MSA_DRIFT = {"حيث", "مليء", "مليئة", "بعيدا", "بعيداً", "لذا", "كذلك"}
_EGY_DRIFT_EXTRA = {"كيف"}   # Egyptian wants إزاي; كيف is native in Najdi

# Egyptian هـ-future — forbidden outside Egyptian replies. Two shapes:
#  2nd/3rd person: ه + ي/ت/ن + stem (هيكون، هتكون، هنروح) — regex + lookalike whitelist.
#  1st person: ه attaches straight to the verb (هقولك، هخبرك) — regex would swallow ordinary
#  nouns (هجوم، هدوء), so these are a curated list of common forms.
_HA_FUTURE_RE = re.compile(r"\bه[يتن][ء-ي]{2,}\b")
_HA_FUTURE_OK = {"هناك", "هنالك", "هيئة", "هيئات", "هيبة", "هيكل", "هياكل", "هيمنة",
                 "هيه", "هيا", "هتاف", "هتف", "هند", "هندي", "هندية", "هنود", "هندسة",
                 "هنيئا", "هنيئاً", "هينزلا",
                 # Suffixed هندسة forms — live FP 2026-07-20: «هندستها» ("its architecture",
                 # step4-golive Fusha run) matched the ه[يتن] regex; the whitelist is
                 # exact-token so construct/possessive forms need their own entries.
                 "هندستها", "هندسته", "هندستهم", "هندستنا"}
_HA_FUTURE_1P = {"هقول", "هقولك", "هقوللك", "هقولكم", "هعمل", "هعمله", "هعملها", "هروح",
                 "هاروح", "هشوف", "هشوفك", "هاخد", "هاخده", "هخبرك", "هخبركم", "هبقى",
                 "هكون", "هقدر", "هلاقي", "هعرف", "هعرفك", "هجيب", "هجيبلك", "هحاول",
                 "هكلمك", "هبعت", "هحكي", "هحكيلك", "هفكر", "هبدأ", "هحط", "هديك", "هساعدك"}
_SHIN_NEG_RE = re.compile(r"\bما\s+[ء-ي]{2,}ش\b")

# تسوي/يسوي/نسوي/سويت — "to do/make," a vetted, high-confidence Najdi marker family
# (routing.py's _NAJDI_MARKERS; Egyptian/Fusha use تعمل/يعمل family instead). Forbidden
# outside Najdi. Found leaking into an Egyptian-routed reply ("لازم تسويها دلوقتي") via
# manual transcript review, 2026-07-21 — not a new candidate needing vetting (already
# vetted when added to _NAJDI_MARKERS), just a gap between that set and this one.
# A plain token set misses this: Arabic attaches object pronouns directly (تسويها,
# يسويها, سويتها, ...) so the whole word never exact-matches a bare root — same reason
# _HA_FUTURE_RE/_SHIN_NEG_RE above are regexes, not sets. The suffix list is closed and
# curated (not \w*) specifically so a derived word like تسويق ("marketing," root سوّق,
# not سوى) never false-matches — verified empirically.
_TSAWWA_SUFFIXES = "|".join(["ها", "ه", "هم", "هن", "كم", "كن", "ك", "ني", "نا"])
_TSAWWA_RE = re.compile(
    rf"\b(?:[تين]سوي(?:{_TSAWWA_SUFFIXES})?"
    rf"|سويت(?:{_TSAWWA_SUFFIXES}|ي|وا)?)\b"
)
_TSAWWA_FORBIDDEN_IN = {"Egyptian", "Fusha"}   # سوى-family allowed only in Najdi

# مرة — "very" IS a genuine leak in Egyptian (Gulf/Levantine usage; EGYPTIAN_CARD teaches
# أوي/جداً instead), but a plain word-set match can't tell that sense apart from the vastly
# more common, completely legitimate "time/occurrence" sense (أول مرة، كل مرة، مرة واحدة،
# مرة ثانية...). Checked every real occurrence flagged across three eval runs (10 total,
# 2026-07-22/27, eval/BASELINES.md) — ALL TEN were "time" usage, zero genuine intensifier
# leaks found. Same shape of mistake as جداً/دول (see routing.py's DIALECT_REPAIR_MAP
# comment and _EGY's دول note above) — removed from the plain FORBIDDEN set below, replaced
# with a context-aware regex so a genuine leak (if one ever occurs) is still caught, while
# the ubiquitous "time" sense no longer false-flags.
_MARRA_TIME_RE = re.compile(
    # [وفل]? — Arabic glues و/ف/ل (and/so/for) directly onto the next word with no space
    # («فكل مرة», «وكل مرة», «لكل مرة»); optional so a bare trigger word still matches too.
    r"\b[وفل]?(?:أول|كل|آخر|تاني|ثاني|كام|من)\s+مرة\b"
    r"|\bمرة\s+(?:واحدة|تانية|ثانية|أخرى|كمان)\b"
)
_MARRA_BARE_RE = re.compile(r"\bمرة\b")


def _marra_leaks(text: str) -> bool:
    """True if `text` contains مرة used OUTSIDE a known time-of-occurrence construction —
    i.e. the genuine Gulf/Levantine "very" leak, not the ordinary "time" sense."""
    safe_spans = [m.span() for m in _MARRA_TIME_RE.finditer(text)]
    return any(not any(a <= s and e <= b for a, b in safe_spans)
               for s, e in (m.span() for m in _MARRA_BARE_RE.finditer(text)))


_MARRA_FORBIDDEN_IN = {"Egyptian"}   # مرة-as-"time" is fine everywhere; only Egyptian
                                     # forbids the intensifier sense (Najdi's is correct).

# What is forbidden INSIDE a reply routed to each dialect. إيش/حاجة notes: إيش is acceptable
# Najdi on this branch (rule 2 says وش/إيش); حاجة is Najdi-forbidden (Najdi=شي).
FORBIDDEN: dict[str, set] = {
    "Najdi":    _EGY | _HIJAZI_ONLY | _JIDDAN | {"وايد", "شنو", "كمان", "بدي", "حاجة"},
    # كثير added — EGYPTIAN_CARD (routing.py) explicitly forbids it ("a lot=كتير never
    # كثير/وايد") but it wasn't in this set before; a leaking كثير in an Egyptian reply
    # passed undetected. Found by direct diff against routing.py's glossary prose
    # (najdi-q2-wrong-elegant-papert.md, Part B.3). مرة is handled separately below
    # (_marra_leaks) — it's a homograph a plain word-set can't disambiguate.
    # NOTE: _JIDDAN deliberately excluded here (unlike Najdi/Fusha above/below) — see
    # the 2026-07-22 correction: جداً/جدا is genuinely correct Egyptian for "very", it
    # was wrongly forbidden for two prior sessions.
    # الذي، تمشى، تأكل added — EGYPTIAN_CARD's relative-pronoun/colloquial-spelling notes
    # (routing.py, 2026-07-22 live-testing review) forbid all three but none were in this
    # set before.
    "Egyptian": _NAJDI_GULF | _HIJAZI_ONLY | {"إيش", "ايش", "زين", "وين", "مشكور",
                                              "كثير", "الذي", "تمشى", "تأكل"},
    # Fusha must contain no dialect function words at all:
    "Fusha":    (_EGY | _NAJDI_GULF | _HIJAZI_ONLY
                 | {"إيش", "ايش", "كمان", "بدي", "يلا", "معليش"}),
}
# Stray-dialect words that belong to NO target dialect (Levantine هيك, Maghrebi مزيان —
# both reached live replies on the old branch 2026-07-08; شو — Levantine "what," found
# leaking into a Najdi-routed reply via manual transcript review, 2026-07-21). Forbidden
# everywhere:
_STRAY = {"هيك", "مزيان", "شو"}
for _d in FORBIDDEN:
    FORBIDDEN[_d] |= _STRAY

_HA_FORBIDDEN_IN = {"Najdi", "Fusha"}   # هـ-future allowed only in Egyptian

_AR_WORD_RE = re.compile(r"[ء-ي]+")
# TTS_LANG_TO_DIALECT is imported from routing (top of file) — single source of truth.


def find_leaks(text: str, dialect: str) -> tuple[list[str], list[str]]:
    """Return (leaks, msa_drift) for a reply routed to `dialect` ("Najdi"/"Egyptian"/"Fusha").
    leaks = cross-dialect tokens + هـ-future / ش-negation patterns; msa_drift = soft MSA words."""
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
    if dialect in _TSAWWA_FORBIDDEN_IN:
        for m in _TSAWWA_RE.findall(text):
            leaks.append(f"سوى-verb:{m}")
    if dialect in _MARRA_FORBIDDEN_IN and _marra_leaks(text):
        leaks.append("مرة")
    drift_set = _MSA_DRIFT | (_EGY_DRIFT_EXTRA if dialect == "Egyptian" else set())
    drift = sorted(words & drift_set) if dialect in ("Najdi", "Egyptian") else []
    return leaks, drift
