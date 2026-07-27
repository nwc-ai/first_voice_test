"""
routing.py — language/dialect detection and text-acceptance policy.

Shared by server.py (utterance routing) and tts_omnivoice_v1.py (per-sentence
CATT gate). Lives in its own module because the TTS module needs the same Najdi
detector, and importing server from the TTS module would be circular.
"""

import re
from typing import Any, Optional

# ── Arabic normalization ──────────────────────────────────────────────────────
# STT/LLM output varies in hamza seating and diacritics; normalize before any
# lexicon match so أبغى/ابغى or a diacritized reply still hit the markers.
_TASHKEEL_RE = re.compile(r"[ً-ْٰ]")  # harakat, shadda, sukun, dagger alif


def normalize_ar(text: str) -> str:
    text = _TASHKEEL_RE.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي")
    return text


# ── Najdi lexical detector ────────────────────────────────────────────────────
# Distinctly-Najdi words only (normalized forms). Words shared with MSA
# (رقم, ضغط, محطة, شكراً…) or pan-Arabic (بس, خلاص, في, مرة) are deliberately
# EXCLUDED — including them would falsely flag Fusha speech as Najdi.
# Known accepted risk: عاد/زين/ماله also exist as MSA words; as single markers
# they can misfire on rare Fusha sentences — acceptable for this heuristic.
_NAJDI_MARKERS = {normalize_ar(w) for w in {
    "وش", "ايش", "ليش", "وين", "عاد", "الحين", "بعدين", "شوي", "صج", "زين",
    "ابغى", "يبيلك", "ماله", "هيه", "خربان", "واطي", "فاضي", "ممتلي", "بطي",
    "ادري",  # from "ما أدري" when tokenized

    # Added via eval (dialect_eval_holdout_report.md): each entry below is either
    # a grammatical conjugation of an already-vetted word above, or a standalone
    # word that essentially never occurs in real Fusha text — NOT a general
    # "sounds informal" word (those were deliberately left out; see the report
    # for the ones considered and rejected: قدر/يقدر family collides with real
    # MSA usage "لا يقدر على"/"بقدر الإمكان"; راح alone collides with MSA "went";
    # وسايل/يقرا are just hamza-dropped spellings of ordinary MSA words, not a
    # dialect signal).
    #
    # REMOVED 2026-07-24 (full-eval finding): اللي، عشان، لسا/لسه، يلا used to be
    # here as "accepted collision" Najdi markers, but they are genuinely pan-dialect
    # (common in real Egyptian speech too, not Najdi-exclusive), and Najdi-first
    # short-circuit precedence in build_turn() meant any Egyptian utterance using
    # one of them misrouted to Najdi. Quantified via eval/dialect_id_cases.jsonl's
    # collision rows: removing them fixes 6 of 8 affected rows (Egyptian text now
    # correctly routes Egyptian, or ambiguous text correctly routes None) at the
    # cost of one accepted regression (a genuine Najdi sentence whose ONLY marker
    # was لسه — "لسه ما وصلت الفاتورة" — now goes undetected). See BASELINES.md's
    # 2026-07-24 entry for the full before/after.
    "تبغى", "يبغى", "نبغى", "تبغين",           # ابغى/أبغى conjugations
    "تشوف", "يشوف", "نشوف", "شاف", "شفت",       # "to see" — Fusha uses يرى/رأى
    "تسوي", "يسوي", "نسوي", "سويت",             # "to do/make" — NOT bare "سوى" (= MSA "except")
    "ماي", "مويه",                              # colloquial "water" — Fusha writes الماء/المياه
    "عيال", "عيالي", "عيالك", "عيالنا", "عيالهم", # "kids/family" — Fusha uses الأطفال/الأولاد
}}  # markers stored normalized — input text is normalized before matching
# Multi-word Najdi expressions (matched on the normalized text as phrases).
_NAJDI_PHRASES_RE = re.compile(
    r"\bما ادري\b|\bما عندي\b|\bما في\b|\bكيف الحال\b",
    re.UNICODE,
)
# Tokenizer for the single-word marker check. NOTE: a naive "split on
# non-Arabic-letter runs" using the raw Unicode block ؀-ۿ (U+0600-U+06FF) is
# WRONG — that block also contains Arabic punctuation (، ؛ ؟ are U+060C/061B/
# 061F, all inside it), so "الحين؟" or "زين،" never split and the marker never
# matched — exactly the normal way Arabic is punctuated (no space before ، or
# ؟). \w+ uses Python's real Unicode word/punctuation categories instead of a
# guessed code-point range, so it correctly treats Arabic letters as word
# characters and Arabic punctuation as separators.
_AR_WORD_RE = re.compile(r"\w+", re.UNICODE)


def looks_najdi(text: str) -> bool:
    """True if the text carries distinctly-Najdi vocabulary. Used to (a) route the
    LLM reply dialect for spoken Arabic and (b) gate CATT tashkeel per sentence —
    CATT is MSA-trained and mis-vocalizes Najdi words."""
    norm = normalize_ar(text)
    words = set(_AR_WORD_RE.findall(norm))
    # Also check each word with a leading "ال" (definite article) stripped —
    # Arabic nouns constantly appear with or without it attached as one token
    # (e.g. "الماي" vs "ماي"), and exact-token matching alone misses that whole
    # class. Safe for markers that don't themselves start with "ال" (those,
    # like "الحين", already match directly — stripping never overrides them).
    stripped = {w[2:] for w in words if w.startswith("ال") and len(w) > 4}
    if (words | stripped) & _NAJDI_MARKERS:
        return True
    return bool(_NAJDI_PHRASES_RE.search(norm))


# ── Egyptian (Masri) lexical detector ─────────────────────────────────────────
# Ported from the omnivoice-tts branch (last commit with Egyptian, 756fa89) and adapted
# to this branch's normalization convention. NOT consumed by llm.py yet — build_turn wires
# it in at plan Step 4, behind looks_najdi (Najdi always evaluated first, short-circuit:
# the owner-accepted 2026-07-20 invariant — any utterance that routes Najdi today keeps
# routing Najdi). اللي/عشان/لسه/يلا were REMOVED from _NAJDI_MARKERS on 2026-07-24 for being
# pan-dialect rather than Najdi-exclusive (see that set's own comment) — the short-circuit
# precedence itself is unchanged, only the marker set shrank, so this no longer collides.
#
# Set decisions carried over as documented (do not "fix" without the eval):
# - bare عشان EXCLUDED (shared across dialects; removed from _NAJDI_MARKERS too as of
#   2026-07-24) — only علشان (with the ل) is the Egyptian-leaning variant.
# - إمتى EXCLUDED (overlaps Hijazi/Gulf).
# - إيه/كام/فين kept — on the old branch these three lifted Egyptian discrimination
#   58%→92% on a 100-question test. Known accepted FP: normalized ايه is also the Najdi
#   "yes" (watch case in eval; v1.1 candidate: demote bare ايه).
# - دول DEMOTED to weak (new vs the old branch): bare دول is the everyday MSA plural of
#   دولة in construct state («دول الخليج، دول العالم») — as a strong marker it flips real
#   Fusha questions to Egyptian. As weak it can only support other markers.
_EGYPTIAN_MARKERS = {normalize_ar(w) for w in {
    "إزاي", "ازاي", "إزيك", "ازيك", "عايز", "عاوز", "عايزة", "دلوقتي", "دلوقت",
    "كده", "كدا", "علشان", "النهاردة", "إمبارح", "امبارح", "أهو",
    "إيه", "ايه", "كام", "فين",
    "ليه",  # "why" — already taught in EGYPTIAN_CARD's own word list ("why=ليه"); sits
            # alongside إيه/فين/كام (its documented siblings) and was simply missing from
            # this set. No collision: zero occurrences in any Najdi/Fusha-expected row
            # across dialect_id_cases.jsonl/test_routing.py (2026-07-22 full-eval finding).
}}
# WEAK markers count 0.5 — high-frequency words shared with other colloquials (and دول with
# MSA, see above): they can support a strong marker or pair up, never decide alone.
_EGYPTIAN_WEAK = {normalize_ar(w) for w in {"مش", "ده", "دي", "دول"}}


def looks_egyptian(text: str) -> bool:
    """True if the text carries distinctly-Egyptian vocabulary (score ≥ 1.0: one strong
    marker, or two weak ones). Mirror of looks_najdi EXCEPT no definite-article stripping:
    Egyptian markers are function words that never take ال, and stripping would manufacture
    false hits from ordinary MSA (e.g. الدول → دول).

    Does strip a leading و/ف/ب (conjunction/preposition) from each token before matching —
    a DIFFERENT, safe class of prefix: Arabic glues these directly onto the next word with
    no space (e.g. وإزاي، فعايز), so exact-token matching alone misses that whole class
    (2026-07-22 full-eval finding; mirrors looks_najdi's ال-stripping for the same reason).
    Safe because these markers don't collide with MSA to begin with — that's the whole
    point of a "distinctly-Egyptian vocabulary" set, unlike ال which manufactures MSA
    false hits."""
    norm = normalize_ar(text)
    words = set(_AR_WORD_RE.findall(norm))
    stripped = {w[1:] for w in words if w[:1] in "وفب" and len(w) > 3}
    all_words = words | stripped
    score = len(all_words & _EGYPTIAN_MARKERS) + 0.5 * len(all_words & _EGYPTIAN_WEAK)
    return score >= 1.0


# ── Egyptian explicit-request pattern (guarded) ───────────────────────────────
# Ported from omnivoice-tts server.py (the 2026-07 review's scar tissue — both guards fix
# CONFIRMED live false positives). NOT consumed by requested_dialect yet — plan Step 4
# consults it AFTER the existing Najdi/Fusha patterns (last position: first-match-wins
# keeps every currently-matching text routing exactly as today).
_NEG_BEFORE_RE = re.compile(
    r"\bلا\b|\bما\b|\bمو\b|\bمش\b|\bبلاش\b|\bبدون\b|don'?t|do\s+not|\bnot\b|\bnever\b|\bstop\b",
    re.IGNORECASE | re.UNICODE,
)


def _negated(text: str, start: int) -> bool:
    """True when a negation token appears in the ~20 chars before position `start` —
    «لا ترد بالمصري» must not commit the very dialect the user is forbidding."""
    return bool(_NEG_BEFORE_RE.search(text[max(0, start - 20):start]))


def _en_dialect_req(name_re: str) -> str:
    """English dialect-name request pattern. The bare name only counts WITH request
    context — a dialect noun ('<name> arabic/dialect/accent') or a speak-verb
    ('reply/speak/say it in <name>') — so proper nouns ('the Egyptian Museum') never
    trigger a dialect request."""
    return (rf"\b(?:{name_re})\s+(?:arabic|dialect|accent)\b"
            rf"|\b(?:reply|respond|answer|speak|say\s+it|talk|switch(?:\s+to)?|use)\s+"
            rf"(?:in\s+|into\s+|to\s+)?(?:the\s+)?(?:{name_re})\b")


# The Arabic arms require a request prefix (بالمصري / لهجة مصرية) so a bare adjective or
# proper noun («المتحف المصري»، «الثورة المصرية») is never mistaken for a dialect request.
# NOTE (?:ال)? — the old branch wrote ال? which is literal ا + optional ل, silently
# requiring the article; «بلهجة مصرية» never matched there. Fixed here (guards unchanged).
_EGYPTIAN_REQUEST_RE = re.compile(
    _en_dialect_req("egyptian|masri") + r"|بالمصري(?:ة|ه)?|(?:لهجة|لغة)\s+(?:ال)?مصري(?:ة|ه)?",
    re.IGNORECASE | re.UNICODE,
)
_EGYPTIAN_REQUEST_PHRASE = "the Egyptian dialect (Masri)"


def requested_egyptian(text: str) -> bool:
    """True when the user explicitly asks for Egyptian and the match is not negated."""
    return any(not _negated(text, m.start()) for m in _EGYPTIAN_REQUEST_RE.finditer(text))


# ── Egyptian dialect card for LLM turn instructions ───────────────────────────
# Ported essentially verbatim from omnivoice-tts@756fa89 _DIALECT_CARDS["Egyptian"] +
# _SPOKEN_REGISTER — the proven artifact; do not "improve" the wording without an A/B
# (eval/dialect_ab.py before/after). Embedded per-turn on Egyptian-routed turns ONLY, the
# exact pattern NAJDI_GLOSSARY uses — nothing Egyptian is ever appended on Najdi turns
# (see the measured pink-elephant result on NAJDI_NO_OTHER_DIALECTS_RULE above).
# Design rules inherited from the old branch: function words + morphology only, no topic
# phrases ("not a checklist" — the model keyword-stuffed bare word lists); دلوقتي is
# meaning-conditioned (present-moment only) because it kept landing in historical
# narration; ده/دي are postposed; the هـ-future is correct EGYPTIAN grammar (it is a leak
# only in Najdi/Fusha, which their own prompts handle).
EGYPTIAN_CARD = (
    "EGYPTIAN usage guide — write natural, fluent Masri as a native speaker would, on any topic. "
    "These are your FUNCTION words, not a checklist; never force them in: "
    "what=إيه، why=ليه، where=فين (never وين)، now=دلوقتي (NEVER الحين/الآن; دلوقتي means "
    "the present moment ONLY — never use it inside past or historical narration)، want=عايز/عاوز "
    "(never أبغى/أبي)، good=كويس، very=أوي or جداً (both natural; NEVER مرة)، a lot=كتير (never كثير/وايد)، "
    "I don't know=مش عارف (never ما أدري)، yes=أيوه، thanks=متشكر، there isn't=مفيش، "
    "full=مليان + noun directly (مليان أحداث — no من). "
    "FUTURE: the هـ prefix (هقولك، هيكون) — never راح or بـ for the future. "
    "NEGATION: مش / ما...ش (معرفش). "
    "Demonstratives ده/دي come AFTER the noun (الزمان ده، الحكاية دي — NEVER ده الزمان). "
    "Avoid MSA connectives (حيث، لذا) — use عشان/علشان. Relative pronoun: اللي (never الذي, "
    "an MSA leak). "
    "Colloquial spelling reflects Egyptian pronunciation, not the MSA spelling: تمشي (not "
    "تمشى), تاكل (not تأكل). "
    "FIELD/STATUS words: working=شغال، broken=بايظ، high=عالي، low=واطي، full=مليان، "
    "empty=فاضي، dirty=وسخ، reading=قراية (not قراءة)، leak=رشح، outage=قطع، really=فعلاً، "
    "settle/sediment=يترسب، dry out=ينشف (not the MSA-flavored يجف). "
    "Technical utility nouns are the same in Egyptian and MSA — use them as-is instead of "
    "hunting for a colloquial replacement: خزان، عداد، تسريب، ضغط، تدفق، محطة، خط، تنبيه، "
    "انقطاع، مشكلة. "
    "If you are genuinely unsure of a fact, say «مش متأكد بصراحة» rather than guessing. "
    "REGISTER: this is a VOICE conversation — answer the way a knowledgeable local TALKS: "
    "address the listener directly, keep a spoken sentence rhythm, and let the dialect's own "
    "grammar carry EVERY sentence — never the tone of a written article or an encyclopedia. "
    "Keep the facts complete; only the voice is conversational."
)

# Grammar rules (not word lists — general patterns to avoid, mirrors NAJDI_GRAMMAR_RULE's
# shape). Found via manual transcript review of both the eval harness and real production
# logs (najdi-q2-wrong-elegant-papert.md, Parts B/D):
#  - ما...ش forced onto a non-verb (e.g. "ما يجفافش" — جفاف is a noun, not a verb stem).
#  - the weak-final/defective (ناقص) verb allomorph: verbs ending in a long vowel (نسي/ينسى,
#    مشى/يمشي-type roots) take a vowel shift before ش, not the plain suffix (تنسى→تنساش,
#    not "تنساش" mis-formed as "تنسىش" or left as MSA "لا تنسى").
#  - بي-habitual prefix wrongly kept after a subjunctive/purpose trigger (عشان، لازم،
#    علشان، إذا، لو) — Egyptian drops the بي- there (يكون not بيكون after عشان/لازم),
#    same mood-conditioned-prefix shape as NAJDI_GRAMMAR_RULE's بـ-future rule.
#  - Form-V/VI verbs kept in their bare MSA shape instead of Egyptian's تت-/اتـ prefix
#    (تجنب→تتجنب، بيفكك→بيتفكك) — scoped to the verbs actually observed in this
#    assistant's domain rather than a fully general rule (real exceptions exist in which
#    prefix a given verb takes; a general rule would misfire on some).
EGYPTIAN_GRAMMAR_RULE = (
    "Grammar note: the ما...ش negation pattern only attaches to an actual conjugated verb "
    "form — never force the ش suffix onto a noun, adjective, or verbal-noun that has no "
    "personal verb conjugation of its own. If what you want to negate is a noun, negate it "
    "with مفيش or مش placed before it instead, the normal way Egyptian negates nouns. "
    "For weak-final verbs whose present tense ends in ى (نسي/تنسى-type roots), the ش "
    "suffix needs a vowel shift first — the negated form is تنساش (ى becomes ا before "
    "ش), never the MSA لا تنسى and never a bare ش glued onto the unshifted تنسى. "
    "Drop the بي- habitual prefix after a subjunctive/purpose trigger (عشان، لازم، علشان، "
    "إذا، لو) — say يكون not بيكون, يروح not بيروح right after one of those words; keep "
    "بي- only for a plain ongoing/habitual statement with no such trigger. "
    "Form-V/VI verbs (reflexive/passive-shaped MSA verbs) take Egyptian's تت-/اتـ prefix, "
    "never the bare MSA form: تتجنب (not تجنب) for avoiding, بيتفكك (not بيفكك) for coming "
    "apart, اتلهم (not استُلهم) for being inspired."
)

# ── Deterministic dialect-repair dictionary ───────────────────────────────────
# Closed-form wrong-word -> right-word pairs per ROUTED dialect, applied as a real
# post-synthesis substitution (generalizes the old Egyptian-only fix_egyptian_leaks,
# which only ever handled جداً->أوي). Every pair here is drawn from an EXISTING mention
# in NAJDI_GLOSSARY / EGYPTIAN_CARD's prose above — never add a pair here without a
# matching prompt-side mention (eval/test_dialect_repair.py checks this mechanically).
#
# ADMISSION BAR (deliberately narrow — do not relax without re-reading this): a pair
# qualifies ONLY if the wrong word is closed-form and unambiguous in a spoken reply —
# i.e. it has no other correct reading (proper noun, homograph, gender/number-dependent
# form) in ANY sentence a user will hear, AND it is actually wrong in that dialect (not
# just informal/uncommon — see the جداً correction below for why this second half of the
# bar matters just as much as the first). مرة does NOT qualify as a MATCH target (only
# as a Najdi replacement VALUE, where NAJDI_GLOSSARY's normal usage already validates
# it) — مرة is itself ambiguous ("very" / "once" / colloquially "wife"). دي/ده (Egyptian
# demonstratives) also do NOT qualify even though they leak (see BASELINES.md,
# Fusha/Najdi ده×1/دي×1): the correct Najdi/Fusha replacement depends on the referent
# noun's gender (هذا vs هذي), which this dictionary cannot know — stays prompt-only.
#
# CORRECTION (owner-confirmed, 2026-07-22): جداً/جدا is genuinely correct, natural
# Egyptian Arabic for "very" — it was WRONGLY treated as a leak here (and in
# EGYPTIAN_CARD, and in eval/leak_lint.FORBIDDEN["Egyptian"]) for two prior sessions,
# apparently inherited from an old branch's design choice that was never actually
# validated against genuine Egyptian usage. Every historical "leak" count driven by
# Egyptian جداً (most of them — see BASELINES.md) overstated the real defect rate. This
# does NOT extend to Najdi: جداً being wrong there (مرة is Najdi's correct word) is a
# separate, independently-confirmed, unchallenged finding across many eval runs.
#
# THIS RAISES THE FLOOR, NOT THE CEILING: a hardcoded dict only fixes words already
# known to be wrong. It is explicitly NOT a fix for a genuinely novel wrong word the
# model produces tomorrow — nothing deterministic can be, by definition. That requires
# either better prompting (own ceiling — see NAJDI_NO_OTHER_DIALECTS_RULE's revert
# above) or detection-plus-human-promotion (see eval/README.md's feedback-loop
# checklist).
DIALECT_REPAIR_MAP: dict[str, dict[str, str]] = {
    "Najdi": {"جداً": "مرة", "جدا": "مرة"},
    # Egyptian: MSA relative pronoun + two colloquial-spelling pairs, all closed-form and
    # unambiguous. NOTE: "مش فيه"→"مفيش" was considered (owner-given example, 2026-07-22
    # live-testing review) and REJECTED for this deterministic map — فيه is itself a
    # genuine Egyptian homograph ("there is" existential vs. "in it" locative — "الفلوس
    # مش فيه" can mean "not inside it"), so "مش فيه" fails the unambiguous-in-ANY-sentence
    # admission bar. EGYPTIAN_CARD already teaches "there isn't=مفيش" in its word list;
    # no further prompt or deterministic change needed for that pattern.
    "Egyptian": {"الذي": "اللي", "تمشى": "تمشي", "تأكل": "تاكل"},
    # Fusha: no entries — جداً is correct in Fusha, nothing to fix.
}

# Canonical tts_language -> curated-dialect-label map. Single source of truth for
# "which curated dialect does this TTS-language value belong to" — eval/leak_lint.py
# imports this instead of keeping its own copy.
TTS_LANG_TO_DIALECT: dict[str, str] = {
    "najdi arabic": "Najdi", "egyptian arabic": "Egyptian", "standard arabic": "Fusha",
}


def _repair_pattern(wrong_word: str) -> Any:
    """Word-boundary-safe pattern for one DIALECT_REPAIR_MAP key. Deliberately a
    RIGHT-SIDE guard only (matches when followed by whitespace/punctuation/EOS) — NOT
    `\\b` on both sides. Verified empirically: Python's \\b treats the tanween
    diacritic (ً, U+064B, a combining mark) as a non-word char, so `\\bجداً\\b` fails to
    match جداً at all. No left-side guard either — Arabic glues the و/ف conjunction
    directly onto the next word with no space (وجداً), a legitimate case this must
    still match; no real root has جدا/جداً as a bound suffix of an unrelated word
    (verified against جدول/جدال, and against جدة [Jeddah] which shares the same
    four-letter prefix but is never confused with جداً)."""
    return re.compile(re.escape(wrong_word) + r'(?=[\s،,.:؟!]|$)', re.UNICODE)


_REPAIR_PATTERNS: dict[str, list[tuple[Any, str]]] = {
    dialect: [(_repair_pattern(w), r) for w, r in pairs.items()]
    for dialect, pairs in DIALECT_REPAIR_MAP.items()
}


def apply_dialect_repairs(text: str, dialect: Optional[str]) -> str:
    """Deterministic post-processing substitution pass for a ROUTED dialect label
    ("Najdi"/"Egyptian"/"Fusha"/None — use TTS_LANG_TO_DIALECT.get(tts_language) to get
    one from a tts_language value). No-ops for Fusha/None/unrecognized labels."""
    for pattern, right_word in _REPAIR_PATTERNS.get(dialect or "", []):
        text = pattern.sub(right_word, text)
    return text


# ── Language detection & acceptance policy (moved verbatim from server.py) ────

ALLOWED_LANGS = {"ar", "en"}

# Whisper mistakes Arabic for these languages — remap them all to ar.
# Includes Arabic-script langs (ur/fa/ps/ug/sd) AND Punjabi (pa) which Whisper
# also confuses with Arabic despite different script.
ARABIC_SCRIPT_REMAP = {"ur", "fa", "ps", "ug", "prs", "ckb", "sd", "pa"}
MIN_TEXT_CHARS = 3
MAX_TEXT_CHARS = 500

# Detects code-switching: text contains both Arabic script and Latin words.
_ARABIC_CHARS_RE = re.compile(r'[؀-ۿ]')
_LATIN_WORDS_RE  = re.compile(r'[a-zA-Z]{2,}')

def is_mixed(text: str) -> bool:
    return bool(_ARABIC_CHARS_RE.search(text)) and bool(_LATIN_WORDS_RE.search(text))

# Explicit output-language requests ("...in Arabic", "بالعربي") — these override
# the auto-detected input language so the user can ask for a reply in any language.
WANTS_ARABIC_RE = re.compile(
    r"\b(in|into|to)\s+arabic\b"
    r"|reply\s+in\s+arabic|answer\s+in\s+arabic|say\s+it\s+in\s+arabic"
    r"|بالعرب|بالعربي|باللغة\s+العربية|بالفصحى|باللهجة",
    re.IGNORECASE | re.UNICODE,
)
WANTS_ENGLISH_RE = re.compile(
    r"\b(in|into|to)\s+english\b"
    r"|reply\s+in\s+english|answer\s+in\s+english|say\s+it\s+in\s+english"
    r"|بالانجليز|بالإنجليز|باللغة\s+الإنجليزية",
    re.IGNORECASE | re.UNICODE,
)

# Specific Arabic dialect requests, checked when the user asks for Arabic output.
# First match wins; falls back to Fusha/MSA when no dialect is named. Supported: Najdi,
# Fusha, Egyptian — a named request for any other dialect (Gulf/Khaleeji, Hijazi, ...)
# simply doesn't match here and falls through to the default Fusha routing in build_turn.
# Egyptian is deliberately checked LAST and via its guarded matcher (requested_egyptian):
# any text the Najdi/Fusha patterns match today keeps resolving exactly as today.
_DIALECT_PATTERNS: list[tuple[str, Any, str]] = [
    ("Najdi", re.compile(r"\bnajdi\b|نجدي|النجدية", re.IGNORECASE | re.UNICODE),
     "the Najdi dialect (use وش/إيش, أبغى, زين, الحين, ماله, يبيلك)"),
    ("Fusha", re.compile(r"\bfus-?ha\b|\bmsa\b|modern\s+standard|classical\s+arabic|الفصحى|فصحى",
                         re.IGNORECASE | re.UNICODE),
     "Modern Standard Arabic (Fusha)"),
]

def requested_dialect(text: str) -> tuple[Optional[str], Optional[str]]:
    """Return (dialect_name, descriptive_phrase) for an explicitly named Arabic dialect, or
    (None, None) for default (Fusha)."""
    for name, pattern, phrase in _DIALECT_PATTERNS:
        if pattern.search(text):
            return name, phrase
    if requested_egyptian(text):
        return "Egyptian", _EGYPTIAN_REQUEST_PHRASE
    return None, None

# Strips CJK, full-width punctuation (？！), and Cyrillic from LLM tokens.
_UNWANTED_SCRIPT_RE = re.compile(
    r"[一-鿿"          # CJK unified ideographs
    r"㐀-䶿"           # CJK extension A
    r"豈-﫿"           # CJK compatibility ideographs
    r"　-〿"           # CJK symbols & punctuation
    r"゠-ヿ"           # katakana
    r"぀-ゟ"           # hiragana
    r"가-힯"           # hangul syllables
    r"＀-￯"           # fullwidth/halfwidth forms incl. ？！
    r"Ѐ-ӿ"           # Cyrillic
    r"Ԁ-ԯ]+",        # Cyrillic supplement
    re.UNICODE,
)

async def filter_cjk(token_gen: Any):
    try:
        async for token in token_gen:
            cleaned = _UNWANTED_SCRIPT_RE.sub("", token)
            if cleaned:
                yield cleaned
    finally:
        # Pass close() through to the source generator so cancelling TTS also
        # tears down the underlying httpx stream (Ollama stops generating).
        aclose = getattr(token_gen, "aclose", None)
        if aclose is not None:
            await aclose()

# Detects ASR stuck-loops: "ا ا ا ا" or "هل هل هل هل"
REPETITION_RE = re.compile(r"(.)\1{4,}|(\b\S+\b)(\s+\2){3,}", re.UNICODE)

# Prompt injection patterns (Arabic + English + Urdu).
# "you are now" requires a role-assignment continuation (a/an/the/my) — the bare
# phrase false-positives on innocent speech like "you are now able to see it".
INJECTION_RE = re.compile(
    r"ignore\s+(previous|prior|all)\s+instructions?"
    r"|تجاهل\s+(التعليمات|الأوامر|السابق)"
    r"|forget\s+(your\s+)?(previous|prior|all)"
    r"|you\s+are\s+now\s+(a|an|the|my)\b"
    r"|نسيان\s+التعليمات"
    r"|<\s*(system|instructions?)\s*>"
    r"|system\s*:",
    re.IGNORECASE | re.UNICODE,
)


# ── Najdi glossary for LLM turn instructions ──────────────────────────────────
# The full MSA→Najdi vocabulary table (user-curated, includes the water-utility
# domain terms). Embedded per-turn on Najdi turns only, so Fusha/English turns
# pay no prompt-token cost for it.
NAJDI_GLOSSARY = (
    "Use these Najdi word choices instead of their MSA equivalents: "
    "وش/إيش (ما/ماذا)، ليش (لماذا)، وين (أين)، كيف الحال (كيف حالك)، زين (جيد)، "
    "عاد (حسناً/إذن)، الحين (الآن)، بعدين (بعد ذلك)، شوي (قليلاً)، صج (حقاً/فعلاً)، "
    "ما أدري (لا أعلم)، أبغى (أريد)، ما عندي (ليس عندي)، خلاص (انتهى/كفى)، "
    "يلا (هيا)، بس (فقط/لكن)، مرة (جداً)، لو سمحت (من فضلك)، هيه (نعم)، "
    "ما في (لا يوجد)، في (يوجد)، شغال (يعمل/يشتغل)، خربان (معطل/خراب)، "
    "عال/عالي (مرتفع)، واطي (منخفض)، فوق (أعلى)، تحت (أسفل)، ممتلي (ممتلئ)، "
    "فاضي (فارغ)، وسخ (ملوث)، بطي (بطيء). "
    "These words are the same in Najdi and MSA — use them as-is: "
    "كثير، طبيعي، رقم، قراءة، معدل، ضغط، تدفق، خزان، عداد، محطة، خط، تنبيه، "
    "مشكلة، انقطاع، تسريب، نظيف، سريع، شكراً، لا."
)

# Grammar rule (not a word list — a general pattern to avoid, so it generalizes to
# vocabulary not covered above instead of only fixing memorized cases). Found via
# eval: the model frequently uses the بـ + imperfective prefix (بيفيدك، بتقوي،
# بيسبب) — that verb-tense marking is Levantine/Egyptian, not Najdi/Gulf at all.
NAJDI_GRAMMAR_RULE = (
    "Grammar note: do NOT put بـ before present-tense verbs (that pattern — بيروح، "
    "بتقول، بنعرف، بيصير — is Levantine/Egyptian Arabic, not Najdi). Use the plain "
    "Najdi imperfective instead: يروح (not بيروح), تقول (not بتقول), نعرف (not بنعرف), "
    "يصير (not بيصير)."
)

# TRIED AND REVERTED (not wired into llm.py — kept here only as a documented dead
# end so it isn't re-attempted blindly). Targeted a second cross-dialect leak
# (Moroccan "باش", Egyptian ما...ش negation) the same way NAJDI_GRAMMAR_RULE
# fixed the بـ-prefix leak. Measured result on the held-out set: the leak rate
# went UP (1.7%→8.3%, all 5 new instances manually confirmed genuine, not a
# counting artifact) instead of down. Likely cause: unlike the بـ-prefix rule
# (which describes a GRAMMAR PATTERN), this rule spells out the exact forbidden
# WORDS ("باش", "ما فيش") in the prompt — a "don't think of a pink elephant"
# effect, where naming the token increases its salience instead of suppressing
# it. If this gets revisited, try phrasing that avoids stating the literal
# banned words (e.g. "negate only with plain ما, never a ما...X pattern") rather
# than repeating this version, and validate on the held-out set before keeping it.
NAJDI_NO_OTHER_DIALECTS_RULE = (
    "Do NOT use باش (that is Moroccan Arabic for 'so that' — use عشان instead). "
    "Do NOT add ش to the end of ما for negation, e.g. ما فيش or ما تنفعش (that is "
    "Egyptian Arabic — Najdi negates with plain ما, e.g. ما في, ما ينفع)."
)
