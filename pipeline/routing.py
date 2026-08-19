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
    "ابغى", "يبيلك", "ماله", "هيه", "يلا", "خربان", "واطي", "فاضي", "ممتلي", "بطي",
    "ادري",  # from "ما أدري" when tokenized

    # Added via eval (dialect_eval_holdout_report.md): each entry below is either
    # a grammatical conjugation of an already-vetted word above, or a standalone
    # word that essentially never occurs in real Fusha text — NOT a general
    # "sounds informal" word (those were deliberately left out; see the report
    # for the ones considered and rejected: قدر/يقدر family collides with real
    # MSA usage "لا يقدر على"/"بقدر الإمكان"; راح alone collides with MSA "went";
    # وسايل/يقرا are just hamza-dropped spellings of ordinary MSA words, not a
    # dialect signal).
    "تبغى", "يبغى", "نبغى", "تبغين",           # ابغى/أبغى conjugations
    "تشوف", "يشوف", "نشوف", "شاف", "شفت",       # "to see" — Fusha uses يرى/رأى
    "تسوي", "يسوي", "نسوي", "سويت",             # "to do/make" — NOT bare "سوى" (= MSA "except")
    "اللي",                                     # colloquial relative pronoun — Fusha uses الذي/التي
    "لسا", "لسه",                               # "still/yet" — Fusha uses لا يزال/بعد
    "ماي", "مويه",                              # colloquial "water" — Fusha writes الماء/المياه
    "عشان",                                     # "so that/because" — Fusha uses لكي/حتى
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


# ── Marker tiers for Arabic dialect routing (Najdi / Egyptian / Fusha) ────────
# اللي/عشان/لسا/لسه/يلا are heavily used in BOTH Najdi and Egyptian (pan-dialectal).
# They STAY in _NAJDI_MARKERS — looks_najdi() above is also the per-sentence CATT
# gate inside the TTS module, and removing them would change protected behavior
# (Najdi recall + tashkeel gating). For ROUTING they are non-decisive: only
# exclusive evidence picks a dialect; shared markers alone keep today's Najdi
# routing (see route_arabic below).
_SHARED_MARKERS = {normalize_ar(w) for w in {"اللي", "عشان", "لسا", "لسه", "يلا"}}
_NAJDI_EXCLUSIVE = _NAJDI_MARKERS - _SHARED_MARKERS   # derived — the literal set is untouched

# Distinctly-Egyptian words only (normalized). Same philosophy as _NAJDI_MARKERS:
# high-precision, exact-token. Negation morphology and demonstratives are the
# strongest published Egyptian signals; discourse particles the noisiest.
_EGY_EXCLUSIVE_MARKERS = {normalize_ar(w) for w in {
    "دلوقتي", "دلوقت",                    # "now" — Najdi uses الحين
    "النهارده", "النهاردة",               # "today" — stored with its ال built in
    "امبارح",                             # "yesterday" — Najdi uses أمس/البارح
    "ازاي", "ازيك",                       # "how / how are you" — Najdi uses كيف
    "عايز", "عايزه", "عايزة", "عاوز", "عاوزه", "عاوزة",   # "want" — Najdi uses أبغى
    "بتاع", "بتاعه", "بتاعة", "بتاعت",     # possessive "of/belonging to"
    "كده", "كدا",                         # "like this" — Najdi uses كذا (different token)
    "ده", "دي",                           # post-nominal demonstratives — Najdi uses هذا/ذا
    "مش",                                 # negator — Najdi uses مو/مب, Fusha ليس
    "فين",                                # "where" — Najdi uses وين
    # Curated suffix-negation forms (a generic ما...ش regex would false-positive
    # on MSA words like مندهش/مفتش — only these vetted forms match):
    "مفيش", "مافيش", "معرفش", "ماعرفش", "مقدرش", "ماقدرش",
    "محدش", "معنديش", "معندكش",

    # Evaluated and REJECTED (do not add back without eval):
    #   ليه    — real Gulf/Najdi usage ("ليه ما جيت؟"); would steal Gulf turns.
    #            Present in EGYPTIAN_CARD as output vocabulary only.
    #   ايه    — Gulf/Najdi إيه = "yes"; matched only via _EGY_PHRASES_RE contexts.
    #   دول    — MSA دُوَل "countries" (الدول ال-strips to دول). Valid Egyptian, but
    #            as a MARKER it misroutes MSA sentences about countries. Never on
    #            any forbidden list either (hard user constraint).
    #   جداً/جدا — valid Egyptian; never a marker, never forbidden (hard constraint).
    #   بكره/برضه/ايوه/خالص/قوي/معلش/حاجة/كمان/منين — shared with Gulf/Hijazi or MSA.
    #   عربية  — "car" collides with "Arabic"; طب — MSA "medicine"; يبقى — MSA "remains".
    #   بص    — STT-confusable with بس; ه-future regex (هروح...) — too broad.
}}
# Multi-word Egyptian expressions — safe contexts for ايه.
_EGY_PHRASES_RE = re.compile(
    r"\bعامل ايه\b|\bعامله ايه\b|\bايه الاخبار\b|\bايه رايك\b",
    re.UNICODE,
)
# ال-strip candidates match only ≥4-char markers: ألفين ("two thousand", common in
# amounts) normalizes to الفين and would ال-strip to فين — an MSA/Najdi sentence
# about money must not route Egyptian. The guard also structurally blocks strip
# matches onto the short markers ده/دي/مش/كده.
_EGY_STRIP_SAFE = {m for m in _EGY_EXCLUSIVE_MARKERS if len(m) >= 4}


def looks_egyptian(text: str) -> bool:
    """True if the text carries distinctly-Egyptian vocabulary. Same algorithm
    family as looks_najdi: normalize → exact tokens → ال-strip → و/ف-deprefix →
    phrases. Used ONLY for routing (route_arabic) — never for the CATT gate."""
    norm = normalize_ar(text)
    words = set(_AR_WORD_RE.findall(norm))
    if words & _EGY_EXCLUSIVE_MARKERS:
        return True
    stripped = {w[2:] for w in words if w.startswith("ال") and len(w) > 4}
    if stripped & _EGY_STRIP_SAFE:
        return True
    # و/ف conjunction-prefix strip (وفين، وعايز، فكده...). Remainder must be ≥3
    # chars so a strip can never yield the 2-char demonstratives: Najdi ودي
    # ("I'd like" — "ودي أشوف") must NOT become دي. بـ is deliberately NOT
    # stripped: بدي is Levantine "I want", and ب+verb is dialect-ambiguous
    # (Egyptian habitual بروح vs Gulf FUTURE بروح — same string, opposite signal).
    deprefixed = {w[1:] for w in words if w[0] in "وف" and len(w) > 3}
    if deprefixed & _EGY_EXCLUSIVE_MARKERS:
        return True
    return bool(_EGY_PHRASES_RE.search(norm))


def looks_najdi_exclusive(text: str) -> bool:
    """looks_najdi restricted to markers NOT shared with Egyptian. Same algorithm.
    Used only for routing priority — looks_najdi itself stays the CATT gate and
    the shared-tier fallback, unchanged."""
    norm = normalize_ar(text)
    words = set(_AR_WORD_RE.findall(norm))
    stripped = {w[2:] for w in words if w.startswith("ال") and len(w) > 4}
    if (words | stripped) & _NAJDI_EXCLUSIVE:
        return True
    return bool(_NAJDI_PHRASES_RE.search(norm))


def route_arabic(text: str) -> str:
    """Route a spoken-Arabic utterance to a reply dialect (tts_language value).

    Priority: Najdi-exclusive > Egyptian-exclusive > shared-only (→ Najdi,
    today's behavior) > Fusha. Shared markers never route to Egyptian; for any
    input with zero Egyptian-exclusive evidence this returns exactly what
    `"najdi arabic" if looks_najdi(text) else "standard arabic"` returned before
    Egyptian existed (branch 1 ∪ branch 3 ≡ looks_najdi). Najdi wins conflicts
    ("وش يعني دلوقتي؟" → Najdi) — protecting the established dialects comes first."""
    if looks_najdi_exclusive(text):
        return "najdi arabic"
    if looks_egyptian(text):
        return "egyptian arabic"
    if looks_najdi(text):
        return "najdi arabic"
    return "standard arabic"


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
# First match wins; falls back to Fusha/MSA when no dialect is named. Najdi,
# Egyptian and Fusha are the supported dialects — a named request for any other
# dialect (Gulf/Khaleeji, Hijazi, ...) simply doesn't match here and falls
# through to the default Fusha routing in build_turn.
# The Egyptian pattern is deliberately TIGHT: this regex runs on every turn, and
# "مصري/egyptian" is common CONTENT ("الاقتصاد المصري", "egyptian pyramids") — a
# bare match would hijack those turns into explicit-dialect requests. Request
# phrasings only.
_DIALECT_PATTERNS: list[tuple[str, Any, str]] = [
    ("Najdi", re.compile(r"\bnajdi\b|نجدي|النجدية", re.IGNORECASE | re.UNICODE),
     "the Najdi dialect (use وش/إيش, أبغى, زين, الحين, ماله, يبيلك)"),
    ("Egyptian", re.compile(
        r"بالمصري|باللهجه\s+المصريه|باللهجة\s+المصرية"
        r"|\b(in|into|to)\s+egyptian\s+arabic\b"
        r"|\begyptian\s+dialect\b"
        r"|speak\s+egyptian|reply\s+in\s+egyptian|answer\s+in\s+egyptian|say\s+it\s+in\s+egyptian",
        re.IGNORECASE | re.UNICODE),
     "Egyptian Arabic (Masri — use دلوقتي, عايز, إزاي, كده, مش)"),
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


# ── Egyptian card for LLM turn instructions ──────────────────────────────────
# Embedded per-turn on Egyptian-routed turns ONLY — Fusha/Najdi/English turns
# never see it (zero token cost, zero cross-dialect prompt risk).
# PINK-ELEPHANT RULE (see NAJDI_NO_OTHER_DIALECTS_RULE above for the measured
# failure): every rule states Egyptian's own correct forms POSITIVELY; no other
# dialect's tokens are ever named as forbidden. جداً/جدا is valid Egyptian and is
# deliberately not discouraged anywhere; دول appears only as positive vocabulary.
EGYPTIAN_CARD = (
    "Use these Egyptian (Masri) word choices instead of their MSA equivalents: "
    "دلوقتي (الآن)، النهارده (اليوم)، امبارح (أمس)، إزاي (كيف)، ايه (ماذا)، فين (أين)، "
    "ليه (لماذا)، عايز/عايزة (أريد)، مش (ليس)، كده (هكذا)، ده/دي/دول (هذا/هذه/هؤلاء)، "
    "اللي (الذي/التي)، برضه (أيضاً)، كمان (أيضاً)، معلش (لا بأس)، خالص (إطلاقاً)، "
    "أوي (كثيراً)، بتاع (خاص بـ)، حاجة (شيء)، عشان (لأن/لكي)، لسه (ما زال). "
    "These words are the same in Egyptian and MSA — use them as-is: "
    "رقم، قراءة، معدل، ضغط، تدفق، خزان، عداد، محطة، خط، تنبيه، "
    "مشكلة، انقطاع، تسريب، نظيف، سريع، شكراً، لا. "
    "Grammar — write natural spoken Cairo Egyptian: "
    "Negate verbs by wrapping them with ما...ش: ما عرفتش، مفيش، ما قدرتش. "
    "Use مش before nouns, adjectives and future verbs: مش عارف، مش هروح. "
    "Mark habitual/progressive present with the بـ prefix: بقول، بيشرح، بنروح — "
    "this prefix is correct Egyptian. Mark future with هـ: هروح، هيبقى، هنشوف. "
    "Put demonstratives AFTER the noun: الكتاب ده، الفكرة دي. "
    "Use بتاع for possession: العداد بتاعك."
)
