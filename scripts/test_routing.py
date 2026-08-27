"""
test_routing.py — no-GPU tests for the Egyptian dialect integration.
====================================================================
Three layers of protection for the Fusha/Najdi baseline:

  1. SNAPSHOT: every row of scripts/fixtures_routing_baseline.json (captured from
     the tree BEFORE the Egyptian work, 60 non-Egyptian utterances) must produce a
     byte-identical (turn_content, tts_language) from the new build_turn, and an
     identical looks_najdi() — proof the protected routes did not move.
  2. DECISION TABLE: Egyptian positives, collision guards, request-hijack guards.
  3. MECHANISMS: _visible_history isolation scenarios, the fanar think-stripper,
     the Egyptian repairs filter, LLM_MODEL config matching.

Run with:
    /home/taha/first_voice_test/.venv/bin/python /home/taha/first_voice_test/scripts/test_routing.py
(no GPU used; imports are heavy because server.py pulls torch — allow ~20 s)
"""

import asyncio
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

if os.environ.get("LLM_MODEL"):
    print("LLM_MODEL is set — this suite validates the DEFAULT (qwen) path and its "
          "baseline snapshot. Run it without LLM_MODEL.")
    sys.exit(2)

from pipeline import llm, routing                       # noqa: E402
from pipeline import tts_voicetut_v1 as vt              # noqa: E402
import server                                           # noqa: E402  (for _visible_history)

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {name}  {detail}")


# ── 1. Baseline snapshot: protected routes byte-identical ─────────────────────
print("[1] baseline snapshot (60 non-Egyptian utterances)")
with open(os.path.join(_PROJECT_ROOT, "scripts", "fixtures_routing_baseline.json"),
          encoding="utf-8") as f:
    baseline = json.load(f)

for row in baseline:
    turn_content, tts_language, _explicit = llm.build_turn(row["text"], row["lang"])
    check(f"snapshot tts_language: {row['text'][:40]}",
          tts_language == row["tts_language"],
          f"expected {row['tts_language']!r} got {tts_language!r}")
    check(f"snapshot turn_content: {row['text'][:40]}",
          turn_content == row["turn_content"], "turn_content diverged")
    check(f"snapshot looks_najdi: {row['text'][:40]}",
          routing.looks_najdi(row["text"]) == row["looks_najdi"], "looks_najdi flipped")

# ── 2. Egyptian routing positives ─────────────────────────────────────────────
print("[2] Egyptian routing")
EGY_POSITIVE = [
    "عايز أعرف الأخبار النهارده.",
    "إزاي أروح المتحف؟",
    "مش فاهم حاجة.",
    "معرفش فين المفاتيح.",
    "عايز أعرف اللي حصل امبارح.",      # shared اللي + exclusive عايز/امبارح
    "عشان كده مش هروح.",               # shared عشان + exclusive كده/مش
    "وفين المحطة؟",                    # و-prefix strip
    "الكتاب ده كويس؟",
    "دلوقتي الدنيا حر.",
    "هو ايه رايك في الموضوع؟",         # phrase context for ايه
]
for t in EGY_POSITIVE:
    check(f"egyptian: {t[:40]}", routing.route_arabic(t) == "egyptian arabic",
          f"got {routing.route_arabic(t)!r}")

# Conflict + collision guards
print("[3] guards")
check("Najdi wins conflicts", routing.route_arabic("وش يعني دلوقتي؟") == "najdi arabic")
check("ودي is not Egyptian", routing.route_arabic("ودي أشوف التقرير.") == "standard arabic")
check("ليه non-decisive", routing.route_arabic("ليه ما جيت؟") == "standard arabic")
check("ألفين strip-guard", routing.route_arabic("عندي ألفين ريال في الحساب.") == "standard arabic")
check("دول (MSA countries) not Egyptian",
      routing.route_arabic("ما عدد الدول العربية؟") == "standard arabic")
check("shared-only stays Najdi (اللي)", routing.route_arabic("اللي صار أمس غريب.") == "najdi arabic")
check("shared-only stays Najdi (عشان)", routing.route_arabic("عشان كذا ما جيت.") == "najdi arabic")
check("جداً not a marker anywhere",
      "جدا" not in routing._EGY_EXCLUSIVE_MARKERS
      and routing.normalize_ar("جداً") not in routing._EGY_EXCLUSIVE_MARKERS)
check("جداً/دول never in repair map",
      "جداً" not in vt._EGY_REPAIR_MAP and "جدا" not in vt._EGY_REPAIR_MAP
      and "دول" not in vt._EGY_REPAIR_MAP)

# requested_dialect: Egyptian requests match, content mentions do not
print("[4] requested_dialect")
check("بالمصري", routing.requested_dialect("اتكلم معايا بالمصري")[0] == "Egyptian")
check("باللهجة المصرية", routing.requested_dialect("رد باللهجة المصرية")[0] == "Egyptian")
check("in egyptian arabic", routing.requested_dialect("say that in egyptian arabic")[0] == "Egyptian")
check("egyptian dialect", routing.requested_dialect("use the egyptian dialect")[0] == "Egyptian")
check("no hijack: الاقتصاد المصري",
      routing.requested_dialect("الاقتصاد المصري في نمو مستمر.")[0] is None)
check("no hijack: egyptian pyramids",
      routing.requested_dialect("tell me about egyptian pyramids")[0] is None)
check("najdi still first priority", routing.requested_dialect("رد بالنجدية مش بالمصري")[0] == "Najdi")

# build_turn Egyptian integration
print("[5] build_turn Egyptian")
tc, tl, ex = llm.build_turn("اتكلم معايا بالمصري", "ar")
check("explicit Egyptian request → route", tl == "egyptian arabic")
check("explicit Egyptian request → explicit flag", ex is True)
check("explicit Egyptian request → card present", routing.EGYPTIAN_CARD in tc)
tc, tl, ex = llm.build_turn("معرفش فين المفاتيح.", "ar")
check("detected Egyptian → route", tl == "egyptian arabic")
check("detected Egyptian → not explicit", ex is False)
check("detected Egyptian → card present", routing.EGYPTIAN_CARD in tc)
check("detected Egyptian → Najdi glossary absent", routing.NAJDI_GLOSSARY not in tc)
tc, tl, ex = llm.build_turn("وش الأخبار اليوم؟", "ar")
check("Najdi turn → card absent", routing.EGYPTIAN_CARD not in tc)
tc, tl, ex = llm.build_turn("أنا رايح الـ meeting دلوقتي.", "mixed")
check("mixed Egyptian → tts None (OmniVoice)", tl is None)
check("mixed Egyptian → Masri instruction", "Egyptian Arabic (Masri)" in tc)
check("mixed Egyptian → card NOT appended", routing.EGYPTIAN_CARD not in tc)

# ── 6. _visible_history isolation scenarios ───────────────────────────────────
print("[6] _visible_history")
H = [
    {"role": "user", "content": "fusha-q", "tag": "standard arabic"},
    {"role": "assistant", "content": "fusha-a", "tag": "standard arabic"},
    {"role": "user", "content": "najdi-q", "tag": "najdi arabic"},
    {"role": "assistant", "content": "najdi-a", "tag": "najdi arabic"},
    {"role": "user", "content": "egy-q", "tag": "egyptian arabic"},
    {"role": "assistant", "content": "egy-a", "tag": "egyptian arabic"},
    {"role": "user", "content": "en-q", "tag": None},
    {"role": "assistant", "content": "en-a", "tag": None},
]
vis = server._visible_history(H, "najdi arabic")
got = [m["content"] for m in vis]
check("najdi turn sees najdi+english only", got == ["najdi-q", "najdi-a", "en-q", "en-a"], str(got))
vis = server._visible_history(H, "egyptian arabic")
got = [m["content"] for m in vis]
check("egyptian turn sees egyptian+english only", got == ["egy-q", "egy-a", "en-q", "en-a"], str(got))
vis = server._visible_history(H, None)
check("english turn sees ALL history", len(vis) == 8)
# NO bypass (owner decision 2026-08-21): an explicit request is filtered like any
# other Arabic turn — the old explicit-sees-all behavior was the Fanar cross-dialect
# copy/contamination vector. English (None-tagged) pairs remain visible, so
# "tell me that in <dialect>" after an ENGLISH answer still has its referent.
vis = server._visible_history(H, "standard arabic")
got = [m["content"] for m in vis]
check("no bypass: fusha turn sees fusha+english only",
      got == ["fusha-q", "fusha-a", "en-q", "en-a"], str(got))
vis = server._visible_history(H, "najdi arabic")
check("tags stripped from prompt messages", all("tag" not in m for m in vis))
check("switch-back restores (najdi visible again after egyptian turns)",
      [m["content"] for m in server._visible_history(H, "najdi arabic")][:2]
      == ["najdi-q", "najdi-a"])

# ── 7. fanar wiring ───────────────────────────────────────────────────────────
print("[7] fanar wiring")
check("default model unchanged when LLM_MODEL unset",
      os.environ.get("LLM_MODEL") is not None or llm.MODEL == "qwen3.5:27b")
cfg = llm.get_model_config("hf.co/mradermacher/Fanar-2-27B-Instruct-i1-GGUF:i1-Q4_K_M")
check("fanar tag matches fanar config", cfg is llm.MODEL_CONFIGS["fanar"])
check("fanar config thinks off", cfg["extra"].get("think") is False)
check("fanar num_ctx matches warm-up", cfg["options"]["num_ctx"] == llm.LLM_NUM_CTX)
check("qwen config untouched",
      llm.get_model_config("qwen3.5:27b") is llm.MODEL_CONFIGS["qwen3.5"])


async def _collect(gen):
    return "".join([t async for t in gen])


async def _gen(tokens):
    for t in tokens:
        yield t

think_cases = [
    (["<think>secret reasoning</think>Hello", " world"], "Hello world"),
    (["<th", "ink>x</th", "ink> Result"], " Result"),
    (["No thinking here."], "No thinking here."),
    (["ok<"], "ok<"),                                     # trailing lookalike survives
    (["<think>never closed..."], ""),                     # unterminated think discarded
]
for toks, want in think_cases:
    got = asyncio.run(_collect(llm.strip_think_tokens(_gen(toks))))
    check(f"strip_think {toks!r}", got == want, f"got {got!r}")

# Identity note: fanar-only, default system prompt byte-identical when unset
if os.environ.get("LLM_MODEL") is None:
    check("ACTIVE_SYSTEM_PROMPT identical to SYSTEM_PROMPT on qwen default",
          llm.ACTIVE_SYSTEM_PROMPT == llm.SYSTEM_PROMPT)
    check("identity note absent on qwen default",
          llm.FANAR_IDENTITY_NOTE not in llm.ACTIVE_SYSTEM_PROMPT)

md_cases = [
    (["**مهم**: نعم"], "مهم: نعم"),
    (["\n1. **زيادة", " الإنتاج**: تزيد"], "\n1. زيادة الإنتاج: تزيد"),
    (["- bullet\n- two"], "bullet\ntwo"),            # bullets at line starts dropped
    (["a - b"], "a - b"),                            # mid-text hyphen preserved
    (["\n-", " item"], "\nitem"),                    # bullet marker split across tokens
    (["\n- 2020-2023 range"], "\n2020-2023 range"),  # only the leading bullet goes
    (["# عنوان\nنص"], " عنوان\nنص"),
]
for toks, want in md_cases:
    got = asyncio.run(_collect(llm.strip_markdown_tokens(_gen(toks))))
    check(f"strip_markdown {toks!r}", got == want, f"got {got!r}")

# ── 8. Egyptian repairs filter ────────────────────────────────────────────────
print("[8] repairs filter")
repair_cases = [
    (["الذي ", "قال تمشى"], "اللي قال تمشي"),
    (["الكتاب الذي", " قرأته"], "الكتاب اللي قرأته"),
    (["تأكل ايه النهارده؟"], "تاكل ايه النهارده؟"),
    (["جداً مهم"], "جداً مهم"),                            # جداً untouched (valid Egyptian)
    (["والذي معه"], "والذي معه"),                          # exact-word only, prefixed word untouched
]
for toks, want in repair_cases:
    got = asyncio.run(_collect(vt._egyptianize_tokens(_gen(toks))))
    check(f"repairs {toks!r}", got == want, f"got {got!r}")

wayed_cases = [
    (["ووايد الناس بتروح هناك"], "وكتير الناس بتروح هناك"),
    (["الجو حلو وايد"], "الجو حلو كتير"),
]
for toks, want in wayed_cases:
    got = asyncio.run(_collect(vt._egyptianize_tokens(_gen(toks))))
    check(f"wayed repair {toks!r}", got == want, f"got {got!r}")

pron_cases = [
    ("المشي مفيد للصحة", "اَلْمَشْي مفيد للصحة"),
    ("بحب مشاهدة الأفلام", "بحب مُشَاهَدَة الأفلام"),
    ("التنبؤ بالطلب مهم", "اَلتَّنَبُّؤ بالطلب مهم"),
    ("والمشي كمان مفيد", "والمشي كمان مفيد"),   # prefixed form untouched (exact-token only)
    ("الوقوف بعرفة يوم عرفة", "الوقوف بِعَرَفَة يوم عَرَفَة"),
    # برا context rules: travel collocation → by-land (tanween); else → outside
    ("بيحبوا يسافروا برا", "بيحبوا يسافروا بَرًّا"),
    ("السفر برا أرخص من الطيران", "السفر بَرًّا أرخص من الطيران"),
    ("ممكن تروح للسفر برا", "ممكن تروح للسفر بَرًّا"),
    ("يوصلوا برا وبحرا", "يوصلوا بَرًّا وبحرا"),
    ("هو قاعد برا دلوقتي", "هو قاعد بَرَّا دلوقتي"),
    ("استنى برا البيت شوية", "استنى بَرَّا البيت شوية"),
    ("فاتورة شهرية في التقويم الهجري", "فاتورة شَهْرِيَّة في التقويم الهِجْرِيّ"),
    ("ويحللوها مخبريًا للتأكد", "ويحللوها مَخْبَرِيًّا للتأكد"),
    ("عشان يضمنوا مستقبل أفضل", "عشان يَضْمَنُوا مستقبل أفضل"),
]
for src, want in pron_cases:
    got = vt._apply_pronunciation_fixes(src)
    check(f"pronunciation {src[:30]!r}", got == want, f"got {got!r}")

# Owner fixes 2026-08-24
pron24_cases = [
    ("لازم يتصل بالشركة ويبلغهم بالمشكلة", "لازم يتصل بالشركة وِيبَلَّغْهُم بالمشكلة"),
    ("يبلغ المبلغ مئة ريال", "يبلغ المبلغ مئة ريال"),            # bare homograph untouched
    ("وبيرفعوا عدد الفرق الفنية", "وبِيِرْفَعُوا عدد الفرق الفنية"),
]
for src, want in pron24_cases:
    got = vt._apply_pronunciation_fixes(src)
    check(f"pron24 {src[:25]!r}", got == want, f"got {got!r}")

check("911 digit-by-digit (Arabic, OmniVoice)",
      "تسعة واحد واحد" in __import__("pipeline.tts_omnivoice_v1", fromlist=["x"])._expand_abbreviations(
          "اتصل على الرقم الموحد 911 فوراً."))
check("911 untouched in English (OmniVoice)",
      "911" in __import__("pipeline.tts_omnivoice_v1", fromlist=["x"])._expand_abbreviations(
          "Please call 911 immediately."))
check("911 digit-by-digit (Egyptian)",
      "تسعة واحد واحد" in vt._expand_abbreviations("اتصل بالرقم 911 حالاً."))
check("الستينيات repair (fanar)",
      asyncio.run(_collect(llm.repair_words(_gen(["في الستينيات بدأت المشاريع"]), llm.FANAR_ARABIC_REPAIRS)))
      == "في الستينات بدأت المشاريع")

check("VoiceTut ref clip exists", os.path.exists(vt._REF_AUDIO), vt._REF_AUDIO)
check("VoiceTut never does CATT", not hasattr(vt, "_add_tashkeel"))

# ── 8b. fanar output guards ───────────────────────────────────────────────────
print("[8b] fanar output guards")
najdi_guard_cases = [
    (["الفترة دي حلوة"], "الفترة هذي حلوة"),
    (["عايز أعرف دلوقتي"], "أبغى أعرف الحين"),
    (["بيسجل كل حاجة"], "يسجل كل حاجة"),          # حاجة deliberately untouched
    (["الشهر دا مهم"], "الشهر هذا مهم"),
    (["فريق قوي جداً"], "فريق قوي جداً"),          # homographs/constraints never repaired
]
for toks, want in najdi_guard_cases:
    got = asyncio.run(_collect(llm.repair_words(_gen(toks), llm.FANAR_NAJDI_REPAIRS)))
    check(f"najdi guard {toks!r}", got == want, f"got {got!r}")

for word in ("قوي", "بقى", "دول", "جداً", "جدا", "اللي", "عشان", "لسه", "يلا"):
    check(f"excluded from najdi guard: {word}", word not in llm.FANAR_NAJDI_REPAIRS)

kingdom_cases = [
    (["السياسات الوطنية للملكة العربية"], "السياسات الوطنية للمملكة العربية"),
    (["في الملكة العربية السعودية"], "في المملكة العربية السعودية"),
]
for toks, want in kingdom_cases:
    got = asyncio.run(_collect(llm.repair_words(_gen(toks), llm.FANAR_ARABIC_REPAIRS)))
    check(f"kingdom repair {toks!r}", got == want, f"got {got!r}")

tc, tl, ex = llm.build_turn("وش الأخبار اليوم؟", "ar")
check("FANAR_ARABIC_NOTE absent on default (qwen) path", llm.FANAR_ARABIC_NOTE not in tc)
check("يبيلك repaired in Egyptian",
      asyncio.run(_collect(vt._egyptianize_tokens(_gen(["اللي يبيلك إياه"])))) == "اللي محتاج إياه")

# ── 9. segment-wise CATT (mocked — no model load) ─────────────────────────────
print("[9] segment-wise CATT")
from pipeline import tts_omnivoice_v1 as ov   # noqa: E402


class _FakeCatt:
    def do_tashkeel(self, text, verbose=False):
        return f"[{text}]"   # wraps each CATT call so segmentation is visible


ov._tashkeel_model = _FakeCatt()   # pre-seed the lazy singleton — real CATT never loads

# (inputs carry no trailing punctuation here — these tests isolate SEGMENTATION;
# the punctuation/pausal layer has its own direct tests below)
check("pure Arabic keeps the single-call path",
      ov._add_tashkeel("نص عربي فقط") == "[نص عربي فقط]")
got = ov._add_tashkeel("يمكنك استخدام تطبيق Hydrate لتتبع الاستهلاك")
check("mixed: English preserved, Arabic segmented",
      got == "[يمكنك استخدام تطبيق] Hydrate [لتتبع الاستهلاك]", repr(got))
got = ov._add_tashkeel("شركة NWC توفر الخدمات")
check("mixed: acronym preserved", got == "[شركة] NWC [توفر الخدمات]", repr(got))
got = ov._add_tashkeel("تطبيقات WaterMinder و Plant Nanny مفيدة")
check("mixed: lone و between English untouched (<2-letter core guard)",
      got == "[تطبيقات] WaterMinder و Plant Nanny [مفيدة]", repr(got))

# digit guard: CATT never sees digit runs (they'd be deleted — verified live)
got = ov._add_tashkeel("الفاتورة رقم 123.5 صدرت أمس")
check("digit run protected from CATT", got == "[الفاتورة رقم] 123.5 [صدرت أمس]", repr(got))

# punctuation guard (2026-08-27): commas/quotes survive, clauses CATT'd separately
got = ov._add_tashkeel("أولاً افتح الصمام، ثم أغلقه")
check("mid-sentence comma survives CATT", got == "[أولاً افتح الصمام]، [ثم أغلقه]", repr(got))
got = ov._add_tashkeel("قال «الماء سر الحياة» بوضوح")
check("guillemets survive CATT", got == "[قال] «[الماء سر الحياة]» [بوضوح]", repr(got))

# symbol expansions (2026-08-27)
sym_cases = [
    (ov, "ارتفع الضغط ٥٠٪ اليوم", "ارتفع الضغط ٥٠ بالمئة اليوم"),
    (ov, "المبلغ 100 ﷼ فقط", "المبلغ 100 ريال فقط"),
    (ov, "المعادلة 2 + 2 = 4 صحيحة", "المعادلة 2 زائد 2 يساوي 4 صحيحة"),
    (vt, "الخصم ٥٠٪ حالياً", "الخصم ٥٠ بالمئة حالياً"),
]
for mod, src, want in sym_cases:
    got = mod._expand_abbreviations(src)
    check(f"symbol {src[:20]!r}", got == want, f"got {got!r}")

# pausal restore still works when the final punctuation SURVIVES CATT
check("pausal with surviving punctuation",
      ov._restore_pausal_form("الحج ركن.", "الْحَجُّ رُكْنٌ.") == "الْحَجُّ رُكْن.")

# number verbalization (integers only, Arabic context only, 911-rule wins first)
num_cases = [
    (ov, "الفاتورة تجيك كل 60 يوم", "الفاتورة تجيك كل ستون يوم"),
    (ov, "كل ٦٠ يوم", "كل ستون يوم"),                       # eastern digits normalized
    (ov, "المبلغ 3.5 ريال", "المبلغ 3.5 ريال"),              # decimals untouched (guard handles)
    (ov, "Please call 911 now", "Please call 911 now"),      # English untouched
    (vt, "العداد قرا 123", "العداد قرا مائة و ثلاثة و عشرون"),
    (ov, "القراءة هي 8192.", "القراءة هي ثمانية آلاف و مائة و اثنان و تسعون."),  # sentence-final int converts
    (ov, "الموعد الساعة 10:30 صباحاً", "الموعد الساعة 10:30 صباحاً"),            # times stay digits (guard handles)
]
for mod, src, want in num_cases:
    got = mod._digits_to_arabic_words(src)
    check(f"num2words {src[:22]!r}", got == want, f"got {got!r}")

got = ov._digits_to_arabic_words(ov._expand_abbreviations("اتصل على الرقم الموحد 911 فوراً."))
check("911 digit-by-digit wins over cardinal",
      "تسعة واحد واحد" in got and "تسعمائة" not in got, repr(got))
check("الرقم survives the BC-abbrev rule (latent-bug regression test)",
      got == "اتصل على الرقم الموحد تسعة واحد واحد فوراً.", repr(got))
check("real ق.م still expands",
      ov._expand_abbreviations("عام 500 ق.م تقريباً") == "عام 500 قبل الميلاد تقريباً")
check("رقم intact in Egyptian module too",
      vt._expand_abbreviations("رقم العداد واضح") == "رقم العداد واضح")

# ── 10. pausal-form (waqf) restoration ────────────────────────────────────────
print("[10] pausal form")
pausal_cases = [
    # (original sentence, CATT-style diacritized, expected)
    ("الحج ركن من أركان الإسلام.", "الْحَجُّ رُكْنٌ مِنْ أَرْكَانِ الْإِسْلَامِ",
     "الْحَجُّ رُكْنٌ مِنْ أَرْكَانِ الْإِسْلَام."),                    # final kasra dropped, period restored
    ("لموسم الحج هذا العام؟", "لِمَوْسِمِ الْحَجِّ هَذَا الْعَامَ",
     "لِمَوْسِمِ الْحَجِّ هَذَا الْعَام؟"),                            # fatha dropped, ؟ restored, mid-sentence حَجِّ untouched
    ("بشكل شهري.", "بِشَكْلٍ شَهْرِيٍّ", "بِشَكْلٍ شَهْرِيّ."),          # tanwīn dropped, shadda KEPT
    ("سافر براً.", "سَافَرَ بَرًّا", "سَافَرَ بَرّا."),                  # tanwīn-fath dropped, final alif kept
    ("كيف الحال", "كَيْفَ الْحَالُ", "كَيْفَ الْحَال"),                 # no punctuation in original — none added
]
for orig, dia, want in pausal_cases:
    got = ov._restore_pausal_form(orig, dia)
    check(f"pausal {orig[:25]!r}", got == want, f"got {got!r}")

print()
print(f"{'ALL PASS' if FAIL == 0 else 'FAILURES'}: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
