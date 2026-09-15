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

# ── 8c. owner fixes 2026-08-27: echo stripper, Egyptian drift retry, word errors ──
print("[8c] echo stripper / drift retry / word errors")
echo_cases = [
    (["وش هي مسؤوليات الشركة؟ الشركة مسؤولة عن التوزيع."], "الشركة مسؤولة عن التوزيع."),
    (["وش هي المسؤوليات", "؟ الجواب هنا."], "الجواب هنا."),          # split across tokens
    (["الشركة مسؤولة. وش بعد؟"], "الشركة مسؤولة. وش بعد؟"),          # first terminator is '.' → untouched
    (["ما هو سؤالك؟"], ""),                                          # echo-only → empty → server fallback
    (["كلام بدون أي نهاية"], "كلام بدون أي نهاية"),                   # no terminator → pass through
    (["ك" * 210 + "؟ نعم."], "ك" * 210 + "؟ نعم."),                  # cap: long first chunk isn't an echo
]
for toks, want in echo_cases:
    got = asyncio.run(_collect(llm.strip_leading_question(_gen(toks))))
    check(f"echo-strip {str(toks)[:30]!r}", got == want, f"got {got!r}")

check("دعوة قضائية phrase repair",
      asyncio.run(_collect(llm.repair_phrases(_gen(["يمكنه رفع دعو", "ة قضائية مدنية"]),
                                              llm.FANAR_PHRASE_REPAIRS)))
      == "يمكنه رفع دعوى قضائية مدنية")
check("دعوة alone untouched",
      asyncio.run(_collect(llm.repair_phrases(_gen(["وصلتني دعوة عشاء"]), llm.FANAR_PHRASE_REPAIRS)))
      == "وصلتني دعوة عشاء")
check("المياة repair",
      asyncio.run(_collect(llm.repair_words(_gen(["شركة المياة مسؤولة"]), llm.FANAR_ARABIC_REPAIRS)))
      == "شركة المياه مسؤولة")
check("بيسوونها repair",
      asyncio.run(_collect(llm.repair_words(_gen(["اللي بيسوونها الشركة"]), llm.FANAR_NAJDI_REPAIRS)))
      == "اللي يسوونها الشركة")

# reply-signal detector (drift check) — real excerpts from the 2026-08-27 session
check("MSA-drifted Egyptian reply → NO signals",
      not routing.reply_has_egyptian_signals(
          "تاريخ الحفاظ على المياه في السعودية، يا سيدي! بدأت الجهود الرسمية للحفاظ على موارد المائية من قبل المملكة منذ فترة طويلة"))
check("good Masri reply → signals (مش)",
      routing.reply_has_egyptian_signals("طيب، إذا كان الفاتورة مش صحيحة، العميل يقدر يعمل إيه؟"))
check("good Masri reply → signals (اللي shared counts for OUTPUT)",
      routing.reply_has_egyptian_signals("الإجراء بيختلف حسب نوع الهوية اللي بتقدمها"))
check("signal set never used for routing",
      routing.route_arabic("الموضوع طيب وفيه حاجة كمان") != "egyptian arabic"
      or routing.looks_egyptian("الموضوع طيب وفيه حاجة كمان"))


async def _run_retry_test(first_text, second_text):
    calls = []

    def make_stream(msgs):
        calls.append([dict(m) for m in msgs])
        async def _s():
            for chunk in ([first_text] if len(calls) == 1 else [second_text]):
                yield chunk
        return _s()

    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    out = "".join([t async for t in llm._with_egy_drift_retry(make_stream, msgs)])
    return out, calls

out, calls = asyncio.run(_run_retry_test(
    "بالتأكيد، تاريخ الحفاظ على المياه في المملكة يعود إلى عقود مضت حيث كانت الجهود الحكومية الرسمية تركز على الترشيد وتطوير البنية التحتية للمياه في جميع المناطق.",
    "طيب، الموضوع ده قديم أوي والناس بتهتم بيه."))
check("drift retry fires on MSA reply", out == "طيب، الموضوع ده قديم أوي والناس بتهتم بيه.", repr(out))
check("retry regenerates with reinforced message",
      len(calls) == 2 and llm._EGY_RETRY_NOTE in calls[1][-1]["content"])
check("original messages not mutated by retry",
      llm._EGY_RETRY_NOTE not in calls[0][-1]["content"])

out, calls = asyncio.run(_run_retry_test(
    "طيب يا فندم، الفاتورة دي بتتحسب كل شهر حسب استهلاكك للمية وبتوصلك في ميعاد ثابت من الشركة المسؤولة.",
    "SHOULD NOT BE CALLED"))
check("no retry on good Masri reply",
      len(calls) == 1 and "SHOULD NOT" not in out, repr(out))

# ── 8d. QA report 2026-08-27 fixes ────────────────────────────────────────────
print("[8d] QA 27/8 fixes")
qa_pron_cases = [
    ("المرحلة الحرجة وصلت", "المرحلة الحَرِجَة وصلت"),
    ("قدم الوثائق المطلوبة", "قدم الوَثَائِق المطلوبة"),
    # water colloquialization REVERTED 2026-09-14 → المياه stays formal everywhere
    ("توزيع المياه على البيوت", "توزيع المياه على البيوت"),
    ("شركة المياه الوطنية مسؤولة", "شركة المياه الوطنية مسؤولة"),
    ("تحلية مياه البحر مهمة", "تحلية مياه البحر مهمة"),
    ("هنبعت الفني حالاً للموقع", "هنبعت الفني حَالَن للموقع"),        # tanween trial
    ("تعال حالًا", "تعال حَالَن"),                                    # mark-before-alif variant
]
for src, want in qa_pron_cases:
    got = vt._apply_pronunciation_fixes(src)
    check(f"qa-pron {src[:24]!r}", got == want, f"got {got!r}")

check("الميه→المويه in Najdi guard",
      asyncio.run(_collect(llm.repair_words(_gen(["تزود الناس بالخدمة والميه النظيفة"]), llm.FANAR_NAJDI_REPAIRS)))
      == "تزود الناس بالخدمة والمويه النظيفة")
check("بالميه→بالمويه (water context)",
      asyncio.run(_collect(llm.repair_regexes(_gen(["يزودهم بالميه ", "النظيفة كل يوم"]), llm.FANAR_NAJDI_REGEX_REPAIRS)))
      == "يزودهم بالمويه النظيفة كل يوم")
check("digit + بالميه stays (percent idiom)",
      asyncio.run(_collect(llm.repair_regexes(_gen(["زاد الضغط 50 بالميه تقريباً"]), llm.FANAR_NAJDI_REGEX_REPAIRS)))
      == "زاد الضغط 50 بالميه تقريباً")
check("علي صيانه phrase repair",
      asyncio.run(_collect(llm.repair_phrases(_gen(["وبتشتغل علي صيان", "ه الشبكات"]), llm.FANAR_PHRASE_REPAIRS)))
      == "وبتشتغل على صيانه الشبكات")
check("the name علي stays safe",
      asyncio.run(_collect(llm.repair_phrases(_gen(["قال علي إن الوضع زين"]), llm.FANAR_PHRASE_REPAIRS)))
      == "قال علي إن الوضع زين")
check("card uses المياه not الميه (revert 2026-09-14)",
      "المياه" in routing.EGYPTIAN_CARD and "الميه (الماء)" not in routing.EGYPTIAN_CARD)
check("Egyptian backstop الميه→المياه", vt._apply_pronunciation_fixes("اشرب الميه") == "اشرب المياه")
check("Egyptian backstop ميه→مياه", vt._apply_pronunciation_fixes("في ميه كتير") == "في مياه كتير")

# ── 8e. owner list 2026-08-28 ─────────────────────────────────────────────────
print("[8e] owner list 28/8")
from pipeline import tts_omnivoice_v1 as ov   # noqa: E402  (reused by later sections)
omni_pron_cases = [
    ("راجع التفاصيل المدخلة في النظام", "راجع التَّفَاصِيل الْمُدْخَلَة في النظام"),
    ("لو حصل تسرب في الخزان", "لو حصل تَسَرُّب في الْخَزَّان"),
    ("المشكلة خطرة خاصةً في الصيف", "المشكلة خَطِرَة خَاصَّةً في الصيف"),
    ("الحجاج يحتاجون خزان إضافي", "الْحُجَّاج يحتاجون خَزَّان إضافي"),
    ("شيك على عدادك الخاص", "شيك على عَدَّادَك الخاص"),
    ("تقدر تلاحظ الفرق في المتحف", "تقدر تُلَاحِظ الفرق في الْمَتْحَف"),
    ("تطبيق تتبع استهلاك المياه", "تطبيق تَتَبُّع استهلاك المياه"),   # collocation fires
    ("لازم تتبع التعليمات بدقة", "لازم تتبع التعليمات بدقة"),         # verb reading untouched
]
for src, want in omni_pron_cases:
    got = ov._apply_pronunciation_fixes(src)
    check(f"omni-pron {src[:22]!r}", got == want, f"got {got!r}")

vt_2808_cases = [
    ("المنطقة دي فيها الملوثات كتير", "الْمِنْطَقَة دي فيها الْمُلَوِّثَات كتير"),
    ("تطبيق تتبع استهلاك المياه", "تطبيق تَتَبُّع استهلاك المياه"),   # تتبع collocation; المياه stays formal (revert)
]
for src, want in vt_2808_cases:
    got = vt._apply_pronunciation_fixes(src)
    check(f"vt-pron {src[:22]!r}", got == want, f"got {got!r}")

check("حئوق→حقوق (display+audio)",
      asyncio.run(_collect(vt._egyptianize_tokens(_gen(["حئوق العميل محفوظة"])))) == "حقوق العميل محفوظة")
check("بيأخدوا→بيخدوا",
      asyncio.run(_collect(vt._egyptianize_tokens(_gen(["هم بيأخدوا القراءة"])))) == "هم بيخدوا القراءة")
check("الموسم الحج word order",
      asyncio.run(_collect(llm.repair_phrases(_gen(["في الموسم الحج يزيد الطلب"]), llm.FANAR_PHRASE_REPAIRS)))
      == "في موسم الحج يزيد الطلب")
check("أوبر→Uber",
      asyncio.run(_collect(llm.repair_words(_gen(["اطلب أوبر من التطبيق"]), llm.FANAR_ARABIC_REPAIRS)))
      == "اطلب Uber من التطبيق")
check("أوبر وكريم pair",
      asyncio.run(_collect(llm.repair_phrases(_gen(["استخدم أوبر أو كريم للوصول"]), llm.FANAR_PHRASE_REPAIRS)))
      == "استخدم Uber أو Careem للوصول")
check("كريم alone stays (name/word)",
      asyncio.run(_collect(llm.repair_phrases(_gen(["الرجل كريم جداً"]), llm.FANAR_PHRASE_REPAIRS)))
      == "الرجل كريم جداً")
check("صج→فعلاً (Najdi output only)",
      asyncio.run(_collect(llm.repair_words(_gen(["صج الوضع تحسن"]), llm.FANAR_NAJDI_REPAIRS)))
      == "فعلاً الوضع تحسن")
check("صج still a routing marker (input side untouched)", routing.looks_najdi("صج الجو حار"))
check("بتدور→تدور",
      asyncio.run(_collect(llm.repair_words(_gen(["ليش بتدور على الملف"]), llm.FANAR_NAJDI_REPAIRS)))
      == "ليش تدور على الملف")
check("Uber/Careem in the Latin-script note", "Uber, Careem" in llm.FANAR_ARABIC_NOTE)

# ── 8f. hidden-issue fixes 2026-08-31 ─────────────────────────────────────────
print("[8f] hidden-issue fixes 31/8")
meters_cases = [
    ("تم التأسيس عام 2026 م رسمياً", "تم التأسيس عام 2026 ميلادي رسمياً"),
    ("عمق الخزان 3 م تقريباً", "عمق الخزان 3 متر تقريباً"),
    ("طول الأنبوب 500 م من المحطة", "طول الأنبوب 500 متر من المحطة"),
    ("عام 1447 هـ الماضي", "عام 1447 هجري الماضي"),      # هـ rule untouched
]
for src, want in meters_cases:
    got = ov._expand_abbreviations(src)
    check(f"meters/year {src[:22]!r}", got == want, f"got {got!r}")

check("diacritized repair target caught (دلوقتِ)",
      asyncio.run(_collect(llm.repair_words(_gen(["دلوقتِ نبدأ الشغل"]), llm.FANAR_NAJDI_REPAIRS)))
      == "الحين نبدأ الشغل")
check("diacritized repair target caught (مِش, Egyptian... via najdi map عايز)",
      asyncio.run(_collect(llm.repair_words(_gen(["هو عَايِز يعرف"]), llm.FANAR_NAJDI_REPAIRS)))
      == "هو أبغى يعرف")
check("unmapped diacritized words keep their marks",
      asyncio.run(_collect(llm.repair_words(_gen(["الْحَجُّ رُكْنٌ مهم"]), llm.FANAR_NAJDI_REPAIRS)))
      == "الْحَجُّ رُكْنٌ مهم")
check("egyptianize tolerant of marks",
      asyncio.run(_collect(vt._egyptianize_tokens(_gen(["الَّذِي قال كده"])))) == "اللي قال كده")

# drift-probe echo blind spot: echo (with Egyptian words) + MSA body → retry fires
out, calls = asyncio.run(_run_retry_test(
    "عايز تعرف ايه اللي بيحصل في الشركة؟ "
    "تقوم الشركة الوطنية بتنفيذ استراتيجيات شاملة لإدارة الموارد المائية في جميع المناطق التابعة لها حسب الخطة المعتمدة رسمياً.",
    "طيب، الشركة بتعمل حاجات كتير عشان الميه توصل لكل الناس."))
check("drift retry fires despite Egyptian echo opener",
      len(calls) == 2 and out.startswith("طيب"), repr(out[:40]))

# ── 8g. dialect number tables (2026-08-31) ────────────────────────────────────
print("[8g] dialect number tables")
from pipeline import arabic_numbers as an   # noqa: E402

msa_nums = [(3, "ثلاثة"), (11, "أحد عشر"), (15, "خمسة عشر"), (23, "ثلاثة وعشرون"),
            (100, "مئة"), (200, "مائتان"), (345, "ثلاثمئة وخمسة وأربعون"),
            (1447, "ألف وأربعمئة وسبعة وأربعون"), (2000, "ألفان"), (10000, "عشرة آلاف")]
for n, want in msa_nums:
    got = an.int_to_words(n, "msa")
    check(f"msa {n}", got == want, f"got {got!r}")

najdi_nums = [(3, "ثَلَاث"), (8, "ثِمَان"), (15, "خَمْسْطَعَش"), (23, "ثَلَاث وْعِشْرِين"),
              (60, "سِتِّين"), (300, "ثَلَاث مِيَّة"), (2000, "أَلْفَين")]
for n, want in najdi_nums:
    got = an.int_to_words(n, "najdi")
    check(f"najdi {n}", got == want, f"got {got!r}")

egy_nums = [(8, "تَمَانْيَة"), (11, "حِدَاشَر"), (13, "تَلَتَّاشَر"), (30, "تَلَاتِين"),
            (300, "تَلْتُمِيَّة"), (3000, "تَلَات آلَاف"), (23, "تَلَاتَة وِعِشْرِين")]
for n, want in egy_nums:
    got = an.int_to_words(n, "egyptian")
    check(f"egy {n}", got == want, f"got {got!r}")

check("decimal egy 3.5", an.decimal_to_words("3", "5", "egyptian") == "تَلَاتَة فَاصْلَة خَمْسَة")
check("time egy 10:30", an.time_to_words("10", "30", "egyptian") == "عَشَرَة وِتَلَاتِين")
# millions/billions (2026-09-14)
check("msa 1,000,000", an.int_to_words(1_000_000, "msa") == "مليون")
check("msa 2,000,000", an.int_to_words(2_000_000, "msa") == "مليونان")
check("msa 3,000,000", an.int_to_words(3_000_000, "msa") == "ثلاثة ملايين")
check("msa 123,456,789",
      an.int_to_words(123_456_789, "msa")
      == "مئة وثلاثة وعشرون مليون وأربعمئة وستة وخمسون ألف وسبعمئة وتسعة وثمانون")
check("egy 3,000,000", an.int_to_words(3_000_000, "egyptian") == "تَلَات مَلَايِين")
check("najdi 2,000,000", an.int_to_words(2_000_000, "najdi") == "مِلْيُونَين")
check("existing thousands unchanged (8192)",
      an.int_to_words(8192, "msa") == "ثمانية آلاف ومئة واثنان وتسعون")
check(">999,999,999 stays digits", an.int_to_words(1_000_000_000, "msa") is None)
# 2026-09-09 corrected forms (Leen's updated block)
check("msa hundred drops silent alif", an.int_to_words(100, "msa") == "مئة")
check("msa 800 attached uses مئة", an.int_to_words(800, "msa") == "ثمانمئة")
check("msa 200 keeps alif (owner)", an.int_to_words(200, "msa") == "مائتان")
check("egy 300 is تَلْتُمِيَّة", an.int_to_words(300, "egyptian") == "تَلْتُمِيَّة")
check("egy 400 unchanged", an.int_to_words(400, "egyptian") == "أَرْبَعُمِيَّة")
check("egy 3000 unchanged", an.int_to_words(3000, "egyptian") == "تَلَات آلَاف")
check("fanar note uses مئة not مائة",
      "مئة" in llm.FANAR_NUMBERS_NOTE and "مائة" not in llm.FANAR_NUMBERS_NOTE)

# Issue 1: digit + spelled-out parenthetical double-reading (2026-09-09)
print("[8h] redundant number parenthetical")
dbl_cases = [
    # the exact live case: paren dropped, digit stays → verbalized once
    (ov, "يوجد 168 (مائة وثمانية وستين) ساعة في الأسبوع",
         "يوجد مئة وثمانية وستون ساعة في الأسبوع"),
    # eastern digits + attached و connector
    (ov, "المبلغ ٣٤٥ (ثلاثمئة وخمسة وأربعون) ريال",
         "المبلغ ثلاثمئة وخمسة وأربعون ريال"),
    # Egyptian route
    (vt, "فيه 11 (حداشر) لاعب", "فيه حِدَاشَر لاعب"),
    # SAFETY: a non-number parenthetical must SURVIVE
    (ov, "اتصل بشركة المياه الوطنية (NWC) فوراً", "اتصل بشركة المياه الوطنية (NWC) فوراً"),
    (ov, "المدة 60 (يوماً) كاملة", "المدة ستون (يوماً) كاملة"),   # "يوماً" isn't a number-word
    # SAFETY: a paren NOT right after digits is untouched
    (ov, "الرقم مذكور (مائة وستون) هنا", "الرقم مذكور (مائة وستون) هنا"),
]
for mod, src, want in dbl_cases:
    got = mod._digits_to_arabic_words(src, "msa" if mod is ov else "egyptian")
    check(f"paren {src[:26]!r}", got == want, f"got {got!r}")

# reverse double-read (2026-09-10): number-WORD then (pure digits) → drop the paren
rev_cases = [
    (ov, "غليان الماء مية (١٠٠) درجة", "غليان الماء مية درجة"),   # Najdi-ish word + eastern digits
    (ov, "العدد مئة وثمانية وستون (168) بالضبط", "العدد مئة وثمانية وستون بالضبط"),
    (vt, "تقريباً حداشر (11) لاعب", "تقريباً حداشر لاعب"),
    # SAFETY: pure-digits paren after a NON-number word must SURVIVE
    (ov, "الصفحة (100) مفيدة", "الصفحة (مئة) مفيدة"),   # "الصفحة" not a number → paren KEPT, 100 verbalized inside
    (ov, "اتصل (NWC) الآن", "اتصل (NWC) الآن"),        # not digits → untouched
]
for mod, src, want in rev_cases:
    got = mod._digits_to_arabic_words(src, "msa" if mod is ov else "egyptian")
    check(f"revparen {src[:24]!r}", got == want, f"got {got!r}")

# ── 8i. number/acronym edge fixes (2026-09-09) ────────────────────────────────
print("[8i] number/acronym edge fixes")
def _stage(s, d="msa", mod=ov):
    return mod._digits_to_arabic_words(mod._expand_abbreviations(s), d)
edge = [
    ("negative", "الحرارة -5 مئوية", "الحرارة ناقص خمسة مئوية"),
    ("range",    "المدة 60-70 يوم", "المدة ستون إلى سبعون يوم"),
    ("subtract", "الناتج 10 - 3", "الناتج عشرة ناقص ثلاثة"),
    ("phone",    "اتصل على 0501234567 حالاً", "اتصل على صفر خمسة صفر واحد اثنان ثلاثة أربعة خمسة ستة سبعة حالاً"),
    ("time_sec", "الوقت 10:30:45 مساءً", "الوقت عشرة وثلاثين وخمسة وأربعين مساءً"),
    ("time_plain", "الساعة 10:30", "الساعة عشرة وثلاثين"),
    ("dash-bullet safe", "الماء - مورد مهم", "الماء - مورد مهم"),   # dash not before a digit
]
for label, s, want in edge:
    got = _stage(s)
    check(f"edge {label}", got == want, f"got {got!r}")

# sentence-initial NWC now expands; English stays; mid-sentence still works
check("NWC sentence-initial expands",
      ov._apply_pronunciation_fixes("NWC مسؤولة عن المياه") == "إن دبليو سي مسؤولة عن المياه")
check("NWC mid-sentence still expands",
      ov._apply_pronunciation_fixes("تفحص شركة NWC العينات") == "تفحص شركة إن دبليو سي العينات")
check("NWC in English untouched",
      ov._apply_pronunciation_fixes("please contact NWC support") == "please contact NWC support")

# diacritic-tolerant repair_phrases / repair_regexes (the slip-class fix)
check("phrase repair tolerant of harakat",
      asyncio.run(_collect(llm.repair_phrases(_gen(["رفع دَعوة قضائية مدنية"]), llm.FANAR_PHRASE_REPAIRS)))
      == "رفع دعوى قضائية مدنية")
check("regex repair tolerant of harakat",
      asyncio.run(_collect(llm.repair_regexes(_gen(["يزودهم بِالميه النظيفة"]), llm.FANAR_NAJDI_REGEX_REPAIRS)))
      == "يزودهم بالمويه النظيفة")
check("regex repair still digit-guarded (percent idiom)",
      asyncio.run(_collect(llm.repair_regexes(_gen(["زاد 50 بالميه"]), llm.FANAR_NAJDI_REGEX_REPAIRS)))
      == "زاد 50 بالميه")

# ── 8j. equations, dates, celsius (2026-09-09) ────────────────────────────────
print("[8j] equations / dates / celsius")
eq_date = [
    ("equation ×/=", "الناتج 4 × 7 = 28", "الناتج أربعة في سبعة يساوي ثمانية وعشرون"),
    ("divide",       "8 ÷ 2 صحيح", "ثمانية على اثنان صحيح"),
    ("date DMY",     "استقل في 14/8/1947 ميلادي",
                     "استقل في أربعة عشر أغسطس ألف وتسعمئة وسبعة وأربعون ميلادي"),
    ("date eastern", "بتاريخ ١/١/٢٠٢٤",
                     "بتاريخ واحد يناير ألفان وأربعة وعشرون"),
    ("invalid date kept as numbers", "الكود 45/99/2000",
                     "الكود خمسة وأربعون/تسعة وتسعون/ألفان"),   # 99>12 → not a date
]
for label, s, want in eq_date:
    got = ov._digits_to_arabic_words(ov._expand_abbreviations(s), "msa")
    check(f"eqdate {label}", got == want, f"got {got!r}")
# dates work on the Egyptian route too
check("date egyptian route",
      vt._digits_to_arabic_words(vt._expand_abbreviations("في 3/6/2010"), "egyptian")
      == "في تَلَاتَة يونيو أَلْفِين وِعَشَرَة")
check("سيليزية→مئوية repair",
      asyncio.run(_collect(llm.repair_words(_gen(["مئة درجة سيليزية بالضبط"]), llm.FANAR_ARABIC_REPAIRS)))
      == "مئة درجة مئوية بالضبط")

# ── 8k. QA report 2026-09-14 ──────────────────────────────────────────────────
print("[8k] QA 14/9 fixes")
# Issue 2: Egyptian water colloquialization REVERTED → المياه stays formal
check("Egyptian water stays المياه (revert)",
      vt._apply_pronunciation_fixes("فيه تسريب في المياه") == "فيه تسريب في المياه")
check("Egyptian مياه not colloquialized",
      vt._apply_pronunciation_fixes("اشرب مياه نظيفة") == "اشرب مياه نظيفة")
# Issue 3: بيعتبر diacritized (Egyptian)
check("بيعتبر→بِيِعْتَبِر", vt._apply_pronunciation_fixes("ده بيعتبر مهم") == "ده بِيِعْتَبِر مهم")
# Issue 1: جدة/العمرة — Egyptian + Najdi (no-CATT) deterministic entries
check("egy جدة→جِدَّة", vt._apply_pronunciation_fixes("زرت جدة") == "زرت جِدَّة")
check("egy العمرة→الْعُمْرَة", vt._apply_pronunciation_fixes("أديت العمرة") == "أديت الْعُمْرَة")
check("najdi/omni جدة→جِدَّة", ov._apply_pronunciation_fixes("زرت جدة") == "زرت جِدَّة")
check("najdi/omni العمرة→الْعُمْرَة", ov._apply_pronunciation_fixes("أديت العمرة") == "أديت الْعُمْرَة")
# Issue 1 Fusha: post-CATT ج-vowel correction (CATT's جَدّة → جِدّة, city reading)
def _apply_post_catt(s):
    for p, r in ov._POST_CATT_FIXES:
        s = p.sub(r, s)
    return s
check("post-CATT جَدَّةِ→جِدَّةِ",
      _apply_post_catt("مدينة جَدَّةِ الجميلة") == "مدينة جِدَّةِ الجميلة")
check("post-CATT no-op when already correct", _apply_post_catt("مدينة جِدَّةِ") == "مدينة جِدَّةِ")
# Issue 4: fanar-only Najdi non-answer note
check("FANAR_NAJDI_NOTE absent on qwen default", os.environ.get("LLM_MODEL") is not None
      or llm.FANAR_NAJDI_NOTE not in llm.build_turn("وش الأخبار؟", "ar")[0])
check("sentence-final time converts",
      ov._digits_to_arabic_words("الموعد الساعة 10:30.") == "الموعد الساعة عشرة وثلاثين.")
check("sentence-final decimal converts",
      ov._digits_to_arabic_words("المبلغ 3.5.") == "المبلغ ثلاثة فاصلة خمسة.")
check("route dispatch: najdi numbers on najdi turns",
      ov._digits_to_arabic_words("عندي 15 عداد", "najdi") == "عندي خَمْسْطَعَش عداد")
check("route dispatch: msa default",
      ov._digits_to_arabic_words("عندي 15 عداد") == "عندي خمسة عشر عداد")
check("route dispatch: voicetut egyptian default",
      vt._digits_to_arabic_words("عندي 15 عداد") == "عندي خَمَسْتَاشَر عداد")
check("leakage impossible by construction",
      an.int_to_words(13, "najdi") != an.int_to_words(13, "egyptian"))
check("fanar numbers note absent on qwen default",
      llm.FANAR_NUMBERS_NOTE not in llm.build_turn("وش الأخبار؟", "ar")[0])

# NWC letter-name expansion (2026-08-31): Arabic context only, both modules
nwc_cases = [
    (ov, "في شركة المياه الوطنية (NWC) من خلال خطوات", "في شركة المياه الوطنية (إن دبليو سي) من خلال خطوات"),
    (ov, "تتبع شركة المياه الوطنية NWC إجراءات صارمة", "تتبع شركة المياه الوطنية إن دبليو سي إجراءات صارمة"),
    (ov, "Please contact NWC support now", "Please contact NWC support now"),   # English untouched
    (vt, "شركة المياه الوطنية NWC بتفحص العينات", "شركة المياه الوطنية إن دبليو سي بتفحص العينات"),
]
for mod, src, want in nwc_cases:
    got = mod._apply_pronunciation_fixes(src)
    check(f"NWC {src[:24]!r}", got == want, f"got {got!r}")

check("VoiceTut ref clip exists", os.path.exists(vt._REF_AUDIO), vt._REF_AUDIO)
check("VoiceTut never does CATT", not hasattr(vt, "_add_tashkeel"))

# two-voice split (2026-09-10): Fusha → dedicated clip, else → Saudi/default
check("Fusha ref clip exists", os.path.exists(ov._FUSHA_REF_AUDIO), ov._FUSHA_REF_AUDIO)
ov._clone_prompt = "SAUDI"          # seed the module globals (no model load needed)
ov._fusha_clone_prompt = "FUSHA"
check("Fusha route → Fusha voice", ov._clone_for("standard arabic") == "FUSHA")
check("Najdi route → Saudi voice", ov._clone_for("najdi arabic") == "SAUDI")
check("English/mixed → Saudi voice", ov._clone_for(None) == "SAUDI")
ov._clone_prompt = ov._fusha_clone_prompt = None   # reset so a real load rebuilds them

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
# ov (tts_omnivoice_v1) already imported above in section 8e


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
    (ov, "المبلغ 3.5 ريال", "المبلغ ثلاثة فاصلة خمسة ريال"),  # decimals now verbalized (2026-08-31)
    (ov, "Please call 911 now", "Please call 911 now"),      # English untouched
    (vt, "العداد قرا 123", "العداد قرا مِيَّة وِتَلَاتَة وِعِشْرِين"),          # Egyptian table
    (ov, "القراءة هي 8192.", "القراءة هي ثمانية آلاف ومئة واثنان وتسعون."),   # tight و, table form (مئة)
    (ov, "الموعد الساعة 10:30 صباحاً", "الموعد الساعة عشرة وثلاثين صباحاً"),   # ‹H وM› (owner 2026-08-31)
]
for mod, src, want in num_cases:
    got = mod._digits_to_arabic_words(src)
    check(f"num2words {src[:22]!r}", got == want, f"got {got!r}")

got = ov._digits_to_arabic_words(ov._expand_abbreviations("اتصل على الرقم الموحد 911 فوراً."))
check("911 digit-by-digit wins over cardinal",
      "تسعة واحد واحد" in got and "تسعمئة" not in got and "تسعمائة" not in got, repr(got))
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

# ── 11. GPU pre-flight threshold logic (no GPU needed) ────────────────────────
print("[11] GPU pre-flight")
os.environ["GPU_MIN_FREE_GB"] = "5"
check("GPU_MIN_FREE_GB overrides required-free", server._required_free_gb() == 5.0)
del os.environ["GPU_MIN_FREE_GB"]
_req = server._required_free_gb()   # calls _llm_already_resident (best-effort; may hit Ollama)
check("default required-free is a sane positive number",
      isinstance(_req, float) and _req >= 9.0, str(_req))

print()
print(f"{'ALL PASS' if FAIL == 0 else 'FAILURES'}: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
