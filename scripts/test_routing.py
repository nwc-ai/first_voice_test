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
vis = server._visible_history(H, "najdi arabic", False)
got = [m["content"] for m in vis]
check("najdi turn sees najdi+english only", got == ["najdi-q", "najdi-a", "en-q", "en-a"], str(got))
vis = server._visible_history(H, "egyptian arabic", False)
got = [m["content"] for m in vis]
check("egyptian turn sees egyptian+english only", got == ["egy-q", "egy-a", "en-q", "en-a"], str(got))
vis = server._visible_history(H, None, False)
check("english turn sees ALL history", len(vis) == 8)
vis = server._visible_history(H, "standard arabic", True)
check("explicit request sees ALL history", len(vis) == 8)
vis = server._visible_history(H, "najdi arabic", False)
check("tags stripped from prompt messages", all("tag" not in m for m in vis))
check("switch-back restores (najdi visible again after egyptian turns)",
      [m["content"] for m in server._visible_history(H, "najdi arabic", False)][:2]
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

check("VoiceTut ref clip exists", os.path.exists(vt._REF_AUDIO), vt._REF_AUDIO)
check("VoiceTut never does CATT", not hasattr(vt, "_add_tashkeel"))

print()
print(f"{'ALL PASS' if FAIL == 0 else 'FAILURES'}: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
