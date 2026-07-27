"""
test_tts_args.py — G4: the TTS argument freeze (fake model, no GPU, ~seconds + torch import)
=============================================================================================
Pins exactly what reaches OmniVoice.generate() and when CATT fires, per routed language:

  language ∈ {None, "standard arabic", "najdi arabic"}  → TODAY'S EXACT CALL:
      generate(text=..., voice_clone_prompt=<saudi>)  — NO language kwarg at all
  language == "egyptian arabic" (clip present)         → generate(text=...,
      voice_clone_prompt=<egyptian>, language="egyptian arabic")
  language == "egyptian arabic" (clip missing)         → falls back to the Saudi call
  CATT: fires ONLY on "standard arabic" + non-Najdi sentence; NEVER for Egyptian.

Run:
    .venv/bin/python eval/test_tts_args.py     # exit 0 = all pass
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tts_omnivoice_v1 as tts  # noqa: E402

FAILURES: list[str] = []


def check(name: str, actual, expected) -> None:
    if actual != expected:
        FAILURES.append(f"FAIL {name}: expected {expected!r}, got {actual!r}")
        print(FAILURES[-1])
    else:
        print(f"  ok {name}")


calls: list[dict] = []


class FakeModel:
    def generate(self, **kwargs):
        calls.append(kwargs)
        return [np.zeros(240, dtype=np.float32)]


# Install the fake model + prompts directly (bypasses _get_model's real load: it returns
# _model when already set). CATT is stubbed to a visible marker so we assert the GATE,
# not the CATT model itself.
tts._model = FakeModel()
tts._clone_prompt = "SAUDI_PROMPT"
tts._egy_clone_prompt = "EGY_PROMPT"
tts.CATT_ENABLED = True
tts._add_tashkeel = lambda text: "TASHKEEL::" + text


def synth(text: str, language):
    calls.clear()
    tts._synthesize_mp3_blocking(text, language)
    assert len(calls) == 1, f"expected exactly 1 generate() call, got {len(calls)}"
    return calls[0]


AR_MSA   = "هذا اختبار بسيط"                 # no Najdi markers
AR_NAJDI = "وش الوضع الحين"                  # Najdi markers → CATT must skip
AR_EGY   = "عايز أعرف إزاي أروح المطار"

print("\n== generate() kwargs per language ==")
kw = synth(AR_MSA, None)
check("None → saudi prompt",            kw.get("voice_clone_prompt"), "SAUDI_PROMPT")
check("None → NO language kwarg",       "language" in kw, False)

kw = synth(AR_NAJDI, "najdi arabic")
check("najdi → saudi prompt",           kw.get("voice_clone_prompt"), "SAUDI_PROMPT")
check("najdi → NO language kwarg",      "language" in kw, False)
check("najdi → no CATT",                kw["text"].startswith("TASHKEEL::"), False)

kw = synth(AR_MSA, "standard arabic")
check("fusha → saudi prompt",           kw.get("voice_clone_prompt"), "SAUDI_PROMPT")
check("fusha → NO language kwarg",      "language" in kw, False)
check("fusha non-Najdi → CATT fires",   kw["text"].startswith("TASHKEEL::"), True)

kw = synth(AR_NAJDI, "standard arabic")
check("fusha-routed Najdi reply → CATT skipped", kw["text"].startswith("TASHKEEL::"), False)

kw = synth(AR_EGY, "egyptian arabic")
check("egyptian → egyptian prompt",     kw.get("voice_clone_prompt"), "EGY_PROMPT")
check("egyptian → language kwarg",      kw.get("language"), "egyptian arabic")
check("egyptian → NEVER CATT",          kw["text"].startswith("TASHKEEL::"), False)

print("\n== fallback: Egyptian clip missing ==")
tts._egy_clone_prompt = None
kw = synth(AR_EGY, "egyptian arabic")
check("missing clip → saudi prompt",    kw.get("voice_clone_prompt"), "SAUDI_PROMPT")
check("missing clip → NO language kwarg", "language" in kw, False)
tts._egy_clone_prompt = "EGY_PROMPT"

print("\n== module surface ==")
check("egyptian NOT in CATT languages", "egyptian arabic" in tts._TASHKEEL_LANGUAGES, False)
check("gen lock exists",                hasattr(tts, "_gen_lock"), True)
check("egyptian clip path on disk",     os.path.exists(tts._EGY_REF_AUDIO), True)

print(f"\n{'=' * 60}")
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(" ", f)
    sys.exit(1)
print("ALL TTS-ARG TESTS PASSED")
