"""
test_tts_args.py — G4: the TTS argument freeze (fake model, no GPU, ~seconds + torch import)
=============================================================================================
Pins exactly what reaches OmniVoice.generate()/VoiceTut-TTS and when CATT fires, per routed
language:

  language ∈ {None, "standard arabic", "najdi arabic"}  → TODAY'S EXACT CALL, UNCHANGED:
      generate(text=..., voice_clone_prompt=<saudi>)  — NO language kwarg at all,
      VoiceTut-TTS never even called.
  language == "egyptian arabic", VoiceTut-TTS available (default, VOICETUT_TTS_ENABLED=1)
      → VoiceTut synthesizes it; OmniVoice.generate() is NEVER called (2026-08-13: the
      default Egyptian engine, promoted after live-testing beat Habibi-TTS and
      Lahgtna-OmniVoice, both fully removed — see eval/BASELINES.md).
  language == "egyptian arabic", VoiceTut unavailable/disabled, clip present
      → falls back to OmniVoice: generate(text=..., voice_clone_prompt=<egyptian>,
      language="egyptian arabic")
  language == "egyptian arabic", VoiceTut unavailable/disabled AND clip also unavailable
      → Saudi
  CATT: fires ONLY on "standard arabic" + non-Najdi sentence; NEVER for Egyptian (true
  regardless of whether VoiceTut or OmniVoice ends up handling the sentence).

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
# not the CATT model itself. _synthesize_egyptian_voicetut is stubbed the same way — it's a
# third-party model this project doesn't own, exactly like CATT above; this test pins the
# BRANCHING logic in _synthesize_mp3_blocking, not VoiceTut-TTS's own internals.
tts._model = FakeModel()
tts._clone_prompt = "SAUDI_PROMPT"
tts._egy_clone_prompt = "EGY_PROMPT"
tts.CATT_ENABLED = True
tts._add_tashkeel = lambda text: "TASHKEEL::" + text

_voicetut_result = {"value": None}   # None = "VoiceTut unavailable/disabled/failed for this sentence"
_voicetut_calls: list[str] = []


def _fake_voicetut(text: str):
    _voicetut_calls.append(text)
    return _voicetut_result["value"]


tts._synthesize_egyptian_voicetut = _fake_voicetut


def synth(text: str, language, expect_voicetut_call=None):
    calls.clear()
    _voicetut_calls.clear()
    tts._synthesize_mp3_blocking(text, language)
    if expect_voicetut_call is not None:
        called = len(_voicetut_calls) == 1
        check(f"VoiceTut-TTS called for {text[:12]!r}", called, expect_voicetut_call)
    if language == "egyptian arabic" and _voicetut_result["value"] is not None:
        assert len(calls) == 0, f"VoiceTut succeeded but OmniVoice.generate() was ALSO called: {calls}"
        return None
    assert len(calls) == 1, f"expected exactly 1 generate() call, got {len(calls)}"
    return calls[0]


AR_MSA   = "هذا اختبار بسيط"                 # no Najdi markers
AR_NAJDI = "وش الوضع الحين"                  # Najdi markers → CATT must skip
AR_EGY   = "عايز أعرف إزاي أروح المطار"

print("\n== generate() kwargs per language (VoiceTut unavailable in this stub run) ==")
kw = synth(AR_MSA, None, expect_voicetut_call=False)
check("None → saudi prompt",            kw.get("voice_clone_prompt"), "SAUDI_PROMPT")
check("None → NO language kwarg",       "language" in kw, False)

kw = synth(AR_NAJDI, "najdi arabic", expect_voicetut_call=False)
check("najdi → saudi prompt",           kw.get("voice_clone_prompt"), "SAUDI_PROMPT")
check("najdi → NO language kwarg",      "language" in kw, False)
check("najdi → no CATT",                kw["text"].startswith("TASHKEEL::"), False)

kw = synth(AR_MSA, "standard arabic", expect_voicetut_call=False)
check("fusha → saudi prompt",           kw.get("voice_clone_prompt"), "SAUDI_PROMPT")
check("fusha → NO language kwarg",      "language" in kw, False)
check("fusha non-Najdi → CATT fires",   kw["text"].startswith("TASHKEEL::"), True)

kw = synth(AR_NAJDI, "standard arabic", expect_voicetut_call=False)
check("fusha-routed Najdi reply → CATT skipped", kw["text"].startswith("TASHKEEL::"), False)

kw = synth(AR_EGY, "egyptian arabic", expect_voicetut_call=True)
check("egyptian, VoiceTut unavailable → egyptian prompt", kw.get("voice_clone_prompt"), "EGY_PROMPT")
check("egyptian, VoiceTut unavailable → language kwarg",  kw.get("language"), "egyptian arabic")
check("egyptian → NEVER CATT",          kw["text"].startswith("TASHKEEL::"), False)

print("\n== VoiceTut-TTS succeeds → OmniVoice.generate() never called ==")
_voicetut_result["value"] = np.zeros(240, dtype=np.float32)
result = synth(AR_EGY, "egyptian arabic", expect_voicetut_call=True)
check("VoiceTut success → OmniVoice.generate() never called", result, None)
_voicetut_result["value"] = None

print("\n== fallback: Egyptian clip missing (VoiceTut also unavailable) ==")
tts._egy_clone_prompt = None
kw = synth(AR_EGY, "egyptian arabic", expect_voicetut_call=True)
check("missing clip → saudi prompt",    kw.get("voice_clone_prompt"), "SAUDI_PROMPT")
check("missing clip → NO language kwarg", "language" in kw, False)
tts._egy_clone_prompt = "EGY_PROMPT"

print("\n== module surface ==")
check("egyptian NOT in CATT languages", "egyptian arabic" in tts._TASHKEEL_LANGUAGES, False)
check("gen lock exists",                hasattr(tts, "_gen_lock"), True)
check("egyptian clip path on disk",     os.path.exists(tts._EGY_REF_AUDIO), True)
check("voicetut lock exists",           hasattr(tts, "_voicetut_lock"), True)
check("voicetut ENABLED by default",    tts.VOICETUT_ENABLED, True)
check("habibi symbols fully removed",   hasattr(tts, "_habibi_lock"), False)
check("lahgtna symbols fully removed",  hasattr(tts, "_lahgtna_lock"), False)
check("egy tashkeel symbols fully removed", hasattr(tts, "_egy_tashkeel_lock"), False)

print(f"\n{'=' * 60}")
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(" ", f)
    sys.exit(1)
print("ALL TTS-ARG TESTS PASSED")
