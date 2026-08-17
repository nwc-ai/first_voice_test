# ========================== WebSocket-ready TTS module (OmniVoice + VoiceTut-TTS) =========
#
# In-process TTS using k2-fsa/OmniVoice (omnilingual zero-shot voice cloning, 24 kHz) for
# Najdi/Fusha/English/mixed, and VoiceTut-TTS (an OmniVoice fine-tune, Egyptian-specialized
# checkpoint) for Egyptian-routed turns specifically — see the VOICETUT_ENABLED block below
# for why (promoted to the default 2026-08-13 after live-testing beat the two prior Egyptian
# candidates, Habibi-TTS and Lahgtna-OmniVoice — both fully removed, see eval/BASELINES.md).
#
# Public API is identical to the previous Silma module (drop-in for server.py):
#   await stream_tts_to_ws(token_gen, ws, cancel_event, on_first_audio=None)
#
# Structure, sentence-flushing, abbreviation/opener handling, MP3 encoding, the
# sentence-queue + background synth worker, the on_first_audio / tts_end protocol,
# and the 3-point cancellation are all carried over verbatim from the proven Silma
# module (tts_silma_v1.py) — only the model load + the per-sentence synthesis call
# are engine-specific.
#
# OmniVoice is a zero-shot voice-cloner: it needs a short reference clip + its
# transcript to define the voice. We reuse the Saudi reference clip (default) and a
# separate Egyptian reference clip (Egyptian-routed turns, OmniVoice fallback path only).
# =========================================================================================

import asyncio
import os
import re
import threading
from typing import Any, AsyncIterator, Optional

import numpy as np
import torch

from routing import apply_dialect_repairs, looks_najdi, TTS_LANG_TO_DIALECT

# CATT tashkeel (diacritization) — Fusha-only. CATT is an MSA-trained diacritizer; applying it
# to Najdi text mis-vocalizes dialect words (e.g. مرة "very" comes back misread as the unrelated
# MSA noun "a time/once"). Gated two ways: the turn's `language` must be Fusha AND the sentence
# being synthesized must not itself look Najdi (the LLM can reply in Najdi even on a turn routed
# as Fusha — the reply text is the ground truth, not the user's input). CATT_ENABLED=0 reverts
# to plain (undiacritized) text in one env var without a code change.
CATT_ENABLED = os.environ.get("CATT_ENABLED", "1") == "1"
_TASHKEEL_LANGUAGES = {"standard arabic"}

# ── VoiceTut-TTS (Egyptian-only second TTS engine — the DEFAULT Egyptian engine) ─────────
# mohammedaly22/VoiceTut-TTS: a fine-tune of THIS project's own k2-fsa/OmniVoice base
# (confirmed via HF's own structured tags: base_model:k2-fsa/OmniVoice +
# base_model:finetune:k2-fsa/OmniVoice, and config.json's model_type == "omnivoice") — loads
# through the ALREADY-INSTALLED omnivoice==0.1.5 PyPI package, zero new pip packages, verified
# hands-on (despite the model's own GitHub repo claiming OmniVoice must be installed from
# source — that claim did not hold up). No monkeypatches, no torchaudio shims, no dependency
# conflicts.
#
# HISTORY (see eval/BASELINES.md for the full record): this project tried three Egyptian
# engines in sequence — Habibi-TTS (SWivid/Habibi-TTS, F5-TTS architecture, the original
# default from 2026-07-30) and Lahgtna-OmniVoice (oddadmix/lahgtna-omnivoice-v2, an opt-in
# A/B candidate from the same date) — before VoiceTut-TTS. Owner live-tested all three by
# ear; VoiceTut-TTS won, and Habibi/Lahgtna were both FULLY REMOVED 2026-08-13 (code,
# packages, downloaded checkpoints) rather than left dormant — this is now the only Egyptian-
# specialized engine in the pipeline, promoted from "opt-in A/B candidate #3" to the default.
#
# WHY IT WON: best license/training-data story of the three — Apache-2.0 confirmed in BOTH
# the model card prose and HF's structured `license` API field (no discrepancy, unlike
# Habibi's Apache-vs-cc-by-nc-sa-4.0 mismatch or Lahgtna's missing tag), and ~380h of
# disclosed, dialect-tagged (arz) Egyptian YouTube podcast training audio — more Egyptian-
# specific data than either prior candidate ever disclosed. Root cause this whole chain of
# candidates exists for: OmniVoice's own base training data is only ~23h Egyptian vs ~204h
# Najdi/~1484h MSA (2026-07-30 research, see eval/BASELINES.md). One honest caveat: the
# downloaded checkpoint is ~6.9GB on disk, larger than its card-stated 0.6B-parameter
# backbone would suggest — not a blocker (plenty of disk free), just a claim-vs-reality gap
# worth noting for the record.
#
# VOICETUT_TTS_ENABLED=0 reverts Egyptian-routed turns to OmniVoice's own Egyptian clone
# prompt (the pre-2026-07-30 behavior) in one env var, no code change needed.
VOICETUT_ENABLED = os.environ.get("VOICETUT_TTS_ENABLED", "1") == "1"
_VOICETUT_MODEL_ID = "mohammedaly22/VoiceTut-TTS"
# Measured hands-on: VoiceTut-TTS's raw output runs on the quieter side (peak ~0.33-0.60
# across 6 test sentences) than OmniVoice's own levels (~0.47-0.89). Peak-normalize so
# Egyptian replies don't sound quieter than Najdi/Fusha ones.
_VOICETUT_TARGET_PEAK = 0.75

_voicetut_model = None
_voicetut_clone_prompt = None
_voicetut_lock = threading.Lock()
_voicetut_load_failed = False   # sticky — don't retry a slow failing load on every single turn


# ── Sentence boundary constants (verbatim from tts_silma_v1.py) ──────────────────────────
HARD_BREAK = {'!', '?', '؟'}
SOFT_BREAK = {'.', ',', '،', ';', ':'}
SOFT_BREAK_MIN = 40
FIRST_SOFT_MIN = 20  # the very first flush happens earlier — cuts time-to-first-audio
_HEAD_PROBE_CHARS = 30  # leading text buffered before deciding whether to strip a filler opener

# ── Arabic abbreviation / glued-digit expander (verbatim from tts_silma_v1.py) ───────────
# Spelled-out forms read better aloud; runs on each sentence before synthesis.
_ABBREV_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r'(\d)\s*هـ(?=[\s،,.:؟!]|$)', re.UNICODE), r'\1 هجري'),
    (re.compile(r'(\d)\s*م(?=[\s،,.:؟!]|$)', re.UNICODE), r'\1 ميلادي'),
    (re.compile(r'ق\.?\s*م(?=[\s،,.:؟!]|$)', re.UNICODE), 'قبل الميلاد'),
    (re.compile(r'(\d)\s*%', re.UNICODE), r'\1 بالمئة'),
    (re.compile(r'\bد\.\s+', re.UNICODE), 'دكتور '),
    (re.compile(r'\bأ\.\s+', re.UNICODE), 'أستاذ '),
    (re.compile(r'\bإلخ\b', re.UNICODE), 'وما إلى ذلك'),
    # Separate digits glued to Arabic letters (e.g. "و2013" → "و 2013").
    (re.compile(r'([؀-ۿ])(\d)', re.UNICODE), r'\1 \2'),
    (re.compile(r'(\d)([؀-ۿ])', re.UNICODE), r'\1 \2'),
]

def _expand_abbreviations(text: str) -> str:
    for pattern, replacement in _ABBREV_RULES:
        text = pattern.sub(replacement, text)
    return text


# Deterministic dialect-leak repair (جداً→أوي/مرة, generalized across dialects — see
# routing.apply_dialect_repairs) is applied per fully-materialized sentence in
# synth_worker(), right before synthesis, not per-LLM-token — a wrong word can split
# across two streamed tokens. Was formerly a hand-written Egyptian-only fixup here
# (fix_egyptian_leaks); superseded by routing.DIALECT_REPAIR_MAP, which also covers
# Najdi's identical مرة/جداً leak. See routing.py and the project plan
# najdi-q2-wrong-elegant-papert.md for the admission bar and the KNOWN, ACCEPTED
# TRADEOFF this creates against _emit()'s "transcript and audio stay identical"
# invariant (the transcript may still show جداً for this one word while the audio
# correctly says the dialect-appropriate replacement).

SAMPLE_RATE = 24000  # OmniVoice output sample rate

# Reference clip + its exact transcript define the cloned voice (Saudi male — the DEFAULT).
_REF_AUDIO = os.path.join(os.path.dirname(__file__), "voices", "silma-tts-saudi-24k.wav")
_REF_TEXT  = "الثقافة السعودية فيها عراقة وتاريخ عميق، وقيم إسلامية راسخة، وعادات وتقاليد قبلية أصيلة متوارثة."

# Egyptian reference clip — v4, owner-provided 2026-07-20 (supersedes v3, which was restored
# from the omnivoice-tts branch; v1/v2 and a synthetic NAMAA clip were rejected by ear —
# this voice has a history of needing iterations, judge it by ear before trusting it).
# The transcript must be the EXACT words spoken in the clip, verbatim as supplied.
# Used ONLY when a turn is routed "egyptian arabic"; every other turn takes the Saudi path
# with the exact legacy generate() call.
_EGY_REF_AUDIO = os.path.join(os.path.dirname(__file__), "voices", "omnivoice-tts-egyptian-24k-v4.wav")
_EGY_REF_TEXT  = "إِزَّيَّكْ النهارده؟ الجو جامد أوي والحمدلله. قالوا إن الخزان الجديد جاهز، بس هو اتشاف كام مرة؟ رأيك ايه، نروح نشوفه بقى؟"

_MODEL_ID = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
_DEVICE   = os.environ.get("OMNIVOICE_DEVICE", "cuda:0")

# ── Lazy model singleton ──────────────────────────────────────────────────────────────────
_model = None
_model_lock = threading.Lock()
_clone_prompt = None   # reusable VoiceClonePrompt — built once with the model; passing the
                       # raw ref WAV per sentence made OmniVoice re-load/re-tokenize the
                       # reference clip on EVERY sentence (needless first-audio latency).
_egy_clone_prompt = None   # Egyptian VoiceClonePrompt — None when the clip is missing, in
                           # which case Egyptian-routed turns silently fall back to the
                           # Saudi voice (the server must never be hostage to this asset).

# Serializes OmniVoice generate() calls. Ported from the omnivoice-tts branch: generate()
# is not documented thread-safe, and a barge-in cancels the synth worker task while its
# to_thread call is still running on the GPU — without this lock the NEXT turn's first
# sentence can run generate() concurrently with that orphaned call on the same model
# object. Latent with one voice, consequential with two clone prompts in play.
_gen_lock = threading.Lock()

# ── Lazy tashkeel model singleton (same shape as _model/_model_lock above) ────────────────
_tashkeel_model = None
_tashkeel_lock = threading.Lock()


def load_models():
    """Optional warm-up hook — call from FastAPI lifespan so the first user
    does not pay the model load cost."""
    if not os.path.exists(_REF_AUDIO):
        raise FileNotFoundError(
            f"OmniVoice reference audio not found: {_REF_AUDIO}\n"
            f"Place the Saudi reference WAV at that path before starting the server."
        )
    if not os.path.exists(_EGY_REF_AUDIO):
        # Soft failure by design: the Saudi pipeline must start regardless; Egyptian-routed
        # turns will fall back to the Saudi voice until the clip is restored
        # (git show omnivoice-tts:voices/omnivoice-tts-egyptian-24k-v3.wav > voices/...).
        print(f"[tts] WARNING: Egyptian reference clip missing ({_EGY_REF_AUDIO}) — "
              f"Egyptian turns will use the Saudi voice.")
    _get_model()
    if CATT_ENABLED:
        print("[tts] loading CATT tashkeel model...")
        _get_tashkeel_model()
        print("[tts] CATT tashkeel ready.")
    else:
        print("[tts] CATT tashkeel disabled (CATT_ENABLED=0) — replies stay undiacritized.")
    if VOICETUT_ENABLED:
        print("[tts] loading VoiceTut-TTS (Egyptian)...")
        _get_voicetut_model()
        if _voicetut_model is not None:
            print("[tts] VoiceTut-TTS ready.")
        # else: _get_voicetut_model() already printed why — soft-fail, Egyptian turns fall
        # back to OmniVoice's existing clip+language-id path, never a startup failure.
    else:
        print("[tts] VoiceTut-TTS disabled (VOICETUT_TTS_ENABLED=0) — Egyptian stays on OmniVoice.")


def _get_model():
    global _model, _clone_prompt, _egy_clone_prompt
    with _model_lock:
        if _model is None:
            from omnivoice import OmniVoice  # type: ignore[import-untyped]
            _model = OmniVoice.from_pretrained(_MODEL_ID, device_map=_DEVICE, dtype=torch.float16)
            _clone_prompt = _model.create_voice_clone_prompt(_REF_AUDIO, _REF_TEXT)
            # Precompute the Egyptian clone prompt too (old-branch warm-up decision): the
            # clip encode costs ~16 ms + disk I/O — paying it at startup means the FIRST
            # Egyptian turn has no extra latency. Missing clip → stays None (Saudi fallback).
            if os.path.exists(_EGY_REF_AUDIO):
                _egy_clone_prompt = _model.create_voice_clone_prompt(_EGY_REF_AUDIO, _EGY_REF_TEXT)
        return _model


def _get_voicetut_model():
    """Lazily load the VoiceTut-TTS checkpoint + build its voice-clone prompt from the SAME
    Egyptian reference clip/transcript OmniVoice's own Egyptian voice uses — via the SAME
    already-installed omnivoice PyPI package's OmniVoice class (verified hands-on: no new
    package needed despite this model's own GitHub docs claiming otherwise). Third-party
    checkpoint, not internal code: any failure is caught and sticky — logs once, returns
    None forever after, never crashes startup or a turn."""
    global _voicetut_model, _voicetut_clone_prompt, _voicetut_load_failed
    with _voicetut_lock:
        if _voicetut_model is not None or _voicetut_load_failed:
            return _voicetut_model, _voicetut_clone_prompt
        try:
            from omnivoice import OmniVoice  # type: ignore[import-untyped]
            _voicetut_model = OmniVoice.from_pretrained(_VOICETUT_MODEL_ID, device_map=_DEVICE,
                                                         dtype=torch.float16)
            _voicetut_clone_prompt = _voicetut_model.create_voice_clone_prompt(_EGY_REF_AUDIO, _EGY_REF_TEXT)
        except Exception as e:
            print(f"[tts] VoiceTut-TTS load failed, Egyptian turns will use the "
                  f"next engine instead: {type(e).__name__}: {e}")
            _voicetut_load_failed = True
            _voicetut_model = None
        return _voicetut_model, _voicetut_clone_prompt


def _synthesize_egyptian_voicetut(text: str) -> Optional[np.ndarray]:
    """Synthesize Egyptian-dialect text via VoiceTut-TTS. Returns a 24kHz mono float32
    waveform (peak-normalized — see _VOICETUT_TARGET_PEAK), or None on ANY failure — caller
    falls back to OmniVoice's existing Egyptian path exactly as if this engine were disabled."""
    if not VOICETUT_ENABLED:
        return None
    model, clone_prompt = _get_voicetut_model()
    if model is None:
        return None
    try:
        # NOTE: no _gen_lock here — the caller (_synthesize_mp3_blocking) already holds it
        # for the whole Egyptian branch; _gen_lock is a plain threading.Lock (not reentrant),
        # so acquiring it again here would deadlock.
        audio = model.generate(text=text, voice_clone_prompt=clone_prompt)
        waveform = np.asarray(audio[0], dtype=np.float32)
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak > 1e-6:
            waveform = np.clip(waveform * (_VOICETUT_TARGET_PEAK / peak), -1.0, 1.0)
        return waveform
    except Exception as e:
        print(f"[tts] VoiceTut-TTS synthesis failed, falling back: {type(e).__name__}: {e}")
        return None



def _get_tashkeel_model():
    global _tashkeel_model
    with _tashkeel_lock:
        if _tashkeel_model is None:
            import catt_tashkeel
            _tashkeel_model = catt_tashkeel.CATTEncoderDecoder()
        return _tashkeel_model


def _add_tashkeel(text: str) -> str:
    """Diacritize Arabic text via CATT for pronunciation precision. CATT is a third-party ONNX
    model, not internal code — falls back to the plain (undiacritized) text on any error rather
    than let a tashkeel hiccup break a turn's audio."""
    try:
        return _get_tashkeel_model().do_tashkeel(text, verbose=False)
    except Exception as e:
        print(f"[tts] tashkeel failed, using plain text: {type(e).__name__}: {e}")
        return text



# ── Sentence boundary helper (verbatim) ──────────────────────────────────────────────────

def _should_flush(buffer: str, char: str, first: bool = False) -> bool:
    if char in HARD_BREAK:
        return True
    min_len = FIRST_SOFT_MIN if first else SOFT_BREAK_MIN
    if char in SOFT_BREAK and len(buffer) >= min_len:
        return True
    return False


# ── Filler-opener stripper (verbatim) ────────────────────────────────────────────────────
_OPENER_RE = re.compile(
    r'^\s*(?:'
    r'(?:yes,?\s+)?of\s+course'
    r'|sure'
    r'|certainly'
    r'|absolutely'
    r'|definitely'
    r'|great'
    r'|بكل\s+تأكيد|بالتأكيد|طبعاً|طبعا|أكيد|بالطبع'
    r')'
    r'\s*[,،.!?؟:؛]+\s*',
    re.IGNORECASE | re.UNICODE,
)

def _strip_openers(text: str) -> str:
    return _OPENER_RE.sub('', text, count=1).lstrip()


# ── Blocking synthesis helpers (run via asyncio.to_thread) ───────────────────────────────

def _synthesize_mp3_blocking(text: str, language: Optional[str] = None) -> bytes:
    """TTS inference + LAME MP3 encode in one blocking call (one to_thread dispatch). Returns
    a complete MP3 container — browser decodeAudioData requires this. `language` is used to
    gate CATT tashkeel (Fusha-only) and to pick the synthesis engine below; it is never passed
    to OmniVoice itself for any non-Egyptian turn, so generation for Najdi/Fusha/English/mixed
    is unchanged from before either diacritization or VoiceTut-TTS were added.

    Egyptian-routed turns try up to THREE tiers in order, each an already-independently-
    sensible fallback: (1) VoiceTut-TTS, ONLY if VOICETUT_ENABLED (the default Egyptian
    engine — see the VOICETUT_ENABLED block above) — (2) OmniVoice's own Egyptian clone
    prompt + native language="egyptian arabic" id if VoiceTut is disabled/unavailable/fails —
    (3) the plain Saudi voice if the Egyptian clip itself is missing.
    Every other language value keeps TODAY'S EXACT call: Saudi prompt, no language kwarg at all."""
    import lameenc
    if CATT_ENABLED and language in _TASHKEEL_LANGUAGES and not looks_najdi(text):
        text = _add_tashkeel(text)
    model = _get_model()
    waveform: Optional[np.ndarray] = None
    with _gen_lock:
        if language == "egyptian arabic":
            waveform = _synthesize_egyptian_voicetut(text)
        if waveform is None:
            # OmniVoice.generate returns a list of float32 np.ndarray (T,) at 24 kHz.
            if language == "egyptian arabic" and _egy_clone_prompt is not None:
                audio = model.generate(
                    text=text,
                    voice_clone_prompt=_egy_clone_prompt,
                    language="egyptian arabic",
                )
            else:
                audio = model.generate(
                    text=text,
                    voice_clone_prompt=_clone_prompt,
                )
            waveform = audio[0]
    pcm_int16 = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
    enc = lameenc.Encoder()
    enc.set_bit_rate(64)
    enc.set_in_sample_rate(SAMPLE_RATE)
    enc.set_channels(1)
    enc.set_quality(7)   # 7=fastest — 64 kbps speech is transparent at any quality setting
    mp3 = enc.encode(np.ascontiguousarray(pcm_int16).tobytes())
    mp3 += enc.flush()
    return mp3


async def _synthesize_mp3(text: str, language: Optional[str] = None) -> bytes:
    return await asyncio.to_thread(_synthesize_mp3_blocking, text, language)


# ── Public WebSocket API (identical signature to the Silma module) ───────────────────────

async def stream_tts_to_ws(
    token_gen: AsyncIterator[str],
    ws,
    cancel_event: asyncio.Event,
    on_first_audio=None,
    language: Optional[str] = None,
) -> None:
    """
    Consume an async token generator, synthesise sentence-by-sentence with OmniVoice,
    and send audio + text events over a WebSocket connection.

    Tokens stream to the browser continuously while a single background worker
    synthesises queued sentences — the LLM is never stalled by GPU synthesis.

    Message types:
      JSON  {"event":"token","text":...}  — emitted text (display)
      bytes <raw MP3>                      — one complete MP3 per sentence
      JSON  {"event":"tts_end"}            — all audio sent

    Cancellation checked at three points: (a) token loop top, (b) before synth, (c) after synth.
    """
    sentence_queue: asyncio.Queue = asyncio.Queue()

    async def synth_worker():
        nonlocal on_first_audio
        while True:
            sentence = await sentence_queue.get()
            if sentence is None:
                return
            if cancel_event.is_set():                               # (b)
                continue   # keep draining so the producer's sentinel is reached
            try:
                text_to_synth = _expand_abbreviations(sentence)
                text_to_synth = apply_dialect_repairs(text_to_synth, TTS_LANG_TO_DIALECT.get(language))
                audio_bytes = await _synthesize_mp3(text_to_synth, language)
            except Exception as e:
                print(f"[tts] synthesis failed, skipping sentence: {type(e).__name__}: {e}")
                continue
            if cancel_event.is_set():                               # (c)
                continue
            if on_first_audio is not None:
                await on_first_audio()
                on_first_audio = None
            await ws.send_bytes(audio_bytes)

    worker = asyncio.create_task(synth_worker())
    buffer = ""
    flushed_any = False

    async def _emit(text_chunk: str) -> None:
        # Send text to the browser (display) AND feed it into the sentence buffer (TTS),
        # so the response box and the spoken audio always carry IDENTICAL text.
        nonlocal buffer, flushed_any
        if not text_chunk:
            return
        await ws.send_json({"event": "token", "text": text_chunk})
        for char in text_chunk:
            buffer += char
            if _should_flush(buffer, char, first=not flushed_any):
                sentence = buffer.strip()
                buffer = ""
                if sentence:
                    await sentence_queue.put(sentence)
                    flushed_any = True

    head = ""
    head_done = False

    try:
        async for token in token_gen:
            if cancel_event.is_set():                               # (a)
                break

            if not head_done:
                # Buffer the leading edge so a filler opener can be stripped from display
                # and audio together. Decide on punctuation or _HEAD_PROBE_CHARS.
                head += token
                if len(head) >= _HEAD_PROBE_CHARS or any((c in HARD_BREAK or c in SOFT_BREAK) for c in token):
                    await _emit(_strip_openers(head))
                    head = ""
                    head_done = True
                continue

            await _emit(token)

        # Response ended before the head threshold (very short reply) — emit what we have.
        if not head_done and head and not cancel_event.is_set():
            await _emit(_strip_openers(head))
        # Flush any remaining text that didn't end with punctuation.
        if buffer.strip() and not cancel_event.is_set():
            await sentence_queue.put(buffer.strip())
    finally:
        # Close the LLM stream promptly so a barge-in stops Ollama generating.
        aclose = getattr(token_gen, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                pass
        await sentence_queue.put(None)   # sentinel — worker exits after the queue drains
        if cancel_event.is_set():
            # Barge-in: cancel the worker task so we don't wait for the in-flight
            # GPU synthesis to finish (can take 2-5s). The underlying thread still
            # runs to completion, but this task's await returns as soon as the
            # current to_thread call exits — no extra sentences are processed.
            worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[tts] synthesis worker error: {e}")

    if not cancel_event.is_set():
        await ws.send_json({"event": "tts_end"})
