# ========================== WebSocket-ready TTS module (OmniVoice) =======================
#
# In-process TTS using k2-fsa/OmniVoice (omnilingual zero-shot voice cloning, 24 kHz).
#
# Public API is identical to the previous Silma module (drop-in for server.py):
#   await stream_tts_to_ws(token_gen, ws, cancel_event, on_first_audio=None)
#
# Structure, sentence-flushing, abbreviation/opener handling, MP3 encoding, the
# sentence-queue + background synth worker, the on_first_audio / tts_end protocol,
# and the 3-point cancellation are all carried over from the earlier Silma TTS module
# (not in this repo — preserved on the `multi-engine-snapshot` branch); only the model
# load + the per-sentence synthesis call are OmniVoice-specific.
#
# OmniVoice is a zero-shot voice-cloner: it needs a short reference clip + its transcript to define the
# voice. A per-dialect voice registry (_VOICES) selects the clip per turn — Saudi (default) for
# Najdi/Hijazi/Fusha/English, Egyptian (v3 clip) for Egyptian-routed turns — and server.py also passes an
# OmniVoice `language=` dialect ID per turn to pin pronunciation.
# =========================================================================================

import asyncio
import os
import re
import threading
from typing import Any, AsyncIterator, Optional

import numpy as np
import torch

# ── Sentence boundary constants (carried over from the earlier Silma TTS module) ─────────
HARD_BREAK = {'!', '?', '؟'}
SOFT_BREAK = {'.', ',', '،', ';', ':'}
SOFT_BREAK_MIN = 40
FIRST_SOFT_MIN = 20  # the very first flush happens earlier — cuts time-to-first-audio
_HEAD_PROBE_CHARS = 30  # leading text buffered before deciding whether to strip a filler opener

# ── Arabic abbreviation / glued-digit expander (carried over from the earlier Silma module) ──
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

SAMPLE_RATE = 24000  # OmniVoice output sample rate

# Saudi DEFAULT voice (registry key "saudi") — used for Najdi/Hijazi/Fusha/English. Egyptian voice below.
_REF_AUDIO = os.path.join(os.path.dirname(__file__), "voices", "silma-tts-saudi-24k.wav")
_REF_TEXT  = "الثقافة السعودية فيها عراقة وتاريخ عميق، وقيم إسلامية راسخة، وعادات وتقاليد قبلية أصيلة متوارثة."

# Egyptian reference clip (user-provided) + its exact transcript — used for Egyptian-routed turns.
_EGY_REF_AUDIO = os.path.join(os.path.dirname(__file__), "voices", "omnivoice-tts-egyptian-24k-v3.wav")
_EGY_REF_TEXT  = "في الغالب بتبقى أكتر من الطفل اللي اتولد في آخر السنة، وبالتالي أداؤه في اللعب هيكون أحسن، فده هيلفت نظر المدربين فهيهتموا بيه"

# Voice registry: key → (reference clip, its exact transcript). OmniVoice CLONES the reference, so the
# chosen clip IS the spoken voice. server.py picks the key per turn from the routed dialect (Egyptian-routed
# → "egyptian", everything else → "saudi"). Add a new voice later by dropping a WAV + one entry here.
DEFAULT_VOICE = "saudi"
_VOICES: dict[str, tuple[str, str]] = {
    "saudi":    (_REF_AUDIO, _REF_TEXT),
    "egyptian": (_EGY_REF_AUDIO, _EGY_REF_TEXT),
}

def _resolve_voice(key: Optional[str]) -> tuple[str, str]:
    """Map a voice key → (ref_audio, ref_text); fall back to the Saudi default for an unknown key or a
    missing file, so a bad/typo'd key can never break synthesis."""
    ref_audio, ref_text = _VOICES.get(key or DEFAULT_VOICE, _VOICES[DEFAULT_VOICE])
    if not os.path.exists(ref_audio):
        print(f"[tts] voice clip for '{key}' missing ({ref_audio}) — falling back to Saudi")
        return _VOICES[DEFAULT_VOICE]
    return ref_audio, ref_text

_MODEL_ID = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
_DEVICE   = os.environ.get("OMNIVOICE_DEVICE", "cuda:0")

# ── Lazy model singleton ──────────────────────────────────────────────────────────────────
_model = None
_model_lock = threading.Lock()


def load_models():
    """Optional warm-up hook — call from FastAPI lifespan so the first user
    does not pay the model load cost. Validates EVERY registry voice clip exists."""
    for _key, (_ref_audio, _ref_text) in _VOICES.items():
        if not os.path.exists(_ref_audio):
            raise FileNotFoundError(
                f"OmniVoice reference audio for voice '{_key}' not found: {_ref_audio}\n"
                f"Place the reference WAV at that path before starting the server."
            )
    _get_model()


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from omnivoice import OmniVoice  # type: ignore[import-untyped]
            _model = OmniVoice.from_pretrained(_MODEL_ID, device_map=_DEVICE, dtype=torch.float16)
        return _model


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

def _synthesize_mp3_blocking(text: str, ref_audio: str = _REF_AUDIO, ref_text: str = _REF_TEXT,
                             language: Optional[str] = None) -> bytes:
    """OmniVoice inference + LAME MP3 encode in one blocking call (one to_thread dispatch).
    Returns a complete MP3 container — browser decodeAudioData requires this.
    ref_audio/ref_text select the cloned voice (default = Saudi). `language` is an OmniVoice dialect
    ID (e.g. "egyptian arabic" → arz) that pins pronunciation to one dialect; None = language-agnostic."""
    import lameenc
    model = _get_model()
    # OmniVoice.generate returns a list of float32 np.ndarray (T,) at 24 kHz.
    audio = model.generate(
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        language=language,
    )
    pcm_int16 = (np.clip(audio[0], -1.0, 1.0) * 32767).astype(np.int16)
    enc = lameenc.Encoder()
    enc.set_bit_rate(64)
    enc.set_in_sample_rate(SAMPLE_RATE)
    enc.set_channels(1)
    enc.set_quality(7)   # 7=fastest — 64 kbps speech is transparent at any quality setting
    mp3 = enc.encode(np.ascontiguousarray(pcm_int16).tobytes())
    mp3 += enc.flush()
    return mp3


async def _synthesize_mp3(text: str, ref_audio: str = _REF_AUDIO, ref_text: str = _REF_TEXT,
                          language: Optional[str] = None) -> bytes:
    return await asyncio.to_thread(_synthesize_mp3_blocking, text, ref_audio, ref_text, language)


# ── Public WebSocket API (identical signature to the Silma module) ───────────────────────

async def stream_tts_to_ws(
    token_gen: AsyncIterator[str],
    ws,
    cancel_event: asyncio.Event,
    on_first_audio=None,
    voice: Optional[str] = None,
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
    ref_audio, ref_text = _resolve_voice(voice)   # pick the cloned voice for this whole turn

    async def synth_worker():
        nonlocal on_first_audio
        while True:
            sentence = await sentence_queue.get()
            if sentence is None:
                return
            if cancel_event.is_set():                               # (b)
                continue   # keep draining so the producer's sentinel is reached
            try:
                audio_bytes = await _synthesize_mp3(_expand_abbreviations(sentence), ref_audio, ref_text, language)
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
