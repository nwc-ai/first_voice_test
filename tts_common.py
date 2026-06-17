# ========================== Shared TTS helpers ==========================
#
# Logic shared by every TTS backend so they behave IDENTICALLY in the live
# pipeline: sentence-boundary detection, Arabic abbreviation/number expansion,
# filler-opener stripping, MP3 encoding, and the token→sentence→audio streaming
# loop (with barge-in cancellation and the on_first_audio / tts_end protocol).
#
# This is the layer whose ABSENCE caused the earlier "LLM text but no voice"
# failure for OmniVoice/Qwen-TTS: those engines only produced a WAV and never
# had the WAV→complete-MP3-per-sentence→ws.send_bytes path the browser requires.
# By centralizing it here, any engine that returns audio bytes plays correctly.
#
# NOTE: the constants + helpers mirror tts_silma_v1.py (kept identical on purpose
# so Silma and the HTTP backends split sentences and clean text the same way).
# =========================================================================

import asyncio
import io
import re
from typing import AsyncIterator, Awaitable, Callable

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

# ── Sentence boundary constants (identical to tts_silma_v1.py) ──────────────────
HARD_BREAK = {'!', '?', '؟'}
SOFT_BREAK = {'.', ',', '،', ';', ':'}
SOFT_BREAK_MIN = 40
FIRST_SOFT_MIN = 20      # the very first flush happens earlier — cuts time-to-first-audio
_HEAD_PROBE_CHARS = 30   # leading text buffered before deciding whether to strip a filler opener


def _should_flush(buffer: str, char: str, first: bool = False) -> bool:
    if char in HARD_BREAK:
        return True
    min_len = FIRST_SOFT_MIN if first else SOFT_BREAK_MIN
    if char in SOFT_BREAK and len(buffer) >= min_len:
        return True
    return False


# ── Arabic abbreviation / glued-digit expander (identical to tts_silma_v1.py) ───
_ABBREV_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r'(\d)\s*هـ(?=[\s،,.:؟!]|$)', re.UNICODE), r'\1 هجري'),
    (re.compile(r'(\d)\s*م(?=[\s،,.:؟!]|$)', re.UNICODE), r'\1 ميلادي'),
    (re.compile(r'ق\.?\s*م(?=[\s،,.:؟!]|$)', re.UNICODE), 'قبل الميلاد'),
    (re.compile(r'(\d)\s*%', re.UNICODE), r'\1 بالمئة'),
    (re.compile(r'\bد\.\s+', re.UNICODE), 'دكتور '),
    (re.compile(r'\bأ\.\s+', re.UNICODE), 'أستاذ '),
    (re.compile(r'\bإلخ\b', re.UNICODE), 'وما إلى ذلك'),
    # Separate digits glued to Arabic letters so the number normalizer can see them.
    (re.compile(r'([؀-ۿ])(\d)', re.UNICODE), r'\1 \2'),
    (re.compile(r'(\d)([؀-ۿ])', re.UNICODE), r'\1 \2'),
]


def _expand_abbreviations(text: str) -> str:
    for pattern, replacement in _ABBREV_RULES:
        text = pattern.sub(replacement, text)
    return text


# ── Filler-opener stripper (identical to tts_silma_v1.py) ───────────────────────
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


# ── MP3 encoding (sample-rate aware) ────────────────────────────────────────────
def pcm_to_mp3(pcm_int16: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono int16 PCM → complete MP3 bytes at the given sample rate.

    The browser's decodeAudioData requires a COMPLETE MP3 container per call.
    sample_rate must be the audio's real rate (16 kHz Qwen, 24 kHz Silma, …) —
    getting it wrong is exactly what makes the browser silently fail to decode.
    """
    import lameenc
    enc = lameenc.Encoder()
    enc.set_bit_rate(64)
    enc.set_in_sample_rate(int(sample_rate))
    enc.set_channels(1)
    enc.set_quality(5)
    mp3 = enc.encode(pcm_int16.tobytes())
    mp3 += enc.flush()
    return mp3


def wav_bytes_to_mp3(wav_bytes: bytes) -> bytes:
    """Read a WAV byte string (any sample rate, mono/stereo) → MP3 at its real SR.

    Used by the HTTP backends: a TTS service returns a WAV, we decode it, take
    channel 0, and re-encode MP3 at the WAV's own sample rate. This removes the
    sample-rate-mismatch bug class — the service never has to know the browser.
    """
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="int16", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data[:, 0]
    return pcm_to_mp3(np.ascontiguousarray(data), sr)


# ── Generic token→sentence→audio streaming loop ─────────────────────────────────
async def stream_sentences(
    token_gen: AsyncIterator[str],
    ws,
    cancel_event: asyncio.Event,
    on_first_audio,
    synth_fn: Callable[[str], Awaitable[bytes]],
) -> None:
    """Drive any TTS backend over a WebSocket using one shared, proven path.

    `synth_fn(text) -> mp3_bytes` is the ONLY engine-specific piece; everything
    else (token display, opener strip, sentence boundaries, abbreviation/number
    expansion, the background synth worker, barge-in cancellation, on_first_audio,
    and the final tts_end) is identical for every backend.

    Events sent:  {"event":"token","text":...} per emitted text · raw MP3 bytes
    per sentence · {"event":"tts_end"} at the end (unless cancelled).
    """
    sentence_queue: asyncio.Queue = asyncio.Queue()
    first_audio_cb = {"fn": on_first_audio}

    async def synth_worker():
        while True:
            sentence = await sentence_queue.get()
            if sentence is None:
                return
            if cancel_event.is_set():
                continue                                   # (b) before synthesis
            try:
                audio_bytes = await synth_fn(_expand_abbreviations(sentence))
            except Exception as e:
                # LOUD on purpose — a silent skip is what hid "no voice" before.
                print(f"[tts] synthesis FAILED for {sentence!r}: {type(e).__name__}: {e}")
                continue
            if not audio_bytes:
                print(f"[tts] synthesis returned EMPTY audio for {sentence!r}")
                continue
            if cancel_event.is_set():
                continue                                   # (c) after synthesis
            if first_audio_cb["fn"] is not None:
                await first_audio_cb["fn"]()
                first_audio_cb["fn"] = None
            await ws.send_bytes(audio_bytes)

    worker = asyncio.create_task(synth_worker())
    buffer = ""
    flushed_any = False

    async def _emit(text_chunk: str) -> None:
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
            if cancel_event.is_set():                       # (a) top of token loop
                break
            if not head_done:
                head += token
                if len(head) >= _HEAD_PROBE_CHARS or any((c in HARD_BREAK or c in SOFT_BREAK) for c in token):
                    await _emit(_strip_openers(head))
                    head = ""
                    head_done = True
                continue
            await _emit(token)

        if not head_done and head and not cancel_event.is_set():
            await _emit(_strip_openers(head))
        if buffer.strip() and not cancel_event.is_set():
            await sentence_queue.put(buffer.strip())
    finally:
        aclose = getattr(token_gen, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                pass
        await sentence_queue.put(None)
        try:
            await worker
        except Exception as e:
            print(f"[tts] synthesis worker error: {e}")

    if not cancel_event.is_set():
        await ws.send_json({"event": "tts_end"})
