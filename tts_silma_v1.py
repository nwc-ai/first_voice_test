# ========================== WebSocket-ready TTS module (Silma) ==========================
#
# Drop-in replacement for tts_habibi_v3.py (edge_tts) using Silma TTS locally on GPU.
#
# Public API is identical to tts_habibi_v3.py:
#   await stream_tts_to_ws(token_gen, ws, cancel_event, on_first_audio=None)
#
# Silma TTS is a zero-shot voice-cloning model — it needs a short reference audio clip
# to define the voice style. We use the Arabic sample bundled with the package.
# =========================================================================================

import asyncio
import os
import re
import threading
from typing import AsyncIterator

import numpy as np
import torch
import torchaudio

# ── Sentence boundary constants (copied verbatim from tts_habibi_v3.py) ──────────────────
HARD_BREAK = {'!', '?', '؟'}
SOFT_BREAK = {'.', ',', '،', ';', ':'}
SOFT_BREAK_MIN = 40
FIRST_SOFT_MIN = 20  # the very first flush happens earlier — cuts time-to-first-audio
_HEAD_PROBE_CHARS = 30  # leading text buffered before deciding whether to strip a filler opener

# ── Arabic abbreviation expander ──────────────────────────────────────────────────────────
# Runs on each sentence buffer before Silma synthesises it.
# LLM is also instructed via system prompt — this is the safety net.
_ABBREV_RULES: list[tuple[re.Pattern, str]] = [
    # Calendar: 1744هـ → 1744 هجري  |  1744 هـ → 1744 هجري
    (re.compile(r'(\d)\s*هـ(?=[\s،,.:؟!]|$)', re.UNICODE), r'\1 هجري'),
    # Calendar: 1932م → 1932 ميلادي  (only م directly after digits)
    (re.compile(r'(\d)\s*م(?=[\s،,.:؟!]|$)', re.UNICODE), r'\1 ميلادي'),
    # Before Common Era: ق.م or ق. م
    (re.compile(r'ق\.?\s*م(?=[\s،,.:؟!]|$)', re.UNICODE), 'قبل الميلاد'),
    # Percent: 30% → 30 بالمئة
    (re.compile(r'(\d)\s*%', re.UNICODE), r'\1 بالمئة'),
    # Academic titles before a space+name: د. → دكتور
    (re.compile(r'\bد\.\s+', re.UNICODE), 'دكتور '),
    # Academic title: أ. → أستاذ
    (re.compile(r'\bأ\.\s+', re.UNICODE), 'أستاذ '),
    # Etc: إلخ → وما إلى ذلك
    (re.compile(r'\bإلخ\b', re.UNICODE), 'وما إلى ذلك'),
    # Separate digits glued to Arabic letters so Silma's number normalizer can SEE the
    # number and speak it as words. Logs showed "و2013" surviving as raw digits because the
    # conjunction was fused to it. Runs LAST so the calendar rules above (1932م etc.) match first.
    (re.compile(r'([؀-ۿ])(\d)', re.UNICODE), r'\1 \2'),
    (re.compile(r'(\d)([؀-ۿ])', re.UNICODE), r'\1 \2'),
]

def _expand_abbreviations(text: str) -> str:
    for pattern, replacement in _ABBREV_RULES:
        text = pattern.sub(replacement, text)
    return text

SAMPLE_RATE = 24000  # Silma TTS output sample rate (from config.yaml)

# Reference audio that defines the voice style — a custom 24 kHz mono recording.
# MUST stay under Silma's 8.05s reference limit: above that, Silma clips the audio
# AND ignores _REF_TEXT, re-transcribing the clip with its own (error-prone) ASR.
# This clip is ~7.64s and _REF_TEXT is the exact transcript of what is spoken in it,
# so Silma uses the voice whole and conditions on the correct text.
_REF_AUDIO = os.path.join(os.path.dirname(__file__), "voices", "silma-tts-saudi-24k.wav")
_REF_TEXT  = "الثقافة السعودية فيها عراقة وتاريخ عميق، وقيم إسلامية راسخة، وعادات وتقاليد قبلية أصيلة متوارثة."

# ── Lazy model singleton ──────────────────────────────────────────────────────────────────
_model = None
_model_lock = threading.Lock()


# PyTorch 2.11+cu130 on RTX 5090 crashes inside torchaudio.transforms.MelSpectrogram when
# it calls spec_f.abs() on a complex CUDA tensor.  Running the mel transform on CPU sidesteps
# the broken kernel; the rest of Silma inference (the flow-matching model) stays on GPU.
_mel_stft_cache: dict = {}

def _vocos_mel_on_cpu(waveform, n_fft=1024, n_mel_channels=100, target_sample_rate=24000,
                      hop_length=256, win_length=1024):
    device = waveform.device
    key = (n_fft, n_mel_channels, target_sample_rate, hop_length, win_length)
    if key not in _mel_stft_cache:
        _mel_stft_cache[key] = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mel_channels,
            power=1,
            center=True,
            normalized=False,
            norm=None,
        ).cpu()
    mel_stft = _mel_stft_cache[key]
    if len(waveform.shape) == 3:
        waveform = waveform.squeeze(1)
    assert len(waveform.shape) == 2
    mel = mel_stft(waveform.cpu())
    mel = mel.clamp(min=1e-5).log()
    return mel.to(device)


def load_models():
    """
    Optional warm-up hook — call from FastAPI lifespan so the first user
    does not pay the ~10-15 s model load cost.
    """
    _get_model()


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            import silma_tts.model.modules as _m
            _m.get_vocos_mel_spectrogram = _vocos_mel_on_cpu
            from silma_tts.api import SilmaTTS
            _model = SilmaTTS()
        return _model


# ── Sentence boundary helper (same logic as tts_habibi_v3.py) ────────────────────────────

def _should_flush(buffer: str, char: str, first: bool = False) -> bool:
    if char in HARD_BREAK:
        return True
    min_len = FIRST_SOFT_MIN if first else SOFT_BREAK_MIN
    if char in SOFT_BREAK and len(buffer) >= min_len:
        return True
    return False


# ── Filler-opener stripper ───────────────────────────────────────────────────────────────
# Removes throwaway lead-ins ("Of course!", "Sure,", "بالتأكيد،") from the START of a
# response — but ONLY when the opener is a standalone phrase terminated by punctuation.
# Openers that flow into content ("I'd be happy to provide…") are left alone to avoid
# awkward truncation. Applied to display + audio together so they never diverge.
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
    r'\s*[,،.!?؟:؛]+\s*',     # must be followed by punctuation → standalone filler only
    re.IGNORECASE | re.UNICODE,
)

def _strip_openers(text: str) -> str:
    return _OPENER_RE.sub('', text, count=1).lstrip()


# ── Blocking synthesis helpers (run via asyncio.to_thread) ───────────────────────────────

def _synthesize_pcm_blocking(text: str) -> np.ndarray:
    """
    Run one Silma inference pass — BLOCKING, must be called via asyncio.to_thread.
    Returns int16 mono PCM at SAMPLE_RATE.
    """
    model = _get_model()
    wav, _sr, _ = model.infer(
        ref_file=_REF_AUDIO,
        ref_text=_REF_TEXT,
        gen_text=text,
        show_info=lambda *a, **kw: None,  # silence console prints
        progress=None,                    # no tqdm bars in pipeline mode
        normalize_numbers=True,
        force_tashkeel=True,
    )
    # wav is float32 in [-1, 1] — convert to int16 for lameenc
    # NOTE: VRAM is reclaimed once per TURN via torch.cuda.empty_cache() in
    # respond_loop's finally block — NOT per sentence (that made synthesis slow,
    # since every following allocation had to go cold to the driver).
    return (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)


def _pcm_to_mp3_blocking(pcm_int16: np.ndarray) -> bytes:
    """
    Encode mono int16 PCM → complete MP3 bytes — BLOCKING.
    Browser requires a complete MP3 container per ws.send_bytes call.
    """
    import lameenc
    enc = lameenc.Encoder()
    enc.set_bit_rate(64)
    enc.set_in_sample_rate(SAMPLE_RATE)
    enc.set_channels(1)
    enc.set_quality(5)   # 2=best/slowest … 7=fastest
    mp3 = enc.encode(pcm_int16.tobytes())
    mp3 += enc.flush()
    return mp3


async def _synthesize_mp3(text: str) -> bytes:
    pcm = await asyncio.to_thread(_synthesize_pcm_blocking, text)
    mp3 = await asyncio.to_thread(_pcm_to_mp3_blocking, pcm)
    return mp3


# ── Public WebSocket API (identical signature to tts_habibi_v3.stream_tts_to_ws) ─────────

async def stream_tts_to_ws(
    token_gen: AsyncIterator[str],
    ws,
    cancel_event: asyncio.Event,
    on_first_audio=None,
) -> None:
    """
    Consume an async token generator, synthesise sentence-by-sentence with Silma TTS,
    and send audio + text events over a WebSocket connection.

    Tokens stream to the browser continuously while a single background worker
    synthesises queued sentences — the LLM is never stalled by GPU synthesis,
    so sentence N+1 is already fully generated while sentence N is on the GPU.

    Message types sent:
      JSON  {"event": "token", "text": "<token>"}   — one per LLM token
      bytes  <raw MP3>                               — one complete MP3 per sentence
      JSON  {"event": "tts_end"}                    — signals all audio has been sent

    Cancellation is checked at three points per sentence:
      (a) top of token loop  (b) before synthesis  (c) after synthesis
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
                audio_bytes = await _synthesize_mp3(_expand_abbreviations(sentence))
            except Exception as e:
                print(f"[tts] synthesis failed, skipping sentence: {e}")
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
        # so the response box and the spoken audio always carry IDENTICAL text. The opener
        # strip happens before _emit, so it removes the filler from both at once.
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
                # and audio together. Decide as soon as we hit punctuation or _HEAD_PROBE_CHARS;
                # this adds at most ~10 chars over the normal first-flush, so TTFA is unaffected.
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
        # Close the LLM stream promptly so a barge-in stops Ollama generating
        # instead of leaving it running until garbage collection.
        aclose = getattr(token_gen, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                pass
        await sentence_queue.put(None)   # sentinel — worker exits after the queue drains
        try:
            await worker
        except Exception as e:
            print(f"[tts] synthesis worker error: {e}")

    if not cancel_event.is_set():
        await ws.send_json({"event": "tts_end"})
