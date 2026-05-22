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
import threading
from importlib.resources import files
from typing import AsyncIterator

import numpy as np
import torch
import torchaudio

# ── Sentence boundary constants (copied verbatim from tts_habibi_v3.py) ──────────────────
HARD_BREAK = {'!', '?', '؟'}
SOFT_BREAK = {'.', ',', '،', ';', ':'}
SOFT_BREAK_MIN = 40

SAMPLE_RATE = 24000  # Silma TTS output sample rate (from config.yaml)

# Reference audio that defines the voice style — bundled with silma_tts package.
# This is a native Arabic female voice at 24 kHz.
_REF_AUDIO = str(files("silma_tts").joinpath("infer/ref_audio_samples/ar.ref.24k.wav"))
_REF_TEXT  = "ويدقق النظر في القرآن الكريم وسائر الكتب السماوية ويتبع مسالك الرسل العظام عليهم الصلاة والسلام."

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

def _should_flush(buffer: str, char: str) -> bool:
    if char in HARD_BREAK:
        return True
    if char in SOFT_BREAK and len(buffer) >= SOFT_BREAK_MIN:
        return True
    return False


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

    Message types sent:
      JSON  {"event": "token", "text": "<token>"}   — one per LLM token
      bytes  <raw MP3>                               — one complete MP3 per sentence
      JSON  {"event": "tts_end"}                    — signals all audio has been sent

    Cancellation is checked at three points per sentence:
      (a) top of token loop  (b) before synthesis  (c) after synthesis
    """
    buffer = ""

    async for token in token_gen:
        if cancel_event.is_set():                                   # (a)
            break

        await ws.send_json({"event": "token", "text": token})

        for char in token:
            buffer += char
            if _should_flush(buffer, char):
                if buffer.strip() and not cancel_event.is_set():    # (b)
                    audio_bytes = await _synthesize_mp3(buffer.strip())
                    if not cancel_event.is_set():                   # (c)
                        if on_first_audio is not None:
                            await on_first_audio()
                            on_first_audio = None
                        await ws.send_bytes(audio_bytes)
                buffer = ""

    # Flush any remaining text that didn't end with punctuation
    if buffer.strip() and not cancel_event.is_set():
        audio_bytes = await _synthesize_mp3(buffer.strip())
        if not cancel_event.is_set():
            if on_first_audio is not None:
                await on_first_audio()
                on_first_audio = None
            await ws.send_bytes(audio_bytes)

    if not cancel_event.is_set():
        await ws.send_json({"event": "tts_end"})
