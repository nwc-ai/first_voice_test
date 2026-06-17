"""
Qwen3-TTS-KSA microservice  (port 8772)
=======================================
Standalone TTS service in its OWN venv (isolated deps: qwen-tts pins
transformers==4.57.3, which conflicts with the main pipeline's 4.57.6 — exactly
why this runs as a separate process).

Contract used by first_voice_test/tts_http_client.py:
  GET  /health      -> {"ready": bool}
  POST /synthesize  {"text": str, "lang": "ar"|"en"} -> WAV bytes (native SR, mono)

This service ONLY produces a WAV. The main pipeline turns it into a complete
MP3-per-sentence and streams it to the browser (the layer that was missing
before, which is why Qwen produced "no voice" in the browser).

Reuses the proven Qwen3Backend load/synth logic recovered from
/home/taha/devproject (git e1123d0).
"""

import asyncio
import io
import os
import sys
import threading
import time
from typing import Any, Optional

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

# ── Defensive torchaudio/torchcodec shim (same box issue Silma hit) ──────────────
# torchaudio >=2.9 routes load() through torchcodec, which needs CUDA NPP libs
# absent here. Block it and back torchaudio.load with soundfile so any internal
# torchaudio.load call in qwen_tts can't crash on import/use.
if "torchcodec" not in sys.modules:
    sys.modules["torchcodec"] = None  # type: ignore[assignment]
try:
    import torchaudio as _ta  # type: ignore[import-untyped]

    def _sf_load(uri: Any, frame_offset: int = 0, num_frames: int = -1,
                 normalize: bool = True, channels_first: bool = True,
                 format: Optional[str] = None, buffer_size: int = 4096,
                 backend: Optional[str] = None):
        data, sr = sf.read(uri if hasattr(uri, "read") else str(uri),
                           dtype="float32", always_2d=False)
        if frame_offset > 0:
            data = data[frame_offset:]
        if num_frames > 0:
            data = data[:num_frames]
        t = torch.from_numpy(data.copy())
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return t, sr
    _ta.load = _sf_load  # type: ignore[assignment]
except Exception as _e:
    print(f"[qwen-tts] torchaudio shim skipped: {_e}", flush=True)

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_ID  = os.environ.get("QWEN_TTS_MODEL", "vadimbelsky/qwen3-TTS-KSA")
DEVICE    = os.environ.get("QWEN_TTS_DEVICE", "cuda")
PORT      = int(os.environ.get("QWEN_TTS_PORT", "8772"))
# Optional override; otherwise the first speaker the model supports is used.
SPEAKER_OVERRIDE = os.environ.get("QWEN_TTS_SPEAKER", "").strip() or None

# qwen3-TTS-KSA is a CUSTOM_VOICE model: it has baked-in speaker(s) selected by
# name (generate_custom_voice) — NOT reference-clip voice cloning. The supported
# speaker/language lists are discovered from the model at load time.

# ── Lazy model singleton ─────────────────────────────────────────────────────
_model: Optional[Any] = None
_speaker: Optional[str] = None     # chosen speaker name
_languages: list = []              # supported languages (lowercased), [] if unconstrained
_lock = threading.Lock()


def _load() -> None:
    global _model, _speaker, _languages
    from qwen_tts import Qwen3TTSModel  # type: ignore[import-untyped]
    print(f"[qwen-tts] loading '{MODEL_ID}' on {DEVICE} ...", flush=True)
    t0 = time.monotonic()
    _model = Qwen3TTSModel.from_pretrained(MODEL_ID, dtype=torch.float16, device_map=DEVICE)

    speakers = _model.get_supported_speakers() or []
    langs = _model.get_supported_languages() or []
    _languages = [str(x) for x in langs]
    print(f"[qwen-tts] supported speakers: {speakers}", flush=True)
    print(f"[qwen-tts] supported languages: {_languages}", flush=True)

    if SPEAKER_OVERRIDE:
        _speaker = SPEAKER_OVERRIDE
    elif speakers:
        _speaker = speakers[0]
    else:
        _speaker = None  # model doesn't constrain speakers
    print(f"[qwen-tts] using speaker: {_speaker!r}", flush=True)

    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"[qwen-tts] ready in {time.monotonic()-t0:.1f}s — VRAM {vram:.2f} GB", flush=True)


def _get_model() -> Any:
    global _model
    with _lock:
        if _model is None:
            _load()
    return _model


def _pick_language(lang: str):
    """Map a request lang ('ar'/'en') to a language the model accepts.
    Returns None (→ 'Auto') when the model doesn't constrain languages."""
    if not _languages:
        return None
    want = "arabic" if (lang or "").lower().startswith("ar") else "english"
    for L in _languages:
        if want in L.lower():
            return L
    return _languages[0]


def _synth_wav(text: str, lang: str) -> bytes:
    """Blocking synthesis → mono WAV bytes at the model's native sample rate."""
    model = _get_model()
    t0 = time.monotonic()
    wavs, fs = model.generate_custom_voice(
        text=text,
        speaker=_speaker,
        language=_pick_language(lang),
    )
    audio: np.ndarray = wavs[0]          # float32 (T,)
    print(f"[qwen-tts] {len(text)} chars | spk={_speaker} | wall={time.monotonic()-t0:.2f}s | "
          f"audio={len(audio)/fs:.2f}s @ {fs}Hz", flush=True)
    buf = io.BytesIO()
    sf.write(buf, audio, fs, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ── HTTP API ─────────────────────────────────────────────────────────────────
app = FastAPI()


class SynthReq(BaseModel):
    text: str
    lang: str = "ar"


@app.get("/health")
def health() -> dict:
    return {"ready": _model is not None}


@app.post("/synthesize")
async def synthesize(req: SynthReq) -> Response:
    wav = await asyncio.to_thread(_synth_wav, req.text, req.lang)
    return Response(content=wav, media_type="audio/wav")


if __name__ == "__main__":
    # Warm the model at startup so the first live request isn't a cold load.
    try:
        _get_model()
    except Exception as e:
        print(f"[qwen-tts] WARN: startup warm-load failed ({type(e).__name__}: {e}); "
              f"will retry on first request.", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
