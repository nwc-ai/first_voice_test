"""
OmniVoice microservice  (port 8771)
===================================
Standalone TTS service in its OWN venv (isolated deps). The main first_voice_test
pipeline calls it over HTTP when you pick **omnivoice** in the TTS dropdown.

OmniVoice (k2-fsa/OmniVoice) is a Qwen3-0.6B-based omnilingual zero-shot TTS
(600+ languages, incl. Arabic) that clones a voice from a short reference clip.

Contract used by first_voice_test/tts_http_client.py:
  GET  /health      -> {"ready": bool}
  POST /synthesize  {"text": str, "lang": "ar"|"en"} -> WAV bytes (24 kHz mono)

The service only produces a WAV; the main pipeline turns it into complete MP3
per sentence and streams it to the browser (the proven path Qwen/Silma use).
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

# ── Defensive torchaudio/torchcodec shim (same box issue Silma/Qwen hit) ─────────
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
    print(f"[omnivoice] torchaudio shim skipped: {_e}", flush=True)

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_ID  = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
DEVICE    = os.environ.get("OMNIVOICE_DEVICE", "cuda:0")
PORT      = int(os.environ.get("OMNIVOICE_PORT", "8771"))
REF_AUDIO = os.environ.get("OMNIVOICE_REF",
                           os.path.join(os.path.dirname(__file__), "reference.wav"))
# Transcript of reference.wav (the Saudi clip) — same text the Silma module uses.
# OmniVoice needs the reference transcript for zero-shot cloning.
REF_TEXT  = os.environ.get(
    "OMNIVOICE_REF_TEXT",
    "الثقافة السعودية فيها عراقة وتاريخ عميق، وقيم إسلامية راسخة، "
    "وعادات وتقاليد قبلية أصيلة متوارثة.",
)

# ── Lazy model singleton ─────────────────────────────────────────────────────
_model: Optional[Any] = None
_lock = threading.Lock()


def _load() -> None:
    global _model
    from omnivoice import OmniVoice  # type: ignore[import-untyped]
    print(f"[omnivoice] loading '{MODEL_ID}' on {DEVICE} ...", flush=True)
    t0 = time.monotonic()
    _model = OmniVoice.from_pretrained(MODEL_ID, device_map=DEVICE, dtype=torch.float16)
    if not os.path.exists(REF_AUDIO):
        raise FileNotFoundError(f"reference clip not found: {REF_AUDIO}")
    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"[omnivoice] ready in {time.monotonic()-t0:.1f}s — VRAM {vram:.2f} GB", flush=True)


def _get_model() -> Any:
    global _model
    with _lock:
        if _model is None:
            _load()
    return _model


def _synth_wav(text: str, lang: str) -> bytes:
    """Blocking zero-shot synthesis → mono 24 kHz WAV bytes.
    lang is accepted for the shared contract; OmniVoice is omnilingual and infers
    the language from the text itself, so it is not passed through."""
    model = _get_model()
    t0 = time.monotonic()
    audio = model.generate(text=text, ref_audio=REF_AUDIO, ref_text=REF_TEXT)
    wav: np.ndarray = audio[0]           # list of np.ndarray (T,) @ 24 kHz
    sr = 24000
    print(f"[omnivoice] {len(text)} chars | wall={time.monotonic()-t0:.2f}s | "
          f"audio={len(wav)/sr:.2f}s @ {sr}Hz", flush=True)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
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
    try:
        _get_model()
    except Exception as e:
        print(f"[omnivoice] WARN: startup warm-load failed ({type(e).__name__}: {e}); "
              f"will retry on first request.", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
