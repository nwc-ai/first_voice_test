# ====================== WebSocket TTS client for remote engines ======================
#
# Drop-in equivalent of tts_silma_v1.stream_tts_to_ws, but instead of synthesising
# in-process it calls a remote TTS microservice (OmniVoice :8771, Qwen-TTS :8772, …)
# that runs in its OWN venv — so each engine's conflicting deps (e.g. Qwen needs
# transformers==4.57.3) stay isolated from this pipeline.
#
# The remote service does ONE thing: POST /synthesize {text, lang} -> WAV bytes.
# All the browser-critical work (sentence splitting, complete-MP3-per-sentence
# encoding at the right sample rate, barge-in cancellation, on_first_audio, tts_end)
# happens HERE via the shared tts_common.stream_sentences — the exact proven path
# Silma uses. This is why these engines will now actually produce voice in the
# browser, where before they were stuck at "returns a WAV".
# =====================================================================================

import asyncio
from typing import AsyncIterator

import httpx

import tts_common


async def stream_tts_to_ws(
    token_gen: AsyncIterator[str],
    ws,
    cancel_event: asyncio.Event,
    on_first_audio=None,
    *,
    service_url: str,
    lang: str = "ar",
) -> None:
    """Stream a remote TTS engine to the browser. Same contract as the Silma module
    plus `service_url` (the engine's base URL) and `lang` (forwarded to the service)."""
    client = httpx.AsyncClient(timeout=120)

    async def synth_fn(text: str) -> bytes:
        # POST one sentence → WAV bytes → MP3 at the WAV's real sample rate.
        # The blocking WAV-decode + MP3-encode runs off the event loop.
        resp = await client.post(
            f"{service_url}/synthesize",
            json={"text": text, "lang": lang},
        )
        resp.raise_for_status()
        wav_bytes = resp.content
        if not wav_bytes:
            raise RuntimeError(f"{service_url} returned empty body")
        return await asyncio.to_thread(tts_common.wav_bytes_to_mp3, wav_bytes)

    try:
        await tts_common.stream_sentences(
            token_gen, ws, cancel_event, on_first_audio, synth_fn
        )
    finally:
        await client.aclose()
