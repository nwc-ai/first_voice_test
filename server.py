"""
server.py — Real-architecture voice pipeline for first_voice_test
=================================================================
Orchestration only — the pipeline pieces live in their own modules:
  stt.py              Silero VAD, FRCRN denoiser, faster-whisper
  routing.py          language/dialect detection, text-acceptance policy
  llm.py              Ollama client, model config, prompt construction
  tts_omnivoice_v1.py OmniVoice synthesis + CATT tashkeel

Architecture:
  - AudioWorklet: continuous 512-sample Float32 chunks at 16kHz
  - Silero VAD: server-side speech onset/end detection
  - Two loops: receive_loop (VAD+STT) + respond_loop (LLM+TTS)
  - asyncio.gather for true concurrency

Run with:
    bash /home/taha/first_voice_test/start_server.sh
"""

import asyncio
import ctypes
import gc
import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Optional

# ── Pre-load CUDA libs globally so torchcodec can find them ──────────────────
# Must use RTLD_GLOBAL so symbols are visible to later-loaded .so files.
_NVIDIA = os.path.join(os.path.dirname(__file__),
                       ".venv/lib/python3.12/site-packages/nvidia")
for _lib in [
    f"{_NVIDIA}/cu13/lib/libnvrtc.so.13",
    f"{_NVIDIA}/cu13/lib/libcublas.so.13",
    f"{_NVIDIA}/cublas/lib/libcublas.so.12",
    f"{_NVIDIA}/cudnn/lib/libcudnn.so.9",
    f"{_NVIDIA}/cuda_nvrtc/lib/libnvrtc.so.12",
]:
    try:
        ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass

import soundfile as _sf  # type: ignore[import-untyped]
import torch
import torchaudio as _torchaudio  # type: ignore[import-untyped]
import uvicorn

# torchaudio 2.11 always routes torchaudio.load() through torchcodec which
# requires CUDA NPP libs not present here. Patch it to use soundfile instead —
# soundfile reads WAV/FLAC natively with no GPU dependency at all.
def _sf_load(path: str, **_: Any) -> tuple:  # type: ignore[return]
    data, sr = _sf.read(str(path), dtype="float32", always_2d=True)  # type: ignore[call-overload]
    return torch.from_numpy(data.T), sr  # type: ignore[return-value]

_torchaudio.load = _sf_load  # type: ignore[assignment]
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(__file__))
import llm
import routing
import stt
import tts_omnivoice_v1  # type: ignore[import-untyped]  # in-process OmniVoice TTS
import time as _time
import datetime

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
LOG_DIR    = os.path.join(os.path.dirname(__file__), "logs")
PERF_LOG   = os.path.join(LOG_DIR, "interactions.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)


def _write_log(entry: dict[str, Any]) -> None:
    try:
        with open(PERF_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  [log] write error: {e}")


# Single-connection enforcement: only one active WebSocket session at a time.
# When a new browser connects, the old session is closed with code 4001 ("superseded")
# so the old tab knows NOT to reconnect — prevents the ping-pong loop where each
# reconnect kills the just-established session, triggering another reconnect.
_active_ws_task: Optional[asyncio.Task] = None
_active_ws_ref:  Optional[Any]          = None   # raw WebSocket for the close-4001 signal


# ── Startup: load all models ──────────────────────────────────────────────────

def _load_all_blocking():
    print("Loading OmniVoice TTS...")
    tts_omnivoice_v1.load_models()
    print("OmniVoice TTS ready.")
    stt.load_models_blocking()


_models_ready = asyncio.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async def _load_and_signal():
        await asyncio.to_thread(_load_all_blocking)
        await llm.warm_llm()       # pin the 27B before announcing 'ready' — no cold first turn
        _models_ready.set()
        print("All models loaded — server ready.")

    asyncio.create_task(_load_and_signal())
    yield  # server binds immediately — page loads while models warm up


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# no-store: the browser must re-fetch these pages every load. Without this it
# serves a cached page after edits/restarts — the "stuck UI, no GET in the
# server log" symptom (the cached page never contacts the server).
_NO_STORE = {"Cache-Control": "no-store, no-cache, must-revalidate"}


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"), headers=_NO_STORE)


@app.get("/review")
async def review_page():
    """HTML dashboard for comparing pipeline performance (reads /logs)."""
    return FileResponse(os.path.join(STATIC_DIR, "review.html"), headers=_NO_STORE)


@app.get("/logs")
async def get_logs() -> dict[str, Any]:
    """Return last 200 interaction log entries as JSON."""
    entries: list[Any] = []
    if os.path.exists(PERF_LOG):
        with open(PERF_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
    return {"entries": entries[-200:], "total": len(entries)}


async def _single_token(text: str):
    yield text


# ── WebSocket endpoint ────────────────────────────────────────────────────────

class _LockedWS:
    """Serializes all sends on a WebSocket through one lock.

    receive_loop, respond_loop, the TTS modules, and the keepalive ping all
    write to the same socket from different coroutines. Starlette does NOT allow
    concurrent sends — overlapping frames corrupt the stream and the browser
    drops the connection. Routing every send through this lock prevents that.
    """

    def __init__(self, ws: WebSocket):
        self._ws = ws
        self._send_lock = asyncio.Lock()

    async def send_json(self, data: Any) -> None:
        async with self._send_lock:
            await self._ws.send_json(data)

    async def send_bytes(self, data: bytes) -> None:
        async with self._send_lock:
            await self._ws.send_bytes(data)

    async def receive(self):
        return await self._ws.receive()

    def __getattr__(self, name):
        # Pass through everything else (client_state, etc.) to the real socket.
        return getattr(self._ws, name)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global _active_ws_task, _active_ws_ref
    await ws.accept()
    # Single-connection policy: supersede the previous session (if any).
    old_task   = _active_ws_task
    old_ws_raw = _active_ws_ref
    _active_ws_task = asyncio.current_task()
    _active_ws_ref  = ws          # raw, before _LockedWS wrap
    if old_task and not old_task.done():
        # Send close code 4001 ("superseded") to the old tab BEFORE cancelling.
        # The old browser sees 4001 in onclose and does NOT reconnect — breaks
        # the ping-pong loop that happens when task.cancel() alone is used.
        async def _close_old():
            if old_ws_raw:
                try:
                    await old_ws_raw.close(code=4001, reason="superseded")
                except Exception:
                    pass
            if not old_task.done():
                old_task.cancel()
        asyncio.create_task(_close_old())
    ws = _LockedWS(ws)  # all subsequent sends are serialized
    active_model = llm.MODEL
    print(f"Browser connected. Model: {active_model}  TTS: omnivoice")

    # Keepalive starts BEFORE the model-loading wait: a cold start takes minutes
    # and the browser watchdog reconnects after 10 s of silence — pings must
    # flow from the moment the socket opens, not from when models are ready.
    async def keepalive_loop():
        """Ping every 3 s so the browser doesn't see a dead connection during
        silent periods (model loading, long LLM thinking)."""
        try:
            while True:
                await asyncio.sleep(3)
                try:
                    await ws.send_json({"event": "ping"})
                except Exception:
                    break
        except asyncio.CancelledError:
            pass

    keepalive_task = asyncio.create_task(keepalive_loop())

    try:
        if not _models_ready.is_set():
            await ws.send_json({"event": "loading", "text": "جاري تحميل النماذج..."})
            await _models_ready.wait()
        await ws.send_json({"event": "ready"})
    except Exception:
        keepalive_task.cancel()
        print("Browser disconnected during model load.")
        return

    cancel_event:    asyncio.Event = asyncio.Event()
    utterance_queue: asyncio.Queue[Optional[tuple[str, str, int, int]]] = asyncio.Queue()
    ai_active      = False  # True while LLM+TTS pipeline is running
    ai_speaking    = False  # True only after first audio chunk has been sent to browser
                            # Barge-in cancels only when AI is speaking, not while thinking
    client_playing = False  # Browser-reported playback state — audio keeps coming out
                            # of the speakers after the server's turn already ended
    # Rolling conversation memory for this connection (clean turns, no per-turn wrappers).
    # Enables natural follow-ups ("وش يعني؟", "tell me more") instead of stateless replies.
    history: list[dict[str, str]] = []
    # Parallel per-PAIR dialect labels for the committed turns above (one label per
    # user+assistant pair): "egyptian" | "najdi" | "fusha" | "en". Used ONLY to clear
    # history at a dialect boundary — see llm.crosses_dialect_boundary.
    history_dialects: list[str] = []

    async def on_speech_start():
        """Called by VAD when speech onset is confirmed.

        Deliberately does NOT cancel the in-flight turn. The client PAUSES playback
        on this event; the turn is cancelled only once STT ACCEPTS the utterance
        (see receive_loop). A false trigger (cough, noise) is rejected by STT and
        the client resumes playback via `speech_rejected` — nothing is lost."""
        try:
            await ws.send_json({"event": "speech_start"})
        except Exception:
            pass

    process_chunk = stt.make_stt_processor(
        on_speech_start,
        is_ai_audible=lambda: ai_speaking or client_playing,
    )

    # ── receive_loop: reads mic chunks, runs VAD, queues utterances ──────────
    async def receive_loop():
        nonlocal client_playing
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    # Log the close code so we can tell WHY: 1000=normal (user/⏹),
                    # 1001=tab closed/navigated away, 1006=abnormal (tunnel/network drop),
                    # 1011=server error.
                    print(f"  receive_loop: websocket.disconnect code={msg.get('code')}")
                    break
                text_payload = msg.get("text")
                if text_payload:
                    # Control messages from the browser — real playback state, so
                    # barge-in stays strict while audio is still audible client-side.
                    try:
                        evt = json.loads(text_payload).get("event")
                    except Exception:
                        evt = None
                    if evt == "playback_start":
                        client_playing = True
                    elif evt == "playback_done":
                        client_playing = False
                    continue
                data = msg.get("bytes")
                if not data:
                    continue

                audio = await process_chunk(data)
                if audio is not None:
                    # Utterance complete. The in-flight turn is NOT cancelled yet —
                    # playback is merely paused client-side. Only an utterance that
                    # survives STT + all filters supersedes the current turn; every
                    # reject path sends `speech_rejected` so the client resumes.
                    async def _reject(reason: str) -> None:
                        print(f"STT rejected ({reason})")
                        try:
                            await ws.send_json({"event": "speech_rejected"})
                        except Exception:
                            pass

                    t_denoise_start = _time.monotonic()
                    audio = await asyncio.to_thread(stt.denoise_blocking, audio)
                    denoise_ms = int((_time.monotonic() - t_denoise_start) * 1000)
                    t_stt_start = _time.monotonic()
                    try:
                        text, lang = await asyncio.to_thread(stt.transcribe_blocking, audio)
                    except Exception as stt_e:
                        if "out of memory" in str(stt_e).lower():
                            print(f"STT OOM — skipping utterance, clearing CUDA cache")
                            try:
                                gc.collect()
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                            await _reject("stt-oom")
                            continue
                        raise
                    stt_ms = int((_time.monotonic() - t_stt_start) * 1000)

                    if not text:
                        await _reject("empty/low-confidence")
                        continue
                    if routing.is_mixed(text):
                        lang = "mixed"
                    if lang not in routing.ALLOWED_LANGS and lang != "mixed":
                        await _reject(f"lang {lang}: {text!r}")
                        continue
                    if len(text) < routing.MIN_TEXT_CHARS or len(text) > routing.MAX_TEXT_CHARS:
                        await _reject(f"length {len(text)} chars: {text!r}")
                        continue
                    if lang == "en" and len(text.split()) < 2:
                        # Single-word English fragments ("Okay.", "So.") burn a full
                        # LLM+TTS turn on nothing — drop them.
                        await _reject(f"en fragment: {text!r}")
                        continue
                    if routing.REPETITION_RE.search(text):
                        await _reject(f"repetition: {text!r}")
                        continue

                    # ── ACCEPTED — this utterance supersedes the in-flight turn ──
                    if ai_speaking:
                        # True barge-in: AI audio is playing — stop LLM/TTS now.
                        cancel_event.set()
                    if ai_active:
                        # AI busy (thinking or speaking) — drain queue so the
                        # latest utterance wins when the current turn finishes.
                        while not utterance_queue.empty():
                            utterance_queue.get_nowait()

                    print(f"STT [{lang}] (denoise {denoise_ms}ms + stt {stt_ms}ms): {text!r}")
                    await utterance_queue.put((text, lang, stt_ms, denoise_ms))
        except Exception as e:
            if "disconnect" not in str(e).lower():
                print(f"receive_loop: {e}")
        finally:
            cancel_event.set()                  # stop in-progress LLM/TTS immediately
            await utterance_queue.put(None)     # sentinel — stops respond_loop

    # ── respond_loop: takes utterances, runs LLM + TTS ───────────────────────
    async def respond_loop():
        nonlocal ai_active, ai_speaking
        try:
            while True:
                item = await utterance_queue.get()
                if item is None:
                    break                        # sentinel received

                text, lang, stt_ms, denoise_ms = item
                cancel_event.clear()
                ai_active   = True
                ai_speaking = False

                if routing.INJECTION_RE.search(text):
                    # Speak a short refusal instead of going silent — a mute assistant
                    # after a transcript reads as "broken", not "blocked".
                    print(f"Injection attempt blocked: {text!r}")
                    await ws.send_json({"event": "transcript", "text": text, "lang": lang})
                    refusal = ("عذراً، ما أقدر أنفذ هذا الطلب." if lang == "ar"
                               else "Sorry, I can't act on that request.")
                    try:
                        await tts_omnivoice_v1.stream_tts_to_ws(  # type: ignore[no-untyped-call]
                            token_gen=_single_token(refusal),
                            ws=ws,
                            cancel_event=cancel_event,
                            on_first_audio=None,
                            language=None,
                        )
                    except Exception as e:
                        print(f"  [warn] refusal TTS failed: {e}")
                        await ws.send_json({"event": "tts_end"})
                    ai_active = False
                    continue

                await ws.send_json({"event": "transcript", "text": text, "lang": lang})
                print(f"LLM start: {text!r}")

                turn_content, tts_language, route_meta = llm.build_turn(text, lang)

                # History clearing at an ARABIC-DIALECT BOUNDARY — rationale, scope, and
                # the never-fires-single-dialect invariant documented on
                # llm.crosses_dialect_boundary (unit-pinned in eval/test_routing.py).
                turn_label = llm.turn_dialect_label(tts_language)
                if history and llm.crosses_dialect_boundary(turn_label, history_dialects):
                    print(f"  [history] cleared at dialect boundary "
                          f"({history_dialects} → {turn_label})")
                    history.clear()
                    history_dialects.clear()

                # Full message list for /api/chat: system + rolling history + this wrapped turn.
                messages = (
                    [{"role": "system", "content": llm.SYSTEM_PROMPT}]
                    + history
                    + [{"role": "user", "content": turn_content}]
                )

                # ── Timing & response collection ──────────────────────────────
                t_llm_start      = _time.monotonic()
                t_first_token:   Optional[float] = None
                t_first_audio:   Optional[float] = None
                response_tokens: list[str]       = []

                def _on_first_token_cb():
                    nonlocal t_first_token
                    t_first_token = _time.monotonic()

                async def _on_first_audio_timed():
                    nonlocal t_first_audio, ai_speaking
                    t_first_audio = _time.monotonic()
                    ai_speaking   = True

                async def _collecting_token_gen():
                    inner = routing.filter_cjk(
                        llm.ollama_chat_token_gen(
                            messages, active_model,
                            on_first_token=_on_first_token_cb,
                        )
                    )
                    try:
                        async for tok in inner:
                            response_tokens.append(tok)
                            yield tok
                    finally:
                        # Propagate close down the chain so a barge-in actually
                        # stops the Ollama generation instead of leaving the 27B
                        # model producing tokens nobody will hear.
                        await inner.aclose()

                try:
                    await tts_omnivoice_v1.stream_tts_to_ws(  # type: ignore[no-untyped-call]
                        token_gen=_collecting_token_gen(),
                        ws=ws,
                        cancel_event=cancel_event,
                        on_first_audio=_on_first_audio_timed,
                        language=tts_language,
                    )
                    t_done = _time.monotonic()

                    final_response = "".join(response_tokens).strip()
                    if not final_response and not cancel_event.is_set():
                        # LLM produced no visible text (e.g. thinking-only response) —
                        # send a fallback so the user knows the model heard them.
                        # Not stored in history (it isn't a real answer).
                        # Guard: skip fallback if barge-in fired — the user already
                        # spoke again and hearing "I didn't catch that" over their
                        # next utterance is confusing.
                        fallback = "I didn't catch that. Could you please repeat?" if lang != "ar" else "عذراً، لم أفهم. ممكن تعيد؟"
                        print(f"  [warn] empty LLM response — sending fallback")
                        await tts_omnivoice_v1.stream_tts_to_ws(  # type: ignore[no-untyped-call]
                            token_gen=_single_token(fallback),
                            ws=ws,
                            cancel_event=cancel_event,
                            on_first_audio=_on_first_audio_timed,
                            language=tts_language,
                        )
                    elif not cancel_event.is_set():
                        # Commit the completed turn to rolling memory — CLEAN user text
                        # (not the wrapped prompt) so per-turn instructions never accumulate.
                        # Barge-in (cancelled, partial answer) is intentionally NOT stored.
                        # Match what was actually SPOKEN (tts_omnivoice_v1's synthesis-side
                        # dialect-repair pass, routing.apply_dialect_repairs) — keeps a
                        # follow-up turn from seeing the model's own raw leak usage (e.g.
                        # جداً) reinforced in its own rolling context. The logged `response`
                        # field and the printed line below intentionally stay RAW —
                        # eval/dialect_purity_lint.py reads logs/interactions.jsonl to
                        # measure the true LLM leak rate; silently "fixing" the log would
                        # blind that measurement.
                        history_response = routing.apply_dialect_repairs(
                            final_response, routing.TTS_LANG_TO_DIALECT.get(tts_language)
                        )
                        history.append({"role": "user", "content": text})
                        history.append({"role": "assistant", "content": history_response})
                        history_dialects.append(turn_label)
                        if len(history) >= llm.MAX_HISTORY_TURNS * 2:
                            del history[: len(history) - llm.MAX_HISTORY_TURNS * 2]
                            del history_dialects[: len(history_dialects) - llm.MAX_HISTORY_TURNS]

                    if final_response:
                        # OmniVoice prints nothing during synthesis, so log the
                        # assistant's reply here for terminal visibility (also in logs/).
                        print(f"  response [{lang}]: {final_response}")
                    print("LLM/TTS done.")

                    llm_ttft_ms    = int((t_first_token  - t_llm_start) * 1000) if t_first_token  else None
                    tts_first_ms   = int((t_first_audio  - t_llm_start) * 1000) if t_first_audio  else None
                    llm_total_ms   = int((t_done         - t_llm_start) * 1000)
                    # What the user actually waits after going silent: VAD tail +
                    # denoise + STT + the whole LLM/TTS turn.
                    e2e_ms         = (stt.MAX_SILENCE_CHUNKS * 32 + denoise_ms + stt_ms
                                      + int((t_done - t_llm_start) * 1000))

                    _write_log({
                        "ts":           datetime.datetime.now().isoformat(timespec="seconds"),
                        "model":        active_model,
                        "lang":         lang,
                        "transcript":   text,
                        "response":     "".join(response_tokens),
                        # Routing ground truth for eval/dialect_purity_lint (labels only —
                        # no content beyond what this row already logs).
                        "route": {
                            "tts_language":    tts_language,
                            "detected":        route_meta["detected"],
                            "requested":       route_meta["requested"],
                            "explicit_arabic": route_meta["explicit_arabic"],
                        },
                        "latency": {
                            "denoise_ms":    denoise_ms,
                            "stt_ms":        stt_ms,
                            "llm_ttft_ms":   llm_ttft_ms,
                            "llm_total_ms":  llm_total_ms,
                            "tts_first_ms":  tts_first_ms,
                            "e2e_ms":        e2e_ms,
                        },
                    })
                    print(f"  [perf] denoise={denoise_ms}ms stt={stt_ms}ms ttft={llm_ttft_ms}ms "
                          f"tts_first={tts_first_ms}ms total={llm_total_ms}ms e2e={e2e_ms}ms")

                except asyncio.CancelledError:
                    print("LLM/TTS cancelled.")
                except Exception as e:
                    print(f"LLM/TTS error: {e}")
                    import traceback; traceback.print_exc()
                finally:
                    ai_active   = False
                    ai_speaking = False
                    # Release PyTorch's reserved-but-unallocated VRAM (OmniVoice scratch
                    # tensors) back to the OS so the next utterance's denoiser and Whisper
                    # have room. Model weights stay in VRAM — only the allocator slack is freed.
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
        except Exception as e:
            print(f"respond_loop: {e}")

    try:
        await asyncio.gather(receive_loop(), respond_loop())
    except asyncio.CancelledError:
        pass   # normal: this session was cancelled by single-connection enforcement
               # (a new tab or reconnect superseded us). Cleanup runs in finally.
    finally:
        keepalive_task.cancel()
        cancel_event.set()
        if _active_ws_task is asyncio.current_task():
            _active_ws_task = None
            _active_ws_ref  = None
        print("Browser disconnected.")


if __name__ == "__main__":
    uvicorn.run(
        app, host="0.0.0.0", port=8765, log_level="info",
        # WebSocket keepalive: the default 20s ping + 20s pong-timeout will auto-CLOSE
        # the socket if the event loop or the SSH tunnel stalls briefly during a heavy
        # turn. We run our own app-level ping (every 3s) and the browser watchdog, and
        # ws.receive() detects real closes — so give the protocol layer a long leash
        # instead of letting it drop a working connection.
        ws_ping_interval=30.0,
        ws_ping_timeout=120.0,
    )
