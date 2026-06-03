"""
server.py — Real-architecture voice pipeline for first_voice_test
=================================================================
Matches the architecture in REALTIME_VOICE_ARCHITECTURE.md:
  - AudioWorklet: continuous 512-sample Float32 chunks at 16kHz
  - Silero VAD: server-side speech onset/end detection
  - Two loops: receive_loop (VAD+STT) + respond_loop (LLM+TTS)
  - asyncio.gather for true concurrency

Run with:
    bash /home/taha/first_voice_test/start_server.sh
"""

import asyncio
import ctypes
import json
import os
import re
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

import httpx
import numpy as np
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
import tts_silma_v1  # type: ignore[import-untyped]

STATIC_DIR  = os.path.join(os.path.dirname(__file__), "static")
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL       = "qwen3.5:9b"
SYSTEM_PROMPT = (
    "You are a voice assistant that supports Arabic dialects and English ONLY. "
    "ABSOLUTE RULES — never break these: "
    "1. If the user speaks English → reply in English only. "
    "2. If the user speaks Arabic → detect their exact dialect and reply in that SAME dialect: "
    "   Najdi (نجدي): use وش/إيش, أبغى, زين, الحين, ماله, يبيلك — "
    "   Hijazi (حجازي): use إيش, وين, كيف, بدي, تعال, ما عندي — "
    "   Egyptian (مصري): use إيه, عايز, دلوقتي, مش, إزيك — "
    "   Levantine (شامي): use شو, هيك, بدي, هلق, كتير — "
    "   Moroccan (مغربي): use واش, بغيت, دابا, مزيان, كيداير — "
    "   Gulf/Khaleeji (خليجي): use شلونك, وايد, يبه, زين, ما أدري. "
    "3. If the user mixes Arabic and English (code-switching) → reply in the same natural mix, matching their Arabic dialect. "
    "4. If Arabic dialect is unclear → use Modern Standard Arabic. "
    "5. NEVER mix two Arabic dialects in one response. "
    "6. NEVER use Chinese, Japanese, Korean, Cyrillic, Vietnamese or any non-Arabic/Latin script. "
    "7. Keep answers short, conversational, no markdown, no symbols."
)

# VAD tuning (matches real architecture)
MIN_SPEECH_CHUNKS  = 4   # 4 × 32 ms ≈ 128 ms to confirm speech onset
MAX_SILENCE_CHUNKS = 25  # 25 × 32 ms ≈ 800 ms silence to end utterance
SAMPLE_RATE        = 16000

# Module-level model singletons — loaded once at startup
_vad_model:     Any = None
_whisper_model: Any = None
_denoiser:      Any = None   # ClearVoice FRCRN — None if failed to load


# ── Startup: load all models ──────────────────────────────────────────────────

def _load_all_blocking():
    global _vad_model, _whisper_model, _denoiser

    print("Loading Silma TTS...")
    tts_silma_v1.load_models()
    print("Silma TTS ready.")

    print("Loading Silero VAD...")
    _vad_model, _ = torch.hub.load(  # type: ignore[misc]
        "snakers4/silero-vad", "silero_vad",
        force_reload=False, trust_repo="check",
    )
    _vad_model.eval()  # type: ignore[union-attr]
    print("Silero VAD ready.")

    print("Loading faster-whisper large-v3...")
    from faster_whisper import WhisperModel  # type: ignore[import-untyped]
    _whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    print("faster-whisper ready.")

    print("Loading FRCRN denoiser...")
    try:
        from clearvoice import ClearVoice  # type: ignore[import-untyped]
        _denoiser = ClearVoice(task="speech_enhancement", model_names=["FRCRN_SE_16K"])
        print("FRCRN denoiser ready.")
    except Exception as e:
        print(f"FRCRN denoiser failed to load — denoising will be skipped: {e}")


_models_ready = asyncio.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async def _load_and_signal():
        await asyncio.to_thread(_load_all_blocking)
        _models_ready.set()
        print("All models loaded — server ready.")

    asyncio.create_task(_load_and_signal())
    yield  # server binds immediately — page loads while models warm up


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ── Per-connection VAD + STT processor ───────────────────────────────────────

def make_stt_processor(on_speech_start: Any):
    """
    Returns an async process_chunk(data: bytes) coroutine.
    Each call processes one 512-sample chunk.
    Returns np.ndarray (full utterance audio) when speech ends, else None.
    """
    speech_buffer:      list[Any] = []
    in_speech:          bool = False
    silence_chunks:     int  = 0
    speech_chunks_count: int = 0

    async def process_chunk(data: bytes) -> Optional[Any]:
        nonlocal speech_buffer, in_speech, silence_chunks, speech_chunks_count

        pcm = np.frombuffer(data, dtype=np.float32).copy()
        tensor = torch.from_numpy(pcm).unsqueeze(0)  # type: ignore[arg-type]

        with torch.no_grad():
            speech_prob: float = _vad_model(tensor, SAMPLE_RATE).item()

        is_speech = speech_prob >= 0.5

        if is_speech:
            speech_buffer.append(pcm)
            silence_chunks = 0
            if not in_speech:
                speech_chunks_count += 1
                if speech_chunks_count >= MIN_SPEECH_CHUNKS:
                    in_speech = True
                    await on_speech_start()
        elif in_speech:
            speech_buffer.append(pcm)
            silence_chunks += 1
            if silence_chunks >= MAX_SILENCE_CHUNKS:
                audio = np.concatenate(speech_buffer)
                speech_buffer      = []
                in_speech          = False
                silence_chunks     = 0
                speech_chunks_count = 0
                return audio
        else:
            speech_chunks_count = 0
            speech_buffer = []   # type: ignore[assignment]  # discard false-start chunks

        return None

    return process_chunk


LANG_PROB_THRESHOLD    = 0.25   # discard if Whisper isn't confident about the language
LANG_PROB_THRESHOLD_AR = 0.10  # Arabic misfires as Urdu/Punjabi/Farsi — only block pure noise
WORD_CONF_THRESHOLD    = 0.3   # discard if mean per-word confidence is too low
ALLOWED_LANGS          = {"ar", "en"}

# Detects code-switching: text contains both Arabic script and Latin words.
_ARABIC_CHARS_RE = re.compile(r'[؀-ۿ]')
_LATIN_WORDS_RE  = re.compile(r'[a-zA-Z]{2,}')

def _is_mixed(text: str) -> bool:
    return bool(_ARABIC_CHARS_RE.search(text)) and bool(_LATIN_WORDS_RE.search(text))

# Whisper mistakes Arabic for these languages — remap them all to ar.
# Includes Arabic-script langs (ur/fa/ps/ug/sd) AND Punjabi (pa) which Whisper
# also confuses with Arabic despite different script.
_ARABIC_SCRIPT_REMAP = {"ur", "fa", "ps", "ug", "prs", "ckb", "sd", "pa"}
MIN_TEXT_CHARS      = 3
MAX_TEXT_CHARS      = 500

# Strips CJK, full-width punctuation (？！), and Cyrillic from LLM tokens.
_UNWANTED_SCRIPT_RE = re.compile(
    r"[一-鿿"          # CJK unified ideographs
    r"㐀-䶿"           # CJK extension A
    r"豈-﫿"           # CJK compatibility ideographs
    r"　-〿"           # CJK symbols & punctuation
    r"゠-ヿ"           # katakana
    r"぀-ゟ"           # hiragana
    r"가-힯"           # hangul syllables
    r"＀-￯"           # fullwidth/halfwidth forms incl. ？！
    r"Ѐ-ӿ"           # Cyrillic
    r"Ԁ-ԯ]+",        # Cyrillic supplement
    re.UNICODE,
)

async def _filter_cjk(token_gen: Any):
    async for token in token_gen:
        cleaned = _UNWANTED_SCRIPT_RE.sub("", token)
        if cleaned:
            yield cleaned

# Detects ASR stuck-loops: "ا ا ا ا" or "هل هل هل هل"
_REPETITION_RE = re.compile(r"(.)\1{4,}|(\b\S+\b)(\s+\2){3,}", re.UNICODE)

# Prompt injection patterns (Arabic + English + Urdu)
_INJECTION_RE = re.compile(
    r"ignore\s+(previous|prior|all)\s+instructions?"
    r"|تجاهل\s+(التعليمات|الأوامر|السابق)"
    r"|forget\s+(your\s+)?(previous|prior|all)"
    r"|you\s+are\s+now\s+"
    r"|نسيان\s+التعليمات"
    r"|<\s*(system|instructions?)\s*>"
    r"|system\s*:",
    re.IGNORECASE | re.UNICODE,
)


def _denoise_blocking(audio: Any) -> Any:
    if _denoiser is None:
        return audio
    try:
        # FRCRN expects (batch, samples) — reshape 1D audio before passing in
        result = _denoiser(audio.reshape(1, -1))  # type: ignore[call-overload]
        if isinstance(result, np.ndarray) and result.size > 0:
            return result.squeeze()
    except Exception as e:
        print(f"Denoiser error (passing audio through): {e}")
    return audio


_TRANSCRIBE_KWARGS: dict[str, Any] = dict(
    beam_size=5,
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 300},
    word_timestamps=True,
)

def _transcribe_blocking(audio: Any) -> tuple[str, str]:
    # First pass: auto language detection
    segments, info = _whisper_model.transcribe(audio, **_TRANSCRIBE_KWARGS)
    lang      = info.language
    lang_prob = info.language_probability

    if lang in _ARABIC_SCRIPT_REMAP:
        # Whisper confused Arabic with an Arabic-script language and transcribed
        # in Urdu/Farsi text. Re-run with language="ar" forced so we get proper
        # Arabic script output instead of Urdu Nastaliq.
        print(f"  whisper: remapped {lang} → ar, re-transcribing in Arabic")
        lang = "ar"
        segments, _ = _whisper_model.transcribe(
            audio, language="ar", **_TRANSCRIBE_KWARGS
        )

    threshold = LANG_PROB_THRESHOLD_AR if lang == "ar" else LANG_PROB_THRESHOLD
    print(f"  whisper: lang={lang} lang_prob={lang_prob:.2f}")
    if lang_prob < threshold:
        print(f"  → dropped: lang_prob {lang_prob:.2f} < {threshold}")
        return "", lang
    segments = list(segments)
    all_words: list[Any] = [w for s in segments for w in (s.words or [])]
    if all_words:
        mean_conf: float = sum(float(w.probability) for w in all_words) / len(all_words)  # type: ignore[union-attr]
        print(f"  whisper: word_conf={mean_conf:.2f}")
        if mean_conf < WORD_CONF_THRESHOLD:
            print(f"  → dropped: word_conf {mean_conf:.2f} < {WORD_CONF_THRESHOLD}")
            return "", lang
    text = " ".join(s.text.strip() for s in segments).strip()
    torch.cuda.empty_cache()   # release Whisper's activation memory
    return text, lang


# ── LLM token generator ───────────────────────────────────────────────────────

async def ollama_token_gen(prompt: str, system: str):
    payload: dict[str, Any] = {
        "model":  MODEL,
        "system": system,
        "prompt": prompt,
        "stream": True,
        "think":  False,   # disable Qwen3.5 thinking mode
    }
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", OLLAMA_URL, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("Browser connected.")

    if not _models_ready.is_set():
        await ws.send_json({"event": "loading", "text": "جاري تحميل النماذج..."})
        await _models_ready.wait()
    await ws.send_json({"event": "ready"})

    cancel_event:    asyncio.Event                      = asyncio.Event()
    utterance_queue: asyncio.Queue[Optional[tuple[str, str]]] = asyncio.Queue()
    ai_active   = False  # True while LLM+TTS pipeline is running
    ai_speaking = False  # True only after first audio chunk has been sent to browser
                         # Barge-in cancels only when AI is speaking, not while thinking

    async def _on_first_audio():
        nonlocal ai_speaking
        ai_speaking = True

    async def on_speech_start():
        """Called by VAD when speech onset is confirmed."""
        # Only cancel if AI is already playing audio (true barge-in).
        # While AI is still thinking (ai_active but not ai_speaking), let the
        # LLM finish so the user actually gets a response.
        if ai_speaking:
            cancel_event.set()
        try:
            await ws.send_json({"event": "speech_start"})
        except Exception:
            pass

    process_chunk = make_stt_processor(on_speech_start)

    # ── receive_loop: reads mic chunks, runs VAD, queues utterances ──────────
    async def receive_loop():
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if not data:
                    continue

                audio = await process_chunk(data)
                if audio is not None:
                    if ai_speaking:
                        # True barge-in: AI is actively playing audio — cancel it.
                        cancel_event.set()
                    if ai_active:
                        # AI is busy (thinking or speaking) — drain queue so the
                        # latest utterance wins when the current turn finishes.
                        while not utterance_queue.empty():
                            utterance_queue.get_nowait()

                    audio = await asyncio.to_thread(_denoise_blocking, audio)
                    text, lang = await asyncio.to_thread(_transcribe_blocking, audio)

                    if not text:
                        continue
                    # Promote to "mixed" if text contains both Arabic and Latin —
                    # overrides Whisper's single-language tag before the ALLOWED_LANGS check.
                    if _is_mixed(text):
                        lang = "mixed"
                    if lang not in ALLOWED_LANGS and lang != "mixed":
                        print(f"STT [{lang}] rejected: {text!r}")
                        continue
                    if len(text) < MIN_TEXT_CHARS or len(text) > MAX_TEXT_CHARS:
                        print(f"STT [{lang}] length-rejected ({len(text)} chars): {text!r}")
                        continue
                    if _REPETITION_RE.search(text):
                        print(f"STT [{lang}] repetition-rejected: {text!r}")
                        continue

                    print(f"STT [{lang}]: {text!r}")
                    await utterance_queue.put((text, lang))
        except Exception as e:
            if "disconnect" not in str(e).lower():
                print(f"receive_loop: {e}")
        finally:
            await utterance_queue.put(None)     # sentinel — stops respond_loop

    # ── respond_loop: takes utterances, runs LLM + TTS ───────────────────────
    async def respond_loop():
        nonlocal ai_active, ai_speaking
        try:
            while True:
                item = await utterance_queue.get()
                if item is None:
                    break                        # sentinel received

                text, lang = item
                cancel_event.clear()
                ai_active   = True
                ai_speaking = False

                if _INJECTION_RE.search(text):
                    print(f"Injection attempt blocked: {text!r}")
                    await ws.send_json({"event": "transcript", "text": text, "lang": lang})
                    await ws.send_json({"event": "tts_end"})
                    ai_active = False
                    continue

                await ws.send_json({"event": "transcript", "text": text, "lang": lang})
                print(f"LLM start: {text!r}")

                if lang == "mixed":
                    lang_instruction = (
                        "The user is mixing Arabic and English (code-switching). "
                        "Reply naturally in the SAME mix of Arabic and English they used. "
                        "For the Arabic parts, match their dialect "
                        "(Najdi, Hijazi, Egyptian, Levantine, Moroccan, or Gulf/Khaleeji). "
                        "Do NOT force a reply into all-Arabic or all-English."
                    )
                elif lang == "ar":
                    lang_instruction = (
                        "The user spoke Arabic. Detect their exact dialect "
                        "(Najdi, Hijazi, Egyptian, Levantine, Moroccan, or Gulf/Khaleeji) "
                        "from their vocabulary and reply in that EXACT same dialect. "
                        "Najdi vs Hijazi: Najdi uses وش/أبغى/الحين; Hijazi uses وين/بدي/كيف. "
                        "Do NOT use Fusha/MSA unless the dialect is completely unclear."
                    )
                else:
                    lang_instruction = "The user spoke English. Reply in English only."
                prompt_with_lang = f"{lang_instruction}\n\nUser: {text}"

                try:
                    await tts_silma_v1.stream_tts_to_ws(  # type: ignore[no-untyped-call]
                        token_gen=_filter_cjk(ollama_token_gen(prompt_with_lang, SYSTEM_PROMPT)),
                        ws=ws,
                        cancel_event=cancel_event,
                        on_first_audio=_on_first_audio,
                    )
                    print("LLM/TTS done.")
                except asyncio.CancelledError:
                    print("LLM/TTS cancelled.")
                except Exception as e:
                    print(f"LLM/TTS error: {e}")
                    import traceback; traceback.print_exc()
                finally:
                    ai_active   = False
                    ai_speaking = False
                    torch.cuda.empty_cache()   # release Silma TTS activation memory
        except Exception as e:
            print(f"respond_loop: {e}")

    try:
        await asyncio.gather(receive_loop(), respond_loop())
    finally:
        cancel_event.set()
        print("Browser disconnected.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
