# ==================== WebSocket-ready TTS module (VoiceTut — EGYPTIAN ONLY) ==============
#
# In-process Egyptian-Arabic TTS using mohammedaly22/VoiceTut-TTS — a fine-tune of the
# exact k2-fsa/OmniVoice model the main pipeline runs (Qwen3-0.6B backbone, 24 kHz,
# ~380 h Egyptian podcasts, language_id=arz). Because it shares the OmniVoice
# architecture, the checkpoint is loaded through the SAME omnivoice API we already
# use — no new dependency (the voicetut-tts pip wrapper internally does exactly this).
#
# This module serves Egyptian-routed turns ONLY. Fusha/Najdi/English stay on
# tts_omnivoice_v1 (PROTECTED — that file is deliberately untouched by the Egyptian
# integration). server.py picks the module per turn via _pick_tts().
#
# Public API is identical to tts_omnivoice_v1 (drop-in for server.py):
#   await stream_tts_to_ws(token_gen, ws, cancel_event, on_first_audio=None, language=None)
#
# Structure, sentence-flushing, abbreviation/opener handling, MP3 encoding, the
# sentence-queue + background synth worker, the on_first_audio / tts_end protocol,
# and the 3-point cancellation are carried over VERBATIM from tts_omnivoice_v1
# (which itself carried them from the proven Silma module). Deliberate deviations:
#   1. No CATT tashkeel anywhere — CATT is MSA-trained and must never touch Egyptian
#      text (and a verbatim copy would lazily build a SECOND CATT instance).
#   2. VoiceTut checkpoint + the Egyptian reference clip/transcript define the voice.
#   3. ensure_loaded() — non-throwing pre-flight used by server.py's dispatch, so a
#      load failure/OOM falls back to OmniVoice instead of a silent turn.
#   4. _egyptianize_tokens — deterministic MSA→Masri lexical repairs on the token
#      stream, upstream of BOTH display and synthesis (spoken == shown text).
#      Gated by EGY_REPAIRS (default on).
# =========================================================================================

import asyncio
import os
import re
import threading
from typing import Any, AsyncIterator, Optional

import numpy as np
import torch

# ── Sentence boundary constants (verbatim from tts_omnivoice_v1.py) ──────────────────────
HARD_BREAK = {'!', '?', '؟'}
SOFT_BREAK = {'.', ',', '،', ';', ':'}
SOFT_BREAK_MIN = 40
FIRST_SOFT_MIN = 20  # the very first flush happens earlier — cuts time-to-first-audio
_HEAD_PROBE_CHARS = 30  # leading text buffered before deciding whether to strip a filler opener

# ── Arabic abbreviation / glued-digit expander (verbatim from tts_omnivoice_v1.py) ───────
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

SAMPLE_RATE = 24000  # VoiceTut inherits OmniVoice's 24 kHz output rate

# Reference clip + its exact transcript define the cloned voice (Egyptian male).
# voices/ sits at the project root, one level above pipeline/.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REF_AUDIO = os.path.join(_PROJECT_ROOT, "voices", "omnivoice-tts-egyptian-24k-v4.wav")
_REF_TEXT  = "إِزَّيَّكْ النهارده؟ الجو جامد أوي والحمدلله. قالوا إن الخزان الجديد جاهز، بس هو اتشاف كام مرة؟ رأيك ايه، نروح نشوفه بقى؟"

_MODEL_ID = os.environ.get("VOICETUT_MODEL", "mohammedaly22/VoiceTut-TTS")
_DEVICE   = os.environ.get("VOICETUT_DEVICE", "cuda:0")

# ── Egyptian lexical repairs (EGY_REPAIRS=0 disables) ─────────────────────────────────────
# Deterministic backstop for residual MSA slips in Egyptian replies. Runs on the token
# stream BEFORE the display emit and the sentence buffer, so the response box and the
# audio carry identical repaired text. Exact whole-word matches only.
# جداً/جدا is VALID Egyptian and is deliberately NOT in this map (hard requirement).
_EGY_REPAIRS_ENABLED = os.environ.get("EGY_REPAIRS", "1") == "1"
_EGY_REPAIR_MAP = {
    "الذي": "اللي", "التي": "اللي", "الذين": "اللي",
    "تمشى": "تمشي",
    "تأكل": "تاكل",
}
_WORD_SPLIT_RE = re.compile(r"(\W+)", re.UNICODE)  # keeps separators as list items


async def _egyptianize_tokens(token_gen: AsyncIterator[str]):
    """Apply _EGY_REPAIR_MAP word-by-word to a token stream. Words can be split
    across tokens, so the last (possibly incomplete) word is held back until its
    boundary arrives."""
    pending = ""
    try:
        async for tok in token_gen:
            pending += tok
            parts = _WORD_SPLIT_RE.split(pending)
            pending = parts.pop() if parts else ""   # possibly-incomplete tail word
            out = "".join(_EGY_REPAIR_MAP.get(p, p) for p in parts)
            if out:
                yield out
        if pending:
            yield _EGY_REPAIR_MAP.get(pending, pending)
    finally:
        aclose = getattr(token_gen, "aclose", None)
        if aclose is not None:
            await aclose()


# ── Lazy model singleton (same shape as tts_omnivoice_v1) ─────────────────────────────────
_model = None
_model_lock = threading.Lock()
_clone_prompt = None   # reusable VoiceClonePrompt — built once with the model; passing the
                       # raw ref WAV per sentence would re-tokenize the clip on EVERY sentence.
_load_failed = False   # remembered so a broken install doesn't retry (and re-log) every turn


def load_models():
    """Warm-up hook. NOT called at startup by default — VoiceTut is lazy-loaded on
    the first Egyptian turn (VOICETUT_PRELOAD=1 in server.py opts into preload)."""
    if not os.path.exists(_REF_AUDIO):
        raise FileNotFoundError(
            f"VoiceTut Egyptian reference audio not found: {_REF_AUDIO}\n"
            f"Place the Egyptian reference WAV at that path."
        )
    _get_model()


def _get_model():
    global _model, _clone_prompt
    with _model_lock:
        if _model is None:
            from omnivoice import OmniVoice  # type: ignore[import-untyped]
            # VoiceTut is an OmniVoice fine-tune — same from_pretrained/generate API,
            # different weights. ~3 GB VRAM at fp16.
            _model = OmniVoice.from_pretrained(_MODEL_ID, device_map=_DEVICE, dtype=torch.float16)
            _clone_prompt = _model.create_voice_clone_prompt(_REF_AUDIO, _REF_TEXT)
        return _model


def ensure_loaded() -> bool:
    """Non-throwing pre-flight for server.py's engine dispatch. Returns True when
    the model is (now) loaded; False on any failure — the caller then falls back
    to OmniVoice for the turn instead of producing text with no audio."""
    global _load_failed
    if _load_failed:
        return False
    try:
        load_models()
        return True
    except Exception as e:
        _load_failed = True
        print(f"[tts-voicetut] load failed — Egyptian turns will use OmniVoice: "
              f"{type(e).__name__}: {e}")
        return False


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
    """VoiceTut inference + LAME MP3 encode in one blocking call (one to_thread
    dispatch). Returns a complete MP3 container — browser decodeAudioData requires
    this. No CATT here ever: Egyptian must not be MSA-diacritized. `language` is
    accepted for API parity with tts_omnivoice_v1 and ignored."""
    import lameenc
    model = _get_model()
    # Same call shape as the proven OmniVoice module: returns list of float32 (T,) @24 kHz.
    audio = model.generate(
        text=text,
        voice_clone_prompt=_clone_prompt,
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


async def _synthesize_mp3(text: str, language: Optional[str] = None) -> bytes:
    return await asyncio.to_thread(_synthesize_mp3_blocking, text, language)


# ── Public WebSocket API (identical signature to tts_omnivoice_v1) ───────────────────────

async def stream_tts_to_ws(
    token_gen: AsyncIterator[str],
    ws,
    cancel_event: asyncio.Event,
    on_first_audio=None,
    language: Optional[str] = None,
) -> None:
    """
    Consume an async token generator, synthesise sentence-by-sentence with VoiceTut,
    and send audio + text events over a WebSocket connection.

    Message types:
      JSON  {"event":"token","text":...}  — emitted text (display)
      bytes <raw MP3>                      — one complete MP3 per sentence
      JSON  {"event":"tts_end"}            — all audio sent

    Cancellation checked at three points: (a) token loop top, (b) before synth, (c) after synth.
    """
    if _EGY_REPAIRS_ENABLED:
        # Upstream of _emit: repairs reach the display tokens AND the sentence
        # buffer identically — spoken and shown text never diverge.
        token_gen = _egyptianize_tokens(token_gen)

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
                audio_bytes = await _synthesize_mp3(_expand_abbreviations(sentence), language)
            except Exception as e:
                print(f"[tts-voicetut] synthesis failed, skipping sentence: {type(e).__name__}: {e}")
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
            # GPU synthesis to finish. The underlying thread still runs to
            # completion, but no extra sentences are processed.
            worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[tts-voicetut] synthesis worker error: {e}")

    if not cancel_event.is_set():
        await ws.send_json({"event": "tts_end"})
