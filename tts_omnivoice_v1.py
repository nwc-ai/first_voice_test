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
# voice. A per-dialect voice registry (_VOICES) selects the clip per turn — currently just the Saudi
# default for every routed dialect (Najdi/Fusha/English) — and server.py also passes an OmniVoice
# `language=` dialect ID per turn to pin pronunciation. (Egyptian support was REMOVED 2026-07-09,
# owner decision — the "egyptian" registry key + its reference clip constants were deleted here;
# the WAV itself is left on disk at voices/omnivoice-tts-egyptian-24k-v3.wav, unreferenced.)
# =========================================================================================

import asyncio
import os
import re
import threading
from typing import Any, AsyncIterator, Optional

import numpy as np
import torch

# CATT tashkeel (diacritization) — DEFAULT ON (owner decision 2026-07-09, re-added after Egyptian/
# Hijazi removal). Same on/off-lever shape as FRCRN_ENABLED: set CATT_ENABLED=0 to revert to plain
# (undiacritized) text in one env var if it sounds bad, without a code change.
# History: tashkeel was evaluated and DROPPED in an earlier session because CATT is MSA-trained and
# mis-vocalizes colloquial words (documented example: علطول→عُلْطُولُ). That reasoning was never
# Egyptian-specific — it's about MSA-vs-dialect in general. Re-verified live on 2026-07-09 against
# real Najdi strings from this repo: Fusha diacritizes cleanly (expected, CATT IS an MSA model), but
# Najdi words get semantically mangled, not just cosmetically odd — مرة ("very") comes back as مَرّةً
# (misread as the MSA noun "a time/once"), and صج ("really") comes back as صَجَّ (misread as an
# unrelated real MSA verb root, "he shouted"). Owner chose to enable it for BOTH Najdi and Fusha
# anyway, accepting that Najdi mispronunciations of this kind can recur — this flag is the fast
# revert if that turns out to sound bad in practice.
CATT_ENABLED = os.environ.get("CATT_ENABLED", "1") == "1"

# Only diacritize text that's actually going to a Najdi or Fusha voice — never English (CATT is
# Arabic-only) and never mixed AR+EN (tts_language=None there specifically to avoid mispronouncing
# the English half; running an Arabic diacritizer over a code-switched string is untested territory).
_TASHKEEL_LANGUAGES = {"najdi arabic", "standard arabic"}

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

# Saudi DEFAULT voice (registry key "saudi") — used for every routed dialect (Najdi/Fusha/English).
_REF_AUDIO = os.path.join(os.path.dirname(__file__), "voices", "silma-tts-saudi-24k.wav")
_REF_TEXT  = "الثقافة السعودية فيها عراقة وتاريخ عميق، وقيم إسلامية راسخة، وعادات وتقاليد قبلية أصيلة متوارثة."

# Voice registry: key → (reference clip, its exact transcript). OmniVoice CLONES the reference, so the
# chosen clip IS the spoken voice. server.py currently only ever picks "saudi" (Egyptian support was
# REMOVED 2026-07-09 — the "egyptian" key + its _EGY_REF_AUDIO/_EGY_REF_TEXT constants were deleted
# here; the reference clip itself is left on disk, unreferenced). Add a new voice later by dropping a
# WAV + one entry here.
DEFAULT_VOICE = "saudi"
_VOICES: dict[str, tuple[str, str]] = {
    "saudi":    (_REF_AUDIO, _REF_TEXT),
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

# generate() is serialized: OmniVoice is not documented thread-safe, and a barge-in orphans an
# in-flight to_thread synthesis (the thread runs to completion) — without this lock the next
# turn's first sentence could run generate() CONCURRENTLY with the orphaned one on the same
# model object, doubling scratch VRAM and slowing both exactly when responsiveness matters.
_gen_lock = threading.Lock()

# Precomputed voice-clone prompts, one per registry voice (built in warm_up()). Passing
# voice_clone_prompt= to generate() skips the per-sentence reference-clip load + silence-trim +
# audio-tokenizer encode (~16 ms/sentence measured, plus disk I/O).
_voice_prompts: dict[str, Any] = {}

# ── Lazy tashkeel model singleton (same shape as _model/_model_lock above) ────────────────
_tashkeel_model = None
_tashkeel_lock = threading.Lock()


def _get_tashkeel_model():
    global _tashkeel_model
    with _tashkeel_lock:
        if _tashkeel_model is None:
            import catt_tashkeel
            _tashkeel_model = catt_tashkeel.CATTEncoderDecoder()
        return _tashkeel_model


def _add_tashkeel(text: str) -> str:
    """Diacritize Arabic text via CATT for pronunciation precision. CATT is a third-party ONNX
    model, not internal code — falls back to the plain (undiacritized) text on any error rather
    than let a tashkeel hiccup break a turn's audio."""
    try:
        return _get_tashkeel_model().do_tashkeel(text, verbose=False)
    except Exception as e:
        print(f"[tts] tashkeel failed, using plain text: {type(e).__name__}: {e}")
        return text


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
    if CATT_ENABLED:
        print("[tts] loading CATT tashkeel model...")
        _get_tashkeel_model()
        print("[tts] CATT tashkeel ready.")
    else:
        print("[tts] CATT tashkeel disabled (CATT_ENABLED=0) — replies stay undiacritized.")


def warm_up():
    """Precompute the clone prompt for every registry voice and run one tiny synthesis, so the
    first real turn pays neither first-inference CUDA kernel cost nor reference encoding.
    Call after load_models() (server lifespan does). language="standard arabic" so this dummy
    call also exercises the CATT tashkeel path — its first ONNX inference is warmed here too,
    not on the first real reply."""
    model = _get_model()
    for key, (ref_audio, ref_text) in _VOICES.items():
        if key not in _voice_prompts and os.path.exists(ref_audio):
            _voice_prompts[key] = model.create_voice_clone_prompt(ref_audio, ref_text)
    _synthesize_mp3_blocking("مرحبا.", DEFAULT_VOICE, "standard arabic")


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

def _synthesize_mp3_blocking(text: str, voice: str = DEFAULT_VOICE,
                             language: Optional[str] = None) -> bytes:
    """OmniVoice inference + LAME MP3 encode in one blocking call (one to_thread dispatch).
    Returns a complete MP3 container — browser decodeAudioData requires this.
    `voice` is a registry key ("saudi"); the precomputed clone prompt is used when
    available, else falls back to per-call ref_audio/ref_text. `language` is an OmniVoice dialect
    ID (e.g. "najdi arabic") that pins pronunciation to one dialect; None = language-agnostic.
    When `language` is Najdi/Fusha and CATT_ENABLED, `text` is diacritized before synthesis —
    display text (server.py's chunk_filter output, sent separately to the browser) is untouched."""
    import lameenc
    if CATT_ENABLED and language in _TASHKEEL_LANGUAGES:
        text = _add_tashkeel(text)
    model = _get_model()
    prompt = _voice_prompts.get(voice)
    # OmniVoice.generate returns a list of float32 np.ndarray (T,) at 24 kHz.
    with _gen_lock:   # strict serialization — see _gen_lock comment
        if prompt is not None:
            audio = model.generate(text=text, voice_clone_prompt=prompt, language=language)
        else:
            ref_audio, ref_text = _resolve_voice(voice)
            audio = model.generate(text=text, ref_audio=ref_audio, ref_text=ref_text, language=language)
    pcm_int16 = (np.clip(audio[0], -1.0, 1.0) * 32767).astype(np.int16)
    enc = lameenc.Encoder()
    enc.set_bit_rate(64)
    enc.set_in_sample_rate(SAMPLE_RATE)
    enc.set_channels(1)
    enc.set_quality(7)   # 7=fastest — 64 kbps speech is transparent at any quality setting
    mp3 = enc.encode(np.ascontiguousarray(pcm_int16).tobytes())
    mp3 += enc.flush()
    return mp3


async def _synthesize_mp3(text: str, voice: str = DEFAULT_VOICE,
                          language: Optional[str] = None) -> bytes:
    return await asyncio.to_thread(_synthesize_mp3_blocking, text, voice, language)


# ── Public WebSocket API (identical signature to the Silma module) ───────────────────────

async def stream_tts_to_ws(
    token_gen: AsyncIterator[str],
    ws,
    cancel_event: asyncio.Event,
    on_first_audio=None,
    voice: Optional[str] = None,
    language: Optional[str] = None,
    is_truncated=None,   # callable → True when the LLM stopped on its token cap
                         # (done_reason=="length"); the unterminated tail is then not spoken
    chunk_filter=None,   # callable(str) -> str applied to each flushed sentence-chunk BEFORE
                         # it is displayed or queued for TTS ("" = drop the chunk entirely).
                         # server.py uses it for the meta-leak filter + dialect word fixups.
) -> None:
    """
    Consume an async token generator, synthesise sentence-by-sentence with OmniVoice,
    and send audio + text events over a WebSocket connection.

    Text is emitted to the browser per sentence-chunk (the same flush points that feed TTS),
    not per token — so `chunk_filter` can rewrite or drop a whole sentence before ANY of it
    is shown, and the response box always carries exactly what the voice says. A single
    background worker synthesises queued sentences — the LLM is never stalled by GPU synthesis.

    Message types:
      JSON  {"event":"token","text":...}  — one flushed sentence-chunk (display)
      bytes <raw MP3>                      — one complete MP3 per sentence
      JSON  {"event":"tts_end"}            — all audio sent

    Cancellation checked at three points: (a) token loop top, (b) before synth, (c) after synth.
    """
    sentence_queue: asyncio.Queue = asyncio.Queue()
    # Pick the cloned voice for this whole turn (unknown/missing key → Saudi default).
    voice_key = voice if voice in _VOICES else DEFAULT_VOICE

    async def synth_worker():
        nonlocal on_first_audio
        while True:
            sentence = await sentence_queue.get()
            if sentence is None:
                return
            if cancel_event.is_set():                               # (b)
                continue   # keep draining so the producer's sentinel is reached
            try:
                audio_bytes = await _synthesize_mp3(_expand_abbreviations(sentence), voice_key, language)
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

    async def _deliver(chunk: str, tts: bool) -> str:
        # The single checkpoint every chunk passes through: filter/rewrite → display →
        # (optionally) TTS queue. Display and audio therefore always carry IDENTICAL text.
        # Returns the delivered text ("" when chunk_filter dropped the chunk).
        if chunk_filter is not None:
            chunk = chunk_filter(chunk)
        if not chunk:
            return ""
        await ws.send_json({"event": "token", "text": chunk + " "})
        if tts:
            await sentence_queue.put(chunk)
        return chunk

    async def _emit(text_chunk: str) -> None:
        # Feed tokens into the sentence buffer; each flushed sentence goes through _deliver.
        # (Display moved from per-token to per-chunk 2026-07-07 so the meta-leak filter and
        # dialect fixups can rewrite/drop a sentence before any of it reaches the browser.)
        nonlocal buffer, flushed_any
        if not text_chunk:
            return
        for char in text_chunk:
            buffer += char
            if _should_flush(buffer, char, first=not flushed_any):
                sentence = buffer.strip()
                buffer = ""
                if sentence and await _deliver(sentence, tts=True):
                    # a filtered-out chunk does NOT count as the first flush, so
                    # FIRST_SOFT_MIN still applies to the real first sentence
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
        # Flush any remaining text that didn't end with punctuation — UNLESS the LLM hit its
        # num_predict cap: that tail is a fragment cut mid-sentence and speaking it aloud sounds
        # broken ("...وبالتالي فإن الحضارة المصرية كا—"). It stays visible in the text box.
        if buffer.strip() and not cancel_event.is_set():
            if is_truncated is not None and is_truncated():
                print(f"[tts] dropping truncated tail (LLM token cap): {buffer.strip()[:60]!r}")
                await _deliver(buffer.strip(), tts=False)   # display-only, never spoken
            else:
                await _deliver(buffer.strip(), tts=True)
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
            # (_gen_lock makes the orphaned thread harmless: the next turn's first
            # synthesis simply queues behind it instead of running concurrently.)
            worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[tts] synthesis worker error: {e}")

    if not cancel_event.is_set():
        await ws.send_json({"event": "tts_end"})
