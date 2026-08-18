"""
stt.py — audio-in half of the pipeline: Silero VAD, FRCRN denoiser, faster-whisper.

Owns the model singletons (loaded once via load_models_blocking) and the
per-connection VAD processor factory. All blocking GPU work here is called
through asyncio.to_thread by server.py.
"""

import gc
import os
import re
from collections import deque
from typing import Any, Optional

import numpy as np
import torch

from .routing import ALLOWED_LANGS, ARABIC_SCRIPT_REMAP

# VAD tuning (matches real architecture)
MIN_SPEECH_CHUNKS       = 4   # 4 × 32 ms ≈ 128 ms to confirm speech onset
MIN_SPEECH_CHUNKS_BARGE = 3   # ≈96 ms onset to interrupt while AI audio is audible.
                              # Lowered 9 → 5 → 3 so speaking cuts the AI off almost immediately.
                              # ASSUMES HEADPHONES — on open speakers the AI's own voice bleeds into
                              # the mic; at 3 chunks that can self-interrupt OR (with echo-cancel
                              # double-talk suppression) still feel sluggish. Headphones make it crisp.
MAX_SILENCE_CHUNKS      = 25  # 25 × 32 ms ≈ 0.8 s silence to end utterance
                              # (pre-roll + stricter onset made the old 1.28 s tail unnecessary;
                              #  raise back toward 40 if users get cut off mid-sentence)
PREROLL_CHUNKS          = 10  # ≈320 ms kept from before VAD onset — first-syllable guard
SAMPLE_RATE             = 16000

# Module-level model singletons — loaded once at startup
_vad_model:     Any = None
_whisper_model: Any = None
_denoiser:      Any = None   # ClearVoice FRCRN — None if failed to load or disabled

# A/B switch: DENOISE_ENABLED=0 disables FRCRN entirely (inference AND the model
# load, freeing its VRAM). Used to measure whether denoising actually helps STT —
# the browser already applies noiseSuppression + echoCancellation, and Whisper
# large-v3 is noise-robust, so the benefit is unproven.
DENOISE_ENABLED     = os.environ.get("DENOISE_ENABLED", "1") == "1"
_FRCRN_MIN_FREE_MB  = 150   # skip denoising if less than this much VRAM is free after cache flush
_FRCRN_MAX_SAMPLES  = SAMPLE_RATE * 4   # skip denoising for clips longer than 4 s —
                                         # FRCRN VRAM scales with length; longer clips
                                         # OOM on this GPU (qwen3.5:27b + OmniVoice loaded).
                                         # Whisper large-v3 handles longer clips fine without it.

LANG_PROB_THRESHOLD    = 0.25   # discard if Whisper isn't confident about the language
LANG_PROB_THRESHOLD_AR = 0.10  # Arabic misfires as Urdu/Punjabi/Farsi — only block pure noise
WORD_CONF_THRESHOLD    = 0.3   # discard if mean per-word confidence is too low


def load_models_blocking() -> None:
    """Load VAD + Whisper (+ FRCRN if enabled). Call once, off the event loop."""
    global _vad_model, _whisper_model, _denoiser

    print("Loading Silero VAD...")
    _vad_model, _ = torch.hub.load(  # type: ignore[misc]
        "snakers4/silero-vad", "silero_vad",
        force_reload=False, trust_repo="check",
    )
    _vad_model.eval()  # type: ignore[union-attr]
    print("Silero VAD ready.")

    print("Loading faster-whisper large-v3...")
    from faster_whisper import WhisperModel  # type: ignore[import-untyped]
    # int8_float16: ~1.5 GB less VRAM than float16, still large-v3, negligible accuracy
    # impact — frees headroom for the in-process OmniVoice + qwen3.5 on one 32 GB GPU.
    _whisper_model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
    print("faster-whisper ready.")

    if DENOISE_ENABLED:
        print("Loading FRCRN denoiser...")
        try:
            from clearvoice import ClearVoice  # type: ignore[import-untyped]
            _denoiser = ClearVoice(task="speech_enhancement", model_names=["FRCRN_SE_16K"])
            print("FRCRN denoiser ready.")
        except Exception as e:
            print(f"FRCRN denoiser failed to load — denoising will be skipped: {e}")
    else:
        print("FRCRN denoiser disabled (DENOISE_ENABLED=0) — raw mic audio goes to Whisper.")


# ── Per-connection VAD processor ──────────────────────────────────────────────

def _reset_vad_states() -> None:
    """Clear Silero's internal LSTM state — call per connection and per utterance
    so state never carries across independent audio segments."""
    try:
        _vad_model.reset_states()  # type: ignore[union-attr]
    except Exception:
        pass  # model not loaded yet, or older silero without reset_states


def make_stt_processor(on_speech_start: Any, is_ai_audible: Any):
    """
    Returns an async process_chunk(data: bytes) coroutine.
    Each call processes one 512-sample chunk.
    Returns np.ndarray (full utterance audio) when speech ends, else None.

    is_ai_audible() — while True (AI audio streaming, or still playing in the
    browser), speech onset needs MIN_SPEECH_CHUNKS_BARGE consecutive chunks
    instead of MIN_SPEECH_CHUNKS, so speaker bleed can't fake a barge-in.
    """
    preroll: deque[Any] = deque(maxlen=PREROLL_CHUNKS)  # recent audio from before onset
    speech_buffer:      list[Any] = []
    in_speech:          bool = False
    silence_chunks:     int  = 0
    speech_chunks_count: int = 0

    _reset_vad_states()

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
                onset_needed = MIN_SPEECH_CHUNKS_BARGE if is_ai_audible() else MIN_SPEECH_CHUNKS
                if speech_chunks_count >= onset_needed:
                    in_speech = True
                    # Prepend pre-roll: VAD confirms onset ~100-300 ms after speech
                    # actually starts, so without this the first syllable is clipped.
                    speech_buffer[:0] = list(preroll)
                    preroll.clear()
                    await on_speech_start()
        elif in_speech:
            speech_buffer.append(pcm)
            silence_chunks += 1
            if silence_chunks >= MAX_SILENCE_CHUNKS:
                audio = np.concatenate(speech_buffer)
                speech_buffer       = []
                in_speech           = False
                silence_chunks      = 0
                speech_chunks_count = 0
                _reset_vad_states()
                return audio
        else:
            # Idle silence or a false start — recycle the dropped chunks into
            # the pre-roll so they're still available if real speech follows.
            if speech_buffer:
                preroll.extend(speech_buffer)
                speech_buffer = []   # type: ignore[assignment]
            preroll.append(pcm)
            speech_chunks_count = 0

        return None

    return process_chunk


# ── Denoising ─────────────────────────────────────────────────────────────────

def denoise_blocking(audio: Any) -> Any:
    global _denoiser
    if _denoiser is None:
        return audio
    if len(audio) > _FRCRN_MAX_SAMPLES:
        return audio   # long clip — skip denoising, pass straight to Whisper
    if torch.cuda.is_available():
        # Flush PyTorch's reserved pool BEFORE checking so mem_get_info() reflects
        # truly available VRAM, not memory still held from the last TTS synthesis.
        torch.cuda.empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < _FRCRN_MIN_FREE_MB * 1024 * 1024:
            return audio
    try:
        result = _denoiser(audio.reshape(1, -1))  # type: ignore[call-overload]
        if isinstance(result, np.ndarray) and result.size > 0:
            return result.squeeze()
    except Exception as e:
        print(f"Denoiser error (passing audio through): {e}")
        if "out of memory" in str(e).lower():
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
    return audio


# ── Transcription ─────────────────────────────────────────────────────────────

_TRANSCRIBE_KWARGS: dict[str, Any] = dict(
    beam_size=5,   # accuracy-first: 5 is Whisper's standard. The extra ~150-250ms lands in the
                   # post-silence 'thinking' window before the LLM's ~1.5s TTFT, so it's imperceptible.
                   # Fewer proper-noun mangles (e.g. 'Indus Valley' → 'index value').
    condition_on_previous_text=False,  # each utterance is independent; cross-segment conditioning
                                       # seeds repetition/drift hallucinations on short clips.
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 300},
    word_timestamps=True,
)

# Arabic decoding bias: a plausible "previous transcript" carrying the water-utility
# domain terms and a few Najdi markers, so عداد/تسريب/انقطاع and dialect spellings
# transcribe correctly. Applied ONLY when the detected language is Arabic.
_AR_INITIAL_PROMPT = (
    "وش صار على قراءة العداد في المحطة؟ أبغى أعرف الحين ليش الضغط عالي والتدفق واطي، "
    "وهل في تسريب أو انقطاع في الخط أو الخزان، والمعدل زين ولا فيه تنبيه."
)

# Above this confidence, trust Whisper's "this sounds like Urdu/Farsi/..." guess as
# genuinely-Arabic-misheard and force it straight to Arabic (the remap's original
# purpose: real Najdi/Gulf speech does get misheard as Urdu). Below it, the guess
# is too shaky to trust blindly — forcing Arabic decoding on audio that might
# actually be English doesn't fail cleanly, it HALLUCINATES a fluent but wrong
# Arabic sentence (observed live: "tell me that in Najdi dialect" came back as
# garbled Arabic about "تسريب نجدي"). That ambiguous band gets a tiebreak instead.
_AR_REMAP_HIGH_CONFIDENCE = 0.85


def _decode_candidate(audio: Any, language: str, kwargs: dict[str, Any]) -> tuple[str, float]:
    """Force-transcribe as `language`; return (text, mean per-word confidence)."""
    segments, _info = _whisper_model.transcribe(audio, language=language, **kwargs)
    segments = list(segments)
    words: list[Any] = [w for s in segments for w in (s.words or [])]
    if not words:
        return "", 0.0
    conf = sum(float(w.probability) for w in words) / len(words)  # type: ignore[union-attr]
    text = " ".join(s.text.strip() for s in segments).strip()
    return text, conf


def transcribe_blocking(audio: Any) -> tuple[str, str]:
    # Detect language first (one cheap encoder pass), then transcribe ONCE with the
    # language fixed — EXCEPT in the ambiguous Arabic-confusable band below, where
    # a tiebreak decode settles it. The old shape transcribed fully, then
    # re-transcribed unconditionally when Whisper mistook Arabic for Urdu/Farsi —
    # doubling STT latency on exactly the dialectal-Arabic utterances that matter
    # most here, so this only pays that cost when the guess is genuinely ambiguous.
    lang, lang_prob, _all_probs = _whisper_model.detect_language(audio)

    if lang in ARABIC_SCRIPT_REMAP and lang_prob < _AR_REMAP_HIGH_CONFIDENCE:
        print(f"  whisper: ambiguous {lang} ({lang_prob:.2f}) — dual-decode tiebreak")
        ar_kwargs = dict(_TRANSCRIBE_KWARGS, initial_prompt=_AR_INITIAL_PROMPT)
        en_text, en_conf = _decode_candidate(audio, "en", _TRANSCRIBE_KWARGS)
        ar_text, ar_conf = _decode_candidate(audio, "ar", ar_kwargs)
        print(f"  whisper: tiebreak en_conf={en_conf:.2f} ar_conf={ar_conf:.2f}")
        if max(en_conf, ar_conf) < WORD_CONF_THRESHOLD:
            print(f"  → dropped: tiebreak confidence too low (en={en_conf:.2f}, ar={ar_conf:.2f})")
            return "", lang
        return (en_text, "en") if en_conf >= ar_conf else (ar_text, "ar")

    if lang in ARABIC_SCRIPT_REMAP:
        # High-confidence Arabic-family misdetection — trust it, decode straight as Arabic.
        print(f"  whisper: remapped {lang} → ar ({lang_prob:.2f}, high-confidence)")
        lang = "ar"

    threshold = LANG_PROB_THRESHOLD_AR if lang == "ar" else LANG_PROB_THRESHOLD
    print(f"  whisper: lang={lang} lang_prob={lang_prob:.2f}")
    if lang_prob < threshold:
        print(f"  → dropped: lang_prob {lang_prob:.2f} < {threshold}")
        return "", lang

    kwargs = dict(_TRANSCRIBE_KWARGS)
    if lang == "ar":
        kwargs["initial_prompt"] = _AR_INITIAL_PROMPT
    segments, _info = _whisper_model.transcribe(audio, language=lang, **kwargs)
    segments = list(segments)
    all_words: list[Any] = [w for s in segments for w in (s.words or [])]
    if all_words:
        mean_conf: float = sum(float(w.probability) for w in all_words) / len(all_words)  # type: ignore[union-attr]
        print(f"  whisper: word_conf={mean_conf:.2f}")
        if mean_conf < WORD_CONF_THRESHOLD:
            print(f"  → dropped: word_conf {mean_conf:.2f} < {WORD_CONF_THRESHOLD}")
            return "", lang
    text = " ".join(s.text.strip() for s in segments).strip()

    # Whisper sometimes mislabels English as Hindi/Turkish/Indonesian/etc.
    # "Hello" is a notorious trigger for Hindi misdetection.
    # If the detected language isn't Arabic/English but the transcript is
    # pure Latin script, it's almost certainly English — remap it.
    if lang not in ALLOWED_LANGS and lang not in ARABIC_SCRIPT_REMAP:
        if text and not re.search(r'[^\x00-\x7FÀ-ɏ -~]', text):
            print(f"  whisper: remapped {lang} → en (Latin script only)")
            lang = "en"

    return text, lang
