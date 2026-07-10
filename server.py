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
import gc
import json
import os
import re
import sys
from collections import deque
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
import tts_omnivoice_v1  # type: ignore[import-untyped]  # in-process OmniVoice TTS
import time as _time
import datetime

STATIC_DIR      = os.path.join(os.path.dirname(__file__), "static")
OLLAMA_URL      = "http://localhost:11434/api/generate"   # used only for the startup warm-up
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"       # conversation turns (carries history)
MAX_HISTORY_TURNS = 3   # rolling memory: keep only the last N user+assistant pairs per connection.
                        # Lowered 6 → 3 — less context re-sent each turn = faster qwen3.5:27b prefill,
                        # while still covering normal follow-ups ("what about X", "وش يعني؟").

# qwen3.5 context window (KV cache size). Default 8192 keeps VRAM low for the in-process
# stack; raise it (e.g. LLM_NUM_CTX=16384) as the team's prompts/reasoning grow — the
# q8_0 KV cache in start_server.sh makes a bigger context affordable. Used by BOTH the
# warm-up and the chat requests so the model loads once at this size (no reload).
LLM_NUM_CTX = int(os.environ.get("LLM_NUM_CTX", "8192"))
LOG_DIR      = os.path.join(os.path.dirname(__file__), "logs")
PERF_LOG     = os.path.join(LOG_DIR, "interactions.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)


def _write_log(entry: dict[str, Any]) -> None:
    try:
        with open(PERF_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  [log] write error: {e}")


# ── TEMP DIAGNOSTIC: trace what stops AI playback mid-reply ──────────────────
# Captures which barge-in path fires (server VAD vs client detector vs disconnect)
# and the measured noise level, to logs/barge_diag.log. REMOVE once the cause is fixed.
_BARGE_DIAG = os.path.join(LOG_DIR, "barge_diag.log")
def _diag(msg: str) -> None:
    try:
        with open(_BARGE_DIAG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat(timespec='milliseconds')} {msg}\n")
    except Exception:
        pass
# Single-connection enforcement: only one active WebSocket session at a time.
# When a new browser connects, the old session is closed with code 4001 ("superseded")
# so the old tab knows NOT to reconnect — prevents the ping-pong loop where each
# reconnect kills the just-established session, triggering another reconnect.
_active_ws_task: Optional[asyncio.Task] = None
_active_ws_ref:  Optional[Any]          = None   # raw WebSocket for the close-4001 signal

MODEL       = "qwen3.5:27b"   # the one and only LLM — warmed at startup so the first turn isn't a
                              # cold load, and pinned in VRAM (a second model alongside it would OOM).
SYSTEM_PROMPT = (
    "You are a voice assistant that supports Arabic dialects and English ONLY. "
    "ABSOLUTE RULES — never break these: "
    "0. LANGUAGE OVERRIDE (highest priority): If the user explicitly asks you to reply in a "
    "   specific language (e.g. 'in Arabic', 'in English', 'بالعربي', 'باللغة العربية'), reply in "
    "   THAT language regardless of which language they wrote their request in. This overrides rules 1-4. "
    "1. Otherwise, if the user speaks English → reply in English only. "
    "2. If the user speaks Arabic → detect their exact dialect and reply in that SAME dialect: "
    "   Najdi (نجدي): use وش/إيش, أبغى, زين, الحين, ماله, يبيلك — "
    "   Gulf/Khaleeji (خليجي): use شلونك, وايد, يبه, زين, ما أدري. "
    "3. If the user mixes Arabic and English (code-switching) → reply in the same natural mix, matching their Arabic dialect. "
    "4. If Arabic dialect is unclear → use Modern Standard Arabic. "
    "5. NEVER mix two Arabic dialects in one response. "
    "6. NEVER use Chinese, Japanese, Korean, Cyrillic, Vietnamese or any non-Arabic/Latin script. "
    "7. ALWAYS reply in complete, natural spoken sentences — never single words or bare fragments. "
    "   Even a simple yes/no must be a full conversational sentence with context. "
    "   BAD: 'نعم' or 'أيوه' or 'Yes'. "
    "   GOOD: 'أيوه، صح كلامك!' or 'إي والله، هذا صحيح.' or 'Yes, absolutely!' "
    "8. Use proper punctuation — REQUIRED for natural speech rhythm: "
    "   commas (،) for pauses, periods (.) to end sentences, "
    "   question marks (؟) for questions, exclamation marks (!) for emphasis. "
    "9. NO markdown — no *, #, -, lists, or headers. Plain flowing sentences only. "
    "10. NEVER start ANY response with filler openers like: Sure, Of course, Certainly, Absolutely, Great, Of course, Happy to help, I'd be happy to. "
    "    Jump straight into the answer. "
    "11. NEVER ask the user for clarification. NEVER say 'could you clarify' or 'which aspect'. "
    "    If the question is broad, give a complete direct answer covering the main points. "
    "12. This is a VOICE assistant — never write abbreviations or symbols; always spell out the full word "
    "the way it is spoken aloud. After a year, write the full word 'هجري' or 'ميلادي' — never the short "
    "forms 'هـ' or 'م'. Likewise, write 'قبل الميلاد' instead of 'ق.م'; write 'بالمئة' instead of '%'; "
    "write 'دكتور' instead of 'د.'; write 'أستاذ' instead of 'أ.'; and write 'وما إلى ذلك' instead of 'إلخ'."
)

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
_denoiser:      Any = None   # ClearVoice FRCRN — None if failed to load


# ── Startup: load all models ──────────────────────────────────────────────────

def _load_all_blocking():
    global _vad_model, _whisper_model, _denoiser

    print("Loading OmniVoice TTS...")
    tts_omnivoice_v1.load_models()
    print("OmniVoice TTS ready.")

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

    print("Loading FRCRN denoiser...")
    try:
        from clearvoice import ClearVoice  # type: ignore[import-untyped]
        _denoiser = ClearVoice(task="speech_enhancement", model_names=["FRCRN_SE_16K"])
        print("FRCRN denoiser ready.")
    except Exception as e:
        print(f"FRCRN denoiser failed to load — denoising will be skipped: {e}")


_models_ready = asyncio.Event()


async def _warm_llm(model: str = MODEL) -> None:
    """Force Ollama to load the default model into VRAM before the first user turn.

    keep_alive:-1 only PINS a model once loaded — it does not pre-load. Without this,
    the first /api/chat call pays the full 27B cold-load (~4.4 s in the logs). One tiny
    throwaway generation here moves that cost into startup, behind the 'loading' screen.
    """
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(OLLAMA_URL, json={
                "model":      model,
                "prompt":     "hi",
                "stream":     False,
                "keep_alive": -1,
                # MUST match the chat requests' num_ctx — otherwise warm-up loads the model
                # at the default 32k context and the first chat request forces a costly
                # reload (and, while pinned, risks a double-load OOM).
                "options":    {"num_predict": 1, "num_ctx": LLM_NUM_CTX},
            })
            resp.raise_for_status()
        print(f"LLM warmed: {model}")
    except Exception as e:
        print(f"LLM warm-up skipped ({model}): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    async def _load_and_signal():
        await asyncio.to_thread(_load_all_blocking)
        await _warm_llm()          # pin the 27B before announcing 'ready' — no cold first turn
        _models_ready.set()
        print("All models loaded — server ready.")

    asyncio.create_task(_load_and_signal())
    yield  # server binds immediately — page loads while models warm up


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    # no-store: the browser must re-fetch index.html every load. Without this it
    # serves a cached page after edits/restarts — the "dropdown stuck on Loading,
    # no GET in the server log" symptom (the cached page never contacts the server).
    return FileResponse(
        os.path.join(STATIC_DIR, "index.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


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


@app.get("/review")
async def review_page():
    """Simple HTML page for comparing model performance."""
    from fastapi.responses import HTMLResponse
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model Performance Review</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; padding: 24px; }
  h1 { font-size: 1.3rem; font-weight: 400; color: #888; margin-bottom: 16px; }
  .controls { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; align-items: center; }
  select, input { background: #1a1a1a; border: 1px solid #333; color: #ccc; padding: 6px 10px; border-radius: 6px; font-size: 0.85rem; }
  button { background: #1e3a1e; border: 1px solid #2a6a2a; color: #6dc96d; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; }
  button:hover { background: #254a25; }
  #count { color: #555; font-size: 0.8rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  th { background: #161616; color: #666; font-weight: 500; padding: 8px 10px; text-align: left; border-bottom: 1px solid #2a2a2a; cursor: pointer; white-space: nowrap; }
  th:hover { color: #aaa; }
  td { padding: 8px 10px; border-bottom: 1px solid #1a1a1a; vertical-align: top; max-width: 280px; }
  tr:hover td { background: #141414; }
  .model-cell { color: #5a9fd4; font-size: 0.75rem; word-break: break-all; }
  .lang-badge { display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 0.7rem; }
  .lang-ar { background: #1a2a1a; color: #6dc96d; }
  .lang-en { background: #1a1a2a; color: #6d9dc9; }
  .lang-mixed { background: #2a2a1a; color: #c9c96d; }
  .ms { color: #888; }
  .ms.fast { color: #4ad94a; }
  .ms.med  { color: #d9a84a; }
  .ms.slow { color: #d94a4a; }
  .text-cell { direction: auto; }
  .sort-asc::after  { content: " ▲"; }
  .sort-desc::after { content: " ▼"; }
  #no-data { text-align: center; color: #444; padding: 60px; }
</style>
</head>
<body>
<h1>Model Performance Review</h1>
<div class="controls">
  <select id="model-filter"><option value="">All models</option></select>
  <select id="lang-filter">
    <option value="">All languages</option>
    <option value="ar">Arabic</option>
    <option value="en">English</option>
    <option value="mixed">Mixed</option>
  </select>
  <input id="search" type="text" placeholder="Search transcript..." style="width:200px">
  <button onclick="loadData()">↻ Refresh</button>
  <span id="count"></span>
</div>
<table id="tbl">
  <thead>
    <tr>
      <th onclick="sortBy('ts')"          data-col="ts">Time</th>
      <th onclick="sortBy('model')"       data-col="model">Model</th>
      <th onclick="sortBy('lang')"        data-col="lang">Lang</th>
      <th class="text-cell">Transcript</th>
      <th class="text-cell">Response</th>
      <th onclick="sortBy('stt')"         data-col="stt">STT</th>
      <th onclick="sortBy('ttft')"        data-col="ttft">TTFT</th>
      <th onclick="sortBy('tts_first')"   data-col="tts_first">TTS 1st</th>
      <th onclick="sortBy('total')"       data-col="total">LLM Total</th>
      <th onclick="sortBy('e2e')"         data-col="e2e">E2E</th>
    </tr>
  </thead>
  <tbody id="tbody"></tbody>
</table>
<div id="no-data" style="display:none">No interactions logged yet.</div>

<script>
let allData = [], sortCol = 'ts', sortDir = -1;

function msClass(v) {
  if (!v && v !== 0) return 'ms';
  if (v < 800)  return 'ms fast';
  if (v < 2000) return 'ms med';
  return 'ms slow';
}
function fmt(v) { return (v != null) ? v + 'ms' : '—'; }
function shortModel(m) {
  const p = m.split('/');
  return p[p.length-1].replace(':latest','');
}
function sortBy(col) {
  if (sortCol === col) sortDir *= -1; else { sortCol = col; sortDir = -1; }
  document.querySelectorAll('th[data-col]').forEach(th => {
    th.classList.remove('sort-asc','sort-desc');
    if (th.dataset.col === col) th.classList.add(sortDir === 1 ? 'sort-asc' : 'sort-desc');
  });
  render();
}
function getVal(e, col) {
  const lat = e.latency || {};
  const m = { ts: e.ts, model: e.model, lang: e.lang, stt: lat.stt_ms, ttft: lat.llm_ttft_ms, tts_first: lat.tts_first_ms, total: lat.llm_total_ms, e2e: lat.e2e_ms };
  return m[col] ?? '';
}
function render() {
  const mf = document.getElementById('model-filter').value;
  const lf = document.getElementById('lang-filter').value;
  const sf = document.getElementById('search').value.toLowerCase();
  let rows = allData.filter(e =>
    (!mf || e.model === mf) &&
    (!lf || e.lang === lf) &&
    (!sf || (e.transcript||'').toLowerCase().includes(sf) || (e.response||'').toLowerCase().includes(sf))
  );
  rows.sort((a,b) => {
    const av = getVal(a,sortCol), bv = getVal(b,sortCol);
    return av < bv ? sortDir : av > bv ? -sortDir : 0;
  });
  document.getElementById('count').textContent = rows.length + ' entries';
  const tbody = document.getElementById('tbody');
  tbody.innerHTML = '';
  document.getElementById('no-data').style.display = rows.length ? 'none' : 'block';
  rows.forEach(e => {
    const lat = e.latency || {};
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="white-space:nowrap;color:#555">${e.ts||''}</td>
      <td class="model-cell">${shortModel(e.model||'')}</td>
      <td><span class="lang-badge lang-${e.lang||'en'}">${e.lang||''}</span></td>
      <td class="text-cell" dir="auto">${(e.transcript||'').slice(0,120)}</td>
      <td class="text-cell" dir="auto">${(e.response||'').slice(0,200)}</td>
      <td class="${msClass(lat.stt_ms)}">${fmt(lat.stt_ms)}</td>
      <td class="${msClass(lat.llm_ttft_ms)}">${fmt(lat.llm_ttft_ms)}</td>
      <td class="${msClass(lat.tts_first_ms)}">${fmt(lat.tts_first_ms)}</td>
      <td class="${msClass(lat.llm_total_ms)}">${fmt(lat.llm_total_ms)}</td>
      <td class="${msClass(lat.e2e_ms)}">${fmt(lat.e2e_ms)}</td>
    `;
    tbody.appendChild(tr);
  });
}
async function loadData() {
  const res = await fetch('/logs');
  const data = await res.json();
  allData = data.entries.reverse();
  const mf = document.getElementById('model-filter');
  const models = [...new Set(allData.map(e => e.model).filter(Boolean))];
  const cur = mf.value;
  mf.innerHTML = '<option value="">All models</option>' + models.map(m => `<option value="${m}"${m===cur?' selected':''}>${m}</option>`).join('');
  render();
}
document.getElementById('model-filter').onchange = render;
document.getElementById('lang-filter').onchange  = render;
document.getElementById('search').oninput        = render;
loadData();
</script>
</body>
</html>"""
    return HTMLResponse(html)


# ── Per-connection VAD + STT processor ───────────────────────────────────────

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


LANG_PROB_THRESHOLD    = 0.25   # discard if Whisper isn't confident about the language
LANG_PROB_THRESHOLD_AR = 0.10  # Arabic misfires as Urdu/Punjabi/Farsi — only block pure noise
WORD_CONF_THRESHOLD    = 0.3   # discard if mean per-word confidence is too low
ALLOWED_LANGS          = {"ar", "en"}

# Detects code-switching: text contains both Arabic script and Latin words.
_ARABIC_CHARS_RE = re.compile(r'[؀-ۿ]')
_LATIN_WORDS_RE  = re.compile(r'[a-zA-Z]{2,}')

def _is_mixed(text: str) -> bool:
    return bool(_ARABIC_CHARS_RE.search(text)) and bool(_LATIN_WORDS_RE.search(text))

# Explicit output-language requests ("...in Arabic", "بالعربي") — these override
# the auto-detected input language so the user can ask for a reply in any language.
_WANTS_ARABIC_RE = re.compile(
    r"\b(in|into|to)\s+arabic\b"
    r"|reply\s+in\s+arabic|answer\s+in\s+arabic|say\s+it\s+in\s+arabic"
    r"|بالعرب|بالعربي|باللغة\s+العربية|بالفصحى|باللهجة",
    re.IGNORECASE | re.UNICODE,
)
_WANTS_ENGLISH_RE = re.compile(
    r"\b(in|into|to)\s+english\b"
    r"|reply\s+in\s+english|answer\s+in\s+english|say\s+it\s+in\s+english"
    r"|بالانجليز|بالإنجليز|باللغة\s+الإنجليزية",
    re.IGNORECASE | re.UNICODE,
)

# Specific Arabic dialect requests, checked when the user asks for Arabic output.
# First match wins; falls back to Fusha/MSA when no dialect is named.
_DIALECT_PATTERNS: list[tuple[str, Any, str]] = [
    ("Najdi", re.compile(r"\bnajdi\b|نجدي|النجدية", re.IGNORECASE | re.UNICODE),
     "the Najdi dialect (use وش/إيش, أبغى, زين, الحين, ماله, يبيلك)"),
    ("Gulf", re.compile(r"\bgulf\b|\bkhal[ei]+ji\b|\bkhaleeji\b|خليجي|الخليجية", re.IGNORECASE | re.UNICODE),
     "the Gulf/Khaleeji dialect (use شلونك, وايد, يبه, زين, ما أدري)"),
    ("Fusha", re.compile(r"\bfus-?ha\b|\bmsa\b|modern\s+standard|classical\s+arabic|الفصحى|فصحى",
                         re.IGNORECASE | re.UNICODE),
     "Modern Standard Arabic (Fusha)"),
]

def _requested_dialect(text: str) -> tuple[Optional[str], Optional[str]]:
    """Return (dialect_name, descriptive_phrase) for an explicitly named Arabic dialect, or
    (None, None) for default (Fusha)."""
    for name, pattern, phrase in _DIALECT_PATTERNS:
        if pattern.search(text):
            return name, phrase
    return None, None

# Lexical Najdi detector for spoken Arabic with no explicitly named dialect — used only to
# decide whether this turn's TTS gets CATT tashkeel (Fusha-only); it does not affect what
# dialect the LLM is told to reply in.
_NAJDI_MARKERS = {"وش", "أبغى", "ابغى", "الحين", "زين", "ماله", "يبيلك", "صج", "عاد", "هيه", "أدري", "ادري"}
_AR_WORD_SPLIT_RE = re.compile(r"[^؀-ۿ]+")  # split on any run of non-Arabic-letter chars

def _looks_najdi(text: str) -> bool:
    words = {w for w in _AR_WORD_SPLIT_RE.split(text) if w}
    return bool(words & _NAJDI_MARKERS)

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
    try:
        async for token in token_gen:
            cleaned = _UNWANTED_SCRIPT_RE.sub("", token)
            if cleaned:
                yield cleaned
    finally:
        # Pass close() through to the source generator so cancelling TTS also
        # tears down the underlying httpx stream (Ollama stops generating).
        aclose = getattr(token_gen, "aclose", None)
        if aclose is not None:
            await aclose()

# Detects ASR stuck-loops: "ا ا ا ا" or "هل هل هل هل"
_REPETITION_RE = re.compile(r"(.)\1{4,}|(\b\S+\b)(\s+\2){3,}", re.UNICODE)

# Prompt injection patterns (Arabic + English + Urdu).
# "you are now" requires a role-assignment continuation (a/an/the/my) — the bare
# phrase false-positives on innocent speech like "you are now able to see it".
_INJECTION_RE = re.compile(
    r"ignore\s+(previous|prior|all)\s+instructions?"
    r"|تجاهل\s+(التعليمات|الأوامر|السابق)"
    r"|forget\s+(your\s+)?(previous|prior|all)"
    r"|you\s+are\s+now\s+(a|an|the|my)\b"
    r"|نسيان\s+التعليمات"
    r"|<\s*(system|instructions?)\s*>"
    r"|system\s*:",
    re.IGNORECASE | re.UNICODE,
)


_FRCRN_MIN_FREE_MB  = 150   # skip denoising if less than this much VRAM is free after cache flush
_FRCRN_MAX_SAMPLES  = SAMPLE_RATE * 4   # skip denoising for clips longer than 4 s —
                                         # FRCRN VRAM scales with length; longer clips
                                         # OOM on this GPU (qwen3.5:27b + OmniVoice loaded).
                                         # Whisper large-v3 handles longer clips fine without it.

def _denoise_blocking(audio: Any) -> Any:
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

    # Whisper sometimes mislabels English as Hindi/Turkish/Indonesian/etc.
    # "Hello" is a notorious trigger for Hindi misdetection.
    # If the detected language isn't Arabic/English but the transcript is
    # pure Latin script, it's almost certainly English — remap it.
    if lang not in ALLOWED_LANGS and lang not in _ARABIC_SCRIPT_REMAP:
        if text and not re.search(r'[^\x00-\x7FÀ-ɏ -~]', text):
            print(f"  whisper: remapped {lang} → en (Latin script only)")
            lang = "en"

    return text, lang


# ── Per-model configuration ───────────────────────────────────────────────────
# Keys are substrings matched against the model name (case-insensitive).
# First match wins. "default" is the fallback (kept so a future model swap
# degrades gracefully instead of crashing).
# "extra" fields are merged directly into the Ollama payload (e.g. think:False).

_STOP_SEQUENCES = ["User:", "user:", "\nUser", "\nالمستخدم:", "Human:", "\nHuman"]

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "qwen3.5": {
        # think:False — voice needs direct, fast answers. With thinking ON the
        # model spends its whole num_predict budget reasoning and never emits a
        # spoken response (empty-response bug).
        # temp lowered 0.7 → 0.5: factual queries fabricated badly at 0.7 (invented
        # parties/dates for Nawaz Sharif). Lower temp = less creative drift, more
        # grounded answers. Trades a little conversational flair for accuracy.
        "extra":   {"think": False},
        "options": {
            "temperature":      0.5,
            "top_p":            0.8,
            "top_k":            20,
            "presence_penalty": 1.5,
            "num_predict":      300,   # hard cap on reply length. Known tradeoff: very long answers
                                       # (~170+ Arabic words) can cut off mid-sentence — accepted for
                                       # now to keep voice replies bounded.
            # Context window (default 8192 via LLM_NUM_CTX). The default-32768 KV cache
            # OOM'd with OmniVoice in-process on one 32 GB GPU; 8192 fits the prompt
            # (system + 3-turn memory + reply ≈ 2.5k tokens) with room to spare. Raise via
            # the LLM_NUM_CTX env var as prompts grow (q8_0 KV cache makes it affordable).
            "num_ctx":          LLM_NUM_CTX,
            "stop":             _STOP_SEQUENCES,
        },
    },
    "default": {
        "extra":   {},
        "options": {
            "temperature": 0.7,
            "top_p":       0.9,
            "top_k":       40,
            "num_predict": 300,
            "stop":        _STOP_SEQUENCES,
        },
    },
}


def _get_model_config(model_name: str) -> dict[str, Any]:
    """Return the config for the given model name, matched by substring."""
    lower = model_name.lower()
    for key, cfg in MODEL_CONFIGS.items():
        if key != "default" and key in lower:
            print(f"  [config] matched '{key}' for model '{model_name}'")
            return cfg
    print(f"  [config] no match for '{model_name}', using default config")
    return MODEL_CONFIGS["default"]


# ── LLM token generator ───────────────────────────────────────────────────────

async def ollama_chat_token_gen(
    messages: list[dict[str, str]],         # [system, ...history..., current user]
    model: str = MODEL,
    on_first_token: Optional[Any] = None,   # callable fired once on first token
):
    """Stream a chat completion from Ollama's /api/chat (carries conversation history)."""
    cfg = _get_model_config(model)
    payload: dict[str, Any] = {
        "model":      model,
        "messages":   messages,
        "stream":     True,
        "keep_alive": -1,   # pin the model in VRAM — a 27B reload after idle costs many seconds
        "options":    cfg["options"],
        **cfg["extra"],
    }
    first = True
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", OLLAMA_CHAT_URL, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                # /api/chat streams {"message": {"role": "assistant", "content": "<tok>"}, ...}
                token = chunk.get("message", {}).get("content", "")
                if token:
                    if first and on_first_token:
                        on_first_token()
                        first = False
                    yield token
                if chunk.get("done"):
                    break


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
    active_model = MODEL
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

    async def on_speech_start():
        """Called by VAD when speech onset is confirmed."""
        # Only cancel if AI is already playing audio (true barge-in).
        # While AI is still thinking (ai_active but not ai_speaking), let the
        # LLM finish so the user actually gets a response.
        if ai_speaking or client_playing:
            _diag(f"[SERVER-VAD] speech_start fired while AI AUDIBLE "
                  f"(ai_speaking={ai_speaking}, client_playing={client_playing}) "
                  f"-> client clearAudioQueue() STOPS PLAYBACK")
        if ai_speaking:
            cancel_event.set()
        try:
            await ws.send_json({"event": "speech_start"})
        except Exception:
            pass

    process_chunk = make_stt_processor(
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
                    # 1011=server error. This is the missing piece for diagnosing auto-closes.
                    print(f"  receive_loop: websocket.disconnect code={msg.get('code')}")
                    _diag(f"[WS-DISCONNECT] code={msg.get('code')} "
                          f"(ai_active={ai_active}, ai_speaking={ai_speaking}, client_playing={client_playing})")
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
                    if ai_speaking:
                        # True barge-in: AI is actively playing audio — cancel it.
                        _diag("[SERVER-UTTERANCE] full utterance completed while ai_speaking -> cancel turn")
                        cancel_event.set()
                    if ai_active:
                        # AI is busy (thinking or speaking) — drain queue so the
                        # latest utterance wins when the current turn finishes.
                        while not utterance_queue.empty():
                            utterance_queue.get_nowait()

                    t_denoise_start = _time.monotonic()
                    audio = await asyncio.to_thread(_denoise_blocking, audio)
                    denoise_ms = int((_time.monotonic() - t_denoise_start) * 1000)
                    t_stt_start = _time.monotonic()
                    try:
                        text, lang = await asyncio.to_thread(_transcribe_blocking, audio)
                    except Exception as stt_e:
                        if "out of memory" in str(stt_e).lower():
                            print(f"STT OOM — skipping utterance, clearing CUDA cache")
                            try:
                                gc.collect()
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                            continue
                        raise
                    stt_ms = int((_time.monotonic() - t_stt_start) * 1000)

                    if not text:
                        continue
                    if _is_mixed(text):
                        lang = "mixed"
                    if lang not in ALLOWED_LANGS and lang != "mixed":
                        print(f"STT [{lang}] rejected: {text!r}")
                        continue
                    if len(text) < MIN_TEXT_CHARS or len(text) > MAX_TEXT_CHARS:
                        print(f"STT [{lang}] length-rejected ({len(text)} chars): {text!r}")
                        continue
                    if lang == "en" and len(text.split()) < 2:
                        # Single-word English fragments ("Okay.", "So.") burn a full
                        # LLM+TTS turn on nothing — drop them.
                        print(f"STT [en] fragment-rejected: {text!r}")
                        continue
                    if _REPETITION_RE.search(text):
                        print(f"STT [{lang}] repetition-rejected: {text!r}")
                        continue

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

                if _INJECTION_RE.search(text):
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

                # A named dialect (Najdi/Gulf/Fusha) counts as an Arabic
                # request on its own — even when "Arabic" isn't said, e.g.
                # "in Najdi Arabic" or "in Gulf Arabic".
                req_name, req_phrase = _requested_dialect(text)
                wants_arabic = req_name is not None or bool(_WANTS_ARABIC_RE.search(text))

                # Decide this turn's TTS language (used only to gate CATT tashkeel to Fusha —
                # see tts_omnivoice_v1.py). Kept independent of the LLM instruction text above.
                if req_name == "Najdi":
                    tts_language = "najdi arabic"
                elif req_name == "Gulf":
                    tts_language = None
                elif wants_arabic:
                    tts_language = "standard arabic"   # Fusha, explicitly named or default
                elif lang == "ar":
                    tts_language = "najdi arabic" if _looks_najdi(text) else "standard arabic"
                else:
                    tts_language = None   # English or mixed AR+EN
                print(f"  [tts-lang] {tts_language}")

                if wants_arabic:
                    dialect = req_phrase or "Modern Standard Arabic (Fusha)"
                    print(f"  [lang] explicit Arabic request → {dialect}")
                    lang_instruction = (
                        "The user EXPLICITLY asked you to reply in Arabic — honor this "
                        "regardless of the language they wrote in. Reply ONLY in Arabic, "
                        f"using {dialect}. Do NOT refuse and do NOT reply in English."
                    )
                elif _WANTS_ENGLISH_RE.search(text):
                    print("  [lang] explicit English request")
                    lang_instruction = (
                        "The user EXPLICITLY asked you to reply in English — honor this "
                        "regardless of the language they wrote in. Reply ONLY in English."
                    )
                elif lang == "mixed":
                    lang_instruction = (
                        "The user is mixing Arabic and English (code-switching). "
                        "Reply naturally in the SAME mix of Arabic and English they used. "
                        "For the Arabic parts, match their dialect (Najdi or Gulf/Khaleeji). "
                        "Do NOT force a reply into all-Arabic or all-English."
                    )
                elif lang == "ar":
                    lang_instruction = (
                        "The user spoke Arabic. Detect their exact dialect "
                        "(Najdi or Gulf/Khaleeji) "
                        "from their vocabulary and reply in that EXACT same dialect. "
                        "Do NOT use Fusha/MSA unless the dialect is completely unclear."
                    )
                else:
                    lang_instruction = "The user spoke English. Reply in English only."
                # Per-turn wrapper: lang routing + style + anti-hallucination. This wraps ONLY
                # the current user message; the clean `text` is what gets stored in history,
                # so these instructions never accumulate across turns.
                turn_content = (
                    f"{lang_instruction}\n\n"
                    "IMPORTANT: Reply in complete spoken sentences with proper punctuation. "
                    "Never reply with a single word or short fragment — always a full natural sentence. "
                    "Do NOT start with: Sure, Certainly, Of course, Absolutely, Great, Happy to help. "
                    "Do NOT ask for clarification — answer directly and completely. "
                    "If you are not certain of a fact, say you are not sure rather than guessing. "
                    "Do NOT invent names, dates, places, or events. "
                    "No markdown.\n\n"
                    f"User: {text}"
                )
                # Full message list for /api/chat: system + rolling history + this wrapped turn.
                messages = (
                    [{"role": "system", "content": SYSTEM_PROMPT}]
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
                    inner = _filter_cjk(
                        ollama_chat_token_gen(
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
                        history.append({"role": "user", "content": text})
                        history.append({"role": "assistant", "content": final_response})
                        if len(history) >= MAX_HISTORY_TURNS * 2:
                            del history[: len(history) - MAX_HISTORY_TURNS * 2]

                    if final_response:
                        # OmniVoice (unlike Silma) prints nothing during synthesis, so log
                        # the assistant's reply here for terminal visibility (also in logs/).
                        print(f"  response [{lang}]: {final_response}")
                    print("LLM/TTS done.")

                    llm_ttft_ms    = int((t_first_token  - t_llm_start) * 1000) if t_first_token  else None
                    tts_first_ms   = int((t_first_audio  - t_llm_start) * 1000) if t_first_audio  else None
                    llm_total_ms   = int((t_done         - t_llm_start) * 1000)
                    # What the user actually waits after going silent: VAD tail +
                    # denoise + STT + the whole LLM/TTS turn.
                    e2e_ms         = (MAX_SILENCE_CHUNKS * 32 + denoise_ms + stt_ms
                                      + int((t_done - t_llm_start) * 1000))

                    _write_log({
                        "ts":           datetime.datetime.now().isoformat(timespec="seconds"),
                        "model":        active_model,
                        "lang":         lang,
                        "transcript":   text,
                        "response":     "".join(response_tokens),
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
