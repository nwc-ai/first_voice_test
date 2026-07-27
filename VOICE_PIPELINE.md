# Voice Pipeline — Full Technical Reference

Owner-level deep reference for the complete audio-in → audio-out flow. Every
setting below is the *current* value in code as of this writing — grep the
cited file if you suspect drift.

```
Browser (AudioWorklet, 512-sample Float32 @ 16 kHz mic capture)
   │  raw PCM chunks over WS binary frames
   ▼
server.py  receive_loop()
   │
   ├─ stt.py   Silero VAD (per-chunk onset/end, barge-in threshold)
   ├─ stt.py   FRCRN denoise (ClearVoice, ≤4s clips only)
   └─ stt.py   faster-whisper large-v3 (lang detect → transcribe → confidence gates)
   │  accepted utterance (text, lang) → asyncio.Queue
   ▼
server.py  respond_loop()
   │
   ├─ routing.py   injection filter, Najdi/Fusha detection, dialect override
   ├─ llm.py       build_turn() wraps text with a per-turn instruction
   ├─ llm.py       Ollama /api/chat streaming (qwen3.5:27b) → token generator
   ├─ routing.py   filter_cjk() strips CJK/Cyrillic tokens from the stream
   └─ tts_omnivoice_v1.py
        sentence-boundary flush → CATT tashkeel (Fusha only) → abbreviation
        expansion → OmniVoice zero-shot clone → MP3 (lameenc) → ws.send_bytes
   ▼
Browser: ordered decode (decodeAudioData), gapless sample-accurate playback,
         server-driven barge-in (pause on speech_start, resume on
         speech_rejected, hard-clear on accepted transcript)
```

Two concurrent coroutines per WebSocket connection (`asyncio.gather`):
`receive_loop` (mic → VAD → STT → queue) and `respond_loop` (queue → LLM →
TTS → WS). They share `cancel_event` (barge-in) and `ai_active`/`ai_speaking`
flags, not a lock — `receive_loop` never blocks waiting on `respond_loop`.

---

## 1. Client (`static/index.html`)

### 1.1 Mic capture

- `getUserMedia` with `echoCancellation: true, noiseSuppression: true, autoGainControl: true` — all three browser DSP stages are ON. This is why `stt.py`'s FRCRN denoiser is "under evaluation for removal" (`CLAUDE.md`): the browser may already be doing most of the work.
- Two separate `AudioContext`s:
  - `audioCtx` — created at `sampleRate: 16000`, mic capture only.
  - `playCtx` — created at native device rate, playback only. Decoding OmniVoice's 24 kHz MP3s inside the 16 kHz context would resample down and cut everything above 8 kHz.
- An inline `AudioWorkletProcessor` (`MicSender`) buffers samples into fixed **512-sample** frames and posts each as a `Float32Array` via `port.postMessage`. 512 samples @ 16 kHz = **32 ms per chunk** — this is the fundamental tick rate the whole VAD state machine in `stt.py` is built on (`MIN_SPEECH_CHUNKS`, `MAX_SILENCE_CHUNKS`, etc. are all counts of these 32 ms chunks).
- Chunks are only sent once `serverReady` is true (gated on the server's `{"event":"ready"}`) — otherwise raw PCM would pile up in the socket buffer while models are still loading.

### 1.2 WebSocket lifecycle

- URL: `ws(s)://<host>/ws`, `binaryType = 'arraybuffer'`.
- **Single-connection enforcement** is server-driven: if a second tab/reconnect opens `/ws`, the server closes the old socket with **code 4001** ("superseded"). The client checks `ev.code === 4001` in `onclose` and does **not** reconnect — this specifically breaks a ping-pong reconnect loop that existed when `task.cancel()` alone was used to kill stale sessions.
- **Watchdog**: every 2.5 s the client checks `Date.now() - lastMsgTime`. If no server message (including the 3 s keepalive ping) has arrived for **20 s**, it declares the connection half-dead, force-disconnects, and reconnects. 20 s (not the server's 3 s ping interval) leaves headroom for a brief SSH-tunnel/network stall during a long reply.
- **Reconnect backoff**: starts at 500 ms, doubles each attempt, capped at 10 s. Reset to 500 ms as soon as a session reaches `ready`.
- Two control messages sent *from* client *to* server (text frames, not binary): `{"event":"playback_start"}` / `{"event":"playback_done"}` — the server uses these to know when AI audio is *actually* still audible client-side (see §3.3, barge-in).

### 1.3 Playback engine

- Each incoming binary WS frame is one complete MP3 (one sentence). `queueMp3()` chains `decodeAudioData` calls through a serialized `decodeChain` promise — necessary because `decodeAudioData` on different-length MP3s can resolve **out of order**, which would scramble sentence playback order without this.
- Sample-accurate gapless scheduling: each decoded buffer is scheduled to start at `nextStartTime`, which advances by `duration - SEAM_OVERLAP` (**0.05 s**) each time — this 50 ms overlap swallows LAME encoder padding/silence between adjacent sentence MP3s so there's no audible gap or click at the seam.
- `playGen` is a generation counter bumped on every `clearAudioQueue()` — in-flight decodes check `gen !== playGen` and silently drop themselves, so a barge-in mid-decode can't schedule stale audio after the queue was cleared.
- Barge-in is **entirely server-driven** — there is no client-side energy/VAD detector. The client only reacts to `speech_start` (pause), `speech_rejected` (resume), and `transcript` (hard clear + resume). Rationale (from code comment): Silero VAD discriminates real speech from noise far better than raw client-side loudness thresholding could.

### 1.4 UI event → state mapping

| Server event | Client action |
|---|---|
| `loading` | disable connect button, show "جاري تحميل النماذج..." |
| `ready` | enable mic streaming, reset reconnect backoff, status → "يستمع..." (listening) |
| `ping` | keepalive no-op (just feeds watchdog `lastMsgTime`) |
| `speech_start` | `pausePlayback()` (suspend `playCtx`, does NOT clear queue) |
| `speech_rejected` | `resumePlayback()` — false trigger, nothing lost |
| `transcript` | `clearAudioQueue()` + `resumePlayback()` + reset transcript/response boxes — this is the only event that actually discards the old turn's audio |
| `token` | append to `response-box` text |
| binary frame | `queueMp3()` |
| `tts_end` | reset to "listening" status if nothing is still playing (otherwise the last source's `onended` does it) |

---

## 2. Server orchestration (`server.py`)

### 2.1 Startup

- `lifespan()` starts a background task (`_load_and_signal`) that: loads OmniVoice + CATT (`tts_omnivoice_v1.load_models()`), loads Silero VAD + faster-whisper + FRCRN (`stt.load_models_blocking()`), then **warms the LLM** (`llm.warm_llm()` — one throwaway generation to force qwen3.5:27b into VRAM before announcing ready, avoiding a ~4.4 s cold-load on the user's first turn).
- The FastAPI app **binds immediately** (`yield` before loading finishes) — the page loads and the WS accepts connections while models are still warming; the client just sees a `loading` event and waits on `_models_ready` (an `asyncio.Event`).
- CUDA library preloading hack at the top of the file: manually `ctypes.CDLL(..., RTLD_GLOBAL)`-loads `libnvrtc`, `libcublas` (both cu12 and cu13 variants), `libcudnn` so symbols are visible to later `.so` loads — needed because torchcodec/faster-whisper/OmniVoice each bundle their own copies and load order matters.
- `torchaudio.load` is monkey-patched to use `soundfile` instead of the default torchcodec backend, because torchcodec requires CUDA NPP libs not present on this box; soundfile reads WAV/FLAC with zero GPU dependency.

### 2.2 Single-connection enforcement

Module-level globals `_active_ws_task` / `_active_ws_ref` track the one live session. A new connection sends the old raw socket a `close(code=4001, reason="superseded")` **before** cancelling its task — ordering matters, since cancelling first can trigger the old client's reconnect logic before it ever sees the 4001.

### 2.3 `_LockedWS` wrapper

All sends (`send_json`, `send_bytes`) go through one `asyncio.Lock`. Reason: `receive_loop`, `respond_loop`, the TTS module, and the 3 s keepalive ping loop all write to the same socket from different coroutines, and Starlette does not allow concurrent sends — unserialized writes corrupt the frame stream and the browser drops the connection.

### 2.4 Keepalive

A dedicated task pings every **3 s** from the moment the socket opens (before model loading even completes) — the 20 s client watchdog needs a heartbeat during the cold-start window, which can take minutes.

### 2.5 `receive_loop()` — accept/reject pipeline

For every VAD-completed utterance (`stt.process_chunk` returns non-`None`), in order:

1. `stt.denoise_blocking` (FRCRN, timed)
2. `stt.transcribe_blocking` (Whisper, timed) — wrapped in a try/except for CUDA OOM: on OOM it does `gc.collect()` + `torch.cuda.empty_cache()` and rejects the utterance rather than crashing the connection.
3. **Reject if:**
   - `text` is empty (STT returned nothing / below confidence gates — see §3)
   - `routing.is_mixed(text)` → force `lang = "mixed"` (code-switch, always allowed)
   - else `lang not in {"ar","en"}` → reject
   - `len(text) < 3` or `> 500` chars → reject
   - `lang == "en"` and fewer than 2 words → reject (single-word English fragments like "Okay." burn a full LLM+TTS turn on nothing)
   - `routing.REPETITION_RE` matches (ASR stuck-loop, e.g. "ا ا ا ا") → reject
4. **On accept:**
   - if `ai_speaking` (audio actually playing) → `cancel_event.set()` — this is the **true barge-in** trigger, distinct from the earlier client-side `speech_start` pause.
   - if `ai_active` (LLM/TTS running at all, thinking or speaking) → drain `utterance_queue` so only the **latest** utterance survives (last-one-wins semantics if the user speaks again before the previous turn is even dequeued).
   - push `(text, lang, stt_ms, denoise_ms)` onto `utterance_queue`.

Every reject path calls back into a `_reject(reason)` closure that sends `{"event":"speech_rejected"}` — this is what tells the client it's safe to resume any paused playback.

On loop exit (any reason, including disconnect): `cancel_event.set()` (kill any in-flight turn) then push a `None` sentinel to unblock `respond_loop`.

### 2.6 `respond_loop()` — LLM + TTS turn

1. Dequeue; `None` → exit.
2. **Injection check first** (`routing.INJECTION_RE`): if matched, log + send the transcript event, then speak a short **hardcoded refusal** ("عذراً، ما أقدر أنفذ هذا الطلب." / "Sorry, I can't act on that request.") via a single-token generator (`_single_token`) straight through the TTS module — never reaches the LLM.
3. Otherwise: send `{"event":"transcript"}`, call `llm.build_turn(text, lang)` → `(turn_content, tts_language)`.
4. Assemble the full `/api/chat` message list: `[system] + history + [{"role":"user","content":turn_content}]`. **Only the raw user message and the raw LLM output ever get committed to `history`** — the per-turn wrapper (language instruction, style rules, Najdi glossary) is never persisted, so those instructions can't accumulate turn over turn.
5. Stream tokens through `routing.filter_cjk()` (strips CJK/Cyrillic/fullwidth-punctuation code points from the LLM's raw output — a defensive filter against a known qwen3.5 failure mode of occasionally emitting stray CJK glyphs) into `tts_omnivoice_v1.stream_tts_to_ws()`.
6. **Empty-response fallback**: if the LLM produced no text at all (e.g. a thinking-only response slipped through) and the turn wasn't cancelled, speak a fixed fallback line ("I didn't catch that. Could you please repeat?" / "عذراً، لم أفهم. ممكن تعيد؟"). This fallback is explicitly **not** stored in history and is skipped entirely if barge-in fired (so the user doesn't hear "I didn't catch that" bleeding over their next utterance).
7. **History trimming**: after a successful (non-cancelled) turn, append the user+assistant pair, then trim from the front once `len(history) >= MAX_HISTORY_TURNS * 2` (`MAX_HISTORY_TURNS = 3`, so max 6 stored messages = 3 user+assistant pairs).
8. **Timing captured per turn**: `llm_ttft_ms` (time to first LLM token), `tts_first_ms` (time to first audio byte reaching the client), `llm_total_ms` (full turn), and a synthesized `e2e_ms` = `MAX_SILENCE_CHUNKS * 32ms` (VAD tail the user actually waited through) `+ denoise_ms + stt_ms + llm_total_ms` — i.e. everything the user experiences as "silence after I stopped talking" through "audio starts."
9. On any exception/cancellation, `finally` always resets `ai_active`/`ai_speaking` to `False` and calls `torch.cuda.empty_cache()` — releases OmniVoice's reserved-but-unallocated VRAM scratch space back to the allocator pool so the *next* utterance's denoiser/Whisper pass has room (model weights themselves stay pinned in VRAM; this only frees allocator slack).

### 2.7 Logging

Every completed turn is appended as one JSON line to `logs/interactions.jsonl` (gitignored — **never commit this**, `nwc-voice`/`first_voice_test` are public repos) via `_write_log()`. Fields: `ts`, `model`, `lang`, `transcript`, `response`, and the `latency` block above. `/logs` returns the last 200 entries as JSON; `/review` serves a dashboard (`static/review.html`) that reads `/logs`.

### 2.8 Uvicorn transport settings

`ws_ping_interval=30.0`, `ws_ping_timeout=120.0` — the protocol-layer ping is deliberately loose (the app already runs its own 3 s ping + the client's 20 s watchdog); a tight default (20 s/20 s) would auto-close a connection during a heavy turn if the event loop or an SSH tunnel stalls briefly.

---

## 3. STT — `stt.py`

### 3.1 Silero VAD state machine

Runs **once per 32 ms chunk** (512 samples @ 16 kHz), per connection, via a closure (`make_stt_processor`) holding `preroll`, `speech_buffer`, `in_speech`, `silence_chunks`, `speech_chunks_count`.

| Constant | Value | Meaning |
|---|---|---|
| `speech_prob >= 0.5` | — | Silero's own speech/non-speech decision threshold per chunk |
| `MIN_SPEECH_CHUNKS` | 4 (≈128 ms) | consecutive speech chunks needed to confirm onset **when the AI is silent** |
| `MIN_SPEECH_CHUNKS_BARGE` | 3 (≈96 ms) | consecutive speech chunks needed to confirm onset **while AI audio is audible** (barge-in). Lowered progressively (9→5→3) so interrupting the AI feels near-instant. **Assumes headphones** — on open speakers, the AI's own voice bleeding into the mic can self-interrupt at this low a threshold. |
| `MAX_SILENCE_CHUNKS` | 25 (≈0.8 s) | trailing silence to close an utterance. Comment notes: if users start getting cut off mid-sentence, raise back toward 40; the pre-roll + stricter onset made the old 1.28 s tail unnecessary. |
| `PREROLL_CHUNKS` | 10 (≈320 ms) | audio retained from *before* confirmed onset and prepended once onset fires — Silero confirms onset ~100–300 ms after speech actually starts, so without this the first syllable is clipped. |

`is_ai_audible()` passed into the processor = `ai_speaking OR client_playing` (server + client-reported playback state combined) — this decides which onset threshold (barge-in vs normal) applies each chunk.

VAD LSTM state (`_vad_model.reset_states()`) is reset per connection **and** per completed utterance — state must never leak across independent audio segments.

On idle/false-start chunks (not in speech), dropped `speech_buffer` content is recycled into `preroll` rather than discarded, so a real utterance immediately following a false start still gets its pre-roll.

### 3.2 FRCRN denoising

- Model: ClearVoice `FRCRN_SE_16K`, gated by env var `DENOISE_ENABLED` (default **on**, `1`). Explicitly called out in code as "under evaluation for removal" — the browser's own `noiseSuppression`/`echoCancellation` plus Whisper large-v3's inherent noise robustness make the benefit unproven.
- `_FRCRN_MAX_SAMPLES = 4 s` — clips longer than this **skip denoising entirely** (FRCRN VRAM scales with clip length and longer clips OOM on this GPU with qwen3.5:27b + OmniVoice already resident). Whisper handles long clips fine without denoising.
- `_FRCRN_MIN_FREE_MB = 150` — before denoising, flushes the CUDA cache and checks free VRAM via `torch.cuda.mem_get_info()`; if under 150 MB free, skips denoising rather than risk an OOM.
- Any denoiser exception (including OOM) falls back to passing the raw audio through unmodified — denoising is treated as best-effort, never a hard dependency.

### 3.3 faster-whisper transcription

- Model: `large-v3`, `device="cuda"`, `compute_type="int8_float16"` — chosen specifically to save ~1.5 GB VRAM vs plain float16 with "negligible accuracy impact," freeing headroom for OmniVoice + qwen3.5:27b sharing the same 32 GB GPU.
- Fixed decode kwargs (`_TRANSCRIBE_KWARGS`):
  - `beam_size=5` — Whisper's standard beam width, chosen deliberately over a faster/smaller beam because the extra ~150–250 ms is masked by the post-silence "thinking" window before the LLM's own ~1.5 s time-to-first-token; buys fewer proper-noun mangles (e.g. "Indus Valley" → "index value").
  - `condition_on_previous_text=False` — each utterance transcribed independently; cross-segment conditioning was found to seed repetition/drift hallucinations on short clips.
  - `vad_filter=True`, `vad_parameters={"min_silence_duration_ms": 300}` — Whisper's own internal VAD filter (separate from the Silero pass upstream) trims internal silence during decoding.
  - `word_timestamps=True` — needed to compute per-word confidence for `WORD_CONF_THRESHOLD` gating.

**Language detection & decode strategy** (`transcribe_blocking`):
1. One cheap `detect_language()` encoder pass first.
2. If the detected language is in `ARABIC_SCRIPT_REMAP` (`{ur, fa, ps, ug, prs, ckb, sd, pa}` — languages Whisper confuses with Arabic, including Arabic-script langs *and* Punjabi) **and** `lang_prob < 0.85` (`_AR_REMAP_HIGH_CONFIDENCE`): run a **dual-decode tiebreak** — force-decode once as `en`, once as `ar` (with the Arabic decode using `_AR_INITIAL_PROMPT`, a water-utility-domain Najdi-flavored priming string), compare mean per-word confidence, and take whichever is higher (or drop entirely if both are below `WORD_CONF_THRESHOLD = 0.3`). This exists because forcing Arabic decode on audio that's actually English doesn't fail cleanly — it **hallucinates a fluent but wrong Arabic sentence** (observed: "tell me that in Najdi dialect" → garbled Arabic about "تسريب نجدي").
3. If in `ARABIC_SCRIPT_REMAP` but `lang_prob >= 0.85`: trust it and force straight to `ar` (no tiebreak — the old code path re-transcribed unconditionally on every Urdu/Farsi misfire, doubling STT latency on exactly the dialectal-Arabic utterances that matter most; the confidence gate limits the expensive dual-decode to genuinely ambiguous cases only).
4. Language-probability gate: `LANG_PROB_THRESHOLD = 0.25` for non-Arabic, but only `LANG_PROB_THRESHOLD_AR = 0.10` for Arabic — Arabic legitimately misfires as Urdu/Punjabi/Farsi at low confidence, so the bar is deliberately lower to avoid over-rejecting real Arabic speech; below threshold → drop.
5. Single force-decode at the resolved language (Arabic decode gets `_AR_INITIAL_PROMPT` too, for domain-term bias — عداد/تسريب/انقطاع and Najdi spellings).
6. Word-confidence gate: `WORD_CONF_THRESHOLD = 0.3` on mean per-word probability — below this, drop the utterance even though language detection passed.
7. **Latin-script English rescue**: if the final detected language isn't `ar`/`en` and isn't in the remap set, but the transcribed text is pure Latin-range script, remap to `en` — catches Whisper's habit of mislabeling English (esp. "Hello") as Hindi/Turkish/Indonesian/etc.

`_AR_INITIAL_PROMPT` is a fixed Arabic sentence about meter readings/pressure/leaks — a "prior transcript" hint purely for domain-vocabulary decoding bias, not conversation content.

---

## 4. Routing — `routing.py`

Single source of truth for language/dialect logic, shared by `server.py` (accept/reject + LLM turn wrapping) and `tts_omnivoice_v1.py` (per-sentence CATT gate) — this module has no dependency on either, avoiding a circular import.

### 4.1 Arabic normalization (`normalize_ar`)

Strips all harakat/shadda/sukun/dagger-alif (`[ً-ْٰ]`), collapses `أ/إ/آ → ا`, `ى → ي`. Applied before every lexicon match so hamza-seating/diacritic variance in STT or LLM output doesn't cause a marker miss.

### 4.2 Najdi lexical detector (`looks_najdi`)

- `_NAJDI_MARKERS`: a curated set of **distinctly-Najdi-only** normalized words (وش, ايش, ليش, وين, عاد, الحين, بعدين, شوي, صج, زين, ابغى, يبيلك, ماله, هيه, يلا, خربان, واطي, فاضي, ممتلي, بطي, ادري, plus conjugation families added via eval: تبغى/يبغى/نبغى/تبغين, تشوف/يشوف/نشوف/شاف/شفت, تسوي/يسوي/نسوي/سويت, اللي, لسا/لسه, ماي/مويه, عشان, عيال+possessives). Words shared with MSA or pan-Arabic (رقم, ضغط, بس, خلاص, في, مرة) are **deliberately excluded** to avoid false-flagging Fusha.
- Two words are called out as known accepted-risk overlaps with MSA (عاد, زين) — single-marker false positives on rare Fusha sentences are tolerated for this heuristic.
- Documented rejects (kept as guidance, not implemented): قدر/يقدر family (collides with MSA "لا يقدر على"), راح alone (collides with MSA "went"), وسايل/يقرا (just hamza-dropped MSA spellings, not a dialect signal).
- `_NAJDI_PHRASES_RE`: multi-word phrase matches — `ما ادري`, `ما عندي`, `ما في`, `كيف الحال`.
- Tokenization uses `\w+` (Python Unicode word-char classes), **not** a naive Arabic-block regex — the naive approach (`؀-ۿ`) also matches Arabic punctuation (، ؛ ؟), which would prevent tokens like "الحين؟" from ever splitting off the marker.
- Also checks each word with a leading `ال` (definite article) stripped, when the stripped form is >4 chars — Arabic nouns constantly appear with/without the attached article as one token (الماي vs ماي) and exact-token matching alone misses that.

`looks_najdi` is used in **two independent places**: (a) `llm.build_turn` to decide the LLM reply-dialect instruction from the user's *input* text, and (b) `tts_omnivoice_v1._synthesize_mp3_blocking` to re-check the LLM's *output* text per sentence before deciding whether to run CATT — the LLM can drift into Najdi on a turn that was routed as Fusha, and the TTS gate uses the actual output as ground truth, not the input routing decision.

### 4.3 Language acceptance policy

- `ALLOWED_LANGS = {"ar", "en"}`.
- `ARABIC_SCRIPT_REMAP = {ur, fa, ps, ug, prs, ckb, sd, pa}` (see §3.3).
- `MIN_TEXT_CHARS = 3`, `MAX_TEXT_CHARS = 500`.
- `is_mixed(text)`: true if the text contains **both** an Arabic-script character (`[؀-ۿ]`) and a 2+ letter Latin word (`[a-zA-Z]{2,}`) — code-switching, always allowed regardless of `ALLOWED_LANGS`.
- `REPETITION_RE`: catches ASR stuck-loops — either a single character repeated 5+ times, or a word repeated 4+ times with whitespace between (`(.)\1{4,}` or `(\b\S+\b)(\s+\2){3,}`).

### 4.4 Explicit language/dialect override

- `WANTS_ARABIC_RE` / `WANTS_ENGLISH_RE`: match phrases like "in Arabic", "بالعربي", "reply in English", etc. — these **override the auto-detected input language** entirely, so a user can speak English and ask for an Arabic reply (or vice versa).
- `_DIALECT_PATTERNS` (checked via `requested_dialect`): explicit `najdi`/`نجدي`/`النجدية` → Najdi; `fusha`/`msa`/`modern standard`/`classical arabic`/`الفصحى` → Fusha. **Only these two dialects are recognized** — a named request for Gulf/Khaleeji/Hijazi/Egyptian/any other dialect simply doesn't match and falls through to default Fusha routing (per `CLAUDE.md`: those two dialects were explicitly removed from scope).
- A named dialect request counts as an implicit Arabic request even without the word "Arabic" appearing (e.g. "in Najdi Arabic").

### 4.5 Output filtering

- `filter_cjk` / `_UNWANTED_SCRIPT_RE`: strips CJK unified ideographs + extension A, CJK compat ideographs, CJK symbols/punctuation, katakana, hiragana, hangul, fullwidth/halfwidth forms (incl. fullwidth `？！`), and both Cyrillic blocks — applied as an async generator wrapper around the raw LLM token stream, forwarding `.aclose()` so cancelling TTS actually tears down the underlying Ollama HTTP stream.

### 4.6 Prompt injection guard

`INJECTION_RE` matches: "ignore previous/prior/all instructions", Arabic تجاهل التعليمات/الأوامر/السابق, "forget your previous/prior/all", "you are now a/an/the/my ..." (requires a role-assignment continuation — the bare phrase "you are now" alone would false-positive on innocent speech like "you are now able to see it"), نسيان التعليمات, `<system>`/`<instructions>` tags, `system:`. On match, `respond_loop` **never calls the LLM** — it speaks a hardcoded refusal directly.

### 4.7 Najdi glossary & grammar rule (embedded in Najdi-routed turns only)

- `NAJDI_GLOSSARY`: an explicit MSA→Najdi word-substitution table (وش/إيش↔ما/ماذا, ليش↔لماذا, وين↔أين, etc.) plus a list of words that are identical in both dialects (رقم, قراءة, معدل, ضغط, تدفق, خزان, عداد, محطة, ...) — appended only when `tts_language == "najdi arabic"`, so Fusha/English turns pay zero extra prompt-token cost.
- `NAJDI_GRAMMAR_RULE`: forbids the بـ + imperfective-verb pattern (بيروح، بتقول، بيصير) — found via eval to be a **Levantine/Egyptian** grammatical leak the model produces even when otherwise speaking Najdi vocabulary; instructs the plain Najdi imperfective instead (يروح، تقول، يصير).
- `NAJDI_NO_OTHER_DIALECTS_RULE`: **documented dead end, not wired in.** Targeted a second leak (Moroccan باش, Egyptian ما...ش negation) by naming the exact forbidden words/patterns in the prompt. Measured result on the held-out eval set: the leak rate went **up** (1.7% → 8.3%), a "don't think of a pink elephant" effect where naming the token increases its salience. Kept in the file only as a documented dead end — don't re-attempt this exact phrasing; if revisited, describe the *pattern* to avoid (e.g. "negate only with plain ما") rather than spelling out the literal banned words.

---

## 5. LLM — `llm.py`

### 5.1 Model & endpoint

- **Model: `qwen3.5:27b`, hard-locked** — no model selector, because a second LLM alongside it would OOM the 32 GB GPU (also hosting OmniVoice + Whisper + FRCRN).
- Ollama `/api/chat` (`OLLAMA_CHAT_URL`) is used for actual turns (carries message history natively); `/api/generate` (`OLLAMA_URL`) is used **only** for the startup warm-up ping.
- `keep_alive: -1` on every request — pins the model in VRAM indefinitely; a 27B cold-reload after idle eviction costs multiple seconds.

### 5.2 Context window

- `LLM_NUM_CTX` env var, default **8192** tokens. Chosen because Ollama's default 32768 KV cache OOM'd with OmniVoice resident in-process on the same GPU. 8192 comfortably fits system prompt + 3-turn history + reply (~2.5k tokens) with headroom.
- **Must match** between `warm_llm()` and the actual chat requests — if the warm-up loads the model at a different `num_ctx`, the first real chat request forces a reload (and while pinned via `keep_alive:-1`, risks a double-load OOM). Both paths read the same `LLM_NUM_CTX`.
- `start_server.sh` runs Ollama with `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` — these roughly halve KV-cache VRAM (near-lossless), making a larger `LLM_NUM_CTX` affordable if ever raised (e.g. `LLM_NUM_CTX=16384 bash start_server.sh`). **These flags only take effect on a fresh `ollama serve`** — if Ollama is already running, `start_server.sh` leaves it as-is; you must kill/`ollama stop` first to pick them up.

### 5.3 Generation options (`qwen3.5` config in `MODEL_CONFIGS`)

| Option | Value | Rationale |
|---|---|---|
| `think` | `False` (in `extra`, not `options`) | With thinking ON, the model spends its whole `num_predict` budget reasoning and never emits a spoken response at all (an observed empty-response bug). Voice needs direct, fast answers. |
| `temperature` | **0.5** | Lowered from an earlier 0.7 — at 0.7, factual queries fabricated details badly (invented parties/dates for a real person). Trades a little conversational flair for grounding. |
| `top_p` | 0.8 | |
| `top_k` | 20 | |
| `presence_penalty` | 1.5 | |
| `num_predict` | **300** | Hard cap on reply length — **owner-locked, known accepted tradeoff**: replies over ~170+ Arabic words can truncate mid-sentence. Kept at 300 to bound voice-reply latency/length. Do not change without checking with the owner. |
| `num_ctx` | `LLM_NUM_CTX` (8192 default) | see §5.2 |
| `stop` | `["User:", "user:", "\nUser", "\nالمستخدم:", "Human:", "\nHuman"]` | stops the model from hallucinating a continued multi-turn transcript inside one response |

A `default` config exists as a fallback (temp 0.7, top_p 0.9, top_k 40, same `num_predict`/`stop`) purely so a future model swap degrades gracefully instead of crashing `get_model_config` — not currently reachable since `MODEL` is hardcoded to `qwen3.5:27b`.

### 5.4 System prompt (`SYSTEM_PROMPT`)

Sent once per turn as the `system` message (never mutated). Key rules, in priority order:
0. **Explicit language override** — if the user asks for a specific reply language, honor it regardless of what language they spoke in; overrides everything below.
1. English input → English-only reply.
2. Najdi Arabic input → Najdi reply (explicit marker examples embedded: وش/إيش, أبغى, زين, الحين, ماله, يبيلك).
3. Code-switched input → reply in the same natural mix, matching the user's Arabic dialect.
4. Fusha or unclear-dialect input → Fusha reply ("never force a regional dialect on a Fusha speaker").
5. Never mix two Arabic dialects in one response.
6. Never use Chinese/Japanese/Korean/Cyrillic/Vietnamese or any non-Arabic/Latin script (backstopped programmatically by `routing.filter_cjk`).
7. Always full spoken sentences, never bare words/fragments — even yes/no needs to be a complete sentence (explicit good/bad examples given).
8. Proper punctuation required (commas، periods, ؟, !) — this directly feeds the TTS sentence-flush logic in §6.
9. No markdown (*, #, -, lists, headers) — this is a spoken-audio pipeline, no visual rendering.
10. Never open with filler ("Sure", "Of course", "Certainly", ...) — backstopped by `tts_omnivoice_v1._strip_openers` as a second layer in case the model slips one in anyway.
11. Never ask for clarification — always give a complete direct answer even to a broad question.
12. **Voice-specific spell-out rules**: write out abbreviations the way they're spoken aloud — full "هجري"/"ميلادي" not "هـ"/"م", "قبل الميلاد" not "ق.م", "بالمئة" not "%", "دكتور"/"أستاذ" not "د."/"أ.", "وما إلى ذلك" not "إلخ". (Backstopped by `_ABBREV_RULES` in the TTS module in case the model still emits the short form — see §6.2.)
13. Never claim to perform a real-world physical action (dispatch a truck, open a ticket, etc.) — must state it cannot do that and direct the user elsewhere.

### 5.5 Per-turn wrapping (`build_turn`)

Called once per accepted utterance; returns `(turn_content, tts_language)`. **Only this turn's wrapping is sent to the LLM — the raw `text` is what's stored in history**, so none of this accumulates across turns.

Decision order for `tts_language` (gates CATT tashkeel downstream, §6.1 — does **not** get passed to OmniVoice itself):
1. Explicit dialect request (`requested_dialect`) → `"najdi arabic"` if Najdi named, else `"standard arabic"` if any Arabic requested.
2. Explicit "reply in Arabic" without a named dialect → `"standard arabic"`.
3. Input language is `ar` → `"najdi arabic"` if `looks_najdi(text)` else `"standard arabic"`.
4. Otherwise (English or mixed) → `None` (no tashkeel).

The `lang_instruction` text embedded in the turn covers the same branches (explicit Arabic/English request, mixed code-switch, Najdi-detected, Fusha-default, English-default) with slightly different wording than the system prompt — this is deliberately redundant reinforcement, not a duplicate source of truth (the system prompt sets the *policy*, the turn instruction restates it *for this specific utterance*).

If `tts_language == "najdi arabic"`, both `NAJDI_GLOSSARY` and `NAJDI_GRAMMAR_RULE` (§4.7) are appended to the turn instruction.

Every turn also appends a fixed style/anti-hallucination block: complete sentences, no filler openers, no clarification requests, say "not sure" instead of guessing, don't invent names/dates/places/events, no markdown.

### 5.6 Token generator (`ollama_chat_token_gen`)

Streams `/api/chat` line-by-line NDJSON; each line's `message.content` is one token, yielded as soon as parsed. Fires `on_first_token` callback exactly once (used by `server.py` to timestamp `llm_ttft_ms`). Stops on `chunk.get("done")`. `httpx.AsyncClient(timeout=120)`.

### 5.7 Warm-up (`warm_llm`)

One `/api/generate` call with `prompt: "hi"`, `stream: False`, `num_predict: 1`, `keep_alive: -1`, and **the same `num_ctx` as real chat requests** (critical — see §5.2). Runs during startup, behind the client's "loading models" screen, so the first real user turn never pays the cold-load cost.

---

## 6. TTS — `tts_omnivoice_v1.py`

### 6.1 Model & voice cloning

- Engine: `k2-fsa/OmniVoice` (`_MODEL_ID`, env-overridable via `OMNIVOICE_MODEL`), loaded via `OmniVoice.from_pretrained(model_id, device_map=OMNIVOICE_DEVICE, dtype=torch.float16)`. `OMNIVOICE_DEVICE` defaults to `"cuda:0"`.
- **Zero-shot voice cloning**: needs a short reference clip + its exact transcript. Reference: `voices/silma-tts-saudi-24k.wav` (Saudi male voice) with a fixed transcript describing Saudi culture/heritage (`_REF_TEXT`).
- The `VoiceClonePrompt` (`_clone_prompt`) is built **once** at model load (`model.create_voice_clone_prompt(ref_audio, ref_text)`) and reused for every sentence — passing the raw reference WAV per sentence would make OmniVoice re-load/re-tokenize it every single time, adding needless first-audio latency.
- Output: 24 kHz float32 PCM (`SAMPLE_RATE = 24000`) — this is why the client uses a separate native-rate `playCtx` rather than the 16 kHz mic context (§1.1).
- `language` parameter is **never passed to OmniVoice** — it exists solely to gate CATT tashkeel (see below); voice generation itself is language-agnostic to this flag.

### 6.2 CATT tashkeel (diacritization)

- `CATT_ENABLED` env var, default **on** (`1`). Loaded lazily (`catt_tashkeel.CATTEncoderDecoder`), same lazy-singleton-with-lock pattern as the main model.
- Applied **only** when `language in {"standard arabic"}` (`_TASHKEEL_LANGUAGES`) **and** `not looks_najdi(sentence)` — double-gated, because the LLM can drift into Najdi vocabulary even on a turn routed as Fusha, and the per-sentence re-check uses the actual output text as ground truth rather than trusting the turn-level routing decision.
- Why gated at all: CATT is MSA-trained and **mis-vocalizes Najdi words** — the example given in comments is مرة ("very" in Najdi) coming back diacritized as the unrelated MSA noun "a time/once."
- Failure mode: any exception from CATT falls back to the plain undiacritized text rather than dropping the sentence — treated as a nice-to-have pronunciation aid, never a hard dependency.
- `CATT_ENABLED=0` reverts to plain text globally with zero code change.

### 6.3 Sentence-boundary flush logic

| Constant | Value | Meaning |
|---|---|---|
| `HARD_BREAK` | `{!, ?, ؟}` | always flush immediately regardless of buffer length |
| `SOFT_BREAK` | `{., ',', '،', ';', ':'}` | flush only if buffer length ≥ `min_len` |
| `SOFT_BREAK_MIN` | 40 chars | minimum buffer length for a soft-break flush, steady state |
| `FIRST_SOFT_MIN` | 20 chars | minimum buffer length for the **very first** flush only — deliberately lower to cut time-to-first-audio on the opening sentence |

`_should_flush(buffer, char, first)` is called per-character as tokens stream in (`_emit`). Every flushed sentence is pushed onto `sentence_queue`, consumed by a single background `synth_worker` task — **tokens keep streaming to the browser (text display) continuously while GPU synthesis happens in the background**, so the LLM/text stream is never stalled waiting on OmniVoice.

### 6.4 Filler-opener stripping (second layer, backstops system prompt rule 10)

- `_HEAD_PROBE_CHARS = 30` — the leading edge of the response is buffered (not flushed/emitted) until either 30 characters accumulate or a hard/soft-break character appears in the current token.
- `_OPENER_RE` strips a single leading filler phrase (English: "of course"/"yes, of course"/"sure"/"certainly"/"absolutely"/"definitely"/"great"; Arabic: بكل تأكيد/بالتأكيد/طبعاً/طبعا/أكيد/بالطبع) followed by punctuation, from that head buffer — applied to **both** the WS `token` display event and the audio, so the transcript box and spoken audio always match.
- If the whole response ends before the head threshold is reached (a very short reply), whatever's buffered is still run through `_strip_openers` before being emitted.

### 6.5 Abbreviation/glued-digit expansion (`_expand_abbreviations`, runs on every flushed sentence before synthesis)

Second-layer backstop for system prompt rule 12 — regex substitutions applied to each sentence right before synthesis:
- `<digit> هـ` → `<digit> هجري`
- `<digit> م` → `<digit> ميلادي`
- `ق.م` / `ق م` → `قبل الميلاد`
- `<digit>%` → `<digit> بالمئة`
- `د. ` → `دكتور `
- `أ. ` → `أستاذ `
- `إلخ` → `وما إلى ذلك`
- Arabic letter immediately glued to a digit (either order, e.g. `و2013`) → inserts a space, so digits read as separate tokens instead of mangling into the adjacent word.

### 6.6 Synthesis + encoding (`_synthesize_mp3_blocking`, one `asyncio.to_thread` dispatch per sentence)

1. Apply CATT tashkeel if gated on (§6.2).
2. `model.generate(text=text, voice_clone_prompt=_clone_prompt)` → list of float32 PCM arrays; take `[0]`.
3. Clip to `[-1, 1]`, scale to int16 PCM.
4. `lameenc.Encoder()`: `bit_rate=64` kbps, `in_sample_rate=24000`, `channels=1` (mono), `quality=7` (fastest — comment: "64 kbps speech is transparent at any quality setting," so there's no accuracy cost to using the fastest LAME preset here).
5. Returns one complete, self-contained MP3 byte string — required because the browser's `decodeAudioData` needs a full container per call, not a raw PCM stream.

### 6.7 `stream_tts_to_ws` — the public contract

Signature: `stream_tts_to_ws(token_gen, ws, cancel_event, on_first_audio=None, language=None)`.

- A single `synth_worker` background task drains `sentence_queue` (FIFO — preserves sentence order even though synthesis is off the main token-consuming path) and calls `ws.send_bytes(mp3)` per sentence, plus fires `on_first_audio()` **exactly once**, right before the very first `send_bytes` call (used by `server.py` to timestamp `tts_first_ms`/set `ai_speaking = True`).
- **Cancellation is checked at exactly 3 points**, matching the module docstring/`CLAUDE.md` contract:
  (a) top of the token-consuming loop (`if cancel_event.is_set(): break`) — stop pulling more LLM tokens;
  (b) inside `synth_worker`, before calling `_synthesize_mp3` — skip queued sentences that are now stale, but keep draining the queue so the sentinel is still reached;
  (c) inside `synth_worker`, after synthesis completes but before `send_bytes` — a barge-in that fires *during* the ~2–5 s GPU synthesis call still prevents that already-synthesized audio from being sent.
- On cancellation, the `finally` block also calls `worker.cancel()` — this is on top of the check-points above, specifically so the caller doesn't have to **wait** for an in-flight GPU synthesis call (which can take 2–5 s) to finish before `stream_tts_to_ws` returns; the underlying blocking thread keeps running to completion in the background regardless (Python threads can't be preempted), but the coroutine unblocks immediately.
- Both `token_gen.aclose()` (propagates the cancellation down into `routing.filter_cjk` → `llm.ollama_chat_token_gen` → the underlying `httpx` stream, so Ollama actually stops generating tokens server-side rather than continuing to burn GPU on a response nobody will hear) and the `sentence_queue` sentinel (`None`) are pushed in a `finally`, so cleanup happens on every exit path (normal completion, exception, or cancellation).
- `{"event":"tts_end"}` is sent **only if the turn was not cancelled** — a barge-in ends the turn silently from the client's perspective (no explicit "end" signal), since the client already reacted to `speech_start`/`transcript` instead.

---

## 7. Infra / environment (`start_server.sh`, `requirements.txt`)

- Server binds `0.0.0.0:8765` (see `server.py` `__main__`).
- `LD_LIBRARY_PATH` is hand-assembled to point at the venv-bundled NVIDIA CUDA 13 libs (`cu13`, `cublas`, `cudnn`, `cuda_nvrtc`) plus a separately-installed newer Ollama build's own `lib/ollama` — needed because system CUDA and the venv's bundled CUDA must both resolve correctly for faster-whisper/OmniVoice/torch to share the GPU.
- Ollama auto-start logic: only starts a fresh `ollama serve` if `localhost:11434/api/tags` isn't already responding; if Ollama is already running, `OLLAMA_FLASH_ATTENTION`/`OLLAMA_KV_CACHE_TYPE=q8_0` are **not** retroactively applied — you must stop the existing Ollama process first.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — reduces CUDA memory fragmentation, important on a GPU shared across Whisper + FRCRN + OmniVoice + Ollama's own VRAM allocation.
- Key pinned dependencies (`requirements.txt`): `faster-whisper==1.1.1`, `clearvoice==0.1.2`, `omnivoice==0.1.5` (note: pulls `transformers>=5.3`; verified faster-whisper + clearvoice both still run fine on transformers 5.x), `catt-tashkeel==1.0.2`, `lameenc==1.8.2`. PyTorch/torchaudio (`2.11.0+cu130`) are installed separately outside `pip install -r` to match the exact CUDA 13.0 toolkit on this machine.

---

## 8. End-to-end timeline for one turn (what the numbers in `logs/interactions.jsonl` mean)

```
user stops talking
 │
 ├─ MAX_SILENCE_CHUNKS × 32ms  (≈800ms)   VAD trailing-silence wait, part of e2e_ms
 ├─ denoise_ms                            FRCRN (or 0 if skipped/disabled)
 ├─ stt_ms                                faster-whisper detect+transcribe
 ├─ llm_ttft_ms                           time to Ollama's first streamed token
 │      (from turn start, i.e. includes any Ollama prefill of system+history+turn)
 ├─ tts_first_ms                          time to first MP3 byte reaching the browser
 │      (from turn start — includes LLM prefill + first-sentence flush wait +
 │       first OmniVoice synthesis call)
 └─ llm_total_ms / e2e_ms                 full turn / total user-perceived wait
```

`e2e_ms` in `server.py` is explicitly computed as `MAX_SILENCE_CHUNKS*32 + denoise_ms + stt_ms + (t_done - t_llm_start)*1000` — i.e. it's the full "what did the human actually wait through" number, not just an internal processing time.
