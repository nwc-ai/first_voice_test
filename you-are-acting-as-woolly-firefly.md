# Technical Review — first_voice_test (Arabic/English dialect voice assistant)

**Scope:** architecture, model quality, performance, correctness — not enterprise infra (dev-stage, single-user, local).
**Evidence:** full read of all code/docs (3,229 lines), the installed `omnivoice` package source, 49 logged turns (`logs/interactions.jsonl`), `barge_diag.log`, git history; 4 web-research agents on mid-2026 alternatives + 22 code findings each adversarially verified (21 confirmed, 1 refuted). Several claims were **measured on this machine** during review (STT/TTS timings on the idle 5090).

---

## Context

The goal is a multilingual conversational voice assistant (English, Fusha, Najdi, Hijazi, Egyptian, AR↔EN code-switching) on one RTX 5090 (32 GB). This document is the requested critical review; the "Prioritized recommendations" section is the implementation plan if approved.

**Bottom line up front:** all three model choices (whisper large-v3, OmniVoice, qwen3.5:27b) survive comparison against mid-2026 alternatives — none should be swapped today. The real problems are: an **OmniVoice weights license the docs get wrong (CC-BY-NC, non-commercial)**, a confirmed reply-truncation bug (`num_predict=300`), the open barge-in/false-positive bug killing reply audio, ~1.3 s of recoverable latency, several dialect-routing false positives, and **no eval harness** — which makes every quality claim unfalsifiable.

---

## 1. Overall architecture

### How it works
Browser mic (16 kHz AudioWorklet, 512-sample frames) → WebSocket → two concurrent asyncio loops per connection: `receive_loop` (Silero VAD with 320 ms pre-roll + 800 ms silence tail → FRCRN denoise for ≤4 s clips → faster-whisper large-v3 with two-pass language logic + dialect hotwords → confidence/length/repetition gates) and `respond_loop` (lexical dialect routing → committed per-turn instruction → qwen3.5:27b streamed from Ollama → sentence-chunked OmniVoice synthesis with per-dialect voice clip + `language=` ID → one 64 kbps MP3 per sentence). Browser decodes in a serialized chain, schedules gaplessly. Barge-in is server-VAD driven; `cancel_event` propagates `aclose()` down to the Ollama stream.

### Is the architecture right for the goal? Yes.
Research confirmed the cascade is still correct in July 2026: **no open speech-to-speech model outputs Arabic speech at all** (Moshi en-only; GLM-4-Voice/Step-Audio zh/en; Qwen3-Omni has no Arabic talker output and needs ≥79 GB). The only omni model with Arabic output (Qwen3.5-Omni, Mar 2026) is cloud-API-only and MSA-only. Nothing end-to-end can replicate this project's core feature: per-turn dialect control via committed LLM instruction + per-dialect reference clip + TTS `language=` ID.

### Major strengths
1. **Latency-conscious streaming design**: token streaming, sentence-level TTS with early first flush, background synth worker, model warm-up behind the loading screen, `keep_alive:-1` + matched `num_ctx` (prevents a double-load OOM).
2. **The dialect engine is a real committed design** — server-side detection, request parsing with proper-noun guards, per-dialect voice + pronunciation pinning. Marker-exclusion decisions (عشان, وين) show real error analysis.
3. **Concurrency discipline**: `_LockedWS`, `aclose()` propagation, 3-point TTS cancellation, socket-scoped browser handlers, 4001 supersede protocol. The cancellation story is unusually complete.
4. **Observability**: per-turn latency JSONL + `/review` dashboard — this review's measurements exist because of it.
5. **Honest docs** recording what was tried and dropped (tashkeel/CATT, client RMS barge-in, Urdu remap).

### Major weaknesses
1. **No eval harness** (the referenced "100-question test" isn't in the repo). Dialect-ID accuracy, per-dialect WER, TTS dialect fidelity: all unmeasured.
2. **~2.6–2.8 s median wait** (silence→first audio): 800 ms fixed VAD tail + 220–330 ms STT + ~576 ms TTFT + **~1.04 s median first-token→first-audio gap**. ~1.3 s is recoverable without changing any model (see §8).
3. **Code-switching is structurally handicapped in STT** — whisper forces one language token per pass; the forced-`ar` re-pass biases against intra-utterance mixing.
4. **Dialect detection is precision-tuned, recall-poor** — and several markers are misassigned (see §2.4), actively misrouting common utterances.
5. **`server.py` monolith** (1,350 lines, 6 concerns + ~150 lines of inline dashboard HTML).

---

## 2. Component-by-component

### 2.1 VAD — Silero
**Implemented:** onset 4 chunks (~128 ms) / 3 while AI audible (~96 ms); end 25 chunks (800 ms); pre-roll recycling; per-utterance state reset.
**Suitable:** yes — language-agnostic, industry standard. **The model is right; the policy around it needs work:**
- The fixed 800 ms tail is ~30% of perceived latency. 2025–26 practice (Pipecat smart-turn v3, LiveKit) is: short trigger (~200 ms) + a semantic turn classifier + long fallback. **smart-turn v3 supports Arabic** (88.6% acc, 8 MB int8 ONNX, 12 ms CPU inference, BSD-2, usable standalone). Expected saving ~500–600 ms/turn; risk ~7% premature cutoff (tune threshold; per-dialect accuracy unpublished — verify by ear).
- `speech_start` unconditionally destroys client audio (`clearAudioQueue()`), even when STT later rejects the noise. `barge_diag.log` shows this firing repeatedly with `client_playing=True` — **this is the open "reply dies mid-tail" bug**. Verified failure: after a false barge-in, if the captured audio fails STT gates, no new turn starts, no `tts_end` is sent, and the UI hangs in "تتحدث..." with dead air.
**Verdict: keep Silero. Change the policy: pause (don't destroy) on `speech_start`, commit the kill only when STT accepts text, send a recovery event otherwise. Then adopt smart-turn v3 endpointing.**

### 2.2 Denoiser — ClearVoice FRCRN
**Implemented:** ≤4 s clips only, VRAM-gated, on top of browser `noiseSuppression:true`. Ran on only **9 of 49 logged turns** (44–56 ms each); holds ~0.5 GB VRAM.
**Evidence is against it.** Three studies (2022–2026): "When De-noising Hurts" (arXiv 2512.17562) — raw audio beat enhanced in **40/40 configurations**; arXiv 2603.04710 — degradation grows with Whisper model size (large-v3 = worst case); Iwamoto et al. (2201.06685) — artifacts, not residual noise, drive the harm. FRCRN specifically: unverified, hence A/B.
**Verdict: drop, pending a cheap A/B** (replay logged short utterances with/without; keep raw unless FRCRN measurably wins). Frees ~0.5 GB VRAM, removes a serial GPU step, unifies the ≤4 s / >4 s signal chain, deletes an OOM-recovery path.

### 2.3 STT — faster-whisper large-v3 (int8_float16)
**Implemented:** utterance-level, beam 5, word timestamps, two-pass language logic, Arabic hotwords on forced-`ar` passes, confidence/length/repetition gates.
**Reality check from published benchmarks** (Open Universal Arabic ASR Leaderboard, Interspeech 2025): whisper-large-v3 is the **best open general Arabic+English model** (29.87% avg WER across 4 Arabic test sets) — but per-dialect SADA WER is sobering: **MSA 27.95 / Najdi 48.58 / Hijazi 49.99 / Egyptian 59.28**. Dialect WER ≈ 2× MSA. Your unmeasured-dialect-accuracy risk is real and quantified.
**Strengths:** the two-pass LID design is genuinely good engineering; hotwords are a cheap dialect lever; gates keep junk from the LLM.
**Confirmed defects:** (a) the distribution-branch re-decode is gated by the **first-pass** language probability, not re-decode quality — a clean forced-en transcription of accented English gets silently dropped when P(en)=0.20 < 0.25 (server.py:736); (b) silent discards: 500-char cap kills long questions after paying full STT cost, `MIN_TEXT_CHARS=3` kills «لا», the repetition regex kills emphatic «لا لا لا لا» and «ههههه» — all with zero UI feedback; (c) the beam-5 comment's justification is factually wrong (STT is serial, not hidden), though measured cost is only ~46 ms; `word_timestamps` costs ~9 ms and duplicates `avg_logprob`; (d) unreachable dead code at server.py:755-758.
**Verdict: keep the engine (see §3 — do NOT switch to large-v3-turbo: it's measurably worse on Arabic). Fix the gates and silent-drop behavior. The evidence-backed accuracy path is dialect fine-tuning, not an engine swap.**

### 2.4 Language / dialect detection & routing
**Implemented:** whisper LID → `_requested_dialect` (explicit requests) → `_detect_dialect` (marker counting) → committed routing; Fusha default.
**Fusha-default policy: right call.** Mechanism: right for stage. But verified false positives, by execution:
- **«أيوه» is in `_HIJAZI_MARKERS` — it's the canonical Egyptian "yes."** An Egyptian saying «أيوه يا فندم» is routed Hijazi: Saudi voice + Hijazi pronunciation + Hijazi wording. «تمام» is pan-dialect; «هلا» is quintessentially Najdi/Gulf — «هلا والله، وش الأخبار» ties Najdi/Hijazi 1-1 → falls to Fusha, losing a clearly Najdi utterance. Conversely «مش/ده/دي» (Egyptian set) appear in urban Hijazi speech → Egyptian voice for a Hijazi speaker.
- **`_WANTS_ARABIC_RE` false positives (server.py:1109):** "How do you say good morning **in Arabic**?" from an English learner → reply forced fully Arabic ("Do NOT reply in English"). **Negation-blind:** «لا ترد باللهجة المصرية، رد بالفصحى» matches the Egyptian pattern first-match-wins → replies in Egyptian, exactly what the user forbade.
**Verdict: keep the marker approach; fix the marker sets (remove تمام/هلا/أيوه from Hijazi; demote مش/ده/دي to tiebreaker weight), add negation/verb-context guards to request parsing. Then measure. An ML dialect-ID model (MARBERT/NADI-class, CPU-viable) is the recall upgrade — only after the eval set exists to prove it.**

### 2.5 LLM — qwen3.5:27b via Ollama
**Identified:** Qwen3.5-27B (Alibaba, 2026-02-24, Apache-2.0), hybrid linear-attention, 262K native context, multimodal; Ollama tag = q4_K_M (17 GB file, ~15.7 GB serving). Strong English (MMLU-Pro 86.1); multilingual MMLU 85.9. **No Arabic-dialect benchmark of this model exists anywhere** — it's on no Arabic leaderboard; its Najdi/Hijazi/Egyptian generation quality is publicly unmeasured.
**Literature warning that applies directly:** AL-QASIDA (2412.04193) and the AMIYA 2026 shared task both find that **all** general LLMs drift to MSA when asked to generate dialect — understanding ≫ generation; no open model does faithful Najdi/Egyptian zero-shot. Your prompt-level steering sits exactly on the documented weak spot.
**Confirmed bug:** `num_predict: 300` with its own comment saying 300 truncates (~170 Arabic words ≈ >300 tokens). The dangling fragment IS spoken (tts module flushes the un-terminated buffer at stream end) — users hear replies stop mid-word. Also: `presence_penalty 1.5` is aggressive for Arabic function words (A/B), 3-turn history is tight but fine for now.
**Verdict: keep (see §5); raise num_predict now; A/B Fanar-2-27B for dialect faithfulness.**

### 2.6 Prompting
**Implemented:** 13-rule system prompt + per-turn wrapper (clean text stored in history — the right pattern), dialect-specific abstention phrasing.
**Issue:** rules 7+11 ("complete direct answers", "never ask clarification") push long answers straight into the `num_predict=300` wall — the prompt and the sampling config fight each other.
**Verdict: keep structure; align length budget (raise num_predict, and/or add a brevity rule for voice); measure dialect fidelity once the harness exists.**

### 2.7 Conversation memory
Rolling 3-pair window, cancelled turns correctly excluded. **Keep.** (Note: session history dies on every reconnect, and the SSH tunnel makes reconnects frequent — accept for now.)

### 2.8 TTS — OmniVoice (k2-fsa)
**Identified:** 0.6–0.8B discrete non-AR diffusion-LM (Qwen3-0.6B backbone), 581k hours / 646 languages, released 2026-03-31. The dialect IDs the project uses are real, documented conditioning targets — training hours: **MSA 1483 h, Najdi 203 h, Gulf 98 h, Hijazi 22 h, Egyptian 23 h**. So Hijazi/Egyptian are the thin dialects — matches the v1→v2→v3 Egyptian clip iteration history. Best published Arabic intelligibility of any open TTS (Arabic WER 1.39% on MiniMax-24; beats ElevenLabs' 10.95% avg across languages).
**Three verified, actionable facts:**
1. **LICENSE: weights are CC-BY-NC** (HF README: code Apache-2.0, *model* CC-BY-NC due to Emilia training data). CLAUDE.md's Apache assumption is wrong. If this assistant is or becomes commercial (client demos count as commercial use in most readings), this must be resolved. MIT-licensed fallback exists (Chatterbox v3) at the cost of dialect pinning.
2. **The project runs the slow inference setting.** `generate()` defaults to `num_step=32`; the documented fast mode `num_step=16` roughly **halves synthesis RTF** (paper Table 8: 0.0598→0.0319). Expected ~1 s → ~0.5 s per sentence. One kwarg + an ear check.
3. **No streaming, ever** — open unanswered GitHub issue #77; the paper states no acceleration technique exists for this architecture. Sentence-chunking (what you do) is the correct usage; a community "streaming" fork is just chunking.
Minor: per-sentence voice-clone prompt rebuild measured at **only ~16 ms/sentence** (hypothesized big cost did not materialize) — still a free two-line fix via `create_voice_clone_prompt()`. First-token→first-audio gap (~1.04 s median) is dominated by head-probe (30 chars) + first-flush gating (punctuation + ≥20 chars) + whole-first-sentence synthesis (~408 ms measured idle, more under LLM contention).
**Verdict: keep the engine — nothing verifiably beats its Najdi-vs-Hijazi-vs-Egyptian pinning (§4). Apply num_step=16, tune first-flush, resolve the license question.**

### 2.9 Audio streaming & playback (browser)
Serialized decode chain, generation counter, gapless scheduling, playback-state reporting, socket-scoped handlers. **Measured non-issues:** MP3 encode ~6 ms/sentence; synthesis worker outruns playback >10× (RTF ~0.09); switching to PCM/Opus would buy nothing. **Keep as-is** except the `speech_start` destroy-vs-pause policy (§2.1).

### 2.10 Orchestration
Two-loop producer/consumer + queue + sentinel: correct. Confirmed defects worth fixing opportunistically: barge-in **orphans the in-flight GPU synthesis** — the to_thread keeps running `generate()` while the next turn's STT/synthesis start, unserialized, on the same model object (add a `threading.Lock` around `model.generate()`); empty-LLM fallback races a queued utterance and can overwrite `t_first_audio` after `t_done` (corrupts `tts_first_ms` logs); `receive_loop`'s catch-all + `"disconnect" in str(e)` matching kills the session (and 3-turn history) on any transient non-OOM error; two fire-and-forget `create_task` calls hold no reference (GC risk, documented asyncio footgun); per-turn `empty_cache()` runs on the event loop (measured cheap here thanks to `expandable_segments`, but trivially moved off-loop). **Keep the architecture; fix these in one small pass.**

---

## 3. STT recommendation

**Keep faster-whisper large-v3.** No open ≤8 GB model beats it as a single en+ar+dialect engine (July 2026). Explicitly do **not** move to large-v3-turbo (33.30% vs 29.87% avg Arabic WER — worse). Ruled out by research: Voxtral (Arabic >45% WER, unusable), Kyutai (no Arabic), Canary/Parakeet (no Arabic).

| Candidate | Type | EN | MSA | Najdi/Hijazi | Egyptian | Code-switch | Streaming | VRAM | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **whisper large-v3 (current)** | local | strong | 16–28% WER | ~49% WER | ~49–59% WER | weak (1 lang token/pass) | no | ~3 GB | **baseline — keep** |
| **#1 Nemotron-3.5-ASR-Streaming-0.6B** (NVIDIA, Jun 2026) | local | strong | ar "transcription-ready", 12–13% WER (NVIDIA's own eval, likely MSA-heavy) | **unverified** | **unverified** | segment-level auto-LID (replaces the two-pass hack) | **yes, 80 ms–1.12 s chunks** | ~1.5–2 GB | A/B on your eval set; wins on latency + LID if dialect WER holds within ~5 pts |
| **#2 Fine-tuned whisper large-v3** (LoRA on SADA ~600 h Saudi + MGB-3 Egyptian + ArzEn CS → CTranslate2) | local | unchanged | unchanged | **evidence: dialect FT worth 15–30 WER pts** (ArTST: Saudi 27.3, Egy 33.2 after FT) | ✔ | ArzEn-ST recipe exists | no | same 3 GB | **best accuracy path, zero pipeline change** |
| **#3 ElevenLabs Scribe v2 Realtime** (cloud) | cloud | best | best | **best measured Saudi CS** | best (13.2% WER on Egyptian+Saudi AR-EN code-switch benchmark, best of 5 providers) | **best** | yes, ~150 ms | none | reference/cloud option only — audio leaves the machine; $0.39/hr |

**Order of operations:** (1) build the eval set first — without it every alternative is unfalsifiable; (2) A/B Nemotron for latency+LID; (3) LoRA-fine-tune large-v3 for dialect accuracy. Fallback latency fix preserving current quality: SimulStreaming/WhisperLiveKit wrappers around the same weights.

## 4. TTS recommendation

**Keep OmniVoice.** It is the only engine with verified per-dialect conditioning at Najdi/Hijazi/Egyptian granularity plus the best published open-model Arabic intelligibility. Every alternative offers at best country-level Arabic or clip-only accent control — the exact mechanism that caused the Saudi/Egyptian word-mixing that `language=` fixed.

| Candidate | Dialect pinning | Arabic quality | English | Code-switch | Cloning | Streaming/latency | License | Verdict |
|---|---|---|---|---|---|---|---|---|
| **OmniVoice (current)** | ✔ ars/acw/arz/arb IDs | best published (AR WER 1.39%) | good (646-lang model) | unverified — test by ear | ✔ 8 s clips | no streaming; **num_step=16 halves RTF** | code Apache / **weights CC-BY-NC** | **keep** |
| **#1 Habibi-TTS** (F5-based, Jan 2026) | ✔ explicit Najdi/Hijazi/Gulf/Egy tags | ≈ElevenLabs parity, 3× OmniVoice's Egyptian hours | not evaluated | **explicitly unsupported** (stated in paper) | ✔ | no | SAU/unified CC-BY-NC-SA; **EGY/MSA Apache-2.0** | A/B its **Apache EGY checkpoint** vs OmniVoice arz for Egyptian turns |
| **#2 Chatterbox Multilingual v3** (Resemble, Jun 2026) | ✘ clip-only | Arabic supported, dialect via clip only | strong | speaker-consistent across switches (claimed) | ✔ 10 s | TTFB <300 ms; streaming fork | **MIT incl. weights** | the commercial-license fallback |
| **#3 OpenAudio S1 / Fish S2 Pro** | ✘ | Tier-2 Arabic | excellent | unverified | ✔ | **true streaming, TTFA ~100 ms** | restrictive research license | only if streaming TTFA ever becomes top priority |

Cloud (labeled): ElevenLabs has **no dialect-pinning parameter** ("Arabic (Saudi Arabia, UAE)" locale only, no Egyptian) — would regress the solved pronunciation problem; Azure ar-SA/ar-EG are fixed catalog voices, MSA-with-accent, no cloning.

**Actions instead of switching:** (1) `num_step=16` A/B by ear (biggest TTS latency lever); (2) precompute voice-clone prompts at startup (small, free); (3) **resolve the CC-BY-NC weights question** — if commercial use is on the roadmap, either obtain clearance or plan the Chatterbox/Habibi-EGY contingency; (4) lower first-flush thresholds.

## 5. LLM recommendation

**Keep qwen3.5:27b for now** — current, Apache-2.0, 15.7 GB, 576 ms TTFT, strongest English in its class. But know that its dialect generation is publicly unbenchmarked and the literature says every general model drifts to MSA under dialect instructions.

| Candidate | English | MSA | Dialect generation | Code-switch | VRAM (Q4) | Ollama | License | Verdict |
|---|---|---|---|---|---|---|---|---|
| **qwen3.5:27b (current)** | best in class (MMLU-Pro 86.1) | strong (inferred; multilingual 85.9) | **unbenchmarked** | works (user experience); unbenchmarked | 15.7 GB | ✔ | Apache-2.0 | **keep** |
| **#1 Fanar-2-27B-Instruct** (QCRI, Dec 2025; Gemma-3-27B + 120–166B Arabic tokens) | good (MMLU 78.9) — a step below | ArabicMMLU 74.7, dialectal Belebele 86.8 | **only 27B-class model with documented Gulf/Egyptian dialect training** | plausible (bilingual), unverified | ~18–19 GB (est.) | community GGUF import | Apache-2.0 | **the A/B to run** — dialect faithfulness vs English trade |
| **#2 qwen3.6:27b** (Apr 2026) | better agentic/coding, MMLU-Pro flat | same | same (no multilingual claims) | same | identical | ✔ one-line swap | Apache-2.0 | cheap try; expect no dialect change |
| **#3 Jais-2-8B-Chat** | well below 27B | strong Arabic-native | competitive among small models | unverified | ~5–6 GB | import | open | only if VRAM must be freed |

Ruled out: ALLaM-34B (weights closed — app-only in KSA), Aya Expanse 32B & Command R7B Arabic (CC-BY-NC), Jais-2-70B (VRAM). Cloud comparator: GPT-5 is the measured best at Saudi/Egyptian dialect steering (4.53/5 vs 3.41 best open).
**If Najdi drift persists across both A/Bs, the evidenced path is a small LoRA on SADA2022-style Saudi data** (Saudi-Dialect-ALLaM recipe, AMIYA findings) — not another base-model swap.

## 6. Hardware suitability

RTX 5090 32 GB is well matched. Measured residency ≈ 21.6 GB (LLM 15.7 + OmniVoice 2.4 + whisper 3 + FRCRN/VAD 0.5), **~10 GB headroom**. Existing hygiene is correct (q8_0 KV, flash attention, matched warm-up ctx, expandable segments). Alternatives all fit: Fanar-2 (+2–3 GB), Nemotron ASR (−1 to −1.5 GB vs whisper if adopted), smart-turn v3 (CPU). What does NOT fit and shouldn't be attempted: a second LLM, Jais-2-70B, Qwen3-Omni (≥79 GB). Better models than qwen3.5:27b exist (e.g., 70B-class Arabic models) but are unrealistic on one 32 GB card sharing with STT+TTS — the 27B-class is the right weight.

## 7. Code quality

Good for its stage: comments explain *why* with measurements, blocking work is off-loop, cancellation is complete. Confirmed issues (all adversarially verified):

**Correctness/UX (fix):** num_predict truncation (§2.5); false-barge-in tail-kill with no recovery (§2.1); `_WANTS_ARABIC_RE`/dialect-request false positives incl. negation (§2.4); marker misassignments (§2.4); silent STT discards (500-char/3-char/repetition — «لا» is dropped!); distribution-branch LID gate drops clean re-decodes; injection regex blocks benign speech ("you are now speaking too fast", "the solar system: …") and yields silent dead air — anchor tightly or make log-only; catch-all + substring exception matching kills sessions and history.

**Concurrency (small pass):** unserialized `model.generate()` across orphaned barge-in thread and next turn; empty-LLM fallback races queued utterance + corrupts `tts_first_ms`; unreferenced `create_task`s; `empty_cache()` on the event loop.

**Hygiene:** unreachable code at server.py:755-758 (advertises a Latin-script protection that never runs); ~150-line inline `/review` HTML → `static/`; dead model-selector UI + `?model=` param; 2 superseded Egyptian clips in git; `logs/interactions.jsonl` (real user speech transcripts) is git-tracked — gitignore it; dialect→(voice,language,instruction) mapping spread over 4 if/elif sites → one table; `word_timestamps` → `avg_logprob` gate (~9 ms); beam-5 comment factually wrong (real cost ~46 ms, measured).

**Verified non-issues (leave alone):** per-turn `empty_cache()` cost (0–6 ms with expandable_segments), MP3 encoding (~6 ms), sequential sentence synthesis (RTF 0.09, outruns playback 10×), `_LockedWS`, browser playback pipeline, single-connection design.

## 8. Prioritized recommendations

### Critical (must fix)
1. **OmniVoice license**: weights are CC-BY-NC, not Apache as documented. Decide: non-commercial OK / seek clearance / plan Chatterbox-v3 (MIT) or Habibi-EGY (Apache) contingency. Update CLAUDE.md regardless.
2. **`num_predict` 300 → ~500–600** + handle `done_reason=="length"` gracefully (stop speaking dangling fragments). One-line + guard.
3. **Build the eval harness** (the repo has none): ~1–2 h ground truth from logged turns + SADA test-clean (Najdi/Hijazi) + MGB-3 (Egyptian) + ArzEn (code-switch) slices; script WER-per-dialect + dialect-ID accuracy + a routing regression suite. Every other model decision is blind without this.
4. **Fix the false-barge-in audio kill** (the open `barge_diag` bug): pause playback on `speech_start`, destroy only when STT accepts the utterance, send a recovery/status event otherwise.

### High-value (ordered by expected payoff)
5. **TTS `num_step=16`** — ~0.5 s off every turn's first audio; A/B by ear vs 32.
6. **Semantic endpointing** (pipecat smart-turn v3, CPU, BSD-2, Arabic-capable) — ~0.5–0.6 s; fallback: speculative STT inside the existing 800 ms tail (~0.25–0.43 s, zero behavior change).
7. **First-flush tuning** — allow first flush at a word boundary ~12–15 chars without punctuation; keep later sentences at 40. Cuts into the measured ~1.04 s first-token→first-audio gap.
8. **Dialect-marker fixes** (أيوه/تمام/هلا out of Hijazi; مش/ده/دي demoted) + **request-parsing guards** (negation; verb-context for English "in Arabic"; request-prefix for English dialect names).
9. **Stop silently discarding speech**: truncate instead of dropping >500 chars; MIN 3→2 (or whitelist لا); repetition rule ≥6 or strip-don't-drop; always send feedback to the UI.
10. **Fix distribution-branch gate** (gate on winner margin or word-confidence only).
11. **FRCRN A/B → likely delete** (evidence says enhancement hurts robust ASR; it only ran on 9/49 turns anyway).
12. **A/B Fanar-2-27B vs qwen3.5:27b** on Najdi/Egyptian faithfulness, code-switching, TTFT — by ear + eval set.
13. **Concurrency pass**: lock around `model.generate()`, fallback-race fix, task references, `empty_cache` off-loop.
14. **Warm-inference Whisper + OmniVoice at startup** (symmetrical with `_warm_llm`; also builds cached clone prompts).

*Combined expected latency: ~2.6–2.8 s → ~1.3–1.6 s median without changing any model.*

### Nice-to-have
15. Split `server.py` (vad/stt/dialect/llm/routes); move `/review` HTML to static; delete dead code (unreachable remap, model selector); one dialect-routing table.
16. Precompute voice-clone prompts (~16 ms/sentence + removes disk I/O).
17. Injection filter: anchor tightly or log-only; speak a refusal instead of dead air.
18. Typed exceptions (WebSocketDisconnect, torch OOM) instead of substrings.
19. gitignore `interactions.jsonl`; remove superseded voice clips; drop `word_timestamps` for `avg_logprob`.
20. Later (post-eval): whisper dialect LoRA (15–30 WER pts per literature); Nemotron streaming-ASR A/B; MARBERT dialect-ID second opinion; Habibi-EGY vs OmniVoice-arz Egyptian A/B.

### Already good — do not change
Cascade architecture (no Arabic S2S alternative exists) · Silero VAD model · faster-whisper large-v3 engine (and do NOT move to turbo) · OmniVoice engine · qwen3.5:27b as default LLM · sentence-level MP3 streaming design · browser playback client · two-loop orchestration + cancellation · per-turn prompt-wrapper pattern · Fusha-default policy · 3-turn memory · VRAM management strategy.

---

## Verification

- **Latency:** `/review` dashboard + `logs/interactions.jsonl` medians before/after each change (stt_ms, llm_ttft_ms, tts_first_ms, e2e_ms).
- **Quality:** the §8.3 eval harness (WER per dialect, dialect-ID accuracy, routing regression cases incl. the confirmed false positives: «أيوه يا فندم», «هلا والله وش الأخبار», "how do you say X in Arabic", «لا ترد باللهجة المصرية»).
- **By ear** (matches owner's working style): num_step 16 vs 32; FRCRN on/off replay; Fanar-2 vs qwen3.5 dialect conversations; smart-turn premature-cutoff feel.
- **Live:** `test_local.py` for TTS/LLM smoke; a real mic session per dialect through the browser; `barge_diag.log` should go quiet on the false-barge fix (then delete the TEMP diagnostic).
