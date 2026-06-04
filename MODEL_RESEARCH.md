# Voice Assistant Model Research — Najdi & Hijazi Arabic Dialect Support

> Research conducted: June 2026  
> Context: Real-time Arabic voice assistant on RTX 5090 (32GB VRAM)  
> Requirements: Najdi, Hijazi, Fusha MSA, English, Arabic+English code-switching

---

## TL;DR — What to Change

| Component | Current | Action | Why |
|---|---|---|---|
| **STT** | faster-whisper large-v3 | ✅ Keep | Best zero-shot option available locally |
| **LLM** | qwen3.5:9b | 🔄 Switch → **SILMA-9B-Instruct** | Purpose-built for Najdi + Hijazi + Fusha + English |
| **TTS** | Silma TTS | ✅ Keep | Already supports Najdi/Hijazi; pronunciation issues were caused by LLM bad output |

---

## 1. STT — faster-whisper large-v3

### Verdict: Keep it, but it has real limitations on Najdi/Hijazi

**Current performance on Saudi dialects:**
- No published benchmark exists specifically for Najdi or Hijazi with Whisper large-v3
- The only Saudi-specific dataset (SADA) shows pre-trained Whisper WER exceeds **100%** without fine-tuning — it hallucinates heavily on Saudi dialects
- On the Open Universal Arabic ASR Leaderboard, MMS 1B (pre-trained, no fine-tuning) scores: **42.7% WER on Hijazi**, **73.2% WER on Najdi** — indicating how hard these dialects are for any off-the-shelf model

**Best alternatives benchmarked:**

| Model | Saudi WER | Notes |
|---|---|---|
| Munsit pipeline | 35.68% | Best accuracy but complex, proprietary |
| MMS 1B (fine-tuned) | 40.9% | Best documented open-source result on SADA |
| SeamlessM4T-v2 (fine-tuned) | 38.54% | Best code-switching support |
| Whisper large-v3 (fine-tuned) | ~39–41% | Solid but not best-in-class |
| Whisper large-v3 (zero-shot) | >100% on Saudi | Hallucinates heavily |

**Key findings:**
- **No off-the-shelf model reliably transcribes Najdi/Hijazi.** Fine-tuning on Saudi dialect data is the only real fix for both Whisper and MMS.
- Fine-tuned Whisper-small matched Whisper large-v3 on Arabic dialects with 70% faster inference — smaller fine-tuned model beats larger zero-shot model.
- Whisper large-v3 handles **Arabic+English code-switching** reasonably well. SeamlessM4T-v2 is marginally better but much more complex to integrate.
- faster-whisper supports real-time streaming via chunked processing (~1–2s chunks) — current architecture is valid.

**Recommendation:**  
Keep faster-whisper large-v3 for now. It is the most practical zero-shot option for a local setup. For significant Najdi/Hijazi accuracy improvement, the path is fine-tuning Whisper on a Saudi dialect dataset (5,000–10,000 samples minimum). This is a future improvement, not a blocker today.

---

## 2. LLM — The Critical Change Needed

### Verdict: Switch to SILMA-9B-Instruct

**Root cause of current problem:**  
qwen3.5:9b with `think:False` in Ollama produces degraded Arabic grammar. The `think:False` parameter is unreliable across all Qwen3.x models in Ollama — it sometimes still generates internal `<think>` tokens that corrupt Arabic output. Additionally, Q4_K_M quantization degrades non-English language quality more than English. The combination results in broken Arabic text that TTS faithfully (and badly) pronounces.

**Model comparison for Najdi + Hijazi + Fusha + English:**

| Model | Najdi/Hijazi | Fusha | English | Code-switch | VRAM (Q4) | Status |
|---|---|---|---|---|---|---|
| **SILMA-9B-Instruct** | ✅ Native | ✅ | ✅ | ✅ | ~5.8 GB | **Recommended** |
| ALLaM-7B-Instruct | ✅ (with LoRA) | ✅ | ✅ | ✅ | ~4.5 GB | Saudi SDAIA model; needs fine-tuning for best results |
| Falcon-H1 Arabic 7B | ✅ | ✅ | ✅ | ✅ | ~4.5 GB | #1 on AraDice dialect benchmark (Jan 2026, UAE's TII) |
| Jais-2 7B | ✅ Gulf region | ✅ | ✅ | ✅ | ~4.5 GB | 17 dialects in training data |
| qwen2.5:7b | ❌ defaults to Fusha | ✅ Good | ✅ | ⚠️ | ~4.7 GB | Needs LoRA fine-tuning for Saudi dialects |
| qwen3:8b | ❌ defaults to Fusha | ✅ | ✅ | ⚠️ | ~5.0 GB | Same think:False issues as 3.5; no dialect advantage |
| qwen3.5:9b | ❌ poor grammar | ⚠️ Poor | ✅ | ⚠️ | ~6.6 GB | think:False damages Arabic quality; not recommended |

**Why SILMA-9B-Instruct:**
- Built by the same team as Silma TTS (Silma AI)
- Explicitly trained on Saudi Najdi, Modern Standard Arabic (Fusha), and English within a single unified pipeline
- Native code-switching between Arabic dialects and English
- No fine-tuning required — plug and play
- ~5.8 GB VRAM (Q4) — fits comfortably alongside Whisper + Silma TTS on RTX 5090
- Sub-300ms TTFT target, designed for voice agent use cases

**Why qwen2.5:7b is not enough (even with better prompting):**  
Research shows Qwen2.5-7B-Instruct in baseline form defaults to MSA regardless of vocabulary hints in the prompt. Achieving 84% authentic Saudi dialect output requires LoRA fine-tuning with ~5,000+ Saudi dialect training pairs — significant engineering work.

**Why NOT qwen3:8b:**  
No dialect advantage over qwen2.5:7b, same think:False reliability issues as qwen3.5, and higher VRAM consumption than alternatives with actual dialect training.

---

## 3. TTS — Silma TTS

### Verdict: Keep it — the pronunciation problem was the LLM, not TTS

**What Silma TTS supports:**
- Explicitly supports **Saudi Najdi**, **Hijazi**, **Modern Standard Arabic (Fusha)**, and **Arabic+English code-switching** in a unified pipeline
- Preserves dialect-specific phonological features: /q/ preservation, dialect verb conjugations, Gulf interrogatives
- Silma Saudi TTS v2.0 is optimized for sub-300ms TTFT — suitable for streaming voice agents

**Why pronunciation sounded bad:**  
Silma TTS reads exactly what the LLM produces. When qwen3.5:9b output grammatically broken Arabic text (due to think:False degradation), Silma TTS faithfully pronounced broken text. Fix the LLM output quality → pronunciation quality fixes itself automatically.

**Best alternatives if needed:**

| TTS System | Najdi | Hijazi | Fusha | Arabic+English | Notes |
|---|---|---|---|---|---|
| **Silma TTS** (current) | ✅ | ✅ | ✅ | ✅ | Purpose-built for Saudi; keep |
| **Habibi TTS** | ✅ Explicit | ✅ Explicit | ✅ | ✅ | Open-source, 12+ dialects, benchmarked on Najdi+Hijazi |
| XTTS v2 | ⚠️ Fine-tune | ⚠️ Fine-tune | ✅ | ⚠️ | Possible but requires dialect data |
| Kokoro TTS | ❌ | ❌ | ❌ | ❌ | Does not support Arabic at all |
| Fish Speech | ⚠️ Unclear | ⚠️ Unclear | ✅ | ⚠️ | Arabic support undocumented |

---

## 4. Code-Switching (Arabic + English)

| Component | Status | Notes |
|---|---|---|
| STT | ✅ Handles it | Whisper large-v3 transcribes mixed Arabic+English reasonably well |
| LLM (SILMA-9B) | ✅ Native | Designed for intra-sentence Arabic+English switching |
| TTS (Silma) | ✅ Native | Handles Arabic+English in one utterance naturally |

The full pipeline supports code-switching end-to-end with the proposed changes.

---

## 5. Scaling — 50 Concurrent Users

**Current single-server capacity on RTX 5090 (~24 GB available VRAM):**

| Component | VRAM | Notes |
|---|---|---|
| faster-whisper large-v3 | ~3 GB | Shared across all connections |
| SILMA-9B-Instruct (Q4) | ~5.8 GB | One instance, queued requests |
| Silma TTS | ~2–3 GB | One instance, queued requests |
| **Total** | ~12 GB | ~12 GB headroom remaining |

- **Comfortable capacity:** ~20–30 concurrent users (sequential LLM/TTS per user, parallel VAD/STT)
- **Bottleneck:** LLM inference (Ollama handles one request at a time by default; `OLLAMA_NUM_PARALLEL` can increase this at the cost of VRAM per slot)
- **For 50+ users:** Requires either a second GPU or a switch to vLLM (batched inference) for the LLM component

**Scaling path when needed:**
1. Set `OLLAMA_NUM_PARALLEL=4` — allows 4 simultaneous LLM requests (uses ~23 GB total)
2. Move to vLLM with continuous batching — handles 50+ efficiently on one RTX 5090
3. Second GPU — straightforward horizontal scale

---

## Sources

- [SILMA-9B-Instruct-v1.0 — Silma AI](https://silma.ai/arabic-llm)
- [Silma Saudi TTS model — Najdi dialect](https://silma.ai/saudi-tts-model)
- [Habibi TTS: Open-Source Unified-Dialectal Arabic Speech Synthesis](https://arxiv.org/abs/2601.13802)
- [Saudi-Dialect-ALLaM LoRA Fine-Tuning Study](https://arxiv.org/html/2508.13525v1)
- [Arabic ASR on SADA Large-Scale Saudi Corpus](https://arxiv.org/html/2508.12968v1)
- [NADI 2025: First Multidialectal Arabic Speech Processing Shared Task](https://arxiv.org/abs/2509.02038)
- [Overcoming Data Scarcity in Multi-Dialectal Arabic ASR via Whisper Fine-Tuning](https://arxiv.org/pdf/2506.02627)
- [Open Universal Arabic ASR Leaderboard](https://arxiv.org/pdf/2412.13788)
- [Falcon-H1 Arabic Model Launch — TII](https://huggingface.co/blog/tiiuae/falcon-h1-arabic/)
- [Ultimate Guide: Best Open Source LLM for Arabic 2026](https://www.siliconflow.com/articles/en/best-open-source-LLM-for-Arabic)
- [Qwen3.5 Quantization Study](https://kaitchup.substack.com/p/qwen35-quantization-similar-accuracy)
- [DialectalArabicMMLU: Benchmarking Arabic Dialect LLMs](https://arxiv.org/html/2510.27543v1)
