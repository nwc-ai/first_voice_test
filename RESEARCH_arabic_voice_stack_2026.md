# Best Local Arabic Voice Stack for an RTX 5090 (2026)

**Question:** Best local (self-hosted, no-cloud) Arabic **STT → LLM → TTS** for a single NVIDIA RTX 5090 (32 GB VRAM, CUDA 13, Blackwell), ranked by latency and naturalness/accuracy — for a real-time bilingual voice assistant with barge-in.

**Ranking criteria (in priority order):**
1. **Saudi dialect quality** — Najdi & Hijazi specifically, plus MSA (Fusha). *This is the #1 criterion.*
2. Strong English alongside Arabic.
3. **Arabic↔English code-switching** (mixed in one utterance) — first-class requirement.

> **Method & honesty note.** This report is the synthesis of a deep-research workflow (fan-out web search → source fetch → 3-vote adversarial verification → cited synthesis). Findings are tagged `high / medium / low` confidence. The single most important caveat: **almost no public benchmark directly measures Najdi or Hijazi at scale.** The strongest, most actionable evidence is for the **LLM** stage; the weakest is for **TTS**, where the only Saudi-dialect comparison found is a vendor's self-published benchmark. Treat naturalness/dialect claims that lack independent listening tests as *unverified*.

---

## TL;DR — Bottom line

| Stage | Current choice | Verdict | Recommended action |
|---|---|---|---|
| **STT** | faster-whisper large-v3 | ⚠️ **Weak on dialect** | Either **fine-tune** on Saudi dialect data, or switch to a **dialect-tuned Conformer-CTC**. Off-the-shelf Whisper is a liability for Najdi/Hijazi input. |
| **LLM** | Qwen2.5:7b | ⚠️ **MSA-default** | Move to the **ALLaM family (SDAIA)** — the only LLM explicitly built and benchmarked on Najdi + Hijazi. 7B fits unquantized; 34B fits quantized in 32 GB. |
| **TTS** | evaluating Silma | ❓ **Unproven** | A **KSA-dialect track exists** (SILMA v2 KSA), but there is **no independent MOS evidence**. Run your own A/B listening test before committing. |

**One-line recommendation:** Keep a Whisper-family STT *only if you fine-tune it* (or adopt a dialect-tuned Conformer-CTC); adopt **ALLaM** for the LLM to actually satisfy the Saudi-dialect requirement; and **do not trust any TTS dialect claim — including SILMA's — until you verify it by ear.**

---

## 1. Speech-to-Text (STT)

### Key finding: your current STT is the weakest link for Saudi dialect

`faster-whisper large-v3` uses the same weights as `openai/whisper-large-v3`. On dialectal Arabic, those weights are provably weak out-of-the-box:

- **NADI 2025 multidialectal ASR benchmark:** zero-shot Whisper-large-v3 is the official baseline at **93.9 average WER**. The winning system (Munsit) reached **35.69 WER** — a **~62% relative reduction**, achieved purely through dialect-specific fine-tuning. *(high confidence, 3–0)*
- **Open Universal Arabic ASR Leaderboard** (arXiv 2412.13788, Interspeech 2025): Whisper-large-v3 ranks only **3rd at 29.87% avg WER**; `large-v3-turbo` is *worse* at **33.30%**. *(high confidence, 3–0)*

> **Implication:** turbo trades accuracy for speed and is the wrong direction for a dialect-first pipeline. And general Whisper, unmodified, will mis-transcribe Najdi/Hijazi input frequently enough to poison the downstream LLM.

### The current open leaderboard leader: NVIDIA Conformer-CTC-large-Arabic

On the Open Universal Arabic ASR Leaderboard, **NVIDIA Conformer-CTC-large-Arabic (+ 4-gram LM)** is **#1 overall**:

| Model | Avg WER | Avg CER | Model size (VRAM) |
|---|---|---|---|
| **Conformer-CTC-large-Arabic + LM** | **25.71%** | **10.02%** | **~0.48 GB** |
| Whisper-large-v3 | 29.87% | — | ~2.88 GB |
| Whisper-large-v3-turbo | 33.30% | — | smaller |

*(high confidence, 3–0; 5 test sets: SADA, Common Voice 18.0, MASC test-clean, MASC test-noisy, MGB-2)*

The Conformer's **~0.48 GB footprint vs Whisper's ~2.88 GB** is a real advantage in a pipeline that must also hold an LLM + TTS in 32 GB.

### The catch: *every* model collapses on Saudi/Gulf dialects

The #1 model's own per-dialect numbers on the **SADA** Saudi corpus show the problem — and SADA is the **only** surfaced benchmark that breaks out Najdi/Hijazi by name:

| Dialect | WER (Conformer-CTC #1) |
|---|---|
| MSA (Fusha) | **19.23%** |
| Najdi | **36.34%** |
| Hijazi | **36.96%** |
| Khaliji / Gulf | **48.23%** |
| Egyptian | 40.97% |

*(high confidence, 3–0)* The leaderboard paper states plainly that all models "achieve their best results on MSA, but exhibit a significant decline when applied to dialects."

NADI 2025 and the Casablanca dataset cover only 8 dialects (Algerian, Egyptian, **Emirati**, Jordanian, Mauritanian, Moroccan, Palestinian, Yemeni) — **Emirati is the sole Gulf representative, and Saudi/Najdi/Hijazi are absent entirely.** So your #1 criterion is essentially **unmeasured by public leaderboards**, and the best available Saudi-dialect WER (~36%) is high.

### The proven lever: dialect-specific fine-tuning

Munsit won NADI 2025 (**35.69% avg WER / 12.21% CER** over 8 dialects) using a **Conformer + CTC** architecture and dialect-specific training — confirming that **dialect-tuned training, not a bigger general model, is the path**. *(high confidence, 3–0)* (Munsit did not test Saudi dialects; its best Gulf result was Emirati at 22.67% WER.) For code-switching specifically, **SAGE** was surfaced as a relevant code-switch-tuned approach.

### STT recommendation (ranked)

1. **Best accuracy path:** Fine-tune a **Conformer-CTC (NeMo)** on Saudi dialect + code-switched data. Smallest VRAM footprint, leaderboard-leading base, proven dialect-tuning ceiling.
2. **Lowest-effort improvement:** Fine-tune **Whisper-large-v3** on Saudi/code-switched data (the NADI result shows ~62% relative WER gains are achievable). Keeps your existing faster-whisper integration.
3. **Do not** ship `whisper-large-v3-turbo` for this use case — it is *less* accurate on Arabic, and the latency win does not justify the dialect error rate.

---

## 2. Large Language Model (LLM)

### Winner: ALLaM (SDAIA) — the only LLM built for Saudi dialects

**ALLaM** (Saudi Data & AI Authority) is the **only model family explicitly developed and benchmarked on Najdi + Hijazi**, which is exactly your #1 criterion.

- **ALLaM-34B** UI evaluation: **code-switching 4.92/5**, **MSA 4.74/5** — directly addressing two of your three hard constraints. *(high confidence)*
- **Fit on 32 GB:** ALLaM-**7B** runs **unquantized**; ALLaM-**34B** fits **quantized** (e.g. 4-bit) within 32 GB — so you can run the larger, better-dialect model on a single RTX 5090.

> **Important nuance:** base **Qwen2.5-7B** (your current choice) *and* base **ALLaM-7B** both **default to MSA** and underperform on Saudi-dialect *generation* without fine-tuning or careful prompting. The dialect advantage shows up most strongly in the **34B** ALLaM and/or with dialect-targeted prompting/fine-tuning.

### Strong second: Fanar-1-9B (QCRI)

- **8.7B params, ~17.4 GB in BF16** — fits comfortably.
- Covers **Gulf, Levantine, and Egyptian** dialects — but **does not name Najdi/Hijazi** specifically. Excellent generalist Arabic model; weaker guarantee on your exact target dialects than ALLaM.

### LLM recommendation (ranked for this project)

| Rank | Model | VRAM (RTX 5090) | Najdi/Hijazi | Code-switch | Note |
|---|---|---|---|---|---|
| 1 | **ALLaM-34B (quantized)** | fits 4-bit in 32 GB | ✅ explicit | ✅ 4.92/5 | Best dialect match; verify latency at your token budget |
| 2 | **ALLaM-7B** | unquantized | ⚠️ MSA-default | good | Lighter; pair with dialect prompting/fine-tune |
| 3 | **Fanar-1-9B** | ~17.4 GB BF16 | ❌ not by name | good | Strong Gulf/Levantine/Egyptian generalist |
| — | Qwen2.5-7B *(current)* | unquantized | ❌ MSA-default | decent | Keep only as fallback; not Saudi-tuned |

Others surfaced (Jais, AceGPT, Command-R, Gemma) are viable Arabic-capable models but were not shown to beat ALLaM on the Najdi/Hijazi-specific requirement.

> **Latency consideration:** the 34B quantized model will be slower per token than your current 7B. For a barge-in voice assistant, validate that time-to-first-token and tokens/sec stay within your interactivity budget before committing — you may prefer ALLaM-7B + dialect fine-tuning if 34B is too slow.

---

## 3. Text-to-Speech (TTS)

### Honest verdict: no independent evidence exists

This is the weakest-evidence stage of the entire report. **There is no independent MOS (naturalness) benchmark** found for any local Arabic TTS on Saudi dialects.

- The **only** Saudi (KSA) dialect comparison surfaced is **SILMA's own self-published benchmark** (SILMA TTS v2 "KSA" track vs Hamsa, Gemini, ElevenLabs). *(low confidence — vendor-published, not independently verified)*
- This establishes that **a KSA-dialect TTS track exists** — which is genuinely useful — but it **cannot prove SILMA is objectively best**, because the comparison was run and reported by the vendor.

### What this means for you

- **Do not** adopt any TTS — SILMA included — on the strength of a naturalness/dialect claim alone.
- **Run your own A/B listening test.** Synthesize the same set of (a) MSA, (b) Najdi-flavored, (c) Hijazi-flavored, and (d) **code-switched Arabic-with-embedded-English** sentences across SILMA TTS v2 (KSA), XTTS-v2/Coqui, F5-TTS, Fish-Speech, and MMS-TTS, then rate them by ear with native Saudi listeners.
- The **code-switching test is critical**: the hard requirement is that the TTS pronounce embedded English words correctly *inside* an Arabic sentence. This is exactly the failure mode that vendor MOS averages hide — test it explicitly.

### TTS recommendation

1. **Shortlist SILMA TTS v2 (KSA track)** as the leading *candidate* — it is the only one with an explicit Saudi-dialect mode — but treat its quality as **unverified pending your own test**.
2. **Benchmark it head-to-head by ear** against XTTS-v2, F5-TTS, and Fish-Speech on your own Najdi/Hijazi + code-switch sentence set. This listening test is the deliverable that actually decides the TTS, not any published number.

---

## Recommended pipeline (net synthesis)

```
                 ┌─────────────────────────────────────────────┐
  User speaks →  │ STT: Conformer-CTC (NeMo) OR Whisper-large-v3│
   (Najdi/       │      — FINE-TUNED on Saudi + code-switch data│  ← #1 fix
    Hijazi/      └─────────────────────────────────────────────┘
    mixed)                         │ text
                 ┌─────────────────▼───────────────────────────┐
                 │ LLM: ALLaM-34B (4-bit) or ALLaM-7B           │  ← swap from Qwen2.5
                 │      — only family built on Najdi + Hijazi   │
                 └─────────────────┬───────────────────────────┘
                                   │ tokens
                 ┌─────────────────▼───────────────────────────┐
                 │ TTS: SILMA v2 (KSA) — CANDIDATE only,        │  ← verify by ear
                 │      validate vs XTTS-v2/F5/Fish by listening│
                 └─────────────────────────────────────────────┘
```

**VRAM budget on 32 GB (rough):** Conformer-CTC STT ~0.5 GB + ALLaM-34B 4-bit ~18–20 GB + TTS ~2–5 GB leaves headroom; with ALLaM-7B unquantized you have ample room for larger TTS or higher STT batch.

---

## Biggest risks / what is *not* measured

1. **Najdi/Hijazi STT accuracy is essentially unmeasured** by public leaderboards. The best surfaced Saudi-dialect WER (~36%, on SADA) is high. **Your own dialect-specific eval set is mandatory** — you cannot rely on published leaderboards to predict your accuracy.
2. **ALLaM-7B vs 34B latency tradeoff** is unresolved for real-time barge-in — must be measured on the 5090.
3. **All TTS naturalness/dialect claims are unverified.** The only Saudi comparison is vendor self-published. A listening test is non-negotiable.
4. **Code-switching** is under-benchmarked across all three stages; SAGE (STT) and ALLaM's 4.92/5 (LLM) are the only concrete signals — TTS code-switch quality has no data at all.

---

## Sources

Primary evidence (high-confidence findings):

- **Open Universal Arabic ASR Leaderboard** — Wang, Alhmoud, Alqurishi, arXiv **2412.13788** (Interspeech 2025): https://arxiv.org/pdf/2412.13788 · https://arxiv.org/html/2412.13788v1
- **NADI 2025** multidialectal ASR shared task: https://nadi.dlnlp.ai/2025/
- **Munsit** (NADI 2025 ASR winner), arXiv **2508.08912**: https://arxiv.org/pdf/2508.08912 · https://arxiv.org/html/2508.08912v1
- **Casablanca** dialect dataset, arXiv **2410.04527**: https://arxiv.org/pdf/2410.04527
- NADI 2025 organizers' overview, arXiv **2509.02038**

LLM / TTS (model cards & vendor sources):
- **ALLaM** (SDAIA) — model card / evaluation reporting code-switching 4.92/5, MSA 4.74/5 (ALLaM-34B)
- **Fanar-1-9B** (QCRI) — model card (8.7B, ~17.4 GB BF16; Gulf/Levantine/Egyptian)
- **SILMA TTS v2** — vendor self-published KSA-dialect benchmark *(low confidence — not independently verified)*

> **Confidence legend:** STT findings are **high confidence** (3–0 adversarial verification against primary benchmarks). LLM findings are **medium–high** (model-card / evaluation sourced). TTS findings are **low confidence** (vendor self-report only) — verify independently.

---

*Generated from a deep-research workflow run (105 agents, adversarial 3-vote verification). Last updated 2026-06-08.*
