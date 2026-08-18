# What changed since commit 5b2b733 (July 12 → August 17, 2026)

**Baseline:** `5b2b733` — "Commit before the Demo on July 12, 2026." This is the last commit before any Egyptian-dialect work began. Najdi and Fusha quality was good at this point.

**Current HEAD:** `be90ab1` — "Finalised the TTS(VoiceTut-TTS) and the LLM(Fanar) for the Egyptian Dialect - 17 August, 2026."

Three real commits sit between them (`5850170`, `42d612a`, `be90ab1`), but the actual work happened in many small, individually-tested steps — most of it recorded in `eval/BASELINES.md`, which is the authoritative append-only log this document is built from.

---

## TL;DR — what actually happened to Najdi and Fusha

**Fusha: untouched, flawless throughout.** Every eval run in this entire five-week window — including the full 245-question sweeps — measured **zero leaks, zero routing errors, zero grammar issues** for Fusha. Nothing done for Egyptian ever changed Fusha's code path.

**Najdi: one real, deliberate, measured tradeoff.** Two changes affected Najdi:

1. **2026-07-24** — four words (اللي، عشان، لسه، يلا) were removed from the Najdi marker list because they're common in Egyptian too, not Najdi-exclusive. This traded Najdi routing recall **88% → 82%** for Egyptian routing recall **64% → 80%**. One specific accepted regression: a Najdi sentence whose *only* marker was لسه now goes undetected.
2. **2026-07-27** — history-clearing on a dialect switch, originally Egyptian-only, was generalized to all Arabic dialect pairs. This means a Najdi↔Fusha switch now also clears the 3-turn rolling context, which it didn't before. Accepted as a cost of a simpler, symmetric rule; a debounce alternative was considered and deferred.

**Everything else was Egyptian-specific and structurally could not touch Najdi/Fusha** — a hard byte-invariant (enforced by `eval/golden_prompts.py`) requires every Najdi/Fusha/English/mixed prompt to stay byte-identical to the pre-Egyptian baseline. That gate stayed green through every single step below.

The one live-mic LLM experiment (Fanar-2/Fanar-1) is opt-in via an env var and never changed the shipped default (qwen3.5:27b).

---

## Phase 1 — Egyptian dialect reintroduction (July 20–22)

Egyptian Arabic was reintroduced under a hard invariant: every byte sent on Najdi/Fusha/English/mixed turns must stay identical to the pre-Egyptian baseline. Najdi detection always wins first over Egyptian for genuinely Najdi-exclusive markers.

- **Routing**: added `looks_egyptian`/`requested_egyptian` detection to `routing.py` — guarded so "Egyptian Museum" or a negated request never misfires.
- **Prompt card**: `EGYPTIAN_CARD` (vocabulary, grammar rules) appended only on Egyptian-routed turns — explicitly never mentioned on Najdi turns (a measured "pink elephant" regression made this a hard rule).
- **History clearing**: rolling conversation history clears when a turn crosses from one Arabic dialect to another — Egyptian-only at first, generalized later (see Phase 2).
- **TTS**: a dedicated Egyptian reference voice clip (own voice-clone prompt) wired into `tts_omnivoice_v1.py`.
- **Corrections made in this phase**, all owner-confirmed against real usage:
  - جداً/جدا was wrongly treated as an Egyptian leak — it's genuinely correct Egyptian Arabic. Removed from the forbidden list and the auto-repair map. (This does *not* apply to Najdi — جداً→مرة repair stays correct there.)
  - دول was wrongly hard-forbidden outside Egyptian — it's a genuine MSA homograph ("countries"). Removed.
  - New lexical repairs added for Egyptian: الذي→اللي، تمشى→تمشي، تأكل→تاكل.
  - New grammar rules added for Egyptian negation/verb-form patterns.
- **Full 245-question eval** (`dialect_eval_full.py`, native dialect-phrased Arabic, real routing) established the working baseline: Fusha 65/65 clean; Najdi 65 questions, 12 leaky; Egyptian 115 questions, 9 leaky, ~30 routing misses (mostly the known Najdi-first collision).
- Two real Egyptian routing gaps found and fixed: `ليه` (why) was missing from the marker list; و/ف/ب-prefixed markers (e.g. وإزاي) weren't matching due to Arabic's glued-conjunction spelling.

## Phase 2 — Marker rebalancing and history-clearing generalization (July 24–27)

This is the phase with the two real Najdi-affecting changes described in the TL;DR above.

- **2026-07-24**: اللي/عشان/لسه/يلا removed from the Najdi marker set (detailed in the TL;DR). Verified impact by running the real detection code against every labeled test row before/after, not estimated.
- **2026-07-27**: history-clearing generalized from Egyptian-only to all Najdi/Fusha/Egyptian pairs (detailed in the TL;DR). Verified: gate suite identical before/after this change specifically, isolating its effect from the marker-set change three days earlier.
- **Re-ran the full 245-question eval** after both changes: Egyptian routing improved (32→22 routing-bad), Najdi routing worsened slightly (11→15, matching the predicted marker-removal cost exactly), Fusha unchanged (0→0).
- **A false alarm corrected**: مرة had been flagged as a repeated Egyptian leak across ten-plus instances over several runs. A full manual check of every flagged instance found **all of them were the correct word** (مرة meaning "time/occurrence," e.g. أول مرة) — the leak *detector* was broken, not the model's output. Fixed the detector with a context-aware regex; no actual leak has ever been confirmed.
- **منى → منا typo fix**: a recurring place-name misspelling (Mina), appearing on both Najdi and Fusha replies, fixed with a narrowly-scoped regex that can't collide with the common word منا ("from us").
- **English explicit-dialect-request bug fixed**: requests like "Answer me in Masri please" (with an object pronoun) weren't recognized as Egyptian requests — the model would hallucinate a fake self-imposed rule and reply in English instead. Fixed.
- **Fanar-2 wired for a live mic/browser test** (owner-requested, not adopted as default): an env var (`LLM_MODEL_OVERRIDE`) lets the pipeline swap qwen3.5:27b for Fanar-2-27B-Instruct for a one-off test. Unset behaves byte-identically to before. An offline 245-question comparison found Fanar-2 fixes a Najdi verb-morphology leak qwen3.5 can't, but fabricates specific-sounding operational facts roughly 3× more often — recommendation was to keep qwen3.5 in production; the owner tested Fanar-2 live anyway as a deliberate one-off.

## Phase 3 — The Egyptian TTS engine search (July 30 – August 13)

This phase is entirely about *voice quality* for Egyptian — none of it touches Najdi/Fusha routing, prompts, or history logic. Root cause, found in OmniVoice's own docs: OmniVoice's base model saw only ~23 hours of Egyptian training audio vs. ~204h Najdi and ~1,484h MSA — which is why Egyptian sounded worse than Najdi/Fusha even after the dialect logic above was solid.

Four alternative Egyptian voices were tried, each judged by the owner's own live listening test — the only method trusted in this project:

| Candidate | Result |
|---|---|
| **Habibi-TTS** (F5-TTS arch., Egyptian-specialized checkpoint) | Made default 2026-07-30. Live-tested, pronunciation quality judged unsatisfactory. |
| **Lahgtna-OmniVoice** (fine-tune of this project's own OmniVoice base) | Wired as an opt-in A/B candidate 2026-07-30. Also judged unsatisfactory. |
| **EGTTS-V0.1** (XTTS-v2 fine-tune) | Wired 2026-08-11, needed three runtime monkeypatches to install. Live-tested 2026-08-12, judged "very very very bad." Fully removed same day. |
| **VoiceTut-TTS** (`mohammedaly22/VoiceTut-TTS`, also a fine-tune of this project's own OmniVoice base) | Wired 2026-08-13. First candidate to actually win — cleanest license story (Apache-2.0, no discrepancy) and most disclosed Egyptian training data (~380h) of anything tried. **Promoted to the permanent default.** |

Once VoiceTut-TTS won, **Habibi-TTS and Lahgtna-OmniVoice were both fully removed** — not left dormant: code deleted, all pip packages uninstalled, downloaded model checkpoints deleted from disk (~3.65GB freed). EGTTS-V0.1 had already been fully removed the same way when it failed on 2026-08-12.

**A side quest during this phase**: Fanar-1-9B-Instruct was wired as a second LLM live-test candidate (2026-08-07), live-tested, found instruction-following too weak, and fully removed the next day.

**A diacritizer detour, tried twice**: CAMeL Tools' BERT-based Egyptian disambiguator was wired as an opt-in Egyptian tashkeel step (mirroring CATT's role for Fusha) on 2026-08-10 — live-tested, found "not working perfectly," fully removed the same day (including its GPL v2 runtime data dependency). Re-added 2026-08-13 for a second live test at the owner's explicit request, after research turned up independent corroboration from a different team (Lahgtna's own model card documents that diacritized Egyptian text "loses coherence and babbles"). The second live test also came back bad ("completely bad") and it was removed again the same day. **Conclusion, now confirmed twice by ear plus one independent third party**: Egyptian colloquial text doesn't have the grammatical ambiguity (i'rab/case endings) that tashkeel exists to resolve for MSA, so a diacritizer has nothing useful to fix.

---

## Where things stand now

- **Najdi & Fusha**: same routing/prompt logic as the pre-Egyptian baseline, plus the two accepted changes in the TL;DR (marker trim, generalized history-clearing). Fusha has never regressed. Najdi's routing recall sits at 82% (was 88%), a known and accepted tradeoff for Egyptian's gains.
- **Egyptian**: routed via `looks_egyptian`/`requested_egyptian`; replies use `EGYPTIAN_CARD` grammar/vocabulary guidance; synthesized via **VoiceTut-TTS** (default) with plain OmniVoice's Egyptian voice clip as the sole fallback; no diacritization step.
- **LLM**: still locked to **qwen3.5:27b** by default. Fanar-2 remains available as an explicit, opt-in, one-off override (`LLM_MODEL_OVERRIDE`) — never the shipped default.
- **Every step above passed the full non-regression gate suite** (`test_routing.py`, `golden_prompts.py`, `dialect_id_eval.py`, `test_tts_args.py`, `test_dialect_repair.py`, `test_leak_lint.py`) before being considered done — nothing here shipped without a green gate run.

**Full detail, dated and in order, for every item above**: `eval/BASELINES.md` (append-only — nothing in it has ever been edited or deleted, only added to).
