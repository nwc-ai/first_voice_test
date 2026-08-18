# Egyptian dialect support — everything that was built, why, and what happened to it

This document explains every piece of Egyptian-dialect work in this pipeline in detail —
not just what changed, but why each decision was made and what was learned. For a shorter,
diff-style summary against the pre-Egyptian baseline, see `EGYPTIAN_DIALECT_CHANGELOG.md`.
The dated, line-level source of truth behind every claim here is `eval/BASELINES.md`
(append-only — nothing in it has ever been edited or deleted).

---

## 1. The governing constraint

Egyptian Arabic was reintroduced on 2026-07-20 under one hard invariant: **every byte sent
on Najdi/Fusha/English/mixed turns must stay identical to the pre-Egyptian baseline.**
`eval/golden_prompts.py` enforces this mechanically — it hashes the exact prompt bytes for a
fixed set of non-Egyptian cases and fails loudly if anything moves. In practice this meant
Egyptian was added strictly *additively*: new detection, a new prompt card, a new voice —
none of it touching the code path other dialects already used. Two later changes (marker
rebalancing, generalized history-clearing) deliberately crossed that line in a measured,
owner-approved way — see §2 and §3.

---

## 2. Detection & routing (`routing.py`)

**`looks_egyptian(text)`** — lexicon-based detection against `_EGYPTIAN_MARKERS`, a set of
distinctly-Egyptian vocabulary (إيه، فين، كام، دلوقتي، عايز، etc.). Two real gaps were found
and fixed after the initial rollout:
- **ليه** (why) was missing entirely, despite sitting right next to إيه/فين/كام (markers that
  *were* included) and being taught in `EGYPTIAN_CARD`'s own word list. A plain oversight,
  not a deliberate exclusion — added.
- **و/ف/ب-prefixed markers didn't match.** Arabic glues conjunctions/prepositions directly
  onto the next word with no space — وإزاي، وعايز normalize to one token that never equals
  the bare marker. Added prefix-stripping for these three letters specifically (unlike the
  definite article ال, which `looks_egyptian()`'s own docstring explicitly does NOT strip,
  since stripping ال would manufacture false hits from ordinary MSA).

**`requested_egyptian(text)`** — detects explicit requests ("reply in Masri/Egyptian").
Guarded against two known false-positive shapes from the start: a proper noun ("the Egyptian
Museum") never fires, and a negated request ("لا ترد بالمصري" / "don't answer me in Masri")
never fires. Later extended (2026-07-27) to tolerate an object pronoun between the verb and
the dialect name for English requests — "Answer **me** in Masri please" was originally
missed (only the pronoun-less form matched), and the live consequence was real: the model,
never told this was an explicit Egyptian request, hallucinated a fake self-imposed rule
("my instructions require me to speak only English or Najdi") and replied in English.

**دول** — a genuine MSA homograph (دول = "these," also read as "duwal" = "countries," e.g.
"دول الخليج"). Demoted to a *weak* signal on the detection side specifically because of this
ambiguity — it never independently triggers Egyptian routing, only reinforces alongside a
stronger marker.

**Najdi-first precedence** — `looks_najdi()` always short-circuits before `looks_egyptian()`
is even evaluated, for genuinely Najdi-exclusive markers. This is an explicit, accepted
design choice: any text carrying one of those markers routes Najdi even if Egyptian markers
are also present. The cost of this precedence is what the marker-rebalancing work below
addresses.

**Marker rebalancing (2026-07-24)** — اللي، عشان، لسه، يلا were removed from
`_NAJDI_MARKERS` entirely (not moved to `_EGYPTIAN_MARKERS` either — they simply stop being a
dialect-discriminating signal). These four words are common in **both** Najdi and Egyptian
colloquial speech, so using them to route to Najdi specifically was never well justified — it
was inherited from before Egyptian existed as a routing target at all. Verified by running
the real detection code against every labeled test row before/after, not estimated:

| Metric | Before | After |
|---|---|---|
| Najdi routing recall | 88% (15/17) | 82% (14/17) |
| Egyptian routing recall | 64% (16/25) | 80% (20/25) |

One specific accepted regression: a genuinely-Najdi sentence whose *only* marker was لسه now
goes undetected (routes to `None`/unclear instead of Najdi).

---

## 3. Prompt-side guidance

**`EGYPTIAN_CARD`** — a dedicated prompt card injected only on Egyptian-routed turns.
Contains vocabulary choices built up over several rounds of real-usage review — e.g.
"settle/sediment=يترسب", "there isn't=مفيش", "dry out=جف→نشف" — plus short grammatical
guidance sentences like "technical utility nouns are the same in Egyptian and MSA."

**`EGYPTIAN_GRAMMAR_RULE`** — pattern-level guidance (not word lists, since these are
morphological patterns no word list could cover):
- ما...ش negation onto a noun instead of a verb (invalid: "ما يجفافش"; valid: "ما يجفش").
- Weak-final-verb ش-negation vowel shift (تنسى → تنساش).
- بي-habitual-prefix drop after subjunctive/purpose triggers — عشان/لازم/إذا/لو take يكون,
  not بيكون.
- Form-V/VI تت-/اتـ prefix requirement (تتجنب, not تجنب).

Every one of these carries the same honest ceiling every prompt-side grammar rule in this
project has: it reduces the error rate, it doesn't guarantee elimination, and there's no
automated way to verify morphology/agreement patterns beyond spot-checking real transcripts.

**`NAJDI_NO_OTHER_DIALECTS_RULE`** — Najdi-routed replies must never mention Egyptian
material. This exists because of a measured "pink elephant" effect: mentioning Egyptian on a
Najdi turn measurably increased actual Egyptian-word leakage into that Najdi reply, not just
a theoretical risk.

**History clearing on a dialect switch** — originally scoped to Egyptian only
(`crosses_egyptian_boundary`): the rolling 3-turn conversation history clears whenever a turn
crosses from one Arabic dialect to a different one, to stop the model stylistically imitating
its own immediately-prior reply in the wrong dialect (the documented mechanism behind a 67%
pre-fix Najdi leak rate). Generalized 2026-07-27 to any Najdi/Fusha/Egyptian pair
(`crosses_dialect_boundary`) once the original Egyptian-only scoping was traced back to a
byte-invariance constraint specific to that week's rollout, not a belief that Najdi↔Fusha
switches are safe to leave uncleared. **Known, accepted cost**: Fusha is a default fallback
bucket, not a positively-detected register, and Najdi recall is imperfect — so this
introduces a real (if lower-stakes) spurious-clear rate for Najdi↔Fusha that Egyptian's own
boundary never had, since Egyptian requires distinctive markers or an explicit request to
trigger at all.

---

## 4. Deterministic text repairs

**`DIALECT_REPAIR_MAP`** (`routing.py`, applied via `apply_dialect_repairs()` right before
synthesis, and on the assistant turn stored into rolling history) — word-level, regex-based
substitutions for patterns confirmed to be genuine, recurring model habits, not one-off
noise:
- Egyptian: الذي→اللي، تمشى→تمشي، تأكل→تاكل.
- Najdi: جداً/جدا→مرة (unaffected by the Egyptian جداً correction below — this repair is
  independently confirmed correct for Najdi, across many runs).
- **مش فيه→مفيش was considered and rejected** for this map — فيه is itself a homograph
  (existential "there is" vs. locative "in it," e.g. "الفلوس مش فيه" can mean "not inside
  it"), so a blind substitution fails the map's own admission bar ("unambiguous in ANY
  sentence"). Left as a prompt-only nudge in `EGYPTIAN_CARD` instead.

**`_MINA_MISSPELL_RE`** — a dialect-agnostic fix (the only repair in this map that isn't
keyed per-dialect, since the typo appeared on both Najdi- and Fusha-routed replies) for a
recurring منى (Mina, the place name) → منا misspelling. Gated on adjacency to a specific set
of domain nouns (خزان|محطات|مخيمات|منطقة) so it can never fire on the extremely common word
منا ("from us," من+نا — واحد منا، طلب منا، قريب منا).

**Detection-only fixes in `leak_lint.py`** (no repair needed, just correct flagging):
شو (Levantine "what") added to the always-forbidden list; a suffix-aware `_TSAWWA_RE` regex
added for the تسوي/يسوي/نسوي/سويت family, since Arabic attaches object pronouns directly
onto the verb (تسويها never exact-matches a bare-root word-list entry).

---

## 5. Three corrections — mistakes found *in the detectors themselves*

Three separate times, something flagged as an Egyptian "leak" turned out, on close
inspection, to be genuinely correct usage — each time because the detector didn't account for
a second meaning of the same word:

| Word | Was treated as | Actually | Fix |
|---|---|---|---|
| جداً/جدا | Forbidden leak, auto-repaired to أوي | Completely normal, correct Egyptian Arabic | Removed from Egyptian's forbidden list and repair map (Najdi's own جداً→مرة repair is a *different* correction and stays correct — Najdi's جداً leak is real) |
| دول | Hard-forbidden outside Najdi/Fusha | A genuine MSA word ("countries"), unrelated to the Egyptian demonstrative معنى | Removed from the forbidden list |
| مرة | Flagged 10+ times across several eval runs as a repeated leak | Every single flagged instance, checked individually, was مرة meaning "time/occurrence" (أول مرة، كل مرة) — never the "very" sense actually forbidden | Replaced the plain word-match with a context-aware regex (`_marra_leaks`/`_MARRA_TIME_RE`) that only flags مرة outside a known time-of-occurrence construction |

The repeated lesson: a raw leak count is not evidence on its own — every flagged instance
needs to actually be read before being treated as real.

---

## 6. Voice — diacritization

Unlike Fusha (which gets CATT tashkeel before every synthesis), **Egyptian replies are never
diacritized**, and this was a deliberate, twice-tested decision, not an oversight:

- **Research first (2026-07-30)**: no production-ready Egyptian-specific diacritizer existed
  at the time. More fundamentally, Egyptian colloquial writing lacks MSA's i'rab (case
  endings) — the actual thing tashkeel disambiguates — so there's architecturally little for
  a diacritizer to fix in the first place.
- **Tried anyway, 1st time (2026-08-10)**: CAMeL Tools' BERT-based Egyptian disambiguator —
  a real, working, downloadable model — was wired in as an opt-in candidate at the owner's
  explicit request despite the research above. Its runtime dependency
  (`morphology-db-egy-r13`) is GPL v2 (a different license category from anything else in
  this project), carefully isolated behind the feature flag so it stays fully inert with the
  flag off. Live-tested: **"not working perfectly."** Fully removed the same day — code,
  9 pip packages uninstalled, the GPL v2 data deleted from disk.
- **Independent corroboration found afterward**: Lahgtna-OmniVoice's own model card (a
  completely unrelated project) documents the same finding from a different angle — its v1/v2
  trained on diacritized Egyptian text and "lose coherence and babble"; v3 switched to raw,
  non-diacritized text specifically to fix this.
- **Tried again, 2nd time (2026-08-13)**, at the owner's explicit request, armed with that
  corroboration. Live-tested against VoiceTut-TTS this time (a different engine than the
  first test): **"completely bad."** Removed again, same day, same discipline.
- **Current state**: no diacritization step for Egyptian at all. The conclusion is now backed
  by two independent live tests against two different TTS engines, plus one independent third
  party's own findings — not planned to be tried a third time absent a genuinely different
  candidate.

---

## 7. Voice — the TTS engine search

**Root cause** (found directly in OmniVoice's own docs, `docs/languages.md`): the base
OmniVoice model was trained on only **~23.2 hours** of Egyptian audio, vs. ~204h Najdi and
~1,484h MSA. That gap — not the reference clip, not the `language=` kwarg — is why Egyptian
sounded worse than Najdi/Fusha even once routing and prompting were solid.

Four alternative engines were installed and live-tested, each judged the only way voice
quality is ever judged in this project: by ear, not by benchmark.

**Habibi-TTS** (`SWivid/Habibi-TTS`, F5-TTS architecture, an Egyptian-specialized checkpoint,
37–103h Egyptian training audio) — made the default 2026-07-30. Real install obstacles hit
and solved: installed with `--no-deps` to stop its stated `torch<2.9.0` pin from downgrading
this box's `torch==2.11.0+cu130`; needed a soundfile-backed monkeypatch for `torchaudio.load()`
since this torchaudio build wants the separate `torchcodec` package + system FFmpeg (absent,
no sudo). License caveat: stated Apache-2.0 in the README's prose but tagged
`cc-by-nc-sa-4.0` at the HuggingFace-repo level — an unresolved discrepancy, accepted for
local testing only. **Live-tested: pronunciation quality unsatisfactory.**

**Lahgtna-OmniVoice** (`oddadmix/lahgtna-omnivoice-v2`) — wired 2026-07-30 as a zero-cost A/B
candidate specifically because it's a genuine continued-training fine-tune of this project's
own OmniVoice base (confirmed via real HuggingFace Trainer checkpoint artifacts and a
`train_config.json` showing 20,000 training steps) — same loader class already in
production, no new package, no CUDA risk. Training data itself was undisclosed; the HF
weights repo carries no license tag. Its raw output ran noticeably quieter than Habibi's, so
it was peak-normalized to match. **Live-tested: also unsatisfactory.**

**EGTTS-V0.1** (`OmarSamir/EGTTS-V0.1`, an XTTS-v2 architecture fine-tune) — wired
2026-08-11 as a third candidate. Real install work: the model card's own install command
didn't work on this Python version (blocked at `<3.12`), so an actively-maintained fork's
PyPI package was substituted instead; three separate runtime monkeypatches were needed (a
`transformers` function the model's code imports by name but this build removed; a
`torchcodec`-availability gate bypassed; the same soundfile-backed `torchaudio.load()` shim
as Habibi). License: Coqui Public Model License (CPML) — explicitly non-commercial, the most
restrictive license of any candidate tried. **Live-tested 2026-08-12: "very very very bad."**
Fully removed the same day — code, 6 pip packages uninstalled, ~5.3GB checkpoint deleted.

**VoiceTut-TTS** (`mohammedaly22/VoiceTut-TTS`) — wired 2026-08-13. Also a genuine fine-tune
of this project's own OmniVoice base, confirmed via HuggingFace's own structured
`base_model:k2-fsa/OmniVoice` tag and `config.json`'s `model_type == "omnivoice"`. Verified
hands-on that it loads through the *already-installed* `omnivoice` package with **zero new
pip packages** — despite its own GitHub docs claiming OmniVoice must be installed from
source, that claim did not hold up under real testing. Best license/data story of anything
tried: Apache-2.0 confirmed with no discrepancy in either the model card or HuggingFace's
structured license field, and **~380 hours** of disclosed, dialect-tagged Egyptian YouTube
podcast training audio — more Egyptian-specific training data than any other candidate in
this project has ever disclosed. **Live-tested and confirmed** — the first candidate to
actually win a live comparison. **Promoted to the permanent default.**

Once VoiceTut-TTS won, **Habibi-TTS and Lahgtna-OmniVoice were both fully removed** — not left
dormant alongside the new winner: code deleted, every pip package uninstalled, every
downloaded checkpoint deleted from disk (~3.65GB freed in total). Every removal in this whole
search — Habibi and Lahgtna here, EGTTS-V0.1 earlier — followed the identical discipline:
actually uninstall packages, actually delete downloaded model data, remove every code/test/doc
reference, and record a new dated entry in `eval/BASELINES.md` explaining what was tried and
why it didn't work (the entry stays forever — append-only — even once the code is gone).

**Current fallback chain**: Egyptian-routed turns try, in order, VoiceTut-TTS (default) →
OmniVoice's own Egyptian voice-clone prompt + `language="egyptian arabic"` (if VoiceTut is
disabled, unavailable, or fails for that sentence) → the Saudi voice (if the Egyptian
reference clip itself is missing). Every tier is a soft-fail — a third-party model failure
never crashes a turn or blocks server startup.

**The Egyptian reference clip itself** needed real iteration before any of the above: v1, v2,
and a synthetic NAMAA-sourced clip were all rejected by ear before v4 (the current clip) was
settled on. The code comment next to it says outright: "this voice has a history of needing
iterations, judge it by ear before trusting it."

---

## 8. A side quest: an alternative LLM, tested for Egyptian quality too

Two alternative LLMs were live-tested against qwen3.5:27b, gated entirely behind an opt-in
env var (`LLM_MODEL_OVERRIDE`) that never changes without being explicitly set — the shipped
default has stayed qwen3.5:27b throughout.

- **Fanar-2-27B-Instruct** — wired 2026-07-27. An offline 245-question comparison found it
  genuinely fixes a Najdi verb-morphology leak qwen3.5 can't get past, but fabricates
  specific-sounding operational facts (tank levels, pressure readings) roughly 3× more often,
  plus more false physical-capability claims and two more dialect-request misses.
  Recommendation given: keep qwen3.5 in production. Kept available for further live,
  hands-on testing at the owner's explicit request — never adopted as the default.
- **Fanar-1-9B-Instruct** — wired 2026-08-07 as a second, smaller candidate. Live-tested,
  found instruction-following too weak to be worth pursuing further, removed the very next
  day.

---

## 9. Evaluation infrastructure built specifically for this work

- **`eval/dialect_id_eval.py` + `dialect_id_cases.jsonl`** — routing recall/precision per
  dialect against hand-labeled examples; the numbers quoted throughout this document (82%
  Najdi, 80% Egyptian recall) come from here.
- **`eval/dialect_eval_full.py` + `dialect_eval_questions.json` /
  `dialect_eval_holdout_questions.json`** — the 245-question native-Arabic eval (real routing
  *and* reply quality measured together, unlike the English-question harness below), the
  largest single measurement tool built for this work.
- **`eval/dialect_ab.py`** — an English-question A/B harness calling Ollama directly
  (LLM + prompt layer only — by its own docstring, it structurally cannot validate any
  TTS-side fix, a distinction that mattered several times when checking whether a given
  before/after number was even a valid comparison for a given change).
- **`eval/test_dialect_repair.py` / `eval/test_leak_lint.py`** — regression pins for every
  repair/detection fix, sourced from real transcript text wherever one existed.
- **`eval/golden_prompts.py`** — the byte-invariant gate (G1/G2/G3) that made the governing
  constraint in §1 mechanically enforceable rather than just a stated intention.
- **`eval/quality_lint.py`** — an LLM-judge quality checker, built and validated against known
  defects, run once for a full 45-turn sweep, then discontinued per owner instruction after it
  was found to fabricate/contradict itself (flagged a correct reply as wrong, wrote "not a
  defect" in its own reasoning while still surfacing the finding). Quality checking since then
  has been direct manual transcript review.
- **`eval/BASELINES.md`** — the append-only dated log underlying every claim in this document.

---

## 10. Documented limitations — investigated, understood, and deliberately not fixed

- **Egyptian gender agreement** (بيبقى→بتبقى class, when the subject is feminine). No safe
  regex/lookup shortcut exists — Arabic's free word order and pro-drop break any "nearest
  noun" heuristic. The two real fixes (a live self-check pass, or a reliable Egyptian-dialect
  parser) are either ruled out (adds latency to every turn) or don't exist in deployable form
  (checked: no production-quality, dialect-tolerant Arabic dependency parser was found).
- **ج loanword pronunciation** (اوكسجين pronounced with Egyptian's standard hard "g" ج when
  the source loanword needs a soft "j"). Confirmed by reading OmniVoice's own source: there is
  no word/phoneme-level pronunciation override for Arabic at all. A hard TTS-layer wall.
- **Egyptian's سوى-verb-family leak** (a Najdi verb habit leaking into Egyptian replies) and
  **the positional مرة-as-"very" repair** — both real, recurring patterns, documented but
  deliberately not attempted. Each would need meaningfully more complex, error-prone
  conjugation-aware or positional regex work than any repair currently in the map.
- **Hallucination mitigation** — prompt-side anti-hallucination wording measurably helps but
  plateaus (the model still occasionally invents confident-sounding specific facts). The
  actual fix, retrieval-augmentation, is a materially bigger effort, ruled out of scope here.

---

## 11. Where things stand today

- Egyptian is detected via `looks_egyptian()`/`requested_egyptian()`, replies use
  `EGYPTIAN_CARD` + `EGYPTIAN_GRAMMAR_RULE` guidance, and history clears on any dialect
  boundary crossing.
- Egyptian replies are synthesized via **VoiceTut-TTS** (default), falling back to plain
  OmniVoice's Egyptian voice clip, then the Saudi voice.
- Egyptian replies are **not** diacritized.
- The LLM is still **qwen3.5:27b** by default; Fanar-2 remains available only as an explicit,
  opt-in override.
- Every item in this document passed the full non-regression gate suite
  (`test_routing.py`, `golden_prompts.py`, `dialect_id_eval.py`, `test_tts_args.py`,
  `test_dialect_repair.py`, `test_leak_lint.py`) before being considered done.
