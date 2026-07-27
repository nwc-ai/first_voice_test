"""
quality_lint.py — LLM-judge quality findings (soft, informational — NOT a gate)
=============================================================================================
leak_lint.py can only ever recognize whether a token is a member of a curated forbidden/drift
set — it has no way to catch grammar, spelling, or factual errors. A manual transcript review
(2026-07-21) found six defect classes leak_lint structurally cannot see, several of them sitting
in existing dialect_ab.py reports tagged "clean": grammatical-person errors (advice self-framed
instead of addressing the listener), gender-agreement inconsistency within one reply, invented/
non-existent Arabic words, merged-word typos, factual/historical hallucination, and wrong-but-
real-word usage.

This module feeds each generated reply (plus the question and routed dialect) to a second,
locally-available LLM acting as a judge, with a JSON-schema-constrained prompt, and returns a
list of finding dicts. Findings are SOFT/INFORMATIONAL ONLY:
  - An LLM judge is probabilistic — false positives and false negatives are both expected, and
    the rate can drift between runs on identical input.
  - This project already measured what happens when a noisy/untrustworthy signal gets treated
    as a hard gate: NAJDI_NO_OTHER_DIALECTS_RULE (routing.py) was reverted after a documented
    regression. Nothing here raises, exits non-zero, or is wired into a lettered gate.

Judge model: qwen3:32b (overridable via QUALITY_JUDGE_MODEL env var / --judge-model CLI flag
at the two call sites, dialect_ab.py and dialect_purity_lint.py). A different checkpoint
lineage than the live-serving qwen3.5:27b — partial, not complete, mitigation of
self-referential bias (same Qwen pretraining lineage/tokenizer, so some correlated blind spots
may not be caught either — a known, documented limitation, not hidden). At 32B it has
meaningfully better general reasoning / JSON-schema reliability than the smaller Arabic-tuned
alternatives also available locally (ALLaM-7B, SILMA-9B) — those are a v1.1 candidate: run as
a second opinion restricted to the invented_word/wrong_real_word categories, surface a finding
only if both judges agree. Not implemented in v1.

HARD PRECONDITION: never run this concurrently with the live server. qwen3.5:27b is pinned via
keep_alive:-1 (llm.py) — Ollama will NOT auto-evict it to make room for a second ~20GB model.
Stop the live server AND run `ollama stop qwen3.5:27b` first. check_judge_model_available()
prints a loud (non-fatal) warning if qwen3.5:27b is still resident.

Usage (library — imported by dialect_ab.py --judge and dialect_purity_lint.py --judge):
    findings = judge_reply(question, dialect, reply, client)

Usage (standalone, for iterating on the judge prompt):
    .venv/bin/python eval/quality_lint.py --dialect Egyptian --question "..." --reply "..."
"""

import argparse
import json
import os
import subprocess
import sys

import httpx

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
JUDGE_MODEL = os.environ.get("QUALITY_JUDGE_MODEL", "qwen3:32b")

FINDING_CATEGORIES = (
    "person_perspective", "gender_agreement", "invented_word",
    "merged_word_typo", "factual_hallucination", "wrong_real_word",
)

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category":      {"type": "string", "enum": list(FINDING_CATEGORIES)},
                    "span":          {"type": "string"},        # verbatim substring of the reply
                    "confidence":    {"type": "string", "enum": ["high", "medium", "low"]},
                    "note":          {"type": "string"},        # short human-readable explanation
                    "suggested_fix": {"type": ["string", "null"]},
                },
                "required": ["category", "span", "confidence", "note", "suggested_fix"],
            },
        },
    },
    "required": ["findings"],
}

# One paragraph per category, independently tunable. Bias toward flag-for-human-review rather
# than the judge asserting a confident correction it might itself get wrong (esp. category 5 —
# the judge can hallucinate too).
_JUDGE_INSTRUCTIONS = """\
You are a strict but careful reviewer of Arabic voice-assistant replies. You will be given
the DIALECT the reply was supposed to be in ("Najdi", "Egyptian", or "Fusha"), the QUESTION
that produced it, and the REPLY text. Find defects ONLY in these six categories. If you find
nothing in a category, do not report anything for it — an empty findings list is a valid and
expected answer; do not invent a finding just to have something to say.

1. person_perspective: the reply is voice advice given TO the listener. Any actionable
   instruction or advice (e.g. "check the pump", "you should check X") must be phrased as
   speaking TO the listener (imperative or "you" framing — تأكد، لازم تشوف، you should check),
   not as the assistant's own first-person intention to act (أنا لازم أفحص، سأتحقق من، I will
   check X). Do NOT flag legitimate first-person self-reference where the assistant is
   describing its own nature/limits (e.g. "أنا مساعد صوتي ولا أقدر أرسل فريق صيانة" is correct,
   not a defect).

2. gender_agreement: Arabic verbs/pronouns addressing "you" carry a grammatical gender
   (masculine تفعل/ـك vs feminine تفعلين/ـكِ). Flag ONLY if the SAME reply switches between
   masculine and feminine forms when addressing the SAME listener (inconsistency within one
   reply). Do NOT flag a reply that consistently uses one gender throughout — that is a
   stylistic default, not an error.

3. invented_word: flag any token that is not a real word in Arabic at all — a form that
   looks like it could be a conjugation but isn't (e.g. "تغل" is not a real verb form; the
   intended word was probably "تقفل"). Only flag tokens you are genuinely confident do not
   exist in ANY register of Arabic — do not flag rare-but-real classical/MSA words, proper
   nouns, or dialect words you are simply unfamiliar with.

4. merged_word_typo: flag any run of characters that is clearly two or more words glued
   together with a missing space (e.g. "فيهالأيام" = "في هالأيام"). This is a typography
   defect, not a grammar defect — flag only cases where a word boundary is visibly missing.

5. factual_hallucination (bias toward flag-for-human-review, not asserting a correction):
   flag any specific proper noun or confident historical/factual claim (street names,
   building names, architectural-style names, dates, named historical events/relationships
   between places) that sounds suspiciously overly specific for a domain you are not
   confident about. You are NOT being asked to fact-check or supply the correct answer —
   you may be wrong too. Only flag that the claim is SPECIFIC AND CONFIDENT-SOUNDING enough
   that a human should verify it; do not state what the correct fact actually is, and do not
   mark something as flagged just because it is a true, well-known fact stated plainly
   (e.g. "Jeddah is on the Red Sea coast" is common knowledge, not a flag).

6. wrong_real_word: flag a real, correctly-conjugated Arabic word that is nonetheless the
   WRONG word for its context (a semantic/register mismatch) — e.g. "يترسخ" (becomes
   entrenched) used where "يترسب" (settles/precipitates) was clearly meant. Only flag when
   you are confident the word is real Arabic but semantically wrong here, not merely an
   unusual stylistic choice.

Only flag tokens/spans that literally appear verbatim in the REPLY text (copy the exact
substring into "span" — do not paraphrase or normalize it).
"""


def _build_user_message(question: str, dialect: str, reply: str) -> str:
    return f"DIALECT: {dialect}\n\nQUESTION: {question}\n\nREPLY: {reply}"


def judge_reply(question: str, dialect: str, reply: str, client: httpx.Client,
                 model: str = JUDGE_MODEL) -> list[dict]:
    """Return a list of finding dicts (possibly empty). Never raises — a judge-call failure
    (model not loaded, malformed JSON, timeout, ...) degrades to an empty list plus a printed
    warning, matching dialect_ab.py's own per-turn try/except around run_turn."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _JUDGE_INSTRUCTIONS},
            {"role": "user",   "content": _build_user_message(question, dialect, reply)},
        ],
        "stream": False,
        "think": False,   # same rationale as llm.py's qwen3.5 config — judging should be
                          # fast and direct, not burn the token budget on hidden reasoning.
        "format": JUDGE_SCHEMA,
        "options": {
            "temperature": 0.15,   # near-deterministic — judging, not creative generation.
            "top_p":       0.9,
            "num_predict": 500,
        },
    }
    try:
        r = client.post(OLLAMA_CHAT_URL, json=payload, timeout=120.0)
        r.raise_for_status()
        content = (r.json().get("message") or {}).get("content", "")
        parsed = json.loads(content)
        findings = parsed.get("findings", [])
        if not isinstance(findings, list):
            raise ValueError(f"'findings' is not a list: {type(findings)}")
    except Exception as e:
        print(f"[quality_lint] judge call failed, returning no findings: {type(e).__name__}: {e}")
        return []

    # Defensive re-validation — never trust the judge's JSON blindly, even schema-constrained.
    clean: list[dict] = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        category = f.get("category")
        span = f.get("span")
        if category not in FINDING_CATEGORIES or not isinstance(span, str) or not span:
            continue
        confidence = f.get("confidence") if f.get("confidence") in ("high", "medium", "low") else "medium"
        note = f.get("note") if isinstance(f.get("note"), str) else ""
        suggested_fix = f.get("suggested_fix") if isinstance(f.get("suggested_fix"), str) else None
        if category == "factual_hallucination":
            # The judge can hallucinate a "correction" too — never surface one for this
            # category, regardless of what the model returned (belt-and-suspenders on top
            # of the prompt instruction, which the model may not perfectly obey).
            suggested_fix = None
        clean.append({"category": category, "span": span, "confidence": confidence,
                      "note": note, "suggested_fix": suggested_fix})
    return clean


def check_judge_model_available(model: str = JUDGE_MODEL) -> None:
    """Print a loud, non-fatal warning if qwen3.5:27b is still resident in Ollama — it is
    pinned via keep_alive:-1 (llm.py) and will NOT be auto-evicted to make room for a second
    ~20GB judge model, risking an OOM on a shared GPU. Call this once at CLI-entry time when
    --judge is set; never blocks execution, just warns."""
    try:
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10)
        out = result.stdout
    except Exception as e:
        print(f"[quality_lint] WARNING: couldn't run `ollama ps` to check VRAM state ({e}) — "
              f"proceeding, but verify manually that qwen3.5:27b is NOT loaded.")
        return
    if "qwen3.5:27b" in out:
        print("=" * 78)
        print("[quality_lint] WARNING: qwen3.5:27b is still resident in Ollama (pinned via")
        print("  keep_alive:-1 — it will NOT be auto-evicted). Loading a second ~20GB judge")
        print(f"  model ({model}) on top of it risks an OOM on this GPU.")
        print("  Stop the live server, then run: ollama stop qwen3.5:27b")
        print("=" * 78)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dialect", required=True, choices=["Najdi", "Egyptian", "Fusha"])
    ap.add_argument("--question", required=True)
    ap.add_argument("--reply", required=True)
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    args = ap.parse_args()

    check_judge_model_available(args.judge_model)
    with httpx.Client() as client:
        results = judge_reply(args.question, args.dialect, args.reply, client, model=args.judge_model)
    if not results:
        print("No findings.")
    for f in results:
        arrow = f" → suggested: {f['suggested_fix']}" if f["suggested_fix"] else ""
        print(f"[{f['category']}][{f['confidence']}] «{f['span']}»{arrow} — {f['note']}")
    sys.exit(0)
