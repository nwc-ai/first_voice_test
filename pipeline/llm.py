"""
llm.py — Ollama client, model configuration, and prompt construction.

Everything the pipeline needs to turn an accepted utterance into a token stream:
the system prompt, the per-turn language/dialect instruction (build_turn), the
streaming /api/chat generator, and the startup warm-up.
"""

import json
import os
from typing import Any, Optional

import httpx

from .routing import (
    EGYPTIAN_CARD,
    NAJDI_GLOSSARY,
    NAJDI_GRAMMAR_RULE,
    WANTS_ARABIC_RE,
    WANTS_ENGLISH_RE,
    looks_egyptian,
    looks_najdi,
    requested_dialect,
    route_arabic,
)

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

# Default LLM — warmed at startup so the first turn isn't a cold load, and pinned
# in VRAM (a second model alongside it would OOM). LLM_MODEL is an OPT-IN override
# for live A/B testing (e.g. Fanar-2 27B); unset ⇒ behavior is byte-identical to
# the hardcoded default. Switching models requires a restart (and restarting
# Ollama, so the previously pinned 27B is released — two 27Bs don't fit):
#   LLM_MODEL='hf.co/mradermacher/Fanar-2-27B-Instruct-i1-GGUF:i1-Q4_K_M' \
#       bash scripts/start_server.sh
MODEL = os.environ.get("LLM_MODEL", "qwen3.5:27b")

SYSTEM_PROMPT = (
    "You are a voice assistant that supports Arabic dialects and English ONLY. "
    "ABSOLUTE RULES — never break these: "
    "0. LANGUAGE OVERRIDE (highest priority): If the user explicitly asks you to reply in a "
    "   specific language (e.g. 'in Arabic', 'in English', 'بالعربي', 'باللغة العربية'), reply in "
    "   THAT language regardless of which language they wrote their request in. This overrides rules 1-4. "
    "1. Otherwise, if the user speaks English → reply in English only. "
    "2. If the user speaks Najdi Arabic (نجدي) → reply in Najdi "
    "   (use وش/إيش, أبغى, زين, الحين, ماله, يبيلك). "
    "3. If the user mixes Arabic and English (code-switching) → reply in the same natural mix, matching their Arabic dialect. "
    "4. If the user speaks Fusha (Modern Standard Arabic), or their dialect is unclear → reply in Fusha. "
    "   Fusha is a fully valid reply mode — never force a regional dialect on a Fusha speaker. "
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
    "write 'دكتور' instead of 'د.'; write 'أستاذ' instead of 'أ.'; and write 'وما إلى ذلك' instead of 'إلخ'. "
    "13. NEVER claim to perform, schedule, or confirm a real-world physical action (e.g. dispatching a "
    "    maintenance team, sending a truck, opening a ticket, fixing something). You are a voice assistant "
    "    with no ability to do any of that. If asked, clearly say you cannot perform the action yourself and "
    "    direct the user to contact the relevant team or service channel instead."
)

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
    "fanar": {
        # Fanar-2-27B-Instruct (Gemma-3-27B base, community GGUF) — opt-in A/B
        # model via LLM_MODEL. think:False — Ollama recognizes the model's
        # thinking capability and maps this onto the template's no_thinking
        # empty-<think> prefill; strip_think_tokens (below) is the safety net if
        # reasoning text still leaks into content. Sampling: Gemma-3-family
        # defaults (top_p 0.95 / top_k 64) with temperature lowered for factual
        # grounding, mirroring the qwen3.5 rationale. Tunable during A/B.
        "extra":   {"think": False},
        "options": {
            "temperature":  0.6,
            "top_p":        0.95,
            "top_k":        64,
            "num_predict":  300,
            # MUST match the warm-up num_ctx or the model reloads at first chat
            # (and risks a double-load OOM while pinned) — same lesson as qwen.
            "num_ctx":      LLM_NUM_CTX,
            "stop":         _STOP_SEQUENCES,
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


def get_model_config(model_name: str) -> dict[str, Any]:
    """Return the config for the given model name, matched by substring."""
    lower = model_name.lower()
    for key, cfg in MODEL_CONFIGS.items():
        if key != "default" and key in lower:
            print(f"  [config] matched '{key}' for model '{model_name}'")
            return cfg
    print(f"  [config] no match for '{model_name}', using default config")
    return MODEL_CONFIGS["default"]


# ── Per-turn language routing → LLM instruction + TTS language ────────────────

def build_turn(text: str, lang: str) -> tuple[str, Optional[str], bool]:
    """Decide this turn's reply-language instruction and TTS language.

    Returns (turn_content, tts_language, explicit):
      turn_content — the wrapped user message sent to the LLM (instruction + style
        rules + the raw text). Only the CLEAN text is stored in history, so these
        per-turn instructions never accumulate across turns.
      tts_language — selects the TTS engine (Egyptian → VoiceTut, everything else
        → OmniVoice) and gates CATT tashkeel to Fusha (the TTS module re-checks
        each synthesized sentence with the same Najdi detector, so a reply that
        comes back Najdi is never MSA-diacritized regardless of this value).
      explicit — the user explicitly requested an output language/dialect this
        turn ("بالمصري", "in English", ...). Used by server.py to bypass the
        Arabic dialect-history isolation: restating a prior answer in another
        dialect NEEDS that prior answer visible.
    """
    # A named dialect (Najdi/Egyptian/Fusha) counts as an Arabic request on its
    # own — even when "Arabic" isn't said, e.g. "in Najdi Arabic". Other named
    # dialects (Gulf, Hijazi, ...) aren't recognized and fall through to the
    # lang-detected routing below.
    req_name, req_phrase = requested_dialect(text)
    wants_arabic = req_name is not None or bool(WANTS_ARABIC_RE.search(text))
    explicit = wants_arabic or bool(WANTS_ENGLISH_RE.search(text))

    if req_name == "Najdi":
        tts_language = "najdi arabic"
    elif req_name == "Egyptian":
        tts_language = "egyptian arabic"
    elif wants_arabic:
        tts_language = "standard arabic"   # Fusha, explicitly named or default
    elif lang == "ar":
        # Tiered dialect router: Najdi-exclusive > Egyptian-exclusive >
        # shared-markers(→Najdi, pre-Egyptian behavior) > Fusha. For any input
        # with no Egyptian-exclusive evidence this returns exactly what the old
        # `"najdi arabic" if looks_najdi(text) else "standard arabic"` returned.
        tts_language = route_arabic(text)
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
    elif WANTS_ENGLISH_RE.search(text):
        print("  [lang] explicit English request")
        lang_instruction = (
            "The user EXPLICITLY asked you to reply in English — honor this "
            "regardless of the language they wrote in. Reply ONLY in English."
        )
    elif lang == "mixed":
        if looks_egyptian(text) and not looks_najdi(text):
            # Exclusive Egyptian evidence and zero Najdi evidence (shared markers
            # included — any Najdi hint keeps the original instruction below).
            lang_instruction = (
                "The user is mixing Arabic and English (code-switching). "
                "Reply naturally in the SAME mix of Arabic and English they used. "
                "For the Arabic parts, use Egyptian Arabic (Masri) — their Arabic "
                "carries Egyptian markers. "
                "Do NOT force a reply into all-Arabic or all-English."
            )
        else:
            lang_instruction = (
                "The user is mixing Arabic and English (code-switching). "
                "Reply naturally in the SAME mix of Arabic and English they used. "
                "For the Arabic parts, use Najdi if their Arabic carries Najdi markers, "
                "otherwise use Fusha (Modern Standard Arabic). "
                "Do NOT force a reply into all-Arabic or all-English."
            )
    elif lang == "ar":
        if tts_language == "najdi arabic":
            lang_instruction = (
                "The user spoke Najdi Arabic. Reply ONLY in the Najdi dialect — "
                "do not switch to Fusha/MSA and do not mix in other dialects."
            )
        elif tts_language == "egyptian arabic":
            lang_instruction = (
                "The user spoke Egyptian Arabic (اللهجة المصرية). Reply ONLY in "
                "everyday spoken Egyptian Arabic — do not switch to Fusha/MSA and "
                "do not mix in other dialects."
            )
        else:
            lang_instruction = (
                "The user spoke Arabic without clear dialect markers. Reply in "
                "Modern Standard Arabic (Fusha). Do NOT force a regional dialect."
            )
    else:
        lang_instruction = "The user spoke English. Reply in English only."

    # Najdi turns (detected or explicitly requested) get the full MSA→Najdi
    # vocabulary glossary so replies use authentic word choices instead of
    # MSA scaffolding with dialect sprinkles, plus a grammar rule against the
    # Levantine/Egyptian بـ-prefix leak found via eval.
    # NAJDI_NO_OTHER_DIALECTS_RULE (باش/ش-negation) was tried and REVERTED — see
    # its docstring in routing.py, it measurably increased the leak it targeted.
    if tts_language == "najdi arabic":
        lang_instruction += "\n" + NAJDI_GLOSSARY + "\n" + NAJDI_GRAMMAR_RULE
    # Egyptian turns (detected or explicitly requested) get the Masri card —
    # vocabulary + positively-phrased grammar. Never appended on any other route.
    elif tts_language == "egyptian arabic":
        lang_instruction += "\n" + EGYPTIAN_CARD

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
    return turn_content, tts_language, explicit


# ── LLM token generator ───────────────────────────────────────────────────────

_THINK_OPEN  = "<think>"
_THINK_CLOSE = "</think>"


async def strip_think_tokens(token_gen: Any):
    """Remove <think>...</think> spans from a token stream.

    Safety net for Fanar-2 (Gemma-3 GGUF): its thinking delimiters are PLAIN TEXT,
    not special tokens. think:False in the fanar config should suppress reasoning
    at the template level; if it ever leaks anyway, this keeps reasoning text out
    of the spoken reply. Tags may arrive split across tokens, so a small tail is
    buffered until it can't be a tag prefix. Applied ONLY when "fanar" is in the
    model name — the qwen path never goes through here.
    """
    buf = ""
    thinking = False
    try:
        async for tok in token_gen:
            buf += tok
            out = ""
            while True:
                if thinking:
                    idx = buf.find(_THINK_CLOSE)
                    if idx == -1:
                        # Discard thinking text, keep only a tail that could be
                        # the start of the close tag.
                        buf = buf[-(len(_THINK_CLOSE) - 1):] if buf else ""
                        break
                    buf = buf[idx + len(_THINK_CLOSE):]
                    thinking = False
                else:
                    idx = buf.find(_THINK_OPEN)
                    if idx == -1:
                        # Emit everything except a tail that could be the start
                        # of an open tag.
                        keep = 0
                        for k in range(min(len(_THINK_OPEN) - 1, len(buf)), 0, -1):
                            if _THINK_OPEN.startswith(buf[-k:]):
                                keep = k
                                break
                        out += buf[:len(buf) - keep]
                        buf = buf[len(buf) - keep:]
                        break
                    out += buf[:idx]
                    buf = buf[idx + len(_THINK_OPEN):]
                    thinking = True
            if out:
                yield out
        if buf and not thinking:
            yield buf   # trailing partial-tag lookalike that never completed
    finally:
        aclose = getattr(token_gen, "aclose", None)
        if aclose is not None:
            await aclose()


async def ollama_chat_token_gen(
    messages: list[dict[str, str]],         # [system, ...history..., current user]
    model: str = MODEL,
    on_first_token: Optional[Any] = None,   # callable fired once on first token
):
    """Stream a chat completion, with the fanar think-stripper when applicable."""
    inner = _ollama_chat_stream(messages, model, on_first_token)
    if "fanar" in model.lower():
        inner = strip_think_tokens(inner)
    try:
        async for token in inner:
            yield token
    finally:
        # Propagate close so cancelling TTS tears down the httpx stream.
        aclose = getattr(inner, "aclose", None)
        if aclose is not None:
            await aclose()


async def _ollama_chat_stream(
    messages: list[dict[str, str]],
    model: str = MODEL,
    on_first_token: Optional[Any] = None,
):
    """Raw streaming chat completion from Ollama's /api/chat (carries conversation history)."""
    cfg = get_model_config(model)
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


async def warm_llm(model: str = MODEL) -> None:
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
