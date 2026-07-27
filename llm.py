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

from routing import (
    EGYPTIAN_CARD,
    EGYPTIAN_GRAMMAR_RULE,
    NAJDI_GLOSSARY,
    NAJDI_GRAMMAR_RULE,
    WANTS_ARABIC_RE,
    WANTS_ENGLISH_RE,
    looks_egyptian,
    looks_najdi,
    requested_dialect,
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

MODEL = "qwen3.5:27b"   # the one and only LLM — warmed at startup so the first turn isn't a
                        # cold load, and pinned in VRAM (a second model alongside it would OOM).

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
    "    direct the user to contact the relevant team or service channel instead. "
    "14. When giving advice, suggestions, or directions, address the listener directly — a second-person "
    "    imperative or a general recommendation. NEVER phrase advice as a first-person statement of your "
    "    own intention or want (announcing what YOU plan to do, instead of telling the LISTENER what to "
    "    do). GOOD: 'لازم تسوي كذا' or 'you should do this' (speaks directly to the listener). This "
    "    applies in every language and dialect you reply in. "
    "15. Arabic replies must address the listener with ONE consistent grammatical gender (masculine OR "
    "    feminine second-person forms) from the first word to the last — never switch between masculine "
    "    and feminine endings within the same reply. If you are unsure of the caller's gender, pick either "
    "    form and hold it consistently for the whole response; consistency matters far more than guessing right."
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

def build_turn(text: str, lang: str) -> tuple[str, Optional[str], dict[str, Any]]:
    """Decide this turn's reply-language instruction and TTS language.

    Returns (turn_content, tts_language, route_meta):
      turn_content — the wrapped user message sent to the LLM (instruction + style
        rules + the raw text). Only the CLEAN text is stored in history, so these
        per-turn instructions never accumulate across turns.
      tts_language — gates CATT tashkeel to Fusha (the TTS module re-checks each
        synthesized sentence with the same Najdi detector, so a reply that comes
        back Najdi is never MSA-diacritized regardless of this value).
      route_meta — how the decision was reached, for the interactions.jsonl route
        block (the purity linter's ground truth): requested = explicitly named
        dialect or None; detected = spoken-dialect detection result ("najdi"/None)
        on plain-Arabic turns; explicit_arabic = an explicit Arabic request fired.
    """
    # A named dialect (Najdi/Fusha/Egyptian) counts as an Arabic request on its own —
    # even when "Arabic" isn't said, e.g. "in Najdi Arabic". Other named dialects
    # (Gulf, Hijazi, ...) aren't recognized and fall through to the lang-detected
    # routing below (Fusha, unless dialect markers are present).
    req_name, req_phrase = requested_dialect(text)
    wants_arabic = req_name is not None or bool(WANTS_ARABIC_RE.search(text))
    detected: Optional[str] = None

    if req_name == "Najdi":
        tts_language = "najdi arabic"
    elif req_name == "Egyptian":
        tts_language = "egyptian arabic"
    elif wants_arabic:
        tts_language = "standard arabic"   # Fusha, explicitly named or default
    elif lang == "ar":
        # Spoken-dialect detection. NAJDI FIRST, short-circuit — the frozen invariant
        # (owner decision 2026-07-20): any utterance that routed Najdi before Egyptian
        # existed keeps routing Najdi, including Egyptian speech carrying pan-dialect
        # Najdi markers (اللي/عشان/لسه/يلا) — that recall cap is measured in eval/.
        # looks_egyptian is only ever consulted on turns that previously routed Fusha.
        detected = ("najdi" if looks_najdi(text)
                    else "egyptian" if looks_egyptian(text)
                    else None)
        tts_language = {"najdi": "najdi arabic",
                        "egyptian": "egyptian arabic"}.get(detected, "standard arabic")
    else:
        tts_language = None   # English or mixed AR+EN (mixed deliberately unchanged in v1)
    print(f"  [tts-lang] {tts_language}")
    route_meta: dict[str, Any] = {"requested": req_name, "detected": detected,
                                  "explicit_arabic": wants_arabic}

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
        lang_instruction = (
            "The user is mixing Arabic and English (code-switching). "
            "Reply naturally in the SAME mix of Arabic and English they used. "
            "For the Arabic parts, use Najdi if their Arabic carries Najdi markers, "
            "otherwise use Fusha (Modern Standard Arabic). "
            "Do NOT force a reply into all-Arabic or all-English."
        )
    elif lang == "ar":
        if detected == "najdi":
            lang_instruction = (
                "The user spoke Najdi Arabic. Reply ONLY in the Najdi dialect — "
                "do not switch to Fusha/MSA and do not mix in other dialects."
            )
        elif detected == "egyptian":
            lang_instruction = (
                "The user spoke EGYPTIAN Arabic (Masri). Reply ONLY in natural spoken "
                "Egyptian — do NOT drift to Fusha/MSA mid-reply and never mix in "
                "another dialect."
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
    # Egyptian turns get their own card the same way — and ONLY Egyptian turns:
    # nothing Egyptian may ever be appended on a Najdi turn (pink-elephant result).
    # NAJDI_NO_OTHER_DIALECTS_RULE (باش/ش-negation) was tried and REVERTED — see
    # its docstring in routing.py, it measurably increased the leak it targeted.
    if tts_language == "najdi arabic":
        lang_instruction += "\n" + NAJDI_GLOSSARY + "\n" + NAJDI_GRAMMAR_RULE
    elif tts_language == "egyptian arabic":
        lang_instruction += "\n" + EGYPTIAN_CARD + "\n" + EGYPTIAN_GRAMMAR_RULE

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
        "When you mention a specific street, building, neighborhood, person's name, or historical date "
        "tied to a real place, only include it if you are highly confident it is factually correct. If "
        "you are not, describe it in general terms instead of inventing a specific-sounding name or claim "
        "to fill the gap — a true general answer is better than a precise-sounding wrong one. "
        "No markdown.\n\n"
        f"User: {text}"
    )
    return turn_content, tts_language, route_meta


# ── History policy: clearing at Arabic-dialect boundaries ─────────────────────

def turn_dialect_label(tts_language: Optional[str]) -> str:
    """Coarse per-turn dialect label for history bookkeeping: the routed Arabic dialect,
    or "en" for English/mixed turns (tts_language None)."""
    return {"najdi arabic": "najdi", "egyptian arabic": "egyptian",
            "standard arabic": "fusha"}.get(tts_language or "", "en")


_ARABIC_DIALECTS = {"najdi", "fusha", "egyptian"}


def crosses_dialect_boundary(turn_label: str, history_dialects: list[str]) -> bool:
    """True when this turn crosses from one Arabic dialect to a DIFFERENT one relative to
    the committed history — the trigger for clearing the rolling history (owner decision
    2026-07-20 for Egyptian only, generalized to all Arabic-dialect pairs 2026-07-27).

    English/mixed turns never trigger clearing in either direction — no Arabic dialect to
    protect. Same-dialect repeats never clear. In a conversation using only one Arabic
    dialect (or none) this can never return True.

    Why clearing at all: the old branch's documented live bug — with cross-dialect history
    in context, a "Najdi" answer came back as the earlier Egyptian answer nearly verbatim,
    and in-context Egyptian exemplars were the mechanism behind the pre-cards 67% Najdi leak
    rate. Originally scoped to the Egyptian boundary only for rollout byte-invariance reasons
    (that rollout is done); generalized because the underlying imitation mechanism is not
    Egyptian-specific — Najdi↔Fusha switches carry the same risk. Accepted cost: a follow-up
    like «وش يعني؟» right after ANY Arabic-to-different-Arabic-dialect switch loses its
    referent (now also true for Najdi↔Fusha, not just Egyptian crossings) — plus a higher
    spurious-clear rate specifically for Najdi↔Fusha than Egyptian ever had, since Fusha is
    the default fallback and Najdi recall (82%, eval/dialect_id_eval.py) misses some
    genuinely-Najdi turns to "no marker detected" rather than a real register switch. Accepted
    tradeoff, not a bug — see eval/BASELINES.md's 2026-07-27 entry."""
    if turn_label not in _ARABIC_DIALECTS:
        return False
    return any(d in _ARABIC_DIALECTS and d != turn_label for d in history_dialects)


# ── LLM token generator ───────────────────────────────────────────────────────

async def ollama_chat_token_gen(
    messages: list[dict[str, str]],         # [system, ...history..., current user]
    model: str = MODEL,
    on_first_token: Optional[Any] = None,   # callable fired once on first token
):
    """Stream a chat completion from Ollama's /api/chat (carries conversation history)."""
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
