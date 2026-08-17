"""
test_local.py — Full pipeline test for tts_omnivoice_v1.py
==========================================================
No microphone needed. This script:
  1. Sends a hardcoded question through the REAL llm.py path (llm.build_turn +
     llm.ollama_chat_token_gen) — same model, same config, same message shape
     server.py's respond_loop actually uses, so it inherits LLM_MODEL_OVERRIDE
     for free and exercises the true streaming /api/chat path (not a hand-rolled
     /api/generate call) — see eval/BASELINES.md's 2026-07-27 Fanar-2 live-test entry.
  2. Streams the LLM tokens through tts_omnivoice_v1.py
  3. Saves each synthesized sentence as an MP3 file in ./test_output/
  4. Prints timing info so you can see how fast each step is

Run with:
    /home/taha/first_voice_test/.venv/bin/python test_local.py
    LLM_MODEL_OVERRIDE=<ollama tag> /home/taha/first_voice_test/.venv/bin/python test_local.py
"""

import asyncio
import os
import sys
import time

# Add project root to path so we can import tts_omnivoice_v1 / llm / routing
sys.path.insert(0, os.path.dirname(__file__))
import llm
import routing
import tts_omnivoice_v1

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")

# The question we "ask" the AI — simulates what a user would say
TEST_PROMPT = "اشرح لي ما هو الذكاء الاصطناعي في جملتين بالعربية"
# Translation: "Explain to me what artificial intelligence is in two sentences in Arabic"


# ── Mock WebSocket ────────────────────────────────────────────────────────────────────────
# The real WebSocket is a browser connection. Here we simulate it:
#   send_json → prints to console
#   send_bytes → saves MP3 to file

class MockWebSocket:
    def __init__(self):
        self.mp3_count = 0
        self.total_bytes = 0
        self.tokens = []

    async def send_json(self, data):
        if data.get("event") == "token":
            text = data["text"]
            self.tokens.append(text)
            print(text, end="", flush=True)
        elif data.get("event") == "tts_end":
            print("\n[tts_end received]")

    async def send_bytes(self, data: bytes):
        self.mp3_count += 1
        self.total_bytes += len(data)
        filename = os.path.join(OUTPUT_DIR, f"sentence_{self.mp3_count:03d}.mp3")
        with open(filename, "wb") as f:
            f.write(data)
        print(f"\n  [MP3 saved: {filename} — {len(data):,} bytes]")


# ── LLM token generator (mirrors server.py's respond_loop exactly) ────────────────────────
# No hand-rolled /api/generate payload here anymore — this calls the same llm.py functions
# the live server uses, so it's model-aware (LLM_MODEL_OVERRIDE) and exercises the real
# streaming /api/chat path, not a separate one-off code path.

_full_response_chars: list[str] = []   # collected across the run for the <think> leak check


async def ollama_token_gen(prompt: str):
    turn_content, tts_language, route_meta = llm.build_turn(prompt, "ar")
    print(f"  [route] {route_meta}  tts_language={tts_language}")
    messages = [
        {"role": "system", "content": llm.SYSTEM_PROMPT},
        {"role": "user", "content": turn_content},
    ]
    async for tok in routing.filter_cjk(llm.ollama_chat_token_gen(messages, llm.MODEL)):
        _full_response_chars.append(tok)
        yield tok


# ── Main test ─────────────────────────────────────────────────────────────────────────────

async def run_test():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("FULL PIPELINE TEST")
    print("=" * 60)
    print(f"Model:    {llm.MODEL}")
    print(f"Question: {TEST_PROMPT}")
    print()

    # Step 1: Warm up the OmniVoice model (loads it into GPU memory)
    print("Loading OmniVoice TTS model into GPU... (this takes ~15 seconds the first time)")
    t0 = time.perf_counter()
    await asyncio.to_thread(tts_omnivoice_v1.load_models)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s\n")

    # Step 2: Run the full pipeline
    ws = MockWebSocket()
    cancel_event = asyncio.Event()

    first_audio_time = None
    t_start = time.perf_counter()

    def on_first_audio_sync():
        nonlocal first_audio_time
        first_audio_time = time.perf_counter() - t_start

    async def on_first_audio():
        on_first_audio_sync()

    print("LLM response (streaming):")
    print("-" * 40)

    await tts_omnivoice_v1.stream_tts_to_ws(  # type: ignore[no-untyped-call]
        token_gen=ollama_token_gen(TEST_PROMPT),
        ws=ws,
        cancel_event=cancel_event,
        on_first_audio=on_first_audio,
    )

    total_time = time.perf_counter() - t_start

    # The offline eval (eval/dialect_eval_full.py) only ever checked the final NON-streamed
    # response for a leftover <think> tag. The live path is stream:True, read token-by-token
    # (llm.ollama_chat_token_gen) — whether Ollama could ever leak literal <think>/</think>
    # tag characters into the streamed content deltas themselves was never actually observed
    # either way. Check it here, on the real streamed text, before trusting a live run.
    full_response = "".join(_full_response_chars)
    think_leak = "<think" in full_response

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Time to first audio:  {first_audio_time:.1f}s" if first_audio_time else "Time to first audio:  (none)")
    print(f"Total time:           {total_time:.1f}s")
    print(f"MP3 files saved:      {ws.mp3_count}")
    print(f"Total audio bytes:    {ws.total_bytes:,}")
    print(f"Output directory:     {OUTPUT_DIR}")
    print(f"<think> leak in streamed text: {'YES — FAIL' if think_leak else 'no'}")
    print()
    if ws.mp3_count > 0:
        print("To listen to the audio, copy the MP3 files to your local machine:")
        print(f"  scp taha@devserver:{OUTPUT_DIR}/*.mp3 ~/Desktop/")
    print("=" * 60)

    assert not think_leak, (
        f"Literal '<think' substring found in the streamed response text — "
        f"the model/GGUF template is leaking reasoning-tag characters into the live "
        f"token stream. Full response: {full_response!r}"
    )


if __name__ == "__main__":
    asyncio.run(run_test())
