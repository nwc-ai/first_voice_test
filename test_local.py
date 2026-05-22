"""
test_local.py — Full pipeline test for tts_silma_v1.py
=====================================================
No microphone needed. This script:
  1. Sends a hardcoded question to Ollama (qwen2.5:7b) as the LLM
  2. Streams the LLM tokens through tts_silma_v1.py
  3. Saves each synthesized sentence as an MP3 file in ./test_output/
  4. Prints timing info so you can see how fast each step is

Run with:
    /home/taha/first_voice_test/.venv/bin/python test_local.py
"""

import asyncio
import os
import sys
import time

import httpx

# Add project root to path so we can import tts_silma_v1
sys.path.insert(0, os.path.dirname(__file__))
import tts_silma_v1

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"

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


# ── Ollama token generator ────────────────────────────────────────────────────────────────

async def ollama_token_gen(prompt: str):
    """Stream tokens from Ollama one at a time."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", OLLAMA_URL, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                import json
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break


# ── Main test ─────────────────────────────────────────────────────────────────────────────

async def run_test():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("FULL PIPELINE TEST")
    print("=" * 60)
    print(f"Question: {TEST_PROMPT}")
    print()

    # Step 1: Warm up the Silma model (loads it into GPU memory)
    print("Loading Silma TTS model into GPU... (this takes ~15 seconds the first time)")
    t0 = time.perf_counter()
    await asyncio.to_thread(tts_silma_v1.load_models)
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

    await tts_silma_v1.stream_tts_to_ws(
        token_gen=ollama_token_gen(TEST_PROMPT),
        ws=ws,
        cancel_event=cancel_event,
        on_first_audio=on_first_audio,
    )

    total_time = time.perf_counter() - t_start

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Time to first audio:  {first_audio_time:.1f}s" if first_audio_time else "Time to first audio:  (none)")
    print(f"Total time:           {total_time:.1f}s")
    print(f"MP3 files saved:      {ws.mp3_count}")
    print(f"Total audio bytes:    {ws.total_bytes:,}")
    print(f"Output directory:     {OUTPUT_DIR}")
    print()
    if ws.mp3_count > 0:
        print("To listen to the audio, copy the MP3 files to your local machine:")
        print(f"  scp taha@devserver:{OUTPUT_DIR}/*.mp3 ~/Desktop/")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_test())
