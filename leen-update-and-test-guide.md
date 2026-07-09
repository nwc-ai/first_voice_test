# Update & Test Guide — new version (commit `23663f6`, July 2026)

Hi Leen 👋 — this guide updates your copy of the voice assistant on the dev server to the
latest version and walks you through testing it. Your Python environment and models from June
**still work — nothing needs reinstalling.** The whole update is a `git pull` plus re-entering
your paths, then a ~15-minute test.

**Time needed:** ~10 min update + ~15 min testing.

---

## Step 0 — Coordinate the GPU (important!)

The server's GPU (32 GB) can only fit **one** running assistant at a time. Before you start:

```bash
nvidia-smi
```

- If you see a `python .../server.py` process, **Taha's assistant is running — ping him
  first** so he shuts it down. Only one assistant can run at a time.
- The **`ollama` process (~16 GB) is SHARED — leave it running.** There is one Ollama for the
  whole machine (Taha's); your start script detects and reuses it automatically. You do NOT
  need your own Ollama install. Never kill the `ollama` process — only your own `server.py`.
- If `curl -s localhost:11434/api/tags` returns nothing (Ollama down, e.g. after a reboot),
  ask Taha to start it — don't try to start your own.

---

## Step 1 — Get the new code

Your local copy has two kinds of local changes that would block the update, so we clear them
first (they are safe to discard — your path edits get re-entered in Step 2, and the log file
just contains your old test conversations):

```bash
cd ~/first_voice_test

# 1. Discard your local copies of files the update replaces:
git checkout -- start_server.sh logs/interactions.jsonl

# 2. Get the new version (branch: omnivoice-tts):
git fetch origin
git checkout omnivoice-tts
git pull origin omnivoice-tts
```

You should now see commit `23663f6` at the top of `git log --oneline -1`.

---

## Step 2 — Re-enter YOUR paths in `start_server.sh`

The script hardcodes Taha's home directory. Edit these lines (same edit you did in June —
replace `/home/taha` with your own home, and point `OLLAMA_NEW` at whatever worked for you
in June):

| Line | What it is |
|---|---|
| 14 | `VENV=/home/taha/first_voice_test/.venv/...` → your venv path |
| 15, 32 | Ollama paths — **leave them as-is.** They're only used to START Ollama, and you use Taha's shared Ollama (see Step 0); your script will find it already running and skip these lines. |
| 50 | the final `exec .../python .../server.py` → your paths |

That's the **only** file you need to touch. Everything else in the code uses relative paths.

> Don't commit this edit — it's your personal copy. If a future `git pull` complains again,
> repeat Step 1.1 + Step 2.

---

## Step 3 — Start it

```bash
bash ~/first_voice_test/start_server.sh
```

Wait for: `All models loaded — server ready.` (first start after the update may re-verify
model files for a minute; no new downloads are expected).

Then from **your laptop**, open your SSH tunnel and the UI as usual:

```bash
ssh -L 8765:localhost:8765 <you>@<devserver>
```

Open **http://localhost:8765** and press **Ctrl+Shift+R** (hard refresh — the browser page
changed in this update; a cached old page will misbehave).

> Note: only ONE browser tab can be connected — opening a second tab disconnects the first.
> That's by design, not a bug.

---

## What changed since June (so you know what "correct" looks like)

1. **Fusha (فصحى) is now the default** when you ask for Arabic without naming a dialect.
2. **Per-dialect pronunciation is pinned** (`language=` ID) — no more Saudi/Egyptian word-mixing
   in one sentence, back when Egyptian was still a target dialect (removed 2026-07-09 — see #6 below).
3. **Dialect quality guards**: the assistant should never say جداً in a dialect reply
   (auto-corrected to أوي/مرة), never narrate its own rules, and never copy an
   earlier answer's dialect into a new one.
4. **Text appears in sentence chunks**, not word-by-word. **This is expected** — the text and
   the voice are now guaranteed identical.
5. **Barge-in is smarter**: background noise/voices while it's speaking now cause a ~1-second
   *pause then resume*, instead of killing the reply. A real interruption (you speaking
   clearly into the mic) still stops it.
6. **"Saudi dialect" works** (treated as Najdi). **Gulf/Khaleeji, Hijazi, and Egyptian were all
   removed** — asking for any of them gets a polite Fusha reply naming the supported dialects
   (Najdi and Fusha only, now).
7. **New dashboard**: http://localhost:8765/review shows every turn with its dialect routing
   and timings.

---

## Step 4 — Test checklist (~15 min, speak into your mic)

Go through these in order. For each: ✅ = behaves as described. Note the time of anything odd.

| # | Say this | Expect |
|---|---|---|
| 1 | "Hey, how are you?" | English reply, natural |
| 2 | "Tell me about the history of coffee **in Arabic**." | Fusha reply, Saudi voice |
| 3 | Same question "**in Egyptian dialect**" | Egyptian was removed (2026-07-09) — Fusha answer that starts by naming the supported dialects, same as Gulf/Hijazi below |
| 4 | Same question "**in Najdi dialect**" | Saudi voice; NO Egyptian words (no ده/دي/مش), NO جداً |
| 5 | Same question "**in Hijazi dialect**" | Hijazi was removed (2026-07-09) — Fusha answer that starts by naming the supported dialects, same as Gulf below |
| 6 | Same question "**in Saudi dialect**" | Behaves exactly like Najdi (new) |
| 7 | Anything "**in Gulf dialect**" | Fusha answer that starts by naming the supported dialects (new) |
| 8 | Speak Arabic directly: «وش أفضل مطعم بالرياض؟» | Najdi-flavored reply |
| 9 | While it's speaking: **interrupt with a real question** | It stops and answers your new question |
| 10 | While it's speaking: have someone talk nearby / play noise (don't speak into the mic yourself) | Audio pauses ~1 s, then **resumes** the same reply (new — the old version killed it) |
| 11 | Watch the text box during any reply | Text arrives in sentence chunks, matching the voice exactly |
| 12 | Open http://localhost:8765/review | Table of your turns with a Dialect column |

---

## If something goes wrong

- **Server won't start / CUDA out of memory** → almost always Taha's assistant is still
  running. `nvidia-smi`, coordinate, retry.
- **Replies fail with LLM/connection errors** → Ollama is down
  (`curl -s localhost:11434/api/tags` returns nothing). Ask Taha to start it.
- **Port 8765 already in use** → an old server of yours is still alive:
  `pkill -u $(whoami) -f server.py` then start again.
- **Browser connects then drops with code 1005/1006** → your SSH tunnel dropped; the page
  auto-reconnects. Re-open the tunnel if it persists.
- **UI looks broken / buttons do nothing** → you skipped the hard refresh (Ctrl+Shift+R).
- **A reply sounded wrong** (wrong dialect, weird words, audio cut) → note the **time** and
  what you said, and tell Taha — every turn is recorded with full diagnostics in
  `logs/interactions.jsonl` on your account, so the exact turn can be pulled up and analyzed.


