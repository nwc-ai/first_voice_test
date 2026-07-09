# SETUP — standing up first_voice_test on a machine

This is the manual runbook to get the voice assistant running. It's reconstructed from `start_server.sh`,
`requirements.txt`, and the code — there is no one-shot installer. Read **CLAUDE.md** (overview) and
**ARCHITECTURE.md** (how it all works) alongside this.

> ⚠️ **Paths are hardcoded.** `start_server.sh` references `/home/taha/first_voice_test` and
> `/home/taha/.ollama_bin_new`. If your username/path differ, edit those (see §6). Everything in
> `server.py`/`tts_omnivoice_v1.py` uses paths relative to the file, so only `start_server.sh` needs editing.

---

## 1. Prerequisites (hardware & system)
- **NVIDIA GPU with ~24 GB+ free VRAM** — developed on an **RTX 5090 (32 GB)**. The full stack (qwen3.5:27b
  ~15.7 GB + OmniVoice ~2.4 GB + Whisper large-v3 ~3 GB + FRCRN/VAD ~0.5 GB) needs a big card.
- **CUDA 13.0** drivers (torch is the `+cu130` build). A different CUDA version means a different torch build.
- **Linux**, Python **3.12**.
- **~20 GB disk** for the venv + model downloads (HF cache).
- No sudo / no Docker assumed (that's why it uses a uv-venv, not system packages).
- ffmpeg is **not** required (a pydub warning about it is harmless).

---

## 2. Get the code + the voice clips
The repo's **reference voice clips are mandatory** — `load_models()` aborts at boot if any registry clip is
missing. Both active clips are committed in git under `voices/`:
- `voices/silma-tts-saudi-24k.wav` — Saudi default voice (Najdi/Fusha/English)
- `voices/omnivoice-tts-egyptian-24k-v3.wav` — active Egyptian voice

(The superseded Egyptian v1/v2 clips were deleted from the tree on 2026-07-06 — recover from git history
if ever needed.)

---

## 3. Python environment (uv venv + CUDA-13 torch)
The project uses a **uv-managed venv** (it has no `pip`; use `uv pip`). torch is installed **separately** from
`requirements.txt` because it must match CUDA 13.

```bash
cd <project>                                   # the first_voice_test folder
uv venv .venv --python 3.12                    # venv MUST live at <project>/.venv (the code preloads
                                               # .venv/lib/python3.12/site-packages/nvidia/* CUDA libs)

# 1) PyTorch built for CUDA 13 (pulls the nvidia-*-cu13 libs the server preloads). Verify the index/version
#    for your environment; this matches the dev machine:
uv pip install --python .venv/bin/python torch==2.11.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130

# 2) The rest of the deps:
uv pip install --python .venv/bin/python -r requirements.txt
```
`requirements.txt` = fastapi, uvicorn, httpx, numpy, soundfile, lameenc, faster-whisper==1.1.1,
clearvoice==0.1.2, omnivoice==0.1.5 (the last pulls `transformers>=5.3`).

Sanity check:
```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.11.0+cu130 True
.venv/bin/python -m py_compile server.py tts_omnivoice_v1.py && echo OK
```

---

## 4. Ollama + the LLM
The LLM runs in a **separate Ollama process** (the server only talks to it over HTTP at `localhost:11434`).
- Install Ollama (the dev box used a custom build at `~/.ollama_bin_new`; any Ollama ≥ 0.30 with CUDA-13
  support works). `start_server.sh` launches it with `OLLAMA_FLASH_ATTENTION=1` + `OLLAMA_KV_CACHE_TYPE=q8_0`.
- Make the model **`qwen3.5:27b`** available (the exact tag the server requests):
  ```bash
  ollama list            # confirm qwen3.5:27b is present
  # if not, pull/create it under that exact tag (source depends on how you obtain qwen3.5:27b)
  ```
  > The server is **hard-locked** to `qwen3.5:27b` ([server.py](server.py) `MODEL`); a different model needs a
  > one-line change there.

---

## 5. The other models (auto-download on first run)
No manual step — these download to the HF cache / torch hub the first time the server loads (needs internet):
- **OmniVoice** `k2-fsa/OmniVoice` (HF) — the TTS.
- **faster-whisper large-v3** (HF) — STT.
- **Silero VAD** via `torch.hub` (`snakers4/silero-vad`).
- **ClearVoice FRCRN_SE_16K** — the denoiser; ClearVoice fetches its weights (the repo's `checkpoints/` is
  gitignored, so a clone re-downloads them). If FRCRN fails to load, the server logs it and **skips denoising**
  (non-fatal).

First startup therefore takes several minutes (downloads + warm-up). The browser shows a "loading models" screen.

---

## 6. Fix the hardcoded paths (only if your path/user differ)
Edit **`start_server.sh`**:
- `VENV=/home/taha/first_voice_test/.venv/lib/python3.12/site-packages/nvidia` → your project's `.venv` path.
- `OLLAMA_NEW=/home/taha/.ollama_bin_new` → wherever your `ollama` binary + libs live (or just use a
  system-installed `ollama` and simplify the launch line).
- The final `exec .../.venv/bin/python .../server.py` → your project path.

---

## 7. Run it
```bash
bash <project>/start_server.sh
```
This starts Ollama (if not already up), then the FastAPI server on **port 8765**. Wait for
`All models loaded — server ready.`

**Access (the dev setup uses an SSH tunnel):** from your laptop —
```bash
ssh -L 8765:localhost:8765 \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ExitOnForwardFailure=yes \
    <user>@<host>
```
then open **http://localhost:8765**, click 🎙️, and speak. (Mic needs a secure context — `localhost` over the
tunnel works; a raw remote IP over plain HTTP will be blocked by the browser. Use the tunnel, or put it behind HTTPS.)

---

## 8. Verify it works
- Browser: button turns green ("يستمع…"); speak Arabic → you see a transcript, a streamed reply, and hear audio.
- Detected Egyptian → Egyptian voice + Egyptian words; a clearly-Najdi sentence → Saudi voice + Najdi words; unclear/no-marker Arabic → **Fusha** (Saudi voice, MSA); English → English.
- **Dashboard:** open `http://localhost:8765/review` — per-turn latency + transcript/response table (from `logs/interactions.jsonl`).
- **No-mic smoke test** (TTS+LLM only): `.venv/bin/python test_local.py` → writes MP3s to `test_output/`.

---

## 9. Useful env vars
| Var | Default | Effect |
|---|---|---|
| `LLM_NUM_CTX` | `8192` | qwen3.5 context / KV-cache size (used by warm-up *and* chat — keep them equal) |
| `OMNIVOICE_MODEL` | `k2-fsa/OmniVoice` | TTS model id |
| `OMNIVOICE_DEVICE` | `cuda:0` | TTS device |
| `FRCRN_ENABLED` | `0` (**off**) | `1` loads the FRCRN denoiser again (A/B only — evidence says enhancement hurts Whisper) |
| `SAVE_UTTERANCES` | `0` | `1` saves each accepted utterance's raw audio + manifest row to `logs/utterances/` (eval ground truth) |
| `OLLAMA_FLASH_ATTENTION` / `OLLAMA_KV_CACHE_TYPE` | `1` / `q8_0` | set on a fresh `ollama serve` (halve KV VRAM) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | reduce CUDA fragmentation |

Run with a bigger context, e.g.: `LLM_NUM_CTX=16384 bash start_server.sh`.

---

## 10. Troubleshooting
- **`OmniVoice reference audio for voice 'egyptian' not found`** at boot → the active Egyptian clip is
  missing. Put `voices/omnivoice-tts-egyptian-24k-v3.wav` in place (the registry's active Egyptian voice).
- **Browser WS keeps closing (`1006`/`1005`)** while idle → it's the **SSH tunnel** dropping, not the app;
  use the keepalive flags in §7. The browser auto-reconnects.
- **CUDA / `libcublas` / torchcodec import errors** → torch isn't the `+cu130` build, or the venv isn't at
  `<project>/.venv` (the server preloads `.venv/lib/python3.12/site-packages/nvidia/*` and patches
  `torchaudio.load`). Re-do §3.
- **VRAM OOM** → confirm only this stack is on the GPU; FRCRN denoise is auto-skipped for clips > 4 s and when
  free VRAM is low. The 27B + OmniVoice + Whisper must fit on one card.
- **First turn very slow / empty** → models still warming up, or Ollama doesn't have `qwen3.5:27b` loaded;
  check the server log and `curl localhost:11434/api/tags`.

---

## 11. What is NOT in the repo
- The `.venv` and all model weights (downloaded per §3–§5).
- The conversation/decision history — but the *decisions* are captured in CLAUDE.md ("Key decisions") and
  ARCHITECTURE.md (§13 known issues).
- Per-machine Claude memory notes (they live in `~/.claude`, not the folder) — their content is mirrored into
  CLAUDE.md / ARCHITECTURE.md so a fresh session understands the system from the repo alone.
