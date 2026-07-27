# OmniVoice TTS — Download & Setup Guide

Step-by-step reconstruction of how the TTS stack (`k2-fsa/OmniVoice` + CATT
tashkeel) is installed and wired into this project, verified against the
**actual running environment** on `devserver` (not assumed) — package
versions, model cache contents, and library paths below were all read
directly off disk. Companion to [VOICE_PIPELINE.md](VOICE_PIPELINE.md), which
covers how the TTS module *behaves* once it's running; this document covers
how to get it installed in the first place.

---

## 0. What you're installing

Two independent pieces, both driven from `tts_omnivoice_v1.py`:

1. **OmniVoice** (`omnivoice` PyPI package, model ID `k2-fsa/OmniVoice`) — the
   zero-shot voice-cloning TTS engine itself. 24 kHz output.
2. **CATT tashkeel** (`catt-tashkeel` PyPI package) — an Arabic diacritization
   model, applied to Fusha sentences only, before they're handed to OmniVoice.

Neither needs a manual `git clone` — both ship as ordinary pip packages. What
*does* need care is the CUDA/PyTorch stack underneath them, because this
server shares one GPU across OmniVoice + faster-whisper + Ollama, and their
CUDA dependencies don't naturally agree with each other (see §4).

---

## 1. Prerequisites (verified on this server)

| Requirement | Verified value on `devserver` |
|---|---|
| GPU | NVIDIA GeForce **RTX 5090**, 32607 MiB VRAM |
| NVIDIA driver | 580.126.09 (system-level install — outside this repo's scope; a driver supporting CUDA 13.0 must already be installed by the box owner/sysadmin) |
| OS Python | 3.12.3 at `/usr/bin/python3` |
| Package manager | [`uv`](https://docs.astral.sh/uv/) 0.11.16 (`~/.local/bin/uv`) — this project uses `uv`, not raw `pip`, to create and populate the venv |
| ffmpeg | **not installed** system-wide — harmless (see §8), only triggers a `pydub` warning; nothing in this pipeline actually needs it |

`nvcc`/the full CUDA toolkit is **not** installed system-wide either — every
CUDA runtime library the pipeline needs comes down as pip packages alongside
PyTorch and faster-whisper (see §4). You only need the driver at the system
level.

---

## 2. Create the virtual environment

```bash
cd /home/taha/first_voice_test
uv venv .venv --python /usr/bin/python3
source .venv/bin/activate
```

**Known gotcha, hit on this server**: a `uv venv` environment does **not**
include `pip` — `python -m pip` inside it fails with `No module named pip`.
This is expected and fine; use `uv pip install ...` (see §3) instead of
activating and running plain `pip install`. If a `.venv` ever gets into a
broken state, the recovery used here was simply:

```bash
rm -rf .venv
uv venv .venv --python /usr/bin/python3
source .venv/bin/activate
```

— then re-run §3 and §4 below to repopulate it. This does not touch any
model weights cached in `~/.cache/huggingface` (see §5), so re-creating the
venv is cheap even after the first real setup.

---

## 3. Install PyTorch matching CUDA 13.0

This machine's driver/toolchain needs the **cu130** PyTorch build
specifically — not the default `pip install torch` (which resolves to a cu12
build) and not a version mismatch with `torchaudio`. This is exactly why
`requirements.txt` carries torch as a **comment**, not a real pinned line —
it must be installed in this separate, explicit step:

```bash
uv pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
```

Verified installed on this server: `torch 2.11.0+cu130`, `torchaudio
2.11.0+cu130`.

Do **not** let a later `uv pip install -r requirements.txt` implicitly
upgrade/replace these — if you ever see torch silently downgrade to a cu12
build, faster-whisper/OmniVoice will still *load* (they don't hard-require
cu13) but you lose the driver-matched build spec this box was set up for.

---

## 4. Install the rest of the TTS/STT stack

```bash
uv pip install -r requirements.txt
```

The relevant TTS-related lines in `requirements.txt`:

```
lameenc==1.8.2          # PCM → MP3 encoding for the WS audio frames
omnivoice==0.1.5        # TTS (in-process). Pulls transformers>=5.3
catt-tashkeel==1.0.2    # Arabic diacritization (tashkeel), Fusha-only
```

Verified installed versions on this server (via `importlib.metadata`, since
`pip show` isn't available — see the gotcha in §2):

| Package | Version |
|---|---|
| `omnivoice` | 0.1.5 |
| `catt-tashkeel` | 1.0.2 |
| `transformers` | 5.12.1 (pulled in by `omnivoice>=5.3`'s constraint — verified faster-whisper and clearvoice both still run fine on transformers 5.x, per the `requirements.txt` comment) |
| `lameenc` | 1.8.2 |

### 4.1 Where the CUDA runtime libraries actually come from

You never install CUDA itself — `pip`/`uv` pulls the runtime `.so` files as
ordinary packages, as a transitive dependency of torch (cu13 variants) *and*
of faster-whisper's `ctranslate2` backend (cu12 variants — `ctranslate2`
wheels are built against CUDA 12). Both sets end up installed side by side.
Confirmed present in this venv's `site-packages/nvidia/` and top-level
dist-info entries:

- **cu13** (from the torch/torchaudio cu130 wheels in §3): `nvidia_cublas-13.1.0.3`, `nvidia_cuda_nvrtc-13.0.88`, `nvidia_cuda_runtime-13.0.96`, `nvidia_cudnn_cu13-9.19.0.56`, `nvidia_cusparselt_cu13-0.8.0`, `nvidia_nccl_cu13-2.28.9`, `nvidia_nvshmem_cu13-3.4.5`, plus `nvidia_cuda_cupti`/`nvidia_cufft`/`nvidia_cufile`/`nvidia_curand`/`nvidia_cusolver`/`nvidia_cusparse`/`nvidia_nvjitlink`/`nvidia_nvtx`
- **cu12** (pulled in by `ctranslate2`/faster-whisper): `nvidia_cublas_cu12-12.9.2.10`, `nvidia_cuda_nvrtc_cu12-12.9.86`, `nvidia_cudnn_cu12-9.22.0.52`

This mixed cu12+cu13 layout is exactly why `server.py` manually preloads
**both** sets of `.so` files with `RTLD_GLOBAL` at import time (see §7) — if
you only load one CUDA major version's libraries, whichever library loaded
second can't resolve symbols the first one already claimed.

---

## 5. The OmniVoice model weights — first-run download

The model is **not** bundled in the pip package — it's pulled from the
Hugging Face Hub the first time `OmniVoice.from_pretrained("k2-fsa/OmniVoice", ...)`
runs (in `tts_omnivoice_v1._get_model()`), and cached locally after that.

Verified on this server: `~/.cache/huggingface/hub/models--k2-fsa--OmniVoice`
— **3.1 GB**. OmniVoice also pulls its own vocoder dependency
(`charactr/vocos-mel-24khz` was observed alongside it in the same HF cache),
so the very first load can take a few minutes on a fresh box depending on
network speed; every load after that is local-disk-cache fast (~15 s per
`test_local.py`'s own comment, mostly GPU weight transfer, not download).

- No Hugging Face account/token is required for this model (no gated-repo
  prompt was hit setting this up).
- If the server has no internet egress to `huggingface.co`, pre-warm the
  cache from a machine that does and copy `~/.cache/huggingface/hub/models--k2-fsa--OmniVoice`
  (and the vocos-mel-24khz folder) over, or set `HF_HUB_OFFLINE=1` once cached.
- `OMNIVOICE_MODEL` env var overrides the model ID if you ever need a
  different OmniVoice checkpoint (default `k2-fsa/OmniVoice`).
- `OMNIVOICE_DEVICE` env var overrides the device placement (default
  `cuda:0`).

---

## 6. CATT tashkeel — no download step needed

Unlike OmniVoice, `catt-tashkeel`'s ONNX models ship **inside the pip
package itself** — verified: `site-packages/catt_tashkeel/onnx_models/{eo_model,ed_model}`,
163 MB total. `uv pip install catt-tashkeel==1.0.2` is the entire setup —
nothing to download at runtime, no HF cache entry for it.

- `CATT_ENABLED` env var (default `1`) gates it on/off globally; loaded
  lazily via `_get_tashkeel_model()` on first Fusha sentence, same
  lazy-singleton-with-lock pattern as the OmniVoice model itself.
- If you want to skip it entirely (e.g. to A/B whether diacritization is
  worth the extra inference step per Fusha sentence), `CATT_ENABLED=0
  bash start_server.sh` — no reinstall needed.

---

## 7. The reference voice clip (voice-cloning input)

OmniVoice is zero-shot: it clones whatever voice you hand it as a short
reference clip + that clip's exact transcript. This project's reference:

- File: `voices/silma-tts-saudi-24k.wav`
- Verified format: **24000 Hz, mono, 7.64 s**
- Exact transcript (hardcoded in `tts_omnivoice_v1.py` as `_REF_TEXT`, must
  match the clip word-for-word): *"الثقافة السعودية فيها عراقة وتاريخ عميق، وقيم إسلامية راسخة، وعادات وتقاليد قبلية أصيلة متوارثة."*

**To swap in a different voice**, both of these must change together:
1. Record/obtain a clean mono WAV, ideally already at 24 kHz (matching
   OmniVoice's native output rate avoids an implicit resample) — a few
   seconds of clear, natural speech is enough.
2. Get its *exact* transcript (the literal words spoken, not a paraphrase).
3. Update `_REF_AUDIO` (the file path) and `_REF_TEXT` (the transcript) at
   the top of `tts_omnivoice_v1.py` together — a mismatched transcript
   measurably degrades clone quality since OmniVoice uses it to align audio
   to text internally.

The `VoiceClonePrompt` built from this pair (`model.create_voice_clone_prompt(ref_audio, ref_text)`)
is constructed **once** at model load and reused for every sentence — see
`VOICE_PIPELINE.md` §6.1 for why (re-tokenizing the reference per sentence
was measured to add needless first-audio latency).

---

## 8. Sanity-check the install

### 8.1 Direct pipeline smoke test (no browser, no mic)

```bash
/home/taha/first_voice_test/.venv/bin/python test_local.py
```

This sends a hardcoded Arabic prompt to Ollama, streams the response through
the real `tts_omnivoice_v1.stream_tts_to_ws`, and writes one MP3 per sentence
to `./test_output/`. Requires Ollama already running with `qwen3.5:27b`
pulled (see `CLAUDE.md` / `start_server.sh` — this test script does **not**
start Ollama itself). Expect ~15 s for the first OmniVoice load, then
per-sentence MP3s appearing with timing printed to stdout. Pull the files
locally to listen:

```bash
scp taha@devserver:/home/taha/first_voice_test/test_output/*.mp3 ~/Desktop/
```

### 8.2 Full pipeline (browser)

```bash
bash /home/taha/first_voice_test/start_server.sh
```

then open `http://<server>:8765/`. The client shows "جاري تحميل النماذج..."
(loading models) until `tts_omnivoice_v1.load_models()` (OmniVoice + CATT)
and `stt.load_models_blocking()` (VAD + Whisper + FRCRN) both finish and the
LLM warm-up completes — see `VOICE_PIPELINE.md` §2.1.

### 8.3 Known harmless warning

```
Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work
```

Comes from `pydub` (a transitive dependency, imported somewhere in the
OmniVoice/CATT chain). Does not affect this pipeline — audio I/O here goes
through `soundfile`/`lameenc` directly, not `pydub`/`ffmpeg`. Safe to ignore;
install `ffmpeg` via the system package manager only if some *other* tool you
add later actually needs it.

---

## 9. Runtime wiring you should know about (not a setup step, but explains why §4 matters)

`server.py` does two things at import time that only make sense in light of
the mixed cu12/cu13 library layout from §4.1:

1. **Manual CUDA lib preload with `RTLD_GLOBAL`** — before anything else
   imports torch/torchaudio/faster-whisper, it does:
   ```python
   ctypes.CDLL(f"{NVIDIA}/cu13/lib/libnvrtc.so.13", mode=ctypes.RTLD_GLOBAL)
   ctypes.CDLL(f"{NVIDIA}/cu13/lib/libcublas.so.13", mode=ctypes.RTLD_GLOBAL)
   ctypes.CDLL(f"{NVIDIA}/cublas/lib/libcublas.so.12", mode=ctypes.RTLD_GLOBAL)
   ctypes.CDLL(f"{NVIDIA}/cudnn/lib/libcudnn.so.9", mode=ctypes.RTLD_GLOBAL)
   ctypes.CDLL(f"{NVIDIA}/cuda_nvrtc/lib/libnvrtc.so.12", mode=ctypes.RTLD_GLOBAL)
   ```
   `RTLD_GLOBAL` (not the default `RTLD_LOCAL`) makes these symbols visible
   to `.so` files loaded *later* by other packages — needed because
   OmniVoice, faster-whisper/ctranslate2, and torch itself each bundle their
   own copies of overlapping CUDA libs, and load order otherwise determines
   which copy "wins," unpredictably.
2. **`torchaudio.load` monkey-patched to use `soundfile`** — torchaudio 2.11
   routes `.load()` through `torchcodec` by default, which needs CUDA NPP
   libraries not present on this box. The patch swaps in a `soundfile`-based
   loader for WAV/FLAC (zero GPU dependency) — this is what actually reads
   `voices/silma-tts-saudi-24k.wav` at startup.

`start_server.sh` sets the equivalent `LD_LIBRARY_PATH` for the Ollama
process and the Python process before `exec`, pointing at
`.venv/lib/python3.12/site-packages/nvidia/{cu13,cublas,cudnn,cuda_nvrtc}/lib`
plus `/usr/lib/x86_64-linux-gnu` and Ollama's own bundled lib dir.

---

## 10. Historical note — standalone microservice prototype (not current architecture)

Before OmniVoice was folded into `server.py` as an in-process module
(current design — see `VOICE_PIPELINE.md` §6), it was first prototyped as a
**separate HTTP microservice** in a scratch `tts_experiments/` directory
(untracked in git, not present on disk anymore): a dedicated
`.venv-omnivoice` venv, a `setup_omnivoice.sh` install script, and a
`service_omnivoice` process on port 8771 that the main server would call
over HTTP. That approach was abandoned in favor of loading OmniVoice
directly inside `server.py`'s own process (see commit `09772ca`, "Switch TTS
to OmniVoice with full pipeline fixes") — avoiding an HTTP hop per sentence
measurably helped first-audio latency, at the cost of both processes now
sharing one GPU's VRAM directly (which is why the VRAM-budget tuning
throughout `stt.py`/`llm.py`/`tts_omnivoice_v1.py` — int8_float16 Whisper,
`LLM_NUM_CTX=8192`, `torch.cuda.empty_cache()` after every turn — exists at
all). Mentioned here only so the `tts_omnivoice_service`/`setup_omnivoice.sh`
references you may see in old shell history make sense — there is nothing to
recreate; the current single-process design in `tts_omnivoice_v1.py` is the
one to build on.

---

## 11. Full command sequence (copy-paste order)

```bash
# 1. venv
cd /home/taha/first_voice_test
uv venv .venv --python /usr/bin/python3
source .venv/bin/activate

# 2. PyTorch matched to this box's CUDA 13.0 driver
uv pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130

# 3. Everything else (STT + TTS + web server)
uv pip install -r requirements.txt

# 4. Place the reference voice clip (already committed in this repo)
ls voices/silma-tts-saudi-24k.wav

# 5. First real run — this is what triggers the ~3.1GB OmniVoice model
#    download from Hugging Face Hub (one-time; cached after)
bash start_server.sh
# or, for a no-browser smoke test instead:
# .venv/bin/python test_local.py
```

---

## Sources

- [Previous PyTorch Versions](https://pytorch.org/get-started/previous-versions/) — confirms the `--index-url https://download.pytorch.org/whl/cu130` pattern for CUDA-13.0-matched wheel builds
