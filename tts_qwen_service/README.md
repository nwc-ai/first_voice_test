# Qwen3-TTS-KSA service (port 8772)

A standalone TTS microservice in its **own venv** (isolated deps — `qwen-tts` pins
`transformers==4.57.3`, which conflicts with the main pipeline). The main
`first_voice_test` server calls it over HTTP when you pick **qwen** in the TTS dropdown.

## One-time setup
```bash
bash /home/taha/tts_qwen_service/setup.sh
```
Creates `.venv`, installs torch (cu130, for the RTX 5090) + `qwen-tts==0.1.1` + FastAPI,
and sanity-checks the imports.

## Run
```bash
bash /home/taha/tts_qwen_service/start.sh
```
Loads `vadimbelsky/qwen3-TTS-KSA` (~5–6 GB VRAM) and serves on `:8772`. Wait for
`[qwen-tts] ready in N.Ns`.

## Test the service alone
```bash
curl -X POST localhost:8772/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"مرحبا، كيف حالك اليوم؟","lang":"ar"}' --output t.wav
# play t.wav — confirms synthesis works
curl localhost:8772/health   # {"ready": true}
```

## Use it live (the real test)
1. Start this service (above).
2. Start the main pipeline: `bash /home/taha/first_voice_test/start_server.sh`.
3. Hard-refresh the browser; the **TTS** dropdown now shows `qwen` (only appears when
   this service answers `/health`). Pick `qwen`, click 🎙️, speak → you should HEAR Qwen.
   If silent, the main server log prints the exact `[tts]` error (no silent failures).

## VRAM note (32 GB RTX 5090)
qwen3.5:27b (~19 GB) + main pipeline (~5 GB) + this service (~5–6 GB) is **tight**.
If you hit CUDA OOM, switch the LLM dropdown to **ALLaM-7B** (frees ~15 GB) while testing Qwen TTS,
or stop the main pipeline's Silma by testing one engine at a time. Watch `nvidia-smi`.

## Troubleshooting
- **torch import fails / `cuda False`**: the cu130 wheel version differs on this box. Match how the
  main venv installed `torch==2.11.0+cu130`, then re-run the `qwen-tts` install line in `setup.sh`.
- **`sox` / libsox error on import**: `uv pip install --python .venv/bin/python sox` already pulled by
  qwen-tts; if it complains about the binary, it's only needed for some codepaths — synthesis via
  `generate_voice_clone` does not require the sox CLI.
