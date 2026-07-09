"""
stt_eval.py — per-dialect WER + language-ID accuracy through the PRODUCTION STT path (GPU)
===========================================================================================
Feeds labeled audio through server._transcribe_blocking — the exact production configuration
(two-pass language logic, hotwords, gates, beam size) — and reports WER per dialect plus
en/ar language-ID accuracy.

Manifest: JSONL rows of
    {"audio": "/path/clip.wav", "text": "reference transcript", "dialect": "Najdi", "lang": "ar"}
dialect ∈ Najdi|Egyptian|MSA|null; lang ∈ ar|en|mixed.
SAVE_UTTERANCES=1 on the server writes logs/utterances/manifest.jsonl in this format —
correct its `text` fields by hand and fill `dialect` before trusting the numbers.

Run (needs the GPU — Whisper is loaded fresh, ~3 GB; fine alongside the idle server):
    .venv/bin/python eval/stt_eval.py logs/utterances/manifest.jsonl

WER uses standard Arabic normalization (diacritics stripped, alef/hamza variants folded,
tatweel removed, punctuation dropped) so spelling variance doesn't count as errors.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np       # noqa: E402
import soundfile as sf   # noqa: E402
import server            # noqa: E402

# ── Arabic-aware normalization ────────────────────────────────────────────────────────────
_DIACRITICS = re.compile(r"[ً-ٰٟـ]")          # harakat + tatweel
_PUNCT      = re.compile(r"[^\w\s؀-ۿ]|_", re.UNICODE)

def normalize(text: str) -> list[str]:
    text = _DIACRITICS.sub("", text)
    text = (text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
                .replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي"))
    text = _PUNCT.sub(" ", text.lower())
    return text.split()

def edit_distance(a: list[str], b: list[str]) -> int:
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb)))
        prev = cur
    return prev[-1]

# ── Run ───────────────────────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(2)
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
rows = [r for r in rows if r.get("text")]
if not rows:
    print("manifest is empty")
    sys.exit(2)

print("Loading faster-whisper large-v3 (production config)...")
from faster_whisper import WhisperModel  # noqa: E402
server._whisper_model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")

stats: dict[str, dict[str, float]] = {}
lid_hits, lid_total = 0, 0
for r in rows:
    audio, sr = sf.read(r["audio"], dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / sr))
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    hyp, lang, meta = server._transcribe_blocking(audio)

    ref_w, hyp_w = normalize(r["text"]), normalize(hyp)
    errs = edit_distance(ref_w, hyp_w)
    d = r.get("dialect") or "unlabeled"
    s = stats.setdefault(d, {"errs": 0, "words": 0, "n": 0, "dropped": 0})
    s["errs"] += errs; s["words"] += len(ref_w); s["n"] += 1
    if not hyp:
        s["dropped"] += 1
    if r.get("lang") in ("ar", "en"):
        lid_total += 1
        lid_hits += (lang == r["lang"])
    gate = f"  DROPPED-BY:{meta['dropped']}" if meta.get("dropped") else ""
    print(f"  [{d:>9}] WER {errs}/{len(ref_w)}  lang={lang}{gate}  {r['audio'].split('/')[-1]}")

print(f"\n{'dialect':>10}  {'WER':>7}  {'clips':>5}  {'dropped-by-gates':>16}")
for d, s in sorted(stats.items()):
    wer = 100 * s["errs"] / max(s["words"], 1)
    print(f"{d:>10}  {wer:6.1f}%  {int(s['n']):>5}  {int(s['dropped']):>16}")
if lid_total:
    print(f"\nlanguage-ID accuracy (ar/en): {lid_hits}/{lid_total} ({100*lid_hits/lid_total:.0f}%)")
