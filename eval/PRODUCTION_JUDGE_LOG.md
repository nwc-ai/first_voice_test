# Production judge log — trend ledger for `eval/dialect_purity_lint.py --judge`

Append-only, same convention as `eval/BASELINES.md` ("do not edit past rows; append") — but
tracks a **traffic-driven trend line** over real production logs, not per-change events.
`BASELINES.md` answers "did this specific change regress anything"; this file answers "how is
real, unseen user traffic looking over time." Full judge reports stay in `logs/judge_runs/`
(gitignored — the repo is public and reports may echo model output, same reasoning as
`logs/ab_runs/`).

## Row schema

| date | log window (`--since`) | rows sampled / total routed | leak-rate by dialect | judge finding counts by category | notable new pattern | report path |
|---|---|---|---|---|---|---|

## Cadence — manual-but-disciplined, event-triggered, NOT cron

Deliberately not automated on a timer. Reasoning:

1. `--judge` requires stopping the live server and `ollama stop qwen3.5:27b` first (hard VRAM
   precondition — `qwen3.5:27b` is pinned via `keep_alive:-1` and won't be auto-evicted for a
   second ~20GB model; `quality_lint.check_judge_model_available()` enforces this
   defensively). A cron job that periodically takes down a live, user-facing voice server is a
   worse failure mode than an occasionally-missed manual run.
2. There is currently **zero** recorded production traffic (`logs/interactions.jsonl` doesn't
   exist yet as of this file's creation) — automating against an empty log is premature.

**Concrete triggers instead of a calendar:**
- After every deploy that reaches production.
- Whenever `logs/interactions.jsonl` has grown by a meaningful batch (~50-100 new routed rows)
  since the last entry below.

Revisit cron (the `schedule`/`CronCreate` tooling exists in this environment) only once there
is steady real traffic volume **and** a safe, automated mechanism for stopping/restarting the
live server around the judge run — neither exists today, so this is explicitly deferred, not
adopted.

## How to run

```bash
# 1. Stop the live server, then:
ollama stop qwen3.5:27b
# 2. Run the judge pass (uses qwen3:32b — already pulled locally):
.venv/bin/python eval/dialect_purity_lint.py --judge --judge-sample 25 --since <date> \
    > logs/judge_runs/<timestamp>.md
# 3. Append a summary row below, referencing the saved report.
```

## Log

_No entries yet — this ledger activates once `eval/dialect_purity_lint.py --judge` has been
run against real, non-empty production traffic. Not a defect; expected at the time this file
was created (`najdi-q2-wrong-elegant-papert.md`, Part A.4)._
