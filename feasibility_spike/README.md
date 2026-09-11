# Feasibility Spike (self-contained, additive)

This folder is a **self-contained** set of feasibility / verification experiments run against
`f1tenth_gym` + Stable-Baselines3. It does **not** import from or modify any code in the rest of
the repository — everything here lives under `feasibility_spike/` so the team's pipeline
(`train.py`, `evaluate.py`, `sb3_wrapper.py`, `src/`) is untouched.

## What it establishes (go/no-go evidence)

| Script | Question | Evidence produced |
|---|---|---|
| `scripts/q1_roundtrip.py` | Does the sim install & run headless? | `env.reset()`/`env.step()` round trip, spaces, 5-tuple check |
| `scripts/q3_checkenv.py` | Is it SB3-compatible? | prints spaces, runs `stable_baselines3 ... check_env` |
| `scripts/q4_ppo.py` | Does PPO train, how fast? | measured FPS, extrapolated run time, return trend |
| `scripts/q5_sac.py` | Does SAC run within memory? | measured FPS + peak RSS, replay-buffer math |
| `scripts/fix_tracks.py` | track filename shim | aliases `<name>.yaml` → `<name>_map.yaml` |
| `scripts/sb3_wrapper.py` | minimal single-agent SB3 adapter | flatten obs → Box, action (2,)↔(1,2); optional `scan_beams` downsample |

Full write-ups are in `reports/` (`spike_report.md`, `spike_report_comprehensive.md`, `journal.md`)
and the exact environment in `pip_freeze.txt`.

> Note: this spike's `sb3_wrapper.py` is a **feasibility harness** (uses the `features`
> observation, no reward shaping) and is intentionally separate from the project's own
> `sb3_wrapper.py` at the repo root. It is here as evidence of the environment working, not as
> part of the training pipeline.

## How to run

Needs a Python 3.11 env with `f1tenth_gym` (v1.0.0 branch), `stable-baselines3`, `torch`, `psutil`.
The exact working set is pinned in `pip_freeze.txt`.

```bash
# from this folder
cd scripts
python q1_roundtrip.py       # ~10 s: reset/step round trip
python q3_checkenv.py        # ~10 s: SB3 check_env
python q4_ppo.py             # ~2-3 min: PPO 100k, prints FPS + trend
python q5_sac.py             # ~15-25 min: SAC 50k, prints peak RSS
```

Rendering is off for all timing runs (headless). Measured numbers and hardware are recorded in
`reports/spike_report.md`.
