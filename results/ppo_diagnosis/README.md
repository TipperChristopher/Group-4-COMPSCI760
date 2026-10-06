# results/ppo_diagnosis — raw evidence behind CONTEXT.md

Everything here was copied (not recomputed) from the run directories and the
evaluation scratch folder on 2026-10-06 by `diagnostics/archive_to_repo.py`.
All runs: `train.py --algo PPO --diversity 1 --total-timesteps 2000000`, canonical
seed-123 pool, `CRASH_PENALTY 5` unless stated. Hardware: one Windows desktop,
CPU only (torch 2.14.0+cpu, SB3 2.9.0, Python 3.12.10).

## `runs_summary.csv`

One row per run: every hyperparameter that differs from SB3 defaults, plus
`last200k_mean_m` / `last200k_lap_rate` (training log, **stochastic** policy,
final 200 k env steps) and `max_laps`.

**`status_in_config` is misleading for killed runs.** `train.py` only writes
`completed` at the end; a run killed mid-way still says `running`. Actual state:

| run | real state |
|---|---|
| V0–V7, G999_gamma, G999L99_full, P40_penalty40 | completed, 2 M |
| X1_v4_5M | **killed** at 2.54 M (plateau identical to V4) |
| R2_repeat10_v6, R3_repeat10_v4, R4_repeat25 | **killed** (to reduce machine load; all 0% laps) |
| G999_s1, G999_s2, G999_ns2048, G999_SAC, R1_repeat10_alone | **still running at archive time** — logs are a snapshot |

## `runs/<run>/`

`run_config.json` (commit, reward hash, all args), `progress.csv` (per-update
SB3 diagnostics: std, approx_kl, clip_fraction, explained_variance),
`monitor_0.monitor.csv` (per episode: return, length, progress_m, laps).
Completed runs also carry `final_model.zip` + `vecnormalize.pkl`.
`G999_gamma` also keeps its 800 k checkpoint (best held-out score of any checkpoint).

## Evaluation CSVs (team `evaluate.py`, deterministic, `--target-laps 1`, `cl_grid_static`, 15 k step cap)

| folder | what | tracks × episodes |
|---|---|---|
| `eval_sweep_1M/` | all 8 sweep variants at the matched 1.0 M checkpoint | test_track_0..4 + Spielberg, Silverstone × 5 |
| `eval_ck800k/`, `eval_ck2M/` | V4, G999_gamma, G999L99_full, P40 at 800 k and 2 M | same |
| `eval_g999_curve/` | G999_gamma at every 200 k checkpoint 400 k–2 M | same |
| `eval_width_oursac/` | our `sac_overfit` (penalty 40) at 1.2 M and 2 M on test / narrowA / narrowB / real | 5 tracks × 10 (real 2 × 10) |
| `repro_desmond_grid/` | re-evaluation of Desmond's seed-0 grid weights (noise floor, penalty comparison, spawn probes) | see file names |

**Warning — `repro_desmond_grid/SACd*_trainpool.csv` and `repro_SACd1_oursynth.csv`
were evaluated on our *seed-0* `synthetic_track_*` maps, which are NOT Desmond's
training tracks (0/20 match).** They are kept only as the record of that mistake;
do not cite them as training-track performance. See wiki `incidents.md`.

5 eval seeds give only **3 distinct spawn poses** per track under `cl_grid_static`
(seeds 0≡1, 2≡3), so "25 episodes" on 5 tracks is ~15 distinct starts.

## Not in git

- `models/` (gitignored): all intermediate checkpoints, ~7 MB per run.
- Desmond's seed-0 grid bundle (`Group4_DiversityGrid_seed0.zip`, 343 MB).
- The scratch folder `../_verify/` on the original machine.
