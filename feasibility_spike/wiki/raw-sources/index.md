# Raw-sources registry

Every source the wiki compiles from. Referenced by in-repo path or URL — not
copied (all sources are stable and in-repo). Reliability tiers: `high`
(source code, deterministic JSON results), `mixed` (our narrative writeups),
`unverified` (external, until cross-referenced).

## bucket: results (JSON, high)

Under `feasibility_spike/reward_experiment/results/`.

| path | slug | topics |
|---|---|---|
| eval_sac_1m.json | sac-unseen-eval | SAC 1.2M unseen-track laps |
| eval_ppo_vn_2m.json | ppo-unseen-eval | PPO 2M unseen-track laps |
| eval_gap_protocol.json | gap-eval | gap-follower (threshold 20) |
| eval_random_protocol.json | random-eval | random baseline |
| eval_sac_nar7.json | nar7-eval | SAC on narrow tracks |
| learning_curve_cp40_vn_1000000.json | ppo-lc-1m | PPO learning curve |
| learning_curve_cp40_sac_vn_1000000.json | sac-lc-1m | SAC learning curve |
| learning_curve_cp40_vn_2000000.json | ppo-lc-2m-p40 | PPO 2M plateau |
| learning_curve_cp40_vn_old_2000000.json | ppo-lc-2m-old | PPO 2M old reward |
| learning_curve_cp5_vn_2000000.json | ppo-lc-2m-p5 | PPO 2M penalty 5 |
| learning_curve_cp40_sac_vn_2000000.json | sac-lc-2m | SAC 2M (partial, 1.2M) |
| reward_scan2.json | reward-scan | V1-vs-V2 reward on same rollout |
| baselines.json | baselines | aligned gap + random |
| spawn_study_Spielberg.json | spawn-spielberg | spawn offset cl vs rl |
| spawn_study_Silverstone.json | spawn-silverstone | spawn offset cl vs rl |

Other `results/*.json` (completion_*, ent/bonus/shape/warmup ablations,
train_compare, rl_before_after_PPO, reward_scan[v1]) — provenance only, see
`results.md`.

## bucket: writeups (our markdown, mixed)

Under `feasibility_spike/reward_experiment/` and COMPSCI 760 root.

| path | slug | topics |
|---|---|---|
| reward_experiment/RESULTS.md | results-writeup | results narrative |
| reward_experiment/REWARD_ANALYSIS.md | reward-writeup | reward + single-env caveat |
| reward_experiment/SPAWN_FIX_RESULTS.md | spawn-writeup | spawn fix verification |
| reward_experiment/METHODOLOGY_PLAN.md | methodology-plan | protocol grounding |
| reward_experiment/RESUME.txt | resume | resumable run commands |
| ../../METHODOLOGY_PLAN.md, TEAM_ANALYSIS.md, MEETING_BRIEF.md | root-docs | planning docs |

## bucket: code (source, high)

| path | slug | topics |
|---|---|---|
| team_repo/train.py | train | training entry (team) |
| team_repo/evaluate.py | evaluate | eval entry (team) |
| team_repo/sb3_wrapper.py | sb3-wrapper | frozen reward (team) |
| team_repo/src/track_pool.py | track-pool | stratified sampler (team) |
| spike/f1tenth_gym_v1/.../base_classes.py | base-classes | RaceCar.scan_simulator (ray-caster bug) |
| spike/f1tenth_gym_v1/.../reset/__init__.py, reset/utils.py | reset | spawn modes |
| spike/f1tenth_gym_v1/.../f110_env.py | f110-env | seed→spawn (L366-367) |
| spike/f1tenth_gym_v1/.../random_trackgen.py | trackgen | TRACK_TURN_RATE curvature cap |
| reward_experiment/learning_curve.py, eval_vs_baseline.py, baselines.py, _gen_narrow.py, new_reward_wrapper.py | our-tools | our additive tooling |

## bucket: team (deck + handovers)

| path | slug | topics |
|---|---|---|
| Group4_ProjectUpdate_FINAL2.pptx | deck | Presentation-2 deck |
| _v2_build_src.pptx | deck-src | v2 source the build loads |
| Desmond handover (reward V1/interim/V2, protocol) | desmond-handover | reward + protocol history |

## bucket: diagnosis (2026-10, on `team/crash-penalty-flag`, high)

| path | slug | topics |
|---|---|---|
| results/ppo_diagnosis/runs_summary.csv | runs-summary | every PPO/SAC run: config + last-200k metrics |
| results/ppo_diagnosis/runs/<run>/{run_config.json,progress.csv,monitor_0.monitor.csv} | run-logs | per-update + per-episode logs |
| results/ppo_diagnosis/eval_sweep_1M/ | sweep-eval | 8 variants, matched 1 M, held-out |
| results/ppo_diagnosis/eval_ck800k/, eval_ck2M/ | ck-eval | V4/G999/G999L99/P40 at 800 k and 2 M |
| results/ppo_diagnosis/eval_g999_curve/ | g999-curve | G999 every 200 k, held-out |
| results/ppo_diagnosis/eval_width_oursac/ | width-oursac | our SAC on test/narrowA/narrowB/real |
| results/ppo_diagnosis/repro_desmond_grid/ | grid-repro | re-eval of Desmond's weights (noise floor, penalty compare); `*_trainpool` files are on the WRONG maps |
| results/grid_s0/*.csv | grid-s0 | Desmond's seed-0 grid evaluation CSVs |
| results/track_geometry/summary.json, track_summary.csv | geometry | width/curvature distributions |
| tracks/manifest.json | manifest | canonical splits + checksums |
| diagnostics/*.py | diag-scripts | the analysis scripts (index in diagnostics/README.md) |
| CONTEXT.md | context | team-facing summary of the October work (mixed) |

## bucket: external bundle (high, not in git)

| path | slug | topics |
|---|---|---|
| Group4_DiversityGrid_seed0.zip (343 MB; Aolin's `COMPSCI 760/_verify/`) | grid-bundle | Desmond's 8 trained cells, logs, checkpoints |
| context_for_aolin.md (COMPSCI 760 root) | desmond-notes | Desmond's notes on the grid (mixed) |

## bucket: sessions (mixed)

| path | slug | topics |
|---|---|---|
| `~/.pi/agent/sessions/--E--OneDrive - The University of Auckland-Desktop-COMPSCI 760--/2026-10-03T23-42-25-433Z_01a10425-c099-7f4d-b774-f991e0ae401f.jsonl` | session-2026-10-04 | status audit → grid verification → PPO diagnosis → wiki ingest (Aolin's machine only) |
| `~/.pi/runs/2026-10-04-cs760-final-presentation-plan/journal.md` | journal-2026-10 | attempt-by-attempt record incl. retractions (Aolin's machine only) |
