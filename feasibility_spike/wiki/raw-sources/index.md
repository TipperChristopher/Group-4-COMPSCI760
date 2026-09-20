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
