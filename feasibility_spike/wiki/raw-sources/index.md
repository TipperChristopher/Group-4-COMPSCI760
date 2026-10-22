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
| `~/.pi/runs/2026-10-04-cs760-final-presentation-plan/journal.md` | journal-2026-10 | attempt-by-attempt record incl. retractions, Attempts 1–40 (2026-10-04 → 10-23) (Aolin's machine only) |

## bucket: final grids and final test (2026-10-08 – 10, high)

| path | slug | topics |
|---|---|---|
| `origin/final-grid` @ `7552398`: `final_grid/DECISION.md`, job files, `check_setup.py` | grid-a | grid A decision (defaults, penalty 40) and setup |
| `origin/final-grid-v2` @ `fdf6252`: `final_grid_v2/DECISION_v2.md` (`985b4bb`), jobs + checker (`a22bc32`), fingerprint (`16440e6`), `results/final_test_v2/selection.csv` (`dd87128`), `results/final_test/` | grid-b | grid B pre-registration, selection, final test, baseline configs |
| `COMPSCI 760/G4_full_bundle_2026-10-10.zip` (9.6 MB, 1345 entries, sha256 `b7526991≈348a5b`); extracted to `_verify/G4_full_bundle_2026-10-10/` | final-bundle | repo snapshot `fdf6252`: pilot, both grids, selection, raw test CSVs, docs (no models) |
| `_verify/G4_full_bundle_2026-10-10/_verify_{summary,selection,configs,tracks,ladders}.py`, `_still_learning.py` | final-verify | our arm's-length recomputation |
| `_verify/final_pptx_review/{check_claims,check2,insights,seed_numbers,width_same_method,width_vs_score}.py` | final-analysis | seed-level follow-ups, insights, widths, width-vs-score |
| `G4_selected_models.zip` (team; not on Aolin's machine) | selected-models | the 48 selected checkpoints (sha256s in `selection.csv` / `test_plan.csv`) |

## bucket: tuning and late diagnosis (2026-10-07 – 09, high unless noted)

| path | slug | topics |
|---|---|---|
| `COMPSCI 760/tuning_vw_val_results.zip`; copy + notes in `results/tuning_vw_val_verified/` | tuning-vw | teammate tuning round (16 runs), time cost, SAC γ 0.999 |
| `origin/desmond/parallel-ppo` @ `058bb93` | tuning-3 | tuning round 3 and its pre-registered rule |
| `results/ppo_diagnosis/eval_ab_own_track/`, `diagnostics/ab_own_track_eval.py` | ab-stability | entropy vs time-cost stability A/B (d1, 3 seeds) |
| `results/vw_p5_tests/` (ppo/, train_tracks/), `diagnostics/vw_p5_*_eval_watcher.py`, `diagnostics/vw_train_tracks_eval.py` | vw-p5 | PPO control vs sigma-fix at vw d20 penalty 5; training vs unseen tracks |
| `_verify/s1p5/` (results/, logs/; not in git) | s1p5 | SAC γ 0.999 at penalty 5 (stopped ~1.81M) |
| `COMPSCI 760/Group4_PPO_fix_and_final_config_2026-10-09.zip` (15.1 MB, 429 entries) | ppo-handoff | our teammate handoff: README, EXPLAINER, figures fig0–fig5, data, scripts |

## bucket: presentation (2026-10-10 – 11, outside git)

| path | slug | topics |
|---|---|---|
| `COMPSCI 760/Group 4 Final Results.pptx` | deck-draft | team first draft (17 slides) |
| `COMPSCI 760/Group 4 Final Results v2.pptx`; build `_verify/final_pptx_review/build_deck.py`, `make_figs.py` | deck-v2 | our corrected 18-slide version |
| `COMPSCI 760/Group 4 Final Results (1).pptx` | deck-final | the team's merged final deck |
| `COMPSCI 760/Group 4 Final Results (1) - checked.pptx` / `.pdf`; `_verify/final_check/apply_fixes.py` | deck-checked | final deck + three fixes |
| `COMPSCI 760/Group 4 Final Talk Script.pdf`; `_verify/script_check/` | talk-script | team talk script; timing and checks |
| `COMPSCI 760/Aolin - talk script and QA (final).md` | aolin-qa | Aolin's final talk part and Q&A prep (mixed) |

## bucket: external references (checked via Crossref, 2026-10-11)

| reference | slug | used for |
|---|---|---|
| Kunda et al. 2025, *Scientific Reports* 15:43940, doi:10.1038/s41598-025-27702-6; text saved at `_verify/final_pptx_review/kunda.txt` | kunda-2025 | anchor paper (PPO vs SAC in racing; tuned PPO tested on 21 unseen tracks) |
| Sezer & Gokasan 2012, *Robotics and Autonomous Systems* 60(9):1123–1134, doi:10.1016/j.robot.2012.05.021 | sezer-2012 | follow-the-gap baseline |

## bucket: sessions — complete list for this project (mixed)

All under `~/.pi/agent/sessions/--E--OneDrive - The University of Auckland-Desktop-COMPSCI 760--/` (Aolin's machine only). Timestamps are UTC.

| file | span | content | ingested |
|---|---|---|---|
| `2026-08-13T10-16-29-491Z_019ffa9f-≈.jsonl` (11 MB) | 08-13 → 10-03 | setup, feasibility spike, Presentation 2, September pilots | 2026-09-20 (wiki created from it) |
| 15 files `2026-09-12` – `2026-09-17` (30–830 KB) | single runs | subagent sessions (no user messages) | not needed |
| `2026-10-03T23-42-25-433Z_01a10425-≈.jsonl` (4.3 MB) | 10-04 → 10-08 | status audit, grid verification, PPO diagnosis, RL teaching, A/B launch | 2026-10-06 (A/B results added 10-09; rest 2026-10-23) |
| `2026-10-05T00-28-34-453Z_01a10976-≈.jsonl` | 10-05 | one "hi" message, no work | nothing to ingest |
| `2026-10-07T06-57-22-363Z_01a11527-09bb-7531-9b54-d7ef84979a55.jsonl` (21 MB) | 10-07 → 10-23 | fork of the above: tuning-round verification, penalty-5 tests, grid B build, final bundle verification, deck review + v2, final deck check, talk script, Q&A prep, this ingest | 2026-10-23 |
