# Slide animations

Made by `tools/animate/make_slide_gifs.py` (read-only on code, models and results).
Each panel replays a frozen test checkpoint with its own VecNormalize statistics, through
`evaluate.py`'s own `build_env`, `load_vecnormalize` and `run_episode` (imported unchanged):
deterministic policy, one lap, `cl_grid_static` spawn, 15,000-step cap, eval seed 0, which is
episode 0 of the final test. Every replay reproduced its `test_raw` row exactly (laps, steps, outcome).

**Track rule** (stated before rendering, applied mechanically to the `test_raw` CSVs): among the
set's tracks where the right-hand agent completes the lap in test episode 0, pick the track where
the left-hand agent's episode-0 score is closest to its overall mean on that set (all test
episodes). The tables below list every qualifying track, ordered by that distance.

## anim1_diversity: 1 track vs 100 tracks (real)

- `final_SAC_d100_s1` completes the lap in episode 0 on **18** real tracks; `final_SAC_d1_s1`'s overall real mean is 0.1027.
- **Chosen track: Oschersleben**

| track | `final_SAC_d100_s1` lap time | `final_SAC_d1_s1` episode 0 | distance to its mean |
|---|---|---|---|
| **Oschersleben** | 35.6 s | 0.0955 (crash, 2.8 s) | 0.0072 |
| Budapest | 48.1 s | 0.0908 (crash, 3.9 s) | 0.0119 |
| Sepang | 58.6 s | 0.1151 (crash, 5.1 s) | 0.0124 |
| SaoPaulo | 43.0 s | 0.0842 (timeout, 150.0 s) | 0.0185 |
| Melbourne | 57.3 s | 0.0818 (crash, 4.5 s) | 0.0209 |
| Spielberg | 41.9 s | 0.1244 (crash, 4.9 s) | 0.0217 |
| Nuerburgring | 55.0 s | 0.0785 (crash, 4.3 s) | 0.0242 |
| Sakhir | 53.8 s | 0.0620 (crash, 3.3 s) | 0.0407 |
| Zandvoort | 50.0 s | 0.0586 (crash, 2.7 s) | 0.0441 |
| Catalunya | 50.1 s | 0.1545 (crash, 6.9 s) | 0.0518 |
| Silverstone | 58.9 s | 0.0481 (crash, 2.6 s) | 0.0546 |
| MoscowRaceway | 42.5 s | 0.0433 (crash, 1.9 s) | 0.0594 |
| BrandsHatch | 41.4 s | 0.1645 (crash, 16.7 s) | 0.0618 |
| Sochi | 59.7 s | 0.1663 (crash, 7.5 s) | 0.0636 |
| Spa | 70.1 s | 0.0379 (crash, 2.6 s) | 0.0648 |
| Montreal | 45.4 s | 0.0119 (crash, 5.5 s) | 0.0908 |
| IMS | 34.5 s | 0.1936 (crash, 113.7 s) | 0.0909 |
| MexicoCity | 45.0 s | 0.2016 (timeout, 150.0 s) | 0.0989 |

- *Trained on 1 track*: `final_SAC_d1_s1` (grid A, SAC, checkpoint final: `models/SAC_1tracks_s1_final_SAC_d1_s1/final_model.zip` + `models/SAC_1tracks_s1_final_SAC_d1_s1/vecnormalize.pkl`). Replay on Oschersleben: crash after 279 steps (2.8 s), 24.9 m, 0.0955 laps; matches its test_raw row.
- *Trained on 100 tracks*: `final_SAC_d100_s1` (grid A, SAC, checkpoint final: `models/SAC_100tracks_s1_final_SAC_d100_s1/final_model.zip` + `models/SAC_100tracks_s1_final_SAC_d100_s1/vecnormalize.pkl`). Replay on Oschersleben: finished after 3558 steps (35.6 s), 260.4 m, 0.9989 laps, lap completed; matches its test_raw row.
- Speed ×6 at 20 fps; loop 8.5 s (140 frames, 1.5 s summary frame first, 1 s hold); GIF 64 colours, 1200×560 px.
- Files: `anim1_diversity.gif` (0.08 MB), `anim1_diversity.png` (0.06 MB, 1600 px wide), `anim1_diversity_trajectories.csv` (pose every step).
- Map check: 100.0%, 100.0% of trajectory points lie on free map cells.

## anim2_ppo_fix: PPO before vs after the fix (vw_test)

- Median seed check, A PPO d100 seeds (vw_test laps): s0 0.0882, s2 0.1000, s1 0.1508 → **final_PPO_d100_s2**
- Median seed check, B PPO d100 seeds (vw_test laps): s0 0.4359, s1 0.4613, s2 0.7840 → **v2_PPO_d100_s1**
- `v2_PPO_d100_s1` completes the lap in episode 0 on **5** vw_test tracks; `final_PPO_d100_s2`'s overall vw_test mean is 0.1000.
- **Chosen track: vw_test_track_17**

| track | `v2_PPO_d100_s1` lap time | `final_PPO_d100_s2` episode 0 | distance to its mean |
|---|---|---|---|
| **vw_test_track_17** | 23.9 s | 0.1104 (crash, 2.1 s) | 0.0104 |
| vw_test_track_18 | 25.4 s | 0.0762 (crash, 8.6 s) | 0.0238 |
| vw_test_track_11 | 23.4 s | 0.0564 (crash, 1.4 s) | 0.0436 |
| vw_test_track_5 | 23.5 s | 0.0532 (crash, 1.4 s) | 0.0468 |
| vw_test_track_9 | 27.9 s | 0.0464 (crash, 1.4 s) | 0.0536 |

- *Default PPO*: `final_PPO_d100_s2` (grid A, PPO, checkpoint final: `models/PPO_100tracks_s2_final_PPO_d100_s2/final_model.zip` + `models/PPO_100tracks_s2_final_PPO_d100_s2/vecnormalize.pkl`). Replay on vw_test_track_17: crash after 209 steps (2.1 s), 18.2 m, 0.1104 laps; matches its test_raw row.
- *Fixed PPO (γ 0.999, 8192-step rollouts, entropy)*: `v2_PPO_d100_s1` (grid B, PPO, checkpoint final: `models/PPO_100tracks_s1_v2_PPO_d100_s1/final_model.zip` + `models/PPO_100tracks_s1_v2_PPO_d100_s1/vecnormalize.pkl`). Replay on vw_test_track_17: finished after 2388 steps (23.9 s), 164.8 m, 0.9982 laps, lap completed; matches its test_raw row.
- Speed ×4 at 20 fps; loop 8.6 s (141 frames, 1.5 s summary frame first, 1 s hold); GIF 64 colours, 1200×560 px.
- Files: `anim2_ppo_fix.gif` (0.09 MB), `anim2_ppo_fix.png` (0.06 MB, 1600 px wide), `anim2_ppo_fix_trajectories.csv` (pose every step).
- Map check: 100.0%, 100.0% of trajectory points lie on free map cells.

## Caption numbers (from the test_raw CSVs, all test episodes)

- `final_SAC_d100_s1` on real: lap completion **84.8%** (195/230 episodes); episode 0 completed on 18 of 23 tracks.
- `v2_PPO_d100_s1` on vw_test: lap completion **14.5%** (29/200 episodes); episode 0 completed on 5 of 20 tracks.
- `final_PPO_d100_s2` on vw_test: lap completion **0.0%** (0/200 episodes); episode 0 completed on 0 of 20 tracks.
