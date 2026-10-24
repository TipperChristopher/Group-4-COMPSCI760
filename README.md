# Autonomous Racing RL: Zero-Shot Generalization in F1TENTH

This repository contains the training and evaluation pipeline for analyzing zero-shot generalization differences between on-policy (PPO) and off-policy (SAC) reinforcement learning algorithms under a time-optimal racing objective. 

The agent is trained on procedurally generated closed-loop circuits and evaluated zero-shot on real Formula One circuits (Silverstone, Monza, Spa, etc.).

## 1. Installation

Clone this repository and install the required dependencies for Python 3.13:
```bash
git clone [https://github.com/TipperChristopher/f1tenth-rl-generalization.git](https://github.com/TipperChristopher/f1tenth-rl-generalization.git)
cd f1tenth-rl-generalization
pip install -r requirements.txt

2. Generate the Track Pool
Clear out legacy ghost files, generate the synthetic tracks, and patch the 4-column CSV boundaries:

Bash
python generate_track_pool.py

Warning: generate_track_pool.py deletes every existing synthetic_track_* folder
before writing. Do not re-run it on a machine whose training tracks are in use.

2b. Held-out Validation and Test Tracks
Three synthetic splits are defined in tracks/manifest.json, with a checksum for
every file:

  train  synthetic_track_0..99   the training pool (generator indices 0-99)
  val    val_track_0..9          checkpoint selection (indices 100-109)
  test   test_track_0..19        reporting (indices 110-129)

All 130 come from one stream: random_trackgen.py with seed 123, the seed
generate_track_pool.py uses by default. val and test are the next 30 draws
after the training set, so they share its generator and settings but are
disjoint from it, and their distinct names keep them out of every training
pool. Recreate the held-out sets, or check an existing install, with:

Bash
python tracks/make_heldout_tracks.py            # build val/test, verify train
python tracks/make_heldout_tracks.py --verify   # check every file, write nothing
python tracks/verify_heldout_tracks.py          # load, spawn and drive checks

Do not use make_synth_tracks.py --seed 0 to build a training pool: it writes
different tracks under the same synthetic_track_N names. --verify catches this.

2c. Real Circuits (zero-shot test)
The manifest's "real" split holds the 23 f1tenth_racetracks circuits (Austin,
BrandsHatch, Budapest, Catalunya, Hockenheim, IMS, Melbourne, MexicoCity,
Montreal, Monza, MoscowRaceway, Nuerburgring, Oschersleben, Sakhir, SaoPaulo,
Sepang, Shanghai, Silverstone, Sochi, Spa, Spielberg, YasMarina, Zandvoort).
Download and install them, and check them, with:

Bash
python tracks/install_real_tracks.py            # download, fix names, verify checksums
python tracks/verify_real_tracks.py             # load, loop, spawn and length checks
python tracks/verify_real_tracks.py --sim       # plus one gap-follower episode each

install_real_tracks.py applies the only fix these need (the <name>_map.yaml
copy that fix_tracks.py makes). Do not run patch_yamls.py for them: it edits
the training tracks' yaml files and none of the real circuits.

Six circuits are longer than 450 m (Melbourne, Sepang, Shanghai, Silverstone,
Sochi, Spa). Under the 15,000-step evaluation cap one lap at 3 m/s needs at
most 450 m, so a slow policy can be truncated there before finishing a lap.
Report lap completion on these separately, or alongside progress, not as a
plain failure.

2d. Narrow Test Tracks (width ablation)
Every training track has a half-width of about 1.47 m; real circuits have a
median of 1.07 m and a minimum near 0.65 m. Two splits isolate width:

  narrowA  narrowA_track_0..19   test shapes at half-width ~1.07 m (WIDTH 7.5)
  narrowB  narrowB_track_0..19   test shapes at half-width ~0.75 m (WIDTH 5.52)

narrowX_track_k has exactly the centreline and map frame of test_track_k;
only the walls move. Rebuild or check them with:

Bash
python tracks/make_narrow_tracks.py              # build, verify vs test, record in manifest
python tracks/make_narrow_tracks.py --calibrate  # WIDTH -> half-width fit, writes nothing
python tracks/verify_real_tracks.py --split narrowA
python analysis/track_geometry.py                # confirms centreline identical to test

Do not make narrow tracks by editing WIDTH in random_trackgen.py and
rerunning it: a different WIDTH can change which attempts the generator
retries (shifting the whole RNG stream), and its tight_layout() call can
rescale the track. make_narrow_tracks.py avoids both.

2e. Varied-Width Tracks (vw_train / vw_val / vw_test)
All 130 synthetic shapes again, each with one constant half-width in
0.60-1.50 m, assigned in track order by a van der Corput sequence so every
nested training pool (first 1, 5, 20, 100) spans the range evenly:

  vw_train  vw_synthetic_track_0..99   shapes of synthetic_track_0..99
  vw_val    vw_val_track_0..9          shapes of val_track_0..9
  vw_test   vw_test_track_0..19        shapes of test_track_0..19

vw_train is a training split; val/test stay held out:

Bash
python tracks/make_vw_tracks.py                  # build, verify vs originals, record in manifest
python analysis/track_geometry.py --twin-check vw_train vw_val vw_test
python train.py --algo SAC --diversity 20 --seed 0 --track-prefix vw_synthetic_track_ --run-tag vw
python evaluate.py --algo SAC --diversity 20 --seed 0 --run-tag vw --track-set vw_test

3. Train the Agents
Launch the Stable-Baselines3 training grid. You can specify the algorithm, track diversity, and random seed:

Bash
python train.py --algo PPO --diversity 5 --seed 42

4. Zero-Shot Evaluation
Test the frozen policy against held-out F1 circuits (e.g., Monza, Silverstone) to generate the generalization performance graph:

Bash
python evaluate.py

A bare python evaluate.py (no --track-set) still runs only Spielberg, Monza
and Silverstone. For all 23 real circuits use --track-set real.

Evaluate on a named split with --track-set. Every split, real included, is
verified against the manifest's checksums before anything runs:

Bash
python evaluate.py --algo SAC --diversity 5 --seed 42 --track-set val
python evaluate.py --algo SAC --diversity 5 --seed 42 --track-set test
python evaluate.py --algo SAC --diversity 5 --seed 42 --track-set real
python evaluate.py --baseline gap --track-set test

If issues running the code:
Install the Missing Dependency
Install the exact version pinned in your experimental design:

PowerShell
py -m pip install gymnasium==0.29.1
If other Stable-Baselines3 packages are not yet installed in that specific environment, install them alongside:

PowerShell
py -m pip install stable-baselines3==2.9.0 shimmy

If issues when updating the F1Tenth code: 
Check your status:

PowerShell
git status
(If you see red text, your files are modified but not committed).

Stage everything:

PowerShell
git add .
Lock in the commit:

PowerShell
git commit -m "Update pipeline with new F1TENTH evaluation metrics"
Push to GitHub:

PowerShell
git push -u origin main

## Reproducing the final report (COMPSCI 760, Group 4)

The report's code footnote cites grid A at branch `final-grid` (commit `7552398`), and grid B plus the final test at branch `final-grid-v2` (commit `fdf6252`). `main` contains both.

### What is where
| Report item | Location |
|---|---|
| Track splits, varied widths (Sec. III-B, Table I) | `tracks/manifest.json`, `tracks/make_vw_tracks.py` (sections 2b–2e above) |
| Tuning rule and results (Sec. III-C) | `tuning3_DECISION_RULE.md`, `results/tuning/` |
| Grid A decision and job files | `final_grid/DECISION.md`; `final_grid/final_jobs_desmond.txt`, `final_grid/final_jobs_grant.txt`, `final_grid/final_jobs_chris.txt` |
| Grid B pre-registration and job files | `final_grid_v2/DECISION_v2.md`; `final_grid_v2/jobs_v2_chris.txt`, `final_grid_v2/jobs_v2_grant.txt`, `final_grid_v2/jobs_v2_desmond_ppo.txt` |
| Training logs for all 48 runs (Fig. 4) | `results/run_logs/grid_{A,B}/<run>/` |
| Validation scores and checkpoint selection | grid A: `results/final_grid_{desmond,grant,chris}/`, `results/final_test/selection.csv`; grid B: `results/final_test_v2/`, with the raw vw_val CSVs in `results/final_grid_v2_grant/`, `results/final_grid_v2_ppo_chris/` and `results/tuning/v2_SAC_*` (Desmond's 6 grid B SAC runs, saved to `results/tuning/` because the queue used the default results folder). The two `summary_*_jobs_v2_chris.csv` files in `results/tuning/` are grid B queue summaries, not tuning results. |
| Final test, per episode (Tables III–V) | `results/final_test/test_raw_*.csv`, `test_summary.csv`, `test_plan.csv`, `start_poses.csv` |
| Statistics (Tables III–V, Wilcoxon, Holm, Spearman) | `python analysis/report/report_stats.py` |
| Checkpoint selection check (recomputes all 48 choices from the vw_val CSVs) | `python analysis/report/check_selection.py` |
| Training curves (Fig. 4) | `python analysis/report/training_curves.py` |
| Slide animations | `tools/animate/make_slide_gifs.py`, `results/animations/` |

### Re-running the experiment
1. Install pinned dependencies (Windows, Python 3.12.10 64-bit, PowerShell 7). Clone with `git clone -c core.autocrlf=false --recurse-submodules https://github.com/TipperChristopher/Group-4-COMPSCI760.git`, then in the repo folder:
   ```
   py -3.12 -m venv .venv
   .venv\Scripts\python.exe -m pip install pip==26.2.1
   .venv\Scripts\python.exe -m pip install --no-deps -r final_grid/requirements-lock.txt
   git submodule update --init
   .venv\Scripts\python.exe -m pip install --no-deps -e ./f1tenth_gym
   ```
   `git submodule update --init` checks out `f1tenth_gym` at the pinned upstream commit `5a301bd` (branch `v1.0.0`). `--no-deps` is required: see the header of `final_grid/requirements-lock.txt`.
2. Build and verify tracks. Each command checks every file against `tracks/manifest.json` and refuses to install anything that differs:
   ```
   python tracks/make_vw_tracks.py
   python tracks/install_real_tracks.py
   python tracks/verify_real_tracks.py
   ```
3. Train one run. Each run is 2M steps on one environment; all 48 job lines are in the job files above. Grid A SAC example (`final_grid/final_jobs_desmond.txt`): `final_SAC_d100_s0 | --algo SAC --diversity 100 --seed 0 --track-prefix vw_synthetic_track_`, which runs as `python train.py --algo SAC --diversity 100 --seed 0 --track-prefix vw_synthetic_track_ --run-tag final_SAC_d100_s0`. Grid B PPO example (`final_grid_v2/jobs_v2_desmond_ppo.txt`): `v2_PPO_d100_s0 | --algo PPO --diversity 100 --seed 0 --track-prefix vw_synthetic_track_ --crash-penalty 5 --gamma 0.999 --n-steps 8192 --ent-coef 0.01` (adds `--crash-penalty 5 --gamma 0.999 --n-steps 8192 --ent-coef 0.01`).
4. Validate checkpoints on vw_val (5 episodes per track) and select the best of 400k / 800k / 1.2M / 1.6M / final by mean fractional laps; ties go to higher lap %, then the earlier checkpoint. The queue trains a job file's runs and evaluates every checkpoint on vw_val, for example `.\launch_queue.ps1 -Jobs final_grid\final_jobs_desmond.txt -ResultsDir results\final_grid_desmond -MaxThreads 16 -Checkpoints 200000,400000,600000,800000,1000000,1200000,1400000,1600000,1800000`. The rule (`final_grid_v2/DECISION_v2.md`, "Checkpoint selection") was applied to those CSVs and frozen in `results/final_test/selection.csv` (grid A) and `results/final_test_v2/selection.csv` (grid B; `candidate_sources.csv` lists where each candidate was scored).
5. Final test, run once: 10 episodes per track on `vw_test` and `real`, deterministic policy. Every job's exact command is in `results/final_test/test_plan.csv`, for example `python evaluate.py --algo SAC --diversity 1 --seed 0 --run-tag final_SAC_d1_s0 --track-set vw_test --episodes 10 --max-steps 15000 --reset-type cl_grid_static --target-laps 1 --eval-seed 0 --no-plot --out results/final_test/test_raw_final_SAC_d1_s0_vw_test.csv --checkpoint 800000`.

Notes:
- Evaluations reproduce exactly on the same machine but can differ slightly across CPUs.
- The final test ran on one machine.
- Trained model checkpoints are not in git because of their size. They are available from the authors on request.
