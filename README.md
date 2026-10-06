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
