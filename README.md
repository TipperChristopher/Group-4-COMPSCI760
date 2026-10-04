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

3. Train the Agents
Launch the Stable-Baselines3 training grid. You can specify the algorithm, track diversity, and random seed:

Bash
python train.py --algo PPO --diversity 5 --seed 42

4. Zero-Shot Evaluation
Test the frozen policy against held-out F1 circuits (e.g., Monza, Silverstone) to generate the generalization performance graph:

Bash
python evaluate.py

Evaluate on a named split with --track-set. Synthetic splits are verified
against the manifest's checksums before anything runs:

Bash
python evaluate.py --algo SAC --diversity 5 --seed 42 --track-set val
python evaluate.py --algo SAC --diversity 5 --seed 42 --track-set test
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
