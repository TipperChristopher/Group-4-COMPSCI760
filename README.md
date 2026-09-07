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

3. Train the Agents
Launch the Stable-Baselines3 training grid. You can specify the algorithm, track diversity, and random seed:

Bash
python train.py --algo PPO --diversity 5 --seed 42

4. Zero-Shot Evaluation
Test the frozen policy against held-out F1 circuits (e.g., Monza, Silverstone) to generate the generalization performance graph:

Bash
python evaluate.py

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

.