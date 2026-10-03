# How to train with the FROZEN reward + env (for teammates)

This folder (feasibility_spike/reward_experiment) contains the trainer used for
every pilot in the deck. It vendors the FROZEN progress reward, so it does NOT
depend on the team repo's sb3_wrapper.py — important because main still carries
the OLD reward (time-alive + low-speed penalty + 5M steps) until the reward
branch is merged.

## The frozen protocol (what the deck reports)
- Reward: 1.0 x metres of centreline progress - CRASH_PENALTY on collision; TIME_COST = 0
  (in new_reward_wrapper.py; crash penalty passed via --penalty, frozen per cell)
- VecNormalize: observations only (norm_reward=False), identical for PPO and SAC
- One env + track pool (n_envs=1, by design — do not change)
- Budget: 2M env-steps, 20 segments of 100k; eval between segments is deterministic
- Eval protocol: centreline spawn (cl_grid_static), 5 fixed seeds (0-4), one-lap
  termination, 15,000-step cap

## Train (examples)
PY="E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760/spike/venv/Scripts/python.exe"
cd "E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760/team_repo/feasibility_spike/reward_experiment"

# PPO, frozen protocol (penalty 40), 2M, resumable:
"$PY" -u learning_curve.py --algo ppo --vecnormalize --penalty 40 --steps 2000000 --segments 20 --save-model saved_models/ppo_run1

# SAC, team's frozen penalty 5, 2M, resumable:
"$PY" -u learning_curve.py --algo sac --vecnormalize --penalty 5 --steps 2000000 --segments 20 --save-model saved_models/sac_run1

# Old-reward ablation (V1, for the reward comparison):
"$PY" -u learning_curve.py --algo ppo --vecnormalize --old-reward --penalty 40 --steps 2000000 --segments 20 --save-model saved_models/ppo_old1

# Optional: also keep every segment's model (overfitting probe):
#   add:  --keep-checkpoints

## Resume after any interruption / PC restart
Just re-run the SAME command. The trainer saves model.zip + vecnormalize.pkl +
state.json after every 100k segment and continues from the last completed one.
See RESUME.txt for the exact commands of the runs in progress.

## Evaluate a finished model on unseen tracks (same protocol, vs baselines)
"$PY" -u eval_vs_baseline.py --model saved_models/<run_dir> --algo PPO \
    --tracks synthetic_track_0 synthetic_track_1 synthetic_track_2 Spielberg Silverstone \
    --seeds 0 1 2 3 4 --tag <run_name>

# Baselines (no training needed):
"$PY" -u eval_vs_baseline.py --algo gap    --tracks ... --seeds 0 1 2 3 4 --tag gap_protocol
"$PY" -u eval_vs_baseline.py --algo random --tracks ... --seeds 0 1 2 3 4 --tag random_protocol

## Results land in
- results/learning_curve_cp*.json   (per-segment curve: laps, ep_rew_mean, ep_len_mean)
- results/eval_*.json               (per-seed unseen-track evals)
- Deck figures regenerate via:  _make_reward_compare_fig.py, gen_gap_figure.py
- Deck rebuild:                  _insert_our_slides.py (from Group4_ProjectUpdate_v2.pptx)

## Known gotchas
- The sim's ray-caster is a process-wide class attribute: one track per process.
- Team main-branch sb3_wrapper.py has the OLD reward; use this folder's wrappers.
- Crash penalty must be FROZEN identically across cells (pilots used 40; the
  team's frozen value per the handover is 5 — decide before the grid).
