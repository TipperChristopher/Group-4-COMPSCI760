#!/bin/bash
cd "E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760/team_repo/feasibility_spike/reward_experiment"
PY="E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760/spike/venv/Scripts/python.exe"
for N in 100000 300000 600000 1000000; do
  TD="saved_models/_ck_eval_$N"
  mkdir -p "$TD"
  cp "saved_models/sac_ckpts/ckpt_${N}.zip" "$TD/final_model.zip" 2>/dev/null || { echo "MISSING ckpt_$N"; continue; }
  cp "saved_models/sac_ckpts/vecnormalize_${N}.pkl" "$TD/vecnormalize.pkl" 2>/dev/null
  echo "=== evaluating checkpoint $N ==="
  "$PY" -u eval_vs_baseline.py --model "$TD" --algo SAC \
      --tracks synthetic_track_1 synthetic_track_2 Spielberg Silverstone \
      --seeds 0 1 2 3 4 --tag "sac_ck$N" 2>&1 | tail -6
done
echo "ALL CHECKPOINT EVALS DONE"
