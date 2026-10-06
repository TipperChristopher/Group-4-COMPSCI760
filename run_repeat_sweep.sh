#!/bin/bash
# Background experiment batch: credit assignment (action repeat), budget check,
# and the width ablation on a policy that actually drives.
set -u
cd "$(dirname "$0")"
PY="E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760/spike/venv/Scripts/python.exe"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
mkdir -p sweep_logs

train () {  # tag, extra args...
  local tag="$1"; shift
  nohup "$PY" -u train.py --algo PPO --diversity 1 --seed 0 \
      --torch-threads 1 --run-tag "$tag" "$@" > "sweep_logs/${tag}.log" 2>&1 &
  echo "  train $tag : $*"
}

# --- Credit assignment: hold each action for 10 physics steps (100 Hz -> 10 Hz) ---
# GAE window 0.168 s -> 1.68 s; track visible at 10 m/s goes 1.7 m -> 16.8 m.
train R1_repeat10_alone   --total-timesteps 2000000 --action-repeat 10
train R2_repeat10_v6      --total-timesteps 2000000 --action-repeat 10 --n-epochs 4 --batch-size 256
train R3_repeat10_v4      --total-timesteps 2000000 --action-repeat 10 --n-steps 8192
train R4_repeat25         --total-timesteps 2000000 --action-repeat 25

# --- Budget check: the best fixed variant, extended past 2M ---
train X1_v4_5M            --total-timesteps 5000000 --n-steps 8192

echo
echo "launched $(jobs -p | wc -l) training runs -> sweep_logs/"
