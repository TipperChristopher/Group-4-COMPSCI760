#!/bin/bash
# PPO fix sweep — one variable at a time, d1, seed 0, 2M steps, canonical pool.
#
# Evidence this targets: in all four grid cells PPO's policy std collapses
# 1.0 -> ~0.03 by ~0.5M steps, after which approx_kl runs 0.25-0.38 (healthy
# 0.01-0.03) and clip_fraction 0.51-0.57 (healthy 0.1-0.2), while
# explained_variance stays 0.88-0.97. Distance never leaves 25-45 m and no
# episode in ~24k ever completes a lap. The critic is fine; the policy is not.
#
# V0 is a REPRODUCTION CHECK: it must match Desmond's PPO_d1 run, which would
# prove our regenerated track pool and this train.py are equivalent to his.
set -u
cd "$(dirname "$0")"
PY="E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760/spike/venv/Scripts/python.exe"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
mkdir -p sweep_logs

launch () {  # name, extra args...
  local name="$1"; shift
  if [ -f "sweep_logs/${name}.done" ]; then echo "skip $name (done)"; return; fi
  echo "launch $name : $*"
  nohup "$PY" -u train.py --algo PPO --diversity 1 --seed 0 \
      --total-timesteps 2000000 --torch-threads 1 --run-tag "$name" "$@" \
      > "sweep_logs/${name}.log" 2>&1 &
}

#                          ent    target_kl  n_steps  n_epochs/batch  log_std_init  lr
launch V0_baseline
launch V1_ent        --ent-coef 0.01
launch V2_kl         --target-kl 0.03
launch V3_ent_kl     --ent-coef 0.01 --target-kl 0.03
launch V4_nsteps     --n-steps 8192
launch V5_logstd     --log-std-init -1.0
launch V6_updates    --n-epochs 4 --batch-size 256
launch V7_lr         --learning-rate 0.0001

echo
echo "launched: $(jobs -p | wc -l) processes"
echo "logs in sweep_logs/ ; models in models/PPO_1tracks_s0_<tag>/"
