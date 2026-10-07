#!/bin/bash
# Arm A (ent_coef 0.01) vs Arm B (time_cost 0.0015) on the working base:
#   gamma 0.999 / n_steps 8192 / crash penalty 5 / d1 / seeds 0-2, 2M steps.
# 2 jobs in parallel, waves by seed. One variable per arm:
#   ENT001: entropy bonus gives sigma a second, outcome-independent voter.
#   TC0015: per-step time bleed makes the stall attractor cost money
#           (0.0015 x 3000 = 4.5 < CRASH_PENALTY 5: no suicide preference).
set -u
cd "$(dirname "$0")"
PY="E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760/spike/venv/Scripts/python.exe"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
mkdir -p sweep_logs

run () {  # tag seed ent_coef time_cost
  local tag="$1" seed="$2" ent="$3" tc="$4"
  local mark="sweep_logs/${tag}_s${seed}.done"
  if [ -f "$mark" ]; then echo "skip $tag s$seed (done)"; return; fi
  echo "[$(date '+%F %T')] start $tag s$seed" >> sweep_logs/ab_sweep.log
  ( nohup "$PY" -u train.py --algo PPO --diversity 1 --seed "$seed" \
      --total-timesteps 2000000 --torch-threads 1 --run-tag "$tag" \
      --gamma 0.999 --n-steps 8192 --ent-coef "$ent" --time-cost "$tc" \
      > "sweep_logs/${tag}_s${seed}.log" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] done  $tag s$seed (exit $rc)" >> sweep_logs/ab_sweep.log
    touch "$mark" ) &
}

wait_marks () {  # args: marker paths; blocks until all exist
  while true; do
    local missing=""
    for m in "$@"; do [ -f "$m" ] || missing="$missing $m"; done
    [ -z "$missing" ] && return 0
    sleep 60
  done
}

run ENT001 0 0.01 0.0
run TC0015 0 0.0 0.0015
wait_marks sweep_logs/ENT001_s0.done sweep_logs/TC0015_s0.done
run ENT001 1 0.01 0.0
run TC0015 1 0.0 0.0015
wait_marks sweep_logs/ENT001_s1.done sweep_logs/TC0015_s1.done
run ENT001 2 0.01 0.0
run TC0015 2 0.0 0.0015
wait_marks sweep_logs/ENT001_s2.done sweep_logs/TC0015_s2.done
echo "[$(date '+%F %T')] ab_sweep complete" >> sweep_logs/ab_sweep.log
