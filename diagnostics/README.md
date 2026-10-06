# Diagnostics

Analysis scripts quoted in `../CONTEXT.md`. These are the exact scripts that produced
the numbers in that document, kept in-tree so the results can be re-derived.

**They contain absolute paths** to this machine (`E:/OneDrive - .../COMPSCI 760`).
If you clone elsewhere, edit the `ROOT` / `REPO` constants at the top of each file.

| script | what it does | section |
|---|---|---|
| `build_canonical_tracks.py` | Installs the canonical seed-123 training pool. **Run this first on a fresh machine** — `train.py` refuses to train on non-canonical maps. | §9 |
| `basin_check.py` | Rolls out trained policies, reports undiscounted and γ-discounted return. Produced the §4 table. | §4 |
| `obs_redundancy.py` | Measures how much a 108-beam scan changes per 100 Hz step. | §5 |
| `did_it_brake2.py` | Speed trajectory over the final 2 s before a crash. Shows the fixed PPO *does* brake; the baseline accelerates into the wall. | §4 |
| `g999_ckpt_curve.py` | Evaluates the γ run at every 200 k checkpoint on held-out tracks. | §6.4 |
| `width_eval_oursac.py` | Our SAC on test / narrowA / narrowB / real — identical centrelines, only walls move. Produced §7. | §7 |
| `eval_lap_break.py` | Matched-checkpoint evaluation of the sweep variants. | §6 |
| `eval_ppo_sweep.py` | Evaluates every sweep variant at a common checkpoint. | §6.1 |
| `penalty_compare.py` | Our penalty-40 SAC vs Desmond's penalty-5 SAC on tracks unseen by both. | — |
| `spawn_probe.py` | Enumerates distinct spawn poses and their outcomes. Showed 30 eval seeds give only 9 distinct spawns, and one of them always crashes. | §8 |
| `speed_vs_robust.py` | Lap time vs spawn-robustness across checkpoints of one run. | §6.4 |
| `sac_train_tracks.py` | SAC cells evaluated on the local track set. | §8 |
