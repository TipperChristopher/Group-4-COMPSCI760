# Spawn / reset protocol — root cause, fix, and before/after evidence

All numbers below are reproducible via the additive scripts in this folder
(`spawn_study.py`, `seed_probe.py`, `evaluate_centreline.py`). No team code was
modified. JSON artifacts: `results/spawn_study_Spielberg.json`,
`results/spawn_study_Silverstone.json`, `results/rl_before_after_PPO.json`.

## 1. Root cause (verified in source)
- Default reset is `rl_grid_static` (`f110_env.py:234`, `reset/__init__.py`) → spawn on the **raceline**.
- Real circuits ship a raceline file; **synthetic tracks do not**, so `track.raceline`
  silently falls back to `track.centerline` (`track/track.py:137-142`).
- Result: **training (synthetic) spawns on the centreline; evaluation (real circuits)
  spawns on the raceline**, ~0.8 m off-centre near the kerb. Same code, different behaviour.
- The exact spawn point is drawn by `np.random.choice` over the first ~1 m of the start
  line (`reset/masked_reset.py`), so an *unpinned* single episode is a dice roll.

## 2. The fix (eval-only, one line)
In `evaluate.py build_env`, add `reset_config` to the env config:
```python
config={"num_agents": 1, "timestep": 0.01, "map": track_name,
        "reset_config": {"type": "cl_grid_static"}},   # <-- spawn on centreline, matching training
```
Also pin `--eval-seed` and evaluate over a fixed set of seeds (`--episodes K`).
Training is untouched (`train.py` uses the default, which is the centreline for synthetic
tracks anyway). For a single agent, `cl` vs `rl` changes only the reference line — the
`move_laterally` flag is a no-op on the pose (`reset/utils.py`, offset applied only if
`n_agents > 1`).

## 3. Env-level proof — the fix moves the spawn to the centreline
Spawn seed 0, measured off-centreline distance:

| circuit | `rl_grid_static` (before) | `cl_grid_static` (after) |
|---|---|---|
| Spielberg | 0.809 m (on raceline) | **0.000 m** (on centreline) |
| Silverstone | 0.790 m | **0.000 m** |

## 4. Seed propagation — VERIFIED (this was the teammate's open question)
`f110_env.reset(seed)` reseeds the global numpy RNG (`f110_env.py:366-367`), so the spawn
is a deterministic function of the seed. `seed_probe.py` confirms end-to-end:
- `DummyVecEnv.seed(s)` gives the **identical** spawn to `env.reset(seed=s)` directly.
- Same seed → identical spawn; different seed → different spawn (reproducible).
- **VecNormalize does NOT break propagation** (the RL eval stack is safe).
- Without reseeding, consecutive resets cycle through ~4 discrete start-line points (this is
  the drift that makes an unpinned episode unreliable).

**Conclusion: pinning `--eval-seed` gives fair, reproducible spawns through the full
DummyVecEnv + VecNormalize eval stack.**

## 5. Before/after outcomes (5 spawn seeds each)
Follow-the-Gap (no training):

| circuit | BEFORE (raceline) | AFTER (centreline) |
|---|---|---|
| Spielberg | 1.999 laps, finish 5/5 | 1.999 laps, finish 5/5 |
| Silverstone | 0.177 laps, crash 5/5 | 0.176 laps, crash 5/5 |

PPO trained on 5 tracks (Desmond's `PPO_5tracks_s42`), zero-shot:

| circuit | BEFORE (raceline) | AFTER (centreline) |
|---|---|---|
| Spielberg | 0.106 laps, crash 5/5 (std 0.001) | 0.106 laps, crash 5/5 (std 0.000) |
| Silverstone | 0.155 laps, crash 5/5 (std 0.001) | 0.154 laps, crash 5/5 (std 0.000) |

## 6. Honest reconciliation + what the fix actually buys us
- **We did NOT reproduce the teammate's "seeds 3-4 finish (2.0 laps)" bimodality** on
  Silverstone with the vendored gap-follower — here it crashes at ~0.18 laps on every seed.
  That was likely their specific gap-follower variant/config; **we need their exact command
  to confirm.**
- The spawn protocol changes the *outcome* only for **borderline-capable** policies (ones
  that finish *sometimes*). Policies that clearly fail (our PPO) or clearly succeed
  (gap on Spielberg) are unaffected. Completion-rate "instability" lives exactly in the
  borderline regime — and the **diversity sweep will produce many borderline policies**, so
  fixing this *before* the main experiments is the right call.
- Net: the fix is a **correctness / fairness / reproducibility** fix (eval spawn now matches
  training; pinned seed → std→0), not a performance rescue. The presentation claim "same
  starting positions, same protocol" becomes *true*.

## 7. Also fixed here
- `Silverstone_map.yaml` was missing (only `Silverstone.yaml` existed) → the loader tried to
  re-download and failed. Created the alias so Silverstone loads. (Monza still not downloaded.)

## 8. Still open
- Get the teammate's exact gap-follower command to confirm/deny the bimodality.
- Add more real circuits (Monza + others) once map filenames are sorted.
- Resolve the eval episode-cap inconsistency (wrapper 3000 vs `--max-steps` 10000) before
  the formal grid.
