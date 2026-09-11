# F1TENTH Feasibility Spike — Report

**Environment:** Windows 11, Python **3.11.2** (clean venv), CPU torch, RTX 3060 Ti present but unused (MlpPolicy + single env → CPU is standard). 16 cores, ~40 GB free RAM. Simulator: **`f1tenth_gym` v1.0.0 branch** (gymnasium-native), commit `5a301bd`.

## 1. Verdict: **GO WITH CAVEATS**

The machinery works end-to-end: install, LiDAR+state observations, SB3 PPO and SAC training loops all run headless, and PPO return trends upward. Two caveats shape the experiment design: **(a) SAC throughput is slow** — a 5M-step SAC run is ~28 h on CPU, well over the 6 h alarm; **(b) the default 1M replay buffer with full LiDAR needs ~8.9 GB**, not "<1 GB" as assumed. Both are manageable (parallelize, downsample LiDAR, and/or size the step budget), and none is a blocker. The 1/5/20/100 track sweep is fully supported.

## 2. Answers to Q1–Q5

**Q1 — Does it install?** YES. Clean `pip install -e .` on Python 3.11.2 pulled **prebuilt cp311 wheels** for every native dep (numba 0.67, llvmlite 0.49, scipy 1.17, numpy 2.4.6, shapely 2.1, opencv 4.14, pygame 2.6) — zero build-from-source, zero conflicts. `env.reset()`+`env.step()` round trip succeeds; `step()` returns a 5-tuple. *The README "Windows must use Python 3.8 (as of 10-2021)" note is a stale known-issue block that contradicts v1.0.0's own `pyproject` (`python>=3.9`); it is obsolete now that cp311 wheels exist.* **One trivial blocker:** the loader expects `<track>_map.yaml` but the live `api.f1tenth.org` tarball ships `<track>.yaml` — fixed by a 6-line rename shim (`fix_tracks.py`), ~2 min.

**Q2 — How many tracks, and can more be generated?** (the key question) **Strong yes.** Three pools: (i) **~20 real closed circuits** — Austin, Silverstone, Monza, Spa, Spielberg, Catalunya, Sepang, Sakhir, Shanghai, Zandvoort, … — fetched on demand from `http://api.f1tenth.org/<name>.tar.xz` (endpoint **live**, verified). (ii) **5 legacy stock maps** on the old branch. (iii) **Unlimited procedural generation**: `examples/random_trackgen.py` emits N random **closed-loop** circuits (`--n_maps --seed`) in the native format (occupancy `.png`/`.pgm` + `_centerline.csv`). **Format per track:** YAML spec (`image`, `resolution`, `origin`, thresholds) + occupancy image + centerline/raceline CSV (`x_m, y_m, w_tr_right_m, w_tr_left_m`). The 1/5/20/100 design holds, and unlike MetaDrive these are closed loops → **lap time is available**.

**Q3 — Gymnasium-compatible for SB3?** YES, via a thin wrapper. API is **modern gymnasium 0.29.1** (5-tuple step). Raw spaces are multi-agent-shaped: `observation = Dict('agent_0': Dict('scan': Box(1080), 'pose_x', 'pose_y', 'delta', 'linear_vel_x', 'pose_theta'))`, `action = Box((1,2))` = [steer ∈ ±0.419 rad, speed ∈ −5..20 m/s]. SB3 does **not** accept this natively; a ~40-line single-agent wrapper (flatten `agent_0` → `Box(1085)`, action `(2,)↔(1,2)`, scalar reward) makes **`stable_baselines3.common.env_checker.check_env` PASS**. (`observation_config.type="features"` with `"scan"` in the feature list yields LiDAR+state; obs is finite, no NaN/inf.)

**Q4 — Does PPO train, how fast?** YES. PPO/MLP, 100k steps, headless, obs=1085 (1080 LiDAR + 5 state): **757 env-steps/s (measured)**. Episode return trended **upward** (first-third 12.3 → last-third 37.0). Extrapolation: **one 5M run ≈ 1.8 h** (under the 6 h alarm). 12 PPO cells ≈ 22 h serial.

**Q5 — Does SAC run without memory problems?** Runs cleanly (50k steps, no crash) but **two corrections to the assumption**. Speed: **48.7 steps/s** → a 5M SAC run ≈ **28.5 h**. Memory: peak RSS at 50k was only 0.94 GB *because the buffer was 5 % full*; the **default 1M buffer with full LiDAR occupies ~8.9 GB once filled** (measured by touching the full arrays) — **not "<1 GB"**. Fits this 40 GB box; would **OOM a 16 GB laptop**. **Mitigation (measured):** downsample LiDAR to 108 beams → obs 113 → **buffer 0.9 GB** *and* **SAC 81 steps/s** (5M ≈ 17 h) — fixes memory ~10× and speed ~1.7×. Alternatively shrink `buffer_size`.

## 3. Measured numbers

| Metric | PPO | SAC (full LiDAR) | SAC (108-beam) |
|---|---|---|---|
| FPS (measured, CPU) | **757** | **48.7** | 81.2 |
| Time / 5M-step run | **1.8 h** | **28.5 h** | 17.1 h |
| Time / 12 runs (serial) | 22 h | 342 h | 205 h |
| Peak RSS @ spike run | ~0.9 GB | 0.94 GB (50k) | 0.6 GB |
| **Full 1M buffer (steady state)** | n/a | **8.9 GB** | 0.9 GB |
| Return trending up? | **Yes** (12→37) | not evaluated (expected flat early) | — |

**Stock map count:** ~20 real circuits (api.f1tenth.org DB) + 5 legacy + **unlimited procedural**. **Obs:** 1080-beam LiDAR + 5 state. **Action:** [steer, speed], continuous.
Full grid (24 runs @ 5M): ~15 days serial full-LiDAR / ~9.5 days downsampled; **~1–2 days at 8-way parallel** on the 16-core box.

## 4. Blockers found

| # | Blocker | Severity | Effort to resolve |
|---|---|---|---|
| 1 | Track loader expects `<name>_map.yaml`; server ships `<name>.yaml` | Trivial | **~2 min** — `fix_tracks.py` rename shim (done) |
| 2 | SB3 rejects raw multi-agent Dict obs / (1,2) action | Expected | **~30 min** — `sb3_wrapper.py` (done, check_env passes) |
| 3 | Default 1M buffer + full LiDAR = ~8.9 GB (OOM risk on ≤16 GB) | Medium | **~10 min** — downsample beams (108→0.9 GB) or set `buffer_size` |
| 4 | SAC 5M run ≈ 28 h on CPU (≫6 h) | Medium–High | Parallelize (16 cores) + downsample (→17 h) + optional CUDA torch / smaller budget |
| 5 | README py3.8/Windows note is stale | Cosmetic | none — ignore; use py3.11 |

## 5. Single biggest risk

**SAC wall-clock throughput.** The bottleneck is SAC's per-step gradient update (`train_freq=1`), which makes it ~15× slower than PPO on the *same* environment — not the simulator and not obs size. A single 5M-step SAC run is 17–28 h; the 12 SAC cells dominate the compute budget. The project is feasible **only if** the team (a) parallelizes runs across the 16 cores, and (b) reduces the SAC cost — downsample LiDAR beams, cap the per-cell step budget below 5M, and/or move SAC's gradient step to the GPU (CUDA torch, untested here). PPO alone is comfortably feasible. Secondary risk: the ~8.9 GB default buffer must be shrunk (or beams downsampled) before running on any ≤16 GB machine.

---
### Attachments (in this folder)
- **Install commands that worked:** `py -3.11 -m venv venv` → `pip install -e ./f1tenth_gym_v1` → `pip install stable-baselines3 torch` → `python fix_tracks.py`
- `pip_freeze.txt` — full environment (35 pkgs; key: f1tenth_gym@5a301bd, gymnasium 0.29.1, stable_baselines3 2.9.0, torch 2.13.0+cpu, numba 0.67.0, numpy 2.4.6)
- `sb3_wrapper.py` — minimal single-agent SB3 adapter (the only "code" the spike required)
- `q1_roundtrip.py`, `q3_checkenv.py`, `q4_ppo.py`, `q5_sac.py`, `fix_tracks.py` — evidence scripts
