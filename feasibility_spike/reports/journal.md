# Run Journal — F1TENTH Feasibility Spike

**Status:** DONE — all 5 questions answered with evidence. Verdict: GO WITH CAVEATS. Report at `report.md`.
**Last updated:** end of spike.

## §1. User request (verbatim)
> read the plan, and i want to know whether u can verify that or nmot, this would be the repo
> do it

**Agent paraphrase:** Read `plan.md` (an F1TENTH feasibility spike brief), judge whether I can actually verify its 5 go/no-go questions in this environment, then execute the spike and produce the one-page report the plan asks for.

## §2. Success criteria (the plan's own deliverable)
- [x] Q1 install: round trip OK on py3.11; pip_freeze.txt saved. MET.
- [x] Q2 tracks: ~20 stock circuits + api.f1tenth.org + unlimited procedural (random_trackgen.py). MET.
- [x] Q3 API: gymnasium 0.29.1 5-tuple; wrapper required; check_env PASSED. MET.
- [x] Q4 PPO: 757 FPS, 5M=1.83h, return trend up. MET.
- [x] Q5 SAC: 48.7 FPS, peak RSS 0.94GB@50k, full buffer 8.9GB (NOT <1GB). MET (assumption corrected).
- [x] Deliverable: report.md + pip_freeze.txt + sb3_wrapper.py + q*.py. MET.

## §3. Environment (measured)
- Windows 11, Git Bash (MINGW64). Python 3.11 & 3.12 available via `py` launcher (3.14 is default but too new).
- RTX 3060 Ti 8GB, 16 CPU, ~40GB free RAM, internet OK. No conda → use venv on py 3.11.

## §4. Constraints from plan
- No reward fn / sampler / eval harness / grid. No HP tuning. MLP not CNN. Rendering off for timing. Report broken-but-fixable with effort.

## Attempts

### Attempt 1 — Q2 (tracks) — recon of both f1tenth_gym versions
- Action: cloned `main` (legacy 0.2.1) and `v1.0.0`; inspected setup, maps, track code.
- Observed:
  - Legacy `main`: `gym==0.19.0` (old API), `numpy<=1.22`, Windows needs py3.8. 5 stock maps (berlin/levine/skirk/stata_basement/vegas).
  - `v1.0.0`: `python>=3.9`, `gymnasium ^0.29.1` (modern 5-tuple), `numpy>=1.18`. `maps/` empty except `.gitkeep`; tracks fetched on demand from `http://api.f1tenth.org/<name>.tar.xz` (endpoint LIVE — Spielberg 142KB, Austin 166KB served 200 OK).
  - f1tenth_racetracks DB = ~20 real closed circuits (Austin, Silverstone, Monza, Spa, Spielberg, Catalunya, Sepang, Sakhir, ...).
  - `examples/random_trackgen.py` in BOTH repos: generates N random CLOSED-LOOP tracks (`--n_maps --seed`), outputs `mapN_map.png`+`.pgm`+`mapN_centerline.csv`. Adapted from CarRacing-v0.
- Verdict: PASS. Q2 answered: stock ~20 + API + unlimited procedural generation. Supports 1/5/20/100. Closed loops (lap time available), unlike MetaDrive.
- Decision: target **v1.0.0** (gymnasium-native) for the SB3 spike.

### Attempt 2 — Q1 (install + round trip)
- Action: py3.11 venv; `pip install -e f1tenth_gym_v1`; then round-trip script.
- Observed: install clean — prebuilt cp311 wheels for numba 0.67/llvmlite 0.49/scipy 1.17/numpy 2.4/gymnasium 0.29.1, ZERO build-from-source. README "Windows must use py3.8 (10-2021)" is a STALE known-issue block (present in both branches) contradicting v1.0.0 pyproject `python>=3.9`; obsolete because cp311 wheels now exist. First run FAILED: loader wants `<name>_map.yaml`, api.f1tenth.org tarball ships `<name>.yaml` — filename drift. FIX: `fix_tracks.py` aliases yaml (~2 min). After fix: make/reset/step OK, step returns 5-tuple, reward 0.01.
- Verdict: PASS (with 1 trivial fixable blocker). Q1 answered.

### Attempt 3 — Q3 (SB3 compatibility)
- Action: built `sb3_wrapper.py` (flatten nested agent_0 dict -> Box, action (2,)->(1,2), scalar reward); ran `check_env`.
- Observed: raw obs = `Dict(agent_0: Dict(...))`, action `Box((1,2))` -> SB3 needs wrapper (confirmed). With `features=[scan]+state`: wrapped obs `Box(2165,)` = 2160 LiDAR beams + 5 state; action `Box(2,)`. `check_env` PASSED. Obs finite (min -1.06/max 30.01, no NaN/inf over 50 steps). Only a cosmetic 'obs not within space' warn (raw scan 30.011 > declared 30.0).
- Verdict: PASS. Q3 answered: modern gymnasium 5-tuple; ~40-line single-agent wrapper required (standard).

### Attempt 4 — Q4/Q5 timing+memory, and a BUG that inflated obs 2x
- Action: PPO 100k, SAC 50k, direct buffer-touch, downsample probe.
- Observed (first pass): obs_dim=2165. Chased it -> `_flatten` added scan TWICE (explicit block + feature loop both include 'scan'). Raw scan is 1080 beams; true full obs = 1085, not 2165. FIXED (skip 'scan' in loop), re-verified obs_dim 1085 / 113.
  - CORRECTED PPO: 757 FPS (obs 1085), 5M=1.83h, 24-run(mixed illustrative)=44h, return 12->37 UP.
  - CORRECTED SAC: 48.7 FPS, 5M=28.5h, peak RSS 0.94GB@50k (buffer 5% full). Direct full-1M-buffer touch @obs1085 = 8.90GB (NOT <1GB).
  - Downsample 108 beams (obs 113): buffer 0.9GB AND SAC 81.2 FPS (5M=17.1h) -> fixes memory ~10x + speed ~1.7x.
- Verdict: PASS. Lesson: verify obs_dim before trusting FPS/memory; the double-scan bug made memory 2x too high and FPS pessimistic. Re-ran after fix.

## Lessons
- README "known issues" can be stale (py3.8/Windows) and contradict pyproject; trust the empirical install (cp311 wheels).
- "pip install succeeded" != runs: the yaml-name drift only appeared at env load.
- Always print obs_dim before trusting throughput/memory numbers — a silent double-count doubled memory and halved apparent FPS.
- SAC per-step gradient update (train_freq=1) is the throughput bottleneck, not obs size or the sim; it makes SAC ~15x slower than PPO on the same env.
