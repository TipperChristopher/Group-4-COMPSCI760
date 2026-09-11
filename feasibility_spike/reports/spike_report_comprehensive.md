# F1TENTH Feasibility Spike — Comprehensive Report

**Project:** COMPSCI 760 — PPO vs SAC across a varying number of training tracks
**Question the spike answers:** Can `f1tenth_gym` support this experiment, or should the proposal name MetaDrive instead?
**Nature of the spike:** a go/no-go check, executed end-to-end on real hardware. Not the experiment itself.
**Date:** 2026-08-13 · **Machine:** Windows 11, 16-core CPU, RTX 3060 Ti (8 GB), ~40 GB free RAM.

---

## 0. Bottom line

> **VERDICT: GO WITH CAVEATS.**

Everything the project depends on works: the simulator installs cleanly on Python 3.11, exposes a modern Gymnasium interface, produces LiDAR + vehicle-state observations, and both **PPO and SAC train** through Stable-Baselines3 with no crashes. PPO's episode return trends upward within 100k steps, confirming the learning loop is wired correctly. The track supply — the project's independent variable — is **not** a constraint: ~20 real circuits ship on demand plus an in-repo generator for **unlimited procedural closed-loop tracks**, so the 1/5/20/100 sweep is fully buildable.

Two caveats govern the experiment design, neither a blocker:

1. **SAC is slow on CPU.** A 5M-step SAC run is ~28 h (the plan's own 6 h alarm is exceeded). The bottleneck is SAC's per-step gradient update, not the simulator.
2. **The default replay buffer is large.** With full LiDAR, SB3's default 1M-transition buffer occupies **~8.9 GB** once filled — not the "<1 GB" the brief assumed. Fine on this 40 GB box; an OOM risk on a 16 GB laptop.

Both are mitigated by parallelizing across the 16 cores and downsampling the LiDAR beam count (a standard practice, e.g. BDEvan5's `f1tenth_benchmarks`).

---

## 1. How the spike was run (method)

Each of the plan's five questions was answered with a **command that ran and a number that was measured**, never an estimate. Where a claim could be self-validating (e.g. "install succeeded"), a second, independent check was added (e.g. an actual `reset()/step()` round trip; touching the full replay buffer to force physical memory commit). The working notes, hypotheses, and the one bug caught mid-flight are recorded in `journal.md`.

**Deliverables produced (all in `spike/`):**

| File | Purpose |
|---|---|
| `report.md` | the one-page deliverable the brief asked for |
| `REPORT_comprehensive.md` | this document |
| `journal.md` | live run journal — hypotheses, attempts, the bug, lessons |
| `sb3_wrapper.py` | the only "code" the spike required — a single-agent SB3 adapter |
| `q1_roundtrip.py` `q3_checkenv.py` `q4_ppo.py` `q5_sac.py` | evidence scripts, one per question |
| `fix_tracks.py` | 6-line shim for the track-filename blocker |
| `pip_freeze.txt` | exact reproducible environment (35 packages) |

---

## 2. Environment & install decisions

| Item | Value | Note |
|---|---|---|
| OS | Windows 11 (Git Bash / MINGW64) | — |
| Python | **3.11.2** (clean venv) | 3.14 is the machine default but too new; 3.11 is ideal for the RL stack |
| Simulator | `f1tenth_gym` **v1.0.0 branch**, commit `5a301bd` | gymnasium-native rewrite (see §3) |
| RL lib | `stable_baselines3` 2.9.0 | |
| DL backend | `torch` 2.13.0 **+cpu** | SB3 recommends CPU for `MlpPolicy` + single env; GPU wouldn't be the bottleneck |
| Native deps | numba 0.67, llvmlite 0.49, scipy 1.17, numpy 2.4.6, shapely 2.1, opencv 4.14, pygame 2.6 | all installed as **prebuilt cp311 wheels** |

**Install commands that worked (reproducible):**

```bash
py -3.11 -m venv venv
./venv/Scripts/python -m pip install --upgrade pip setuptools wheel
git clone --depth 1 --branch v1.0.0 https://github.com/f1tenth/f1tenth_gym.git f1tenth_gym_v1
./venv/Scripts/python -m pip install -e ./f1tenth_gym_v1
./venv/Scripts/python -m pip install stable-baselines3 torch psutil
./venv/Scripts/python fix_tracks.py     # one-time filename fix, see §4 / Q1
```

---

## 3. The critical version decision: legacy `main` vs `v1.0.0`

`f1tenth_gym` has two very different lines, and choosing correctly is what makes this a GO rather than a NO-GO.

| | Legacy `main` (`f110_gym 0.2.1`) | **`v1.0.0` (used)** |
|---|---|---|
| API | `gym==0.19.0` — **old-style**, 4-tuple step | `gymnasium ^0.29.1` — **modern 5-tuple** |
| numpy | `numpy<=1.22.0` — hard upper cap | `numpy>=1.18` — no cap |
| Python | "Windows must use 3.8 (2021)" | `python>=3.9` |
| Rendering | `pyglet<1.5` | `pygame` |
| Stock maps | 5 (berlin, levine, skirk, stata_basement, vegas) | on-demand DB + generator |
| SB3 fit | needs a gym→gymnasium shim **and** a numpy downgrade that fights SB3/torch | drops straight in with a thin wrapper |

Had the spike used the legacy branch on Windows, it would have collided with the old numpy pin against modern SB3/torch and the old `gym` API — likely producing a false NO-GO. The v1.0.0 branch removes both problems. **Recommendation: the project must pin the v1.0.0 branch (or later), not `pip install f1tenth-gym` / the default `main`.**

> **On the README warning you flagged** — "*Library support issues on Windows. You must use Python 3.8 as of 10-2021*" — it appears in **both** branch READMEs but is a **stale "Known issues" block**. It contradicts v1.0.0's own `pyproject` (`python>=3.9`), and its root cause (no Windows wheels for numba/llvmlite/scipy above 3.8 in 2021) no longer holds: the install here pulled cp311 wheels for all of them with zero source builds. Empirically obsolete for v1.0.0 on Python 3.11.

---

## 4. Findings per question (with evidence)

### Q1 — Does it install? **YES** (one trivial, fixed blocker)

- **Install:** `pip install -e .` completed with no build-from-source and no dependency conflicts.
- **Round trip (`q1_roundtrip.py`):** `make()` → `reset()` → `step()` succeeds headless; `step()` returns a **5-tuple** (Gymnasium), reward emitted (`0.01`).
- **Blocker found & fixed:** the loader requests `<track>_map.yaml`, but the live tarball from `http://api.f1tenth.org/<track>.tar.xz` ships `<track>.yaml` — a filename drift between the code and the current track server. Symptom:

  ```
  FileNotFoundError: ...\maps\Spielberg\Spielberg_map.yaml
  ```

  **Fix (`fix_tracks.py`, ~2 min):** after download, alias `<stem>.yaml → <stem>_map.yaml` for each track dir. Idempotent; covers procedurally-generated tracks too.

- **Note on the process:** "pip install succeeded" did **not** imply "it runs" — this bug only surfaced at env-load time. That is exactly why the spike insists on a real round trip as Q1's evidence.

### Q2 — How many tracks, and can more be generated? **STRONG YES** (the key question)

Three independent supply routes:

1. **~20 real closed circuits**, downloaded on demand from `http://api.f1tenth.org/<name>.tar.xz` (endpoint verified live — Spielberg 142 KB, Austin 167 KB, HTTP 200). The F1TENTH racetrack database:

   > Austin, BrandsHatch, Budapest, Catalunya, Hockenheim, IMS, Melbourne, Mexico City, Montreal, Monza, MoscowRaceway, Nuerburgring, Oschersleben, Sakhir, SaoPaulo, Sepang, Shanghai, Silverstone, Sochi, Spa, Spielberg, YasMarina, Zandvoort (≈23 dirs).

2. **5 legacy stock maps** on the old branch (berlin, levine, skirk, stata_basement, vegas).

3. **Unlimited procedural generation** — `examples/random_trackgen.py` ships in-repo, generating N random **closed-loop** circuits (`--n_maps --seed`). It builds checkpoints on a circle, connects them, and closes the loop, then writes each track in the native format.

**File format per track:**

- YAML spec: `image`, `resolution` (m/px), `origin [x,y,θ]`, `negate`, `occupied_thresh`, `free_thresh`.
- Occupancy grid: `.png` (+ `.pgm`).
- Centerline CSV: header `# x_m, y_m, w_tr_right_m, w_tr_left_m`; optional `_raceline.csv`.

**Verdict:** the 1/5/20/100 design is fully supported (20 real for the ≤20 cells; procedural fill to 100 and beyond for held-out test sets). Because these are **closed loops**, **lap time is a valid metric** — the specific weakness the brief noted for the MetaDrive fallback (open point-to-point routes) does not apply here.

### Q3 — Gymnasium-compatible for SB3? **YES, via a thin wrapper**

- **API style:** modern **gymnasium 0.29.1**, 5-tuple `step()`.
- **Raw spaces** (multi-agent-shaped):
  - `observation = Dict('agent_0': Dict('scan': Box(1080), 'pose_x', 'pose_y', 'delta', 'linear_vel_x', 'pose_theta'))`
  - `action = Box(low=[-0.4189, -5.0], high=[0.4189, 20.0], shape=(1,2))` = **[steering rad, speed m/s]**
- **SB3 acceptance:** not native — the doubly-nested per-agent Dict and the `(1,2)` action need adapting. A **~40-line single-agent wrapper** (`sb3_wrapper.py`) flattens `agent_0` to `Box(1085)`, maps action `(2,) ↔ (1,2)`, and unwraps scalar reward/terminated/truncated. With it, **`stable_baselines3.common.env_checker.check_env` PASSES**.
- **Observation choice:** `observation_config.type="features"` with `"scan"` in the feature list yields **LiDAR + state** (the project's stated observation). Verified finite over 50 steps (min −1.06, max 30.01, no NaN/inf), so training will not NaN. (One cosmetic warning — a beam reads 30.011 vs a declared max of 30.0 — is harmless.)

### Q4 — Does PPO train, and how fast? **YES**

PPO / `MlpPolicy`, 100k steps, headless, obs = 1085 (1080 LiDAR + 5 state), defaults, CPU:

| Measured | Value |
|---|---|
| Throughput | **757 env-steps/s** |
| Wall time (100k) | 132 s |
| Episodes completed | 43 |
| Return trend | first-third **12.3** → last-third **37.0** → **UP** ✔ |
| Extrapolated 5M-step run | **1.8 h** (under the 6 h alarm) |
| 12 PPO cells, serial | ~22 h |

The upward return trend confirms the loop learns (not just runs). PPO is comfortably feasible.

### Q5 — Does SAC run without memory problems? **YES it runs — but two assumptions were wrong**

SAC / `MlpPolicy`, 50k steps, same track, default `buffer_size=1_000_000`:

| Measured | Value |
|---|---|
| Throughput | **48.7 env-steps/s** |
| Wall time (50k) | 1027 s |
| Peak RSS at 50k | **0.94 GB** |
| Extrapolated 5M-step run | **~28.5 h** |

- **Memory correction.** The 0.94 GB peak is misleading: at 50k steps the 1M buffer is only ~5 % full. Physical memory is committed lazily as rows are **written**, so RSS grows with fill. Forcing the full buffer to commit (touching every row) measured **8.90 GB** for the full-LiDAR obs (1085-dim). The brief's "default 1M buffer well under 1 GB" is **false for full LiDAR** — the real steady state is ~8.9 GB. Fits this 40 GB machine; would OOM a 16 GB laptop.

  Buffer footprint vs beam count (obs + next_obs, float32, 1M transitions):

  | LiDAR beams | obs dim | Full 1M buffer |
  |---|---|---|
  | 1080 (full) | 1085 | **8.9 GB** |
  | 108 | 113 | **0.9 GB** ✔ (matches the brief's assumption) |
  | 54 | 59 | 0.47 GB |
  | 20 | 25 | 0.20 GB |

- **Speed correction & mitigation.** Downsampling LiDAR to 108 beams (obs 113) both shrinks the buffer to 0.9 GB **and** raises SAC throughput to **81 steps/s** (5M ≈ 17 h) — memory ~10× smaller, speed ~1.7× faster. Downsampling alone does not get SAC under 6 h; the residual cost is SAC's per-step gradient update.

---

## 5. Consolidated numbers

| Metric | PPO | SAC (full LiDAR) | SAC (108-beam) |
|---|---|---|---|
| FPS (measured, CPU) | **757** | **48.7** | 81.2 |
| Wall time / 5M-step run | **1.8 h** | **28.5 h** | 17.1 h |
| 12 runs, serial | 22 h | 342 h | 205 h |
| Peak RSS during spike | ~0.9 GB | 0.94 GB (buffer 5 % full) | 0.6 GB |
| **Full 1M buffer (steady state)** | n/a | **8.9 GB** | 0.9 GB |
| Return trending up in spike? | **Yes** (12→37) | not evaluated (expected flat early) | — |

**Environment facts:** stock maps ≈ 20 real circuits (+5 legacy) + unlimited procedural · obs = 1080-beam LiDAR + 5 state · action = [steer ±0.419 rad, speed −5…20 m/s], continuous.

### Full-grid compute estimate (24 runs @ 5M steps each)

| Scenario | Serial | 8-way parallel on 16 cores |
|---|---|---|
| Full LiDAR (1080 beams) | ~15 days | **~2 days** |
| Downsampled (108 beams) | ~9.5 days | **~1.2 days** |

Each SB3 CPU run mostly occupies 1–2 cores, so 6–8 concurrent runs fit comfortably. Parallelization is the single biggest lever and turns a 2-week serial job into ~1–2 wall-clock days.

---

## 6. Blockers & effort

| # | Blocker | Severity | Effort | Status |
|---|---|---|---|---|
| 1 | Loader expects `<name>_map.yaml`; server ships `<name>.yaml` | Trivial | ~2 min (`fix_tracks.py`) | **fixed** |
| 2 | SB3 rejects raw multi-agent Dict obs / `(1,2)` action | Expected | ~30 min (`sb3_wrapper.py`) | **fixed, check_env passes** |
| 3 | Default 1M buffer + full LiDAR ≈ 8.9 GB (OOM ≤16 GB) | Medium | ~10 min (downsample beams or set `buffer_size`) | mitigation measured |
| 4 | SAC 5M run ≈ 28 h on CPU (≫6 h) | Medium–High | parallelize + downsample (→17 h) ± CUDA / smaller budget | design-level |
| 5 | README py3.8/Windows note is stale | Cosmetic | none — ignore | n/a |
| 6 | Must pin v1.0.0 branch, not default `main`/PyPI | Setup | pin in requirements | documented |

---

## 7. A process note: the bug that would have corrupted the numbers

During Q4/Q5 the wrapped observation came out **2165-dim**, but the raw scan is 1080 beams (true obs 1085). Tracing it: `_flatten` added the scan **twice** — once in an explicit `"scan"` branch and again in the feature loop (which also contained `"scan"`), i.e. `1080 + 1080 + 5 = 2165`. This silently:

- inflated the SAC memory figure **2×** (reported 17.5 GB → corrected **8.9 GB**), and
- **understated** PPO/SAC throughput (obs twice as wide as reality).

After fixing the double-count (obs verified 1085 / 113) the affected runs were **re-executed**; PPO rose 613 → **757 FPS**, and all §5 numbers are the corrected values. **Lesson recorded:** print and check `obs_dim` before trusting any throughput or memory number — a one-line shape bug can quietly double a headline figure.

---

## 8. Recommendations if the project proceeds on F1TENTH

1. **Pin `f1tenth_gym` v1.0.0** (gymnasium-native). Keep the `fix_tracks.py` shim and the `sb3_wrapper.py` adapter in the repo from day one.
2. **Downsample LiDAR to ~100 beams.** Fixes the buffer to <1 GB, speeds SAC ~1.7×, and matches established F1TENTH RL practice (BDEvan5). Keep full state features.
3. **Budget SAC deliberately.** At ~17–28 h per 5M SAC run, either cap the per-cell step budget (RL here plateaus early anyway) or accept multi-day parallel schedules. Consider a CUDA torch build to accelerate SAC's gradient step (untested here — the 3060 Ti was idle behind CPU torch).
4. **Parallelize across the 16 cores** — the difference between a 2-week serial run and a ~1–2 day campaign.
5. **Reuse an existing reward/observation setup** (BDEvan5's `f1tenth_benchmarks`) rather than inventing one; the brief already flags it as a working reference for LiDAR + speed observations.
6. **Set expectations on driving quality:** published work notes online RL in F1TENTH plateaus at suboptimal returns and often fails to complete laps (SAC the most stable). Poor lap completion at these budgets is expected and is **not** a failure of the machinery — which is what this spike validated.

---

## 9. Single biggest risk

**SAC wall-clock throughput.** SAC's per-step gradient update (`train_freq=1`) makes it ~15× slower than PPO on the *identical* environment — the limiter is the learner, not the simulator or the observation size. A single 5M-step SAC run is 17–28 h, and the twelve SAC cells dominate the entire compute budget. The project is feasible **only with** run-level parallelization plus at least one SAC cost reduction (downsampled beams, a capped step budget, or GPU gradient updates). PPO on its own is comfortably within budget. The secondary risk is memory: the default replay buffer (~8.9 GB at full LiDAR) must be shrunk or the beams downsampled before running on any ≤16 GB machine.

---

## 10. Caveat on scope

The folder contained only `plan.md`; no repository was attached with the request ("this would be the repo" pointed to nothing present). The spike therefore used the **official F1TENTH GitHub simulator** the plan names as its reference point. If a specific fork was intended, the same evidence scripts re-run against it in minutes.

---

## Appendix A — the SB3 adapter (`sb3_wrapper.py`)

The only code the project genuinely needs to add to make SB3 consume the env: flatten `agent_0`'s `{scan, state…}` dict into one `Box` vector, convert the action between `(2,)` and `(1,2)`, and pass the env's own reward through as a scalar. Optional `scan_beams=N` uniformly subsamples the LiDAR (the Q5 mitigation). No reward engineering, no tuning — a faithful feasibility harness.

## Appendix B — reproduce the evidence

```bash
./venv/Scripts/python q1_roundtrip.py    # Q1: reset/step round trip + spaces
./venv/Scripts/python q3_checkenv.py     # Q3: prints spaces, runs check_env
./venv/Scripts/python q4_ppo.py          # Q4: PPO 100k, FPS, return trend, extrapolations
./venv/Scripts/python q5_sac.py          # Q5: SAC 50k, peak RSS, buffer math
```

## Appendix C — key package versions (`pip_freeze.txt`)

```
f1tenth_gym @ github.com/f1tenth/f1tenth_gym@5a301bd (v1.0.0 branch, editable)
gymnasium==0.29.1
stable_baselines3==2.9.0
torch==2.13.0+cpu
numba==0.67.0      llvmlite==0.49.0
numpy==2.4.6       scipy==1.17.1
shapely==2.1.2     opencv-python==4.14.0.94
pygame==2.6.1      psutil==7.2.2
```
