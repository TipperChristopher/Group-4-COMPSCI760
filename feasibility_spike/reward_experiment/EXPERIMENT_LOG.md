# EXPERIMENT LOG — complete session record (handover for the next session)

Branch: `experiments/feasibility-spike` (merged with origin/main, pushed, all work
additive under `feasibility_spike/`). Venv python:
`E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760/spike/venv/Scripts/python.exe`
Repo: `E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760/team_repo`
This folder: `feasibility_spike/reward_experiment/`

---

## 0. Where things stand (30-second summary)

Two open problems after the pilots:
1. **PPO cannot hold a policy** — peaks 0.34 laps @ 300k then oscillates/degrades to
   ~0.10-0.13; reward swings +37..-19. Verified NOT fixable by: more steps (2M, 3M),
   VecNormalize, crash-penalty value, reward variant, or dense forward-proximity
   shaping. Root cause: on-policy forgetting + single-env variance + safety-blind
   reward. SAC (replay buffer) completes laps under identical conditions.
2. **Zero-shot fails on real circuits** — SAC trained on ONE synthetic track completes
   UNSEEN synthetic tracks (1.0 laps, 5/5 on track_2) but crashes within metres of the
   spawn on real circuits. Verified at action level: creeps ~20 s then panic-steers
   into the wall; reproduces on a NARROW synthetic track (same spawn scan). Not
   overfitting (checkpoint study: unseen performance rises monotonically 0.42->1.00).
   It is INPUT-DISTRIBUTION coverage + a generator ceiling (synthetic corners capped
   ~0.54 1/m vs real 1.27). The next session's job: the fixes in section 4.

---

## 1. The frozen protocol (what every pilot used)

- Reward: `1.0 * metres of centreline progress - CRASH_PENALTY (on collision)`,
  TIME_COST = 0 (new_reward_wrapper.py). Pilots used penalty 40; the TEAM's frozen
  value per the handover is 5. **DECISION NEEDED before the grid.**
- VecNormalize: observations only (norm_reward=False), identical for PPO and SAC.
- One env + track-pool sampler (n_envs=1 by design — do not change).
- Budget: 2M env-steps (20 x 100k segments; eval after each segment is deterministic).
- Eval: centreline spawn (cl_grid_static), 5 fixed seeds 0-4, one-lap termination,
  15,000-step cap, greedy policy.
- Baselines: Follow-the-Gap (safe_threshold=20, team's commit 3616d54) + Random,
  through the identical eval.

Key harness files: `learning_curve.py` (trainer: resumable checkpoints, --old-reward,
--shaping, --keep-checkpoints), `eval_vs_baseline.py` (RL + baselines, same protocol),
`reward_scan2.py`, `spawn_study.py`, `seed_probe.py`, `gen_gap_figure.py`,
`_make_reward_compare_fig.py`, `_insert_our_slides.py` (deck builder).
Docs: `INVESTIGATION.md` (the two problems + fixes), `REWARD_ANALYSIS.md`,
`SPAWN_FIX_RESULTS.md`, `TRAIN_README.md`, `RESUME.txt`.

---

## 2. What we ran: why -> result -> meaning

### A. Spawn / protocol investigation
- **Why:** teammate found train (centreline) vs eval (raceline, near kerb) mismatch +
  single-episode instability; proposal claimed "same starting positions".
- **Ran:** traced the library (rl_grid_static default; raceline==centreline fallback
  on synthetic; MaskedResetFn samples ~1 m of start line from the global RNG);
  seed_probe verified `--eval-seed` propagates through DummyVecEnv AND VecNormalize;
  spawn_study before/after on Spielberg/Silverstone.
- **Got:** fix = `cl_grid_static` + pinned seeds; off-centreline 0.79-0.81 -> 0.00 m;
  ~1 m start line yields only 3-4 DISTINCT spawn positions (so "5 seeds" weights ~3
  positions). On Silverstone the raceline spawn was accidentally RESCUING the
  gap-follower 2/5 seeds (team CSV); centreline makes it a deterministic 5/5 crash at
  the same hairpin. Honest scope: a fairness/reproducibility fix, not a performance one.
- **Meaning:** protocol is now fair and reproducible; spawn is folded into one bullet
  on the deck's protocol slide.

### B. Reward investigation
- **Why:** old reward's optimum was a crawl (time payment); need the precise
  before/after + whether the fix changed agent performance.
- **Ran:** reward_scan2.py (same straight-line rollout scored under BOTH formulas,
  frozen penalty 40); trained PPO under three reward variants to 2M (original V1
  incl. -5-on-terminated, new-p5, new-p40); early completion_old/new ablation re-read.
- **Got:** OLD formula `1.0/s + 0.1*speed - 0.5*|steer| - 5(term)`: standstill 100.00,
  crawl 0.25 = 263 peak, 20 m/s = 189 (slower = higher). NEW: standstill 0.00, wall
  crash = -22 at ANY speed (metres only). All THREE PPO variants plateau ~0.1 laps
  over 2M — the reward change fixed the INCENTIVE, not PPO's stability.
- **Meaning:** reward slide says exactly that (design framing, not "bug"). SAC was
  never trained under the old reward (open question if we want it).

### C. PPO vs SAC
- **Ran:** matched learning curves (VN, penalty 40, 1M then 2M), PPO-2M and SAC-1M
  in the deck; SAC re-run reproduced bit-identically (determinism proof).
- **Got:** PPO peaks 0.34 @ 300k -> 0.13 @ 2M, reward std 14 > mean 10; SAC completes
  2 laps from 400k and keeps getting faster (40.3 -> 24.7 s @ 1M; 22.7 s @ 1.2M).
  Crash forensics: PPO's crash SPEED rises with training (2.3 -> 7.0 m/s) and the
  final policy accelerates into the sharpest corner (no braking).
- **Meaning:** under the fixed-budget single-env protocol, replay-buffer SAC wins;
  PPO's problem is memory/variance, not reward or budget. (Survey predicted this.)

### D. Baselines (aligned protocol)
- **Ran:** gap-follower + random through the identical eval (5 tracks x 5 seeds).
- **Got:** gap completes Spielberg (1.0 lap, 67 s), fails Silverstone at the same
  hairpin every seed; on synthetic_0 gap is seed-sensitive (0.09-1.33, mean 0.74);
  random ~0.01-0.05 everywhere.
- **Meaning:** floor/ceiling brackets for the RL rows; the reactive baseline currently
  wins out-of-distribution — the gap the diversity sweep must close.

### E. Zero-shot generalization + the action-level mechanism
- **Ran:** PPO-2M and SAC-1M evals on unseen synthetic + real circuits (5 seeds).
- **Got:** SAC: synth_1 0.80 (2/5 finish), synth_2 1.00 (5/5, 22.4 s), Spielberg 0.106,
  Silverstone 0.057 (5/5 crash). PPO: 0.13-0.24 on synthetic, ~0.013 on real (= random).
  Action-level replay (verified, not assumed): on real circuits SAC CREEPS ~20 s at
  0.2-0.7 m/s then commands full lock + ~17 m/s and rams the wall from standstill;
  crash at ~34 m (Spielberg) / ~26 m (Silverstone), on the opening stretch, well
  before the sharp corners it never reaches (an earlier "2.7-10.7 m" figure of ours
  was a VecNormalize-loader artifact — see wiki/incidents.md). The SAME
  failure reproduces on a NARROW synthetic track (parallel narrow-track experiment).
- **Meaning:** not memorization (unseen synthetic completed) — it is scan-to-action
  brittleness outside the training manifold ("coverage, not real-track magic").
  Geometry audit: generator caps corner sharpness ~0.54 (TRACK_TURN_RATE=0.31);
  real circuits reach 1.27 (Spielberg) on laps 2-2.5x longer with more straights.

### F. Overfitting diagnostic (your "fewer steps" question)
- **Ran:** eval of SAC checkpoints 100k/300k/600k/1M zero-shot (from the parallel
  session's completed keep-checkpoints run; evidence eval_sac_ck*.json).
- **Got:** synth_1: 0.42 -> 0.69 -> 0.91 -> 1.00 (finishes 0 -> 0 -> 2 -> 5/5);
  real circuits flat-low (0.02-0.17) at EVERY checkpoint.
- **Meaning:** NOT overfitting — longer training monotonically helps unseen-synthetic
  performance; the real-circuit gap exists from the first checkpoint. "Train less"
  is not the fix; coverage is.

### G. Slides
- Built `Group4_ProjectUpdate_FINAL2.pptx` from the team's v2 (originals untouched)
  with our protocol / reward / PPO-vs-SAC / why-PPO / zero-shot / conclusions slides,
  dark theme, speaker notes with evidence. Regenerate via `_insert_our_slides.py`.

---

## 3. Current state of runs (resumable via RESUME.txt)

| dir (saved_models/) | state |
|---|---|
| sac_overfit | 7/20 (700k) — RUNNING (2M keep-checkpoints extension) |
| sac_vn_cp40 | 12/20 (1.2M) — paused (main SAC; model used for SAC-1M evals) |
| sac_ckpts | 10/10 (1M) — complete (source of the overfit diagnostic) |
| ppo_old_v1 / ppo_new_p5 / ppo_vn_cp40 | 20/20 — complete (reward-variant set) |

Resume = rerun the same learning_curve.py command (auto-detects state.json).

---

## 4. Next steps (the fix queue, ranked)

1. **Widen the synthetic generator** (raise TRACK_TURN_RATE / allow smaller radii so
   synthetic max_k ~ 1.3, plus width variety) — NECESSARY or the diversity sweep has a
   built-in ceiling. Then train ONE SAC track on the wider pool and re-eval real
   circuits. *(Not done yet — this directly answers "did we train on improved tracks".)*
2. **PPO n_envs=8 CONSTANT ablation** (2M, frozen protocol) — the key remaining PPO
   question (variance vs update count).
3. **Domain randomization on scans** (jitter/dropout) for SAC — widens the effective
   input manifold, orthogonal to the diversity axis.
4. **All-around clearance shaping** (our forward-only version got gamed) — braking
   gradient for both algos.
5. **Diversity sweep** 2 x 4 x >=3 seeds with bootstrap CIs, on the FINAL frozen setup
   (crash penalty decided, generator frozen).
6. Optional: SAC under the old reward (answers "did the reward matter for SAC?").

## 5. Open decisions
- Crash penalty: 5 (team frozen per handover) vs 40 (our sweep) — FREEZE before grid.
- Generator widening = a protocol change; freeze the new generator + re-verify.
- Main now contains the merged reward (PR #2); our vendored wrappers are consistent
  with it but redundant (cleanup optional).

## 6. Honesty notes
- All pilots n=1 training seed; eval uses 5 spawn seeds (which map to ~3-4 distinct
  spawn positions — report that if asked).
- Tests: `tests/check_reward.py` + `tests/check_experiment.py` both green after
  generating the 20-track pool.
