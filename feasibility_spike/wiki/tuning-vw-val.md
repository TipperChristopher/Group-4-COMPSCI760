---
title: Teammate tuning round on vw_val — verified (time-cost reward + final-grid settings)
type: synthesis
updated: 2026-10-07
sources:
  - _verify/teammate_tuning_vw/tuning_vw_val_results/ (raw CSVs; canonical copy: desmond/heldout-tracks @ 6e5b60e)
  - results/tuning_vw_val_verified/ (README + summary copies + verification notes)
---

# Tuning round on vw_val — what it is, and what we verified

A teammate ran a pre-registered tuning round to pick the final-grid settings:
**16 runs = 8 settings × 2 seeds (98, 99)**, each 2 M steps on the first 20
varied-width tracks (`vw_synthetic_track_0..19`, half-widths 0.63–1.44 m),
crash penalty 40, scored on **vw_val** (10 unseen varied-width tracks × 5
episodes, deterministic, centreline spawn, one-lap termination). Code:
`desmond/heldout-tracks` @ `6e5b60e`, reward sha `9b73ea…`, all 16 runs clean
and completed.

| setting | algorithm | flags on top of defaults | what it tests |
|---|---|---|---|
| S0 | SAC | none (γ 0.99) | baseline |
| S1 | SAC | `--gamma 0.999` | longer value horizon |
| S2 | SAC | `--gamma 0.999 --time-cost 0.01` | **S1 + per-step time cost (anti-crawl)** |
| S3 | SAC | `--gamma 0.999 --buffer-size 2000000` | no buffer overwriting |
| P0 | PPO | none (γ 0.99, n_steps 2048) | baseline |
| P1 | PPO | `--gamma 0.999 --n-steps 8192` | our d1 fix, at d20 |
| P2 | PPO | `--gamma 0.999 --n-steps 8192 --time-cost 0.01` | **P1 + the time cost** |
| P3 | PPO | `--gamma 0.999 --n-steps 8192 --target-kl 0.03 --ent-coef 0.01` | our proposed stability arm |

## What we verified (arm's length, from raw files)

- **Configs:** all 16 `run_config.json` files match the table above, status
  completed, no dirty files, commit `6e5b60e`. (P0's `n_steps` is `None` = the
  SB3 default 2048, as documented.)
- **The time cost really reaches the reward.** From raw per-episode logs:
  `(progress − return)/length` is 0.0100 per step in the four S2/P2 runs and
  0.0000 in the other twelve; crash episodes pay exactly 40 in all 16.
  *Correction to their README:* the claim "0.0100–0.0100 per step" is slightly
  overstated — ~12% of episodes (lap-boundary progress-wrap logging) show a
  small negative tail; the dominant value is exactly 0.0100.
- **Scores reproduce.** All 30 spot-checked checkpoint scores recomputed from
  the 80 raw eval CSVs match their `summary.csv` to ≤5e-5.
- **Same eval defects as the grid:** 5 episodes/track give 3 distinct spawns
  (eval_seed 0≡1, 2≡3) — their caveat, confirmed.
- **Evaluation uses the canonical reward** (no time cost), so all settings are
  scored on the same scale: a lap episode returns exactly its progress, a crash
  exactly progress − 40.
- **Internals spot checks:** S1's critic loss explodes to 3789 (their note 1
  ✓). But *their claim that the entropy coefficient blows up is not supported
  by the log* — `ent_coef` declines smoothly 0.884 → 0.013, never rising.
  P3's std rises 1.0 → 2.05 (their 1.5–1.9, roughly ✓). P0 s98 explained
  variance ends at −38.8 (their "about −3.5" was at an earlier point).

## How the time cost works (the mechanism)

`--time-cost 0.01` subtracts 0.01 from every step's reward. At speed v (m/s)
each 0.01 s step earns 0.01·v progress, so the net is

```
0.01·v − 0.01 = 0.01·(v − 1)
```

**Break-even at exactly 1 m/s.** Any speed below 1 m/s loses reward every step;
crawling is no longer free. In value terms a constant per-step cost is worth
`0.01/(1−γ)`: **1.0 at γ 0.99, 10.0 at γ 0.999** — the anti-crawl force is
10× stronger under the long horizon. That matches the data: the tax rescues
the γ-0.999 runs (S1 → S2) but changes little for γ-0.99-style baselines.

*Why it was tried:* progress-only reward gives slow-and-safe a free ride, and
γ 0.999 makes it worse (slow progress still counts almost fully). Crawling was
observed: S0 s99 trains at 2.43 m/s with 84% of episodes timing out; S1 seeds
crawl at 0.38–2.29 m/s.

## Results (best-checkpoint fractional laps on vw_val, mean of 2 seeds)

| setting | seed 99 | seed 98 | average | vs baseline | notes |
|---|---|---|---|---|---|
| S0 (SAC base) | 0.453 | 0.566 | **0.509** | — | crawling ×1 |
| S1 (γ .999) | 0.277 | 0.214 | 0.246 | −0.264 | collapses after 400 k |
| S2 (+ time cost) | 0.483 | 0.396 | 0.439 | **−0.070** | no crawling; 86% training crash |
| S3 (+ 2M buffer) | 0.277 | 0.214 | 0.246 | −0.264 | identical to S1 until 1 M |
| P0 (PPO base) | 0.175 | 0.109 | **0.142** | — | |
| P1 (our fix) | 0.028 | 0.126 | 0.077 | −0.064 | **does not beat P0 at d20** |
| P2 (+ time cost) | 0.071 | 0.491 | 0.281 | +0.139 | **one seed only** (gap 0.42) |
| P3 (ent+KL) | 0.044 | 0.034 | 0.039 | −0.103 | optimiser stable, policy crawls |

## Verdict

- **The time cost is a legitimate, well-calibrated anti-crawl mechanism**
  (break-even 1 m/s; the 1/(1−γ) interaction is a nice first-principles point
  for the report). It does exactly what it says: taxed runs train at 7.3–7.7
  m/s and never crawl.
- **It does not beat the baselines at d20.** S2 scores below plain SAC (0.439
  vs 0.509) and crashes 86% of training episodes. P2's lead over P0 rests on
  one seed (0.491 vs 0.071, gap 0.42 ≈ the known noise floor). So: relevant as
  a negative result and a mechanism — **not as the final-grid setting**.
- **The bundle answers three of our open questions anyway:** (1) γ 0.999 alone
  *breaks* SAC at d20 (Q11 — see open-questions); (2) our PPO fix does not
  transfer to d20 varied-width training in 2 seeds (Q10, partial); (3) the
  ent_coef+target_kl stability arm fails — it stabilised the optimiser (KL
  0.007) but the entropy bonus inflated σ to ~2.0 and the policy crawled.

## Follow-up (2026-10-07): does the time cost cause early crashes? Why did crawling appear?

**Early-crash test (last 400 training episodes per run, exact crash flag).** The wrapper's
design note warns that any per-step cost invites suicide once idling costs more than a crash.
In bootstrapped form: crash beats idling forever iff `penalty < time_cost/(1−γ)`. With
c = 0.01 at γ 0.999 that threshold is 10, so **penalty 40 is safe; penalty 5 would not be.**
The data agrees — taxed runs crash often but *late*, at speed:

| run | speed m/s | crash % | median m at crash | crashes < 10 m |
|---|---|---|---|---|
| S2 s99 / s98 (taxed) | 7.7 / 7.3 | 86 / 86 | 62 / 66 | 0% / 1% |
| S1 s99 / s98 (untaxed γ .999) | 2.3 / 1.1 | 100 / 64 | 1.6 / 1.5 | **100% / 100%** |
| P1 s99 (untaxed) | 1.5 | 33 | 8.8 | 84% |
| P2 s99 (taxed) | 4.2 | 62 | 9.6 | 61% |

The early crashes are in *untaxed* runs (S1's diverged critic, PPO collapse), not caused by the tax.

**Why crawling appeared here and not in our d1 runs.** SB3 bootstraps through the 3000-step
time limit, so the critic never sees a deadline: the only time pressure is γ. The design
note's "the step limit already supplies the time pressure" holds at γ 0.99 (1 s horizon) but
not at γ 0.999. Toy model (illustrative, not measured; FAST = 8 m/s crashing after 3 s,
CRAWL = slow, never crashes): at γ 0.99 FAST wins in every penalty/time-cost combination; at
γ 0.999 a crawl beats fast-but-crashing — narrowly with penalty 5 (17.0 vs 20.0), by a wide
margin with penalty 40 (−8.9 vs 20.0). **γ 0.999 + penalty 40 had never been tested before
this round** (our P40 runs were at γ 0.99; our γ 0.999 runs used penalty 5).

**P1 is not a one-variable test of our fix.** It changes three things at once versus our
validation: penalty 40 (not 5), 20 tracks (not 1), half-widths down to 0.63 m (ours ~1.47 m;
our d1 policy scored 0/50 at 0.75 m). Its optimiser side does reproduce — std 0.30–0.34 and
KL 0.005–0.030 at 1 M (ours: 0.39, 0.047) — but it does not lap, and by 2 M std is 0.06–0.07
and KL 0.19 / 2.5 (the late drift again). Clean test still needed: P1 with penalty 5 at d20.

**SAC.** S1's critic loss (median) goes 0.10 → 20.8 → 24.1 between 400 k and 1.2 M, exactly when
its score falls 0.28 → 0.02; S0 falls smoothly 0.31 → 0.03; S2 stays 0.06–0.09. The time cost
kept the γ-0.999 critic stable; why is open. Lap *completion* on vw_val at the best checkpoint:
S0 6% / 42%, S2 4% / 4%, P0 0 / 0, P1 0 / 0, P2 0 / 18%.

## See also

- [final-run-plan.md](final-run-plan.md) — SAC γ rule revised with this data
- [open-questions.md](open-questions.md) — Q10, Q11 status
- [grid-verification.md](grid-verification.md) — the eval noise floor that makes the 0.42 seed gap suspicious
