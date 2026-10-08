# Final-grid settings: the tuning-round-3 decision

**Verdict: no time cost. SAC = S0 (defaults), PPO = P0 (defaults).**

The rule was fixed before round 3 ran, in `tuning3_DECISION_RULE.md` (branch
`desmond/parallel-ppo`, commit 058bb93). It was applied on 2026-10-09 to every
candidate's vw_val results. Nothing was evaluated on any other track set.

## The rule (verbatim)

> Candidates without time cost. SAC: default (S0). PPO: default (P0), par8_P0, par8_P1, par8_G998.
> Candidates with time cost. SAC: S4 (time cost only), S2 (10 s + time cost), S5 (5 s + time cost, new). PPO: P4 (time cost only), P2 (10 s + 8192 + time cost), par8_G998TC (new).
> Score: best vw_val checkpoint per run, averaged over seeds 99/98/97/96. Settings with only 2 seeds are compared on the shared seeds only.
> Reliability bar: a setting can only replace its algorithm's default if it beats the default's average on at least 3 of 4 seeds. Otherwise that algorithm stays on its default.
> Step 1, reward: use the time cost for both algorithms only if, for both SAC and PPO, the best time-cost candidate that passes the reliability bar is ≥ the best no-time-cost candidate − 0.03. Otherwise no time cost, for both.
> Step 2, setting: each algorithm takes its best candidate under the chosen reward that passes the reliability bar. If none passes, it uses its default.
> Fallback: if the time cost isn't adopted and no tuned setting passes the bar, both algorithms use defaults (S0 and P0).
> Disclosure: PPO received more tuning (about 9 settings) than SAC (6).

## The candidate results

All runs used 20 varied-width training tracks (`vw_synthetic_track_0..19`), crash
penalty 40 and 2M steps. Each run was evaluated on vw_val (10 tracks × 5
episodes, deterministic, one lap) at 400k, 800k, 1.2M, 1.6M and the final model;
every evaluation has 50 rows.

**Score** = mean fractional laps of a run's best checkpoint. ✓ = beats the
default's 4-seed average.

### SAC: default S0 averages 0.5509

| setting | flags beyond the default | s99 | s98 | s97 | s96 | mean | seeds beating 0.5509 | passes bar |
|---|---|---|---|---|---|---|---|---|
| **S0** (default) | none | .4525 | .5664 | .5127 | .6718 | **0.5509** | — | default |
| S4 | `--time-cost 0.01` | .5434 | .5901 ✓ | .4435 | .3696 | 0.4867 | 1/4 | ✗ |
| S2 | `--gamma 0.999 --time-cost 0.01` | .4830 | .3959 | – | – | 0.4395 | 0/2 | ✗ |
| S5 | `--gamma 0.998 --time-cost 0.01` | .3049 | .1180 | .3443 | .3056 | 0.2682 | 0/4 | ✗ |

S2 was run on seeds 99 and 98 only. On those shared seeds S0 averages 0.5095,
and S2 is below it on both.

### PPO: default P0 averages 0.1238

| setting | flags beyond the default | s99 | s98 | s97 | s96 | mean | seeds beating 0.1238 | passes bar |
|---|---|---|---|---|---|---|---|---|
| **P0** (default) | none | .1746 | .1091 | .1000 | .1114 | **0.1238** | — | default |
| par8_P0 | `--n-envs 8` | .0950 | .1132 | .0724 | .1150 | 0.0989 | 0/4 | ✗ |
| P1 | `--gamma 0.999 --n-steps 8192` | .0284 | .1264 ✓ | – | – | 0.0774 | 1/2 | ✗ |
| par8_P1 | `--n-envs 8 --gamma 0.999 --n-steps 1024` | .0414 | .1072 | .5228 ✓ | .0874 | 0.1897 | 1/4 | ✗ |
| par8_G998 | `--n-envs 8 --gamma 0.998 --n-steps 1024` | .3964 ✓ | .1118 | .0809 | .2500 ✓ | 0.2098 | 2/4 | ✗ |
| P4 | `--time-cost 0.01` | .0728 | .1083 | .2245 ✓ | .0671 | 0.1182 | 1/4 | ✗ |
| **P2** | `--gamma 0.999 --n-steps 8192 --time-cost 0.01` | .0706 | .4907 ✓ | .2869 ✓ | .2595 ✓ | **0.2769** | **3/4** | **✓** |
| par8_G998TC | `--n-envs 8 --gamma 0.998 --n-steps 1024 --time-cost 0.01` | .2819 ✓ | .0162 | .3697 ✓ | .1222 | 0.1975 | 2/4 | ✗ |

P1 was run on seeds 99 and 98 only. On those shared seeds P0 averages 0.1419,
and P1 beats it on neither. par8_G998TC's seed 96 (0.1222) misses P0's average
by 0.0016.

## Working

1. **Step 1, reward.** Time cost is adopted only if, for both algorithms, the best
   time-cost candidate that passes the bar is ≥ the best no-time-cost candidate − 0.03.
   - PPO: P2 passes (3/4), and 0.2769 ≥ 0.1238 − 0.03 = 0.0938. This side holds.
   - SAC: no time-cost candidate passes the bar (S4 1/4, S2 0/2, S5 0/4). This
     side fails. It would fail without the bar too: the best time-cost mean,
     S4 at 0.4867, is below S0's 0.5509 − 0.03 = 0.5209.
   - **Result: no time cost, for both algorithms.**
2. **Step 2, settings under no time cost.**
   - SAC: the only no-time-cost candidate is the default, so **S0**.
   - PPO: none of par8_P0 (0/4), P1 (1/2), par8_P1 (1/4) or par8_G998 (2/4)
     passes the bar, so the default, **P0**.
3. **Fallback:** gives the same answer, S0 and P0.

## The cost of the rule

P2 is the only tuned setting that reliably beat its default: PPO 0.277 against
0.124, better on 3 of 4 seeds. The pre-registered rule requires both algorithms
to use the same reward, so that the comparison between them stays fair. SAC
rejects the time cost, so P2 is set aside. The final grid therefore compares the
two algorithms at their defaults under the same reward, rather than at the best
setting each could reach. The write-up should say so, together with the
disclosure that PPO received more tuning (about 9 settings) than SAC (6).

## Final settings and job files

Crash penalty 40 and time cost 0 are the values in `sb3_wrapper.py`, so neither
needs a flag:

```
SAC:  --algo SAC --diversity <1|5|20|100> --seed <0|1|2> --track-prefix vw_synthetic_track_
PPO:  --algo PPO --diversity <1|5|20|100> --seed <0|1|2> --track-prefix vw_synthetic_track_
```

These are the winning tuning lines `tune_S0_s99` and `tune_P0_s99` in
`tuning_jobs.txt`, with only diversity and seed changed. All 24 lines in
`final_jobs_{desmond,grant,chris}.txt` were diffed against them:

- each of the 24 cells (SAC/PPO × diversity 1/5/20/100 × seed 0/1/2) appears
  exactly once
- every tag is `final_<algo>_d<N>_s<seed>`
- 0 lines differ in anything except `--diversity`, `--seed` and the tag

`check_setup.py` repeats this check on every machine.

| person | job file | runs | `-MaxThreads` | results folder |
|---|---|---|---|---|
| Desmond | `final_jobs_desmond.txt` | SAC seeds 0, 1 (8) | 16 | `results\final_grid_desmond` |
| Grant | `final_jobs_grant.txt` | SAC seed 2 (4) | 8 | `results\final_grid_grant` |
| Chris | `final_jobs_chris.txt` | PPO seeds 0, 1, 2 (12) | 10 | `results\final_grid_chris` |

Sources of the scores:
- S0, S2, S4, P0, P1, P2 and P4: `results/tuning/` in the main checkout
- S5 and par8_G998TC: `results/tuning3/`, branch `desmond/parallel-ppo`
- par8_P0, par8_P1 and par8_G998: `results/followup_parallel/`, branch `desmond/parallel-ppo`
