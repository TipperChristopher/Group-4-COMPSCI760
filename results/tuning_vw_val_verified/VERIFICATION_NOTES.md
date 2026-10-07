# Verification notes (2026-10-07, arm's length)

Raw data: `_verify/teammate_tuning_vw/` (not in git; 22 MB). Canonical copy:
`desmond/heldout-tracks` @ `6e5b60e`. What was checked and the result:

| check | result |
|---|---|
| 16 run_config.json vs README table | match (P0 n_steps None = SB3 default 2048); all completed, clean, commit 6e5b60e |
| time cost in raw rewards | 0.0100/step in S2/P2, 0.0000 in the other 12; small negative tail on ~12% of episodes = lap-wrap progress-logging artifact (their "0.0100-0.0100" claim slightly overstated) |
| crash penalty | 40.00 in all 16 runs |
| 30 spot-checked eval scores vs summary.csv | max |delta| = 5e-5 |
| crawl speeds (last 200 eps) | match their table exactly (2.43 / 0.38 / 7.72 / 7.34 / 5.65 / 0.26 m/s) |
| eval reward protocol | canonical (no time cost): lap ep return = progress; crash = progress - 40 |
| spawn distinctness | 3 distinct poses per track (eval_seed 0=1, 2=3) - their caveat confirmed |
| S1 internals | critic loss max 3789 (their note 1 ok); ent_coef 0.884 -> 0.013, NO blowup (their "entropy coefficient blows up" not supported by the column) |
| P3 std | 1.0 -> 2.05 (their 1.5-1.9 roughly ok) |
| P0 s98 explained_variance | ends at -38.8 (their "about -3.5" was earlier in training) |

Self-error during verification: a first-pass crash-rate flag (r < progress - 20) mislabels
long taxed no-crash episodes as crashes; corrected flag reproduces their ~86% table.
