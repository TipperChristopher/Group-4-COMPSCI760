Candidates without time cost. SAC: default (S0). PPO: default (P0), par8_P0, par8_P1, par8_G998.
Candidates with time cost. SAC: S4 (time cost only), S2 (10 s + time cost), S5 (5 s + time cost, new). PPO: P4 (time cost only), P2 (10 s + 8192 + time cost), par8_G998TC (new).
Score: best vw_val checkpoint per run, averaged over seeds 99/98/97/96. Settings with only 2 seeds are compared on the shared seeds only.
Reliability bar: a setting can only replace its algorithm's default if it beats the default's average on at least 3 of 4 seeds. Otherwise that algorithm stays on its default.
Step 1, reward: use the time cost for both algorithms only if, for both SAC and PPO, the best time-cost candidate that passes the reliability bar is ≥ the best no-time-cost candidate − 0.03. Otherwise no time cost, for both.
Step 2, setting: each algorithm takes its best candidate under the chosen reward that passes the reliability bar. If none passes, it uses its default.
Fallback: if the time cost isn't adopted and no tuned setting passes the bar, both algorithms use defaults (S0 and P0).
Disclosure: PPO received more tuning (about 9 settings) than SAC (6).
