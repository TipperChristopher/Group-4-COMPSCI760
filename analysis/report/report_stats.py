"""Report statistics (Tables III-V, Wilcoxon, Holm, Spearman) from the committed final-test files.

    python analysis/report/report_stats.py

Reads only results/final_test/test_raw_*.csv, results/final_test/selection.csv (grid A)
and results/final_test_v2/selection.csv (grid B). Prints:
  * Table III: per grid x algorithm x diversity x test set, mean and SD over the 3 seeds of
    each run's mean fractional laps (all tracks x 10 episodes); lap % at d100; baselines.
  * Per-seed gains d100 - d1 (same seed).
  * 16 two-sided Wilcoxon signed-rank tests (scipy defaults) on per-track means, each the
    average over 3 seeds x 10 episodes of fractional laps:
      RQ2  d100 vs d1, per grid x set x algorithm (8 tests)
      RQ3  SAC gain vs PPO gain (gain = d100 - d1 per track), per grid x set (4 tests)
      RQ1  SAC vs PPO at d100, per grid x set (4 tests)
    with Holm step-down adjustment over the 4 RQ3 tests and over all 16.
  * Spearman correlation between each run's vw_val_laps (selection files) and its test
    score, per test set (48 runs each).
  * Table V for the d100 agents (episodes pooled over the 3 seeds): median metres travelled
    (progress_m), mean speed = progress_m / (length / 100) in m/s, crash %, timeout %.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

REPO = pathlib.Path(__file__).resolve().parents[2]
TEST = REPO / "results" / "final_test"
SELECTION = {"A": TEST / "selection.csv", "B": REPO / "results" / "final_test_v2" / "selection.csv"}
SETS = {"vw_test": "synthetic (vw_test)", "real": "real circuits"}
ALGOS = ("SAC", "PPO")
DIVS = (1, 5, 20, 100)
SIM_HZ = 100


def load():
    runs, eps = [], []
    for grid, path in SELECTION.items():
        sel = pd.read_csv(path)
        for _, r in sel.iterrows():
            for ts in SETS:
                d = pd.read_csv(TEST / f"test_raw_{r.tag}_{ts}.csv")
                d = d.assign(grid=grid, algorithm=r.algorithm, diversity=int(r.diversity),
                             seed=int(r.seed), tag=r.tag, test_set=ts)
                eps.append(d)
                runs.append(dict(grid=grid, algorithm=r.algorithm, diversity=int(r.diversity),
                                 seed=int(r.seed), tag=r.tag, test_set=ts, vw_val_laps=r.vw_val_laps,
                                 laps=d.laps.mean(), lap_pct=100 * d.completed_lap.mean()))
    return pd.DataFrame(runs), pd.concat(eps, ignore_index=True)


def fmt(p: float) -> str:
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def holm(p: dict) -> dict:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    keys = sorted(p, key=p.get)
    m, out, running = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * p[k]))
        out[k] = running
    return out


def main() -> int:
    runs, eps = load()
    pd.set_option("display.width", 200)

    print("=" * 78, "\nTABLE III: fractional laps, mean ± SD over 3 seeds (lap % at d100)\n" + "=" * 78)
    for grid in SELECTION:
        for ts, label in SETS.items():
            print(f"Grid {grid}, {label}")
            for algo in ALGOS:
                cells = []
                for d in DIVS:
                    g = runs[(runs.grid == grid) & (runs.algorithm == algo) & (runs.diversity == d) & (runs.test_set == ts)]
                    cells.append(f"d{d} {g.laps.mean():.3f} ± {g.laps.std(ddof=1):.3f}")
                    if d == 100:
                        cells[-1] += f" (lap {g.lap_pct.mean():.1f}% ± {g.lap_pct.std(ddof=1):.1f})"
                print(f"   {algo}: " + " | ".join(cells))
    print("Baselines (single run each)")
    for b in ("gap", "random"):
        for ts, label in SETS.items():
            d = pd.read_csv(TEST / f"test_raw_baseline_{b}_{ts}.csv")
            print(f"   {b:<6} {label:<20} laps {d.laps.mean():.3f}, lap {100 * d.completed_lap.mean():.1f}%")

    print("\n" + "=" * 78, "\nPER-SEED GAINS d100 - d1 (fractional laps, same seed)\n" + "=" * 78)
    for grid in SELECTION:
        for ts, label in SETS.items():
            for algo in ALGOS:
                g = runs[(runs.grid == grid) & (runs.algorithm == algo) & (runs.test_set == ts)].set_index(["diversity", "seed"]).laps
                gains = [g[(100, s)] - g[(1, s)] for s in (0, 1, 2)]
                print(f"   grid {grid} {label:<20} {algo}: " + ", ".join(f"s{s} {v:+.3f}" for s, v in enumerate(gains))
                      + f"   mean {np.mean(gains):+.3f}")

    # Per-track means: average over 3 seeds x 10 episodes.
    track = eps.groupby(["grid", "test_set", "algorithm", "diversity", "track"]).laps.mean()

    def vec(grid, ts, algo, d):
        return track.loc[(grid, ts, algo, d)].sort_index()

    tests = {}
    for grid in SELECTION:
        for ts in SETS:
            for algo in ALGOS:
                a, b = vec(grid, ts, algo, 100), vec(grid, ts, algo, 1)
                tests[("RQ2", grid, ts, f"{algo} d100 vs d1")] = (wilcoxon(a, b).pvalue, (a - b).median(), len(a))
            gs = vec(grid, ts, "SAC", 100) - vec(grid, ts, "SAC", 1)
            gp = vec(grid, ts, "PPO", 100) - vec(grid, ts, "PPO", 1)
            tests[("RQ3", grid, ts, "SAC gain vs PPO gain")] = (wilcoxon(gs, gp).pvalue, (gs - gp).median(), len(gs))
            s, p = vec(grid, ts, "SAC", 100), vec(grid, ts, "PPO", 100)
            tests[("RQ1", grid, ts, "SAC vs PPO at d100")] = (wilcoxon(s, p).pvalue, (s - p).median(), len(s))
    p_all = {k: v[0] for k, v in tests.items()}
    h16 = holm(p_all)
    h4 = holm({k: v for k, v in p_all.items() if k[0] == "RQ3"})

    print("\n" + "=" * 78, "\nWILCOXON SIGNED-RANK (two-sided, scipy defaults) on per-track means\n" + "=" * 78)
    print(f"   {'RQ':<4}{'grid':<5}{'set':<9}{'comparison':<22}{'n':>3} {'median diff':>12} {'p':>9} {'Holm-4':>8} {'Holm-16':>8}")
    for k in sorted(tests, key=lambda k: (k[0], k[1], k[2], k[3])):
        p, med, n = tests[k]
        print(f"   {k[0]:<4}{k[1]:<5}{k[2]:<9}{k[3]:<22}{n:>3} {med:>+12.4f} {fmt(p):>9} "
              f"{(fmt(h4[k]) if k in h4 else ''):>8} {fmt(h16[k]):>8}")

    print("\n" + "=" * 78, "\nSPEARMAN: selected checkpoint's vw_val_laps vs its test score (48 runs per set)\n" + "=" * 78)
    for ts, label in SETS.items():
        g = runs[runs.test_set == ts]
        r = spearmanr(g.vw_val_laps, g.laps)
        print(f"   {label:<20} rho = {r.statistic:.3f} (p = {r.pvalue:.2e}, n = {len(g)})")

    print("\n" + "=" * 78, "\nTABLE V: d100 agents, episodes pooled over 3 seeds\n" + "=" * 78)
    d100 = eps[eps.diversity == 100].copy()
    d100["speed"] = d100.progress_m / (d100.length / SIM_HZ)
    for grid in SELECTION:
        for algo in ALGOS:
            for ts, label in SETS.items():
                g = d100[(d100.grid == grid) & (d100.algorithm == algo) & (d100.test_set == ts)]
                print(f"   grid {grid} {algo} {label:<20} median {g.progress_m.median():6.1f} m | mean speed "
                      f"{g.speed.mean():5.2f} m/s | crash {100 * (g.outcome == 'crash').mean():5.1f}% | "
                      f"timeout {100 * (g.outcome == 'timeout').mean():5.1f}% | n {len(g)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
