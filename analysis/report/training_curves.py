"""Report Fig. 4: training curves of the 100-track runs, grid A (left) vs grid B (right).

    python analysis/report/training_curves.py [--out analysis/report/output/fig4_training_curves.png]

Reads results/run_logs/grid_{A,B}/<run>/monitor_0.monitor.csv (one row per training
episode; the first line is a JSON header). For each run, episodes are placed at the
cumulative environment step where they ended, averaged in 50k-step bins (metres travelled
along the centreline per episode, column progress_m), then smoothed with a trailing
rolling mean over 4 bins (200k steps). The line is the mean over the 3 seeds; the shaded
band is the min-max across seeds. SAC blue, PPO orange.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
LOGS = REPO / "results" / "run_logs"
BIN = 50_000
ROLL = 4
TOTAL = 2_000_000
COLOURS = {"SAC": "#2a78d6", "PPO": "#eb6834"}
GRIDS = {"A": ("grid_A", "final"), "B": ("grid_B", "v2")}


def run_curve(monitor: pathlib.Path) -> pd.Series:
    m = pd.read_csv(monitor, skiprows=1)
    end_step = m["l"].cumsum()
    binned = m.groupby(end_step // BIN)["progress_m"].mean()
    binned = binned.reindex(range(TOTAL // BIN)).interpolate(limit_direction="both")
    return binned.rolling(ROLL, min_periods=1).mean()


def curves(grid: str, algo: str, diversity: int = 100) -> pd.DataFrame:
    folder, prefix = GRIDS[grid]
    runs = sorted((LOGS / folder).glob(f"{algo}_{diversity}tracks_s*_{prefix}_{algo}_d{diversity}_s*"))
    if len(runs) != 3:
        raise SystemExit(f"grid {grid} {algo} d{diversity}: expected 3 seeds, found {len(runs)}")
    return pd.DataFrame({r.name: run_curve(r / "monitor_0.monitor.csv") for r in runs})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO / "analysis" / "report" / "output" / "fig4_training_curves.png")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    x = (np.arange(TOTAL // BIN) + 1) * BIN / 1e6        # bin end, millions of steps
    for ax, grid in zip(axes, ("A", "B")):
        for algo in ("SAC", "PPO"):
            c = curves(grid, algo)
            mean, lo, hi = c.mean(axis=1), c.min(axis=1), c.max(axis=1)
            ax.fill_between(x, lo, hi, color=COLOURS[algo], alpha=0.2, lw=0)
            ax.plot(x, mean, color=COLOURS[algo], lw=2, label=f"{algo} (mean of 3 seeds)")
            print(f"grid {grid} {algo} d100: final value {mean.iloc[-1]:.1f} m "
                  f"(seed range {lo.iloc[-1]:.1f}-{hi.iloc[-1]:.1f} m)")
        ax.set_title(f"Grid {grid}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Training steps (millions)")
        ax.set_xlim(0, TOTAL / 1e6)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Metres per training episode")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
