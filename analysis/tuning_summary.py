"""Summarise a tuning queue: per-run checkpoint scores, per-setting results, winners.

    python analysis/tuning_summary.py --state logs/queue_<name>/state.json \
        --results results/tuning [--baselines S0,P0]

Run by launch_queue.ps1 when the queue finishes, and by `launch_queue.ps1 -Summary`.

Score = mean fractional laps on vw_val (column `laps` of the evaluation CSVs,
5 episodes x 10 tracks, one-lap termination, so it lies in [0, 1]).

Per run: laps / lap completion % / mean progress at every evaluated checkpoint;
the best checkpoint (highest score; tie -> faster mean lap time over completed
laps -> earlier checkpoint); effective crash penalty and time cost from
run_config.json; mean training speed over the last 200 episodes (CRAWLING if
under 2.5 m/s); "declines late" if the final model scores more than 20% below
the best checkpoint.

Per setting (tag tune_<setting>_s<seed>): mean of its seeds' best scores and
the gap between seeds. The winner within each algorithm is chosen among the
candidates only; --baselines (default S0, P0) are shown but never win.
Settings within 0.03 of the winner are marked near-tie.

Writes <results>/summary.csv (one row per run) and summary_settings.csv.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
NEAR_TIE = 0.03
DECLINE = 0.20
CRAWL_MPS = 2.5


def ck_order(label: str) -> float:
    return np.inf if label == "final" else float(label.rstrip("k"))


def run_rows(state: dict, results: pathlib.Path) -> pd.DataFrame:
    rows = []
    for j in state["jobs"]:
        tag, run_dir = j["tag"], ROOT / j["run_dir"]
        m = re.fullmatch(r"[^_]+_(.+)_s(\d+)", tag)
        setting = m.group(1) if m else tag
        cfg_p = run_dir / "run_config.json"
        cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
        rew = cfg.get("effective_reward_constants") or cfg.get("reward_constants") or {}
        row = {"tag": tag, "setting": setting, "algo": j["algo"], "seed": j["seed"],
               "status": j["status"], "train_exit": j.get("exit"),
               "crash_penalty": rew.get("CRASH_PENALTY"), "time_cost": rew.get("TIME_COST")}
        mons = sorted(run_dir.glob("monitor_*.monitor.csv"))   # one per env when n_envs > 1
        if mons:
            # Episodes from all envs, ordered by wall-clock end time (t_start + t).
            parts = []
            for f in mons:
                head = json.loads(f.open().readline().lstrip("#"))
                d = pd.read_csv(f, skiprows=1)
                d["_end"] = head.get("t_start", 0.0) + d["t"]
                parts.append(d)
            mm = pd.concat(parts).sort_values("_end").tail(200)
            spd = float((mm["progress_m"] / (mm["l"] * 0.01)).mean()) if len(mm) else np.nan
            row["train_speed_last200_mps"] = round(spd, 2)
            row["crawling"] = bool(spd < CRAWL_MPS)
        scores = []
        for f in sorted(results.glob(f"{tag}_*_vw_val.csv")):
            label = f.name[len(tag) + 1:-len("_vw_val.csv")]
            d = pd.read_csv(f)
            lap = d["completed_lap"] == 1
            s = {"label": label, "laps": float(d["laps"].mean()),
                 "lap_pct": 100 * float(lap.mean()), "mean_m": float(d["progress_m"].mean()),
                 "lap_time_s": float((d.loc[lap, "length"] * 0.01).mean()) if lap.any() else np.inf,
                 "episodes": len(d)}
            scores.append(s)
            row[f"{label}_laps"] = round(s["laps"], 4)
            row[f"{label}_lap_pct"] = round(s["lap_pct"], 1)
            row[f"{label}_mean_m"] = round(s["mean_m"], 1)
        if scores:
            best = sorted(scores, key=lambda s: (-round(s["laps"], 6), s["lap_time_s"],
                                                 ck_order(s["label"])))[0]
            row.update({"best_checkpoint": best["label"], "best_laps": round(best["laps"], 4),
                        "best_lap_pct": round(best["lap_pct"], 1),
                        "best_lap_time_s": None if np.isinf(best["lap_time_s"]) else round(best["lap_time_s"], 2),
                        "checkpoints_evaluated": len(scores)})
            final = next((s for s in scores if s["label"] == "final"), None)
            if final is not None:
                row["final_laps"] = round(final["laps"], 4)
                row["declines_late"] = bool(best["laps"] > 0 and final["laps"] < (1 - DECLINE) * best["laps"])
        rows.append(row)
    return pd.DataFrame(rows)


def settings_table(runs: pd.DataFrame, baselines: set[str]) -> pd.DataFrame:
    out = []
    for (algo, setting), g in runs.groupby(["algo", "setting"], sort=False):
        b = g["best_laps"].dropna() if "best_laps" in g else pd.Series(dtype=float)
        out.append({"algo": algo, "setting": setting, "role": "baseline" if setting in baselines else "candidate",
                    "seeds": ",".join(str(s) for s in g["seed"]), "runs_scored": len(b),
                    "mean_best_laps": round(b.mean(), 4) if len(b) else np.nan,
                    "seed_gap": round(b.max() - b.min(), 4) if len(b) > 1 else np.nan,
                    "best_checkpoints": ",".join(str(x) for x in g.get("best_checkpoint", [])),
                    "declines_late": bool(g.get("declines_late", pd.Series(dtype=bool)).fillna(False).any()),
                    "crawling_runs": int(g.get("crawling", pd.Series(dtype=bool)).fillna(False).sum()),
                    "crash_penalty": ",".join(sorted({str(x) for x in g["crash_penalty"]})),
                    "time_cost": ",".join(sorted({str(x) for x in g["time_cost"]}))})
    st = pd.DataFrame(out)
    st["verdict"] = ""
    for algo, g in st.groupby("algo"):
        cand = g[(g.role == "candidate") & g.mean_best_laps.notna()]
        if cand.empty:
            continue
        win = cand.mean_best_laps.max()
        for i in cand.index:
            if st.at[i, "mean_best_laps"] == win:
                st.at[i, "verdict"] = "WINNER"
            elif st.at[i, "mean_best_laps"] >= win - NEAR_TIE:
                st.at[i, "verdict"] = "near-tie"
    return st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--state", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--baselines", default="S0,P0")
    a = ap.parse_args()
    state = json.loads(pathlib.Path(a.state).read_text(encoding="utf-8-sig"))
    results = pathlib.Path(a.results)
    results.mkdir(parents=True, exist_ok=True)
    runs = run_rows(state, results)
    if runs.empty:
        print("No runs in the queue state; nothing to summarise.")
        return 0
    st = settings_table(runs, set(a.baselines.split(",")))
    runs.to_csv(results / "summary.csv", index=False)
    st.to_csv(results / "summary_settings.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    cks = sorted({c[:-5] for c in runs.columns if c.endswith("_laps") and c != "best_laps"},
                 key=ck_order)
    print(f"TUNING SUMMARY (vw_val; score = mean fractional laps)  {len(runs)} runs")
    cols = ["tag", "status", "crash_penalty", "time_cost", "train_speed_last200_mps", "crawling"] + \
           [f"{c}_laps" for c in cks] + ["best_checkpoint", "best_laps", "best_lap_pct", "best_lap_time_s", "declines_late"]
    print(runs[[c for c in cols if c in runs.columns]].to_string(index=False))
    print("\nPER SETTING (baselines shown, never winners; near-tie = within "
          f"{NEAR_TIE} of the winner; declines late = final > {int(DECLINE * 100)}% below best)")
    print(st.to_string(index=False))
    print(f"\nwritten {results / 'summary.csv'} and {results / 'summary_settings.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
