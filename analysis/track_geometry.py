"""Centreline geometry of every track, and where the SAC agents crash on it.

    python analysis/track_geometry.py

PART 1, GEOMETRY. For every track in the train, test and real splits of
tracks/manifest.json, along the centreline as the simulator itself samples it
(Track.from_track_name resamples every centreline at 0.1 m of spline arc
length, so all three sets share one sampling):

  half-width   distance from the centreline to the nearest wall, from the map
               image and its yaml resolution: a Euclidean distance transform
               of the loader's own thresholded occupancy grid, sampled
               bilinearly, minus half a pixel so it measures to the wall's
               edge rather than its pixel centre. Identical for every set; the
               CSV width columns are not used (the synthetic ones are a
               constant 2.0 placeholder).
  curvature    heading change over a fixed 1 m of arc, in 1/m. Not the raw
               spline curvature: that spikes at the CSV knots, and the knots
               are 0.57 m apart on synthetic tracks but 0.30-0.42 m on real
               ones, so raw curvature would measure the sampling as much as
               the track. A fixed arc window measures every set the same way.
               The 2 m window is reported as a sensitivity check.

Written to results/track_geometry/: points_<split>.csv.gz (every 0.1 m),
track_summary.csv, distributions.png.

PART 2, SAC CRASHES. For every SAC episode in results/grid_s0/SAC_d*_{test,
real}.csv that ended in a crash:

  spawn        cl_grid_static does NOT always start at centreline index 0. It
               draws a waypoint uniformly from the first 1 m of the resampled
               centreline (indices 0-9), using numpy's global RNG, which
               f110_env.reset(seed) seeds with the episode's eval_seed. Each
               episode's spawn is reproduced exactly by resetting the same
               evaluation env with that seed (no driving), and the origin is
               read from the wrapper's own progress tracker, the one that
               produced progress_m.
  crash point  (spawn s + progress_m) mod track length.

At the crash point, and as the worst value over the 5 m leading up to it,
half-width and curvature are compared with:

  random points   uniform along the same circuit (track-matched null)
  driven points   the points that same episode passed safely, excluding the
                  5 m before the crash (survival-matched null)
  the training set  "outside training" means narrower than the narrowest
                  point, or sharper than the sharpest point, anywhere on
                  synthetic_track_0..99 (and, per cell, on its own pool).

Every point of every real circuit turns out to be narrower than every point of
every training track (the generator's width is a constant), so "crash outside
the training range" is true of every real-circuit crash by construction. The
informative tests are therefore (a) the crash point's rank within its own
circuit, and (b) at circuit level, whether each cell's mean progress over the
23 circuits rises with circuit width or falls with curvature (Spearman).

Deterministic episodes from the same spawn are identical, so repeated
(cell, circuit, spawn) episodes are counted once.

Written to results/track_geometry/: sac_crash_points.csv, crash_vs_random.png,
summary.json.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import pathlib
import sys
import types
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
if "gym" not in sys.modules:
    sys.modules["gym"] = types.ModuleType("gym")
sys.modules["gym"].__version__ = "0.0.0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, map_coordinates

OUT = REPO / "results" / "track_geometry"
GRID = REPO / "results" / "grid_s0"
SPLITS = ("train", "test", "real")
# Same shapes as test, narrower walls. Measured and checked against test when
# present in the manifest; kept out of the train/test/real plots.
NARROW_SPLITS = ("narrowA", "narrowB")
DS = 0.1                 # simulator's centreline resampling step, metres
CURV_WINDOW_M = 1.0      # primary curvature arc window
CURV_WINDOW_ALT_M = 2.0  # sensitivity check
LOOKBACK_M = 5.0         # "leading up to the crash" window
N_RANDOM = 200           # random points per crash for the plotted null
N_SIM = 20000            # simulations for the out-of-training p-values
RNG = np.random.default_rng(0)

# Validated categorical slots 1-3 (dataviz reference palette, all-pairs pass).
COLOURS = {"train": "#2a78d6", "test": "#eb6834", "real": "#1baf7a"}
CRASH, RANDOM_C, INK, MUTED = "#2a78d6", "#8a8985", "#0b0b0b", "#52514e"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------------ geometry
def windowed_curvature(yaw: np.ndarray, window_m: float) -> np.ndarray:
    """Signed heading change across a centred arc window, per metre. Periodic."""
    h = int(round(window_m / 2 / DS))
    return wrap(np.roll(yaw, -h) - np.roll(yaw, h)) / (2 * h * DS)


def track_geometry(name: str, track=None):
    """Per-point geometry. Pass `track` to measure one not installed in maps/."""
    if track is None:
        from f1tenth_gym.envs.track import Track
        track = Track.from_track_name(name)
    cl = track.centerline
    x, y = cl.xs.astype(float), cl.ys.astype(float)
    yaw = cl.yaws.astype(float)
    n = len(x)
    s = np.arange(n) * DS

    res = float(track.spec.resolution)
    ox, oy = float(track.spec.origin[0]), float(track.spec.origin[1])
    free = track.occupancy_map > 128          # loader's own threshold, row 0 = origin y
    edt = distance_transform_edt(free) * res
    rows = (y - oy) / res - 0.5               # pixel centres sit at +0.5 px
    cols = (x - ox) / res - 0.5
    half_width = map_coordinates(edt, [rows, cols], order=1, mode="nearest") - 0.5 * res

    df = pd.DataFrame({
        "track": name, "idx": np.arange(n), "s_m": np.round(s, 1),
        "x": np.round(x, 3), "y": np.round(y, 3),
        "half_width_m": np.round(half_width, 4),
        "curvature_1m": np.round(windowed_curvature(yaw, CURV_WINDOW_M), 5),
        "curvature_2m": np.round(windowed_curvature(yaw, CURV_WINDOW_ALT_M), 5),
        "curvature_spline": np.round(cl.ks.astype(float), 5),
    })
    meta = {"length_m": float(cl.spline.s[-1]), "resolution_m": res}
    return df, meta


def summarise(split: str, name: str, df: pd.DataFrame, meta: dict) -> dict:
    w = df["half_width_m"].to_numpy()
    k = np.abs(df["curvature_1m"].to_numpy())
    k2 = np.abs(df["curvature_2m"].to_numpy())
    return {
        "split": split, "track": name, "length_m": round(meta["length_m"], 1),
        "n_points": len(df), "map_resolution_m": meta["resolution_m"],
        "width_min_m": round(w.min(), 3), "width_p10_m": round(np.percentile(w, 10), 3),
        "width_median_m": round(np.median(w), 3),
        "curv_max_per_m": round(k.max(), 4), "curv_p90_per_m": round(np.percentile(k, 90), 4),
        "curv_median_per_m": round(np.median(k), 4),
        "min_radius_m": round(1.0 / k.max(), 3),
        "curv2m_max_per_m": round(k2.max(), 4),
        "curv_spline_max_per_m": round(np.abs(df["curvature_spline"]).max(), 4),
    }


def check_narrow(manifest: dict, geo: dict, meta: dict, geo_splits) -> dict | None:
    """Each narrow track against its test twin: same centreline, narrower walls.

    Compares the simulator's resampled centreline (x, y every 0.1 m), the map
    frame (yaml origin and resolution) and curvature exactly, and reports the
    half-width of both.
    """
    import re
    maps = _load("make_heldout_tracks", "tracks/make_heldout_tracks.py").maps_dir()

    def frame(name):
        text = (maps / name / f"{name}_map.yaml").read_text()
        return (re.search(r"origin:\s*\[([^\]]+)\]", text).group(1),
                re.search(r"resolution:\s*(\S+)", text).group(1))

    rows, out = [], {}
    tests = manifest["splits"]["test"]["names"]
    for split in (s for s in NARROW_SPLITS if s in geo_splits):
        for test, name in zip(tests, manifest["splits"][split]["names"]):
            a, b = geo[test], geo[name]
            same_n = len(a) == len(b)
            d_xy = float(np.max(np.hypot(a["x"] - b["x"], a["y"] - b["y"]))) if same_n else np.inf
            d_k = float(np.max(np.abs(a["curvature_1m"] - b["curvature_1m"]))) if same_n else np.inf
            rows.append({
                "split": split, "track": name, "test_twin": test,
                "same_n_points": same_n, "max_centreline_diff_m": d_xy,
                "max_curvature_diff": d_k,
                "same_map_frame": frame(test) == frame(name),
                "same_length": meta[test]["length_m"] == meta[name]["length_m"],
                "test_half_width_median_m": round(float(a["half_width_m"].median()), 4),
                "half_width_median_m": round(float(b["half_width_m"].median()), 4),
                "half_width_min_m": round(float(b["half_width_m"].min()), 4),
                "half_width_max_m": round(float(b["half_width_m"].max()), 4),
            })
        r = [x for x in rows if x["split"] == split]
        out[split] = {
            "n_tracks": len(r),
            "centreline_identical": all(x["same_n_points"] and x["max_centreline_diff_m"] == 0.0
                                        and x["same_length"] for x in r),
            "map_frame_identical": all(x["same_map_frame"] for x in r),
            "curvature_identical": all(x["max_curvature_diff"] == 0.0 for x in r),
            "half_width_median_m": round(float(np.median([x["half_width_median_m"] for x in r])), 4),
            "half_width_min_m": round(min(x["half_width_min_m"] for x in r), 4),
            "half_width_max_m": round(max(x["half_width_max_m"] for x in r), 4),
            "test_half_width_median_m": round(float(np.median([x["test_half_width_median_m"] for x in r])), 4),
        }
    out["per_track"] = rows
    return out if rows else None


# ------------------------------------------------------------------- spawns
def find_progress(env):
    """The F1TenthSB3Wrapper inside the evaluation stack."""
    e = env
    while e is not None:
        if hasattr(e, "progress") and hasattr(e.progress, "_prev_s"):
            return e.progress
        e = getattr(e, "env", None)
    raise RuntimeError("no progress tracker in env stack")


def reproduce_spawns(ev, track: str, seeds, geo: pd.DataFrame) -> dict:
    """Spawn (centreline index, s) per eval seed, exactly as evaluation reset."""
    env = ev.build_env(track, False, 0, 15000, 1, "cl_grid_static")()
    out = {}
    xy = geo[["x", "y"]].to_numpy()
    for seed in sorted(set(int(s) for s in seeds)):
        env.reset(seed=seed)
        core = env.unwrapped
        px, py = float(core.poses_x[0]), float(core.poses_y[0])
        prog = find_progress(env)
        idx = int(np.argmin(np.hypot(xy[:, 0] - px, xy[:, 1] - py)))
        out[seed] = {"spawn_idx": idx, "spawn_s_m": float(prog._prev_s),
                     "spawn_offset_m": float(np.hypot(xy[idx, 0] - px, xy[idx, 1] - py)),
                     "track_len_m": float(prog.length)}
    env.close()
    return out


# ------------------------------------------------------------------ lookups
def lookback_idx(i: int, n: int) -> np.ndarray:
    h = int(round(LOOKBACK_M / DS))
    return np.arange(i - h, i + 1) % n


def worst_in_window(w: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Min half-width and max |curvature| over the lookback window ending at each point."""
    h = int(round(LOOKBACK_M / DS))
    wp = np.concatenate([w[-h:], w])
    kp = np.concatenate([k[-h:], k])
    from numpy.lib.stride_tricks import sliding_window_view
    return (sliding_window_view(wp, h + 1).min(axis=1),
            sliding_window_view(kp, h + 1).max(axis=1))


def rank_low(values: np.ndarray, x: float) -> float:
    """Fraction of values <= x: small means x is unusually low (narrow)."""
    return float(np.mean(values <= x))


def rank_high(values: np.ndarray, x: float) -> float:
    """Fraction of values >= x: small means x is unusually high (sharp)."""
    return float(np.mean(values >= x))


# -------------------------------------------------------------------- plots
def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_distributions(points: dict, ranges: dict, path: pathlib.Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="#fcfcfb")
    specs = [("half_width_m", "Half-width (centreline to nearest wall), m", np.linspace(0.5, 1.7, 61), False),
             ("abs_curv", "|Curvature|, 1/m (heading change over 1 m)", np.linspace(0, 1.65, 67), True)]
    for ax, (col, label, bins, logy) in zip(axes, specs):
        ax.set_facecolor("#fcfcfb")
        style(ax)
        # train drawn last: test follows it almost exactly and would hide it
        for split in ("real", "test", "train"):
            v = points[split][col].to_numpy()
            h, e = np.histogram(v, bins=bins, density=True)
            ax.stairs(h, e, color=COLOURS[split], linewidth=2,
                      label=f"{split} ({points[split]['track'].nunique()} tracks)")
        edge = ranges["width_min"] if col == "half_width_m" else ranges["curv_max"]
        ax.axvline(edge, color=INK, linestyle="--", linewidth=1)
        word = "narrowest" if col == "half_width_m" else "sharpest"
        ax.annotate(f"{word} point on any\ntraining track: {edge:.2f}", (edge, 0.62),
                    xycoords=("data", "axes fraction"),
                    xytext=(-6 if col == "half_width_m" else 6, 0), textcoords="offset points",
                    ha="right" if col == "half_width_m" else "left", color=INK, fontsize=8.5)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(label, color=INK, fontsize=10)
        ax.set_ylabel("density (share of centreline)" + (", log scale" if logy else ""),
                      color=MUTED, fontsize=9)
        ax.legend(frameon=False, fontsize=9, loc="upper left" if col == "half_width_m" else "upper right")
    fig.suptitle("Centreline geometry: training (synthetic_track_0..99) vs held-out test vs real circuits. "
                 "Every real point is narrower than all of training; train and test overlap.",
                 color=INK, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def ecdf(ax, v, colour, label, ls="-"):
    v = np.sort(np.asarray(v))
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=colour,
            linewidth=2, linestyle=ls, label=label)


def plot_crashes(crash: pd.DataFrame, rand: pd.DataFrame, ranges: dict, path: pathlib.Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="#fcfcfb")
    rows = [("half_width_m", "Half-width at crash point, m", ranges["width_min"], "narrowest training point"),
            ("abs_curv", "|Curvature| at crash point, 1/m", ranges["curv_max"], "sharpest training point")]
    for c, split in enumerate(("real", "test")):
        cs, rs = crash[crash["set"] == split], rand[rand["set"] == split]
        for r, (col, label, edge, edge_label) in enumerate(rows):
            ax = axes[r, c]
            ax.set_facecolor("#fcfcfb")
            style(ax)
            ecdf(ax, rs[col], RANDOM_C, f"random points, same circuits")
            ecdf(ax, cs[col], CRASH, f"SAC crash points (n={len(cs)})")
            ax.axvline(edge, color=INK, linestyle="--", linewidth=1)
            lo, hi = ax.get_xlim()
            left_half = edge < 0.5 * (lo + hi)
            ax.annotate(edge_label, (edge, 0.55), xycoords=("data", "axes fraction"),
                        xytext=(5 if left_half else -5, 0), textcoords="offset points",
                        ha="left" if left_half else "right", color=INK, fontsize=8.5)
            ax.set_xlabel(label, color=INK, fontsize=10)
            ax.set_ylabel("cumulative share", color=MUTED, fontsize=9)
            ax.set_title(f"{split} circuits", color=INK, fontsize=10, loc="left")
            ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Where SAC crashes vs random points on the same circuits (all four diversity cells, seed 0)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


# --------------------------------------------------------------------- main
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((REPO / "tracks" / "manifest.json").read_text())

    # ---- Part 1: geometry
    geo_splits = SPLITS + tuple(s for s in NARROW_SPLITS if s in manifest["splits"])
    geo, meta, summaries = {}, {}, []
    for split in geo_splits:
        for name in manifest["splits"][split]["names"]:
            geo[name], meta[name] = track_geometry(name)
            summaries.append(summarise(split, name, geo[name], meta[name]))
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT / "track_summary.csv", index=False)

    narrow_check = check_narrow(manifest, geo, meta, geo_splits)
    if narrow_check:
        pd.DataFrame(narrow_check["per_track"]).to_csv(OUT / "narrow_vs_test.csv", index=False)

    points = {}
    for split in geo_splits:
        names = manifest["splits"][split]["names"]
        df = pd.concat([geo[n] for n in names], ignore_index=True)
        df.to_csv(OUT / f"points_{split}.csv.gz", index=False, compression="gzip")
        df = df.copy()
        df["abs_curv"] = df["curvature_1m"].abs()
        df["abs_curv2"] = df["curvature_2m"].abs()
        points[split] = df

    train = points["train"]
    ranges = {"width_min": float(train["half_width_m"].min()),
              "curv_max": float(train["abs_curv"].max()),
              "curv2_max": float(train["abs_curv2"].max())}
    pool_ranges = {}
    for d in (1, 5, 20, 100):
        pool = train[train["track"].isin([f"synthetic_track_{i}" for i in range(d)])]
        pool_ranges[d] = {"width_min": float(pool["half_width_m"].min()),
                          "curv_max": float(pool["abs_curv"].max())}
    plot_distributions(points, ranges, OUT / "distributions.png")

    # Per-track arrays and the out-of-training share of every circuit.
    arr = {}
    for name, df in geo.items():
        w = df["half_width_m"].to_numpy()
        k = np.abs(df["curvature_1m"].to_numpy())
        k2 = np.abs(df["curvature_2m"].to_numpy())
        ww, kw = worst_in_window(w, k)
        arr[name] = {"w": w, "k": k, "k2": k2, "ww": ww, "kw": kw}

    def ood(w, k, rng_=ranges, k_key="curv_max"):
        return (w < rng_["width_min"]) | (k > rng_[k_key])

    # ---- Part 2: SAC crashes
    ev = _load("evaluate_mod", "evaluate.py")
    rows = []
    for path in sorted(glob.glob(str(GRID / "SAC_d*_*.csv"))):
        stem = pathlib.Path(path).stem               # SAC_d100_real
        _, dpart, split = stem.split("_")
        if split not in ("test", "real"):
            continue
        df = pd.read_csv(path)
        df["set"], df["cell"], df["d"] = split, stem.rsplit("_", 1)[0], int(dpart[1:])
        rows.append(df)
    ep = pd.concat(rows, ignore_index=True)

    spawns = {}
    for track, g in ep.groupby("track"):
        spawns[track] = reproduce_spawns(ev, track, g["eval_seed"], geo[track])
    for col in ("spawn_idx", "spawn_s_m", "spawn_offset_m", "track_len_m"):
        ep[col] = [spawns[t][s][col] for t, s in zip(ep["track"], ep["eval_seed"])]

    # Consistency: a deterministic policy from the same spawn must replay the
    # same episode, and the spawn must lie in the first 1 m of the centreline.
    sig = ep.groupby(["cell", "track", "spawn_idx"]).agg(
        n_sig=("progress_m", lambda v: len(set(np.round(v, 3)))))
    spawn_check = {
        "max_spawn_idx": int(ep["spawn_idx"].max()),
        "spawn_idx_counts": {int(k): int(v) for k, v in ep["spawn_idx"].value_counts().sort_index().items()},
        "max_spawn_offset_from_centreline_point_m": round(float(ep["spawn_offset_m"].max()), 4),
        "same_spawn_different_outcome_groups": int((sig["n_sig"] > 1).sum()),
        "groups": int(len(sig)),
    }

    crashed = ep[ep["outcome"] == "crash"].copy()
    unique = crashed.drop_duplicates(["cell", "track", "spawn_idx"]).copy()

    recs, rand_recs = [], []
    for _, e in unique.iterrows():
        a = arr[e["track"]]
        n, L = len(a["w"]), e["track_len_m"]
        s_crash = (e["spawn_s_m"] + e["progress_m"]) % L
        i = int(round(s_crash / DS)) % n
        pr = pool_ranges[int(e["d"])]

        # points the car passed safely: spawn to crash minus the lookback window
        n_drv = int((e["progress_m"] - LOOKBACK_M) / DS)
        drv = (int(e["spawn_idx"]) + np.arange(max(n_drv, 0))) % n

        rec = {
            "cell": e["cell"], "set": e["set"], "diversity": int(e["d"]), "track": e["track"],
            "episode": int(e["episode"]), "eval_seed": int(e["eval_seed"]),
            "n_identical_episodes": int(((crashed["cell"] == e["cell"]) & (crashed["track"] == e["track"])
                                         & (crashed["spawn_idx"] == e["spawn_idx"])).sum()),
            "spawn_idx": int(e["spawn_idx"]), "spawn_s_m": round(e["spawn_s_m"], 3),
            "progress_m": e["progress_m"], "crash_s_m": round(s_crash, 2), "crash_idx": i,
            "half_width_m": a["w"][i], "curvature_1m_abs": a["k"][i], "curvature_2m_abs": a["k2"][i],
            "lookback_min_width_m": a["ww"][i], "lookback_max_curv": a["kw"][i],
            "outside_train_point": bool(ood(a["w"][i], a["k"][i])),
            "outside_train_lookback": bool(ood(a["ww"][i], a["kw"][i])),
            "narrower_than_train": bool(a["w"][i] < ranges["width_min"]),
            "sharper_than_train": bool(a["k"][i] > ranges["curv_max"]),
            "sharper_than_train_2m": bool(a["k2"][i] > ranges["curv2_max"]),
            "outside_own_pool_point": bool(ood(a["w"][i], a["k"][i], pr)),
            "outside_own_pool_lookback": bool(ood(a["ww"][i], a["kw"][i], pr)),
            # rank within the same circuit: 0 = most extreme point on the circuit
            "width_rank_track": rank_low(a["w"], a["w"][i]),
            "curv_rank_track": rank_high(a["k"], a["k"][i]),
            "lookback_width_rank_track": rank_low(a["ww"], a["ww"][i]),
            "lookback_curv_rank_track": rank_high(a["kw"], a["kw"][i]),
            "n_driven_points": int(len(drv)),
            "width_rank_driven": rank_low(a["w"][drv], a["w"][i]) if len(drv) >= 100 else np.nan,
            "curv_rank_driven": rank_high(a["k"][drv], a["k"][i]) if len(drv) >= 100 else np.nan,
            # out-of-training share of this circuit, for the track-matched null
            "p_ood_point_track": float(ood(a["w"], a["k"]).mean()),
            "p_ood_lookback_track": float(ood(a["ww"], a["kw"]).mean()),
            "p_ood_own_pool_point_track": float(ood(a["w"], a["k"], pr).mean()),
        }
        recs.append(rec)
        for j in RNG.integers(0, n, N_RANDOM):
            rand_recs.append({"set": e["set"], "track": e["track"],
                              "half_width_m": a["w"][j], "abs_curv": a["k"][j]})

    crash = pd.DataFrame(recs)
    crash.to_csv(OUT / "sac_crash_points.csv", index=False, float_format="%.5g")
    rand = pd.DataFrame(rand_recs)
    plot_crashes(crash.rename(columns={"curvature_1m_abs": "abs_curv"}), rand, ranges,
                 OUT / "crash_vs_random.png")

    # ---- statistics
    def null_p(observed: int, p: np.ndarray) -> float:
        """P(at least `observed` out-of-training crashes) if crashes fell at
        uniformly random points of their own circuits (Poisson-binomial)."""
        sims = (RNG.random((N_SIM, len(p))) < p).sum(axis=1)
        return float((np.sum(sims >= observed) + 1) / (N_SIM + 1))

    def block(df: pd.DataFrame) -> dict:
        out = {"n_unique_crashes": int(len(df)),
               "n_crash_episodes": int(df["n_identical_episodes"].sum())}
        for tag, flag, p in (("point", "outside_train_point", "p_ood_point_track"),
                             ("lookback_5m", "outside_train_lookback", "p_ood_lookback_track"),
                             ("own_pool_point", "outside_own_pool_point", "p_ood_own_pool_point_track")):
            obs = int(df[flag].sum())
            exp = float(df[p].sum())
            out[f"outside_training_{tag}"] = {
                "observed": obs, "observed_share": round(obs / max(len(df), 1), 3),
                "expected_if_random": round(exp, 1),
                "expected_share": round(exp / max(len(df), 1), 3),
                "ratio": round(obs / exp, 2) if exp > 0 else None,
                "p_value_one_sided": null_p(obs, df[p].to_numpy()) if len(df) else None,
            }
        out["narrower_than_train"] = int(df["narrower_than_train"].sum())
        out["sharper_than_train"] = int(df["sharper_than_train"].sum())
        out["sharper_than_train_2m_window"] = int(df["sharper_than_train_2m"].sum())
        for col in ("width_rank_track", "curv_rank_track", "lookback_width_rank_track",
                    "lookback_curv_rank_track", "width_rank_driven", "curv_rank_driven"):
            v = df[col].dropna().to_numpy()
            if len(v):
                boot = RNG.choice(v, (5000, len(v))).mean(axis=1)
                out[f"mean_{col}"] = {"mean": round(float(v.mean()), 3),
                                      "ci95": [round(float(np.percentile(boot, 2.5)), 3),
                                               round(float(np.percentile(boot, 97.5)), 3)],
                                      "n": int(len(v))}
        return out

    def distance_block(df: pd.DataFrame) -> dict:
        return {"median_progress_m": round(float(df["progress_m"].median()), 1),
                "share_within_15m_of_spawn": round(float((df["progress_m"] < 15).mean()), 3)}

    results = {}
    for split in ("real", "test"):
        cs = crash[crash["set"] == split]
        results[split] = {"all_cells": {**block(cs), **distance_block(cs)}}
        for cell, g in cs.groupby("cell"):
            results[split][cell] = {**block(g), **distance_block(g)}

    # Circuit level: on real circuits every point is narrower than training,
    # so narrowness can act on the whole circuit rather than at a crash spot.
    # Does a cell get further on wider circuits? (All episodes, crash or not.)
    from scipy.stats import spearmanr
    circuit_level = {}
    real_ep = ep[ep["set"] == "real"]
    real_summary = summary_df[summary_df["split"] == "real"].set_index("track")
    for cell, g in real_ep.groupby("cell"):
        m = g.groupby("track")["progress_m"].mean()
        t = real_summary.loc[m.index]
        circuit_level[cell] = {}
        for col in ("width_median_m", "width_min_m", "curv_max_per_m", "curv_p90_per_m"):
            rho, p = spearmanr(t[col], m)
            circuit_level[cell][f"spearman_mean_progress_vs_{col}"] = {
                "rho": round(float(rho), 3), "p": round(float(p), 4), "n_circuits": int(len(m))}

    circuits = {}
    for name in manifest["splits"]["real"]["names"]:
        a = arr[name]
        circuits[name] = {
            "share_outside_training": round(float(ood(a["w"], a["k"]).mean()), 4),
            "share_narrower": round(float((a["w"] < ranges["width_min"]).mean()), 4),
            "share_sharper": round(float((a["k"] > ranges["curv_max"]).mean()), 4),
            "sac_unique_crashes": int((crash["track"] == name).sum()),
            "sac_crashes_outside_training": int(crash.loc[crash["track"] == name, "outside_train_point"].sum()),
        }

    by_split = {}
    for split in geo_splits:
        sd = summary_df[summary_df["split"] == split]
        p = points[split]
        by_split[split] = {
            "n_tracks": int(len(sd)),
            "width_min_m": round(float(p["half_width_m"].min()), 3),
            "width_median_m": round(float(p["half_width_m"].median()), 3),
            "curv_max_per_m": round(float(p["abs_curv"].max()), 3),
            "curv_p90_per_m": round(float(p["abs_curv"].quantile(0.9)), 3),
            "share_outside_training": round(float(ood(p["half_width_m"].to_numpy(),
                                                      p["abs_curv"].to_numpy()).mean()), 4),
            "share_narrower_than_training": round(float((p["half_width_m"] < ranges["width_min"]).mean()), 4),
            "share_sharper_than_training": round(float((p["abs_curv"] > ranges["curv_max"]).mean()), 4),
        }

    summary = {
        "method": {
            "sampling": f"simulator centreline resampling, every {DS} m of spline arc length",
            "half_width": "EDT of loader's thresholded occupancy, bilinear, minus half a pixel",
            "curvature": f"heading change over a centred {CURV_WINDOW_M} m arc (alt {CURV_WINDOW_ALT_M} m)",
            "lookback_m": LOOKBACK_M,
            "outside_training": "half-width < training min OR |curvature| > training max",
            "dedup": "one record per (cell, circuit, spawn index): deterministic replays are identical",
        },
        "training_range": {k: round(v, 4) for k, v in ranges.items()},
        "training_range_per_pool": {f"d{d}": {k: round(v, 4) for k, v in r.items()}
                                    for d, r in pool_ranges.items()},
        "by_split": by_split,
        "spawn_check": spawn_check,
        "episodes": {"sac_rows": int(len(ep)),
                     "outcomes": {k: int(v) for k, v in ep["outcome"].value_counts().items()},
                     "crash_rows": int(len(crashed)), "unique_crashes": int(len(crash))},
        "sac_crashes": results,
        "real_circuit_level": circuit_level,
        "narrow_vs_test": {k: v for k, v in narrow_check.items() if k != "per_track"} if narrow_check else None,
        "real_circuits": circuits,
    }
    with open(OUT / "summary.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    # ---- console report
    print(f"training range: half-width >= {ranges['width_min']:.3f} m, "
          f"|curvature| <= {ranges['curv_max']:.3f} 1/m (2 m window {ranges['curv2_max']:.3f})")
    for split, b in by_split.items():
        print(f"  {split:<5} {b['n_tracks']:>3} tracks  width min {b['width_min_m']:.2f} "
              f"median {b['width_median_m']:.2f}  |k| max {b['curv_max_per_m']:.2f} "
              f"p90 {b['curv_p90_per_m']:.2f}  outside training {100 * b['share_outside_training']:.1f}% "
              f"(narrower {100 * b['share_narrower_than_training']:.1f}%, "
              f"sharper {100 * b['share_sharper_than_training']:.1f}%)")
    if narrow_check:
        for split, c in narrow_check.items():
            if split == "per_track":
                continue
            print(f"  {split} vs test: centreline identical {c['centreline_identical']}, map frame "
                  f"identical {c['map_frame_identical']}, curvature identical {c['curvature_identical']}; "
                  f"half-width median {c['half_width_median_m']:.3f} m (test {c['test_half_width_median_m']:.3f}), "
                  f"range {c['half_width_min_m']:.3f}-{c['half_width_max_m']:.3f}")
    print(f"spawn check: {spawn_check}")
    print(f"SAC episodes: {summary['episodes']}")
    print("SAC crashes (rank within own circuit: 0.5 = random, low = narrower/sharper):")
    for split in ("real", "test"):
        for cell, b in results[split].items():
            print(f"  {split:<4} {cell:<9} n {b['n_unique_crashes']:>3}  narrower than training "
                  f"{b['narrower_than_train']:>3}  sharper {b['sharper_than_train']:>2}  "
                  f"width rank {b['mean_width_rank_track']['mean']:.2f}  "
                  f"curv rank {b['mean_curv_rank_track']['mean']:.2f}  "
                  f"median progress {b['median_progress_m']:>5.1f} m  "
                  f"<15 m {100 * b['share_within_15m_of_spawn']:>3.0f}%")
    print("real circuits, mean progress vs circuit geometry (Spearman over 23 circuits):")
    for cell, c in circuit_level.items():
        w = c["spearman_mean_progress_vs_width_median_m"]
        k = c["spearman_mean_progress_vs_curv_max_per_m"]
        print(f"  {cell:<9} median width rho {w['rho']:+.2f} (p {w['p']:.3f})   "
              f"max curvature rho {k['rho']:+.2f} (p {k['p']:.3f})")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
