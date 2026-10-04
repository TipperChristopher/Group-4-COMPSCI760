"""Simulator-level checks on the held-out val/test tracks.

    python tracks/verify_heldout_tracks.py

For every track in the manifest's val and test splits this checks that it:

  1. loads in F1TENTH as-is (Track.from_track_name and gym.make), with no
     fix_tracks.py or patch_yamls.py step. Those exist for downloaded real
     circuits and legacy yaml names; the generator's packaging already writes
     the <name>_map.yaml / .pgm / _centerline.csv layout the loader expects.
  2. is a closed, simple loop whose centreline lies in free space.
  3. spawns on the centreline under cl_grid_static: zero lateral offset, no
     collision at reset, free occupancy at the spawn pose.
  4. is not a duplicate or near-duplicate of any training track, nor of any
     other held-out track. Exact duplicates are caught by the occupancy image
     hash. Near-duplicate is judged by WORST-CASE deviation (Hausdorff distance
     between centrelines, in a common frame) against the measured track
     half-width: if one centreline never leaves the other track's corridor, a
     car could drive the same line on both and they are the same track for our
     purposes.

     Mean centreline distance is reported but deliberately NOT used to decide.
     Every track here is a closed loop wound around the same centre within the
     same radius band, so mean nearest-point distance is compressed into a
     narrow range (about 1.2-2.8 m) even for unrelated tracks. A held-out track
     is also compared against more candidates than a training track is, so its
     minimum drifts lower by chance alone. Hausdorff exposes the section where
     two tracks genuinely diverge.
  5. is drivable from the spawn: one gap-follower episode with the committed
     baseline, centreline spawn, one-lap termination.

Results are written to tracks/heldout_verification.json. Exits non-zero if any
hard check fails. A gap-follower crash is reported, not failed: the planner
crashes on some legitimate tracks, so it is evidence a track is not degenerate
only when the car gets well clear of the spawn.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import sys
import types
import warnings

warnings.filterwarnings("ignore")
REPO = pathlib.Path(__file__).resolve().parents[1]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)
if "gym" not in sys.modules:
    sys.modules["gym"] = types.ModuleType("gym")
sys.modules["gym"].__version__ = "0.0.0"

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from shapely.geometry import LinearRing, Polygon

MANIFEST = HERE / "manifest.json"
OUT = HERE / "heldout_verification.json"
CAP = 15000
SEED = 0
DEGENERATE_M = 10.0  # a car that cannot get this far from the spawn is suspect


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def centreline_xy(maps: pathlib.Path, name: str) -> np.ndarray:
    """Centreline in the simulator's world frame (spawn point at the origin)."""
    pts = np.loadtxt(maps / name / f"{name}_centerline.csv", delimiter=",", comments="#")
    return pts[:, :2]


def yaml_origin(maps: pathlib.Path, name: str) -> np.ndarray:
    text = (maps / name / f"{name}_map.yaml").read_text()
    vals = re.search(r"origin:\s*\[([^\]]+)\]", text).group(1).split(",")[:2]
    return np.array([float(v) for v in vals])


def common_xy(maps: pathlib.Path, name: str) -> np.ndarray:
    """Centreline in a frame shared by every track.

    Each track's world frame puts its own spawn at the origin, so two tracks'
    world coordinates are not comparable. Every generated image renders the
    same data window, so subtracting the map origin maps a world point onto the
    shared image grid.
    """
    return centreline_xy(maps, name) - yaml_origin(maps, name)


def mean_dist(a, b, ta, tb) -> float:
    """Symmetric mean nearest-point distance, metres. Reported, not decisive."""
    return 0.5 * (tb.query(a)[0].mean() + ta.query(b)[0].mean())


def hausdorff(a, b, ta, tb) -> float:
    """Largest distance from either centreline to the other, metres."""
    return float(max(tb.query(a)[0].max(), ta.query(b)[0].max()))


def half_width(track, xy_world: np.ndarray) -> float:
    """Median distance from the centreline to the nearest wall, metres."""
    occ = track.occupancy_map
    res = float(track.spec.resolution)
    ox, oy = float(track.spec.origin[0]), float(track.spec.origin[1])
    dt = distance_transform_edt(occ > 128) * res
    c = np.clip(((xy_world[:, 0] - ox) / res).astype(int), 0, occ.shape[1] - 1)
    r = np.clip(((xy_world[:, 1] - oy) / res).astype(int), 0, occ.shape[0] - 1)
    return float(np.median(dt[r, c]))


def loop_checks(xy: np.ndarray, track) -> dict:
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    closure = float(np.linalg.norm(xy[-1] - xy[0]))
    ring = LinearRing(xy)
    occ = track.occupancy_map
    res = float(track.spec.resolution)
    ox, oy = float(track.spec.origin[0]), float(track.spec.origin[1])
    cols = ((xy[:, 0] - ox) / res).astype(int)
    rows = ((xy[:, 1] - oy) / res).astype(int)
    inside = (rows >= 0) & (rows < occ.shape[0]) & (cols >= 0) & (cols < occ.shape[1])
    free = np.zeros(len(xy), dtype=bool)
    free[inside] = occ[rows[inside], cols[inside]] > 128
    return {
        "n_points": int(len(xy)),
        "closure_gap_m": round(closure, 3),
        "median_spacing_m": round(float(np.median(seg)), 3),
        "closed": bool(closure < 3.0 * float(np.median(seg))),
        "simple": bool(ring.is_simple),
        "area_m2": round(float(Polygon(xy).area), 1),
        "length_m": round(float(track.centerline.spline.s[-1]), 1),
        "centreline_free_fraction": round(float(free.mean()), 4),
    }


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    mh = _load("make_heldout_tracks", "tracks/make_heldout_tracks.py")
    maps = mh.maps_dir()
    ev = _load("evaluate_mod", "evaluate.py")
    B = sys.modules["baselines"]
    from f1tenth_gym.envs.track import Track

    train = manifest["splits"]["train"]["names"]
    held = manifest["splits"]["val"]["names"] + manifest["splits"]["test"]["names"]

    # Exact-duplicate check: the occupancy image is the track.
    pgm_sha = {n: manifest["tracks"][n]["files"][f"{n}_map.pgm"]["sha256"]
               for n in train + held}

    # Near-duplicate threshold: the measured track half-width.
    widths = {n: half_width(Track.from_track_name(n), centreline_xy(maps, n))
              for n in train + held}
    threshold = float(min(widths.values()))
    print(f"measured half-width: median {np.median(list(widths.values())):.2f} m, "
          f"min {threshold:.2f} m over {len(widths)} tracks -> near-duplicate if "
          f"worst-case centreline deviation < {threshold:.2f} m")

    xy = {n: common_xy(maps, n) for n in train + held}
    tree = {n: cKDTree(xy[n]) for n in xy}
    train_haus = [min(hausdorff(xy[a], xy[b], tree[a], tree[b]) for b in train if b != a)
                  for a in train]
    print(f"reference, training vs nearest training track: worst-case deviation "
          f"min {min(train_haus):.2f} m, median {np.median(train_haus):.2f} m")

    gap = B.make_baseline("gap")
    print(f"gap-follower safe_threshold = {gap.safe_threshold} beams "
          f"({gap.config()['safe_threshold_deg']} deg)\n")

    results, hard_fail = {}, []
    hdr = (f"  {'track':<14} {'len m':>6} {'closed':>6} {'simple':>6} {'CL free':>8} "
           f"{'|ey|':>6} {'coll':>5} {'nearest (worst-case)':>24} {'gap-follower':>22}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for name in held:
        r = {"split": manifest["tracks"][name]["split"],
             "raw_index": manifest["tracks"][name]["raw_index"]}

        # 1. loads
        track = Track.from_track_name(name)
        r["loads"] = True

        # 2. closed simple loop in free space
        # World frame: the occupancy lookup needs the simulator's coordinates,
        # not the shared comparison frame used by the duplicate check below.
        r.update(loop_checks(centreline_xy(maps, name), track))

        # 3. spawn on the centreline
        env = ev.build_env(name, False, 0, CAP, 1, "cl_grid_static")()
        obs, _ = env.reset(seed=SEED)
        core = env.unwrapped
        px, py, pth = float(core.poses_x[0]), float(core.poses_y[0]), float(core.poses_theta[0])
        _, ey, _ = core.track.cartesian_to_frenet(px, py, pth)
        obs0, _, _, _, info0 = env.step(np.array([0.0, -1.0], dtype=np.float32))
        r["spawn_abs_ey_m"] = round(abs(float(ey)), 4)
        r["spawn_collision"] = bool(info0.get("collision"))
        r["spawn_min_lidar_m"] = round(float(np.asarray(obs[:108]).min()), 3)
        env.close()

        # 4. duplicates
        others = [n for n in train + held if n != name]
        haus = {n: hausdorff(xy[name], xy[n], tree[name], tree[n]) for n in others}
        nearest = min(haus, key=haus.get)
        nearest_train = min(train, key=lambda n: haus[n])
        r["half_width_m"] = round(widths[name], 3)
        r["exact_duplicate_of"] = [n for n in others if pgm_sha[n] == pgm_sha[name]]
        r["nearest_track"] = nearest
        r["nearest_worst_case_m"] = round(haus[nearest], 2)
        r["nearest_mean_distance_m"] = round(
            mean_dist(xy[name], xy[nearest], tree[name], tree[nearest]), 2)
        r["nearest_training_track"] = nearest_train
        r["nearest_training_worst_case_m"] = round(haus[nearest_train], 2)
        r["near_duplicate"] = bool(haus[nearest] < threshold)

        # 5. one gap-follower episode
        env = ev.build_env(name, False, 0, CAP, 1, "cl_grid_static")()
        obs, _ = env.reset(seed=SEED)
        pol = B.make_baseline("gap")
        steps = 0
        for _ in range(CAP):
            a, _ = pol.predict(obs)
            obs, _, term, trunc, info = env.step(a)
            steps += 1
            if term or trunc:
                break
        env.close()
        r["gap_lap_completed"] = bool(info.get("lap_completed"))
        r["gap_outcome"] = ("lap" if r["gap_lap_completed"]
                            else "crash" if info.get("collision") else "timeout")
        r["gap_distance_m"] = round(float(info.get("progress_m", 0.0)), 1)
        r["gap_steps"] = steps

        checks = {
            "loads": r["loads"], "closed": r["closed"], "simple": r["simple"],
            "centreline_in_free_space": r["centreline_free_fraction"] >= 0.99,
            "spawn_on_centreline": r["spawn_abs_ey_m"] < 0.05,
            "no_collision_at_spawn": not r["spawn_collision"],
            "not_exact_duplicate": not r["exact_duplicate_of"],
            "not_near_duplicate": not r["near_duplicate"],
            "clear_of_spawn": r["gap_distance_m"] >= DEGENERATE_M,
        }
        r["checks"] = checks
        failed = [k for k, v in checks.items() if not v]
        if failed:
            hard_fail.append(f"{name}: {', '.join(failed)}")
        results[name] = r

        g = ("LAP " if r["gap_lap_completed"] else f"{r['gap_outcome']} ") + f"{r['gap_distance_m']:.0f}m"
        print(f"  {name:<14} {r['length_m']:>6.0f} {str(r['closed']):>6} {str(r['simple']):>6} "
              f"{r['centreline_free_fraction']*100:>7.1f}% {r['spawn_abs_ey_m']:>6.3f} "
              f"{str(r['spawn_collision']):>5} {nearest + ' ' + format(haus[nearest], '.1f') + 'm':>24} {g:>22}")

    held_haus = [results[n]["nearest_worst_case_m"] for n in held]
    summary = {
        "near_duplicate_rule": ("worst-case centreline deviation (Hausdorff, common "
                                "frame) below the measured track half-width"),
        "near_duplicate_threshold_m": round(threshold, 3),
        "training_vs_nearest_training_worst_case_m": {
            "min": round(min(train_haus), 3),
            "median": round(float(np.median(train_haus)), 3)},
        "heldout_vs_nearest_any_worst_case_m": {
            "min": round(min(held_haus), 3),
            "median": round(float(np.median(held_haus)), 3)},
        "gap_follower": gap.config(),
        "protocol": {"reset_type": "cl_grid_static", "target_laps": 1,
                     "max_steps": CAP, "seed": SEED},
        "hard_failures": hard_fail,
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"summary": summary, "tracks": results}, f, indent=2)
        f.write("\n")

    print()
    for split in ("val", "test"):
        names = manifest["splits"][split]["names"]
        laps = sum(results[n]["gap_lap_completed"] for n in names)
        print(f"  {split:<5} gap-follower laps {laps}/{len(names)}")
    print(f"\nwritten {OUT}")
    if hard_fail:
        print("HARD FAILURES:")
        for f in hard_fail:
            print("  " + f)
        return 1
    print("All hard checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
