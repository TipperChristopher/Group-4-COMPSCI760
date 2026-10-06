"""Checks on the 23 real circuits in the manifest's "real" split.

    python tracks/verify_real_tracks.py          # static checks (seconds)
    python tracks/verify_real_tracks.py --sim    # plus one gap-follower episode each
    python tracks/verify_real_tracks.py --split narrowA [--sim]   # any other split

Static checks, per circuit (or track, with --split):

  1. files match tracks/manifest.json (install with tracks/install_real_tracks.py)
  2. loads in F1TENTH: Track.from_track_name and the evaluation env
  3. closed, simple centreline loop lying in free space
  4. cl_grid_static spawn: on the centreline, in free space, no collision on
     the first step (one zero-speed step, no driving)
  5. length against the evaluation cap: 15,000 steps x 0.01 s = 150 s, so one
     lap at 3 m/s fits only up to 450 m. Longer circuits are flagged, not
     failed: a slow policy can be truncated mid-lap there for reasons that
     have nothing to do with driving ability.

--sim adds one gap-follower episode per circuit (committed baseline,
centreline spawn, one-lap termination, 15,000-step cap). It runs in this one
process; set OMP_NUM_THREADS=1 to keep it to one core. A gap-follower crash is
reported, not failed: the planner crashes on some legitimate tracks.

Results go to tracks/<split>_verification.json (real_verification.json by
default). Exits non-zero on any hard failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

import verify_heldout_tracks as vh  # sets up paths, gym shim, cwd

from scipy.ndimage import distance_transform_edt

CAP = vh.CAP
DT = 0.01
SLOW_SPEED = 3.0
MAX_LEN_AT_SLOW = CAP * DT * SLOW_SPEED  # 450 m


def wall_clearance(track, xy: np.ndarray) -> np.ndarray:
    """Distance from each centreline point to the nearest wall, metres."""
    occ = track.occupancy_map
    res = float(track.spec.resolution)
    ox, oy = float(track.spec.origin[0]), float(track.spec.origin[1])
    dt = distance_transform_edt(occ > 128) * res
    c = np.clip(((xy[:, 0] - ox) / res).astype(int), 0, occ.shape[1] - 1)
    r = np.clip(((xy[:, 1] - oy) / res).astype(int), 0, occ.shape[0] - 1)
    return dt[r, c]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sim", action="store_true",
                    help="Also run one gap-follower episode per circuit.")
    ap.add_argument("--split", default="real",
                    help="Manifest split to check (default real).")
    args = ap.parse_args()
    OUT = vh.HERE / f"{args.split}_verification.json"

    manifest = json.loads(vh.MANIFEST.read_text())
    if args.split not in manifest["splits"]:
        raise SystemExit(f"tracks/manifest.json has no {args.split} split.")
    names = manifest["splits"][args.split]["names"]
    mh = vh._load("make_heldout_tracks", "tracks/make_heldout_tracks.py")
    maps = mh.maps_dir()
    ev = vh._load("evaluate_mod", "evaluate.py")
    B = sys.modules["baselines"]
    from f1tenth_gym.envs.track import Track

    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    results, hard_fail, long_tracks = {}, [], []

    hdr = (f"  {'circuit':<14} {'len m':>6} {'steps@3':>8} {'closed':>6} {'simple':>6} "
           f"{'CL free':>8} {'min clr':>7} {'|ey|':>6} {'spawn clr':>9} {'coll':>5}"
           + (f" {'gap-follower':>20}" if args.sim else ""))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for name in names:
        r = {}
        files = manifest["tracks"][name]["files"]
        try:
            r["checksums_match"] = mh.same(files, mh.checksums(maps, name, list(files)))
        except FileNotFoundError:
            r["checksums_match"] = False
        if not r["checksums_match"]:
            hard_fail.append(f"{name}: files missing or differ from manifest")
            results[name] = r
            print(f"  {name:<14} FILES MISSING OR DIFFER")
            continue

        # loads
        try:
            track = Track.from_track_name(name)
            r["loads"] = True
        except Exception as ex:
            r["loads"] = False
            r["load_error"] = repr(ex)
            hard_fail.append(f"{name}: does not load ({ex})")
            results[name] = r
            print(f"  {name:<14} DOES NOT LOAD: {ex}")
            continue

        # loop geometry, world frame
        xy = vh.centreline_xy(maps, name)
        r.update(vh.loop_checks(xy, track))
        clr = wall_clearance(track, xy)
        r["min_wall_clearance_m"] = round(float(clr.min()), 3)
        r["median_wall_clearance_m"] = round(float(np.median(clr)), 3)
        r["has_raceline_csv"] = (maps / name / f"{name}_raceline.csv").exists()
        r["steps_for_one_lap_at_3mps"] = int(np.ceil(r["length_m"] / SLOW_SPEED / DT))
        r["exceeds_cap_at_3mps"] = bool(r["length_m"] > MAX_LEN_AT_SLOW)
        if r["exceeds_cap_at_3mps"]:
            long_tracks.append(name)

        # spawn: one reset and one zero-speed step
        env = ev.build_env(name, False, 0, CAP, 1, "cl_grid_static")()
        obs, _ = env.reset(seed=vh.SEED)
        core = env.unwrapped
        px, py, pth = float(core.poses_x[0]), float(core.poses_y[0]), float(core.poses_theta[0])
        s0, ey, _ = core.track.cartesian_to_frenet(px, py, pth)
        _, _, _, _, info0 = env.step(np.array([0.0, -1.0], dtype=np.float32))
        env.close()
        r["spawn_pose"] = [round(px, 3), round(py, 3), round(pth, 4)]
        r["spawn_abs_ey_m"] = round(abs(float(ey)), 4)
        r["spawn_wall_clearance_m"] = round(float(wall_clearance(track, np.array([[px, py]]))[0]), 3)
        r["spawn_collision"] = bool(info0.get("collision"))
        r["spawn_min_lidar_m"] = round(float(np.asarray(obs[:108]).min()), 3)

        if args.sim:
            env = ev.build_env(name, False, 0, CAP, 1, "cl_grid_static")()
            obs, _ = env.reset(seed=vh.SEED)
            pol = B.make_baseline("gap")
            steps, t0, info = 0, time.time(), {}
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
            r["gap_wall_seconds"] = round(time.time() - t0, 1)
        elif name in old.get("tracks", {}) and "gap_outcome" in old["tracks"][name]:
            for k, v in old["tracks"][name].items():
                if k.startswith("gap_"):
                    r[k] = v  # keep an earlier --sim result rather than dropping it

        checks = {
            "checksums_match": r["checksums_match"], "loads": r["loads"],
            "closed": r["closed"], "simple": r["simple"],
            "centreline_in_free_space": r["centreline_free_fraction"] >= 0.99,
            "spawn_on_centreline": r["spawn_abs_ey_m"] < 0.05,
            "spawn_in_free_space": r["spawn_wall_clearance_m"] > 0.0,
            "no_collision_at_spawn": not r["spawn_collision"],
        }
        r["checks"] = checks
        failed = [k for k, v in checks.items() if not v]
        if failed:
            hard_fail.append(f"{name}: {', '.join(failed)}")
        results[name] = r

        line = (f"  {name:<14} {r['length_m']:>6.0f} {r['steps_for_one_lap_at_3mps']:>8} "
                f"{str(r['closed']):>6} {str(r['simple']):>6} "
                f"{r['centreline_free_fraction'] * 100:>7.1f}% {r['min_wall_clearance_m']:>7.2f} "
                f"{r['spawn_abs_ey_m']:>6.3f} {r['spawn_wall_clearance_m']:>9.2f} "
                f"{str(r['spawn_collision']):>5}")
        if args.sim:
            g = ("LAP " if r["gap_lap_completed"] else f"{r['gap_outcome']} ") + f"{r['gap_distance_m']:.0f}m"
            line += f" {g:>20}"
        print(line + ("  > 450 m" if r.get("exceeds_cap_at_3mps") else ""))

    summary = {
        "n_circuits": len(names),
        "n_pass": len(names) - len({f.split(':')[0] for f in hard_fail}),
        "cap_steps": CAP,
        "max_length_for_one_lap_at_3mps_m": MAX_LEN_AT_SLOW,
        "longer_than_that": long_tracks,
        "protocol": {"reset_type": "cl_grid_static", "target_laps": 1,
                     "max_steps": CAP, "seed": vh.SEED},
        "hard_failures": hard_fail,
    }
    if any("gap_outcome" in r for r in results.values()):
        summary["gap_follower"] = B.make_baseline("gap").config()
        summary["gap_follower_laps"] = sum(bool(r.get("gap_lap_completed")) for r in results.values())
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"summary": summary, "tracks": results}, f, indent=2)
        f.write("\n")

    print(f"\n  {summary['n_pass']}/{len(names)} circuits pass every hard check")
    print(f"  longer than {MAX_LEN_AT_SLOW:.0f} m (one lap at {SLOW_SPEED:.0f} m/s exceeds the "
          f"{CAP:,}-step cap): {len(long_tracks)} -> {', '.join(long_tracks) or 'none'}")
    if "gap_follower_laps" in summary:
        print(f"  gap-follower laps: {summary['gap_follower_laps']}/{len(names)}")
    print(f"\nwritten {OUT}")
    if hard_fail:
        print("HARD FAILURES:")
        for f in hard_fail:
            print("  " + f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
