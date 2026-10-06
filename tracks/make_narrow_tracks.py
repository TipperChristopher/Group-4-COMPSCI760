"""Narrow copies of the 20 test tracks: same shapes, narrower corridor.

    python tracks/make_narrow_tracks.py               # build, install, record in manifest
    python tracks/make_narrow_tracks.py --calibrate   # measure WIDTH -> half-width only

WHAT THIS PRODUCES

    narrowA  narrowA_track_0 .. narrowA_track_19   half-width ~1.07 m (real-circuit median)
    narrowB  narrowB_track_0 .. narrowB_track_19   half-width ~0.75 m (near the real minimum)

narrowX_track_k has exactly the centreline of test_track_k (raw generator
index 110 + k, seed 123). Only the walls move: the training and test tracks
are drawn with WIDTH 10, which measures as a 1.47 m half-width.

WHY NOT JUST EDIT WIDTH AND RERUN THE GENERATOR

random_trackgen.py retries a track whenever any step raises, and the
wall-offset step (shapely buffer) is one of those steps. A different WIDTH can
make a different attempt fail or succeed, the retry consumes RNG draws, and
every later track in the stream changes shape. Second, convert_track sizes
its axes with tight_layout() around tick labels of the autoscaled wall extent,
so narrower walls can shift the axes and rescale the whole track, centreline
included (this happened on test shape 3 when first tried).

So this script replays the generator's own loop exactly as it ran
(create_track and convert_track at the original WIDTH 10, same seed, same
retries) and checks that all 130 replayed tracks match tracks/manifest.json.
Then, for indices 110-129 only, it redraws that same centreline's walls with
the generator's buffer code at a narrower WIDTH, rendering with the layout
pinned to the WIDTH-10 walls (render() below, checked byte-for-byte against
convert_track at WIDTH 10). The centreline CSV and map frame are therefore
identical to the test track's; only the walls move.

HOW WIDTH MAPS TO HALF-WIDTH

WIDTH is a polygon offset in the generator's data units, not metres (the
feasibility spike's narrow script treated it as metres). Measured on these 20
shapes with analysis/track_geometry.py's half-width (distance transform of the
loader's occupancy grid), the relation is linear; --calibrate prints the fit.
The WIDTH values below come from that fit.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))
import make_heldout_tracks as mh  # noqa: E402

BASE_WIDTH = 10.0
RAW = range(mh.N_TRAIN + mh.N_VAL, mh.N_TOTAL)        # 110..129, the test tracks
# Target half-widths (m) and the generator WIDTH that produces them, from
# --calibrate: half_width = SLOPE * WIDTH + INTERCEPT on the 20 test shapes.
# Calibration (WIDTH 5.0 / 7.5 / 10.0 -> 0.665 / 1.072 / 1.473 m, max residual
# 2 mm): half_width = 0.16163 * WIDTH - 0.14201. The feasibility spike's WIDTH
# 6.0 would give ~0.83 m.
SPLITS = {
    "narrowA": {"target_half_width_m": 1.07, "width": 7.5},
    "narrowB": {"target_half_width_m": 0.75, "width": 5.52},
}
SLOPE, INTERCEPT = 0.16163, -0.14201
CALIBRATION_WIDTHS = (5.0, 7.5, 10.0)


def name_of(split: str, raw_index: int) -> str:
    return f"{split}_track_{raw_index - RAW.start}"


def generator():
    import importlib.util
    import matplotlib
    matplotlib.use("Agg")
    spec = importlib.util.spec_from_file_location("random_trackgen", mh.GENERATOR)
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    src = mh.GENERATOR.read_text()
    # The redraw below copies these two lines; refuse if the generator changed.
    for line in ("track_xy_offset_in = track_poly.buffer(WIDTH)",
                 "track_xy_offset_out = track_poly.buffer(-WIDTH)",
                 f"WIDTH = {BASE_WIDTH}"):
        if line not in src:
            raise SystemExit(f"random_trackgen.py no longer contains '{line}'; "
                             "the narrow redraw would not match the generator.")
    return rt


def render(track, track_int, track_ext, track_id, outdir, layout_walls,
           limits_from_layout_only=False):
    """random_trackgen.convert_track, with the layout pinned to the WIDTH-10 walls.

    convert_track calls tight_layout() before fixing the axis limits, so the
    axes box depends on the tick labels of the autoscaled wall extent. Drawing
    narrower walls can change a tick label and shift the box, which rescales
    the pixel->metre mapping of the whole track, centreline included. Adding
    the WIDTH-10 walls as invisible lines (they count for autoscaling but are
    not drawn) reproduces the original layout exactly. Every other line is
    the generator's. Checked byte-for-byte against convert_track at WIDTH 10.

    Walls WIDER than WIDTH 10 would still enlarge the autoscaled extent. With
    limits_from_layout_only=True the visible walls are added as plain artists,
    which are drawn identically but never enter the data limits, so only the
    WIDTH-10 walls set the layout whatever the width.
    """
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 20)
    for w in layout_walls:
        ax.plot(w[:, 0], w[:, 1], color="black", linewidth=3, visible=False)
    if limits_from_layout_only:
        for w in (track_int, track_ext):
            ax.add_artist(Line2D(w[:, 0], w[:, 1], color="black", linewidth=3))
    else:
        ax.plot(track_int[:, 0], track_int[:, 1], color="black", linewidth=3)
        ax.plot(track_ext[:, 0], track_ext[:, 1], color="black", linewidth=3)
    plt.tight_layout()
    ax.set_aspect("equal")
    ax.set_xlim(-180, 300)
    ax.set_ylim(-300, 300)
    plt.axis("off")

    track_filepath = outdir / f"map{track_id}_map.png"
    plt.savefig(track_filepath, dpi=80)
    plt.close()

    xy_pixels = ax.transData.transform(track)
    origin_x_pix = xy_pixels[0, 0]
    origin_y_pix = xy_pixels[0, 1]
    xy_pixels = xy_pixels - np.array([[origin_x_pix, origin_y_pix]])
    map_origin_x = -origin_x_pix * 0.05
    map_origin_y = -origin_y_pix * 0.05

    cv_img = cv2.imread(str(track_filepath), -1)
    cv_img_bw = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(track_filepath), cv_img_bw)
    cv2.imwrite(str(track_filepath.with_suffix(".pgm")), cv_img_bw)

    with open(track_filepath.with_suffix(".yaml"), "w") as yaml:
        yaml.write(f"image: map{track_id}.pgm\n")
        yaml.write("resolution: 0.062500\n")
        yaml.write(f"origin: [{map_origin_x},{map_origin_y},0.000000]\n")
        yaml.write("negate: 0\n")
        yaml.write("occupied_thresh: 0.45\n")
        yaml.write("free_thresh: 0.196")

    with open(outdir / f"map{track_id}_centerline.csv", "w") as waypoints_csv:
        waypoints_csv.write("#x,y\n")
        for row in xy_pixels:
            waypoints_csv.write(f"{0.05 * row[0]}, {0.05 * row[1]}\n")


def walls(track_xy: np.ndarray, width: float):
    """The generator's wall step (create_track), at a chosen WIDTH."""
    import shapely.geometry as shp
    poly = shp.Polygon(track_xy)
    return (np.array(poly.buffer(width).exterior.xy).T,
            np.array(poly.buffer(-width).exterior.xy).T)


def replay(raw_dir: pathlib.Path, redraw: dict, indices=RAW,
           limits_from_layout_only=False) -> None:
    """Replay random_trackgen.main exactly; also redraw some tracks at other widths.

    raw_dir/base gets every track at WIDTH 10, as the generator wrote it;
    raw_dir/<key> gets the tracks in `indices` redrawn at redraw[key], either
    one WIDTH for all of them or a {raw index: WIDTH} dict; raw_dir/selfcheck
    gets the same tracks at WIDTH 10 through render() in the same mode, which
    must equal base byte-for-byte.
    """
    import contextlib
    import io
    rt = generator()
    np.random.seed(mh.SEED)
    for key in ("base", "selfcheck", *redraw):
        (raw_dir / key).mkdir(parents=True)
    for i in range(mh.N_TOTAL):
        while True:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    track, t_int, t_ext = rt.create_track()
                    rt.convert_track(track, t_int, t_ext, i, raw_dir / "base")
                break
            except Exception:
                continue
        if i in indices:
            # Same buffer code at WIDTH 10 must give the generator's own walls.
            a, b = walls(track, BASE_WIDTH)
            assert np.array_equal(a, t_int) and np.array_equal(b, t_ext), i
            render(track, t_int, t_ext, i, raw_dir / "selfcheck", (t_int, t_ext),
                   limits_from_layout_only)
            for f in (f"map{i}_map.yaml", f"map{i}_map.pgm", f"map{i}_centerline.csv"):
                if (raw_dir / "selfcheck" / f).read_bytes() != (raw_dir / "base" / f).read_bytes():
                    raise SystemExit(f"render() does not reproduce convert_track for {f}; "
                                     "the generator has changed. Nothing written.")
            for key, w in redraw.items():
                ti, te = walls(track, w[i] if isinstance(w, dict) else w)
                render(track, ti, te, i, raw_dir / key, (t_int, t_ext),
                       limits_from_layout_only)


def check_base(pkg: pathlib.Path, manifest: dict) -> None:
    """Every replayed WIDTH-10 track must equal its manifest entry."""
    bad = [mh.split_of(i)[1] for i in range(mh.N_TOTAL)
           if not mh.same(manifest["tracks"][mh.split_of(i)[1]]["files"],
                          mh.checksums(pkg, mh.split_of(i)[1]))]
    if bad:
        raise SystemExit(f"replay does not reproduce {len(bad)} manifest tracks, e.g. {bad[:5]}; "
                         "nothing written.")
    print(f"replay reproduces all {mh.N_TOTAL} manifest tracks exactly (train, val, test)")


def load_from_dir(folder: pathlib.Path, name: str):
    """Track.from_track_name's steps, for a folder outside maps/."""
    from PIL import Image
    from PIL.Image import Transpose
    from f1tenth_gym.envs.track import Track
    from f1tenth_gym.envs.track.raceline import Raceline
    spec = Track.load_spec(track=name, filespec=str(folder / f"{name}_map.yaml"))
    img = Image.open(folder / spec.image).transpose(Transpose.FLIP_TOP_BOTTOM)
    occ = np.array(img).astype(np.float32)
    occ[occ <= 128] = 0.0
    occ[occ > 128] = 255.0
    cl = Raceline.from_centerline_file(folder / f"{name}_centerline.csv")
    return Track(spec=spec, filepath=str(folder / name), ext=".pgm",
                 occupancy_map=occ, centerline=cl, raceline=cl)


def median_half_width(root: pathlib.Path, names) -> np.ndarray:
    import track_geometry as tg
    return np.array([float(np.median(tg.track_geometry(n, load_from_dir(root / n, n))[0]["half_width_m"]))
                     for n in names])


def cmd_calibrate() -> int:
    manifest = json.loads(mh.MANIFEST.read_text())
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t)
        keys = {f"w{w}": w for w in CALIBRATION_WIDTHS}
        replay(t / "raw", keys)
        rows = []
        for key, w in keys.items():
            names = [name_of(key, i) for i in RAW]
            for i, n in zip(RAW, names):
                mh.package(t / "raw" / key, i, t / "pkg", n)
            hw = median_half_width(t / "pkg", names)
            rows.append((w, float(np.median(hw)), float(hw.min()), float(hw.max())))
            print(f"WIDTH {w:>5}: half-width median {np.median(hw):.3f} m "
                  f"(per-track medians {hw.min():.3f}-{hw.max():.3f})")
    w = np.array([r[0] for r in rows])
    h = np.array([r[1] for r in rows])
    slope, intercept = np.polyfit(w, h, 1)
    resid = h - (slope * w + intercept)
    print(f"fit: half_width = {slope:.5f} * WIDTH + {intercept:+.5f}   "
          f"(max residual {np.abs(resid).max() * 1000:.1f} mm)")
    for split, s in SPLITS.items():
        print(f"  {split}: target {s['target_half_width_m']} m -> WIDTH "
              f"{(s['target_half_width_m'] - intercept) / slope:.3f}")
    return 0


def cmd_generate(args) -> int:
    if SLOPE is None or any(s["width"] is None for s in SPLITS.values()):
        raise SystemExit("Run --calibrate first and fill in SLOPE, INTERCEPT and the widths.")
    manifest = json.loads(mh.MANIFEST.read_text())
    maps = mh.maps_dir()
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t)
        replay(t / "raw", {k: s["width"] for k, s in SPLITS.items()})
        for i in range(mh.N_TOTAL):
            mh.package(t / "raw" / "base", i, t / "pkg", mh.split_of(i)[1])
        check_base(t / "pkg", manifest)

        sums = {}
        for split in SPLITS:
            for i in RAW:
                n = name_of(split, i)
                mh.package(t / "raw" / split, i, t / "pkg", n)
                sums[n] = mh.checksums(t / "pkg", n)
                test = mh.split_of(i)[1]
                same_cl = (sums[n][f"{n}_centerline.csv"]["sha256_lf"]
                           == manifest["tracks"][test]["files"][f"{test}_centerline.csv"]["sha256_lf"])
                if not same_cl:
                    raise SystemExit(f"{n}: centreline differs from {test}; nothing installed.")
        print(f"centreline CSVs identical to the matching test track: {len(sums)}/{len(sums)}")

        installed = 0
        for n in sums:
            dest = maps / n
            if dest.exists():
                if mh.same(sums[n], mh.checksums(maps, n)):
                    continue
                if not args.force:
                    raise SystemExit(f"{dest} exists with different contents; pass --force.")
                shutil.rmtree(dest)
            shutil.copytree(t / "pkg" / n, dest)
            installed += 1
        print(f"installed {installed} new, {len(sums) - installed} already present and identical")

    for split, s in SPLITS.items():
        names = [name_of(split, i) for i in RAW]
        manifest["splits"][split] = {
            "names": names,
            "raw_indices": [RAW.start, RAW.stop - 1],
            "same_shapes_as": "test",
            "generator_width": s["width"],
            "base_generator_width": BASE_WIDTH,
            "target_half_width_m": s["target_half_width_m"],
            "width_calibration": {"half_width_m": f"{SLOPE} * WIDTH + {INTERCEPT}",
                                  "measured_with": "analysis/track_geometry.py"},
            "regenerate_with": "python tracks/make_narrow_tracks.py",
        }
        for i, n in zip(RAW, names):
            manifest["tracks"][n] = {"split": split, "raw_index": i, "files": sums[n]}
    mh.write_json(mh.MANIFEST, manifest)
    print(f"manifest updated: {', '.join(SPLITS)} ({len(SPLITS) * len(RAW)} tracks)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--calibrate", action="store_true",
                    help="Measure how WIDTH maps to half-width; write nothing.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite differing narrow tracks. Never touches other splits.")
    args = ap.parse_args()
    return cmd_calibrate() if args.calibrate else cmd_generate(args)


if __name__ == "__main__":
    sys.exit(main())
