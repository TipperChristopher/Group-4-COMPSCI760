"""Varied-width copies of all 130 synthetic tracks: same shapes, one width each.

    python tracks/make_vw_tracks.py            # build, verify, install, record in manifest

WHAT THIS PRODUCES

    vw_train  vw_synthetic_track_0..99   shapes of synthetic_track_0..99  (raw 0-99)
    vw_val    vw_val_track_0..9          shapes of val_track_0..9        (raw 100-109)
    vw_test   vw_test_track_0..19        shapes of test_track_0..19      (raw 110-129)

Each track keeps its original centreline exactly and gets ONE constant
half-width in [0.60, 1.50] m (the originals are all 1.47 m; real circuits run
0.65-1.30 m with median 1.07 m).

HOW WIDTHS ARE ASSIGNED

Track k of a split (k = 0, 1, 2, ...) gets

    u_k        = van der Corput (base 2) of k + 1   -> 0.5, 0.25, 0.75, 0.125, ...
    half-width = 0.60 + 0.90 * u_k

The sequence is deterministic, so it needs no seed, and it is low-discrepancy
in prefix order: the first 1, 5, 20 and 100 training tracks (the nested
diversity pools) each cover the range about as evenly as that many points can.
Track 0 sits at the middle (1.05 m). val and test restart the sequence, so
each split spans the range on its own.

The generator WIDTH for a half-width comes from the calibration measured in
make_narrow_tracks.py (half_width = 0.16163 * WIDTH - 0.14201, 2 mm max
residual). Every built track is measured again with analysis/track_geometry.py
and refused if its median half-width misses the target by more than 1.5 cm.

HOW SHAPES ARE KEPT IDENTICAL

Exactly as in make_narrow_tracks.py: replay the generator at WIDTH 10 (all 130
must match tracks/manifest.json), then redraw only the walls with the layout
pinned to the WIDTH-10 walls. Widths up to 1.50 m put some walls slightly
OUTSIDE the WIDTH-10 walls, so the visible walls are added without entering
the axis limits (render(limits_from_layout_only=True)); that mode is checked
byte-for-byte against the generator at WIDTH 10 on all 130 shapes. Every
centreline CSV must equal its original's or nothing is installed.
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
sys.path.insert(0, str(HERE))
import make_heldout_tracks as mh  # noqa: E402
import make_narrow_tracks as nt  # noqa: E402

HW_MIN, HW_MAX = 0.60, 1.50
TOLERANCE_M = 0.015
VW_SPLIT = {"train": "vw_train", "val": "vw_val", "test": "vw_test"}


def van_der_corput(n: int, base: int = 2) -> float:
    q, denom = 0.0, 1.0
    while n:
        n, r = divmod(n, base)
        denom *= base
        q += r / denom
    return q


def plan() -> list[dict]:
    """One row per raw index 0..129: names, target half-width, generator WIDTH."""
    rows = []
    for i in range(mh.N_TOTAL):
        split, orig = mh.split_of(i)
        k = int(orig.rsplit("_", 1)[1])                 # index within its split
        hw = round(HW_MIN + (HW_MAX - HW_MIN) * van_der_corput(k + 1), 6)
        rows.append({"raw_index": i, "split": VW_SPLIT[split], "twin": orig,
                     "name": f"vw_{orig}", "k": k, "target_half_width_m": hw,
                     "generator_width": round((hw - nt.INTERCEPT) / nt.SLOPE, 6)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true",
                    help="Overwrite differing vw tracks. Never touches other splits.")
    args = ap.parse_args()

    rows = plan()
    manifest = json.loads(mh.MANIFEST.read_text())
    maps = mh.maps_dir()
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t)
        nt.replay(t / "raw", {"vw": {r["raw_index"]: r["generator_width"] for r in rows}},
                  indices=range(mh.N_TOTAL), limits_from_layout_only=True)
        print(f"render() in pinned-limits mode reproduces the generator byte-for-byte "
              f"at WIDTH 10 on all {mh.N_TOTAL} shapes")
        for i in range(mh.N_TOTAL):
            mh.package(t / "raw" / "base", i, t / "base_pkg", mh.split_of(i)[1])
        nt.check_base(t / "base_pkg", manifest)

        for r in rows:
            mh.package(t / "raw" / "vw", r["raw_index"], t / "pkg", r["name"])
            r["files"] = mh.checksums(t / "pkg", r["name"])
            twin_cl = manifest["tracks"][r["twin"]]["files"][f"{r['twin']}_centerline.csv"]["sha256_lf"]
            if r["files"][f"{r['name']}_centerline.csv"]["sha256_lf"] != twin_cl:
                raise SystemExit(f"{r['name']}: centreline differs from {r['twin']}; nothing installed.")
        print(f"centreline CSVs identical to the original track: {len(rows)}/{len(rows)}")

        measured = nt.median_half_width(t / "pkg", [r["name"] for r in rows])
        for r, m in zip(rows, measured):
            r["measured_half_width_m"] = round(float(m), 4)
        err = np.array([r["measured_half_width_m"] - r["target_half_width_m"] for r in rows])
        print(f"measured vs target half-width: max |error| {np.abs(err).max() * 1000:.1f} mm, "
              f"mean {err.mean() * 1000:+.1f} mm")
        off = [r["name"] for r, e in zip(rows, err) if abs(e) > TOLERANCE_M]
        if off:
            raise SystemExit(f"{len(off)} tracks miss their target half-width by more than "
                             f"{TOLERANCE_M * 100:.1f} cm, e.g. {off[:5]}; nothing installed.")

        installed = 0
        for r in rows:
            n, dest = r["name"], maps / r["name"]
            if dest.exists():
                if mh.same(r["files"], mh.checksums(maps, n)):
                    continue
                if not args.force:
                    raise SystemExit(f"{dest} exists with different contents; pass --force.")
                shutil.rmtree(dest)
            shutil.copytree(t / "pkg" / n, dest)
            installed += 1
        print(f"installed {installed} new, {len(rows) - installed} already present and identical")

    for orig_split, split in VW_SPLIT.items():
        mine = [r for r in rows if r["split"] == split]
        manifest["splits"][split] = {
            "names": [r["name"] for r in mine],
            "raw_indices": [mine[0]["raw_index"], mine[-1]["raw_index"]],
            "same_shapes_as": orig_split,
            "half_width_range_m": [HW_MIN, HW_MAX],
            "width_assignment": ("track k: half_width = 0.60 + 0.90 * vdc2(k + 1), "
                                 "van der Corput base 2, deterministic, in track order"),
            "width_calibration": {"half_width_m": f"{nt.SLOPE} * WIDTH + {nt.INTERCEPT}",
                                  "measured_with": "analysis/track_geometry.py"},
            "regenerate_with": "python tracks/make_vw_tracks.py",
        }
        for r in mine:
            manifest["tracks"][r["name"]] = {
                "split": split, "raw_index": r["raw_index"], "same_shape_as": r["twin"],
                "target_half_width_m": r["target_half_width_m"],
                "generator_width": r["generator_width"],
                "measured_half_width_m": r["measured_half_width_m"],
                "files": r["files"],
            }
    mh.write_json(mh.MANIFEST, manifest)
    print(f"manifest updated: {', '.join(VW_SPLIT.values())} ({len(rows)} tracks)")
    for split in VW_SPLIT.values():
        hw = [r["target_half_width_m"] for r in rows if r["split"] == split]
        print(f"  {split:<8} first widths: {', '.join(f'{h:.3f}' for h in hw[:8])} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
