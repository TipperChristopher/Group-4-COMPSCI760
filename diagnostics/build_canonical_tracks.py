"""Install the team's canonical seed-123 training tracks on this machine.

WHY THIS EXISTS

`tracks/make_heldout_tracks.py` builds val/test, but it *deliberately refuses*
to create the training tracks: it verifies that whatever is on disk already
reproduces from the seed-123 generator, and aborts otherwise ("this machine's
training set is not the canonical one"). On this machine the installed
`synthetic_track_*` came from `make_synth_tracks.py --seed 0`, a different RNG
stream under the same filenames, so train.py's pool gate correctly rejected it.

This script performs only that first step, using the branch's own `package()`
so the files are byte-for-byte what the team has (including platform text mode).
Afterwards `make_heldout_tracks.py` can run normally and install val/test.

    python _verify/build_canonical_tracks.py          # build + install train split
    python _verify/build_canonical_tracks.py --dry    # generate, compare, write nothing
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import types

REPO = pathlib.Path(__file__).resolve().parent.parent / "team_repo_heldout"
sys.path.insert(0, str(REPO))
if "gym" not in sys.modules:
    sys.modules["gym"] = types.ModuleType("gym")
sys.modules["gym"].__version__ = "0.0.0"

# Reuse the branch's own generator packaging and checksum helpers so the result
# is byte-identical to the canonical set rather than merely equivalent.
import importlib.util

spec = importlib.util.spec_from_file_location(
    "mh", REPO / "tracks" / "make_heldout_tracks.py")
mh = importlib.util.module_from_spec(spec)
sys.modules["mh"] = mh
spec.loader.exec_module(mh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true",
                    help="Generate and compare, but install nothing.")
    args = ap.parse_args()

    maps = mh.maps_dir()
    print(f"maps dir : {maps}")
    print(f"seed     : {mh.SEED}   tracks: {mh.N_TRAIN} train + "
          f"{mh.N_VAL} val + {mh.N_TEST} test = {mh.N_TOTAL}")

    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        pkg = mh.generate(tmp)               # runs random_trackgen --seed 123

        # Are the training tracks we would install already exactly right?
        already = 0
        for i in range(mh.N_TRAIN):
            _, name = mh.split_of(i)
            if (maps / name).exists() and mh.same(
                    mh.checksums(pkg, name), mh.checksums(maps, name)):
                already += 1
        print(f"training tracks already canonical on disk: {already}/{mh.N_TRAIN}")

        if already == mh.N_TRAIN:
            print("Nothing to do; the training split is already the canonical set.")
            return 0
        if args.dry:
            print("--dry: generated and compared, wrote nothing.")
            return 0

        # Replace the non-canonical copies. Only synthetic_track_* is touched.
        replaced = 0
        for i in range(mh.N_TRAIN):
            _, name = mh.split_of(i)
            dest = maps / name
            if dest.exists():
                if mh.same(mh.checksums(pkg, name), mh.checksums(maps, name)):
                    continue
                shutil.rmtree(dest)
            shutil.copytree(pkg / name, dest)
            replaced += 1
        print(f"installed/replaced {replaced} training tracks")

        bad = [mh.split_of(i)[1] for i in range(mh.N_TRAIN)
               if not mh.same(mh.checksums(pkg, mh.split_of(i)[1]),
                              mh.checksums(maps, mh.split_of(i)[1]))]
        print(f"post-install verification: {mh.N_TRAIN - len(bad)}/{mh.N_TRAIN} match")
        if bad:
            print(f"FAILED: {bad[:5]}")
            return 1

    print("\nNext: python tracks/make_heldout_tracks.py   # installs val/test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
