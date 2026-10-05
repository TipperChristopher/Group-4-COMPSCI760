"""Download and install the 23 real circuits used for zero-shot evaluation.

    python tracks/install_real_tracks.py              # install, verify vs manifest
    python tracks/install_real_tracks.py --manifest   # (re)record checksums

The circuits come from f1tenth_racetracks via http://api.f1tenth.org/<name>.tar.xz.
Downloading uses f1tenth_gym's own find_track_dir, the same code path a bare
Track.from_track_name(<name>) would take, so what is installed here is what
the simulator would fetch on its own.

THE ONE FIX NEEDED

The tarballs ship <name>.yaml and <name>.png, but f1tenth_gym v1.0.0 loads
<name>_map.yaml. This script copies <name>.yaml -> <name>_map.yaml and
<name>.png -> <name>_map.png, exactly what fix_tracks.py does, but only for the
23 circuits rather than for every folder under maps/. The yaml's image field
already names <name>.png, which exists, so no yaml text is edited.

patch_yamls.py is NOT needed and must not be run for this: it rewrites the
yaml text of synthetic_track_0..99, the training tracks, and touches no real
circuit.

WHAT THE MANIFEST RECORDS

For each circuit, the checksums of the files the loader actually reads:
<name>_map.yaml, the image that yaml names, <name>_centerline.csv, and
<name>_raceline.csv where the circuit has one (Montreal and Shanghai do not;
the loader then uses the centreline as the raceline). evaluate.py --track-set
real refuses to run if any of them differs.

Note: the downloaded files are not pinned upstream. If the server's contents
ever change, verification fails here rather than results silently shifting.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_heldout_tracks as mh  # noqa: E402

# The f1tenth_racetracks circuits, as listed in the feasibility spike report.
REAL_CIRCUITS = [
    "Austin", "BrandsHatch", "Budapest", "Catalunya", "Hockenheim", "IMS",
    "Melbourne", "MexicoCity", "Montreal", "Monza", "MoscowRaceway",
    "Nuerburgring", "Oschersleben", "Sakhir", "SaoPaulo", "Sepang", "Shanghai",
    "Silverstone", "Sochi", "Spa", "Spielberg", "YasMarina", "Zandvoort",
]
SOURCE_URL = "http://api.f1tenth.org/<name>.tar.xz"


def real_files(maps: pathlib.Path, name: str) -> list[str]:
    """The files Track.from_track_name reads for this circuit."""
    folder = maps / name
    text = (folder / f"{name}_map.yaml").read_text()
    image = re.search(r"^image:\s*(\S+)", text, re.M).group(1)
    files = [f"{name}_map.yaml", image, f"{name}_centerline.csv"]
    if (folder / f"{name}_raceline.csv").exists():
        files.append(f"{name}_raceline.csv")
    return files


def install(maps: pathlib.Path) -> list[str]:
    from f1tenth_gym.envs.track.utils import find_track_dir

    failed = []
    for name in REAL_CIRCUITS:
        try:
            find_track_dir(name)  # downloads and extracts if missing
        except Exception as ex:  # report, never drop silently
            failed.append(f"{name}: download failed ({ex})")
            continue
        folder = maps / name
        for src, dst in ((f"{name}.yaml", f"{name}_map.yaml"),
                         (f"{name}.png", f"{name}_map.png")):
            if (folder / src).exists() and not (folder / dst).exists():
                shutil.copyfile(folder / src, folder / dst)
                print(f"  {name}: {src} -> {dst}")
        if not (folder / f"{name}_map.yaml").exists():
            failed.append(f"{name}: no {name}_map.yaml after install")
    return failed


def record(maps: pathlib.Path) -> None:
    manifest = json.loads(mh.MANIFEST.read_text())
    for name in REAL_CIRCUITS:
        manifest["tracks"][name] = {
            "split": "real",
            "files": mh.checksums(maps, name, real_files(maps, name)),
        }
    manifest["splits"]["real"] = {
        "names": list(REAL_CIRCUITS),
        "source": SOURCE_URL + " (f1tenth_racetracks)",
        "install_with": "python tracks/install_real_tracks.py",
    }
    mh.write_json(mh.MANIFEST, manifest)
    print(f"manifest updated: {len(REAL_CIRCUITS)} real circuits recorded in {mh.MANIFEST}")


def verify(maps: pathlib.Path) -> list[str]:
    manifest = json.loads(mh.MANIFEST.read_text())
    if "real" not in manifest["splits"]:
        return ["manifest has no real split; run with --manifest"]
    bad = []
    for name in manifest["splits"]["real"]["names"]:
        files = manifest["tracks"][name]["files"]
        try:
            ok = mh.same(files, mh.checksums(maps, name, list(files)))
        except FileNotFoundError:
            ok = False
        if not ok:
            bad.append(f"{name}: differs from manifest")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", action="store_true",
                    help="Record the installed circuits' checksums in tracks/manifest.json.")
    args = ap.parse_args()
    maps = mh.maps_dir()
    failed = install(maps)
    if failed:
        print("COULD NOT INSTALL:")
        for f in failed:
            print("  " + f)
        return 1
    print(f"installed: {len(REAL_CIRCUITS)}/{len(REAL_CIRCUITS)} real circuits in {maps}")
    if args.manifest:
        record(maps)
    bad = verify(maps)
    if bad:
        print("VERIFY FAILED:")
        for b in bad:
            print("  " + b)
        return 1
    print(f"verified: {len(REAL_CIRCUITS)}/{len(REAL_CIRCUITS)} match tracks/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
