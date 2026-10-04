"""Generate the held-out synthetic validation and test tracks, reproducibly.

    python tracks/make_heldout_tracks.py            # build val/test, write manifest
    python tracks/make_heldout_tracks.py --verify   # check files on disk, no writes

WHAT THIS PRODUCES

    train  synthetic_track_0  .. synthetic_track_99   (already exist; never touched)
    val    val_track_0        .. val_track_9          (raw generator indices 100-109)
    test   test_track_0       .. test_track_19        (raw generator indices 110-129)

HOW THE SETS ARE DEFINED

The training tracks were produced by generate_track_pool.py, which runs
random_trackgen.py with no --seed argument, so its default seed of 123. That
generator draws every track from one sequential numpy RNG stream, so track i
depends on all draws for tracks 0..i-1 (including retried attempts). The only
way to reproduce track i is to regenerate 0..i in order.

Regenerating with seed 123 reproduces the existing synthetic_track_0..99
byte-for-byte (yaml, pgm and centreline CSV, 100/100), which proves both that
the generator is deterministic and that its settings are exactly those that
produced the training set. The held-out tracks are therefore the NEXT 30 draws
of that same stream: same generator, same settings, same distribution, and
disjoint from training by construction.

They get distinct name prefixes (val_track_*, test_track_*) because train.py
builds its pool as synthetic_track_{i} for i in range(diversity). A held-out
track can never enter a training pool by any value of --diversity.

SAFETY

This script never writes to a synthetic_track_* folder. It refuses to run if
the training tracks on disk do not reproduce, because that would mean this
machine's training set differs from the canonical one; it refuses to overwrite
an existing val/test track or manifest whose contents differ.

NOTE ON CHECKSUMS

The pgm images are binary and compared exactly. The yaml and csv files are
text written in platform text mode, so their bytes differ between Windows
(CRLF) and Linux (LF) even when the content is identical. The manifest records
a line-ending-normalised sha256 for every file and verification compares that,
so a set regenerated on another OS still verifies. The raw sha256 is recorded
alongside for completeness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
GENERATOR = REPO / "random_trackgen.py"
MANIFEST = pathlib.Path(__file__).resolve().parent / "manifest.json"

SEED = 123                      # generate_track_pool.py's effective seed
N_TRAIN, N_VAL, N_TEST = 100, 10, 20
N_TOTAL = N_TRAIN + N_VAL + N_TEST

# The generator settings that define the track distribution. Read from the
# generator source and asserted, so a silently edited generator cannot produce
# a set that claims to match.
EXPECTED_SETTINGS = {
    "CHECKPOINTS": 16,
    "SCALE": 6.0,
    "TRACK_RAD": "900 / SCALE",
    "TRACK_DETAIL_STEP": "21 / SCALE",
    "TRACK_TURN_RATE": 0.31,
    "WIDTH": 10.0,
}
EXPECTED_RENDER = {
    "figsize_in": [20, 20],
    "xlim": [-180, 300],
    "ylim": [-300, 300],
    "dpi": 80,
    "resolution_m_per_px": 0.0625,
}


def split_of(raw_index: int) -> tuple[str, str]:
    """Map a raw generator index to (split, track name)."""
    if raw_index < N_TRAIN:
        return "train", f"synthetic_track_{raw_index}"
    if raw_index < N_TRAIN + N_VAL:
        return "val", f"val_track_{raw_index - N_TRAIN}"
    return "test", f"test_track_{raw_index - N_TRAIN - N_VAL}"


def maps_dir() -> pathlib.Path:
    """The directory f1tenth_gym's find_track_dir actually searches."""
    import types
    if "gym" not in sys.modules:
        sys.modules["gym"] = types.ModuleType("gym")
    sys.modules["gym"].__version__ = "0.0.0"
    import f1tenth_gym.envs.track.utils as u
    return pathlib.Path(u.__file__).resolve().parents[3] / "maps"


def sha_raw(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha_norm(p: pathlib.Path) -> str:
    """sha256 with CRLF normalised to LF for text files; exact for binary."""
    data = p.read_bytes()
    if p.suffix in (".yaml", ".csv"):
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def track_files(name: str) -> list[str]:
    return [f"{name}_map.yaml", f"{name}_map.pgm", f"{name}_centerline.csv"]


def read_settings() -> dict:
    """Parse the distribution-defining constants out of the generator source."""
    src = GENERATOR.read_text()
    body = src[src.index("def create_track"):src.index("def convert_track")]
    got = {}
    for key in EXPECTED_SETTINGS:
        m = re.search(rf"^\s*{key}\s*=\s*(.+?)\s*$", body, re.M)
        if not m:
            raise SystemExit(f"Could not find {key} in {GENERATOR}")
        raw = m.group(1)
        try:
            got[key] = float(raw) if "." in raw else int(raw)
        except ValueError:
            got[key] = raw
    render = {
        "figsize_in": [int(x) for x in re.search(r"set_size_inches\((\d+),\s*(\d+)\)", src).groups()],
        "xlim": [int(x) for x in re.search(r"set_xlim\((-?\d+),\s*(-?\d+)\)", src).groups()],
        "ylim": [int(x) for x in re.search(r"set_ylim\((-?\d+),\s*(-?\d+)\)", src).groups()],
        "dpi": int(re.search(r"savefig\([^)]*dpi=(\d+)", src).group(1)),
        "resolution_m_per_px": float(re.search(r'resolution:\s*([0-9.]+)', src).group(1)),
    }
    for k, v in EXPECTED_SETTINGS.items():
        if got[k] != v:
            raise SystemExit(f"Generator setting {k} is {got[k]!r}, expected {v!r}. "
                             "The generator has changed since the training set was "
                             "made; refusing to produce a set that claims to match.")
    for k, v in EXPECTED_RENDER.items():
        if render[k] != v:
            raise SystemExit(f"Generator render setting {k} is {render[k]!r}, expected {v!r}.")
    return {"create_track": got, "render": render}


def package(raw_dir: pathlib.Path, i: int, dest: pathlib.Path, name: str) -> None:
    """Byte-for-byte replica of generate_track_pool.py's packaging step.

    Text files are written in platform text mode on purpose: that is what
    generate_track_pool.py does, and reproducing the training files exactly
    depends on it.
    """
    src_yaml = raw_dir / f"map{i}_map.yaml"
    src_pgm = raw_dir / f"map{i}_map.pgm"
    src_csv = raw_dir / f"map{i}_centerline.csv"
    if not (src_yaml.exists() and src_pgm.exists() and src_csv.exists()):
        raise SystemExit(f"Generator did not produce raw track {i}")
    folder = dest / name
    folder.mkdir(parents=True, exist_ok=True)
    with open(src_yaml, "r") as f:
        content = f.read()
    content = re.sub(r"image:\s*.*", f"image: {name}_map.pgm", content)
    with open(folder / f"{name}_map.yaml", "w") as f:
        f.write(content)
    shutil.copy(src_pgm, folder / f"{name}_map.pgm")
    with open(src_csv, "r") as inf, open(folder / f"{name}_centerline.csv", "w") as outf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                outf.write("#x,y,w_left,w_right\n")
            else:
                outf.write(f"{line}, 2.0, 2.0\n")


def generate(tmp: pathlib.Path) -> pathlib.Path:
    """Run the generator for all 130 tracks and package them under final names."""
    raw = tmp / "raw"
    pkg = tmp / "pkg"
    print(f"Running {GENERATOR.name} --n-maps {N_TOTAL} --seed {SEED} ...")
    subprocess.run(
        [sys.executable, str(GENERATOR), "--n-maps", str(N_TOTAL),
         "--seed", str(SEED), "--outdir", str(raw)],
        check=True, stdout=subprocess.DEVNULL,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    for i in range(N_TOTAL):
        _, name = split_of(i)
        package(raw, i, pkg, name)
    return pkg


def checksums(folder_root: pathlib.Path, name: str) -> dict:
    out = {}
    for fn in track_files(name):
        p = folder_root / name / fn
        out[fn] = {"sha256": sha_raw(p), "sha256_lf": sha_norm(p)}
    return out


def same(a: dict, b: dict) -> bool:
    return all(a[fn]["sha256_lf"] == b[fn]["sha256_lf"] for fn in a)


def library_versions() -> dict:
    import cv2, matplotlib, numpy, shapely
    return {"python": sys.version.split()[0], "numpy": numpy.__version__,
            "matplotlib": matplotlib.__version__, "opencv": cv2.__version__,
            "shapely": shapely.__version__}


def git_commit_of(path: pathlib.Path) -> str | None:
    try:
        return subprocess.run(["git", "log", "-1", "--format=%h", "--", str(path)],
                              cwd=REPO, capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def build_manifest(settings: dict, sums: dict) -> dict:
    tracks = {}
    splits = {"train": [], "val": [], "test": []}
    for i in range(N_TOTAL):
        split, name = split_of(i)
        splits[split].append(name)
        tracks[name] = {"split": split, "raw_index": i, "files": sums[name]}
    return {
        "schema": 1,
        "description": ("Synthetic track splits for the F1TENTH PPO-vs-SAC study. "
                        "train is the existing training pool; val (checkpoint "
                        "selection) and test (reporting) are the next 30 draws of "
                        "the same seeded generator stream, disjoint from training."),
        "generator": {
            "script": "random_trackgen.py",
            "script_sha256": sha_raw(GENERATOR),
            "script_last_commit": git_commit_of(GENERATOR),
            "seed": SEED,
            "n_generated": N_TOTAL,
            "settings": settings,
            "packaging": ("generate_track_pool.py layout: <name>/<name>_map.yaml, "
                          "<name>_map.pgm, <name>_centerline.csv with columns "
                          "x,y,w_left,w_right (widths fixed at 2.0)"),
            "library_versions": library_versions(),
            "regenerate_with": "python tracks/make_heldout_tracks.py",
        },
        "splits": {
            "train": {"raw_indices": [0, N_TRAIN - 1], "names": splits["train"]},
            "val": {"raw_indices": [N_TRAIN, N_TRAIN + N_VAL - 1], "names": splits["val"]},
            "test": {"raw_indices": [N_TRAIN + N_VAL, N_TOTAL - 1], "names": splits["test"]},
        },
        "tracks": tracks,
    }


def write_json(path: pathlib.Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def cmd_generate(args) -> int:
    maps = maps_dir()
    settings = read_settings()
    print(f"maps dir: {maps}")
    print("generator settings match the training set's: "
          + ", ".join(f"{k}={v}" for k, v in settings["create_track"].items()))

    with tempfile.TemporaryDirectory() as t:
        pkg = generate(pathlib.Path(t))
        sums = {split_of(i)[1]: checksums(pkg, split_of(i)[1]) for i in range(N_TOTAL)}

        # 1. The training set on disk must be exactly the canonical one.
        bad = []
        for i in range(N_TRAIN):
            name = split_of(i)[1]
            if not (maps / name).exists() or not same(sums[name], checksums(maps, name)):
                bad.append(name)
        print(f"training tracks reproduced exactly: {N_TRAIN - len(bad)}/{N_TRAIN}")
        if bad:
            raise SystemExit(
                f"{len(bad)} training tracks on disk do not match the seed-{SEED} "
                f"generator, e.g. {bad[:5]}. This machine's training set is not the "
                "canonical one. Nothing was written. Do not regenerate them here "
                "without agreeing it with the team: models trained on them depend "
                "on these exact files.")

        # 2. Install val/test, never overwriting a differing copy.
        installed = 0
        for i in range(N_TRAIN, N_TOTAL):
            name = split_of(i)[1]
            dest = maps / name
            if dest.exists():
                if not same(sums[name], checksums(maps, name)):
                    if not args.force:
                        raise SystemExit(f"{dest} exists with different contents. "
                                         "Refusing to overwrite; pass --force.")
                    shutil.rmtree(dest)
                else:
                    continue
            shutil.copytree(pkg / name, dest)
            installed += 1
        print(f"held-out tracks installed: {installed} new, "
              f"{N_VAL + N_TEST - installed} already present and identical")

    manifest = build_manifest(settings, sums)
    if MANIFEST.exists() and not args.force:
        old = json.loads(MANIFEST.read_text())
        if all(same(old["tracks"][n]["files"], manifest["tracks"][n]["files"])
               for n in manifest["tracks"]):
            print(f"manifest unchanged: {MANIFEST}")
            return 0
        raise SystemExit(f"{MANIFEST} exists and differs. Refusing to overwrite; pass --force.")
    write_json(MANIFEST, manifest)
    print(f"manifest written: {MANIFEST}")
    return 0


def cmd_verify(args) -> int:
    """Check every track on disk against the manifest. Writes nothing."""
    if not MANIFEST.exists():
        raise SystemExit(f"No manifest at {MANIFEST}; run without --verify first.")
    manifest = json.loads(MANIFEST.read_text())
    maps = maps_dir()
    counts = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}
    failures = []
    for name, entry in manifest["tracks"].items():
        split = entry["split"]
        counts[split][1] += 1
        if not (maps / name).exists():
            failures.append(f"{name}: missing")
            continue
        if same(entry["files"], checksums(maps, name)):
            counts[split][0] += 1
        else:
            failures.append(f"{name}: contents differ from manifest")
    for split, (ok, total) in counts.items():
        print(f"  {split:<5} {ok}/{total} match the manifest")
    if failures:
        print("FAILURES:")
        for f in failures[:20]:
            print("  " + f)
        return 1
    print("All tracks match the manifest.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verify", action="store_true",
                    help="Check files on disk against the manifest; write nothing.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite differing val/test tracks or manifest. Never "
                         "touches training tracks.")
    args = ap.parse_args()
    return cmd_verify(args) if args.verify else cmd_generate(args)


if __name__ == "__main__":
    sys.exit(main())
