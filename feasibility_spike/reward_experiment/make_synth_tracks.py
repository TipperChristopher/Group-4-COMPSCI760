"""Generate a small synthetic track pool into the INSTALLED f1tenth_gym maps dir.

Self-contained helper for the reward ablation. Reuses the repo's own
random_trackgen.py (the generator) and the same packaging format as
generate_track_pool.py (4-column centreline, <name>_map.yaml/.pgm), but writes
into wherever f1tenth_gym is actually installed, so it works with an editable
install or a site-packages install without assuming a local ./f1tenth_gym dir.

    python make_synth_tracks.py --n 3 --seed 0
"""
import argparse
import os
import pathlib
import re
import subprocess
import sys
import tempfile

import f1tenth_gym.envs.track.utils as _u

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
INSTALLED_MAPS = pathlib.Path(_u.__file__).resolve().parents[3] / "maps"


def package(src_dir, i, dest_dir):
    name = f"synthetic_track_{i}"
    folder = dest_dir / name
    folder.mkdir(parents=True, exist_ok=True)

    yaml_src = src_dir / f"map{i}_map.yaml"
    pgm_src = src_dir / f"map{i}_map.pgm"
    if not yaml_src.exists() or not pgm_src.exists():
        return False

    content = yaml_src.read_text()
    content = re.sub(r"image:\s*.*", f"image: {name}_map.pgm", content)
    (folder / f"{name}_map.yaml").write_text(content)
    (folder / f"{name}_map.pgm").write_bytes(pgm_src.read_bytes())

    csv_src = src_dir / f"map{i}_centerline.csv"
    if csv_src.exists():
        with open(csv_src) as inf, open(folder / f"{name}_centerline.csv", "w") as outf:
            for line in inf:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    outf.write("#x,y,w_left,w_right\n")
                else:
                    outf.write(f"{line}, 2.0, 2.0\n")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Installed f1tenth_gym maps dir: {INSTALLED_MAPS}")
    INSTALLED_MAPS.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        cmd = [sys.executable, str(REPO_ROOT / "random_trackgen.py"),
               "--n-maps", str(args.n), "--seed", str(args.seed), "--outdir", str(tmp)]
        print("Generating raw tracks:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        made = 0
        for i in range(args.n):
            if package(tmp, i, INSTALLED_MAPS):
                made += 1
                print(f"  packaged synthetic_track_{i}")
    print(f"Done. {made} synthetic tracks written to {INSTALLED_MAPS}.")


if __name__ == "__main__":
    main()
