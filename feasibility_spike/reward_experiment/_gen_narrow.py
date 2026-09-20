"""Generate NARROW synthetic tracks (WIDTH 2.2 m -> walls ~1.1 m, matching the
real circuits' spawn clearance) to test the input-shift hypothesis. ADDITIVE:
patches a temp copy of the team's random_trackgen.py; team files untouched.
"""
import pathlib, re, subprocess, sys, tempfile, types

if "gym" not in sys.modules:
    sys.modules["gym"] = types.ModuleType("gym"); sys.modules["gym"].__version__ = "0.0.0"
import f1tenth_gym.envs.track.utils as _u  # noqa: E402

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[1]                      # team_repo
INSTALLED_MAPS = pathlib.Path(_u.__file__).resolve().parents[3] / "maps"

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--width", type=float, default=4.0)
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--seed", type=int, default=200)
ap.add_argument("--name", default="synthetic_narrow")
args = ap.parse_args()
N, WIDTH, SEED = args.n, args.width, args.seed
NAME = args.name

src = (ROOT / "random_trackgen.py").read_text()
assert "WIDTH = 10.0" in src, "generator changed? width patch target not found"
patched = src.replace("WIDTH = 10.0", f"WIDTH = {WIDTH}")
gen = HERE / "_random_trackgen_narrow.py"
gen.write_text(patched)

with tempfile.TemporaryDirectory() as tmp:
    tmp = pathlib.Path(tmp)
    subprocess.run([sys.executable, str(gen), "--n-maps", str(N),
                    "--seed", str(SEED), "--outdir", str(tmp)], check=True)
    for i in range(N):
        name = f"{NAME}_{i}"
        folder = INSTALLED_MAPS / name
        folder.mkdir(parents=True, exist_ok=True)
        yaml_src = tmp / f"map{i}_map.yaml"
        pgm_src = tmp / f"map{i}_map.pgm"
        if not (yaml_src.exists() and pgm_src.exists()):
            continue
        content = yaml_src.read_text()
        content = re.sub(r"image:\s*.*", f"image: {name}_map.pgm", content)
        (folder / f"{name}_map.yaml").write_text(content)
        (folder / f"{name}_map.pgm").write_bytes(pgm_src.read_bytes())
        csv_src = tmp / f"map{i}_centerline.csv"
        if csv_src.exists():
            with open(csv_src) as inf, open(folder / f"{name}_centerline.csv", "w") as outf:
                for line in inf:
                    line = line.strip()
                    if not line: continue
                    outf.write((f"#x,y,w_left,w_right\n" if line.startswith("#") else f"{line}, 2.0, 2.0\n"))
        print(f"packaged {name}")
print("done")
