"""Minimal fix for v1.0.0 vs api.f1tenth.org filename drift:
the loader wants <stem>_map.yaml, the tarball ships <stem>.yaml.
Idempotent: creates the *_map.yaml alias for every downloaded track dir."""
import pathlib, shutil, sys

maps = pathlib.Path(__file__).parent / "f1tenth_gym_v1" / "maps"
n = 0
for d in maps.iterdir():
    if not d.is_dir():
        continue
    want = d / f"{d.stem}_map.yaml"
    have = d / f"{d.stem}.yaml"
    if not want.exists() and have.exists():
        shutil.copyfile(have, want)
        n += 1
        print("aliased", have.name, "->", want.name)
print(f"done. {n} track(s) fixed.")
