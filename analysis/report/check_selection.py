"""Recompute every frozen checkpoint selection from the committed vw_val CSVs and compare.

    python analysis/report/check_selection.py

Pre-registered rule (final_grid_v2/DECISION_v2.md, "Checkpoint selection"): per run, the
best of the 400k / 800k / 1.2M / 1.6M / final checkpoints by vw_val mean fractional laps;
ties go to the higher lap completion %, then the earlier checkpoint.

Candidate CSVs:
  grid A  results/final_grid_<trained_by>/<tag>_<checkpoint>_vw_val.csv, with trained_by
          from results/final_test/selection.csv
  grid B  the paths listed in results/final_test_v2/candidate_sources.csv
Each recomputed choice (checkpoint and score) is compared with the frozen selection file.
Exits 1 if anything differs or a candidate CSV is missing.
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
LABELS = ["400k", "800k", "1200k", "1600k", "final"]
SELECTION = {"A": REPO / "results" / "final_test" / "selection.csv",
             "B": REPO / "results" / "final_test_v2" / "selection.csv"}


def candidates_grid_a(sel: pd.DataFrame) -> dict:
    return {(r.tag, lab): REPO / "results" / f"final_grid_{r.trained_by}" / f"{r.tag}_{lab}_vw_val.csv"
            for _, r in sel.iterrows() for lab in LABELS}


def candidates_grid_b() -> dict:
    src = pd.read_csv(REPO / "results" / "final_test_v2" / "candidate_sources.csv")
    return {(r.tag, r.checkpoint): REPO / r.csv for _, r in src.iterrows()}


def choose(scores: list[tuple[str, float, float]]) -> tuple[str, float]:
    """scores: (label, mean laps, lap %) in checkpoint order."""
    order = {lab: i for i, lab in enumerate(LABELS)}
    best = sorted(scores, key=lambda s: (-round(s[1], 9), -round(s[2], 9), order[s[0]]))[0]
    return best[0], best[1]


def main() -> int:
    ok_all = True
    for grid, path in SELECTION.items():
        sel = pd.read_csv(path)
        cands = candidates_grid_a(sel) if grid == "A" else candidates_grid_b()
        matches, problems = 0, []
        for _, r in sel.iterrows():
            scores = []
            for lab in LABELS:
                f = cands.get((r.tag, lab))
                if f is None or not f.exists():
                    problems.append(f"{r.tag}: missing candidate CSV for {lab} ({f})")
                    break
                d = pd.read_csv(f)
                scores.append((lab, float(d.laps.mean()), 100 * float(d.completed_lap.mean())))
            else:
                lab, laps = choose(scores)
                if lab == r.chosen_checkpoint and round(laps, 4) == round(float(r.vw_val_laps), 4):
                    matches += 1
                else:
                    problems.append(f"{r.tag}: recomputed {lab} ({laps:.4f}) vs frozen "
                                    f"{r.chosen_checkpoint} ({r.vw_val_laps:.4f})")
        print(f"grid {grid}: {matches}/{len(sel)} selections reproduced ({path.relative_to(REPO).as_posix()})")
        for p in problems:
            print(f"   MISMATCH {p}")
        ok_all &= not problems
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
