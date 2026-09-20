---
title: Handoff — file map, commands, and how to resume
type: reference
updated: 2025-09-20
sources:
  - team_repo/feasibility_spike/reward_experiment/RESUME.txt (high)
  - team_repo/feasibility_spike/reward_experiment/_insert_our_slides.py (high)
---

# Handoff

Everything a teammate (or a future session) needs to pick up the work.

## Git state

- **Branch:** `experiments/feasibility-spike`, ~30 commits ahead of `main`,
  all additive, pushed to origin. Author: `aolin yang`.
- **Latest deck commit:** timeline "four bugs" → "two bugs + reward/spawn
  redesign".
- `.gitignore` excludes `logs/` and `saved_models/` (churn) — models are
  regenerated from `RESUME.txt`, not committed.
- **Repo covers `team_repo/` only.** The decks and `_v2_build_src.pptx` live
  at the COMPSCI 760 root, **outside** git.

## Directory map

```
team_repo/
├── train.py, evaluate.py, sb3_wrapper.py, src/   # TEAM code — do not modify
├── feasibility_spike/
│   ├── wiki/                                      # this wiki
│   └── reward_experiment/
│       ├── learning_curve.py     # train PPO/SAC, --save-model, --keep-checkpoints,
│       │                         #   --old-reward, --shaping, resumable per-segment
│       ├── eval_vs_baseline.py   # unseen-track eval (one-lap, centreline, N seeds)
│       ├── baselines.py          # gap-follower (threshold 20) + random
│       ├── spawn_study.py        # spawn-offset measurement (cl vs rl grid)
│       ├── seed_probe.py         # verifies seed→spawn reproducibility
│       ├── reward_scan2.py       # V1-vs-V2 reward on the same rollout
│       ├── _gen_narrow.py        # narrow-track generator (patched temp copy)
│       ├── new_reward_wrapper.py, shaped_reward_wrapper.py, OldRewardWrapper
│       ├── _insert_our_slides.py # deck build (loads _v2_build_src.pptx, inserts 6 slides)
│       ├── _make_*.py, gen_gap_figure.py           # figure scripts
│       ├── results/*.json        # all measured results (see results.md)
│       ├── slides_assets/*.png    # deck figures
│       ├── saved_models/         # checkpoints (gitignored; regenerate via RESUME.txt)
│       ├── logs/                 # run logs (gitignored)
│       └── RESUME.txt            # resumable run commands
```

## Environment

- Venv python: `spike/venv/Scripts/python.exe` (absolute path; quote it).
- A gym shim is required for the SAC eval scripts (see script headers).
- Windows: quote paths (spaces), read files as UTF-8, write output to UTF-8
  files (console gbk codec fails on unicode). Write `.py` files rather than
  heredocs for anything with special chars.

## Resume training

See `RESUME.txt`. Each command auto-detects its checkpoint
(`saved_models/<name>/state.json`) and continues from the last segment.

- **SAC-2M** was paused at 12/20 (1.2M). Rerun command #3 to finish
  (~4 h). Then run its unseen-track eval (command in RESUME.txt).
- **Overfitting probe:** SAC-1M with `--keep-checkpoints` (see RESUME.txt
  "OVERFITTING PROBE"); then eval each `ckpt_*.zip` on `synthetic_nar7_0` +
  Spielberg.

## Rebuild the deck

```bash
cd feasibility_spike/reward_experiment
"$PY" _make_figs.py && "$PY" gen_gap_figure.py   # refresh figures if data changed
"$PY" _insert_our_slides.py                       # -> Group4_ProjectUpdate_FINAL2.pptx
```

- The build loads `_v2_build_src.pptx` (a copy of the team's v2, since the
  original locks in PowerPoint), inserts 6 of our slides, and never touches
  the original slide shapes. `fix_team_text()` applies the one authorized
  wording fix (timeline "four bugs").
- **Close PowerPoint before rebuilding** — an open file causes a
  PermissionError; the script then can't save.

## Deck slide order (FINAL2, 18 slides)

1 Title · 2 Motivation · 3–5 Lit survey · 6 RQ · 7 Datasets · **8 Evaluation
protocol (ours)** · 9 Two bugs (team) · **10 The reward: design & why it
changed (ours)** · 11 Baselines (team) · **12 PPO vs SAC (ours)** · **13 Why
PPO falls short (ours)** · **14 Zero-shot results: RL vs baselines (ours)** ·
15 Timeline (team) · **16 Conclusions & next steps (ours)** · 17 Roles/Q&A
(team) · 18 Sources. Speaker notes carry a "what we say / what we did /
rubric" pack.

## Pre-submission checklist

- Export to PDF as `XX_MethodResults.pdf` (XX = group number); leader uploads.
- Check Montserrat/Inter fonts render on the exporting machine (else Calibri
  fallback shifts spacing).
- Keep only `v2` + `FINAL2`; delete stale deck versions.
- Names: title slide now uses Victory / Grant to match the roles slide
  (Victory = Yi Wei, Grant = Zihang Zhang; team-confirmed). Resolved.
- Rehearse to 8 min (18 slides, 5 speakers).

## See also

- `results.md` — what each JSON contains
- `open-questions.md` — what to run next
- `methodology.md` — the protocol the tools implement
