---
title: Final presentation — deck versions, fixes, talk script and Q&A
type: reference
updated: 2026-10-23
sources:
  - COMPSCI 760 root (outside git): Group 4 Final Results.pptx, Group 4 Final Results v2.pptx, Group 4 Final Results (1).pptx, Group 4 Final Results (1) - checked.pptx/.pdf, Group 4 Final Talk Script.pdf (team + ours)
  - _verify/final_pptx_review/ (v2 build: build_deck.py, make_figs.py, check_claims.py, insights.py) (high)
  - _verify/final_check/ (apply_fixes.py, renders, PDF export) and _verify/script_check/ (timing, aolin_*.txt) (high)
  - final-results.md (the verified numbers every slide was checked against)
  - run journal Attempts 34, 36–39 (mixed)
---

# Final presentation

How the verified results (`final-results.md`) became the final talk: which deck versions exist, what was wrong
in each and how it was fixed, the talk-script review, and the wording the group settled on. The talk was due
shortly after 2026-10-11; **the delivery date, which deck file was projected, and the marks are not recorded in
any session** — confirm with the team before citing them.

## Deck lineage (all in the COMPSCI 760 root, outside git)

| file | when | by | what it is |
|---|---|---|---|
| `Group 4 Final Results.pptx` | 10-10 20:55 | team | first draft, 17 slides; reviewed (journal Attempts 34, 36) |
| `Group 4 Final Results v2.pptx` | 10-11 11:27 | Aolin, via `_verify/final_pptx_review/build_deck.py` | 18 slides: corrections below + new RQ3/RQ4 slide + new sim-to-real slide |
| `Group 4 Final Results (1).pptx` | 10-11 12:31 | team (merged v2 with Grant's and Victory's ideas and figures) | 18 slides; the version the team had |
| `Group 4 Final Results (1) - checked.pptx` / `.pdf` | 10-11 12:39 | Aolin, via `_verify/final_check/apply_fixes.py` | (1) plus three fixes (slides 4, 13, 14 and two notes), sent back to the team |

## First draft → v2: what was wrong and what v2 changed (2026-10-10/11)

| slide (draft) | problem | v2 change |
|---|---|---|
| 3 | "HYPOTHESIS" label overlapped its text | layout fixed |
| 5 | mixed two agents (92% vs 58%) | one agent, 92% → 0%; narrowA = the test tracks (20/20 identical centrelines) |
| 7 | "15% best case" untraceable; really an average | "on average 14% (24% at 100 tracks)", "on one training track" |
| 10 | best seed called "typical" | best seed = validation-best; 3-seed average 47% shown |
| 11–12 | "PPO gains more in grid B" rested on seed 2; "real circuits are narrow" contradicted by the data | new RQ3/RQ4 slide with every seed; new sim-to-real slide (length, not width) |
| 14–15 | conclusions/limitations overclaimed | RQ3 inconclusive; RQ2 seed-consistent + memorisation evidence; bonus finding exact; seeds, budget and hybrid limitations |

Speaker notes were rebalanced to 675 s (Chris 120, Victory 125, Aolin 150, Grant 150, Desmond 130); the
team then rewrote the timings in their talk script (below).

## Final deck (the team's merged version): checks and fixes (2026-10-11)

Rendered and exported through PowerPoint from a copy; all 18 slides export cleanly with no text overflow.

- **Verified against the raw data:** slide 5 bars (`0_pilot/grid_s0_r40{,_vw}/SAC_d100_*.csv`: fixed-width test 92%,
  narrowA 0%, real 0%; varied-width 76% / 68% / 37%; narrowB 2%); slide 7 heatmap cells 39.6 / 62.9 / 44.9 /
  217.7 m and 0 / 0 / 0 / 65% laps (penalty 5, seed 0, 0.8–1.0 M window); the 14.5% / 24% figures; slide 8 runtimes;
  slide 12 Holm 0.036; slides 10/11/15 72% / 47% / 85% consistent; references (Kunda DOI 10.1038/s41598-025-27702-6,
  Sezer & Gokasan DOI 10.1016/j.robot.2012.05.021, both via Crossref).
- **Kunda et al. claim** ("trained on one track; only tuned PPO was tested on unseen tracks (21)"): the abstract and
  Table 7 confirm the best tuned PPO model was evaluated on 21 unseen tracks; training and reporting use the
  re:Invent 2018 track. The paper never says "only one track" literally, so "trained on the re:Invent 2018 track"
  is the safest wording in writing.
- **Fixes applied in the `- checked` copy:**
  1. Slide 4: replay buffer "~8.9 GB → ~0.9 GB" → **"~8.7 GB → ~0.9 GB"** (SB3 buffer, see `final-results.md`).
  2. Slide 13: half-width table 1.06 / 1.11 m → **1.02 / 1.07 m** (same method as slide 5), the chart with the
     "0.0" / "Unseen synthetic" label overlap fixed, and the notes updated.
  3. Slide 14: RQ1 "then it beats SAC on average" → **"then it beats SAC under that reward"** (grid A SAC 0.68 beats
     grid B PPO 0.33 on real circuits), with "SAC works at defaults" so it still fits.
  4. Slide 12 notes: name the Holm family (two grid-B tests; 0.054 if seven).
- **Left to the team / Aolin:** slide 17 contribution row (suggested: "Reward design; PPO failure diagnosis and fix;
  grid B pre-registration and setup checks; independent verification of all results; seed and sim-to-real
  analysis"). Optional, not applied: RQ2 "1-track agents memorise their own track" → "1-track SAC" (the evidence is
  SAC; grid A PPO never learned its own track either).
- **Rendering caveat:** the deck's fonts (Space Grotesk titles, IBM Plex Sans body) are not installed on Aolin's
  machine, so titles render with uneven letter spacing there; export from a machine with the fonts.

## Final deck structure (18 slides, `(1)` version)

| # | slide | speaker in the talk script |
|---|---|---|
| 1–3 | title; motivation (Kunda, Li gap); RQ3 headline + RQ1/RQ2/RQ4 + hypothesis "SAC benefits more" | Chris |
| 4–6 | dataset/environment; width problem (pilot); changes since the proposal + P2 response | Victory |
| 7 | why default PPO failed (heatmap + animation) | Aolin |
| 8–9 | experimental design (48 runs, validation-only decisions); training curves | Grant |
| 10–12 | demo (Oschersleben, 1 vs 100 tracks); main results; RQ3/RQ4 per-seed gains | Desmond |
| 13–14 | sim-to-real (longer, not narrower); conclusions | Aolin |
| 15–16 | limitations and future work; reflection | Grant |
| 17–18 | team, code and GenAI use; references | Chris |

P2 feedback recorded on slide 6: 18.75 / 20 (objectives 1.75/2, changes 2.5/3, results 2.75/3, Q&A 1.75/2).

**Team contributions as stated on slide 17** (Aolin's row is the team's wording):

| member | contribution |
|---|---|
| Christopher Tipper (lead) | literature synthesis, report assembly; ran the 24 PPO runs (grids A and B) |
| Desmond Li | training and evaluation pipeline, track generation and width fix, tuning and job queue, final test and analysis; SAC runs |
| Aolin Yang | reward design, PPO failure diagnosis and fix, independent verification of results |
| Zihang (Grant) Zhang | statistical analysis, figure generation; SAC runs in both grids |
| Yi (Victory) Wei | method refinements, feasibility metrics, slide drafting |

**References on slide 18:** Cobbe et al. 2020 (ICML, procgen); Haarnoja et al. 2018 (SAC, ICML); Kirk et al. 2023
(JAIR 76, zero-shot generalisation survey); Kunda et al. 2025 (Sci. Rep. 15, 43940); Li et al. 2022 (MetaDrive,
IEEE TPAMI); O'Kelly et al. 2020 (F1TENTH, PMLR 123); Raffin et al. 2021 (SB3, JMLR 22(268)); Schulman et al. 2017
(PPO, arXiv:1707.06347); Sezer & Gokasan 2012 (Follow the Gap, RAS 60(9) 1123–1134; implementation adapted from
BDEvan5/f1tenth_benchmarks); F1TENTH racetracks repository (github.com/f1tenth/f1tenth_racetracks).

## Talk script review (`Group 4 Final Talk Script.pdf`, 2026-10-11)

- **Timing:** 1,916 spoken words = 12:47 at the script's own 150 words/min, against its stated 11:40 and a 12:00
  stop. Over budget: Chris 1:46 (1:40), Aolin 1:12 + 1:52 (0:55 + 1:35), Grant 1:46 + 1:15 (1:35 + 1:10), Chris 0:36 (0:25).
- **Errors found (raw-data checked):** "the drop lowers every score" (7/48 agents rise); RQ2 "on every seed" true only
  on synthetic tracks (two grid-B PPO seeds got worse on real); "they crash early because of the reward" shown only
  for SAC; "PPO never completed a single lap" true for the diagnosis run, not for grid A (15 training laps);
  "barely moving" (measured crawl 1.5–4.2 m/s); Q&A "gap-follower uses track geometry" (it uses the same 108 LiDAR
  beams); Q&A "committed to git before the single test run" false for grid A (slide 16 itself says grid A's
  selection was late); Q&A "our claims hold across both [grids]" overclaims.
- **What changed:** with too little time to edit others' parts, only Aolin's part was rewritten (it also removes the
  script's one internal contradiction, RQ2 vs slide 12). Final version: **`COMPSCI 760/Aolin - talk script and QA
  (final).md`** — 323 words ≈ 2:09 for slides 7, 13 and 14 (whole talk ≈ 11:50), a "watch my wording" list, the
  key numbers, and 39 prepared Q&A answers. The older `Aolin - final talk script and QA.md` was overwritten by an
  open editor tab and holds a previous draft; use the `(final)` file.
- **Heads-up sent for Desmond's Q&A:** answer "did you tune on the test set?" with the rule recomputation (the
  selection rule reproduces all 48 picks), not with "committed before the test".

## Wording the group settled on (carry into the report)

- "PPO leads **under grid B's reward**", never "PPO beats SAC".
- "On synthetic tracks, more tracks help every learning agent **on every seed**" — not on real circuits.
- RQ3: "**not supported**" (no consistent difference once both learn), not "refuted" and not "no difference".
- "Real circuits are **not narrower**; they are 2.4× longer" — not "width doesn't matter".
- "The drop **mostly keeps the ranking**" (Spearman 0.92), not "it lowers every score".
- Track-level p-values describe **these three agents per cell**, not the algorithms; show every seed.
- Do not claim longer training would beat the gap-follower.
- The gap-follower is a hand-written rule on the **same LiDAR**, capped at 3–5 m/s; it uses no map.

## Source reliability

- **Deck and script contents: high** (read from the files with python-pptx / pdftotext; renders checked by eye).
- **Who did what, and timing of edits: mixed** (session record and file modification times).
- **Delivery date, projected file and marks: not recorded.**

## See also

- `final-results.md` — every number these slides were checked against
- `report-handoff.md` — what carries over into the report
- `final-run-plan.md` — how grids A and B were set up
- `ppo-diagnosis.md` — the evidence behind slide 7
