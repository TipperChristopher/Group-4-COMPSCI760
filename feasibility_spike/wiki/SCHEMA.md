# SCHEMA — Group-4 F1TENTH RL project wiki

Conventions for this wiki. Read first.

## Purpose

Durable, hand-off-ready record of the Group-4 COMPSCI 760 project: the
research question, what we built, what we measured, the decisions we made
and why, the problems we hit, and what is still open. Written so a teammate
(or a future session) can pick up the thread without re-deriving anything.

## Scope

This wiki documents **the whole project as we understand it**, with emphasis
on the work done in the `feasibility_spike/` area (evaluation methodology,
reward analysis, spawn fix, PPO-vs-SAC pilots, generalization study). It is
additive: it never restates or replaces teammates' own slides or code, it
points to them.

## Layout

```
wiki/
├── SCHEMA.md         # this file — conventions
├── index.md          # canonical browse list of every page
├── log.md            # append-only operation log (what knowledge changed, when)
├── raw-sources/
│   └── index.md      # registry of every source (in-repo path or URL), by bucket
├── overview.md       # project goal, RQ, proposal → now (concept)
├── methodology.md    # evaluation protocol: splits, metric, spawn, budget (reference)
├── reward.md         # reward history V1/interim/V2 + incentive-flaw story (concept)
├── experiments.md    # PPO vs SAC, learning curves, unseen-track eval, nar7, geometry (concept)
├── results.md        # the numbers, in tables, each cited to a JSON file (reference)
├── decisions.md      # key decisions: choice, alternative, rationale (decision)
├── incidents.md      # the two real bugs + the loader pitfall (bug)
├── open-questions.md  # what is not yet answered / not yet run (open-question)
├── handoff.md        # how to resume: file map, commands, RESUME.txt (reference)
└── glossary.md       # domain terms (reference)
```

## Frontmatter (mandatory on every page)

`title`, `type`, `updated`, `sources`. Types used here: `concept`,
`decision`, `bug`, `open-question`, `reference`, `synthesis`, `stub`.

## Rules

- **Data-backed only.** Every quantitative claim cites the file it came
  from — a `results/*.json`, a writeup `.md`, or a source-code `path:line`.
  No number appears without a source. If a value is a point estimate from
  one seed, say so.
- **No hallucination.** If something was not measured, it lives in
  `open-questions.md`, not stated as fact.
- **Not stale.** Each page carries `updated:`. When a result changes, bump
  the date and log it in `log.md`. The single-seed / pilot caveats are part
  of the record, not a footnote.
- **Reference, don't copy.** Source code, JSON results, and the deck are
  referenced by path. Only genuinely ad-hoc external material would be
  copied into `raw-sources/<bucket>/` (none so far).
- **Reliability tiers** on sources: `high` (source code, deterministic
  JSON results), `mixed` (our narrative writeups), `unverified` (external,
  until cross-referenced).

## Paths (this machine)

- Team repo: `team_repo/` (git; branch `experiments/feasibility-spike`).
- Our work: `team_repo/feasibility_spike/reward_experiment/`.
- Results: `.../reward_experiment/results/*.json`.
- Venv python: `spike/venv/Scripts/python.exe`.
- Decks: COMPSCI 760 root (outside git).
