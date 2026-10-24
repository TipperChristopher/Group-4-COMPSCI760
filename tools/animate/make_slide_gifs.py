"""Slide animations: replay frozen test checkpoints and render side-by-side GIFs + PNGs.

    .venv\\Scripts\\python.exe tools\\animate\\make_slide_gifs.py

Read-only on code, models and results; writes only results/animations/.

Replays reuse evaluate.py unchanged (imported, not copied): build_env, load_vecnormalize
and run_episode, with the final test's settings (deterministic, one lap, cl_grid_static,
15,000-step cap, eval seed 0 = test episode 0). A gym wrapper around the env only records
the car pose after every reset and step; it does not change anything the policy or the
simulator sees. Every replay must reproduce its test_raw row (laps, length, outcome) exactly,
or the script stops.

Track choice is a rule stated in advance and applied mechanically to the existing test_raw
CSVs: among the set's tracks where the right-hand ("finisher") agent completes the lap in
test episode 0, pick the track where the left-hand agent's episode-0 score is closest to its
overall mean on that set (all test episodes). If no track qualifies, the earlier rule is used
instead (minimise the sum over both agents of |episode-0 score - overall mean|), and the
README says so.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
TEST = REPO / "results" / "final_test"
SEL = {"A": REPO / "results" / "final_test" / "selection.csv",
       "B": REPO / "results" / "final_test_v2" / "selection.csv"}
SIM_HZ = 100
FPS = 20
HOLD_S = 1.0
SUMMARY_HOLD_S = 1.5              # the GIF opens on the final summary frame for this long
COLOURS = {"SAC": "#2a78d6", "PPO": "#eb6834"}
CRASH_RED = "#d62728"

LOOP_S = (7.0, 9.0)               # whole loop, summary frame and end hold included
# panels: (grid, tag, title); the right-hand panel is the "finisher" in the track rule.
# expect_median: these tags must be the median seed of their cell by test score on the set.
ANIMATIONS = [
    dict(name="anim1_diversity", title="1 track vs 100 tracks", test_set="real",
         panels=[("A", "final_SAC_d1_s1", "Trained on 1 track"),
                 ("A", "final_SAC_d100_s1", "Trained on 100 tracks")]),
    dict(name="anim2_ppo_fix", title="PPO before vs after the fix", test_set="vw_test",
         panels=[("A", "final_PPO_d100_s2", "Default PPO"),
                 ("B", "v2_PPO_d100_s1", "Fixed PPO (γ 0.999, 8192-step rollouts, entropy)")],
         expect_median=[("A", "PPO", 100), ("B", "PPO", 100)]),
]
CAPTIONS = [("final_SAC_d100_s1", "real"), ("v2_PPO_d100_s1", "vw_test"), ("final_PPO_d100_s2", "vw_test")]


def load_evaluate():
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("evaluate", REPO / "evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    sys.modules["evaluate"] = ev
    spec.loader.exec_module(ev)
    return ev


# ------------------------------------------------------------------ choices

def median_seed_tag(grid, algo, diversity, test_set):
    per_run = pd.read_csv(TEST / "test_per_run.csv")
    g = per_run[(per_run.grid == grid) & (per_run.algorithm == algo) &
                (per_run.diversity == diversity) & (per_run.test_set == test_set)].sort_values("laps")
    assert len(g) == 3, (grid, algo, diversity, test_set)
    return g.iloc[1].tag, [(int(r.seed), float(r.laps), r.tag) for _, r in g.iterrows()]


def raw(tag, test_set):
    return pd.read_csv(TEST / f"test_raw_{tag}_{test_set}.csv")


def choose_track(other, finisher, test_set):
    """Tracks where `finisher` completes the lap in episode 0; pick the one where `other`'s
    episode-0 score is closest to its overall mean on the set. Returns (track, table, fallback)."""
    a, b = raw(finisher, test_set), raw(other, test_set)
    a0, b0 = a[a.episode == 0].set_index("track"), b[b.episode == 0].set_index("track")
    mean_b = float(b.laps.mean())
    rows = [dict(track=t, finisher_lap_s=a0.length[t] / SIM_HZ, other_ep0=float(b0.laps[t]),
                 other_outcome=b0.outcome[t], other_len_s=b0.length[t] / SIM_HZ,
                 diff=abs(float(b0.laps[t]) - mean_b))
            for t in a0.index[a0.completed_lap == 1]]
    table = pd.DataFrame(rows).sort_values(["diff", "track"]) if rows else pd.DataFrame()
    if len(table):
        return table.iloc[0].track, table, None
    # Fallback (the earlier rule): minimise the sum over both agents of |ep0 - overall mean|.
    mean_a = float(a.laps.mean())
    cost = ((a0.laps - mean_a).abs() + (b0.laps - mean_b).abs()).sort_values()
    return cost.index[0], table, f"no track qualified; fallback rule picked {cost.index[0]} (cost {cost.iloc[0]:.4f})"


# ------------------------------------------------------------------- replay

def replay(ev, grid, tag, track):
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv

    sel = pd.read_csv(SEL[grid]).set_index("tag").loc[tag]
    poses = []                                   # one list per episode

    class PoseRecorder(gym.Wrapper):
        def _pose(self):
            u = self.env.unwrapped
            return float(u.poses_x[0]), float(u.poses_y[0]), float(u.poses_theta[0])

        def reset(self, **kw):
            out = self.env.reset(**kw)
            poses.append([self._pose()])
            return out

        def step(self, action):
            out = self.env.step(action)
            poses[-1].append(self._pose())
            return out

    factory = ev.build_env(track, False, 0, 15000, 1, "cl_grid_static")   # as evaluate_track(): eval_seed 0
    venv = DummyVecEnv([lambda: PoseRecorder(factory())])
    venv = ev.load_vecnormalize(venv, str(REPO / sel.vecnormalize_path))
    model = ev.ALGOS[sel.algorithm].load(str(REPO / sel.model_path))
    try:
        venv.seed(0)                               # episode 0 of the test
    except Exception:
        pass
    result = ev.run_episode(venv, model, 15000, True, False)    # deterministic, no render
    venv.close()

    test_set = "real" if not track.startswith("vw_") else "vw_test"
    raw = pd.read_csv(TEST / f"test_raw_{tag}_{test_set}.csv")
    row = raw[(raw.track == track) & (raw.episode == 0)].iloc[0]
    got = (f"{result['laps']:.4f}", int(result["length"]), result["outcome"])
    want = (f"{row.laps:.4f}", int(row.length), row.outcome)
    if got != want:
        raise SystemExit(f"REPLAY MISMATCH {tag} on {track}: replay {got} vs test_raw {want}. Stopping.")
    traj = np.array(poses[0][: int(result["length"]) + 1])      # reset pose + one per step
    return dict(tag=tag, grid=grid, algorithm=sel.algorithm, checkpoint=sel.chosen_checkpoint,
                model_path=sel.model_path, vecnormalize_path=sel.vecnormalize_path,
                laps=result["laps"], length=int(result["length"]), outcome=result["outcome"],
                completed=bool(result["completed_lap"]), progress_m=result["progress_m"], traj=traj)


# ------------------------------------------------------------------- render

def track_image(track):
    from f1tenth_gym.envs.track import Track
    t = Track.from_track_name(track)
    res, origin = float(t.spec.resolution), t.spec.origin
    assert abs(float(origin[2])) < 1e-9, "map origin with yaw is not supported"
    occ = t.occupancy_map                         # row 0 = bottom (f1tenth flips the image)
    cl = np.column_stack([t.centerline.xs, t.centerline.ys])
    return occ, res, (float(origin[0]), float(origin[1])), cl


def to_px(xy, res, origin):
    """Metres -> map pixels (x right, y up), using the yaml resolution and origin."""
    return np.column_stack([(xy[:, 0] - origin[0]) / res, (xy[:, 1] - origin[1]) / res])


def corridor_rgb(occ, cl_px):
    """Shade the drivable corridor: the free region of the map image connected to the
    centreline (flood fill, so it follows the real walls), on a light background."""
    from scipy import ndimage
    free = occ > 128
    lab, _ = ndimage.label(free)
    ix = np.clip(np.round(cl_px[:, 0]).astype(int), 0, occ.shape[1] - 1)
    iy = np.clip(np.round(cl_px[:, 1]).astype(int), 0, occ.shape[0] - 1)
    ids = lab[iy, ix]
    ids = ids[ids > 0]
    corridor = lab == np.bincount(ids).argmax() if len(ids) else np.zeros_like(free)
    if corridor.mean() > 0.5:                     # walls not closed: no shading, walls only
        corridor[:] = False
    rgb = np.empty(occ.shape + (3,))
    rgb[:] = np.array([0xf6, 0xf6, 0xf3]) / 255.0          # background
    rgb[corridor] = np.array([0xe2, 0xe5, 0xea]) / 255.0   # track surface
    rgb[~free] = np.array([0x7d, 0x84, 0x8e]) / 255.0      # walls
    return rgb


def on_free_fraction(occ, px):
    ix = np.clip(np.round(px[:, 0]).astype(int), 0, occ.shape[1] - 1)
    iy = np.clip(np.round(px[:, 1]).astype(int), 0, occ.shape[0] - 1)
    return float((occ[iy, ix] > 128).mean())


def render(anim, runs, track, speed, out_dir, gif_size=(1200, 560), png_width=1600):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from PIL import Image

    occ, res, origin, cl = track_image(track)
    pxs = [to_px(r["traj"][:, :2], res, origin) for r in runs]
    cl_px = to_px(cl, res, origin)
    pts = np.vstack([cl_px] + pxs)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.06 * (hi - lo).max()
    x0, y0 = max(lo[0] - pad, 0), max(lo[1] - pad, 0)
    x1, y1 = min(hi[0] + pad, occ.shape[1]), min(hi[1] + pad, occ.shape[0])
    rgb = corridor_rgb(occ, cl_px)
    tri = 0.022 * max(x1 - x0, y1 - y0)

    steps_per_frame = speed * SIM_HZ / FPS
    n_motion = int(math.ceil(max(r["length"] for r in runs) / steps_per_frame)) + 1
    n_frames = n_motion + int(HOLD_S * FPS)

    def make_fig(width_px, height_px, dpi=100):
        fig, axes = plt.subplots(1, 2, figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
        fig.patch.set_facecolor("#f6f6f3")
        artists = []
        for ax, r, title in zip(axes, runs, anim["titles"]):
            ax.imshow(rgb, origin="lower", interpolation="nearest")
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.axis("off")
            ax.set_title(title, fontsize=15, fontweight="bold", pad=6)
            col = COLOURS[r["algorithm"]]
            start = to_px(r["traj"][:1, :2], res, origin)[0]
            ax.plot([start[0]], [start[1]], "o", ms=7, mfc="white", mec="#333333", mew=1.5, zorder=4)
            trail, = ax.plot([], [], color=col, lw=2.2, solid_capstyle="round")
            car = Polygon(np.zeros((3, 2)), closed=True, fc=col, ec="black", lw=0.8, zorder=5)
            ax.add_patch(car)
            status = ax.text(0.5, 0.02, "", transform=ax.transAxes, ha="center", va="bottom",
                             fontsize=14, fontweight="bold")
            cross, = ax.plot([], [], "x", color=CRASH_RED, ms=18, mew=4, zorder=6)
            artists.append((trail, car, status, cross))
        fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.07, wspace=0.03)
        return fig, artists

    def draw_state(artists, k_steps):
        for (trail, car, status, cross), r, px in zip(artists, runs, pxs):
            i = min(int(k_steps), r["length"])
            trail.set_data(px[: i + 1, 0], px[: i + 1, 1])
            x, y = px[i]; th = r["traj"][i, 2]
            c, s = math.cos(th), math.sin(th)
            shape = np.array([[1.0, 0.0], [-0.6, 0.55], [-0.6, -0.55]]) * tri
            car.set_xy(np.column_stack([x + shape[:, 0] * c - shape[:, 1] * s,
                                        y + shape[:, 0] * s + shape[:, 1] * c]))
            if i >= r["length"]:
                if r["outcome"] == "crash":
                    cross.set_data([x], [y]); status.set_text("CRASH"); status.set_color(CRASH_RED)
                elif r["completed"]:
                    status.set_text(f"LAP ✓ {r['length'] / SIM_HZ:.1f} s"); status.set_color("#2b8a3e")
                else:
                    status.set_text("TIME CAP"); status.set_color("#555555")
            else:
                cross.set_data([], []); status.set_text("")

    # ---- GIF
    fig, artists = make_fig(*gif_size)
    fig.text(0.01, 0.015, f"×{speed} speed", fontsize=11, color="#333333", ha="left", va="bottom")
    fig.text(0.99, 0.015, track.replace("_", " "), fontsize=11, color="#333333", ha="right", va="bottom")
    frames = []
    for f in range(n_frames):
        draw_state(artists, min(f, n_motion - 1) * steps_per_frame)
        fig.canvas.draw()
        frames.append(Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                                       bytes(fig.canvas.buffer_rgba())).convert("RGB"))
    plt.close(fig)
    gif = out_dir / f"{anim['name']}.gif"
    # The loop opens on a copy of the final summary frame (both full trails, crash markers,
    # labels), held SUMMARY_HOLD_S, then plays the replay.
    frames = [frames[-1]] + frames
    durations = [int(SUMMARY_HOLD_S * 1000)] + [int(1000 / FPS)] * (len(frames) - 1)
    for colours in (64, 48, 32):
        pal = frames[-1].quantize(colors=colours, method=Image.Quantize.MEDIANCUT)
        q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
        q[0].save(gif, save_all=True, append_images=q[1:], duration=durations, loop=0,
                  optimize=True, disposal=1)
        if gif.stat().st_size < 5e6:
            break
    # ---- PNG: final state, full trails, crash points
    h = int(png_width * gif_size[1] / gif_size[0])
    fig, artists = make_fig(png_width, h)
    fig.text(0.99, 0.015, track.replace("_", " "), fontsize=12, color="#333333", ha="right", va="bottom")
    draw_state(artists, max(r["length"] for r in runs))
    png = out_dir / f"{anim['name']}.png"
    fig.savefig(png, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    loop_s = n_frames / FPS + SUMMARY_HOLD_S
    return dict(gif=gif, png=png, frames=n_frames, loop_s=loop_s, colours=colours,
                free_fraction=[on_free_fraction(occ, p) for p in pxs])


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=pathlib.Path, default=REPO / "results" / "animations")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    ev = load_evaluate()
    report = []
    for anim in ANIMATIONS:
        info = {}
        for (grid, algo, d), (_, tag, _) in zip(anim.get("expect_median", []), anim["panels"]):
            med, seeds = median_seed_tag(grid, algo, d, anim["test_set"])
            info[f"{grid} {algo} d{d} seeds ({anim['test_set']} laps)"] = seeds
            if med != tag:
                raise SystemExit(f"{tag} is not the median seed of grid {grid} {algo} d{d} ({med} is). Stopping.")
        anim["titles"] = [p[2] for p in anim["panels"]]
        other, finisher = anim["panels"][0][1], anim["panels"][1][1]
        track, table, fallback = choose_track(other, finisher, anim["test_set"])
        overall = {t: float(raw(t, anim["test_set"]).laps.mean()) for t in (other, finisher)}
        runs = [replay(ev, grid, tag, track) for grid, tag, _ in anim["panels"]]
        longest_s = max(r["length"] for r in runs) / SIM_HZ
        # Motion about 6 s, so summary (1.5 s) + motion + end hold (1 s) lands in LOOP_S.
        speed = max(1, math.ceil(longest_s / 6.0))
        out = render(anim, runs, track, speed, a.out_dir)
        if not LOOP_S[0] <= out["loop_s"] <= LOOP_S[1]:
            raise SystemExit(f"{anim['name']}: loop {out['loop_s']:.2f} s outside {LOOP_S}")
        traj = pd.concat([pd.DataFrame(dict(tag=r["tag"], step=np.arange(len(r["traj"])),
                                            x=r["traj"][:, 0], y=r["traj"][:, 1], theta=r["traj"][:, 2]))
                          for r in runs])
        traj.to_csv(a.out_dir / f"{anim['name']}_trajectories.csv", index=False, float_format="%.5f")
        report.append(dict(anim=anim, track=track, overall=overall, table=table, fallback=fallback,
                           runs=runs, speed=speed, out=out, info=info))
    write_readme(report, a.out_dir)


def write_readme(report, out_dir):
    L = ["# Slide animations", "",
         "Made by `tools/animate/make_slide_gifs.py` (read-only on code, models and results).",
         "Each panel replays a frozen test checkpoint with its own VecNormalize statistics, through",
         "`evaluate.py`'s own `build_env`, `load_vecnormalize` and `run_episode` (imported unchanged):",
         "deterministic policy, one lap, `cl_grid_static` spawn, 15,000-step cap, eval seed 0, which is",
         "episode 0 of the final test. Every replay reproduced its `test_raw` row exactly (laps, steps, outcome).",
         "",
         "**Track rule** (stated before rendering, applied mechanically to the `test_raw` CSVs): among the",
         "set's tracks where the right-hand agent completes the lap in test episode 0, pick the track where",
         "the left-hand agent's episode-0 score is closest to its overall mean on that set (all test",
         "episodes). The tables below list every qualifying track, ordered by that distance.", ""]
    for rep in report:
        an, out = rep["anim"], rep["out"]
        other, finisher = an["panels"][0][1], an["panels"][1][1]
        L += [f"## {an['name']}: {an['title']} ({an['test_set']})", ""]
        for k, v in rep["info"].items():
            L.append(f"- Median seed check, {k}: " + ", ".join(f"s{s} {l:.4f}" for s, l, _ in v)
                     + f" → **{v[1][2]}**")
        L.append(f"- `{finisher}` completes the lap in episode 0 on **{len(rep['table'])}** "
                 f"{an['test_set']} tracks; `{other}`'s overall {an['test_set']} mean is {rep['overall'][other]:.4f}.")
        if rep["fallback"]:
            L.append(f"- **{rep['fallback']}**")
        L.append(f"- **Chosen track: {rep['track']}**")
        L += ["", f"| track | `{finisher}` lap time | `{other}` episode 0 | distance to its mean |",
              "|---|---|---|---|"]
        for _, t in rep["table"].iterrows():
            mark = "**" if t.track == rep["track"] else ""
            L.append(f"| {mark}{t.track}{mark} | {t.finisher_lap_s:.1f} s | {t.other_ep0:.4f} "
                     f"({t.other_outcome}, {t.other_len_s:.1f} s) | {t['diff']:.4f} |")
        L.append("")
        for r, title in zip(rep["runs"], an["titles"]):
            L.append(f"- *{title}*: `{r['tag']}` (grid {r['grid']}, {r['algorithm']}, checkpoint "
                     f"{r['checkpoint']}: `{r['model_path']}` + `{r['vecnormalize_path']}`). Replay on "
                     f"{rep['track']}: {r['outcome']} after {r['length']} steps ({r['length'] / SIM_HZ:.1f} s), "
                     f"{r['progress_m']:.1f} m, {r['laps']:.4f} laps"
                     f"{', lap completed' if r['completed'] else ''}; matches its test_raw row.")
        sz = lambda p: f"{p.stat().st_size / 1e6:.2f} MB"
        L.append(f"- Speed ×{rep['speed']} at {FPS} fps; loop {out['loop_s']:.1f} s ({out['frames']} frames, "
                 f"{SUMMARY_HOLD_S:g} s summary frame first, 1 s hold); GIF {out['colours']} colours, 1200×560 px.")
        L.append(f"- Files: `{out['gif'].name}` ({sz(out['gif'])}), `{out['png'].name}` ({sz(out['png'])}, 1600 px wide), "
                 f"`{an['name']}_trajectories.csv` (pose every step).")
        L.append(f"- Map check: {', '.join(f'{f:.1%}' for f in out['free_fraction'])} of trajectory points lie on free map cells.")
        L.append("")
    L += ["## Caption numbers (from the test_raw CSVs, all test episodes)", ""]
    for tag, ts in CAPTIONS:
        d = raw(tag, ts)
        e0 = d[d.episode == 0]
        L.append(f"- `{tag}` on {ts}: lap completion **{100 * d.completed_lap.mean():.1f}%** "
                 f"({int(d.completed_lap.sum())}/{len(d)} episodes); episode 0 completed on "
                 f"{int(e0.completed_lap.sum())} of {len(e0)} tracks.")
    L.append("")
    (out_dir / "README.md").write_text("\n".join(L), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n".join(L))


if __name__ == "__main__":
    main()
