"""Copy every piece of evidence behind CONTEXT.md / the wiki into git.

Logs + configs for every run, final weights for completed runs (~330 KB each),
and every evaluation CSV. Nothing is recomputed here; this is a copy.
"""
import glob, json, os, shutil, pathlib
import pandas as pd
ROOT = pathlib.Path(r"E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760")
REPO = ROOT / "team_repo_heldout"; V = ROOT / "_verify"
OUT = REPO / "results/ppo_diagnosis"
(OUT / "runs").mkdir(parents=True, exist_ok=True)
rows = []
for cfgp in sorted(glob.glob(str(REPO / "models/*/run_config.json"))):
    run = pathlib.Path(cfgp).parent; tag = run.name
    cfg = json.load(open(cfgp)); a = cfg["args"]
    dst = OUT / "runs" / tag; dst.mkdir(exist_ok=True)
    for f in ["run_config.json", "progress.csv", "monitor_0.monitor.csv", "per_episode_log.csv"]:
        if (run / f).exists(): shutil.copy2(run / f, dst / f)
    if cfg.get("status") == "completed":
        for f in ["final_model.zip", "vecnormalize.pkl"]:
            if (run / f).exists(): shutil.copy2(run / f, dst / f)
    # also keep the best held-out checkpoint of the gamma run (800k)
    if tag == "PPO_1tracks_s0_G999_gamma":
        for f in ["PPO_checkpoint_800000_steps.zip", "PPO_checkpoint_vecnormalize_800000_steps.pkl"]:
            shutil.copy2(run / f, dst / f)
    e = pd.read_csv(run / "monitor_0.monitor.csv", skiprows=1); e["cum"] = e.l.cumsum()
    W = e[e.cum > e.cum.max() - 200000]
    rep = a.get("action_repeat") or 1
    rows.append(dict(run=tag, algo=a["algo"], seed=a["seed"], diversity=a["diversity"],
        n_steps=a.get("n_steps") or (2048 if a["algo"] == "PPO" else None),
        batch_size=a.get("batch_size") or (64 if a["algo"] == "PPO" else 256),
        n_epochs=a.get("n_epochs") or (10 if a["algo"] == "PPO" else None),
        gamma=a.get("gamma") or 0.99, gae_lambda=a.get("gae_lambda") or (0.95 if a["algo"] == "PPO" else None),
        ent_coef=a.get("ent_coef") if a.get("ent_coef") is not None else (0.0 if a["algo"] == "PPO" else "auto"),
        target_kl=a.get("target_kl"), log_std_init=a.get("log_std_init"), learning_rate=a.get("learning_rate") or 3e-4,
        crash_penalty=cfg["reward_constants"]["CRASH_PENALTY"], action_repeat=rep,
        status_in_config=cfg.get("status"), decisions_logged=int(e.cum.max()),
        physics_steps_logged=int(e.cum.max()) * rep,
        last200k_mean_m=round(W.progress_m.mean(), 1), last200k_lap_rate=round((W.laps >= 1).mean(), 3),
        max_laps=round(e.laps.max(), 3), wall_hours=round((cfg.get("wall_seconds") or 0) / 3600, 2)))
pd.DataFrame(rows).to_csv(OUT / "runs_summary.csv", index=False)
# evaluation CSVs
for src, name in [("ppo_sweep_eval", "eval_sweep_1M"), ("lap_break", "eval_ck800k"),
                  ("lap_break_final", "eval_ck2M"), ("g999_curve", "eval_g999_curve"),
                  ("width_eval", "eval_width_oursac"), ("repro", "repro_desmond_grid")]:
    s = V / src
    if s.exists():
        d = OUT / name; d.mkdir(exist_ok=True)
        for f in glob.glob(str(s / "*.csv")) + glob.glob(str(s / "*.json")):
            shutil.copy2(f, d / os.path.basename(f))
for f in ["eval_ppo_sweep.log", "lap_break.log", "lap_break_final.log", "g999_curve.log", "width_eval.log"]:
    if (V / f).exists():
        (OUT / "logs").mkdir(exist_ok=True); shutil.copy2(V / f, OUT / "logs" / f)
print(pd.DataFrame(rows)[["run","seed","n_steps","gamma","crash_penalty","action_repeat","status_in_config","decisions_logged","last200k_mean_m","last200k_lap_rate"]].to_string(index=False))
tot = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
print(f"\narchived {sum(1 for p in OUT.rglob('*') if p.is_file())} files, {tot/1e6:.1f} MB -> {OUT}")
