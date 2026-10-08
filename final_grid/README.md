# Final grid: running your share of the training

Our study trains two reinforcement-learning algorithms (SAC and PPO) to drive a
simulated F1TENTH car, on 1, 5, 20 or 100 training tracks, with 3 random seeds.
That makes 24 training runs, split between three machines:

| who | job file | runs | `-MaxThreads` | expected time | zip to send |
|---|---|---|---|---|---|
| Desmond | `final_grid\final_jobs_desmond.txt` | SAC, seeds 0 and 1 (8 runs) | 16 | about 7 h | about 720 MB |
| Grant | `final_grid\final_jobs_grant.txt` | SAC, seed 2 (4 runs) | 8 | about 6–7 h | about 360 MB |
| Chris | `final_grid\final_jobs_chris.txt` | PPO, seeds 0, 1 and 2 (12 runs) | 10 | about 3–4.5 h (laptop) | about 85 MB |

The settings were chosen by a rule fixed in advance, as recorded in
`final_grid\DECISION.md`: no time cost, and SAC and PPO at their defaults.

Every run must be **identical to the others apart from the diversity level and
seed**: same code, same Python packages, same tracks, same settings. The setup
script does all the installing, and then checks your machine against Desmond's.

`final_grid\TEAMMATE_SETUP.txt` is the same guide as plain text, written so you
can paste it into an AI assistant. The two files give the same commands.

## Ground rules

- Don't edit any code, settings or job files.
- Don't run `git pull`, `git commit` or `git push` after setup, except the
  one-time pull in step 4 or when Desmond asks.
- Only the automatic vw_val evaluation is allowed. Never evaluate on the `test`,
  `vw_test`, `real`, `narrowA` or `narrowB` track sets; Desmond runs the final
  test once, centrally.
- Only run your own job file.
- If anything fails or looks different from this guide, stop and send Desmond
  the error text.

## What you need

- **Windows 10 or 11**, 64-bit.
- **Python 3.12.10, 64-bit.** Exactly this version.
- **PowerShell 7 (pwsh).** Windows PowerShell 5.1 does not work, because the
  queue launcher needs PowerShell 7.
- **Git for Windows.**
- **About 5 GB of free disk space** on the drive you clone to:
  - about 2.5 GB stays: Python environment 1.2 GB, tracks 0.3 GB, plus your
    runs and your zip
  - up to 1.6 GB more is needed for a while during setup
- **16 GB of RAM recommended.** Each SAC run uses about 1.3 GB by the end.
- **Internet** during setup (about 0.6 GB of downloads). Training needs none.

## 1. Install the tools (once)

1. Git for Windows: https://git-scm.com/download/win (default options).
2. PowerShell 7. In any PowerShell window, run:
   ```
   winget install --id Microsoft.PowerShell -e --source winget
   ```
3. Python 3.12.10, 64-bit:
   https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
   - "Add python.exe to PATH" is optional; the setup script finds Python either way.
   - If you already have another 3.12.x, this installer upgrades it. Other
     Python versions can stay installed.

From now on, use **PowerShell 7**: Start menu → "PowerShell 7".

## 2. Get the code

Use a short local path that **isn't** synced by OneDrive (Documents and Desktop
often are). For example:

```
mkdir C:\work
cd C:\work
git clone -c core.autocrlf=false --recurse-submodules -b final-grid https://github.com/TipperChristopher/Group-4-COMPSCI760.git final-grid
cd final-grid
```

## 3. Set up and check (one command, 30–45 minutes)

Run this with your own name: `desmond`, `grant` or `chris`, as in your job file.

```
pwsh -ExecutionPolicy Bypass -File final_grid\setup_teammate.ps1 -Name grant
```

The script:

1. checks git, the branch and the line endings
2. finds Python 3.12.10
3. creates `.venv` with the exact packages in `final_grid\requirements-lock.txt`
4. sets up the simulator `f1tenth_gym` at the pinned version
5. generates the race tracks (5–15 minutes) and checks every file against
   `tracks\manifest.json`
6. runs `final_grid\check_setup.py`, which:
   - drives the car on 3 tracks with a fixed action sequence, which must match
     Desmond's results exactly
   - runs a short SAC and PPO training to confirm everything works
   - writes `final_grid\fingerprint_<name>.txt` and compares it with
     `final_grid\reference_fingerprint.txt`

If any step fails, it stops with a red **SETUP FAILED** message. Running the
script again is safe: finished steps are only re-checked.

When it ends with `SETUP COMPLETE: this machine MATCHES the reference.`:

1. Send Desmond `final_grid\fingerprint_<name>.txt`.
2. **Wait for Desmond's reply, "MATCH, go", before training.**

If it ends any other way, send Desmond the fingerprint (if it exists) and
`logs\setup_teammate.log`.

## 4. Only if you set up before the job files existed

If your `final_grid` folder has no `final_jobs_<name>.txt`, you set up early. Do
this once, then send Desmond the new fingerprint:

```
git pull
.venv\Scripts\python.exe final_grid\check_setup.py --out final_grid\fingerprint_grant.txt
.venv\Scripts\python.exe final_grid\check_setup.py --compare final_grid\fingerprint_grant.txt
```

The last command must print **MATCH**.

## 5. Keep the computer awake

Training stops if the computer sleeps, hibernates, restarts or you sign out.
Locking the screen (Win+L) is fine. Laptops must stay **plugged in**.

1. Run these commands. They set "never sleep" and "closing the lid does
   nothing", both only while on mains power:
   ```
   powercfg /change standby-timeout-ac 0
   powercfg /change hibernate-timeout-ac 0
   powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
   powercfg /setactive SCHEME_CURRENT
   ```
2. Pause Windows Update (Settings → Windows Update → Pause updates).
3. Laptops: set Settings → System → Power → Power mode to **Best performance**,
   and keep the vents clear. The CPU will run hot at 100%, which is expected.
4. Close games and heavy programs.

## 6. Start the training (after "MATCH, go")

In the repo folder, run your own line:

```
# Desmond
.\launch_queue.ps1 -Jobs final_grid\final_jobs_desmond.txt -ResultsDir results\final_grid_desmond -MaxThreads 16 -Checkpoints 200000,400000,600000,800000,1000000,1200000,1400000,1600000,1800000
# Grant
.\launch_queue.ps1 -Jobs final_grid\final_jobs_grant.txt -ResultsDir results\final_grid_grant -MaxThreads 8 -Checkpoints 200000,400000,600000,800000,1000000,1200000,1400000,1600000,1800000
# Chris
.\launch_queue.ps1 -Jobs final_grid\final_jobs_chris.txt -ResultsDir results\final_grid_chris -MaxThreads 10 -Checkpoints 200000,400000,600000,800000,1000000,1200000,1400000,1600000,1800000
```

It prints `Queue 'final_jobs_<name>' started: ...` and then runs in the
background, so you can close the window. The queue:

- trains every run for 2 million steps
- evaluates each run on the validation tracks (vw_val) every 200,000 steps,
  plus the final model: 10 evaluations per run
- writes a summary named after the queue,
  `results\final_grid_<name>\summary_final_jobs_<name>.csv`

Chris's laptop runs 10 of the 12 PPO runs at once; the last 2 start when
threads free up.

## 7. Check progress (any time, from any PowerShell 7 window)

```
.\launch_queue.ps1 -Jobs final_grid\final_jobs_grant.txt -Status
```

What you should see:

- the scheduler **ALIVE**, with a heartbeat a few seconds old
- jobs going from `running` (or `pending`) to `trained`, then `evaluated`, each
  with Exit 0 and `10 done` evaluations
- `phase: finished` once everything is done

If a job shows a non-zero Exit, or the scheduler says **NOT RUNNING** before
everything is evaluated, send Desmond the `-Status` output and
`logs\queue_final_jobs_<name>\scheduler.log`. Don't delete or restart anything;
finished runs are kept, and Desmond will send the exact commands for the rest.

To stop everything (only if Desmond asks):

```
.\launch_queue.ps1 -Jobs final_grid\final_jobs_grant.txt -Stop
```

## 8. Send the results back

Once the status shows `phase: finished`:

```
pwsh -File final_grid\pack_results.ps1 -Name grant
```

This writes `share\final_<name>.zip`: about 85 MB for Chris, 360 MB for Grant
and 720 MB for Desmond. The zip contains:

- every run folder, with all checkpoints and their normalisation statistics
- the vw_val results, the logs and the queue state
- your fingerprint
- a list of every file with its checksum

**Upload it to [UPLOAD LINK]**, then tell Desmond and include a final `-Status`
output. Keep your copy until Desmond confirms the zip opens.

Afterwards you can restore sleep (30 minutes) and the lid action:

```
powercfg /change standby-timeout-ac 30
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 1
powercfg /setactive SCHEME_CURRENT
```

## Troubleshooting

- **"running scripts is disabled":** start scripts with
  `pwsh -ExecutionPolicy Bypass -File ...` as shown, or first run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that window.
- **"Python 3.12.10 (64-bit) not found":** install it from the link in step 1,
  or pass its path to the setup script: `-Python C:\path\to\python.exe`.
- **"This needs PowerShell 7":** you are in Windows PowerShell. Open
  "PowerShell 7" from the Start menu.
- **The setup fails installing a package:** send Desmond
  `logs\setup_teammate.log`. Don't install other versions yourself.
- **A track check FAILs:** send Desmond the output. Don't regenerate tracks by hand.
- **"Queue ... is already running":** it has already started; use `-Status`.
- **The launcher refuses because of "uncommitted changes":** something in the
  repo was modified. Send Desmond the output of `git status`.
- **The disk is full:** free space outside the repo, then tell Desmond. Don't
  delete anything in the repo's `models\` or `results\` folders.

## For reference: what the files are

| file | what it is |
|---|---|
| `final_grid\TEAMMATE_SETUP.txt` | this guide as plain text |
| `final_grid\DECISION.md` | how the final settings were chosen |
| `final_grid\setup_teammate.ps1` | the one-command setup (step 3) |
| `final_grid\check_setup.py` | prints the fingerprint; `--compare` checks one against Desmond's |
| `final_grid\reference_fingerprint.txt` | Desmond's fingerprint |
| `final_grid\requirements-lock.txt` | exact Python package versions |
| `final_grid\final_jobs_<name>.txt` | each person's training runs |
| `final_grid\pack_results.ps1` | zips your results (step 8) |
| `launch_queue.ps1` | runs the jobs in the background and evaluates checkpoints |
