# Final grid: running your share of the training

Our study trains two reinforcement-learning algorithms (SAC and PPO) to drive a
simulated F1TENTH car, on 1, 5, 20 or 100 training tracks. That makes 8 training
runs. Each of us runs the same 8 runs with a different random **seed**, so the
results can be averaged over 3 seeds:

| seed | who |
|---|---|
| 0 | Desmond |
| 1 | *(Desmond will tell you)* |
| 2 | *(Desmond will tell you)* |

Your runs must be **identical to Desmond's apart from the seed**: same code, same
Python packages, same tracks, same settings. The setup script below does all the
installing, and then checks your machine against Desmond's. You don't need to
understand the code. Please don't edit any file in the repo, because the check
fails if a tracked file changes.

**Time needed:** about 30–45 minutes for setup. Training then takes about 6 hours
on a desktop and 9–12 hours on a laptop, with the computer left on and awake
(overnight is ideal).

## What you need

- **Windows 10 or 11**, 64-bit.
- **About 5 GB of free disk space** on the drive where you put the repo:
  - about 2.5 GB stays: Python environment 1.2 GB, tracks 0.3 GB, results 0.4 GB,
    the zip you send back 0.4 GB
  - about 1.5 GB more is needed for a while during setup
- **16 GB of RAM recommended.** At the end, each of the 4 SAC runs uses about 1.3 GB.
- **Internet** during setup (about 0.6 GB of downloads). Training needs no internet.
- **Access to the GitHub repo** `TipperChristopher/Group-4-COMPSCI760`.

## 1. Install the tools (once)

1. **Git for Windows:** https://git-scm.com/download/win (the default options are fine).
2. **PowerShell 7.** This is not the "Windows PowerShell" that comes with Windows.
   Open Windows PowerShell and run:
   ```
   winget install --id Microsoft.PowerShell -e --source winget
   ```
3. **Python 3.12.10, exactly this version, 64-bit:**
   https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
   - The default options are fine. "Add python.exe to PATH" is optional.
   - If you already have another Python 3.12.x, this installer upgrades it to
     3.12.10.
   - Other Python versions (3.11, 3.13, ...) can stay installed.

From now on, use **PowerShell 7**: Start menu → "PowerShell 7" (the window title
says "PowerShell 7"). Every command below goes in that window.

## 2. Get the code

Put the repo on a local drive, **not** in a OneDrive-synced folder (Documents and
Desktop are often synced). For example `C:\f1tenth`:

```
mkdir C:\f1tenth
cd C:\f1tenth
git clone -c core.autocrlf=false --recurse-submodules -b final-grid https://github.com/TipperChristopher/Group-4-COMPSCI760.git final-grid
cd final-grid
```

`-c core.autocrlf=false` keeps the files byte-identical to Desmond's, and
`--recurse-submodules` also downloads the simulator (`f1tenth_gym`). If you forget
either, the setup script fixes it.

## 3. Set up and check (one command, 30–45 minutes)

Inside the `final-grid` folder, replace `yourname` with your first name
(letters only):

```
pwsh -ExecutionPolicy Bypass -File final_grid\setup_teammate.ps1 -Name yourname
```

The script does these steps, and stops with a red **SETUP FAILED** message if any
of them goes wrong:

1. checks git, the branch and the line endings
2. finds Python 3.12.10
3. creates the Python environment `.venv` with the exact package versions in
   `final_grid\requirements-lock.txt`
4. sets up the simulator `f1tenth_gym` at the pinned version
5. generates the 130 race tracks (5–15 minutes) and checks every file against
   `tracks\manifest.json`
6. runs `final_grid\check_setup.py` (about 1 minute), which:
   - drives the car on 3 tracks with a fixed sequence of actions and checks that
     the results match Desmond's exactly
   - runs a short SAC and PPO training to confirm everything works
   - writes your fingerprint to `final_grid\fingerprint_yourname.txt` and
     compares it with Desmond's

**Good result:** the last lines say
`SETUP COMPLETE: this machine MATCHES the reference.` Send
`final_grid\fingerprint_yourname.txt` to Desmond.

**Anything else:** don't start training. Send Desmond
`final_grid\fingerprint_yourname.txt` (if it exists) and `logs\setup_teammate.log`.

You can run the script again safely: steps that are already done are only
re-checked.

## 4. Keep the computer awake

Training stops if the computer sleeps, hibernates, restarts or you sign out.
Locking the screen (Win+L) is fine. Laptops must stay **plugged in**.

1. Run these commands in PowerShell 7. They set "never sleep" and "closing the lid
   does nothing", both only while on mains power:
   ```
   powercfg /change standby-timeout-ac 0
   powercfg /change hibernate-timeout-ac 0
   powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
   powercfg /setactive SCHEME_CURRENT
   ```
2. **Pause Windows Update for 1 week** (Settings → Windows Update → Pause updates),
   so it can't restart the computer overnight.
3. **Laptops:** set Settings → System → Power → Power mode to **Best
   performance**, and put the laptop on a hard surface so the fans can breathe.
4. Close games and other heavy programs. Light browsing is fine.

To undo all this after training, which sets sleep after 30 minutes and the lid
back to sleep:

```
powercfg /change standby-timeout-ac 30
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 1
powercfg /setactive SCHEME_CURRENT
```

## 5. Start the training

Desmond will tell you when the final job files are ready. Then, in the
`final-grid` folder, replace `N` with your seed and `yourname` with your name:

```
git pull
.venv\Scripts\python.exe final_grid\check_setup.py --out final_grid\fingerprint_yourname.txt
.venv\Scripts\python.exe final_grid\check_setup.py --compare final_grid\fingerprint_yourname.txt
```

The last command must print **MATCH**. If it does, start the training:

```
.\launch_queue.ps1 -Jobs final_grid\final_jobs_seedN.txt -ResultsDir results\final_grid -MaxThreads 12 -Checkpoints 200000,400000,600000,800000,1000000,1200000,1400000,1600000,1800000
```

It prints `Queue 'final_jobs_seedN' started: 8 jobs ...`. Everything now runs in
the background, so you can close the window. The queue then:

- trains all 8 runs, each for 2 million steps
- evaluates each run on the validation tracks every 200,000 steps, plus the final
  model

If it says `No jobs in ...`, you haven't got the real job files yet: run
`git pull` again, or ask Desmond.

## 6. Check progress (from any PowerShell 7 window)

```
cd C:\f1tenth\final-grid
.\launch_queue.ps1 -Jobs final_grid\final_jobs_seedN.txt -Status
```

What healthy looks like:

- the first line says **ALIVE** and the heartbeat is a few seconds old
- the PPO runs reach 2,000,000 steps after about 45–60 minutes
- the SAC runs take about 5–6 hours on a desktop and 9–12 hours on a laptop
- each run ends with `evaluated` and `10 done` in the Evals column
- when everything has finished, the first line says `phase: finished`

## 7. Send the results back

Once the status shows `phase: finished`:

```
pwsh -File final_grid\pack_results.ps1 -Name yourname -Seed N
```

This writes `share\final_yourname_seedN.zip`, about **400 MB**. The zip contains:

- all 8 run folders, with every checkpoint and its normalisation statistics
- the validation results and the logs
- your fingerprint
- a list of every file with its checksum

**Upload it to:** *(Desmond will add the location here)*. Then message Desmond.
Keep your copy until Desmond confirms the zip opens.

## If something goes wrong

- **The setup script fails:** the red message names the step and what to do.
  Fix it and run the same command again. The most common causes:
  - the wrong Python version (you need 3.12.10 exactly)
  - Windows PowerShell instead of PowerShell 7
  - no internet
  - a full disk
- **The fingerprint does not MATCH:** don't train. Send
  `final_grid\fingerprint_yourname.txt` and `logs\setup_teammate.log` to Desmond.
- **The status says `NOT RUNNING` but the phase isn't finished** (for example
  after a restart or sleep): don't delete anything and don't restart the queue.
  - Send Desmond the output of the `-Status` command and the file
    `logs\queue_final_jobs_seedN\scheduler.log`.
  - Finished runs are kept. Only unfinished runs need redoing, and Desmond will
    send the exact commands.
- **A run shows `failed`:** send Desmond `logs\<that run's tag>.err.log` and the
  `-Status` output. The other runs carry on.
- **You need to stop everything** (for example, the laptop is needed):
  ```
  .\launch_queue.ps1 -Jobs final_grid\final_jobs_seedN.txt -Stop
  ```
  Then tell Desmond. Stopped runs have to start again from the beginning.

## For reference: what the files are

| file | what it is |
|---|---|
| `final_grid\setup_teammate.ps1` | the one-command setup (section 3) |
| `final_grid\check_setup.py` | prints the fingerprint; `--compare` checks one against Desmond's |
| `final_grid\reference_fingerprint.txt` | Desmond's fingerprint |
| `final_grid\requirements-lock.txt` | exact Python package versions |
| `final_grid\final_jobs_seed0/1/2.txt` | the 8 training runs for each seed |
| `final_grid\pack_results.ps1` | zips your results (section 7) |
| `launch_queue.ps1` | runs the jobs in the background and evaluates checkpoints |
