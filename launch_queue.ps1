<#
.SYNOPSIS
    Unattended job queue: train runs within a thread budget, then evaluate their
    checkpoints on vw_val only.

.DESCRIPTION
    Reads a job list (one job per line: "<tag> | <train.py arguments>"), and
    runs the jobs in list order within -MaxThreads (SAC = 2 torch threads,
    PPO = 1), starting the next job as soon as enough threads are free. When a
    training run finishes, its checkpoints (-Checkpoints, default 400k, 800k,
    1.2M, 1.6M) and final model are evaluated on --track-set vw_val ONLY
    (--episodes 5 --max-steps 15000, deterministic, 1 thread each) into
    <ResultsDir>\<tag>_<checkpoint>_vw_val.csv. Evaluations run before any
    further training starts. A failed job is recorded and the queue carries on.
    When everything is finished it writes and prints the summary
    (analysis/tuning_summary.py).

    UNATTENDED: the scheduler is started through WMI (Win32_Process.Create), so it
    is not a child of this window: closing the window, or the editor killing the
    terminal's process tree, does not stop it. Training and evaluation processes
    are started by the scheduler, so they are detached too. Windows sign-out,
    restart or sleep DO stop everything (sleep is off on AC on this machine).

    State lives in logs\queue_<name>\ (state.json, heartbeat.txt, scheduler.log);
    job logs are logs\<tag>.log / .err.log and logs\<tag>_eval_<ck>.log.

.EXAMPLE
    .\launch_queue.ps1 -Jobs tuning_jobs.txt             # validate and start (detached)
    .\launch_queue.ps1 -Jobs tuning_jobs.txt -Status     # from any window
    .\launch_queue.ps1 -Jobs tuning_jobs.txt -Stop       # stop scheduler + all its jobs
    .\launch_queue.ps1 -Jobs tuning_jobs.txt -Summary    # (re)compute and print the summary
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Jobs,
    [int]$MaxThreads = 16,
    [int[]]$Checkpoints = @(400000, 800000, 1200000, 1600000),
    [string]$ResultsDir = "results\tuning",
    [string]$Baselines = "S0,P0",
    [int]$PollSeconds = 15,
    [switch]$Status,
    [switch]$Stop,
    [switch]$Summary,
    [switch]$AllowDirty,
    # Python interpreter for train/evaluate. Default: this folder's .venv. Pass the
    # main checkout's .venv when running from a git worktree that has none.
    [string]$Python = "",
    [switch]$Scheduler          # internal: run the scheduler loop in this process
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root
if (-not $Python) { $Python = Join-Path $Root ".venv\Scripts\python.exe" }
$JobsPath = (Resolve-Path $Jobs).Path
$QName = [IO.Path]::GetFileNameWithoutExtension($JobsPath)
$QDir = Join-Path $Root "logs\queue_$QName"
$StatePath = Join-Path $QDir "state.json"
$Heartbeat = Join-Path $QDir "heartbeat.txt"
$StopFlag = Join-Path $QDir "STOP"
$SchedLog = Join-Path $QDir "scheduler.log"
$LogDir = Join-Path $Root "logs"
$ResultsAbs = Join-Path $Root $ResultsDir
$EVAL_SET = "vw_val"     # the ONLY set this queue ever evaluates on

function Log([string]$msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $SchedLog -Value $line
}

function Read-JobList {
    $list = @()
    foreach ($raw in Get-Content $JobsPath) {
        $line = $raw.Trim()
        if (-not $line -or $line.StartsWith("#")) { continue }
        $tag, $rest = $line -split "\|", 2
        $tag = $tag.Trim(); $argv = @($rest.Trim() -split "\s+" | Where-Object { $_ })
        if ($tag -notmatch '^[A-Za-z0-9_-]+$') { throw "Bad tag '$tag'" }
        $get = { param($flag) $i = [array]::IndexOf($argv, $flag); if ($i -ge 0) { $argv[$i + 1] } else { $null } }
        $algo = & $get "--algo"; $div = & $get "--diversity"; $seed = & $get "--seed"
        if (-not ($algo -and $div -and $seed)) { throw "Job '$tag' needs --algo, --diversity and --seed" }
        if ($argv -contains "--run-tag" -or $argv -contains "--torch-threads") {
            throw "Job '$tag': --run-tag and --torch-threads are set by the queue" }
        # torch threads for the learner (SAC 2, PPO 1); with --n-envs N > 1 the job
        # also runs N environment worker processes, so it costs torch + N threads
        # against the budget while the learner itself still gets torch threads.
        $torch = $(if ($algo -eq "SAC") { 2 } else { 1 })
        $nenv = [int]$(if (& $get "--n-envs") { & $get "--n-envs" } else { 1 })
        $list += [ordered]@{
            tag = $tag; algo = $algo; diversity = [int]$div; seed = [int]$seed; args = $argv
            torch_threads = $torch; n_envs = $nenv
            threads = $(if ($nenv -gt 1) { $torch + $nenv } else { $torch })
            run_dir = "models\${algo}_${div}tracks_s${seed}_$tag"
            status = "pending"; pid = $null; started = $null; ended = $null; exit = $null
            evals = @()
        }
    }
    if (-not $list) { throw "No jobs in $JobsPath" }
    $dup = $list.tag | Group-Object | Where-Object Count -gt 1
    if ($dup) { throw "Duplicate tags: $($dup.Name -join ', ')" }
    return $list
}

function Save-State($state) {
    $tmp = "$StatePath.tmp"
    $state | ConvertTo-Json -Depth 8 | Set-Content -Path $tmp -Encoding UTF8
    Move-Item -Force $tmp $StatePath
}

function Load-State { Get-Content $StatePath -Raw | ConvertFrom-Json -AsHashtable }

function Run-Summary {
    & $Python (Join-Path $Root "analysis\tuning_summary.py") --state $StatePath `
        --results $ResultsAbs --baselines $Baselines 2>&1
}

# ======================================================================= STATUS
if ($Status) {
    if (-not (Test-Path $StatePath)) { "No queue state at $StatePath"; return }
    $s = Load-State
    $hb = if (Test-Path $Heartbeat) { Get-Content $Heartbeat -Raw } else { "none" }
    $alive = $s.scheduler_pid -and (Get-Process -Id $s.scheduler_pid -ErrorAction SilentlyContinue)
    $age = if (Test-Path $Heartbeat) { [int]((Get-Date) - (Get-Item $Heartbeat).LastWriteTime).TotalSeconds } else { -1 }
    "Queue '$QName'  scheduler PID $($s.scheduler_pid) $(if ($alive) { 'ALIVE' } else { 'NOT RUNNING' })  " +
        "heartbeat: $($hb.Trim())  ($age s ago)  phase: $($s.phase)"
    if (-not $alive -and $s.phase -notin "finished", "stopped") {
        "WARNING: scheduler is not running but the queue is not finished. Last log lines:"
        if (Test-Path $SchedLog) { Get-Content $SchedLog -Tail 5 }
    }
    $rows = foreach ($j in $s.jobs) {
        $steps = ""; $fps = ""
        $prog = Join-Path $Root (Join-Path $j.run_dir "progress.csv")
        if (Test-Path $prog) {
            $last = Import-Csv $prog | Select-Object -Last 1
            if ($last) { $steps = [long]$last.'time/total_timesteps'; $fps = $last.'time/fps' }
        }
        $ev = @($j.evals)
        $evs = if ($ev.Count) { ($ev | Group-Object { $_.status } | ForEach-Object { "$($_.Count) $($_.Name)" }) -join ", " } else { "" }
        [pscustomobject]@{ Tag = $j.tag; Algo = $j.algo; State = $j.status; Steps = $steps; "Steps/s" = $fps
                           Threads = $j.threads; Exit = $j.exit; Evals = $evs; PID = $j.pid }
    }
    $rows | Format-Table -AutoSize | Out-String -Width 200
    $busy = @($s.jobs | Where-Object { $_.status -eq "running" }).Count
    $evr = @($s.jobs | ForEach-Object { $_.evals } | Where-Object { $_.status -eq "running" }).Count
    "running: $busy training, $evr evaluations"
    return
}

# ========================================================================= STOP
if ($Stop) {
    if (-not (Test-Path $StatePath)) { "No queue state at $StatePath"; return }
    New-Item -ItemType File -Force $StopFlag | Out-Null
    $s = Load-State
    if ($s.scheduler_pid) { Stop-Process -Id $s.scheduler_pid -Force -ErrorAction SilentlyContinue }
    foreach ($j in $s.jobs) {
        foreach ($p in @($j.pid) + @($j.evals | ForEach-Object { $_.pid })) {
            if ($p) { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue }
        }
        if ($j.status -eq "running") { $j.status = "stopped" }
        foreach ($e in $j.evals) { if ($e.status -eq "running") { $e.status = "stopped" } }
    }
    $s.phase = "stopped"; Save-State $s
    "Stopped scheduler and all running jobs of queue '$QName'. Finished runs and results are kept."
    return
}

# ====================================================================== SUMMARY
if ($Summary) {
    if (-not (Test-Path $StatePath)) { "No queue state at $StatePath"; return }
    $st = Load-State     # this queue's own results folder and baselines, not the defaults
    $ResultsAbs = Join-Path $Root $st.results_dir; $Baselines = $st.baselines
    if ($st.python) { $Python = $st.python }
    Run-Summary
    return
}

# ==================================================================== SCHEDULER
if ($Scheduler) {
    $s = Load-State
    # Every setting comes from state.json, never from this command line: arrays
    # (-Checkpoints) do not survive "pwsh -File" as typed parameters.
    $MaxThreads = [int]$s.max_threads; $Checkpoints = @($s.checkpoints | ForEach-Object { [int]$_ })
    $ResultsDir = $s.results_dir; $ResultsAbs = Join-Path $Root $ResultsDir
    $Baselines = $s.baselines; $PollSeconds = [int]$s.poll_seconds
    if ($s.python) { $Python = $s.python }
    $s.scheduler_pid = $PID; $s.phase = "running"; Save-State $s
    try {
    Log "scheduler started, PID $PID, $($s.jobs.Count) jobs, max threads $MaxThreads"
    $procs = @{}       # key -> System.Diagnostics.Process (kept for exit codes)

    function Start-Child($key, $argv, $threads, $log) {
        $env:OMP_NUM_THREADS = "$threads"; $env:MKL_NUM_THREADS = "$threads"
        $env:OPENBLAS_NUM_THREADS = "$threads"; $env:MPLBACKEND = "Agg"; $env:PYTHONUNBUFFERED = "1"
        $p = Start-Process -FilePath $Python -ArgumentList $argv -WorkingDirectory $Root -PassThru `
                -WindowStyle Hidden -RedirectStandardOutput "$log.log" -RedirectStandardError "$log.err.log"
        $null = $p.Handle          # keep a handle so ExitCode is readable after exit
        $procs[$key] = $p
        return $p.Id
    }

    while ($true) {
        Set-Content -Path $Heartbeat -Value ("{0}  pid {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $PID)
        if (Test-Path $StopFlag) { Log "STOP flag found; exiting"; break }

        # --- reap finished processes
        foreach ($j in $s.jobs) {
            if ($j.status -eq "running" -and $procs[$j.tag].HasExited) {
                $j.exit = $procs[$j.tag].ExitCode; $j.ended = (Get-Date -Format s); $j.pid = $null
                $cfg = Join-Path $Root (Join-Path $j.run_dir "run_config.json")
                $ok = ($j.exit -eq 0) -and (Test-Path $cfg) -and ((Get-Content $cfg -Raw | ConvertFrom-Json).status -eq "completed")
                if ($ok) {
                    $j.status = "trained"
                    $labels = @($Checkpoints | ForEach-Object { @{ label = "{0}k" -f ($_ / 1000); steps = $_ } }) +
                              @(@{ label = "final"; steps = $null })
                    $j.evals = @($labels | ForEach-Object { [ordered]@{ label = $_.label; steps = $_.steps
                                  status = "pending"; pid = $null; exit = $null } })
                    Log "$($j.tag) trained OK; queued $($j.evals.Count) vw_val evaluations"
                } else {
                    $j.status = "failed"; Log "$($j.tag) FAILED (exit $($j.exit)); continuing"
                }
            }
            foreach ($e in $j.evals) {
                $k = "$($j.tag)|$($e.label)"
                if ($e.status -eq "running" -and $procs[$k].HasExited) {
                    $e.exit = $procs[$k].ExitCode; $e.pid = $null
                    $e.status = $(if ($e.exit -eq 0) { "done" } else { "failed" })
                    Log "$k eval $($e.status) (exit $($e.exit))"
                }
            }
            if ($j.status -eq "trained" -and -not ($j.evals | Where-Object { $_.status -in "pending", "running" })) {
                $j.status = "evaluated"; Log "$($j.tag) all evaluations finished"
            }
        }

        $used = 0
        foreach ($j in $s.jobs) {
            if ($j.status -eq "running") { $used += $j.threads }
            $used += @($j.evals | Where-Object { $_.status -eq "running" }).Count
        }

        # --- evaluations first (short, 1 thread each)
        foreach ($j in $s.jobs) {
            foreach ($e in $j.evals) {
                if ($e.status -ne "pending" -or $used -ge $MaxThreads) { continue }
                $out = Join-Path $ResultsAbs ("{0}_{1}_{2}.csv" -f $j.tag, $e.label, $EVAL_SET)
                $argv = @("-u", "evaluate.py", "--algo", $j.algo, "--diversity", $j.diversity, "--seed", $j.seed,
                          "--run-tag", $j.tag, "--track-set", $EVAL_SET, "--episodes", "5",
                          "--max-steps", "15000", "--no-plot", "--out", $out)
                if ($e.steps) {
                    $ck = Join-Path $Root (Join-Path $j.run_dir ("{0}_checkpoint_{1}_steps.zip" -f $j.algo, $e.steps))
                    if (-not (Test-Path $ck)) { $e.status = "missing"; Log "$($j.tag) $($e.label): no checkpoint"; continue }
                    $argv += @("--checkpoint", $e.steps)
                }
                if ($argv[[array]::IndexOf($argv, "--track-set") + 1] -ne "vw_val") { throw "refusing non-vw_val evaluation" }
                $e.pid = Start-Child "$($j.tag)|$($e.label)" $argv 1 (Join-Path $LogDir "$($j.tag)_eval_$($e.label)")
                $e.status = "running"; $used += 1
                Log "$($j.tag) eval $($e.label) started PID $($e.pid)"
            }
        }

        # --- training, strictly in list order
        foreach ($j in $s.jobs) {
            if ($j.status -ne "pending") { continue }
            if ($used + $j.threads -gt $MaxThreads) { break }
            $tt = $(if ($j.torch_threads) { $j.torch_threads } else { $j.threads })
            $argv = @("-u", "train.py") + $j.args + @("--run-tag", $j.tag, "--torch-threads", $tt)
            $j.pid = Start-Child $j.tag $argv $tt (Join-Path $LogDir $j.tag)
            $j.status = "running"; $j.started = (Get-Date -Format s); $used += $j.threads
            Log "$($j.tag) started PID $($j.pid) ($($j.threads) threads)"
        }

        Save-State $s
        $open = $s.jobs | Where-Object { $_.status -in "pending", "running", "trained" }
        if (-not $open) { break }
        Start-Sleep -Seconds $PollSeconds
    }

    if (-not (Test-Path $StopFlag)) {
        $s.phase = "summarising"; Save-State $s
        Log "all jobs finished; writing summary"
        Run-Summary | Out-File -FilePath (Join-Path $QDir "summary.txt") -Encoding UTF8
        $s.phase = "finished"
    } else { $s.phase = "stopped" }
    $s.scheduler_pid = $null; Save-State $s
    Set-Content -Path $Heartbeat -Value ("{0}  scheduler exited ({1})" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $s.phase)
    Log "scheduler exiting, phase $($s.phase)"
    } catch {
        Log "SCHEDULER CRASHED: $_ $($_.ScriptStackTrace)"
        try { $s.phase = "crashed"; $s.scheduler_pid = $null; Save-State $s } catch {}
        Set-Content -Path $Heartbeat -Value ("{0}  scheduler CRASHED, see scheduler.log" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
        exit 1
    }
    return
}

# ======================================================================= LAUNCH
if (-not (Test-Path $Python)) { throw "No venv python at $Python" }
if ((Test-Path $StatePath)) {
    $old = Load-State
    if ($old.scheduler_pid -and (Get-Process -Id $old.scheduler_pid -ErrorAction SilentlyContinue)) {
        throw "Queue '$QName' is already running (scheduler PID $($old.scheduler_pid))." }
    throw "Queue state already exists at $QDir. Move or delete it to start this queue afresh."
}
$dirty = git status --porcelain --untracked-files=no | ForEach-Object { ($_.Trim() -split '\s+', 2)[1] } |
         Where-Object { $_ -and $_ -ne "f1tenth_gym" }
if ($dirty -and -not $AllowDirty) {
    throw "Uncommitted changes to tracked code: $($dirty -join ', '). Commit first, or pass -AllowDirty." }

# NB: not $jobs, which PowerShell would treat as the [string] -Jobs parameter.
$jobList = @(Read-JobList)
$tooBig = @($jobList | Where-Object { $_.threads -gt $MaxThreads })
if ($tooBig) { throw "Jobs needing more than -MaxThreads $MaxThreads threads would never start: $($tooBig.tag -join ', ')" }
$clash = @()
foreach ($j in $jobList) {
    if (Test-Path (Join-Path $Root (Join-Path $j.run_dir "run_config.json"))) { $clash += $j.run_dir }
    Get-ChildItem $ResultsAbs -Filter "$($j.tag)_*_vw_val.csv" -ErrorAction SilentlyContinue |
        ForEach-Object { $clash += $_.FullName }
}
$clash = @($clash | Where-Object { $_ })
if ($clash.Count) { throw "Would overwrite existing runs/results: $($clash -join '; '). Nothing started." }

New-Item -ItemType Directory -Force $QDir, $LogDir, $ResultsAbs | Out-Null
Save-State ([ordered]@{ queue = $QName; jobs_file = $JobsPath; created = (Get-Date -Format s)
                        commit = (git rev-parse --short HEAD).Trim(); max_threads = $MaxThreads
                        checkpoints = $Checkpoints; results_dir = $ResultsDir; eval_set = $EVAL_SET
                        baselines = $Baselines; poll_seconds = $PollSeconds; python = $Python
                        scheduler_pid = $null; phase = "starting"; jobs = $jobList })

$pwsh = (Get-Process -Id $PID).Path
$cmd = "`"$pwsh`" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Scheduler " +
       "-Jobs `"$JobsPath`""
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = $cmd; CurrentDirectory = $Root; ProcessStartupInformation = $startup }
if ($r.ReturnValue -ne 0) { throw "Could not start the scheduler (WMI return $($r.ReturnValue))" }
# Do not report success until the scheduler has actually taken over.
$deadline = (Get-Date).AddSeconds(60)
while ((Get-Date) -lt $deadline -and -not (Test-Path $Heartbeat)) { Start-Sleep -Milliseconds 500 }
if (-not (Test-Path $Heartbeat)) {
    throw "Scheduler (PID $($r.ProcessId)) did not start within 60 s. See $SchedLog and $StatePath." }
"Queue '$QName' started: $($jobList.Count) jobs, scheduler PID $($r.ProcessId) (detached via WMI)."
"Status : .\launch_queue.ps1 -Jobs $Jobs -Status"
"Stop   : .\launch_queue.ps1 -Jobs $Jobs -Stop"
"Summary: .\launch_queue.ps1 -Jobs $Jobs -Summary"
