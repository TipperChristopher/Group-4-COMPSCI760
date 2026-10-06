<#
.SYNOPSIS
    Launch, monitor and stop the 8-cell PPO/SAC x diversity training grid.

.DESCRIPTION
    Cells: SAC and PPO at diversity 1, 5, 20, 100. Every cell gets identical
    settings except algorithm and pool size. SAC cells start first because
    they are the slow ones (they bound the wall clock).

    Each cell runs as its own background python process, writing:
        logs\<cell>.log        training output (provenance block, SB3 tables)
        logs\<cell>.err.log    warnings and any crash traceback
        models\<ALGO>_<D>tracks_s<seed>\   run_config.json, progress.csv,
                               monitor_0.monitor.csv, checkpoints every 100k,
                               final_model.zip, vecnormalize.pkl

    stdout and stderr go to separate files because Start-Process cannot send
    both streams of a background process to one file. Processes get their own
    hidden console, so closing this window does not kill the runs.

    Safety: refuses to launch on uncommitted tracked code (each run records
    its commit hash, which must reproduce it) and refuses if any cell's run
    directory already holds a run, before starting anything.

    -RunTag r40 makes a separate grid: cells SAC_d20_s0_r40, run folders
    models\SAC_20tracks_s0_r40\, logs logs\SAC_d20_s0_r40.log and PID file
    logs\grid_s0_r40.pids.json, so nothing collides with the untagged grid.
    Pass the same -Seed and -RunTag to -Status, -Watch and -Stop; they show
    the cells recorded in that grid's PID file.

.EXAMPLE
    .\launch_grid.ps1                      # launch the grid (seed 0, 2M steps)
    .\launch_grid.ps1 -Status              # one-off progress table
    .\launch_grid.ps1 -Watch               # progress table, refreshed every 30 s
    .\launch_grid.ps1 -Stop SAC_d20_s0     # stop one cell
    .\launch_grid.ps1 -Stop all            # stop every cell of this seed
    .\launch_grid.ps1 -Algos SAC -RunTag r40           # SAC-only tagged grid
    .\launch_grid.ps1 -RunTag r40 -Watch               # watch it
    .\launch_grid.ps1 -RunTag r40 -Stop SAC_d20_s0_r40 # stop one tagged cell
#>
[CmdletBinding()]
param(
    [int]$Seed = 0,
    [int]$TrackSeed = 0,
    [long]$TotalTimesteps = 2000000,
    # torch / BLAS threads per process. Measured on a Ryzen 7 9800X3D (8 cores,
    # 16 threads), all 8 cells concurrent, steady-state steps/s per cell:
    #
    #   SAC/PPO threads     SAC         PPO           note
    #   1 / 1               72-77       770-1094
    #   2 / 2               91-97       646-1007
    #   4 / 1               62-74       676-965       20 threads on 16 CPUs
    #   2 / 1               94-102      738-1067      <- default, best for SAC
    #   2 / 1, pinned       81-85       970-1395      SAC confined to one core
    #
    # and with SAC running alone (the last ~85% of the grid, after PPO ends):
    #   SAC 2 threads 95-102, SAC 4 threads 96-105.
    #
    # SAC's per-step gradient update stops scaling past 2 threads, and PPO is
    # bound by the simulator, so 2/1 is best. SAC bounds the wall clock.
    [int]$SacThreads = 2,
    [int]$PpoThreads = 1,
    # Algorithms to launch. SAC cells always start first whatever the order
    # here, because they bound the wall clock.
    [ValidateSet("SAC", "PPO")][string[]]$Algos = @("PPO", "SAC"),
    # Older spelling of -Algos for a single algorithm; kept so existing
    # commands still work.
    [ValidateSet("", "SAC", "PPO")][string]$Only = "",
    # Suffix for run folders, cell names, logs and the PID file (train.py
    # --run-tag). Empty = the original untagged grid.
    [string]$RunTag = "",
    # Pool prefix (train.py --track-prefix). Must name a training split.
    [string]$TrackPrefix = "synthetic_track_",
    # Pin each cell to its own block of whole physical cores. Off by default
    # because it measured WORSE for the grid: it confines each SAC cell's two
    # threads to the SMT siblings of one core (SAC -15%), and pinned SAC cells
    # cannot spread onto the cores PPO frees when it finishes. It speeds PPO
    # up ~30%, which only matters if PPO were the bottleneck.
    [switch]$Pin,
    [double]$StaggerSeconds = 3,
    [switch]$Status,
    [switch]$Watch,
    [string]$Stop,
    [switch]$AllowDirty,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($RunTag -and $RunTag -notmatch '^[A-Za-z0-9-]+$') {
    throw "-RunTag may contain only letters, digits and '-' (got '$RunTag')"
}
$Suffix   = if ($RunTag) { "_$RunTag" } else { "" }

$Python   = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$LogDir   = Join-Path $PSScriptRoot "logs"
$PidFile  = Join-Path $LogDir "grid_s$Seed$Suffix.pids.json"
if ($Only) { $Algos = @($Only) }
# SAC first: it bounds the wall clock.
$Algos    = @(@("SAC", "PPO") | Where-Object { $Algos -contains $_ })
$Div      = @(1, 5, 20, 100)

function Get-Threads([string]$algo) {
    if ($algo -eq "SAC") { return $SacThreads } else { return $PpoThreads }
}

function Read-Pids {
    if (Test-Path $PidFile) { return Get-Content $PidFile -Raw | ConvertFrom-Json }
    return $null
}

function Get-Cells([switch]$Launched) {
    # For -Status/-Watch/-Stop, show the cells this grid actually launched
    # (from its PID file), so a SAC-only tagged grid needs no -Algos repeated.
    $names = $null
    if ($Launched) {
        $pids = Read-Pids
        if ($pids) { $names = @($pids.PSObject.Properties.Name) }
    }
    $cells = @()
    foreach ($a in @("SAC", "PPO")) {
        if (-not $names -and $Algos -notcontains $a) { continue }
        foreach ($d in $Div) {
            $name = "${a}_d${d}_s$Seed$Suffix"
            if ($names -and $names -notcontains $name) { continue }
            $cells += [pscustomobject]@{
                Name   = $name
                Algo   = $a
                Div    = $d
                RunDir = Join-Path $PSScriptRoot "models\${a}_${d}tracks_s$Seed$Suffix"
            }
        }
    }
    return $cells
}

function Show-Status {
    $pids = Read-Pids
    $rows = foreach ($c in Get-Cells -Launched) {
        $cfgPath  = Join-Path $c.RunDir "run_config.json"
        $progPath = Join-Path $c.RunDir "progress.csv"
        $state = "not started"; $steps = 0; $fps = $null; $rew = ""; $len = ""
        if (Test-Path $cfgPath) {
            $cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
            $state = $cfg.status
        }
        if (Test-Path $progPath) {
            $last = Import-Csv $progPath | Select-Object -Last 1
            if ($last) {
                $steps = [long]$last.'time/total_timesteps'
                $fps   = [double]$last.'time/fps'
                $rew   = $last.'rollout/ep_rew_mean'
                $len   = $last.'rollout/ep_len_mean'
            }
        }
        $alive = $false
        if ($pids -and $pids.($c.Name)) {
            $alive = [bool](Get-Process -Id $pids.($c.Name) -ErrorAction SilentlyContinue)
        }
        if ($state -eq "running" -and -not $alive -and $pids) { $state = "DIED (see .err.log)" }
        $eta = ""
        if ($fps -and $fps -gt 0 -and $state -eq "running") {
            $eta = [TimeSpan]::FromSeconds(($TotalTimesteps - $steps) / $fps).ToString("hh\:mm\:ss")
        }
        $pct = if ($TotalTimesteps -gt 0) { "{0,5:N1}%" -f (100.0 * $steps / $TotalTimesteps) } else { "" }
        [pscustomobject]@{
            Cell = $c.Name; State = $state; Steps = $steps; Done = $pct
            "Steps/s" = $fps; "ETA" = $eta
            "ep_rew_mean" = $rew; "ep_len_mean" = $len
            PID = if ($pids) { $pids.($c.Name) } else { "" }
        }
    }
    $rows | Format-Table -AutoSize
}

# ----------------------------------------------------------------- modes
if ($Status) { Show-Status; return }

if ($Watch) {
    while ($true) {
        Clear-Host
        Write-Host ("Grid seed {0}{1}  {2}  (Ctrl+C to stop watching; runs continue)" -f `
            $Seed, $(if ($RunTag) { "  tag $RunTag" } else { "" }), (Get-Date))
        Show-Status
        Start-Sleep -Seconds 30
    }
}

if ($Stop) {
    $pids = Read-Pids
    if (-not $pids) { throw "No PID file at $PidFile" }
    $targets = if ($Stop -eq "all") { (Get-Cells -Launched).Name } else { @($Stop) }
    foreach ($t in $targets) {
        $p = $pids.$t
        if (-not $p) { Write-Warning "Unknown cell '$t'"; continue }
        if (Get-Process -Id $p -ErrorAction SilentlyContinue) {
            Stop-Process -Id $p -Force
            Write-Host "stopped $t (PID $p)"
        } else {
            Write-Host "$t (PID $p) is not running"
        }
    }
    Write-Host "Checkpoints already written stay in models\. A stopped cell keeps its run_config.json, so relaunching it needs train.py --overwrite."
    return
}

# --------------------------------------------------------------- launch
if (-not (Test-Path $Python)) { throw "No venv python at $Python" }

# 1. Clean tree: the commit hash each run records must reproduce it.
$dirty = git status --porcelain --untracked-files=no |
         ForEach-Object { ($_.Trim() -split '\s+', 2)[1] } |
         Where-Object { $_ -and $_ -ne "f1tenth_gym" }
if ($dirty -and -not $AllowDirty) {
    throw "Uncommitted changes to tracked code: $($dirty -join ', '). Commit first, or pass -AllowDirty."
}
$commit = (git rev-parse --short HEAD).Trim()
$branch = (git rev-parse --abbrev-ref HEAD).Trim()

# 2. Never clobber: fail before starting anything if any cell already ran.
$cells = Get-Cells
$taken = $cells | Where-Object { Test-Path (Join-Path $_.RunDir "run_config.json") }
if ($taken) { throw "Run directories already in use: $($taken.Name -join ', '). Nothing was started." }

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$env:MPLBACKEND       = "Agg"
$env:PYTHONUNBUFFERED = "1"

$logical = [Environment]::ProcessorCount
Write-Host ("Launching {0} cells | commit {1} ({2}) | seed {3} | track-seed {4} | {5:N0} steps | threads SAC={6} PPO={7} | pin={8} | tag={9} | prefix={10}" -f `
    $cells.Count, $commit, $branch, $Seed, $TrackSeed, $TotalTimesteps, $SacThreads, $PpoThreads, [bool]$Pin,
    $(if ($RunTag) { $RunTag } else { "(none)" }), $TrackPrefix)

$pidMap = [ordered]@{}
$nextCpu = 0
foreach ($c in $cells) {
    $log = Join-Path $LogDir "$($c.Name).log"
    $err = Join-Path $LogDir "$($c.Name).err.log"
    $threads = Get-Threads $c.Algo
    $argList = @("-u", "train.py",
                 "--algo", $c.Algo, "--diversity", $c.Div,
                 "--seed", $Seed, "--track-seed", $TrackSeed,
                 "--total-timesteps", $TotalTimesteps,
                 "--torch-threads", $threads)
    # Only passed when set, so an untagged default launch runs exactly the
    # command it always did.
    if ($RunTag) { $argList += @("--run-tag", $RunTag) }
    if ($TrackPrefix -ne "synthetic_track_") { $argList += @("--track-prefix", $TrackPrefix) }
    if ($DryRun) {
        Write-Host "DRY RUN [threads=$threads]: $Python $($argList -join ' ') > $log"
        continue
    }
    # Thread caps for every BLAS/OpenMP pool, inherited by this child only:
    # environment variables are read at process start, so setting them just
    # before each launch gives each cell its own value.
    $env:OMP_NUM_THREADS      = "$threads"
    $env:MKL_NUM_THREADS      = "$threads"
    $env:OPENBLAS_NUM_THREADS = "$threads"
    $p = Start-Process -FilePath $Python -ArgumentList $argList `
            -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $log -RedirectStandardError $err
    if ($Pin) {
        # Whole physical cores: logical CPUs 2k and 2k+1 are SMT siblings.
        $width = 2 * [math]::Ceiling($threads / 2)
        if ($nextCpu + $width -le $logical) {
            $mask = (([int64]1 -shl $width) - 1) -shl $nextCpu
            $p.ProcessorAffinity = [IntPtr]$mask
            $nextCpu += $width
        } else {
            Write-Warning "$($c.Name): not enough free logical CPUs to pin; left unpinned"
        }
    }
    $pidMap[$c.Name] = $p.Id
    Write-Host ("  started {0,-18} PID {1,6}  log {2}" -f $c.Name, $p.Id, $log)
    # Written after every launch, so -Status/-Stop see a partly launched grid.
    $pidMap | ConvertTo-Json | Set-Content -Path $PidFile -Encoding UTF8
    Start-Sleep -Seconds $StaggerSeconds
}

if ($DryRun) { return }

$tagArg = if ($RunTag) { " -RunTag $RunTag" } else { "" }
$first  = $cells[0].Name
Write-Host ""
Write-Host "PIDs saved to $PidFile"
Write-Host "Progress table : .\launch_grid.ps1 -Status -Seed $Seed$tagArg     (or -Watch to refresh every 30 s)"
Write-Host "Follow one log : Get-Content logs\$first.log -Tail 30 -Wait"
Write-Host "Stop one cell  : .\launch_grid.ps1 -Stop $first -Seed $Seed$tagArg"
Write-Host "Stop all       : .\launch_grid.ps1 -Stop all -Seed $Seed$tagArg"
