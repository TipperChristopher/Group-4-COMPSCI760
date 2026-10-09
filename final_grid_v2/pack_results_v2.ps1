<#
.SYNOPSIS
    Zip one person's finished final-grid-v2 runs for sending back: share\final_v2_<name>.zip

.DESCRIPTION
    Reads the queue state of final_grid_v2\jobs_v2_<name>.txt
    (logs\queue_final_jobs_<name>\state.json) and packs, for every job in it:
      models\<run>\            everything: run_config.json, progress.csv, monitor CSVs,
                               all checkpoints with their VecNormalize statistics,
                               final_model.zip and vecnormalize.pkl
      <results>\<tag>_*_vw_val.csv   the vw_val evaluation of every checkpoint
      logs\<tag>*.log          training and evaluation logs
    plus the queue folder (state, scheduler log, summary), the results summaries,
    final_grid_v2\fingerprint_<name>.txt and logs\setup_teammate.log, and a
    pack_manifest_<name>.csv listing every file with its size and sha256.
    Paths inside the zip are relative to the repo root, so all bundles can be
    unzipped into one clone side by side (run folders, tags and result folders
    all differ between people).

    Expected size (each SAC run about 90 MB, each PPO run about 7 MB; the
    checkpoints are already compressed):
      chris    12 PPO runs   about  85 MB
      grant     4 SAC runs   about 360 MB
      desmond   8 SAC runs   about 720 MB

.EXAMPLE
    pwsh -File final_grid_v2\pack_results_v2.ps1 -Name grant
#>
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$OutDir = "share",
    # Queue to pack; default jobs_v2_<Name>. Only needed for testing.
    [string]$Queue = "",
    # Pack even though the queue has not finished (e.g. to send a partial result).
    [switch]$Force
)

$ErrorActionPreference = "Stop"
if ($Name -notmatch '^[A-Za-z0-9]+$') { throw "-Name must be letters/digits only" }
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (-not $Queue) { $Queue = "jobs_v2_$Name" }
$StatePath = Join-Path $Root "logs\queue_$Queue\state.json"
if (-not (Test-Path $StatePath)) { throw "No queue state at $StatePath. Was the queue started with final_grid_v2\$Queue.txt?" }
$s = Get-Content $StatePath -Raw | ConvertFrom-Json -AsHashtable

# ------------------------------------------------------------ completeness
$problems = @()
if ($s.phase -ne "finished") { $problems += "queue phase is '$($s.phase)', not 'finished'" }
foreach ($j in $s.jobs) {
    if ($j.status -ne "evaluated") { $problems += "$($j.tag): $($j.status)" }
    $bad = @($j.evals | Where-Object { $_.status -ne "done" })
    if ($bad) { $problems += "$($j.tag): evaluations not done: $(($bad | ForEach-Object { "$($_.label)=$($_.status)" }) -join ', ')" }
}
if ($problems) {
    Write-Host "Not everything has finished:" -ForegroundColor Yellow
    $problems | ForEach-Object { Write-Host "  $_" }
    if (-not $Force) { throw "Wait for the queue to finish (launch_queue.ps1 -Status), or pass -Force to pack anyway." }
}

# ------------------------------------------------------------ collect files
$files = [System.Collections.Generic.List[string]]::new()
function Add-Files($items) { foreach ($f in @($items)) { if ($f -and (Test-Path -LiteralPath $f -PathType Leaf)) { $files.Add((Resolve-Path -LiteralPath $f).Path) } } }
$results = Join-Path $Root $s.results_dir
foreach ($j in $s.jobs) {
    $run = Join-Path $Root $j.run_dir
    if (Test-Path $run) { Add-Files (Get-ChildItem -LiteralPath $run -File -Recurse).FullName }
    else { Write-Host "  missing run folder $($j.run_dir)" -ForegroundColor Yellow }
    Add-Files (Get-ChildItem -LiteralPath $results -File -Filter "$($j.tag)_*_vw_val.csv" -ErrorAction SilentlyContinue).FullName
    Add-Files (Get-ChildItem -LiteralPath (Join-Path $Root "logs") -File -Filter "$($j.tag)*.log").FullName
}
Add-Files (Get-ChildItem -LiteralPath (Join-Path $Root "logs\queue_$Queue") -File).FullName
Add-Files (Get-ChildItem -LiteralPath $results -File -Filter "summary*.csv" -ErrorAction SilentlyContinue).FullName
Add-Files (Join-Path $Root "final_grid_v2\fingerprint_$Name.txt")
$setupLog = Join-Path $Root "logs\setup_teammate.log"
$files = [System.Collections.Generic.List[string]]@($files | Select-Object -Unique)

# --------------------------------------------------------------------- zip
$out = if ([IO.Path]::IsPathRooted($OutDir)) { $OutDir } else { Join-Path $Root $OutDir }
New-Item -ItemType Directory -Force $out | Out-Null
$zipPath = Join-Path $out "final_v2_$Name.zip"
if (Test-Path $zipPath) { throw "$zipPath already exists. Move or delete it first." }
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
$rootPrefix = $Root.TrimEnd('\') + '\'
function Rel($p) { $p.Substring($rootPrefix.Length).Replace('\', '/') }

$manifest = [System.Text.StringBuilder]::new("path,bytes,sha256`n")
$zip = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    $n = 0
    foreach ($f in $files) {
        $entry = Rel $f
        # SB3 model files are zips already; recompressing them only costs time.
        $level = if ($f.EndsWith(".zip")) { [System.IO.Compression.CompressionLevel]::NoCompression }
                 else { [System.IO.Compression.CompressionLevel]::Optimal }
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $f, $entry, $level) | Out-Null
        $h = (Get-FileHash -LiteralPath $f -Algorithm SHA256).Hash.ToLower()
        [void]$manifest.Append("$entry,$((Get-Item -LiteralPath $f).Length),$h`n")
        $n++
        if ($n % 50 -eq 0) { Write-Host "  $n / $($files.Count) files" }
    }
    if (Test-Path $setupLog) {
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $setupLog, "teammate/$Name/setup_teammate.log") | Out-Null
    }
    $me = $zip.CreateEntry("pack_manifest_$Name.csv")
    $w = [System.IO.StreamWriter]::new($me.Open()); $w.Write($manifest.ToString()); $w.Dispose()
} finally { $zip.Dispose() }

# ------------------------------------------------------------------ report
$check = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
$entries = $check.Entries.Count; $check.Dispose()
$runs = foreach ($j in $s.jobs) {
    $cfgPath = Join-Path $Root (Join-Path $j.run_dir "run_config.json")
    $cfg = if (Test-Path $cfgPath) { Get-Content $cfgPath -Raw | ConvertFrom-Json } else { $null }
    $ck = @(Get-ChildItem -LiteralPath (Join-Path $Root $j.run_dir) -Filter "*_checkpoint_*_steps.zip" -ErrorAction SilentlyContinue).Count
    [pscustomobject]@{ Tag = $j.tag; Status = $j.status; Steps = $cfg.num_timesteps
                       Hours = $(if ($cfg.wall_seconds) { [math]::Round($cfg.wall_seconds / 3600, 2) } else { "" })
                       Checkpoints = $ck; "vw_val evals" = @($j.evals | Where-Object { $_.status -eq "done" }).Count }
}
$runs | Format-Table -AutoSize | Out-String -Width 160 | Write-Host
$mb = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
Write-Host "Written $zipPath" -ForegroundColor Green
Write-Host "  $($files.Count) files from $($s.jobs.Count) runs, $entries zip entries, $mb MB"
Write-Host "Upload this file as described in final_grid_v2\TEAMMATE_V2.txt."
