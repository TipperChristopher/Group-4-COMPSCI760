<#
.SYNOPSIS
    Set up a fresh clone of the final-grid branch so training is identical to the
    reference machine, then print and check the setup fingerprint.

.DESCRIPTION
    Run from anywhere, in PowerShell 7 (pwsh), on Windows:

        pwsh -ExecutionPolicy Bypass -File final_grid\setup_teammate.ps1 -Name <yourname>

    Steps, each of which stops the script with a clear error if it fails:
      1. PowerShell 7, git, the repo, branch final-grid, LF line endings
      2. Python 3.12.10 (64-bit)
      3. .venv with exactly the packages in final_grid\requirements-lock.txt
      4. the f1tenth_gym submodule at the pinned commit, installed editable
      5. the varied-width tracks: generated (tracks\make_vw_tracks.py) or verified,
         checksum-identical to tracks\manifest.json
      6. final_grid\check_setup.py: fingerprint to final_grid\fingerprint_<name>.txt,
         compared against final_grid\reference_fingerprint.txt

    Safe to run again: finished steps are verified rather than redone. A full log
    is written to logs\setup_teammate.log.

.PARAMETER Name
    Your name, letters/digits only (used in the fingerprint file name).

.PARAMETER Python
    Path to python.exe 3.12.10, if the script cannot find it by itself.
#>
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$PIN = "5a301bd0ae1ceaf7dec653e7549c8d099db58a6b"       # f1tenth/f1tenth_gym, branch v1.0.0
$PY_VERSION = "3.12.10"
$PY_URL = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
$Root = Split-Path -Parent $PSScriptRoot
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$Lock = Join-Path $Root "final_grid\requirements-lock.txt"

function Fail([string]$msg) {
    Write-Host ""
    Write-Host "SETUP FAILED: $msg" -ForegroundColor Red
    Write-Host "Nothing after this step was done. Fix the problem above and run the script again." -ForegroundColor Red
    Write-Host "Full log: $(Join-Path $Root 'logs\setup_teammate.log')"
    try { Stop-Transcript | Out-Null } catch {}
    exit 1
}
function Step([string]$msg) { Write-Host ""; Write-Host "== $msg" -ForegroundColor Cyan }
function Ok([string]$msg) { Write-Host "   OK  $msg" -ForegroundColor Green }
# Run a native command with its output (stdout and stderr) in the console and the
# log; return its exit code. Warnings on stderr do not stop the script.
function Invoke-Native([scriptblock]$cmd) {
    $eap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { & $cmd 2>&1 | ForEach-Object { "   $_" } | Out-Host } finally { $ErrorActionPreference = $eap }
    return $LASTEXITCODE
}
# Same, but fail loudly on a non-zero exit code.
function Run([string]$what, [scriptblock]$cmd) {
    $rc = Invoke-Native $cmd
    if ($rc -ne 0) { Fail "$what (exit code $rc)" }
}

if ($Name -notmatch '^[A-Za-z0-9]+$') { Write-Host "-Name must be letters/digits only, e.g. -Name alice"; exit 1 }
if ($PSVersionTable.PSVersion.Major -lt 7) {
    Write-Host "This needs PowerShell 7 (pwsh); this is Windows PowerShell $($PSVersionTable.PSVersion)." -ForegroundColor Red
    Write-Host "Install it with:  winget install --id Microsoft.PowerShell --source winget"
    Write-Host "then open 'PowerShell 7' from the Start menu and run this again."
    exit 1
}
Set-Location $Root
New-Item -ItemType Directory -Force (Join-Path $Root "logs") | Out-Null
Start-Transcript -Path (Join-Path $Root "logs\setup_teammate.log") -Force | Out-Null
Write-Host "Setting up $Root for $Name, $(Get-Date -Format 'yyyy-MM-dd HH:mm')"

# ------------------------------------------------------------------ 1. git
Step "1/6  git, branch and line endings"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Fail "git is not installed (https://git-scm.com/download/win)" }
if (-not (Test-Path (Join-Path $Root ".git"))) { Fail "$Root is not a git clone" }
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne "final-grid") { Fail "on branch '$branch'. Run: git switch final-grid   (then git pull)" }
$dirty = @(git status --porcelain --untracked-files=no | Where-Object { $_ -and $_.Substring(3) -ne "f1tenth_gym" })
if ($dirty) { Fail "tracked files are modified: $($dirty -join '; '). Undo with: git restore ." }
# LF checkout everywhere, so file hashes (and the reward-code hash train.py writes
# into every run_config.json) are identical on every machine.
if ((git config core.autocrlf) -ne "false") {
    git config core.autocrlf false
    Run "re-checking out with LF line endings" { git rm -r -q --cached . ; git reset -q --hard }
    Ok "set core.autocrlf false for this clone and re-checked out the files with LF endings"
}
$crlf = @(git ls-files --eol train.py sb3_wrapper.py | Where-Object { $_ -match 'w/crlf' })
if ($crlf) { Fail "files still have CRLF line endings: $crlf" }
Ok "branch final-grid, clean, LF line endings, commit $((git rev-parse --short HEAD).Trim())"

# ---------------------------------------------------------------- 2. python
Step "2/6  Python $PY_VERSION"
function Get-PyVersion([string]$exe) {
    try { $v = & $exe -c "import sys, struct; print('%d.%d.%d-%d' % (*sys.version_info[:3], struct.calcsize('P') * 8))" 2>$null }
    catch { return $null }
    if ($LASTEXITCODE -ne 0) { return $null }
    return "$v".Trim()
}
$want = "$PY_VERSION-64"
if (-not $Python) {
    $cands = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $p = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $p) { $cands += "$p".Trim() }
    }
    $cands += @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe", "C:\Program Files\Python312\python.exe")
    if (Get-Command python -ErrorAction SilentlyContinue) { $cands += (Get-Command python).Source }
    $Python = $cands | Where-Object { $_ -and (Test-Path $_) -and ((Get-PyVersion $_) -eq $want) } | Select-Object -First 1
    if (-not $Python) {
        Fail ("Python $PY_VERSION (64-bit) not found. Install exactly this version from`n   $PY_URL`n" +
              "   (tick 'Add python.exe to PATH' is optional), then run this script again,`n" +
              "   or pass its path: -Python C:\path\to\python.exe")
    }
}
$got = Get-PyVersion $Python
if ($got -ne $want) { Fail "$Python is Python $got, need $want. Install $PY_URL" }
Ok "$Python"

# ------------------------------------------------------------------ 3. venv
Step "3/6  virtual environment .venv from the lock file"
if (Test-Path $VenvPy) {
    $vgot = Get-PyVersion $VenvPy
    if ($vgot -ne $want) { Fail ".venv exists but is Python $vgot. Delete the .venv folder and run again." }
    Ok ".venv already exists (Python $vgot)"
} else {
    Run "creating .venv" { & $Python -m venv (Join-Path $Root ".venv") }
    Ok "created .venv"
}
Run "installing pip 26.2.1" { & $VenvPy -m pip install --disable-pip-version-check -q "pip==26.2.1" }
# --no-deps: the lock lists every package; see the comment at the top of the lock file.
Run "installing the locked packages (torch is about 200 MB to download)" {
    & $VenvPy -m pip install --disable-pip-version-check --no-deps -r $Lock }
Ok "installed final_grid\requirements-lock.txt"

# ------------------------------------------------------------- 4. f1tenth_gym
Step "4/6  f1tenth_gym submodule at $($PIN.Substring(0, 12))"
Run "git submodule sync" { git submodule sync -q }
Run "git submodule update (clones f1tenth_gym from GitHub)" { git submodule update --init }
$head = (git -C f1tenth_gym rev-parse HEAD).Trim()
if ($head -ne $PIN) { Fail "f1tenth_gym is at $head, expected $PIN. Run: git pull, then this script again." }
$subDirty = git -C f1tenth_gym status --porcelain
if ($subDirty) { Fail "f1tenth_gym has local changes:`n$subDirty`nUndo with: git -C f1tenth_gym restore ." }
Run "installing f1tenth_gym (editable)" {
    & $VenvPy -m pip install --disable-pip-version-check -q --no-deps -e (Join-Path $Root "f1tenth_gym") }
$where = & $VenvPy -c "import importlib.util as u; print(u.find_spec('f1tenth_gym').origin)"
if ("$where" -notlike "$(Join-Path $Root 'f1tenth_gym')*") { Fail "f1tenth_gym imports from $where, not from this clone" }
Ok "f1tenth_gym $($PIN.Substring(0, 12)), clean, editable from .\f1tenth_gym"

# ---------------------------------------------------------------- 5. tracks
Step "5/6  varied-width tracks (vw_train, vw_val)"
# Temporary renders (about 1.5 GB) go to the repo's drive, not the system drive.
$tmp = Join-Path $Root "logs\setup_tmp"
New-Item -ItemType Directory -Force $tmp | Out-Null
$oldTemp, $oldTmp = $env:TEMP, $env:TMP
$env:TEMP = $tmp; $env:TMP = $tmp; $env:MPLBACKEND = "Agg"
$verify = "import sys; sys.argv=['x']; sys.path.insert(0, r'$Root\final_grid'); import check_setup as c; " +
          "L=[]; ok=c.check_tracks(L); print('; '.join(f'{k} {v}' for k, v in L)); sys.exit(0 if ok else 1)"
& $VenvPy -c $verify *> $null      # quiet probe: are the tracks already there and correct?
if ($LASTEXITCODE -eq 0) {
    Ok "all tracks already present and identical to tracks\manifest.json"
} else {
    Write-Host "   generating all 130 varied-width tracks (vw_train, vw_val, vw_test): about 5-15 minutes (the tracks were not there yet; this is expected) ..."
    Run "tracks\make_vw_tracks.py" { & $VenvPy (Join-Path $Root "tracks\make_vw_tracks.py") }
    # The generator rewrites the vw entries of the manifest; they must come out identical.
    git diff --quiet -- tracks/manifest.json
    if ($LASTEXITCODE -ne 0) {
        git restore tracks/manifest.json
        Fail "track generation produced a different tracks\manifest.json (restored). Send logs\setup_teammate.log to Desmond."
    }
    if ((Invoke-Native { & $VenvPy -c $verify }) -ne 0) { Fail "generated tracks do not match tracks\manifest.json" }
    Ok "generated; every vw_train and vw_val file matches tracks\manifest.json"
}
$env:TEMP, $env:TMP = $oldTemp, $oldTmp
Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue

# ----------------------------------------------------------- 6. fingerprint
Step "6/6  fingerprint and comparison with the reference machine (1-3 minutes)"
$fp = Join-Path $Root "final_grid\fingerprint_$Name.txt"
$checkOk = (Invoke-Native { & $VenvPy (Join-Path $Root "final_grid\check_setup.py") --out $fp }) -eq 0
Write-Host ""
$match = (Invoke-Native { & $VenvPy (Join-Path $Root "final_grid\check_setup.py") --compare $fp }) -eq 0
Write-Host ""
if ($checkOk -and $match) {
    Write-Host "SETUP COMPLETE: this machine MATCHES the reference." -ForegroundColor Green
} elseif ($checkOk) {
    Write-Host "SETUP COMPLETE, but the fingerprint differs from the reference (see above)." -ForegroundColor Yellow
    Write-Host "Do not start training. Send $fp to Desmond."
} else {
    Fail "check_setup.py found problems (see PROBLEMS above). Send $fp to Desmond."
}
Write-Host "Send this file to Desmond: $fp"
try { Stop-Transcript | Out-Null } catch {}
