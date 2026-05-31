# Final reproducible re-run of every experiment used in the thesis.
# All scripts use set_seed(42) for bit-stable RNG state; the wrapper just
# chains them, logs to one file, and backs up any pre-existing results.
#
# Usage (from project root):
#   .\scripts\run_all_final.ps1
#
# Will:
#   - back up   outputs/results/                  -> outputs/results_backup_<ts>/
#   - log to    outputs/logs/final_run_<ts>.log
#   - continue past per-step failures (so a single broken script does not
#     tank the rest of the matrix); failures are summarised at the end.

$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED  = "1"   # so Tee captures progress as it streams

$startedAt = Get-Date
$ts        = $startedAt.ToString("yyyyMMdd_HHmmss")

# --- venv -------------------------------------------------------------------
if (-not $env:VIRTUAL_ENV) {
    & .\.venv\Scripts\Activate.ps1
}
Write-Host "venv:          $env:VIRTUAL_ENV"
Write-Host "python:        $(python -c 'import sys; print(sys.version.split()[0])')"
Write-Host "torch / cuda:  $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
Write-Host ""

# --- backup -----------------------------------------------------------------
$backupDir = "outputs/results_backup_$ts"
if (Test-Path "outputs/results") {
    Copy-Item -Recurse -Path "outputs/results" -Destination $backupDir
    Write-Host "Backup created: $backupDir"
}

New-Item -ItemType Directory -Force -Path "outputs/logs"    | Out-Null
New-Item -ItemType Directory -Force -Path "outputs/results" | Out-Null
$logFile = "outputs/logs/final_run_$ts.log"
"=== final run started $startedAt ===" | Out-File $logFile -Encoding utf8

# --- step runner ------------------------------------------------------------
$script:results  = @()
$script:failures = @()

function Run-Step {
    param(
        [string]$Name,
        [string]$Cmd
    )

    $banner = "=" * 76
    Write-Host ""
    Write-Host $banner
    Write-Host "  [$(Get-Date -Format HH:mm:ss)] $Name"
    Write-Host "  > $Cmd"
    Write-Host $banner
    Add-Content $logFile "`r`n$banner"
    Add-Content $logFile "[$(Get-Date -Format HH:mm:ss)] $Name"
    Add-Content $logFile "> $Cmd"
    Add-Content $logFile $banner

    $stepStart = Get-Date
    try {
        # Run + tee
        Invoke-Expression $Cmd 2>&1 | Tee-Object -FilePath $logFile -Append
        $code = $LASTEXITCODE
    } catch {
        $code = 1
        Write-Host "ERROR: $($_.Exception.Message)"
        Add-Content $logFile "ERROR: $($_.Exception.Message)"
    }
    $duration = (Get-Date) - $stepStart

    $row = [PSCustomObject]@{
        Step     = $Name
        ExitCode = $code
        Duration = $duration.ToString("hh\:mm\:ss")
    }
    $script:results += $row
    if ($code -ne 0) { $script:failures += $Name }

    Write-Host ""
    Write-Host ("  -> {0}  (exit={1})  elapsed {2}" -f $Name, $code, $row.Duration)
    Add-Content $logFile ("  -> {0}  (exit={1})  elapsed {2}" -f $Name, $code, $row.Duration)
}

# ============================================================================
#  IN-DISTRIBUTION (Drone-Audio-Dataset test split, 410 clips)
# ============================================================================
Run-Step "in-dist  / 01 / CNN14 clean baseline"      "python scripts\evaluate_baseline.py"
Run-Step "in-dist  / 02 / FGSM (5 epsilons)"          "python scripts\run_fgsm.py"
Run-Step "in-dist  / 03 / PGD  (5 epsilons, 40 step)" "python scripts\run_pgd.py"
Run-Step "in-dist  / 04 / EOT-PGD (2 eps, 20 step)"   "python scripts\run_eot_pgd.py"
Run-Step "in-dist  / 05 / baselines (jam + spoof)"    "python scripts\run_baselines.py"
Run-Step "in-dist  / 06 / black-box transfer"         "python scripts\run_blackbox_transfer.py"

# ============================================================================
#  CROSS-DATASET (SWARM, 3 556 clips)
# ============================================================================
Run-Step "cross    / 07 / clean evaluation (CNN14+Proxy)" "python scripts\evaluate_cross_dataset.py"
Run-Step "cross    / 08 / FGSM + PGD (5 eps each)"        "python scripts\run_cross_dataset_attacks.py --attacks FGSM,PGD --epsilons 0.001,0.005,0.01,0.02,0.05"
Run-Step "cross    / 09 / baselines (jam + spoof)"        "python scripts\run_cross_dataset_baselines.py"
Run-Step "cross    / 10 / FGSM transfer"                  "python scripts\run_cross_dataset_transfer.py"

# ============================================================================
#  SUMMARY
# ============================================================================
$totalDur = (Get-Date) - $startedAt
$summary  = @()
$summary += ""
$summary += ("=" * 76)
$summary += "  ALL DONE   total elapsed: " + $totalDur.ToString("hh\:mm\:ss")
$summary += "  log:       $logFile"
$summary += "  backup:    $backupDir"
$summary += ("=" * 76)

$summary  | ForEach-Object { Write-Host $_ ; Add-Content $logFile $_ }
$results  | Format-Table -AutoSize | Out-String | Tee-Object -FilePath $logFile -Append

if ($failures.Count -gt 0) {
    Write-Host ""
    Write-Host "FAILED STEPS:" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host "  - $_" }
    Add-Content $logFile "`r`nFAILED STEPS:"
    $failures | ForEach-Object { Add-Content $logFile "  - $_" }
    exit 1
}

Write-Host ""
Write-Host "All steps succeeded. Refresh figures next."
exit 0
