# Build helper for Windows
# Usage: open PowerShell as Administrator (if needed) and run:
#   ./build_windows.ps1

param(
    [bool]$OneFile = $true,
    [bool]$Windowed = $true,
    [string]$IconPath = 'assets\logo.ico',
    [switch]$Incremental,
    [switch]$RecreateVenv,
    [switch]$UpgradePip,
    [switch]$SkipDependencyInstall,
    [int]$PipRetries = 5,
    [int]$PipTimeoutSec = 60,
    [string]$PipCacheDir = '.pip-cache',
    [switch]$NoPause
)

$ErrorActionPreference = 'Stop'

function Invoke-StepWithRetry {
    param(
        [string]$Name,
        [scriptblock]$Action,
        [int]$Attempts = 3,
        [int]$DelaySeconds = 3,
        [switch]$NonBlocking
    )

    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            & $Action
            if ($LASTEXITCODE -eq 0) {
                return $true
            }
            throw "Command exited with code $LASTEXITCODE"
        }
        catch {
            if ($attempt -ge $Attempts) {
                if ($NonBlocking) {
                    Write-Warning "$Name failed after $Attempts attempts: $($_.Exception.Message)"
                    return $false
                }
                throw "$Name failed after $Attempts attempts: $($_.Exception.Message)"
            }
            Write-Warning "$Name failed (attempt $attempt/$Attempts): $($_.Exception.Message). Retrying in $DelaySeconds s..."
            Start-Sleep -Seconds $DelaySeconds
        }
    }
}

try {
    Push-Location $PSScriptRoot

    $venvRoot = Join-Path $PSScriptRoot '.venv'
    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

    if ($RecreateVenv -and (Test-Path $venvRoot)) {
        Write-Host "Removing existing virtual environment..."
        Remove-Item -Path $venvRoot -Recurse -Force
    }

    if (-not (Test-Path $venvPython)) {
        Write-Host "Preparing virtual environment..."
        python -m venv .venv
    }
    else {
        Write-Host "Reusing existing virtual environment..."
    }

    if (-not (Test-Path $venvPython)) {
        throw "Virtual env python not found at: $venvPython"
    }

    $requirementsPath = Join-Path $PSScriptRoot 'requirements.txt'
    $requirementsHash = (Get-FileHash -Path $requirementsPath -Algorithm SHA256).Hash
    $requirementsStamp = Join-Path $venvRoot '.requirements.sha256'
    $installDeps = -not $SkipDependencyInstall

    if ($installDeps -and (Test-Path $requirementsStamp)) {
        $cachedHash = (Get-Content -Path $requirementsStamp -Raw).Trim()
        if ($cachedHash -eq $requirementsHash) {
            Write-Host "requirements.txt unchanged; skipping dependency install."
            $installDeps = $false
        }
    }

    $pipArgs = @(
        '--disable-pip-version-check',
        '--retries', $PipRetries,
        '--timeout', $PipTimeoutSec
    )

    if ($PipCacheDir -ne '') {
        $cachePath = Join-Path $PSScriptRoot $PipCacheDir
        if (-not (Test-Path $cachePath)) {
            New-Item -Path $cachePath -ItemType Directory | Out-Null
        }
        $pipArgs += @('--cache-dir', $cachePath)
    }

    if ($UpgradePip) {
        Write-Host "Upgrading pip (optional)..."
        Invoke-StepWithRetry -Name "pip upgrade" -Attempts 2 -DelaySeconds 2 -NonBlocking -Action {
            & $venvPython -m pip install --upgrade pip @pipArgs
        } | Out-Null
    }

    if ($installDeps) {
        Write-Host "Installing requirements..."
        Invoke-StepWithRetry -Name "requirements install" -Attempts 3 -DelaySeconds 4 -Action {
            & $venvPython -m pip install @pipArgs -r requirements.txt
        } | Out-Null
        Set-Content -Path $requirementsStamp -Value $requirementsHash -NoNewline
    }

    Write-Host "Running build_exe.py..."
    $argList = @()
    if ($OneFile) { $argList += '--onefile' } else { $argList += '--onedir' }
    if ($Windowed) { $argList += '--windowed' } else { $argList += '--console' }
    if ($Incremental) { $argList += '--incremental' }
    if ($IconPath -ne '') { $argList += "--icon=$IconPath" }

    & $venvPython build_exe.py $argList

    Write-Host "Build complete. Check the 'dist' folder for results."
}
catch {
    Write-Host "Build failed: $($_.Exception.Message)" -ForegroundColor Red
    throw
}
finally {
    Pop-Location
    if (-not $NoPause) {
        Write-Host ""
        Read-Host "Press Enter to exit"
    }
}
