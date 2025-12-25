# Build helper for Windows
# Usage: open PowerShell as Administrator (if needed) and run:
#   ./build_windows.ps1

param(
    [switch]$OneFile = $true,
    [switch]$Windowed = $true,
    [string]$IconPath = ''
)

$ErrorActionPreference = 'Stop'

Write-Host "Preparing virtual environment..."
python -m venv .venv
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$venvPip = Join-Path $PSScriptRoot ".venv\Scripts\pip.exe"

if (-not (Test-Path $venvPython)) {
    throw "Virtual env python not found at: $venvPython"
}

Write-Host "Upgrading pip and installing requirements..."
& $venvPython -m pip install --upgrade pip
& $venvPip install -r requirements.txt

Write-Host "Running build_exe.py..."
$argList = @()
if ($OneFile) { $argList += '--onefile' } else { $argList += '--onedir' }
if ($Windowed) { $argList += '--windowed' } else { $argList += '--console' }
if ($IconPath -ne '') { $argList += "--icon=$IconPath" }

& $venvPython build_exe.py $argList

Write-Host "Build complete. Check the 'dist' folder for results."