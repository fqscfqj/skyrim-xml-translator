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
.\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip and installing requirements..."
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Running build_exe.py..."
$argList = @()
if ($OneFile) { $argList += '--onefile' } else { $argList += '--onedir' }
if ($Windowed) { $argList += '--windowed' } else { $argList += '--console' }
if ($IconPath -ne '') { $argList += "--icon=$IconPath" }

python build_exe.py $argList

Write-Host "Build complete. Check the 'dist' folder for results."