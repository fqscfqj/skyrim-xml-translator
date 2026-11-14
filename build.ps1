$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot 'env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Missing project env. Run: python -m venv env; env\Scripts\python.exe -m pip install -r build-requirements.lock'
}

Push-Location $PSScriptRoot
try {
    & $python (Join-Path $PSScriptRoot 'build_exe.py') @args
    $buildExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $buildExitCode
