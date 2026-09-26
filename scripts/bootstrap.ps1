param([string]$PythonExe = "")
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (Test-Path $VenvPython) {
    & $VenvPython -c "import sys; print(sys.version); assert sys.version_info[:2] == (3, 11), 'Expected Python 3.11; preserve this environment and select a compatible one explicitly.'"
    if ($LASTEXITCODE -ne 0) { throw "Existing environment requires review; it was not changed." }
    Write-Host "Reusing $VenvPython"
    exit 0
}
if (Test-Path $VenvPath) { throw "Incomplete or non-Windows .venv exists. Select a clean target folder; no files were removed." }
if ($PythonExe) {
    $Runtime = $PythonExe
    $RuntimeArgs = @()
} else {
    $Runtime = "py"
    $RuntimeArgs = @("-3.11")
}
& $Runtime @RuntimeArgs -c "import sys, struct; assert sys.version_info[:2] == (3, 11) and struct.calcsize('P') == 8, 'Use 64-bit Python 3.11'"
if ($LASTEXITCODE -ne 0) { throw "A 64-bit Python 3.11 runtime is required for this bootstrap." }
& $Runtime @RuntimeArgs -m venv $VenvPath
if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
Write-Host "Created $VenvPython. No research dependencies were installed."
Write-Host "Next: & '$VenvPython' '$PSScriptRoot\inspect_environment.py'"
