# Runs PsychoPy's bundled Python from the project root, whatever your cwd is.
# Usage (PowerShell):   .\scripts\py.ps1 -m pytest -q
#                       .\scripts\py.ps1 -m src.experiment.run_experiment
#                       .\scripts\py.ps1 tests/ai-tests/run_sim.py --timeline
# Finds the interpreter via $env:PSYCHOPY_PYTHON, then the usual install
# locations. (Plain `python` will NOT work: only PsychoPy's has the right deps.)
$root = Split-Path -Parent $PSScriptRoot
$candidates = @(@(
    $env:PSYCHOPY_PYTHON,
    "$env:LOCALAPPDATA\Programs\PsychoPy\python.exe",
    "C:\Program Files\PsychoPy\python.exe",
    "C:\Program Files (x86)\PsychoPy\python.exe"
) | Where-Object { $_ -and (Test-Path $_) })
if (-not $candidates) {
    Write-Error "PsychoPy's python.exe not found. Install PsychoPy Standalone or set `$env:PSYCHOPY_PYTHON."
    exit 1
}
Set-Location $root
& $candidates[0] @args
exit $LASTEXITCODE
