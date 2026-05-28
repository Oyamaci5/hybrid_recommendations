# scikit-surprise kurulumu (.venv)
#
# Windows: PyPI kaynak derlemesi icin MSVC Build Tools gerekir.
# Surprise, numpy 1.x ile derlenir; numpy>=2 uyumsuzdur.
#
# Kullanim (proje kokunden):
#   powershell -ExecutionPolicy Bypass -File scripts/setup_surprise.ps1
#
# Sonra:
#   .\.venv\Scripts\python.exe scripts/run_surprise_kwm_baseline.py --eval-split official

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Pip = Join-Path $Root ".venv\Scripts\pip.exe"

if (-not (Test-Path $Python)) {
    throw ".venv bulunamadi. Once: python -m venv .venv"
}

Write-Host "numpy<2 kuruluyor (surprise uyumu)..."
& $Pip install "numpy<2"

Write-Host "scikit-surprise kuruluyor..."
& $Pip install scikit-surprise

Write-Host ""
& $Python -c "from surprise import KNNWithMeans; import numpy as np; print('surprise OK, numpy', np.__version__)"
Write-Host ""
Write-Host "Ornek:"
Write-Host "  .\.venv\Scripts\python.exe scripts/run_surprise_kwm_baseline.py --eval-split official"
