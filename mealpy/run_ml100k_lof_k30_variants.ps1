# ML-100K, LOF, K=30 — preprocessing varyantlari (H9_QSA+CDO yok).
# ExecutionPolicy hatasi alirsaniz:
#   powershell -ExecutionPolicy Bypass -File .\run_ml100k_lof_k30_variants.ps1
# veya imzasiz .ps1 engelini atlamak icin: .\run_ml100k_lof_k30_variants.bat

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$algo = @(
    'B0_KMEANS', 'B1_HHO', 'B2_HGS', 'H1_HHO+HGS', 'H4_MFO+HHO',
    'LIT_GOA', 'LIT_GWO', 'LIT_SSA', 'H12_MFO+CDO', 'H13_HHO+GAop'
)

$runs = @(
    @('--lof', '--dataset', '100k', '--algo') + $algo + @('--k', '30', '--zscore'),
    @('--lof', '--dataset', '100k', '--algo') + $algo + @('--k', '30', '--wnmf-features', '20'),
    @('--lof', '--dataset', '100k', '--algo') + $algo + @('--k', '30', '--zscore', '--wnmf-features', '20'),
    @('--lof', '--dataset', '100k', '--algo') + $algo + @('--k', '30', '--pca', '0.80'),
    @('--lof', '--dataset', '100k', '--algo') + $algo + @('--k', '30', '--zscore', '--pca', '0.80')
)

$n = 0
foreach ($argv in $runs) {
    $n++
    Write-Host "`n========== Kosu $n / $($runs.Count) ==========" -ForegroundColor Cyan
    & python generate_assignments.py @argv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "HATA: Kosu $n cikis kodu $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "`nTamamlandi: $($runs.Count) kosu." -ForegroundColor Green
