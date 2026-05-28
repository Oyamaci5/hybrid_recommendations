# kNN fine sweep: K=6,7 x knn=21..25 (cluster_knn, fold 5)
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\mealpy\run_knn_fine_sweep.ps1
#   powershell -ExecutionPolicy Bypass -File .\mealpy\run_knn_fine_sweep.ps1 -Fold 5 -K 6 7 -Knn 21 22 23 24 25

param(
    [int]$Fold = 5,
    [int[]]$K = @(6, 7),
    [int[]]$Knn = @(21, 22, 23, 24, 25),
    [string[]]$Algo = @("HA_AVOAHGS")
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $RepoRoot

$AssignRoot = "mealpy\results\assignments_lof"
$BaseSuffix = "_pruneu5_i10_zscore_euc_imkpp_none_wnmf30"

foreach ($k in $K) {
    $suffix = "${BaseSuffix}_k$k"
    Write-Host "=== K=$k  kNN=$($Knn -join ',') ===" -ForegroundColor Cyan
    python wnmf\wnmf_experiment.py `
        --dataset 100k `
        --eval-split random `
        --fold $Fold `
        --assign-root $AssignRoot `
        --assign-suffix $suffix `
        --mode baselines `
        --no-global `
        --no-cluster-avg `
        --k $k `
        --knn @Knn `
        --similarity pearson `
        --expand-knn `
        --sig-weight 10 `
        --algo @Algo
}

Write-Host "Bitti. Sonuclar: results\wnmf\ml100k\k{K}\fold$Fold\run*" -ForegroundColor Green
