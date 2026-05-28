# Paper-style WNMF-50 assignments (K=27, K=30) + cluster KNNBaseline eval.
# PCA50 klasörleriyle aynı protokol: colzscore, euclidean, random init, gray sheep kapalı.
#
#   powershell -ExecutionPolicy Bypass -File .\mealpy\run_k27_k30_paper_wnmf_knnbasic.ps1
#   powershell -ExecutionPolicy Bypass -File .\mealpy\run_k27_k30_paper_wnmf_knnbasic.ps1 -Phase eval

param(
    [ValidateSet("all", "assign", "eval")]
    [string]$Phase = "all",
    [int[]]$Knn = @(40),
    [int]$Jobs = 4
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $RepoRoot

$AssignRoot = "mealpy\results\assignments"

# Mevcut PCA50 paper assignment setleriyle aynı algoritma listesi
$AlgosK27 = @(
    "B0_KMEANS", "B1_HHO", "B_AVOA", "HA_AVOAHGS", "IWO_HHO", "LIT_GOA", "LIT_GWO"
)
$AlgosK30 = @(
    "B0_KMEANS", "B2_HGS", "B3_MFO", "H4_MFO+HHO", "HA_AVOAHGS", "IWO_HHO", "LIT_GOA"
)

# generate_assignments: {label}{out_suffix}{assign_suffix}
$SuffixK27 = "_colzscore_euc_irand_paper_none_wnmf50_k27_kmref"
$SuffixK30 = "_colzscore_euc_irand_paper_trainonly_official_none_wnmf50_k30_pwcss_kmref"

function Invoke-AssignK27 {
    Write-Host "=== Assign K=27 (paper WNMF-50) ===" -ForegroundColor Cyan
    python -u mealpy\generate_assignments.py `
        --dataset 100k `
        --paper-mode `
        --feature-extraction wnmf `
        --svd-components 50 `
        --k 27 `
        --algo @AlgosK27 `
        --jobs $Jobs `
        --skip-existing
}

function Invoke-AssignK30 {
    Write-Host "=== Assign K=30 (paper WNMF-50, train-only official) ===" -ForegroundColor Cyan
    python -u mealpy\generate_assignments.py `
        --dataset 100k `
        --paper-mode `
        --feature-extraction wnmf `
        --svd-components 50 `
        --k 30 `
        --train-only `
        --eval-split official `
        --fold 1 `
        --cluster-objective wcss `
        --algo @AlgosK30 `
        --jobs $Jobs `
        --skip-existing
}

function Invoke-EvalK27 {
    Write-Host "=== Eval K=27 | Cluster KNNBaseline ===" -ForegroundColor Cyan
    $knnArg = @("--knn") + ($Knn | ForEach-Object { "$_" })
    python -u wnmf\wnmf_experiment.py `
        --dataset 100k `
        --eval-split random `
        --fold 1 `
        --mode baselines `
        --no-global `
        --no-cluster-avg `
        --cluster-knn-variant baseline `
        --cluster-knn-backend surprise `
        --similarity pearson `
        @knnArg `
        --k 27 `
        --algo @AlgosK27 `
        --assign-root $AssignRoot `
        --assign-suffix $SuffixK27 `
        --top-n 10 `
        --relevance-threshold 4.0 `
        --algo-jobs 3
}

function Invoke-EvalK30 {
    Write-Host "=== Eval K=30 | Cluster KNNBaseline (official split) ===" -ForegroundColor Cyan
    $knnArg = @("--knn") + ($Knn | ForEach-Object { "$_" })
    python -u wnmf\wnmf_experiment.py `
        --dataset 100k `
        --eval-split official `
        --fold 1 `
        --mode baselines `
        --no-global `
        --no-cluster-avg `
        --cluster-knn-variant baseline `
        --cluster-knn-backend surprise `
        --similarity pearson `
        @knnArg `
        --k 30 `
        --algo @AlgosK30 `
        --assign-root $AssignRoot `
        --assign-suffix $SuffixK30 `
        --top-n 10 `
        --relevance-threshold 4.0 `
        --algo-jobs 3
}

switch ($Phase) {
    "assign" {
        Invoke-AssignK27
        Invoke-AssignK30
    }
    "eval" {
        Invoke-EvalK27
        Invoke-EvalK30
    }
    "all" {
        Invoke-AssignK27
        Invoke-AssignK30
        Invoke-EvalK27
        Invoke-EvalK30
    }
}

Write-Host "Bitti." -ForegroundColor Green
