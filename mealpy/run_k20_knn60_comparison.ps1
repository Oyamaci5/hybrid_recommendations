# K=20, kNN=60 champion karsilastirma pipeline
# WNMF30 + mkpp (HA_AVOAHGS ayarları); B0_KMEANS + global SVD baseline dahil
#
# powershell -ExecutionPolicy Bypass -File .\mealpy\run_k20_knn60_comparison.ps1 -Phase all
# powershell -ExecutionPolicy Bypass -File .\mealpy\run_k20_knn60_comparison.ps1 -Phase assignment
# powershell -ExecutionPolicy Bypass -File .\mealpy\run_k20_knn60_comparison.ps1 -Phase eval

param(
    [ValidateSet("assignment", "assignment-pca", "export-db", "eval", "eval-pca", "compare", "all")]
    [string]$Phase = "all",
    [int]$Fold = 5,
    [int]$K = 20,
    [int]$Knn = 60,
    [ValidateSet("best_wcss", "latest", "worst_wcss")]
    [string]$DbStrategy = "best_wcss"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $RepoRoot

$AssignRoot = "mealpy\results\assignments_lof"
$WnmfSuffix = "_pruneu5_i10_zscore_euc_imkpp_none_wnmf30_k$K"
$PcaSuffix  = "_pruneu5_i10_zscore_pca95pct_euc_imkpp_none_none20_k$K"

$CompareAlgos = @("B0_KMEANS", "LIT_GWO", "IWO_HHO", "H4_MFO+HHO", "B2_HGS", "LIT_PSO", "HA_AVOAHGS")
$AllEvalAlgos = $CompareAlgos

function Invoke-Wnmf30Assignment {
    param([string[]]$Algos)
    python mealpy\generate_assignments.py `
        --dataset 100k `
        --lof `
        --zscore `
        --preprocess none `
        --feature-extraction wnmf `
        --svd-components 30 `
        --cluster-metric euclidean `
        --init-mode mkpp `
        --k $K `
        --skip-existing `
        --algo @Algos
}

function Invoke-PcaAssignment {
    python mealpy\generate_assignments.py `
        --dataset 100k `
        --lof `
        --zscore `
        --preprocess none `
        --feature-extraction none `
        --pca 0.95 `
        --cluster-metric euclidean `
        --init-mode mkpp `
        --k $K `
        --skip-existing `
        --algo HA_AVOAHGS
}

function Invoke-ExportAll {
    param([string]$Suffix, [string[]]$Algos)
    foreach ($a in $Algos) {
        $outDir = Join-Path $AssignRoot "ml100k\${a}${Suffix}"
        python assignment_db.py export ml100k $a $K $Suffix $outDir $DbStrategy
    }
}

function Invoke-WnmfEval {
    param(
        [string]$Suffix,
        [string[]]$Algos,
        [switch]$IncludeSvdBaseline
    )
    if ($IncludeSvdBaseline) {
        python wnmf\wnmf_experiment.py `
            --dataset 100k `
            --eval-split random `
            --fold $Fold `
            --assign-root $AssignRoot `
            --assign-suffix $Suffix `
            --assign-from-db `
            --assign-db-strategy $DbStrategy `
            --assign-db-overwrite `
            --mode baselines `
            --svd `
            --no-cluster-avg `
            --k $K `
            --knn $Knn `
            --similarity pearson `
            --expand-knn `
            --sig-weight 10 `
            --algo @Algos
    } else {
        python wnmf\wnmf_experiment.py `
            --dataset 100k `
            --eval-split random `
            --fold $Fold `
            --assign-root $AssignRoot `
            --assign-suffix $Suffix `
            --assign-from-db `
            --assign-db-strategy $DbStrategy `
            --assign-db-overwrite `
            --mode baselines `
            --no-global `
            --no-cluster-avg `
            --k $K `
            --knn $Knn `
            --similarity pearson `
            --expand-knn `
            --sig-weight 10 `
            --algo @Algos
    }
}

function Invoke-CompareMetrics {
    python mealpy\summarize_k20_comparison.py --k $K --fold $Fold --knn $Knn
}

switch ($Phase) {
    "assignment" {
        Write-Host "=== WNMF30 assignment K=$K : $($CompareAlgos -join ', ') ===" -ForegroundColor Cyan
        Invoke-Wnmf30Assignment -Algos $CompareAlgos
    }
    "assignment-pca" {
        Write-Host "=== PCA95 assignment K=$K : HA_AVOAHGS ===" -ForegroundColor Cyan
        Invoke-PcaAssignment
    }
    "export-db" {
        Write-Host "=== DB export (WNMF30) ===" -ForegroundColor Cyan
        Invoke-ExportAll -Suffix $WnmfSuffix -Algos $CompareAlgos
        Write-Host "=== DB export (PCA95) ===" -ForegroundColor Cyan
        Invoke-ExportAll -Suffix $PcaSuffix -Algos @("HA_AVOAHGS")
    }
    "eval" {
        Write-Host "=== WNMF eval fold $Fold kNN=$Knn (+ SVD baseline) ===" -ForegroundColor Cyan
        Invoke-WnmfEval -Suffix $WnmfSuffix -Algos $AllEvalAlgos -IncludeSvdBaseline
    }
    "eval-pca" {
        Write-Host "=== PCA eval fold $Fold kNN=$Knn ===" -ForegroundColor Cyan
        Invoke-WnmfEval -Suffix $PcaSuffix -Algos @("HA_AVOAHGS")
    }
    "compare" {
        Invoke-CompareMetrics
    }
    "all" {
        Invoke-Wnmf30Assignment -Algos $CompareAlgos
        Invoke-PcaAssignment
        Invoke-WnmfEval -Suffix $WnmfSuffix -Algos $AllEvalAlgos -IncludeSvdBaseline
        Invoke-WnmfEval -Suffix $PcaSuffix -Algos @("HA_AVOAHGS")
        Invoke-CompareMetrics
    }
}

Write-Host "Bitti." -ForegroundColor Green
