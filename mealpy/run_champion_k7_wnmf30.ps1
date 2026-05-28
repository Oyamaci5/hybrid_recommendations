# Champion pipeline: LOF + zscore + WNMF30 + K=7 + euclidean + mkpp
# Train-only suffix: ..._trainonly_rand_fN_none_wnmf30_k7
#
# powershell -ExecutionPolicy Bypass -File .\mealpy\run_champion_k7_wnmf30.ps1 -Phase trainonly-5fold

param(
    [ValidateSet("assignment", "assignment-5fold", "trainonly-5fold", "export-db", "eval", "eval-5fold", "eval-svd-baseline", "all")]
    [string]$Phase = "all",
    [string[]]$Algo = @("HA_AVOAHGS"),
    [ValidateSet("best_wcss", "latest", "worst_wcss")]
    [string]$DbStrategy = "best_wcss",
    [switch]$TrainOnly,
    [switch]$NoNearestCentroid,
    [switch]$ExpandKnn,
    [switch]$LatentDev,
    [ValidateSet("cluster", "weighted_cluster")]
    [string]$KnnMode = "cluster",
    [string]$ClusterWeightAlpha = "1",
    [string]$ClusterWeightBase = "1.0",
    [ValidateSet("MFO", "IWO", "HA")]
    [string]$CentroidAlgo = "MFO",
    [int]$Fold = 5
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $RepoRoot

$AssignRoot = "mealpy\results\assignments_lof"

function Get-TrainOnlyTag {
    param([int]$FoldNum = 5)
    if ($FoldNum -eq 1) { return '_trainonly_rand_f1' }
    return "_trainonly_rand_f${FoldNum}"
}

function Get-AssignSuffix {
    param([int]$FoldNum = 5, [switch]$UseTrainOnly, [switch]$NoKmRef)
    $kmref = if ($NoKmRef) { '' } else { '_kmref' }
    if ($UseTrainOnly) {
        $tag = Get-TrainOnlyTag -FoldNum $FoldNum
        return "_pruneu5_i10_zscore_euc_imkpp${tag}_none_wnmf30_k7${kmref}"
    }
    return "_pruneu5_i10_zscore_euc_imkpp_none_wnmf30_k7${kmref}"
}

function Get-ChampionDir {
    param([int]$FoldNum = 5, [switch]$UseTrainOnly)
    $suffix = Get-AssignSuffix -FoldNum $FoldNum -UseTrainOnly:$UseTrainOnly
    return Join-Path $AssignRoot "ml100k\HA_AVOAHGS$suffix"
}

$AssignSuffix = Get-AssignSuffix -FoldNum $Fold -UseTrainOnly:$TrainOnly
$ChampionDir  = Get-ChampionDir -FoldNum $Fold -UseTrainOnly:$TrainOnly
$UseNearestCentroid = -not $NoNearestCentroid

function Get-WnmfWCacheDir {
    param([int]$FoldNum = 5)
    return "mealpy\results\latent_dev_cache\fold$FoldNum"
}

function Invoke-WnmfWCache {
    param([int]$FoldNum = 5)
    $cache = Get-WnmfWCacheDir -FoldNum $FoldNum
    $out = Join-Path $cache "user_features.npy"
    if (Test-Path $out) {
        Write-Host "  W cache mevcut: $out" -ForegroundColor DarkGray
        return $cache
    }
    Write-Host "  W cache uretiliyor: fold $FoldNum -> $cache" -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $cache | Out-Null
    python -c @"
import os, sys, numpy as np
sys.path.insert(0, 'mealpy')
from generate_assignments import (
    load_movielens_train_only_100k, prepare_matrix_for_clustering, SEED, DATA_100K,
)
cache = r'$cache'.replace('\\','/')
os.makedirs(cache, exist_ok=True)
matrix = load_movielens_train_only_100k(
    eval_split='random', fold=$FoldNum, random_seed=SEED, ratings_path=DATA_100K,
)
matrix = prepare_matrix_for_clustering(
    matrix, zscore=True, pca_var=None, wnmf_k=30,
    preprocess='none', feature_extraction='wnmf', svd_components=30,
    min_user_ratings=5, min_item_ratings=10,
    wnmf_init_method='inmed', inmed_trim=(5.0, 95.0),
)
np.save(os.path.join(cache, 'user_features.npy'), matrix)
print('W cache:', matrix.shape)
"@
    if (-not (Test-Path $out)) { throw "W cache olusturulamadi: $out" }
    return $cache
}

function Invoke-Assignment {
    param(
        [string[]]$Algos,
        [int]$FoldNum = 5,
        [switch]$UseTrainOnly,
        [switch]$UseLatentDev
    )
    $algoArgs = @()
    if ($Algos.Count -gt 0) {
        $algoArgs = @("--algo") + $Algos
    }
    $trainOnlyArgs = @()
    if ($UseTrainOnly) {
        $trainOnlyArgs = @("--train-only", "--eval-split", "random", "--fold", "$FoldNum")
    }
    $latentArgs = @()
    if ($UseLatentDev) {
        $wPath = Invoke-WnmfWCache -FoldNum $FoldNum
        $latentArgs = @(
            "--fitness", "latent_dev",
            "--wnmf-model-path", $wPath,
            "--centroid-algo", $CentroidAlgo,
            "--centroid-agents", "20",
            "--centroid-iter", "50"
        )
    }
    python mealpy\generate_assignments.py `
        --dataset 100k `
        --lof `
        --zscore `
        --preprocess none `
        --feature-extraction wnmf `
        --svd-components 30 `
        --cluster-metric euclidean `
        --init-mode mkpp `
        --k 7 `
        --skip-existing `
        @trainOnlyArgs `
        @latentArgs `
        @algoArgs
}

function Invoke-ExportFromDb {
    param([int]$FoldNum = 5, [switch]$UseTrainOnly)
    $suffix = Get-AssignSuffix -FoldNum $FoldNum -UseTrainOnly:$UseTrainOnly
    $outDir = Get-ChampionDir -FoldNum $FoldNum -UseTrainOnly:$UseTrainOnly
    python assignment_db.py export ml100k HA_AVOAHGS 7 $suffix $outDir $DbStrategy
}

function Invoke-WnmfEval {
    param(
        [int]$FoldNum = 5,
        [string[]]$WnmfAlgos = @("HA_AVOAHGS"),
        [switch]$ExpandKnn,
        [switch]$UseTrainOnly,
        [switch]$UseNearestCentroid,
        [string]$EvalKnnMode = "cluster",
        [string]$EvalClusterWeightAlpha = "1",
        [string]$EvalClusterWeightBase = "1.0"
    )
    $suffix = Get-AssignSuffix -FoldNum $FoldNum -UseTrainOnly:$UseTrainOnly
    $expandArgs = if ($ExpandKnn) { @("--expand-knn", "--sig-weight", "10") } else { @() }
    $ncArgs = if ($UseNearestCentroid) {
        @("--nearest-centroid", "--centroid-metric", "euclidean")
    } else {
        @()
    }
    $knnArgs = @()
    if ($EvalKnnMode -eq "weighted_cluster") {
        $knnArgs = @(
            "--knn-mode", "weighted_cluster",
            "--cluster-weight-alpha", $EvalClusterWeightAlpha
        )
        if ($EvalClusterWeightAlpha -eq "auto") {
            $knnArgs += @("--cluster-weight-base", $EvalClusterWeightBase)
        }
    }
    # nearest-centroid için best_sol.npy + user_features.npy gerekir; DB export bunları yazmaz.
    $dbArgs = if ($UseNearestCentroid) {
        @()
    } else {
        @("--assign-from-db", "--assign-db-strategy", $DbStrategy, "--assign-db-overwrite")
    }
    python wnmf\wnmf_experiment.py `
        --dataset 100k `
        --eval-split random `
        --fold $FoldNum `
        --assign-root $AssignRoot `
        --assign-suffix $suffix `
        @dbArgs `
        --mode baselines `
        --no-global `
        --no-cluster-avg `
        --k 7 `
        --knn 10 `
        --similarity pearson `
        @expandArgs `
        @ncArgs `
        @knnArgs `
        --algo @WnmfAlgos
}

function Invoke-Cv5MeanSummary {
    param([switch]$UseTrainOnly)
    if ($UseTrainOnly) {
        python mealpy\summarize_cv5.py --train-only
    } else {
        python mealpy\summarize_cv5.py
    }
}

switch ($Phase) {
    "assignment" {
        Write-Host "=== Assignment fold $Fold (TrainOnly=$TrainOnly) ===" -ForegroundColor Cyan
        Invoke-Assignment -Algos $Algo -FoldNum $Fold -UseTrainOnly:$TrainOnly -UseLatentDev:$LatentDev
        if (-not (Test-Path (Join-Path $ChampionDir "assignments.npy"))) {
            throw "Assignment beklenen klasörde yok: $ChampionDir"
        }
    }
    "assignment-5fold" {
        Write-Host "=== Assignment folds 1-5 (TrainOnly=$TrainOnly) ===" -ForegroundColor Cyan
        foreach ($f in 1..5) {
            Write-Host "--- fold $f ---" -ForegroundColor Yellow
            Invoke-Assignment -Algos $Algo -FoldNum $f -UseTrainOnly:$TrainOnly -UseLatentDev:$LatentDev
            $dir = Get-ChampionDir -FoldNum $f -UseTrainOnly:$TrainOnly
            if (-not (Test-Path (Join-Path $dir "assignments.npy"))) {
                throw "Assignment yok: $dir"
            }
        }
    }
    "trainonly-5fold" {
        Write-Host "=== Train-only 5-fold: assignment + eval ===" -ForegroundColor Cyan
        foreach ($f in 1..5) {
            Write-Host "--- fold $f assignment ---" -ForegroundColor Yellow
            Invoke-Assignment -Algos $Algo -FoldNum $f -UseTrainOnly
            $dir = Get-ChampionDir -FoldNum $f -UseTrainOnly
            if (-not (Test-Path (Join-Path $dir "assignments.npy"))) {
                throw "Assignment yok: $dir"
            }
            Write-Host "--- fold $f eval ---" -ForegroundColor Yellow
            Invoke-WnmfEval -FoldNum $f -WnmfAlgos $Algo -UseTrainOnly -UseNearestCentroid:$UseNearestCentroid -ExpandKnn:$ExpandKnn -EvalKnnMode $KnnMode -EvalClusterWeightAlpha $ClusterWeightAlpha -EvalClusterWeightBase $ClusterWeightBase
        }
        Invoke-Cv5MeanSummary -UseTrainOnly
    }
    "export-db" {
        Write-Host "=== DB'den champion assignment export ===" -ForegroundColor Cyan
        python assignment_db.py backfill-suffix 8000
        Invoke-ExportFromDb -FoldNum $Fold -UseTrainOnly:$TrainOnly
    }
    "eval" {
        Write-Host "=== WNMF eval fold $Fold (TrainOnly=$TrainOnly) ===" -ForegroundColor Cyan
        Invoke-WnmfEval -FoldNum $Fold -WnmfAlgos $Algo -UseTrainOnly:$TrainOnly -UseNearestCentroid:$UseNearestCentroid -ExpandKnn:$ExpandKnn -EvalKnnMode $KnnMode -EvalClusterWeightAlpha $ClusterWeightAlpha -EvalClusterWeightBase $ClusterWeightBase
    }
    "eval-5fold" {
        Write-Host "=== WNMF eval folds 1-5 (TrainOnly=$TrainOnly) ===" -ForegroundColor Cyan
        foreach ($f in 1..5) {
            Invoke-WnmfEval -FoldNum $f -WnmfAlgos $Algo -UseTrainOnly:$TrainOnly -UseNearestCentroid:$UseNearestCentroid -ExpandKnn:$ExpandKnn
        }
        Invoke-Cv5MeanSummary -UseTrainOnly:$TrainOnly
    }
    "eval-svd-baseline" {
        Write-Host "=== GLOBAL SVD baseline, folds 1-5 ===" -ForegroundColor Cyan
        foreach ($f in 1..5) {
            python wnmf\wnmf_experiment.py `
                --dataset 100k --eval-split random --fold $f `
                --mode baselines --k 7
        }
    }
    "all" {
        Invoke-ExportFromDb -FoldNum $Fold -UseTrainOnly:$TrainOnly
        Invoke-WnmfEval -FoldNum $Fold -WnmfAlgos $Algo -UseTrainOnly:$TrainOnly -UseNearestCentroid:$UseNearestCentroid -ExpandKnn:$ExpandKnn -EvalKnnMode $KnnMode -EvalClusterWeightAlpha $ClusterWeightAlpha -EvalClusterWeightBase $ClusterWeightBase
    }
}

Write-Host "Bitti. Champion path: $ChampionDir" -ForegroundColor Green
