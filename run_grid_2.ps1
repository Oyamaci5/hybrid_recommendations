# Otomatik üretilmiş grid betiği (senaryo bazlı)
# Çalıştırmadan önce konumun D:\hybrid_recommendations olduğundan emin ol.

function Get-AssignSuffix {
  param(
    [int]$MinU, [int]$MinI,
    [bool]$Zscore,
    [string]$Pca,
    [int]$WnmfFeat,
    [string]$WnmfInit, [double]$TrimLow, [double]$TrimHigh,
    [string]$Metric,
    [bool]$Paper, [bool]$NoGs,
    [string]$Preprocess, [string]$Fe, [int]$Components,
    [int]$K,
    [bool]$KmRefine = $false,
    [string]$Algo = ""
  )
  $s = "_pruneu$($MinU)_i$($MinI)"
  if ($Zscore) { $s += "_zscore" }
  if ($Pca -ne "") { $s += "_pca$([int][math]::Round([double]$Pca * 100))pct" }
  # Yol A.2: legacy --wnmf-features verildiyse SADECE init/trim etiketi düşer.
  # Latent boyut etiketi (_wnmf{N}) zaten aşağıda _$Fe$Components içinde tek sefer yazılır.
  if ($WnmfFeat -gt 0) {
    $s += "_$($WnmfInit)_trim$($TrimLow)_$($TrimHigh)"
  }
  if ($Metric -eq "fuzzy")    { $s += "_fuzzy" }
  elseif ($Metric -eq "euclidean") { $s += "_euc" }
  if ($Paper) { $s += "_paper" }
  elseif ($NoGs) { $s += "_nogs" }
  $s += "_$($Preprocess)_$($Fe)$($Components)_k$($K)"
  # KMeans refinement aktifse ve algoritma B0_KMEANS değilse klasör adına _kmref ekle
  if ($KmRefine -and $Algo -ne "B0_KMEANS") { $s += "_kmref" }
  return $s
}

$ErrorActionPreference = "Stop"
$dataset    = "100k"
$minU       = 5
$minI       = 10
$initMode   = ""
$jobs       = "4"
$useFcm     = $true
$skipExist  = $true
$kmRefine   = $false
$kmRefineIt = ""
$algos      = @("B0_KMEANS", "B2_HGS", "LIT_GOA", "LIT_GWO", "LIT_PSO", "HA_AVOAHGS")

$ks      = @("7", "14", "30")
$gsModes = @("lof", "off")

# Senaryolar — her satır kurallı bir kombinasyon
$scenarios = @(
  @{ Label="WNMF·zscore · wnmf/dim10/none/fuzzy/zs"; Fe="wnmf"; Dim="10"; Pp="none"; Cm="fuzzy"; Pca=""; Zscore=$true },
  @{ Label="WNMF·zscore · wnmf/dim10/none/euclidean/zs"; Fe="wnmf"; Dim="10"; Pp="none"; Cm="euclidean"; Pca=""; Zscore=$true },
  @{ Label="WNMF·zscore · wnmf/dim10/minmax/fuzzy/zs"; Fe="wnmf"; Dim="10"; Pp="minmax"; Cm="fuzzy"; Pca=""; Zscore=$true },
  @{ Label="WNMF·zscore · wnmf/dim10/minmax/euclidean/zs"; Fe="wnmf"; Dim="10"; Pp="minmax"; Cm="euclidean"; Pca=""; Zscore=$true },
  @{ Label="WNMF·zscore · wnmf/dim10/maxabs/fuzzy/zs"; Fe="wnmf"; Dim="10"; Pp="maxabs"; Cm="fuzzy"; Pca=""; Zscore=$true },
  @{ Label="WNMF·zscore · wnmf/dim10/maxabs/euclidean/zs"; Fe="wnmf"; Dim="10"; Pp="maxabs"; Cm="euclidean"; Pca=""; Zscore=$true },
  @{ Label="WNMF·zscore · wnmf/dim20/none/fuzzy/zs"; Fe="wnmf"; Dim="20"; Pp="none"; Cm="fuzzy"; Pca=""; Zscore=$true },
  @{ Label="WNMF·zscore · wnmf/dim20/none/euclidean/zs"; Fe="wnmf"; Dim="20"; Pp="none"; Cm="euclidean"; Pca=""; Zscore=$true },
  @{ Label="WNMF·zscore · wnmf/dim20/minmax/fuzzy/zs"; Fe="wnmf"; Dim="20"; Pp="minmax"; Cm="fuzzy"; Pca=""; Zscore=$true },
  @{ Label="WNMF·zscore · wnmf/dim20/minmax/euclidean/zs"; Fe="wnmf"; Dim="20"; Pp="minmax"; Cm="euclidean"; Pca=""; Zscore=$true },
  @{ Label="WNMF·zscore · wnmf/dim20/maxabs/fuzzy/zs"; Fe="wnmf"; Dim="20"; Pp="maxabs"; Cm="fuzzy"; Pca=""; Zscore=$true },
  @{ Label="WNMF·zscore · wnmf/dim20/maxabs/euclidean/zs"; Fe="wnmf"; Dim="20"; Pp="maxabs"; Cm="euclidean"; Pca=""; Zscore=$true },
  @{ Label="WNMF·zscore · wnmf/dim30/none/fuzzy/zs"; Fe="wnmf"; Dim="30"; Pp="none"; Cm="fuzzy"; Pca=""; Zscore=$true },
  @{ Label="WNMF·zscore · wnmf/dim30/none/euclidean/zs"; Fe="wnmf"; Dim="30"; Pp="none"; Cm="euclidean"; Pca=""; Zscore=$true },
  @{ Label="WNMF·zscore · wnmf/dim30/minmax/fuzzy/zs"; Fe="wnmf"; Dim="30"; Pp="minmax"; Cm="fuzzy"; Pca=""; Zscore=$true },
  @{ Label="WNMF·zscore · wnmf/dim30/minmax/euclidean/zs"; Fe="wnmf"; Dim="30"; Pp="minmax"; Cm="euclidean"; Pca=""; Zscore=$true },
  @{ Label="WNMF·zscore · wnmf/dim30/maxabs/fuzzy/zs"; Fe="wnmf"; Dim="30"; Pp="maxabs"; Cm="fuzzy"; Pca=""; Zscore=$true },
  @{ Label="WNMF·zscore · wnmf/dim30/maxabs/euclidean/zs"; Fe="wnmf"; Dim="30"; Pp="maxabs"; Cm="euclidean"; Pca=""; Zscore=$true },
  @{ Label="NMF·noscaler · nmf/dim10/none/fuzzy/no-zs"; Fe="nmf"; Dim="10"; Pp="none"; Cm="fuzzy"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim10/none/euclidean/no-zs"; Fe="nmf"; Dim="10"; Pp="none"; Cm="euclidean"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim10/minmax/fuzzy/no-zs"; Fe="nmf"; Dim="10"; Pp="minmax"; Cm="fuzzy"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim10/minmax/euclidean/no-zs"; Fe="nmf"; Dim="10"; Pp="minmax"; Cm="euclidean"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim10/maxabs/fuzzy/no-zs"; Fe="nmf"; Dim="10"; Pp="maxabs"; Cm="fuzzy"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim10/maxabs/euclidean/no-zs"; Fe="nmf"; Dim="10"; Pp="maxabs"; Cm="euclidean"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim10/none/fuzzy/no-zs"; Fe="nmf"; Dim="10"; Pp="none"; Cm="fuzzy"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim10/none/euclidean/no-zs"; Fe="nmf"; Dim="10"; Pp="none"; Cm="euclidean"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim20/none/fuzzy/no-zs"; Fe="nmf"; Dim="20"; Pp="none"; Cm="fuzzy"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim20/none/euclidean/no-zs"; Fe="nmf"; Dim="20"; Pp="none"; Cm="euclidean"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim20/minmax/fuzzy/no-zs"; Fe="nmf"; Dim="20"; Pp="minmax"; Cm="fuzzy"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim20/minmax/euclidean/no-zs"; Fe="nmf"; Dim="20"; Pp="minmax"; Cm="euclidean"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim20/maxabs/fuzzy/no-zs"; Fe="nmf"; Dim="20"; Pp="maxabs"; Cm="fuzzy"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim20/maxabs/euclidean/no-zs"; Fe="nmf"; Dim="20"; Pp="maxabs"; Cm="euclidean"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim20/none/fuzzy/no-zs"; Fe="nmf"; Dim="20"; Pp="none"; Cm="fuzzy"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim20/none/euclidean/no-zs"; Fe="nmf"; Dim="20"; Pp="none"; Cm="euclidean"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim30/none/fuzzy/no-zs"; Fe="nmf"; Dim="30"; Pp="none"; Cm="fuzzy"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim30/none/euclidean/no-zs"; Fe="nmf"; Dim="30"; Pp="none"; Cm="euclidean"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim30/minmax/fuzzy/no-zs"; Fe="nmf"; Dim="30"; Pp="minmax"; Cm="fuzzy"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim30/minmax/euclidean/no-zs"; Fe="nmf"; Dim="30"; Pp="minmax"; Cm="euclidean"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim30/maxabs/fuzzy/no-zs"; Fe="nmf"; Dim="30"; Pp="maxabs"; Cm="fuzzy"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim30/maxabs/euclidean/no-zs"; Fe="nmf"; Dim="30"; Pp="maxabs"; Cm="euclidean"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim30/none/fuzzy/no-zs"; Fe="nmf"; Dim="30"; Pp="none"; Cm="fuzzy"; Pca=""; Zscore=$false },
  @{ Label="NMF·noscaler · nmf/dim30/none/euclidean/no-zs"; Fe="nmf"; Dim="30"; Pp="none"; Cm="euclidean"; Pca=""; Zscore=$false },
  @{ Label="PCA·zscore·0.95 · pca/dim10/none/euclidean/pca0.95/zs"; Fe="pca"; Dim="10"; Pp="none"; Cm="euclidean"; Pca="0.95"; Zscore=$true },
  @{ Label="PCA·zscore·0.95 · pca/dim10/none/fuzzy/pca0.95/zs"; Fe="pca"; Dim="10"; Pp="none"; Cm="fuzzy"; Pca="0.95"; Zscore=$true },
  @{ Label="PCA·zscore·0.95 · pca/dim20/none/euclidean/pca0.95/zs"; Fe="pca"; Dim="20"; Pp="none"; Cm="euclidean"; Pca="0.95"; Zscore=$true },
  @{ Label="PCA·zscore·0.95 · pca/dim20/none/fuzzy/pca0.95/zs"; Fe="pca"; Dim="20"; Pp="none"; Cm="fuzzy"; Pca="0.95"; Zscore=$true },
  @{ Label="PCA·zscore·0.95 · pca/dim50/none/euclidean/pca0.95/zs"; Fe="pca"; Dim="50"; Pp="none"; Cm="euclidean"; Pca="0.95"; Zscore=$true },
  @{ Label="PCA·zscore·0.95 · pca/dim50/none/fuzzy/pca0.95/zs"; Fe="pca"; Dim="50"; Pp="none"; Cm="fuzzy"; Pca="0.95"; Zscore=$true }
)

$knns = @("10", "14", "30")
$sims = @("pearson")
$predMode      = "baselines"
$predNoAvg     = $true
$predNoGlobal  = $true
$predAlgoJobs  = ""
$predFolds = @("1")

$counter = 0
$total = 864

foreach ($k in $ks) {
  foreach ($sc in $scenarios) {
    foreach ($gsMode in $gsModes) {
      $useLof      = ($gsMode -eq "lof")
      $noGraySheep = ($gsMode -eq "off")
      $assignRoot  = if ($useLof) { "mealpy\results\assignments_lof" } else { "mealpy\results\assignments" }

      $counter++
      Write-Host ""
      Write-Host "==================================================" -ForegroundColor Cyan
      Write-Host "[$counter / $total] k=$k gs=$gsMode  $($sc.Label)" -ForegroundColor Cyan
      Write-Host "==================================================" -ForegroundColor Cyan

      $args = @(
        "mealpy\generate_assignments.py",
        "--dataset", $dataset,
        "--k", $k,
        "--algo"
      ) + $algos + @(
        "--preprocess", $sc.Pp,
        "--feature-extraction", $sc.Fe,
        "--svd-components", $sc.Dim,
        "--cluster-metric", $sc.Cm,
        "--min-user-ratings", $minU,
        "--min-item-ratings", $minI
      )
      if ($sc.Zscore)    { $args += "--zscore" }
      if ($useLof)       { $args += "--lof" }
      if ($noGraySheep)  { $args += "--no-gray-sheep" }
      if ($useFcm)       { $args += "--fcm" }
      if ($skipExist)    { $args += "--skip-existing" }
      if ($kmRefine)     { $args += "--kmeans-refine" }
      if ($kmRefine -and $kmRefineIt -ne "") { $args += @("--kmeans-refine-iter", $kmRefineIt) }
      if ($initMode)     { $args += @("--init-mode", $initMode) }
      if ($jobs -ne "")  { $args += @("--jobs", $jobs) }
      if ($sc.Pca -ne "" -and $sc.Fe -eq "pca") { $args += @("--pca", $sc.Pca) }

      python @args

      # Atama klasörü suffix'i (kmref durumunda B0_KMEANS hariç _kmref eklenir)
      $suffixBase = Get-AssignSuffix -MinU $minU -MinI $minI `
        -Zscore $sc.Zscore -Pca $sc.Pca `
        -WnmfFeat 0 -WnmfInit "inmed" -TrimLow 5.0 -TrimHigh 95.0 `
        -Metric $sc.Cm -Paper $false -NoGs $noGraySheep `
        -Preprocess $sc.Pp -Fe $sc.Fe -Components ([int]$sc.Dim) -K ([int]$k) `
        -KmRefine $false -Algo "B0_KMEANS"
      $suffixRefined = Get-AssignSuffix -MinU $minU -MinI $minI `
        -Zscore $sc.Zscore -Pca $sc.Pca `
        -WnmfFeat 0 -WnmfInit "inmed" -TrimLow 5.0 -TrimHigh 95.0 `
        -Metric $sc.Cm -Paper $false -NoGs $noGraySheep `
        -Preprocess $sc.Pp -Fe $sc.Fe -Components ([int]$sc.Dim) -K ([int]$k) `
        -KmRefine $kmRefine -Algo "OTHER"

      $algosBase    = $algos | Where-Object { $_ -eq "B0_KMEANS" }
      $algosRefined = $algos | Where-Object { $_ -ne "B0_KMEANS" }

      foreach ($fd in $predFolds) {
        foreach ($knn in $knns) {
          foreach ($sim in $sims) {
            Write-Host "  -> tahmin: fold=$fd knn=$knn sim=$sim mode=$predMode" -ForegroundColor Yellow

            $predGroups = @()
            if ($algosRefined.Count -gt 0) {
              $predGroups += ,@{ Algos = $algosRefined; Suffix = $suffixRefined }
            }
            if ($algosBase.Count -gt 0) {
              $predGroups += ,@{ Algos = $algosBase; Suffix = $suffixBase }
            }

            foreach ($grp in $predGroups) {
              $pargs = @(
                "wnmf\wnmf_experiment.py",
                "--dataset", $dataset,
                "--k", $k,
                "--assign-root", $assignRoot,
                "--assign-suffix", $grp.Suffix,
                "--mode", $predMode,
                "--similarity", $sim,
                "--knn", $knn,
                "--algo"
              ) + $grp.Algos
              if ($predNoAvg)    { $pargs += "--no-cluster-avg" }
              if ($predNoGlobal) { $pargs += "--no-global" }
              if ($predAlgoJobs -ne "") { $pargs += @("--algo-jobs", $predAlgoJobs) }
              if ($fd -ne "") { $pargs += @("--fold", [int]$fd) }
              python @pargs
            }
          }
        }
      }
    }
  }
  }
}

Write-Host ""
Write-Host "Grid tamamlandı: $counter / $total kombinasyon" -ForegroundColor Green