# Otomatik üretilmiş grid betiği
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
    [int]$K
  )
  $s = "_pruneu$($MinU)_i$($MinI)"
  if ($Zscore) { $s += "_zscore" }
  if ($Pca -ne "") { $s += "_pca$([int][math]::Round([double]$Pca * 100))pct" }
  if ($WnmfFeat -gt 0) {
    $s += "_wnmf$($WnmfFeat)_$($WnmfInit)_trim$($TrimLow)_$($TrimHigh)"
  }
  if ($Metric -eq "fuzzy")    { $s += "_fuzzy" }
  elseif ($Metric -eq "euclidean") { $s += "_euc" }
  if ($Paper) { $s += "_paper" }
  elseif ($NoGs) { $s += "_nogs" }
  $s += "_$($Preprocess)_$($Fe)$($Components)_k$($K)"
  return $s
}

$ErrorActionPreference = "Stop"
$dataset    = "100k"
$minU       = 5
$minI       = 10
$initMode   = ""
$jobs       = "4"
$useZscore  = $true
$useLof     = $true
$useFcm     = $true
$skipExist  = $true
$noGraySheep= $true
$algos      = @("B0_KMEANS", "B2_HGS", "H4_MFO+HHO", "LIT_GWO", "LIT_PSO", "HA_AVOAHGS")

$ks   = @("7", "14", "30", "70")
$fes  = @("nmf", "wnmf")
$dims = @("10", "20", "50")
$cms  = @("fuzzy", "euclidean")
$pps  = @("minmax", "maxabs")
$pcas = @("")
$knns = @("10", "14", "30")
$sims = @("pearson")
$predMode      = "baselines"
$predNoAvg     = $true
$predNoGlobal  = $true
$predAlgoJobs  = ""

$counter = 0
$total = 288
$effectiveLof = $useLof -and (-not $noGraySheep)
$assignRoot = if ($effectiveLof) { "mealpy\results\assignments_lof" } else { "mealpy\results\assignments" }

foreach ($k in $ks) {
  foreach ($fe in $fes) {
    foreach ($dim in $dims) {
      foreach ($cm in $cms) {
        foreach ($pp in $pps) {
          foreach ($pca in $pcas) {
            # WNMF + PCA aynı anda çalışmaz, atla
            if ($fe -eq "wnmf" -and $pca -ne "") { continue }

            $counter++
            Write-Host ""
            Write-Host "==================================================" -ForegroundColor Cyan
            Write-Host "[$counter / $total] k=$k fe=$fe dim=$dim cm=$cm pp=$pp pca=$pca" -ForegroundColor Cyan
            Write-Host "==================================================" -ForegroundColor Cyan

            $args = @(
              "mealpy\generate_assignments.py",
              "--dataset", $dataset,
              "--k", $k,
              "--algo"
            ) + $algos + @(
              "--preprocess", $pp,
              "--feature-extraction", $fe,
              "--svd-components", $dim,
              "--cluster-metric", $cm,
              "--min-user-ratings", $minU,
              "--min-item-ratings", $minI
            )
            if ($useZscore)   { $args += "--zscore" }
            if ($useLof)      { $args += "--lof" }
            if ($useFcm)      { $args += "--fcm" }
            if ($skipExist)   { $args += "--skip-existing" }
            if ($noGraySheep) { $args += "--no-gray-sheep" }
            if ($initMode)    { $args += @("--init-mode", $initMode) }
            if ($jobs -ne "") { $args += @("--jobs", $jobs) }
            if ($pca -ne "")  {
              $args += @("--pca", $pca)
              # PCA varken WNMF zaten elendi
            }
            if ($fe -eq "wnmf") {
              $args += @("--wnmf-features", $dim)
            }

            python @args

            # Atama klasörü suffix'i
            $wnmfFeat = if ($fe -eq "wnmf") { [int]$dim } else { 0 }
            $suffix = Get-AssignSuffix -MinU $minU -MinI $minI `
              -Zscore $useZscore -Pca $pca `
              -WnmfFeat $wnmfFeat -WnmfInit "inmed" -TrimLow 5.0 -TrimHigh 95.0 `
              -Metric $cm -Paper $false -NoGs $noGraySheep `
              -Preprocess $pp -Fe $fe -Components ([int]$dim) -K ([int]$k)

            foreach ($knn in $knns) {
              foreach ($sim in $sims) {
                Write-Host "  -> tahmin: knn=$knn sim=$sim mode=$predMode" -ForegroundColor Yellow
                $pargs = @(
                  "wnmf\wnmf_experiment.py",
                  "--dataset", $dataset,
                  "--k", $k,
                  "--assign-root", $assignRoot,
                  "--assign-suffix", $suffix,
                  "--mode", $predMode,
                  "--similarity", $sim,
                  "--knn", $knn,
                  "--algo"
                ) + $algos
                if ($predNoAvg)    { $pargs += "--no-cluster-avg" }
                if ($predNoGlobal) { $pargs += "--no-global" }
                if ($predAlgoJobs -ne "") { $pargs += @("--algo-jobs", $predAlgoJobs) }
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