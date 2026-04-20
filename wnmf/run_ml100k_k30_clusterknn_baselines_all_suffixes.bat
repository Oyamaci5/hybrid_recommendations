@echo off
:: ML-100K: yalnizca adim-3 degerlendirme (cluster_kNN, mean-centered).
:: Her suffix icin: --mode baselines + --no-global + --no-cluster-avg
:: Not: ilgili assignment klasorleri once generate_assignments ile uretilmis olmali.
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "K100K=30"
set "ERR=0"
set "RUNNO=0"

call :run NONE || set ERR=1
call :run _zscore || set ERR=1
call :run _wnmf20 || set ERR=1
call :run _zscore_wnmf20 || set ERR=1
call :run _pca80pct || set ERR=1
call :run _zscore_pca80pct || set ERR=1

if %ERR% neq 0 (
  echo BIR VEYA DAHA FAZLA KOSU HATA ILE BITTI.
  exit /b 1
)
echo Tamamlandi: ML-100K 6 kosu (cluster_knn-only, k=%K100K%).
exit /b 0

:run
set "TAG=%~1"
set /a RUNNO+=1
if /I "%TAG%"=="NONE" (
  echo ========== [!RUNNO!/6] ML-100K cluster_knn-only assign_suffix=[bos] k=%K100K% ==========
  python wnmf_experiment.py --dataset 100k --k %K100K% --mode baselines --no-global --no-cluster-avg --similarity pearson --knn 30
) else (
  echo ========== [!RUNNO!/6] ML-100K cluster_knn-only assign_suffix=%TAG% k=%K100K% ==========
  python wnmf_experiment.py --dataset 100k --k %K100K% --assign-suffix "%TAG%" --mode baselines --no-global --no-cluster-avg --similarity pearson --knn 30
)
if errorlevel 1 (
  echo HATA: suffix=%TAG%
  exit /b 1
)
exit /b 0
