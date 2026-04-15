@echo off
:: Her preprocessing suffix icin tam pipeline: global WNMF + cluster_avg + kNN + SharedV.
:: Oncelik: mealpy/results/assignments_lof/ml100k/*_k30* klasorleri hazir olsun
:: (run_ml100k_lof_k30_assign_none.bat + run_ml100k_lof_k30_variants.bat).
setlocal
cd /d "%~dp0"

echo ========== WNMF 1/6 suffix=(bos, ham clustering) ==========
python wnmf_experiment.py --dataset 100k --k 30 --mode sharedV
if errorlevel 1 exit /b 1

echo ========== WNMF 2/6 --assign-suffix _zscore ==========
python wnmf_experiment.py --dataset 100k --k 30 --assign-suffix _zscore --mode sharedV
if errorlevel 1 exit /b 1

echo ========== WNMF 3/6 --assign-suffix _wnmf20 ==========
python wnmf_experiment.py --dataset 100k --k 30 --assign-suffix _wnmf20 --mode sharedV
if errorlevel 1 exit /b 1

echo ========== WNMF 4/6 --assign-suffix _zscore_wnmf20 ==========
python wnmf_experiment.py --dataset 100k --k 30 --assign-suffix _zscore_wnmf20 --mode sharedV
if errorlevel 1 exit /b 1

echo ========== WNMF 5/6 --assign-suffix _pca80pct ==========
python wnmf_experiment.py --dataset 100k --k 30 --assign-suffix _pca80pct --mode sharedV
if errorlevel 1 exit /b 1

echo ========== WNMF 6/6 --assign-suffix _zscore_pca80pct ==========
python wnmf_experiment.py --dataset 100k --k 30 --assign-suffix _zscore_pca80pct --mode sharedV
if errorlevel 1 exit /b 1

echo Tamamlandi: 6 WNMF kosusu (her biri ayri runs/wnmf/.../runN + DB run).
endlocal
