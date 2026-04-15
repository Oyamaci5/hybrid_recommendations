@echo off
:: ML-1M: her suffix icin --mode sharedV (daha hizli "bakalim" kosusu).
:: K: mealpy/run_ml1m_lof_*.bat ile ayni (varsayilan 30).
:: PCA: ml1m variants --pca 0.90 -> _pca90pct. Suffix klasorleri yoksa run_ml1m_lof_variants.bat.
setlocal
cd /d "%~dp0"

set "K1M=30"

echo ========== WNMF ML-1M 1/6 suffix=[bos] k-1m=%K1M% ==========
python wnmf_experiment.py --dataset 1m --k-1m %K1M% --mode sharedV
if errorlevel 1 exit /b 1

echo ========== WNMF ML-1M 2/6 --assign-suffix _zscore ==========
python wnmf_experiment.py --dataset 1m --k-1m %K1M% --assign-suffix "_zscore" --mode sharedV
if errorlevel 1 exit /b 1

echo ========== WNMF ML-1M 3/6 --assign-suffix _wnmf20 ==========
python wnmf_experiment.py --dataset 1m --k-1m %K1M% --assign-suffix "_wnmf20" --mode sharedV
if errorlevel 1 exit /b 1

echo ========== WNMF ML-1M 4/6 --assign-suffix _zscore_wnmf20 ==========
python wnmf_experiment.py --dataset 1m --k-1m %K1M% --assign-suffix "_zscore_wnmf20" --mode sharedV
if errorlevel 1 exit /b 1

echo ========== WNMF ML-1M 5/6 --assign-suffix _pca90pct ==========
python wnmf_experiment.py --dataset 1m --k-1m %K1M% --assign-suffix "_pca90pct" --mode sharedV
if errorlevel 1 exit /b 1

echo ========== WNMF ML-1M 6/6 --assign-suffix _zscore_pca90pct ==========
python wnmf_experiment.py --dataset 1m --k-1m %K1M% --assign-suffix "_zscore_pca90pct" --mode sharedV
if errorlevel 1 exit /b 1

echo Tamamlandi: ML-1M 6 WNMF kosusu (sharedV, k-1m=%K1M%).
endlocal
