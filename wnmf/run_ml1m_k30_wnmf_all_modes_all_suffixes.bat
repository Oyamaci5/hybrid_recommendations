@echo off
:: ML-1M: tekrarsiz --mode all (baselines + full + sharedV + global), 6 suffix.
:: K: mealpy/run_ml1m_lof_*.bat ile ayni olmali (varsayilan 30).
:: PCA suffix: ml1m variants --pca 0.90 -> _pca90pct (100k 0.80 ile karistirma).
:: _zscore/_wnmf20 vb. klasor yoksa once mealpy\run_ml1m_lof_variants.bat calistir.
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "K1M=30"
set "ERR=0"
set "RUNNO=0"

call :run NONE || set ERR=1
call :run _zscore || set ERR=1
call :run _wnmf20 || set ERR=1
call :run _zscore_wnmf20 || set ERR=1
call :run _pca90pct || set ERR=1
call :run _zscore_pca90pct || set ERR=1

if %ERR% neq 0 (
  echo BIR VEYA DAHA FAZLA KOSU HATA ILE BITTI.
  exit /b 1
)
echo Tamamlandi: ML-1M 6 kosu (her biri --mode all, k-1m=%K1M%).
exit /b 0

:run
set "TAG=%~1"
set /a RUNNO+=1
if /I "%TAG%"=="NONE" (
  echo ========== [!RUNNO!/6] ML-1M --mode all  assign_suffix=[bos]  k-1m=%K1M% ==========
  python wnmf_experiment.py --dataset 1m --k-1m %K1M% --mode all
) else (
  echo ========== [!RUNNO!/6] ML-1M --mode all  assign_suffix=%TAG%  k-1m=%K1M% ==========
  python wnmf_experiment.py --dataset 1m --k-1m %K1M% --assign-suffix "%TAG%" --mode all
)
if errorlevel 1 (
  echo HATA: suffix=%TAG%
  exit /b 1
)
exit /b 0
