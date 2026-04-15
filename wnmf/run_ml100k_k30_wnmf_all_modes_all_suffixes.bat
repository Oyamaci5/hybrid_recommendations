@echo off
:: Tekrarsiz: her preprocessing icin TEK kosu --mode all
:: (kume ort.+kNN + cluster_full WNMF + cluster_sharedV WNMF + global WNMF).
:: 6 suffix = 6 python cagrisi; baselines/sharedV/full ayri ayri kosulmaz.
setlocal EnableDelayedExpansion
cd /d "%~dp0"

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
echo Tamamlandi: 6 kosu (her biri --mode all).
exit /b 0

:run
set "TAG=%~1"
set /a RUNNO+=1
if /I "%TAG%"=="NONE" (
  echo ========== [!RUNNO!/6] --mode all  assign_suffix=[bos] ==========
  python wnmf_experiment.py --dataset 100k --k 30 --mode all
) else (
  echo ========== [!RUNNO!/6] --mode all  assign_suffix=%TAG% ==========
  python wnmf_experiment.py --dataset 100k --k 30 --assign-suffix %TAG% --mode all
)
if errorlevel 1 (
  echo HATA: suffix=%TAG%
  exit /b 1
)
exit /b 0
