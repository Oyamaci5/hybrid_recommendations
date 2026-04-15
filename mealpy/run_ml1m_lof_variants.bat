@echo off
:: ML-1M, LOF, 5 preprocessing varyanti (100k run_ml100k_lof_k30_variants ile ayni mantik).
:: K: varsayilan 150; 100k ile ayni K istiyorsan set K1M=30 yap.
setlocal
cd /d "%~dp0"
set "K1M=30"
set "ALGO=B0_KMEANS B1_HHO B2_HGS H1_HHO+HGS H4_MFO+HHO LIT_GOA LIT_GWO LIT_SSA H12_MFO+CDO H13_HHO+GAop"

echo ========== 1/5 ML-1M --zscore K=%K1M% ==========
python generate_assignments.py --lof --dataset 1m --algo %ALGO% --k-1m %K1M% --zscore
if errorlevel 1 exit /b 1

echo ========== 2/5 ML-1M --wnmf-features 20 K=%K1M% ==========
python generate_assignments.py --lof --dataset 1m --algo %ALGO% --k-1m %K1M% --wnmf-features 20
if errorlevel 1 exit /b 1

echo ========== 3/5 ML-1M --zscore --wnmf-features 20 K=%K1M% ==========
python generate_assignments.py --lof --dataset 1m --algo %ALGO% --k-1m %K1M% --zscore --wnmf-features 20
if errorlevel 1 exit /b 1

echo ========== 4/5 ML-1M --pca 0.80 K=%K1M% ==========
python generate_assignments.py --lof --dataset 1m --algo %ALGO% --k-1m %K1M% --pca 0.90
if errorlevel 1 exit /b 1

echo ========== 5/5 ML-1M --zscore --pca 0.80 K=%K1M% ==========
python generate_assignments.py --lof --dataset 1m --algo %ALGO% --k-1m %K1M% --zscore --pca 0.90
if errorlevel 1 exit /b 1

echo Tamamlandi: 5 ML-1M kosusu + once none icin run_ml1m_lof_assign_none.bat calistir.
endlocal
