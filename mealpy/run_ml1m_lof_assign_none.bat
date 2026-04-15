@echo off
:: ML-1M, LOF, preprocess YOK (none). K: varsayilan 150 (K_1M_DEFAULT); degistirmek icin K1M satirini duzenle.
setlocal
cd /d "%~dp0"
set "K1M=150"
set "ALGO=B0_KMEANS B1_HHO B2_HGS H1_HHO+HGS H4_MFO+HHO LIT_GOA LIT_GWO LIT_SSA H12_MFO+CDO H13_HHO+GAop"

echo ========== ML-1M Assignment: LOF, K=%K1M%, preprocess YOK ==========
python generate_assignments.py --lof --dataset 1m --algo %ALGO% --k-1m %K1M%
if errorlevel 1 exit /b 1
echo Tamamlandi.
endlocal
