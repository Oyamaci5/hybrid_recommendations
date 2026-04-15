@echo off
:: run_ml100k_lof_k30_variants.bat icinde OLMAYAN: hic preprocess yok (DB: preprocessing=none).
:: Once bunu bir kez calistir; sonra wnmf batch suffix'siz klasorleri kullanir.
setlocal
cd /d "%~dp0"
set "ALGO=B0_KMEANS B1_HHO B2_HGS H1_HHO+HGS H4_MFO+HHO LIT_GOA LIT_GWO LIT_SSA H12_MFO+CDO H13_HHO+GAop"
echo ========== Assignment: LOF, K=30, preprocess YOK (none) ==========
python generate_assignments.py --lof --dataset 100k --algo %ALGO% --k 30
if errorlevel 1 exit /b 1
echo Tamamlandi.
endlocal
