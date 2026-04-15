@echo off
setlocal
cd /d "%~dp0"

set "ALGO=B0_KMEANS B1_HHO B2_HGS H1_HHO+HGS H4_MFO+HHO LIT_GOA LIT_GWO LIT_SSA H12_MFO+CDO H13_HHO+GAop"

echo ========== Kosu 1 / 5 --zscore ==========
python generate_assignments.py --lof --dataset 100k --algo %ALGO% --k 30 --zscore
if errorlevel 1 exit /b 1

echo ========== Kosu 2 / 5 --wnmf-features 20 ==========
python generate_assignments.py --lof --dataset 100k --algo %ALGO% --k 30 --wnmf-features 20
if errorlevel 1 exit /b 1

echo ========== Kosu 3 / 5 --zscore --wnmf-features 20 ==========
python generate_assignments.py --lof --dataset 100k --algo %ALGO% --k 30 --zscore --wnmf-features 20
if errorlevel 1 exit /b 1

echo ========== Kosu 4 / 5 --pca 0.80 ==========
python generate_assignments.py --lof --dataset 100k --algo %ALGO% --k 30 --pca 0.80
if errorlevel 1 exit /b 1

echo ========== Kosu 5 / 5 --zscore --pca 0.80 ==========
python generate_assignments.py --lof --dataset 100k --algo %ALGO% --k 30 --zscore --pca 0.80
if errorlevel 1 exit /b 1

echo Tamamlandi: 5 kosu.
endlocal
