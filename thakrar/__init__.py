"""
Thakrar et al. (2025 BigComp) — Algoritma 2 temiz implementasyonu.

"An Improved Recommendation System Using K-Means Clustering and
Matrix Factorization"  (BigComp 2025).

İskelet (sklearn'siz, saf NumPy):
    Matrix Factorization (Alg.4)  →  centroid init  →  K-Means (Alg.5, kendi)
    →  küme-ortalaması tahmini (Alg.6)  →  MAE/RMSE

Modüller:
    matrix_factorization : Alg.4 SGD MF (gözlemlenen oylarda)
    custom_kmeans        : Alg.5 Lloyd K-Means (kendi, boş-küme onarımlı)
    predict              : Alg.6 küme-ortalaması tahmini
    data                 : ML-100K resmi 5-fold yükleyici
    pipeline             : tek konfigürasyon uçtan uca çalıştırma
"""

from . import custom_kmeans, data, matrix_factorization, predict, pipeline  # noqa: F401
