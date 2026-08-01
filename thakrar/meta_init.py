"""
Meta-sezgisel centroid başlatma köprüsü.

Makalede K-Means centroidleri Cuckoo Search (CSO) ile başlatılır (Alg.7). Burada,
mevcut depo altyapısını (mealpy + generate_assignments) yeniden kullanarak AVOA
(ve istenirse HHO/HGS/CSO vb.) ile WCSS-minimize centroidler üretiyoruz. Çıktı,
kendi K-Means'imize (custom_kmeans.kmeans) `init_centroids` olarak verilir.

Akış (makale Alg.2 ile uyumlu):
    P (MF kullanıcı-latent)  →  meta-sezgisel WCSS arama  →  (k, d) centroidler
    →  custom K-Means (Lloyd ile rafine)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_MEALPY = _REPO / "mealpy"
if str(_MEALPY) not in sys.path:
    sys.path.insert(0, str(_MEALPY))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Depo etiketi -> mealpy full_name
META_ALGOS = {
    "avoa": "AVOA.OriginalAVOA",
    "hho": "HHO.OriginalHHO",
    "cso": "CSO.OriginalCSO",
}


def metaheuristic_init(
    X: np.ndarray,
    k: int,
    *,
    algo: str = "avoa",
    epoch: int = 100,
    pop_size: int = 30,
    seed: int = 42,
) -> np.ndarray:
    """
    Meta-sezgisel ile (k, d) başlangıç centroidleri üret (WCSS fitness, öklid).

    Parametreler
    ------------
    X        : (n, d) öznitelik matrisi (MF kullanıcı-latent P).
    k        : küme sayısı.
    algo     : 'avoa' | 'hho' | 'cso' (META_ALGOS anahtarı).
    epoch    : optimizer iterasyon sayısı.
    pop_size : popülasyon boyutu (= başlangıç çözüm sayısı).

    Döndürür
    --------
    centroids : (k, d) float64
    """
    from mealpy_comparison_v2 import get_all_algorithms_v3, mkmeans_plus_plus_init
    from generate_assignments import run_single

    algo = algo.lower()
    if algo not in META_ALGOS:
        raise ValueError(f"Bilinmeyen meta algo: {algo!r}. Seçenekler: {list(META_ALGOS)}")
    full_name = META_ALGOS[algo]

    algo_map = {a["full_name"]: a for a in get_all_algorithms_v3()}
    if full_name not in algo_map:
        raise KeyError(
            f"{full_name} mealpy'de bulunamadı. Mevcut: "
            f"{[n for n in algo_map if n.startswith(algo.upper())]}"
        )
    algo_info = algo_map[full_name]

    X = np.asarray(X, dtype=np.float64)
    init = mkmeans_plus_plus_init(X, K=k, n_solutions=pop_size, seed=seed, metric="euclidean")

    best_sol, best_fit = run_single(
        algo_info, X, k, init, epoch, pop_size,
        metric="euclidean", cluster_objective="wcss",
        fitness_config={"objective": "wcss"},
    )
    return np.asarray(best_sol, dtype=np.float64).reshape(k, X.shape[1])
