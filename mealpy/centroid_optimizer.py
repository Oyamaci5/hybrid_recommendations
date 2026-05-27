"""
centroid_optimizer.py
=====================
Meta-algoritma ile optimal cluster centroid'larını bul.
Fitness (knn_mae): örneklenmiş val seti üzerinde küme-içi kNN MAE.
Fitness (latent_dev): WNMF latent uzayında ortalama sapma (eski proxy).
"""

from __future__ import annotations

import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mealpy import FloatVar
from scipy.spatial.distance import cdist

_MEALPY_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_MEALPY_DIR)
if _MEALPY_DIR not in sys.path:
    sys.path.insert(0, _MEALPY_DIR)

from mealpy_comparison_v2 import get_special_params

_EMPTY_CLUSTER_PENALTY = 1e6

# generate_assignments.py ALGO_CONFIG ile uyumlu kısa adlar
CENTROID_ALGO_MAP = {
    'MFO': 'MFO.OriginalMFO',
    'IWO': 'IWO.OriginalIWO',
    'HA': 'HHO.OriginalHHO',
}

_W_ATTRS = ('W', 'U', 'user_factors', 'user_features')
_W_NPZ_KEYS = ('W', 'U', 'user_factors', 'user_features')
_W_NPY_NAMES = (
    'user_features.npy',
    'user_factors.npy',
    'ml100k_U.npy',
    'ml1m_U.npy',
    'U.npy',
    'W.npy',
)
_MODEL_FILE_NAMES = (
    'wnmf_model.pkl',
    'wnmf_model.joblib',
    'model.pkl',
    'model.joblib',
    'wnmf.pkl',
    'wnmf.joblib',
)


def assign_nearest(W: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """W uzayında her kullanıcıyı en yakın centroid kümesine ata."""
    dists = cdist(np.asarray(W, dtype=np.float64), np.asarray(centroids, dtype=np.float64))
    return dists.argmin(axis=1).astype(np.int32)


def sample_ratings(
    ratings: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rating triplets (u, i, r) içinden n adet örnekle."""
    ratings = np.asarray(ratings, dtype=np.float64)
    if ratings.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if len(ratings) <= n:
        return ratings.copy()
    idx = rng.choice(len(ratings), size=int(n), replace=False)
    return ratings[idx]


def _build_rating_caches(
    ratings: np.ndarray,
) -> Tuple[dict, dict, float, dict]:
    user_ratings: dict = {}
    for row in np.asarray(ratings, dtype=np.float64):
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        user_ratings.setdefault(u, {})[i] = r

    user_means = {
        u: float(np.mean(list(d.values())))
        for u, d in user_ratings.items()
    }
    all_vals = [r for d in user_ratings.values() for r in d.values()]
    global_mean = float(np.mean(all_vals)) if all_vals else 3.0

    item_popularity: dict = {}
    for u, items in user_ratings.items():
        for it in items:
            item_popularity[it] = item_popularity.get(it, 0) + 1

    return user_ratings, user_means, global_mean, item_popularity


def _predict_cluster_knn(
    u: int,
    i: int,
    cid: int,
    assignments: np.ndarray,
    user_ratings: dict,
    user_means: dict,
    global_mean: float,
    item_popularity: dict,
    k_neighbors: int = 20,
    min_common: int = 3,
) -> float:
    """Küme-içi Pearson kNN — wnmf_experiment._predict_knn ile aynı mantık."""
    wnmf_dir = os.path.join(_REPO_ROOT, 'wnmf')
    if wnmf_dir not in sys.path:
        sys.path.insert(0, wnmf_dir)
    from wnmf_experiment import knn_user_similarity

    neighbors = [v for v, a in enumerate(assignments) if int(a) == int(cid) and v != u]
    sims: List[Tuple[float, int]] = []
    for v in neighbors:
        if i not in user_ratings.get(v, {}):
            continue
        s = knn_user_similarity(
            u, v,
            similarity='pearson',
            user_ratings=user_ratings,
            user_means=user_means,
            item_popularity=item_popularity,
            min_common=min_common,
        )
        if abs(s) > 0.0:
            sims.append((s, v))

    base = float(user_means.get(u, global_mean))
    if not sims:
        return float(np.clip(base, 1.0, 5.0))

    sims.sort(key=lambda x: -abs(x[0]))
    top_k = sims[:max(1, int(k_neighbors))]
    num = sum(
        s * (user_ratings[v][i] - user_means.get(v, global_mean))
        for s, v in top_k
    )
    den = sum(abs(s) for s, _ in top_k)
    if den < 1e-8:
        return float(np.clip(base, 1.0, 5.0))
    return float(np.clip(base + num / den, 1.0, 5.0))


def fast_knn_mae(
    train_sample: np.ndarray,
    val_sample: np.ndarray,
    assignments: np.ndarray,
    k: int = 20,
    min_common: int = 3,
) -> float:
    """
    Örneklenmiş val triplets üzerinde küme-içi kNN MAE.
    train_sample: kNN komşuluk indeksi (500 rating yeterli).
    val_sample  : fitness değerlendirmesi (200-300 rating yeterli).
    """
    val_sample = np.asarray(val_sample, dtype=np.float64)
    if val_sample.size == 0:
        return float(_EMPTY_CLUSTER_PENALTY)

    user_ratings, user_means, global_mean, item_popularity = _build_rating_caches(
        train_sample,
    )
    assignments = np.asarray(assignments, dtype=np.int32)
    errors: List[float] = []

    for row in val_sample:
        u, i, r_true = int(row[0]), int(row[1]), float(row[2])
        if u < 0 or u >= len(assignments):
            continue
        cid = int(assignments[u])
        pred = _predict_cluster_knn(
            u, i, cid, assignments,
            user_ratings, user_means, global_mean, item_popularity,
            k_neighbors=k, min_common=min_common,
        )
        errors.append(abs(r_true - pred))

    if not errors:
        return float(_EMPTY_CLUSTER_PENALTY)
    return float(np.mean(errors))


def _as_w_matrix(arr, source: str) -> np.ndarray:
    W = np.asarray(arr, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError(f"{source}: 2D W/U matrisi bekleniyor, shape={W.shape}")
    return W


def _extract_w_from_model_obj(obj, source: str = 'model') -> np.ndarray:
    """WNMFModel, MFModel veya dict içinden kullanıcı latent matrisini çıkar."""
    if isinstance(obj, dict):
        for key in _W_NPZ_KEYS:
            if key in obj and obj[key] is not None:
                return _as_w_matrix(obj[key], f"{source}[{key!r}]")
        raise ValueError(f"{source}: dict içinde W/U/user_factors bulunamadı")

    for attr in _W_ATTRS:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if val is not None:
                return _as_w_matrix(val, f"{source}.{attr}")

    raise ValueError(
        f"{source}: W/U/user_factors attribute bulunamadı "
        f"({', '.join(_W_ATTRS)})"
    )


def _load_pickle_object(path: str):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, 'rb') as fh:
            return pickle.load(fh)


def load_wnmf_w_matrix(path: str) -> np.ndarray:
    """
    WNMF kullanıcı latent matrisini (W / U / user_factors) yükle.

    Desteklenen kaynaklar:
      - .npy dosyası (doğrudan W matrisi)
      - .npz (W, U, user_factors anahtarları)
      - .pkl / .joblib (model.W, model.U, model.user_factors)
      - dizin: user_features.npy, model.pkl, …
    """
    if not path:
        raise ValueError('--fitness latent_dev için --wnmf-model-path zorunlu')

    path = os.path.abspath(path)

    if os.path.isfile(path):
        lower = path.lower()
        if lower.endswith('.npy'):
            W = np.load(path)
            print(f"  W matrisi yüklendi: {path} {W.shape}")
            return _as_w_matrix(W, path)
        if lower.endswith('.npz'):
            data = np.load(path)
            for key in _W_NPZ_KEYS:
                if key in data:
                    print(f"  W matrisi yüklendi: {path} [{key}] {data[key].shape}")
                    return _as_w_matrix(data[key], f"{path}[{key}]")
            raise ValueError(f"{path}: npz içinde W/U/user_factors bulunamadı")
        if lower.endswith(('.pkl', '.pickle', '.joblib')):
            obj = _load_pickle_object(path)
            W = _extract_w_from_model_obj(obj, path)
            print(f"  W matrisi yüklendi: {path} {W.shape}")
            return W
        raise FileNotFoundError(f"--wnmf-model-path dosyası okunamadı: {path}")

    if os.path.isdir(path):
        for name in _W_NPY_NAMES:
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                W = np.load(candidate)
                print(f"  W matrisi yüklendi: {candidate} {W.shape}")
                return _as_w_matrix(W, candidate)
        for name in _MODEL_FILE_NAMES:
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                obj = _load_pickle_object(candidate)
                W = _extract_w_from_model_obj(obj, candidate)
                print(f"  W matrisi yüklendi: {candidate} {W.shape}")
                return W
        raise FileNotFoundError(
            f"--wnmf-model-path dizininde W matrisi/model bulunamadı: {path}"
        )

    raise FileNotFoundError(f"--wnmf-model-path geçersiz: {path}")


def _resolve_centroid_algo_class(algo: str):
    key = (algo or 'MFO').strip().upper()
    full_name = CENTROID_ALGO_MAP.get(key)
    if full_name is None:
        raise ValueError(
            f"Bilinmeyen centroid algo: {algo!r}; "
            f"desteklenen: {', '.join(CENTROID_ALGO_MAP)}"
        )

    import mealpy
    import inspect
    import pkgutil

    for _, modname, _ in pkgutil.walk_packages(
        path=mealpy.__path__,
        prefix=mealpy.__name__ + '.',
        onerror=lambda _x: None,
    ):
        if modname.split('.')[-1] != full_name.split('.')[0]:
            continue
        try:
            module = __import__(modname, fromlist='dummy')
            cls_name = full_name.split('.')[1]
            obj = getattr(module, cls_name, None)
            if inspect.isclass(obj) and hasattr(obj, 'solve'):
                return full_name, obj
        except Exception:
            continue

    raise ImportError(f"mealpy sınıfı bulunamadı: {full_name}")


class CentroidOptimizer:
    """WNMF W uzayında centroid araması (kNN MAE veya latent deviation fitness)."""

    def __init__(
        self,
        W_matrix: np.ndarray,
        K: int,
        n_agents: int = 20,
        n_iter: int = 50,
        algo: str = 'MFO',
        seed: int = 42,
        train_ratings: Optional[np.ndarray] = None,
        val_ratings: Optional[np.ndarray] = None,
        n_train_sample: int = 500,
        n_val_sample: int = 200,
        knn_k: int = 20,
        fitness_mode: str = 'knn_mae',
    ):
        self.W = np.asarray(W_matrix, dtype=np.float64)
        self.K = int(K)
        self.n_users, self.n_features = self.W.shape
        self.n_agents = max(5, int(n_agents))
        self.n_iter = max(1, int(n_iter))
        self.algo = (algo or 'MFO').strip().upper()
        self.seed = int(seed)
        self.fitness_mode = (fitness_mode or 'knn_mae').strip().lower()
        self.knn_k = max(1, int(knn_k))
        self.rng = np.random.default_rng(self.seed)

        self.train_sample = None
        self.val_sample = None
        if self.fitness_mode == 'knn_mae':
            train_ratings = np.asarray(train_ratings, dtype=np.float64) if train_ratings is not None else np.zeros((0, 3))
            val_ratings = np.asarray(val_ratings, dtype=np.float64) if val_ratings is not None else np.zeros((0, 3))
            self.train_sample = sample_ratings(train_ratings, int(n_train_sample), self.rng)
            self.val_sample = sample_ratings(val_ratings, int(n_val_sample), self.rng)
            print(
                f"  CentroidOptimizer knn_mae: train_sample={len(self.train_sample)}, "
                f"val_sample={len(self.val_sample)}, k={self.knn_k}",
                flush=True,
            )

        col_min = self.W.min(axis=0)
        col_max = self.W.max(axis=0)
        self.lb = np.tile(col_min, self.K).tolist()
        self.ub = np.tile(col_max, self.K).tolist()

    def _assign(self, centroids: np.ndarray) -> np.ndarray:
        return assign_nearest(self.W, centroids)

    def _latent_dev_fitness(self, centroids: np.ndarray, assignments: np.ndarray) -> float:
        for cid in range(self.K):
            if not np.any(assignments == cid):
                return float(_EMPTY_CLUSTER_PENALTY)

        total = 0.0
        for cid in range(self.K):
            members = self.W[assignments == cid]
            total += float(np.linalg.norm(members - centroids[cid], axis=1).sum())
        return float(total / self.n_users)

    def fitness(self, individual) -> float:
        centroids = np.asarray(individual, dtype=np.float64).reshape(
            self.K, self.n_features,
        )
        assignments = self._assign(centroids)

        for cid in range(self.K):
            if not np.any(assignments == cid):
                return float(_EMPTY_CLUSTER_PENALTY)

        if self.fitness_mode == 'knn_mae' and self.val_sample is not None:
            return fast_knn_mae(
                self.train_sample,
                self.val_sample,
                assignments,
                k=self.knn_k,
            )

        return self._latent_dev_fitness(centroids, assignments)

    def optimize(self) -> Dict[str, Any]:
        full_name, algo_cls = _resolve_centroid_algo_class(self.algo)
        pop_size = self.n_agents
        if full_name == 'IWO.OriginalIWO':
            pop_size = max(10, pop_size)
        sp = get_special_params(full_name, self.n_iter, pop_size) or {}
        sp.setdefault('epoch', self.n_iter)
        sp.setdefault('pop_size', pop_size)
        if full_name == 'IWO.OriginalIWO':
            sp['seed_max'] = max(4, min(int(sp.get('seed_max', 5)), pop_size // 2))

        problem = {
            'obj_func': self.fitness,
            'bounds': FloatVar(lb=self.lb, ub=self.ub),
            'minmax': 'min',
            'log_to': None,
            'save_population': False,
        }

        model = algo_cls(**sp)
        try:
            model.solve(problem, seed=self.seed)
        except TypeError:
            model.solve(problem)

        best_fit = float(model.g_best.target.fitness)
        individual = np.asarray(model.g_best.solution, dtype=np.float64)
        centroids = individual.reshape(self.K, self.n_features)
        assignments = self._assign(centroids)

        fit_label = 'sample_mae' if self.fitness_mode == 'knn_mae' else 'latent_dev'
        print(
            f"    CentroidOptimizer ({self.algo}/{full_name}): "
            f"{fit_label}={best_fit:.6f}",
            flush=True,
        )

        return {
            'centroids': centroids,
            'assignments': assignments,
            'fitness': best_fit,
            'algo': self.algo,
            'fitness_mode': self.fitness_mode,
        }
