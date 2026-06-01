"""
centroid_optimizer.py
=====================
Meta-algoritma ile optimal cluster centroid'larını bul.

Fitness modları:
  knn_mae       — ClusterPredictor (küme bias SGD + küme-içi cosine/pearson kNN) val MAE
  knn_mae_legacy — eski hızlı Pearson kNN (örneklemeli)
  latent_dev    — WNMF latent uzayında ortalama sapma (proxy)

generate_assignments._run_one_core içinde --algo etiketi (HA_AVOAHGS, B1_HHO, …)
centroid aramasını yürütür; ardından opsiyonel KMeans Lloyd (kmref) uygulanır.
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

# Geriye dönük: doğrudan CentroidOptimizer.optimize() çağrıları için
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
    """Örneklenmiş val triplets üzerinde küme-içi kNN MAE."""
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


def cluster_predictor_mae(
    train_ratings: np.ndarray,
    val_sample: np.ndarray,
    assignments: np.ndarray,
    n_users: int,
    n_items: int,
    *,
    k_neighbors: int = 20,
    sim_metric: str = 'cosine',
    min_support: int = 3,
    bias_epochs: int = 5,
    bias_lr: float = 0.005,
    bias_reg: float = 0.02,
) -> float:
    """Küme ataması için ClusterPredictor fit + val örneklemesi üzerinde MAE."""
    val_sample = np.asarray(val_sample, dtype=np.float64)
    if val_sample.size == 0:
        return float(_EMPTY_CLUSTER_PENALTY)

    assignments = np.asarray(assignments, dtype=np.int32)
    if len(assignments) != int(n_users):
        return float(_EMPTY_CLUSTER_PENALTY)

    wnmf_dir = os.path.join(_REPO_ROOT, 'wnmf')
    if wnmf_dir not in sys.path:
        sys.path.insert(0, wnmf_dir)
    from cluster_predictor import ClusterPredictor, build_rating_matrix

    R = build_rating_matrix(train_ratings, int(n_users), int(n_items))
    pred = ClusterPredictor(
        k_neighbors=max(1, int(k_neighbors)),
        sim_metric=(sim_metric or 'cosine').strip().lower(),
        min_support=max(1, int(min_support)),
        bias_epochs=max(1, int(bias_epochs)),
        bias_lr=float(bias_lr),
        bias_reg=float(bias_reg),
        bias_seed=42,
    )
    pred.fit(R, assignments)
    mae, _ = pred.evaluate(val_sample)
    return float(mae)


def _rating_bounds(train_ratings: np.ndarray) -> Tuple[int, int]:
    tr = np.asarray(train_ratings, dtype=np.float64)
    if tr.size == 0:
        return 0, 0
    return int(tr[:, 0].max()) + 1, int(tr[:, 1].max()) + 1


def _as_w_matrix(arr, source: str) -> np.ndarray:
    W = np.asarray(arr, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError(f"{source}: 2D W/U matrisi bekleniyor, shape={W.shape}")
    return W


def _extract_w_from_model_obj(obj, source: str = 'model') -> np.ndarray:
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
    """WNMF kullanıcı latent matrisini (W / U / user_factors) yükle."""
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


class CentroidFitnessEvaluator:
    """
    WNMF U uzayında centroid → assignment → downstream fitness (knn_mae / latent_dev).
    generate_assignments içindeki meta-algoritmalar bu obj_func ile mealpy problem'i çözer.
    """

    def __init__(
        self,
        W_matrix: np.ndarray,
        K: int,
        *,
        train_ratings: Optional[np.ndarray] = None,
        val_ratings: Optional[np.ndarray] = None,
        n_train_sample: int = 500,
        n_val_sample: int = 200,
        knn_k: int = 20,
        fitness_mode: str = 'knn_mae',
        knn_sim_metric: str = 'cosine',
        min_common: int = 3,
        bias_epochs: int = 5,
        use_native_predictor: bool = True,
        seed: int = 42,
    ):
        self.W = np.asarray(W_matrix, dtype=np.float64)
        self.K = int(K)
        self.n_users, self.n_features = self.W.shape
        self.fitness_mode = (fitness_mode or 'knn_mae').strip().lower()
        self.knn_k = max(1, int(knn_k))
        self.knn_sim_metric = (knn_sim_metric or 'cosine').strip().lower()
        self.min_common = max(1, int(min_common))
        self.bias_epochs = max(1, int(bias_epochs))
        self.use_native_predictor = bool(use_native_predictor)
        self.rng = np.random.default_rng(int(seed))

        self.train_full: Optional[np.ndarray] = None
        self.train_search: Optional[np.ndarray] = None
        self.val_sample: Optional[np.ndarray] = None
        self.n_items_fit = 0

        if self.fitness_mode in ('knn_mae', 'knn_mae_legacy'):
            train_ratings = (
                np.asarray(train_ratings, dtype=np.float64)
                if train_ratings is not None
                else np.zeros((0, 3), dtype=np.float64)
            )
            val_ratings = (
                np.asarray(val_ratings, dtype=np.float64)
                if val_ratings is not None
                else np.zeros((0, 3), dtype=np.float64)
            )
            self.train_full = train_ratings
            _, self.n_items_fit = _rating_bounds(train_ratings)
            self.val_sample = sample_ratings(val_ratings, int(n_val_sample), self.rng)

            if self.use_native_predictor and self.fitness_mode == 'knn_mae':
                # Arama sırasında örneklemeli train → fitness hızlanır
                self.train_search = sample_ratings(
                    train_ratings, int(n_train_sample), self.rng,
                )
                print(
                    f"  CentroidFitness knn_mae (ClusterPredictor): "
                    f"train_search={len(self.train_search):,}, "
                    f"train_full={len(self.train_full):,}, "
                    f"val_sample={len(self.val_sample)}, "
                    f"k={self.knn_k}, sim={self.knn_sim_metric}, "
                    f"bias_ep={self.bias_epochs}",
                    flush=True,
                )
            else:
                self.train_search = sample_ratings(
                    train_ratings, int(n_train_sample), self.rng,
                )
                print(
                    f"  CentroidFitness knn_mae_legacy: "
                    f"train_sample={len(self.train_search)}, "
                    f"val_sample={len(self.val_sample)}, k={self.knn_k}",
                    flush=True,
                )

        col_min = self.W.min(axis=0)
        col_max = self.W.max(axis=0)
        self.lb = np.tile(col_min, self.K).tolist()
        self.ub = np.tile(col_max, self.K).tolist()

    def fitness_label(self) -> str:
        if self.fitness_mode == 'knn_mae':
            return 'cluster_predictor_mae'
        if self.fitness_mode == 'knn_mae_legacy':
            return 'sample_mae'
        return 'latent_dev'

    def make_problem(self) -> dict:
        """mealpy model.solve() için problem sözlüğü."""
        return {
            'obj_func': self.fitness,
            'bounds': FloatVar(lb=self.lb, ub=self.ub, name='centroids'),
            'minmax': 'min',
            'log_to': None,
            'save_population': False,
        }

    def make_flat_fitness_fn(self):
        """LF-HHO / SFOA gibi düz vektör fitness bekleyen optimizörler için."""
        return self.fitness

    def assign(self, centroids: np.ndarray) -> np.ndarray:
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

    def _knn_mae_for_assignments(
        self,
        assignments: np.ndarray,
        *,
        use_full_train: bool = False,
    ) -> float:
        if self.val_sample is None or self.val_sample.size == 0:
            return float(_EMPTY_CLUSTER_PENALTY)

        if self.fitness_mode == 'knn_mae' and self.use_native_predictor:
            train_use = (
                self.train_full
                if use_full_train and self.train_full is not None
                else self.train_search
            )
            if train_use is None or len(train_use) == 0:
                return float(_EMPTY_CLUSTER_PENALTY)
            return cluster_predictor_mae(
                train_use,
                self.val_sample,
                assignments,
                self.n_users,
                max(self.n_items_fit, 1),
                k_neighbors=self.knn_k,
                sim_metric=self.knn_sim_metric,
                min_support=self.min_common,
                bias_epochs=self.bias_epochs,
            )

        train_use = self.train_search
        if train_use is None or len(train_use) == 0:
            return float(_EMPTY_CLUSTER_PENALTY)
        return fast_knn_mae(
            train_use,
            self.val_sample,
            assignments,
            k=self.knn_k,
            min_common=self.min_common,
        )

    def fitness(self, individual) -> float:
        centroids = np.asarray(individual, dtype=np.float64).reshape(
            self.K, self.n_features,
        )
        assignments = self.assign(centroids)

        for cid in range(self.K):
            if not np.any(assignments == cid):
                return float(_EMPTY_CLUSTER_PENALTY)

        if self.fitness_mode in ('knn_mae', 'knn_mae_legacy'):
            return self._knn_mae_for_assignments(assignments, use_full_train=False)

        return self._latent_dev_fitness(centroids, assignments)

    def evaluate_solution(
        self,
        flat_solution: np.ndarray,
        *,
        use_full_train: bool = False,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Çözüm vektörü için fitness + centroid + assignment."""
        centroids = np.asarray(flat_solution, dtype=np.float64).reshape(
            self.K, self.n_features,
        )
        assignments = self.assign(centroids)

        if self.fitness_mode in ('knn_mae', 'knn_mae_legacy'):
            fit = self._knn_mae_for_assignments(
                assignments, use_full_train=use_full_train,
            )
        else:
            fit = self._latent_dev_fitness(centroids, assignments)

        return float(fit), centroids, assignments


class CentroidOptimizer:
    """
    Geriye dönük uyumluluk: yalnızca MFO/IWO/HA ile doğrudan centroid araması.
    Yeni kod: generate_assignments --algo + CentroidFitnessEvaluator kullanır.
    """

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
        knn_sim_metric: str = 'cosine',
        min_common: int = 3,
        bias_epochs: int = 5,
        use_native_predictor: bool = True,
    ):
        self.evaluator = CentroidFitnessEvaluator(
            W_matrix,
            K,
            train_ratings=train_ratings,
            val_ratings=val_ratings,
            n_train_sample=n_train_sample,
            n_val_sample=n_val_sample,
            knn_k=knn_k,
            fitness_mode=fitness_mode,
            knn_sim_metric=knn_sim_metric,
            min_common=min_common,
            bias_epochs=bias_epochs,
            use_native_predictor=use_native_predictor,
            seed=seed,
        )
        self.n_agents = max(5, int(n_agents))
        self.n_iter = max(1, int(n_iter))
        self.algo = (algo or 'MFO').strip().upper()
        self.seed = int(seed)
        self.fitness_mode = self.evaluator.fitness_mode

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

        problem = self.evaluator.make_problem()
        model = algo_cls(**sp)
        try:
            model.solve(problem, seed=self.seed)
        except TypeError:
            model.solve(problem)

        best_fit, centroids, assignments = self.evaluator.evaluate_solution(
            model.g_best.solution,
            use_full_train=(self.fitness_mode == 'knn_mae'),
        )

        fit_label = self.evaluator.fitness_label()
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
