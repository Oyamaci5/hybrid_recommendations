"""
per_cluster_wnmf_optimizer.py
=============================
Meta-algoritma ile her küme için WNMF hiperparametrelerini (latent_dim, lr, reg, epochs)
optimize eder.

Yaklaşım C: Kümeleme sabit (B0_KMEANS veya başka bir algo), meta-sezgisel her kümenin
WNMF parametrelerini bulur.

Arama uzayı: K küme × 4 parametre = 4K boyutlu sürekli vektör.
- latent_dim:    [3, 100]   → integer, küçük kümeler az boyut ister
- learning_rate: [0.001, 0.2] → log-uniform
- regularization:[0.0001, 0.5] → log-uniform
- n_epochs:      [10, 200]  → integer

Kullanım (generate_assignments.py içinden):
    python generate_assignments.py --dataset 100k --lof --k 7 \\
        --fitness per_cluster_wnmf --algo B1_HHO B2_HGS H4_MFO+HHO \\
        --per-cluster-wnmf-assign B0_KMEANS

Kullanım (doğrudan test):
    python per_cluster_wnmf_optimizer.py --dataset 100k \\
        --assign-dir results/assignments_lof/ml100k/B0_KMEANS_pruneu5_i10_zscore_k7
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mealpy import FloatVar

_MEALPY_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_MEALPY_DIR)
_WNMF_DIR = os.path.join(_REPO_ROOT, 'wnmf')
if _MEALPY_DIR not in sys.path:
    sys.path.insert(0, _MEALPY_DIR)
if _WNMF_DIR not in sys.path:
    sys.path.insert(0, _WNMF_DIR)

from mealpy_comparison_v2 import get_special_params

# Default parameter ranges for per-cluster optimization
PC_LATENT_MIN     = 3
PC_LATENT_MAX     = 100
PC_LR_MIN         = 0.001
PC_LR_MAX         = 0.2
PC_REG_MIN        = 0.0001
PC_REG_MAX        = 0.5
PC_EPOCHS_MIN     = 10
PC_EPOCHS_MAX     = 200

# Optimization objective weighting
PC_REG_WEIGHT     = 1.0        # latent_dim small → penalty
PC_VAL_SAMPLE     = 800        # validation sample size for fast eval
PC_GLOBAL_EPOCHS  = 60         # global V training epochs (fewer for speed)
PC_RANDOM_SEED    = 42


class PerClusterWNMFFitness:
    """
    Her küme için WNMF parametrelerini optimize eden fitness sınıfı.

    Arama vektörü yapısı (sürekli, mealpy FloatVar):
        [ld₁ ... ld_K,  lr₁ ... lr_K,  reg₁ ... reg_K,  ec₁ ... ec_K]
        Log-scale dönüşümler:
          ld  ∈ [ld_min, ld_max]        → integer
          lr  ∈ log10[lr_min, lr_max]   → 10^x
          reg ∈ log10[reg_min, reg_max] → 10^x
          ec  ∈ [ec_min, ec_max]        → integer
    """

    def __init__(
        self,
        train: np.ndarray,
        test: np.ndarray,
        assignments: np.ndarray,
        n_items: int,
        K: int,
        *,
        gray_mask: Optional[np.ndarray] = None,
        n_epochs_global: int = PC_GLOBAL_EPOCHS,
        random_seed: int = PC_RANDOM_SEED,
        use_bias: bool = True,
        val_sample: int = PC_VAL_SAMPLE,
        reg_weight: float = PC_REG_WEIGHT,
        verbose: bool = False,
    ):
        self.train = train
        self.test = test
        self.assignments = np.asarray(assignments, dtype=np.int32)
        self.n_items = int(n_items)
        self.K = int(K)
        self.gray_mask = (
            np.asarray(gray_mask, dtype=bool)
            if gray_mask is not None
            else np.zeros(len(assignments), dtype=bool)
        )
        self.n_epochs_global = int(n_epochs_global)
        self.random_seed = int(random_seed)
        self.use_bias = bool(use_bias)
        self.val_sample = int(val_sample)
        self.reg_weight = float(reg_weight)  # latent_dim küçüklük cezası
        self.verbose = bool(verbose)

        self.n_users = len(self.assignments)

        # Pre-compute: kümeleri böl (bir kez)
        self._cluster_train: Dict[int, np.ndarray] = {}
        self._cluster_test: Dict[int, np.ndarray] = {}
        self._cluster_users: Dict[int, List[int]] = {}
        self._build_cluster_splits()

        # Global V cache (sadece latent_dim değişirse rebuild)
        self._global_V_cache: Dict[int, Tuple[np.ndarray, float, np.ndarray]] = {}
        self._eval_rng = np.random.default_rng(random_seed)

        # Validation samples (her küme için örneklenmiş, hız için)
        self._val_samples: Dict[int, np.ndarray] = {}
        self._build_val_samples()

        # Global mean (fallback tahmin için)
        self._global_mean = float(np.mean(train[:, 2]))

    def _build_cluster_splits(self):
        """Train/test'i kümelere böl."""
        n_clusters = int(max(int(self.assignments.max()), 0)) + 1
        # Train split
        for row in self.train:
            u, i, r = int(row[0]), int(row[1]), float(row[2])
            cid = int(self.assignments[u]) if u < self.n_users else 0
            self._cluster_train.setdefault(cid, []).append((u, i, r))

        for cid in list(self._cluster_train.keys()):
            self._cluster_train[cid] = np.array(self._cluster_train[cid], dtype=np.float32)

        # Test split
        for row in self.test:
            u, i, r = int(row[0]), int(row[1]), float(row[2])
            cid = int(self.assignments[u]) if u < self.n_users else 0
            self._cluster_test.setdefault(cid, []).append((u, i, r))

        for cid in list(self._cluster_test.keys()):
            self._cluster_test[cid] = np.array(self._cluster_test[cid], dtype=np.float32)

        # Users
        for u, cid in enumerate(self.assignments):
            self._cluster_users.setdefault(int(cid), []).append(u)

        if self.verbose:
            for cid in range(n_clusters):
                n_train = len(self._cluster_train.get(cid, []))
                n_test = len(self._cluster_test.get(cid, []))
                n_users = len(self._cluster_users.get(cid, []))
                print(f"  Küme {cid}: {n_users} kullanıcı, {n_train} train, {n_test} test")

    def _build_val_samples(self):
        """Her küme için validation örnekleri (hızlı fitness için)."""
        for cid, c_train in self._cluster_train.items():
            if len(c_train) == 0:
                self._val_samples[cid] = np.zeros((0, 3), dtype=np.float32)
                continue
            n_sample = min(self.val_sample, len(c_train))
            idx = self._eval_rng.choice(len(c_train), size=n_sample, replace=False)
            self._val_samples[cid] = c_train[idx].copy()

    # ----------------------------------------------------------------
    # Çözüm vektörü çözümleme
    # ----------------------------------------------------------------
    def _decode_params(self, x: np.ndarray) -> Dict[int, Dict[str, float]]:
        """
        Sürekli çözüm vektörünü küme bazlı parametrelere çevir.

        x yapısı: [ld₁...ld_K, log10(lr₁)...log10(lr_K), log10(reg₁)...log10(reg_K), ec₁...ec_K]
        """
        K = self.K
        params: Dict[int, Dict[str, float]] = {}

        for k in range(K):
            # latent_dim: [ld_min, ld_max] → round to int, at least 2
            ld_raw = float(x[k])
            ld = max(2, int(round(np.clip(ld_raw, PC_LATENT_MIN, PC_LATENT_MAX))))

            # learning_rate: log10 space → 10^x
            lr_raw = float(x[K + k])
            lr = float(np.clip(10.0 ** lr_raw, PC_LR_MIN, PC_LR_MAX))

            # regularization: log10 space → 10^x
            reg_raw = float(x[2 * K + k])
            reg = float(np.clip(10.0 ** reg_raw, PC_REG_MIN, PC_REG_MAX))

            # epochs: [ec_min, ec_max] → round to int
            ec_raw = float(x[3 * K + k])
            ec = max(5, int(round(np.clip(ec_raw, PC_EPOCHS_MIN, PC_EPOCHS_MAX))))

            params[k] = {
                'latent_dim': ld,
                'learning_rate': lr,
                'regularization': reg,
                'n_epochs_cluster': ec,
            }

        return params

    # ----------------------------------------------------------------
    # Global V (cached by latent_dim — en pahalı işlem)
    # ----------------------------------------------------------------
    def _get_global_V(self, latent_dim: int) -> Tuple[np.ndarray, float, np.ndarray]:
        """Global V'yi eğit veya cache'den al."""
        if latent_dim in self._global_V_cache:
            return self._global_V_cache[latent_dim]

        from wnmf_model import WNMFModel

        model = WNMFModel(
            n_users=self.n_users,
            n_items=self.n_items,
            latent_dim=latent_dim,
            learning_rate=0.01,
            regularization=0.01,
            n_epochs=self.n_epochs_global,
            random_seed=self.random_seed,
            use_bias=self.use_bias,
        )

        # Gray sheep'leri düşük ağırlıkla fit et
        if np.any(self.gray_mask):
            user_ids = self.train[:, 0].astype(np.int32)
            sw = np.where(self.gray_mask[user_ids], 0.1, 1.0).astype(np.float32)
            model.fit(self.train, sample_weights=sw, verbose=False)
        else:
            model.fit(self.train, verbose=False)

        V = model.V.copy()
        mu = float(model.mu)
        b_i = model.b_i.copy()

        self._global_V_cache[latent_dim] = (V, mu, b_i)
        return V, mu, b_i

    # ----------------------------------------------------------------
    # Fitness evaluation
    # ----------------------------------------------------------------
    def evaluate(self, x: np.ndarray) -> float:
        """
        Çözüm vektörü için validation MAE hesapla.

        Pipeline:
          1. x → per-cluster params (decode)
          2. En büyük latent_dim ile global V eğit (bir kez)
          3. Her küme için: o kümenin latent_dim'i ile U eğit
             - Global V'yi o kümenin latent_dim'ine kırp/genişlet
          4. Validation MAE hesapla
          5. Regularization penalty ekle (aşırı büyük latent_dim cezası)
        """
        params = self._decode_params(x)

        # Tüm kümeler için en büyük latent_dim'i bul
        # Global V en büyük boyutta eğitilir, küçük kümeler ilk ld sütunlarını kullanır
        max_ld = max(p['latent_dim'] for p in params.values())
        max_ld = max(max_ld, 5)

        # Global V eğit (en büyük latent_dim ile)
        V_big, mu_global, b_i_global = self._get_global_V(max_ld)

        from wnmf_model import ClusterWNMF
        from wnmf_utils import remap_user_ids, split_by_cluster

        total_error = 0.0
        total_count = 0

        for cid in range(self.K):
            p = params[cid]
            ld = p['latent_dim']
            lr = p['learning_rate']
            reg = p['regularization']
            ec = p['n_epochs_cluster']

            # Bu kümenin train verisi
            c_train = self._cluster_train.get(cid)
            if c_train is None or len(c_train) < 5:
                continue

            # Global V'yi bu kümenin latent_dim'ine kırp
            V_k = V_big[:, :ld].copy()

            # Remap user IDs
            c_train_r, _, uid_map, n_loc = remap_user_ids(
                c_train,
                np.empty((0, 3), dtype=np.float32),
                self.n_items,
            )

            if n_loc < 2:
                continue

            # Cluster model
            cluster_model = ClusterWNMF(
                n_users=n_loc,
                n_items=self.n_items,
                latent_dim=ld,
                V_shared=V_k,
                learning_rate=lr,
                regularization=reg,
                n_epochs=ec,
                random_seed=self.random_seed + int(cid),
                use_bias=self.use_bias,
                mu=mu_global,
                b_i_global=b_i_global[:self.n_items] if b_i_global is not None else None,
                cluster_ratings=c_train_r if self.use_bias else None,
            )
            cluster_model.fit_cluster_U(c_train_r, verbose=False)

            # Validation: bu kümenin val örnekleri üzerinde MAE
            c_val = self._val_samples.get(cid)
            if c_val is None or len(c_val) == 0:
                continue

            for row in c_val:
                u_orig = int(row[0])
                i = int(row[1])
                r = float(row[2])

                # Remap user_id
                u_local = uid_map.get(u_orig)
                if u_local is not None and 0 <= u_local < len(cluster_model.U) and 0 <= i < len(cluster_model.V):
                    if self.use_bias:
                        b_i_val = float(cluster_model.b_i[i]) if i < len(cluster_model.b_i) else 0.0
                        pred = (
                            cluster_model.mu
                            + cluster_model.b_u[u_local]
                            + b_i_val
                            + float(np.dot(cluster_model.U[u_local], cluster_model.V[i]))
                        )
                    else:
                        pred = float(np.dot(cluster_model.U[u_local], cluster_model.V[i]))
                else:
                    pred = self._global_mean

                total_error += abs(r - np.clip(pred, 1.0, 5.0))
                total_count += 1

        if total_count == 0:
            return 1e9  # büyük penalty

        mae = total_error / total_count

        # Regularization: büyük latent_dim'lere hafif ceza
        avg_ld = np.mean([p['latent_dim'] for p in params.values()])
        reg_penalty = self.reg_weight * (avg_ld / 100.0)

        return float(mae + reg_penalty)

    # ----------------------------------------------------------------
    # Mealpy problem builder
    # ----------------------------------------------------------------
    def make_problem(self):
        """
        Mealpy uyumlu problem sözlüğü oluştur.

        Arama uzayı: 4K boyutlu sürekli vektör
        """
        K = self.K

        lb_list = []
        ub_list = []

        # latent_dim: [ld_min, ld_max]
        lb_list.extend([float(PC_LATENT_MIN)] * K)
        ub_list.extend([float(PC_LATENT_MAX)] * K)

        # learning_rate: log10 space
        lb_list.extend([np.log10(PC_LR_MIN)] * K)
        ub_list.extend([np.log10(PC_LR_MAX)] * K)

        # regularization: log10 space
        lb_list.extend([np.log10(PC_REG_MIN)] * K)
        ub_list.extend([np.log10(PC_REG_MAX)] * K)

        # n_epochs: [ec_min, ec_max]
        lb_list.extend([float(PC_EPOCHS_MIN)] * K)
        ub_list.extend([float(PC_EPOCHS_MAX)] * K)

        lb = np.array(lb_list, dtype=np.float64)
        ub = np.array(ub_list, dtype=np.float64)

        return {
            "obj_func": self.evaluate,
            "bounds": FloatVar(lb=lb.tolist(), ub=ub.tolist(), name="cluster_params"),
            "minmax": "min",
            "log_to": None,
            "save_population": False,
        }

    def make_flat_fitness_fn(self):
        """mealpy dışı optimizer'lar için düz fitness fonksiyonu."""
        return self.evaluate

    def fitness_label(self) -> str:
        return "per_cluster_wnmf_mae"


# ============================================================
# Helper: load assignments from disk (reuse from wnmf_utils)
# ============================================================
def _get_wnmf_utils():
    from wnmf_utils import (
        load_assignment,
        load_ratings_100k,
        load_ratings_100k_all,
        load_ratings_1m,
        split_by_cluster,
        remap_user_ids,
    )
    return (
        load_assignment,
        load_ratings_100k,
        load_ratings_100k_all,
        load_ratings_1m,
        split_by_cluster,
        remap_user_ids,
    )


def run_per_cluster_optimization(
    matrix: np.ndarray,
    K: int,
    assignments: np.ndarray,
    algo_info: dict,
    algo_map: dict,
    train_ratings: np.ndarray,
    test_ratings: np.ndarray,
    n_items: int,
    init: Optional[List[np.ndarray]] = None,
    opt_epoch: int = 30,
    opt_pop: int = 30,
    gray_mask: Optional[np.ndarray] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, Dict[int, Dict[str, float]]]:
    """
    Meta-algoritma ile per-cluster WNMF parametrelerini optimize et.

    Returns:
        best_solution: En iyi çözüm vektörü (4K boyutlu)
        best_fitness:  En iyi validation MAE
        best_params:   Küme bazlı decode edilmiş parametreler
    """
    fitness = PerClusterWNMFFitness(
        train=train_ratings,
        test=test_ratings,
        assignments=assignments,
        n_items=n_items,
        K=K,
        gray_mask=gray_mask,
        random_seed=seed,
        verbose=verbose,
    )

    problem = fitness.make_problem()

    # Popülasyon başlangıcı: mantıklı varsayılanlar etrafında
    if init is None or len(init) < opt_pop:
        rng = np.random.default_rng(seed)
        lb = np.array(problem["bounds"].lb)
        ub = np.array(problem["bounds"].ub)

        init = []
        # Default center: decent values
        center = np.zeros(4 * K, dtype=np.float64)
        center[:K] = 20.0                          # latent_dim ~ 20
        center[K:2*K] = np.log10(0.01)             # lr ~ 0.01
        center[2*K:3*K] = np.log10(0.01)           # reg ~ 0.01
        center[3*K:4*K] = 50.0                     # epochs ~ 50

        for i in range(opt_pop):
            noise = rng.normal(0, 0.3, 4 * K)  # log-space noise
            x = np.clip(center + noise, lb, ub)
            init.append(x)

    print(f"\n  Per-cluster WNMF optimization: K={K}, {opt_epoch} epochs, {opt_pop} agents")
    print(f"  Search space: {4*K} dimensions")
    print(f"  Fitness: validation MAE (sample={fitness.val_sample}/cluster)", flush=True)

    t0 = time.time()

    sp = get_special_params(algo_info['full_name'], opt_epoch, opt_pop)
    model = algo_info['class'](
        **(sp or {'epoch': opt_epoch, 'pop_size': opt_pop})
    )
    try:
        model.solve(problem, starting_solutions=init[:opt_pop])
    except TypeError:
        model.solve(problem)

    best_sol = np.asarray(model.g_best.solution, dtype=np.float64)
    best_fit = float(model.g_best.target.fitness)
    best_params = fitness._decode_params(best_sol)

    elapsed = time.time() - t0

    # Debug: cluster bazında çıktı
    for cid in sorted(best_params.keys()):
        p = best_params[cid]
        print(
            f"    Küme {cid:2d}: ld={p['latent_dim']:3d}  "
            f"lr={p['learning_rate']:.4f}  "
            f"reg={p['regularization']:.4f}  "
            f"epochs={p['n_epochs_cluster']:3d}"
        )

    print(
        f"  [{algo_info['full_name'].split('.')[0]}] "
        f"Per-cluster WNMF MAE: {best_fit:.6f}  ({elapsed:.1f}s)",
        flush=True,
    )

    return best_sol, best_fit, best_params


# ============================================================
# Standalone test / karşılaştırma
# ============================================================
def compare_with_baseline(
    train: np.ndarray,
    test: np.ndarray,
    assignments: np.ndarray,
    n_items: int,
    K: int,
    gray_mask: Optional[np.ndarray] = None,
    latent_dim: int = 20,
    lr: float = 0.01,
    reg: float = 0.01,
    epochs_cluster: int = 50,
    epochs_global: int = 100,
    seed: int = 42,
    use_bias: bool = True,
) -> Tuple[float, float]:
    """
    Baseline (uniform params) vs optimize edilmiş per-cluster params.
    Returns (baseline_mae, optimized_mae).
    """
    from wnmf_model import WNMFSharedV, WNMFModel
    from wnmf_utils import split_by_cluster, remap_user_ids

    print("\n=== Baseline (uniform params) ===")
    t0 = time.time()

    # Global V
    global_model = WNMFModel(
        n_users=len(assignments),
        n_items=n_items,
        latent_dim=latent_dim,
        learning_rate=lr,
        regularization=reg,
        n_epochs=epochs_global,
        random_seed=seed,
        use_bias=use_bias,
    )
    if gray_mask is not None and np.any(gray_mask):
        user_ids = train[:, 0].astype(np.int32)
        sw = np.where(gray_mask[user_ids], 0.1, 1.0).astype(np.float32)
        global_model.fit(train, sample_weights=sw, verbose=False)
    else:
        global_model.fit(train, verbose=False)

    V_global = global_model.V.copy()
    mu_global = float(global_model.mu)
    b_i_global = global_model.b_i.copy()

    cluster_train, _ = split_by_cluster(train, assignments, gray_mask)
    cluster_test, _ = split_by_cluster(test, assignments, gray_mask)

    total_err, total_n = 0.0, 0
    from wnmf_model import ClusterWNMF

    for cid, c_train in cluster_train.items():
        if len(c_train) < 5:
            continue
        c_train_r, _, uid_map, n_loc = remap_user_ids(
            c_train, np.empty((0, 3), dtype=np.float32), n_items,
        )
        cm = ClusterWNMF(
            n_users=n_loc, n_items=n_items, latent_dim=latent_dim,
            V_shared=V_global, learning_rate=lr, regularization=reg,
            n_epochs=epochs_cluster, random_seed=seed + int(cid),
            use_bias=use_bias, mu=mu_global, b_i_global=b_i_global,
            cluster_ratings=c_train_r if use_bias else None,
        )
        cm.fit_cluster_U(c_train_r, verbose=False)

        c_test = cluster_test.get(cid)
        if c_test is None or len(c_test) == 0:
            continue
        for row in c_test:
            u_orig, i, r = int(row[0]), int(row[1]), float(row[2])
            u_local = uid_map.get(u_orig)
            if u_local is not None and 0 <= u_local < n_loc:
                if use_bias:
                    pred = cm.mu + cm.b_u[u_local] + (
                        cm.b_i[i] if i < len(cm.b_i) else 0
                    ) + float(np.dot(cm.U[u_local], cm.V[i]))
                else:
                    pred = float(np.dot(cm.U[u_local], cm.V[i]))
            else:
                pred = mu_global
            total_err += abs(r - np.clip(pred, 1.0, 5.0))
            total_n += 1

    baseline_mae = total_err / total_n if total_n > 0 else float('inf')
    print(f"  Baseline MAE: {baseline_mae:.6f}  ({time.time() - t0:.1f}s)")

    return baseline_mae, baseline_mae  # second will be replaced by optimized


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Per-cluster WNMF parameter optimizer")
    p.add_argument('--dataset', default='100k', choices=['100k', '1m'])
    p.add_argument('--assign-dir', required=True,
                   help='Assignments klasörü (assignments.npy içermeli)')
    p.add_argument('--algo', default='B1_HHO',
                   help='Meta-algoritma adı (mealpy full name, örn: HHO.OriginalHHO)')
    p.add_argument('--epoch', type=int, default=30)
    p.add_argument('--pop', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-bias', action='store_true')
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    load_assignment, load_ratings_100k, load_ratings_100k_all, load_ratings_1m, _, _ = _get_wnmf_utils()

    DATA_100K = os.path.join(_REPO_ROOT, 'data', 'ml-100k', 'u.data')
    DATA_100K_TRAIN = os.path.join(_REPO_ROOT, 'data', 'ml-100k', 'u1.base')
    DATA_100K_TEST = os.path.join(_REPO_ROOT, 'data', 'ml-100k', 'u1.test')
    DATA_1M = os.path.join(_REPO_ROOT, 'data', 'ml-1m', 'ratings.dat')

    if args.dataset == '100k':
        train_ratings, test_ratings = load_ratings_100k(DATA_100K_TRAIN, DATA_100K_TEST)
        n_items = 1682
    else:
        train_ratings, test_ratings = load_ratings_1m(DATA_1M)
        n_items = int(max(train_ratings[:, 1].max(), test_ratings[:, 1].max())) + 1

    assignments, gray_mask = load_assignment(args.assign_dir)
    K = int(assignments.max()) + 1

    print(f"Dataset     : {args.dataset}")
    print(f"Assignments : {args.assign_dir}")
    print(f"K           : {K}")
    print(f"n_users     : {len(assignments)}")
    print(f"n_items     : {n_items}")

    # Build algo_map
    from mealpy_comparison_v2 import get_all_algorithms_v3
    algo_map = {a['full_name']: a for a in get_all_algorithms_v3()}

    from mealpy.evolutionary_based import GA
    algo_map['GA.EliteMultiGA'] = {'full_name': 'GA.EliteMultiGA', 'class': GA.EliteMultiGA}

    if args.algo in algo_map:
        algo_info = algo_map[args.algo]
    else:
        print(f"Algoritma '{args.algo}' bulunamadı. Kullanılabilir:", list(algo_map.keys())[:10])
        sys.exit(1)

    best_sol, best_fit, best_params = run_per_cluster_optimization(
        matrix=None,
        K=K,
        assignments=assignments,
        algo_info=algo_info,
        algo_map=algo_map,
        train_ratings=train_ratings,
        test_ratings=test_ratings,
        n_items=n_items,
        opt_epoch=args.epoch,
        opt_pop=args.pop,
        gray_mask=gray_mask,
        seed=args.seed,
        verbose=args.verbose,
    )

    # Compare with baseline
    compare_with_baseline(
        train=train_ratings,
        test=test_ratings,
        assignments=assignments,
        n_items=n_items,
        K=K,
        gray_mask=gray_mask,
    )
