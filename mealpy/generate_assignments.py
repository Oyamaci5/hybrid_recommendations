"""
generate_assignments.py
=======================
ALGO_CONFIG listesindeki algoritmalar için (başta B1/B2/H1/H4, sonra B3, hibritler, LIT_*)
ML-100K ve ML-1M üzerinde tek run çalıştırır, assignment kaydeder.

Gray sheep tespiti:
    Varsayılan : Sabit 80. percentile (~%20 gray sheep)
    --lof flag  : LOF tabanlı adaptif threshold (önerilen)

Çıktı yapısı:
    --lof verilmezse : results/assignments/
    --lof verilirse  : results/assignments_lof/

    Her ikisinde de:
    ├── ml100k/
    │   ├── B1_HHO/              ← K=7 (default, eski düzen)
    │   ├── B1_HHO_pruneu5_..._minmax_svd20_k14/  ← --k 14 (yeni düzen)
    │   ├── B1_HHO_k70/          ← K=70 (eski düzen, args yok)
    │   └── H4_MFO+HHO/
    └── ml1m/
        └── ...

Kullanım:
    python generate_assignments.py                            # varsayılan
    python generate_assignments.py --lof                      # LOF gray sheep
    python generate_assignments.py --dataset 100k --algo H4_MFO+HHO
    python generate_assignments.py --last-only               # sadece son algoritma
    python generate_assignments.py --lof --k 70               # LOF + K=70
    python generate_assignments.py --lof --k 20 30 50 90     # aynı koşuda birden fazla K
    python generate_assignments.py --k-100k 70 --k-1m 120     # ayrı K (--k çoklu ile birlikte kullanılmaz)
    python generate_assignments.py --lof --n-neighbors 15 --contamination 0.1
    python generate_assignments.py --jobs 4              # algoritmaları paralel süreçte
    python generate_assignments.py --dataset 100k --paper-mode --algo LIT_GOA --k 30
    python generate_assignments.py --dataset 100k --algo B_AVOA --cluster-metric fuzzy --no-gray-sheep --k 13
    #   → AVOA, FCM objective minimize eder; FCM iterasyonuyla centroid rafine edilir; argmax hard assignment
"""

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(BASE_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, BASE_DIR)
_OPT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'optimizers')
if _OPT_DIR not in sys.path:
    sys.path.insert(0, _OPT_DIR)

try:
    from assignment_db import (
        assign_suffix_from_save_dir,
        finish_run,
        init_db,
        save_assignment as db_save,
        start_run,
    )
    init_db()
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False
    finish_run = None
    start_run = None

from mealpy_comparison_v2 import (
    load_movielens,
    mkmeans_plus_plus_init,
    make_fitness_function,
    MO_WEIGHT_PRESETS,
    compute_wcss_fast,
    compute_fcm_objective,
    detect_gray_sheep,
    get_all_algorithms_v3,
    get_special_params,
    euclidean_distance_batch,
    _fcm_memberships_from_dist,
)
from mealpy import FloatVar
from mealpy.evolutionary_based import GA
try:
    from ga_hho_optimizer import OriginalGAHHO  # pyright: ignore[reportMissingImports]
except ImportError:
    from ga_hho import OriginalGAHHO  # pyright: ignore[reportMissingImports]
from doa_optimizer import OriginalDOA  # pyright: ignore[reportMissingImports]
from lf_hho_optimizer import LevyHHO_Clustering  # pyright: ignore[reportMissingImports]
from sfoa_optimizer import SFOA_Clustering  # pyright: ignore[reportMissingImports]

# ============================================================
# AYARLAR
# ============================================================

DATA_100K = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-100k', 'u.data')
DATA_100K_TRAIN = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-100k', 'u1.base')
DATA_100K_TEST = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-100k', 'u1.test')
DATA_1M   = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-1m',   'ratings.dat')
DATA_FILMTRUST = os.path.join(
    os.path.dirname(BASE_DIR), 'data', 'filmtrust', 'ratings.txt',
)
N_USERS_100K = 943
N_ITEMS_100K = 1682

K_100K_DEFAULT = 7
K_1M_DEFAULT   = 7
K_FILMTRUST_DEFAULT = 15

GLOBAL_EPOCH   = 30
LOCAL_EPOCH    = 20
BASELINE_EPOCH = 200   # önceki baseline: 100
POP_SIZE       = 50    # önceki baseline: 30
SEED           = 42


def _resolve_train_hyperparams(args=None):
    """CLI / modül sabitlerinden meta-algoritma epoch ve pop boyutu."""
    if args is not None:
        be = getattr(args, 'baseline_epoch', None)
        ps = getattr(args, 'pop_size', None)
        if be is not None:
            baseline_epoch = int(be)
        else:
            baseline_epoch = BASELINE_EPOCH
        if ps is not None:
            pop_size = int(ps)
        else:
            pop_size = POP_SIZE
    else:
        baseline_epoch = BASELINE_EPOCH
        pop_size = POP_SIZE
    return baseline_epoch, GLOBAL_EPOCH, LOCAL_EPOCH, pop_size

# Early-stop (--early-stop): blok bazlı kontrol; erken durmazsa max epoch'ta biter
EARLY_STOP_MAX_EPOCH = 200   # 8 blok × patience 8 = en fazla 40 epoch erken; üst sınır 200
EARLY_STOP_BLOCK_SIZE = 5
EARLY_STOP_PATIENCE = 8      # 8 blok = 40 epoch iyileşme yoksa dur
EARLY_STOP_TOLERANCE = 1e-5


def _resolve_ha_epoch_policy(args, matrix, K, pop_size):
    """
    HA_AVOAHGS için epoch/early-stop parametrelerini üretir.

    Adaptif modda mantık:
      - min_epoch: problemin (K, d) karmaşıklığına göre alt sınır
      - max_epoch: n, d, K ve pop büyüklüğüne göre bütçe
      - patience: min_epoch'tan sonra ne kadar bekleyeceğimizi blok cinsinden belirler
      - tolerance: veri büyüdükçe (n*K) daha hassas iyileşme eşiği
    """
    if args is None:
        return (
            EARLY_STOP_MAX_EPOCH,
            EARLY_STOP_PATIENCE,
            EARLY_STOP_TOLERANCE,
            EARLY_STOP_BLOCK_SIZE,
            "fixed",
        )

    block_size = int(max(1, getattr(args, 'early_stop_block', EARLY_STOP_BLOCK_SIZE)))
    if not getattr(args, 'ha_adaptive_epoch', False):
        return (
            int(getattr(args, 'early_stop_max_epoch', EARLY_STOP_MAX_EPOCH)),
            int(getattr(args, 'early_stop_patience', EARLY_STOP_PATIENCE)),
            float(getattr(args, 'early_stop_tolerance', EARLY_STOP_TOLERANCE)),
            block_size,
            "fixed",
        )

    n_users = int(matrix.shape[0])
    n_feats = int(matrix.shape[1])
    K_eff = int(max(1, K))
    pop_eff = int(max(1, pop_size))

    min_epoch = int(max(
        30,
        np.ceil(2.5 * np.sqrt(K_eff) * np.log2(n_feats + 1.0)),
    ))

    target_epoch = int(np.ceil(
        min_epoch
        + 0.35 * np.sqrt((n_users * n_feats) / K_eff)
        + 0.60 * pop_eff
    ))

    cap = int(max(min_epoch + block_size, getattr(args, 'ha_adaptive_max_cap', 600)))
    max_epoch = int(min(cap, max(min_epoch + block_size, target_epoch)))

    span = max(1, max_epoch - min_epoch)
    patience = int(max(4, np.ceil(span / (3.0 * block_size))))
    min_tol = float(max(1e-8, getattr(args, 'ha_adaptive_min_tol', 1e-6)))
    tolerance = float(max(min_tol, 1e-4 / np.sqrt(n_users * K_eff)))

    note = (
        "adaptive "
        f"(min={min_epoch}, target={target_epoch}, max={max_epoch}, "
        f"pat={patience}, tol={tolerance:.2e}, block={block_size})"
    )
    return max_epoch, patience, tolerance, block_size, note


LOF_N_NEIGHBORS   = 20
LOF_CONTAMINATION = 'auto'

ALGO_CONFIG = [
    ('B0_KMEANS',  None, None),
    ('B1_HHO',     'HHO.OriginalHHO', None),
    ('B2_HGS',     'HGS.OriginalHGS', None),
    ('H1_HHO+HGS', 'HHO.OriginalHHO', 'HGS.OriginalHGS'),
    ('H4_MFO+HHO', 'MFO.OriginalMFO', 'HHO.OriginalHHO'),
    ('B3_MFO',     'MFO.OriginalMFO', None),
    ('H9_QSA+CDO',  'QSA.OriginalQSA',  'CDO.OriginalCDO'),
    ('H12_MFO+CDO', 'MFO.OriginalMFO',  'CDO.OriginalCDO'),
    ('H13_HHO+GAop', 'HHO.OriginalHHO', 'GAop'),
    ('H5_GAHHO',   'GAHHO.OriginalGAHHO', None),
    ('H5_EliteGA+HHO', 'GA.EliteMultiGA', 'HHO.OriginalHHO'),
    ('DOA', 'DOA.OriginalDOA', None),
    ('LF_HHO', 'LF_HHO', None),
    ('IWO_HHO', 'IWO_HHO', None),
    ('SFOA', 'SFOA', None),
    ('SFOA_06', 'SFOA_06', None),
    ('LIT_CIRCLESA', 'CircleSA.OriginalCircleSA', None),
    ('AGTO', 'AGTO.OriginalAGTO', None),
    ('LIT_GOA', 'GOA.OriginalGOA', None),
    ('LIT_GWO', 'GWO.OriginalGWO', None),
    ('LIT_SSA', 'SSA.OriginalSSA', None),
    ('LIT_PSO', 'PSO.OriginalPSO', None),
    ('LIT_CSO', 'CSO.OriginalCSO', None),
    ('HA_AVOAHGS', None, None),
    ('B_AVOA', 'AVOA.OriginalAVOA', None),
]

ALGO_LABELS = [c[0] for c in ALGO_CONFIG]

# hybrid_test / rapor ile aynı etiketler; Wilcoxon çiftleri (klasör adı = ilk sütun)
WILCOXON_PAIRS = [
    ('H5_EliteGA+HHO', 'H4_MFO+HHO', 'GA global vs MFO global'),
    ('H5_EliteGA+HHO', 'B1_HHO',     'H5 vs HHO tek'),
    ('H5_EliteGA+HHO', 'B3_MFO',     'H5 vs MFO tek'),
]


def _resolve_pool_workers(requested: Optional[int], n_tasks: int) -> int:
    if n_tasks <= 0:
        return 1
    cpu = os.cpu_count() or 1
    if requested is None or requested <= 0:
        cap = cpu
    else:
        cap = requested
    return max(1, min(cap, n_tasks))


# ============================================================
# GRAY SHEEP TESPİTİ — İKİ MOD
# ============================================================

def detect_gray_sheep_percentile(matrix, assignments, solution, K,
                                 metric: str = 'pearson'):
    """
    Sabit 80. percentile ile gray sheep tespiti.
    Her zaman ~%20 gray sheep üretir.
    Orijinal detect_gray_sheep fonksiyonunu çağırır.
    """
    return detect_gray_sheep(matrix, assignments, solution, K, metric=metric)


def detect_gray_sheep_lof(matrix, assignments, n_neighbors=LOF_N_NEIGHBORS,
                          contamination=LOF_CONTAMINATION):
    """
    LOF tabanlı adaptif gray sheep tespiti.

    Sabit percentile yerine LOF kullanmanın avantajı:
    - Threshold veriden otomatik belirlenir
    - Gray sheep sayısı veri yapısına göre değişir
    - Jüri sorusuna net metodolojik cevap verilebilir:
      'Neden %20?' değil, 'LOF skoru eşiği aştığında gray sheep'

    Parametreler
    ------------
    matrix        : (n_users, n_items) rating matrisi
    assignments   : (n_users,) küme atamaları
    n_neighbors   : LOF komşu sayısı
    contamination : 'auto' veya float (0-0.5)
    """
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import MinMaxScaler, normalize

    n_users  = len(matrix)
    features = _build_lof_features(matrix, assignments)

    lof = LocalOutlierFactor(
        n_neighbors   = min(n_neighbors, n_users - 1),
        contamination = contamination,
        metric        = 'euclidean',
        novelty       = False,
    )
    lof.fit(features)

    lof_scores  = -lof.negative_outlier_factor_
    predictions = lof.fit_predict(features)
    gray_mask   = predictions == -1
    original_gray_mask = gray_mask.copy()

    R_matrix = matrix
    R_centered = R_matrix - R_matrix.mean(axis=1, keepdims=True)
    R_centered = np.nan_to_num(R_centered)
    scaler = MinMaxScaler()
    R_scaled = scaler.fit_transform(R_centered)
    svd_n_components = min(30, R_scaled.shape[1])
    svd = TruncatedSVD(n_components=svd_n_components, random_state=42)
    U_svd = svd.fit_transform(R_scaled)
    U_svd = normalize(U_svd)

    lof_svd = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    svd_outliers = lof_svd.fit_predict(U_svd)

    svd_gray_mask = svd_outliers == -1
    gray_mask = gray_mask | svd_gray_mask

    print(f"Orijinal gray sheep : {original_gray_mask.sum()}")
    print(f"SVD'den eklenen     : {(svd_gray_mask & ~original_gray_mask).sum()}")
    print(f"Toplam gray sheep   : {gray_mask.sum()}")

    if gray_mask.sum() > 0 and (~gray_mask).sum() > 0:
        threshold = float(
            (lof_scores[gray_mask].min() + lof_scores[~gray_mask].max()) / 2
        )
    else:
        threshold = float(lof_scores.mean())

    return {
        'gray_sheep_mask' : gray_mask,
        'lof_scores'      : lof_scores,
        'threshold'       : threshold,
        'gray_sheep_count': int(gray_mask.sum()),
        'gray_sheep_ratio': float(gray_mask.mean()),
    }


def _build_lof_features(matrix, assignments):
    """
    LOF için 4 özellik: ort. rating, rating sayısı, std, küme içi fark.
    Doğrudan seyrek matrisi kullanmak yerine anlamlı özetler kullanılır.
    """
    n_users  = len(matrix)
    features = np.zeros((n_users, 4), dtype=np.float32)

    for u in range(n_users):
        rated   = matrix[u][matrix[u] > 0]
        n_rated = len(rated)

        if n_rated == 0:
            features[u] = [0, 0, 0, 1]
            continue

        avg_r = float(rated.mean())
        std_r = float(rated.std()) if n_rated > 1 else 0.0
        cnt_r = float(n_rated)

        cid          = int(assignments[u])
        cluster_mask = assignments == cid
        if cluster_mask.sum() > 1:
            c_mat   = matrix[cluster_mask]
            c_means = c_mat[:, matrix[u] > 0].mean(axis=0)
            diff    = float(np.abs(rated - c_means).mean())
        else:
            diff = 1.0

        features[u] = [avg_r, cnt_r / matrix.shape[1], std_r, diff]

    for col in range(features.shape[1]):
        s = features[:, col].std()
        if s > 0:
            features[:, col] = (features[:, col] - features[:, col].mean()) / s

    return features


# ============================================================
# ALGORİTMA ÇALIŞTIRICILAR
# ============================================================

def _centroid_search_bounds(matrix: np.ndarray, K: int, margin_frac: float = 0.1):
    """
    K merkez vektörü (düzleştirilmiş K×d) için boyut bazlı lb/ub.
    PCA gibi negatif koordinatlı uzaylarda alt sınır 0'a kilitlenmez.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    col_min = mat.min(axis=0)
    col_max = mat.max(axis=0)
    span = col_max - col_min
    margin = np.maximum(span * margin_frac, 1e-3)
    if float(col_min.min()) >= 0.0 and float(mat.max()) <= 10.0:
        lb_cols = np.maximum(col_min - margin, 0.0)
    else:
        lb_cols = col_min - margin
    ub_cols = col_max + margin
    lb = np.tile(lb_cols, int(K))
    ub = np.tile(ub_cols, int(K))
    return lb, ub


def _centroid_value_upper_bound(matrix: np.ndarray) -> float:
    """Geriye dönük uyumluluk; yeni kod _centroid_search_bounds kullanır."""
    _, ub = _centroid_search_bounds(matrix, K=1)
    return float(np.max(ub))


def _parse_mo_weights_arg(value):
    """--mo-weights: preset adı veya '0.4,0.3,0.3'."""
    from mealpy_comparison_v2 import _normalize_mo_weights
    if value is None or str(value).strip() == '':
        return MO_WEIGHT_PRESETS['default']
    return _normalize_mo_weights(str(value).strip())


def _fitness_config_from_args(args):
    """Meta WCSS/multi-objective fitness yapılandırması."""
    if args is None:
        return {'objective': 'multi'}
    cfg = {
        'objective': getattr(args, 'cluster_objective', 'multi') or 'multi',
        'mo_weights': _parse_mo_weights_arg(getattr(args, 'mo_weights', None)),
        'repulsion_lambda': float(
            getattr(args, 'centroid_repulsion_lambda', 0.0) or 0.0
        ),
        'repulsion_dmin': getattr(args, 'centroid_repulsion_dmin', None),
    }
    if cfg['repulsion_dmin'] is not None:
        cfg['repulsion_dmin'] = float(cfg['repulsion_dmin'])
    if getattr(args, 'cluster_metric', '') == 'fuzzy':
        cfg['fcm_m'] = _fcm_m_from_args(args)
    return cfg


def _make_problem(
    matrix,
    K,
    metric: str = 'pearson',
    cluster_objective: str = 'multi',
    fitness_config: dict | None = None,
):
    lb, ub = _centroid_search_bounds(matrix, K)
    fc = dict(fitness_config or {'objective': cluster_objective})
    fc.setdefault('objective', cluster_objective)
    return {
        "obj_func"       : make_fitness_function(
            matrix, K, metric=metric,
            fcm_m=float(fc.get('fcm_m', 2.0) or 2.0),
            **{k: v for k, v in fc.items() if k != 'fcm_m'},
        ),
        "bounds"         : FloatVar(
                               lb=lb.tolist(),
                               ub=ub.tolist(),
                               name="centroids"
                           ),
        "minmax"         : "min",
        "log_to"         : None,
        "save_population": False,
    }


def _resolve_problem(
    matrix,
    K,
    problem=None,
    metric: str = 'pearson',
    cluster_objective: str = 'multi',
    fitness_config: dict | None = None,
):
    """WCSS problemi veya dışarıdan verilen centroid fitness problemi."""
    if problem is not None:
        return problem
    return _make_problem(
        matrix, K, metric=metric, cluster_objective=cluster_objective,
        fitness_config=fitness_config,
    )


def run_single(algo_info, matrix, K, init, epoch, pop_size, metric: str = 'pearson',
               cluster_objective: str = 'multi', fitness_config: dict | None = None,
               problem=None):
    custom_problem = problem is not None
    problem = _resolve_problem(
        matrix, K, problem=problem,
        metric=metric, cluster_objective=cluster_objective,
        fitness_config=fitness_config,
    )
    sp      = get_special_params(algo_info['full_name'], epoch, pop_size)
    model   = algo_info['class'](**(sp or {'epoch': epoch, 'pop_size': pop_size}))
    try:
        model.solve(problem, starting_solutions=init[:pop_size])
    except TypeError:
        model.solve(problem)
    best_fit = float(model.g_best.target.fitness)
    metric_label = 'fitness' if custom_problem else 'WCSS'
    print(
        f"    {algo_info['full_name'].split('.')[0]} {metric_label}: {best_fit:.4f}"
    )
    return model.g_best.solution, best_fit


def _init_early_stop_population(init, pop_size, lb, ub):
    rng = np.random.default_rng(SEED)
    current_pop = [
        np.asarray(s, dtype=np.float64).copy() for s in init[:pop_size]
    ]
    while len(current_pop) < pop_size:
        current_pop.append(rng.uniform(lb, ub))
    return current_pop


def _run_block_early_stop(
    init,
    max_epoch,
    pop_size,
    patience,
    tolerance,
    block_size,
    algo_short,
    lb,
    ub,
    run_block,
):
    """
    Blok bazlı convergence early-stop.
    run_block(current_pop, block_epochs, block_idx) -> (block_best_sol, block_fit, new_pop)
    """
    block_size = max(1, int(block_size))
    max_epoch = max(1, int(max_epoch))
    patience = max(1, int(patience))

    best_fit = float('inf')
    best_sol = None
    no_improve = 0
    actual_epochs = 0
    convergence_history: list[dict] = []
    current_pop = _init_early_stop_population(init, pop_size, lb, ub)

    def _run_one_block(epochs: int, block_idx: int) -> bool:
        nonlocal best_fit, best_sol, no_improve, actual_epochs, current_pop

        block_best_sol, new_fit, current_pop = run_block(
            current_pop, epochs, block_idx,
        )
        actual_epochs += epochs
        convergence_history.append({
            'epoch': actual_epochs,
            'fitness': new_fit,
            'block': block_idx,
        })

        improvement = best_fit - new_fit
        if improvement > tolerance:
            best_fit = new_fit
            best_sol = np.asarray(block_best_sol, dtype=np.float64).copy()
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"    Block {block_idx}: fitness={new_fit:.6f}, "
            f"improvement={improvement:.2e}, "
            f"no_improve={no_improve}/{patience}"
        )
        return no_improve >= patience

    n_blocks = max_epoch // block_size
    remainder = max_epoch % block_size
    stopped = False

    for block in range(n_blocks):
        if _run_one_block(block_size, block + 1):
            print(f"    Early stop: {actual_epochs} epoch'ta converge")
            stopped = True
            break

    if not stopped and remainder > 0 and actual_epochs < max_epoch:
        if _run_one_block(remainder, n_blocks + 1):
            print(f"    Early stop: {actual_epochs} epoch'ta converge")

    if best_sol is None:
        raise RuntimeError(
            f"{algo_short}: early-stop sonrası geçerli çözüm yok"
        )

    print(
        f"    {algo_short} WCSS: {best_fit:.4f} "
        f"({actual_epochs}/{max_epoch} epoch, {len(convergence_history)} blok)"
    )
    return best_sol, best_fit, actual_epochs, convergence_history


def run_single_with_early_stop(
    algo_info,
    matrix,
    K,
    init,
    max_epoch,
    pop_size,
    metric: str = 'pearson',
    patience: int = EARLY_STOP_PATIENCE,
    tolerance: float = EARLY_STOP_TOLERANCE,
    block_size: int = EARLY_STOP_BLOCK_SIZE,
    cluster_objective: str = 'multi',
    fitness_config: dict | None = None,
    problem=None,
):
    """
    mealpy epoch callback olmadığı için optimizasyonu bloklar halinde çalıştırır.
    patience ardışık blok boyunca tolerance altında iyileşme yoksa durur.
    """
    problem = _resolve_problem(
        matrix, K, problem=problem,
        metric=metric, cluster_objective=cluster_objective,
        fitness_config=fitness_config,
    )
    lb = np.array(problem["bounds"].lb)
    ub = np.array(problem["bounds"].ub)
    algo_short = algo_info['full_name'].split('.')[0]

    def run_block(current_pop, block_epochs, _block_idx):
        sp_block = get_special_params(
            algo_info['full_name'], block_epochs, pop_size,
        )
        block_model = algo_info['class'](
            **(sp_block or {'epoch': block_epochs, 'pop_size': pop_size})
        )
        try:
            block_model.solve(problem, starting_solutions=current_pop)
        except TypeError:
            block_model.solve(problem)

        new_fit = float(block_model.g_best.target.fitness)
        new_sol = np.asarray(
            block_model.g_best.solution, dtype=np.float64,
        ).copy()
        pop = getattr(block_model, 'pop', None)
        if pop:
            new_pop = [
                np.asarray(a.solution, dtype=np.float64).copy() for a in pop
            ]
        else:
            new_pop = current_pop.copy()
            new_pop[0] = new_sol
        return new_sol, new_fit, new_pop

    return _run_block_early_stop(
        init, max_epoch, pop_size, patience, tolerance, block_size,
        algo_short, lb, ub, run_block,
    )


def run_ha_avoahgs_with_early_stop(
    matrix,
    K,
    init,
    max_epoch,
    pop_size,
    metric: str = 'pearson',
    patience: int = EARLY_STOP_PATIENCE,
    tolerance: float = EARLY_STOP_TOLERANCE,
    block_size: int = EARLY_STOP_BLOCK_SIZE,
    p1: float = 0.4,
    hgs_rate: float = 0.7,
    cluster_objective: str = 'multi',
    fitness_config: dict | None = None,
    problem=None,
):
    from optimizers.HA_AVOAHGS import HA_AVOAHGS

    problem = _resolve_problem(
        matrix, K, problem=problem,
        metric=metric, cluster_objective=cluster_objective,
        fitness_config=fitness_config,
    )
    lb = np.array(problem["bounds"].lb)
    ub = np.array(problem["bounds"].ub)

    def run_block(current_pop, block_epochs, _block_idx):
        model = HA_AVOAHGS(
            epoch=block_epochs,
            pop_size=pop_size,
            p1=p1,
            hgs_rate=hgs_rate,
        )
        try:
            model.solve(problem, starting_solutions=current_pop)
        except TypeError:
            model.solve(problem)

        new_fit = float(model.g_best.target.fitness)
        new_sol = np.asarray(model.g_best.solution, dtype=np.float64).copy()
        pop = getattr(model, 'pop', None)
        if pop:
            new_pop = [
                np.asarray(a.solution, dtype=np.float64).copy() for a in pop
            ]
        else:
            new_pop = current_pop.copy()
            new_pop[0] = new_sol
        return new_sol, new_fit, new_pop

    return _run_block_early_stop(
        init, max_epoch, pop_size, patience, tolerance, block_size,
        'HA_AVOAHGS', lb, ub, run_block,
    )


def run_iwo_hho_with_early_stop(
    matrix,
    K,
    init,
    max_epoch,
    pop_size,
    seed,
    metric: str = 'pearson',
    patience: int = EARLY_STOP_PATIENCE,
    tolerance: float = EARLY_STOP_TOLERANCE,
    block_size: int = EARLY_STOP_BLOCK_SIZE,
    cluster_objective: str = 'multi',
    fitness_config: dict | None = None,
    problem=None,
):
    from optimizers.iwo_hho import IWO_HHO_Clustering

    problem = _resolve_problem(
        matrix, K, problem=problem,
        metric=metric, cluster_objective=cluster_objective,
        fitness_config=fitness_config,
    )
    lb = np.array(problem["bounds"].lb)
    ub = np.array(problem["bounds"].ub)

    def run_block(current_pop, block_epochs, _block_idx):
        iwo_hho = IWO_HHO_Clustering(
            epoch=block_epochs,
            pop_size=pop_size,
            seed=seed,
        )
        try:
            best_sol, new_fit = iwo_hho.solve(
                problem, starting_solutions=current_pop,
            )
        except TypeError:
            best_sol, new_fit = iwo_hho.solve(problem)

        new_sol = np.asarray(best_sol, dtype=np.float64).copy()
        last_pop = getattr(iwo_hho, 'last_population', None)
        if last_pop:
            new_pop = [
                np.asarray(p, dtype=np.float64).copy() for p in last_pop
            ]
        else:
            new_pop = current_pop.copy()
            new_pop[0] = new_sol
        return new_sol, float(new_fit), new_pop

    return _run_block_early_stop(
        init, max_epoch, pop_size, patience, tolerance, block_size,
        'IWO_HHO', lb, ub, run_block,
    )


def run_hybrid(g_info, l_info, matrix, K, init, g_epoch, l_epoch, pop_size,
               metric: str = 'pearson', cluster_objective: str = 'multi',
               fitness_config: dict | None = None, problem=None):
    custom_problem = problem is not None
    problem = _resolve_problem(
        matrix, K, problem=problem,
        metric=metric, cluster_objective=cluster_objective,
        fitness_config=fitness_config,
    )

    sp_g    = get_special_params(g_info['full_name'], g_epoch, pop_size)
    g_model = g_info['class'](**(sp_g or {'epoch': g_epoch, 'pop_size': pop_size}))
    try:
        g_model.solve(problem, starting_solutions=init[:pop_size])
    except TypeError:
        g_model.solve(problem)
    best_g_sol = g_model.g_best.solution
    best_g_fit = float(g_model.g_best.target.fitness)
    metric_label = 'fitness' if custom_problem else 'WCSS'
    print(
        f"    Global ({g_info['full_name'].split('.')[0]}) "
        f"{metric_label}: {best_g_fit:.4f}"
    )

    sp_l    = get_special_params(l_info['full_name'], l_epoch, pop_size)
    l_model = l_info['class'](**(sp_l or {'epoch': l_epoch, 'pop_size': pop_size}))
    rng = np.random.default_rng(seed=42)
    lb  = np.array(problem["bounds"].lb)
    ub  = np.array(problem["bounds"].ub)
    search_range = ub - lb           # [0, 5] → range=5
    noise_scale  = 0.02 * search_range   # çözüm uzayının %2'si

    local_start = []
    for i in range(pop_size):
        if i == 0:
            local_start.append(best_g_sol.copy())   # en iyi nokta korunsun
        else:
            noise = rng.normal(0, noise_scale)
            noisy = np.clip(best_g_sol + noise, lb, ub)
            local_start.append(noisy)
    try:
        l_model.solve(problem, starting_solutions=local_start)
    except TypeError:
        l_model.solve(problem)
    best_l_fit = float(l_model.g_best.target.fitness)
    print(
        f"    Local  ({l_info['full_name'].split('.')[0]}) "
        f"{metric_label}: {best_l_fit:.4f}"
    )

    if best_l_fit < best_g_fit:
        best_sol, best_fit, improved = l_model.g_best.solution, best_l_fit, True
    else:
        best_sol, best_fit, improved = best_g_sol, best_g_fit, False
    print(
        f"    Lokal iyilestirdi: {improved}  ->  Final {metric_label}: {best_fit:.4f}"
    )
    return best_sol, best_fit


def run_parallel_hybrid(g_info, l_info, matrix, K, init, g_epoch, l_epoch, pop_size,
                        metric: str = 'pearson', cluster_objective: str = 'multi',
                        fitness_config: dict | None = None, problem=None):
    """Etikette '||' geçen hibrit: iki algoritmayı ayrı çalıştırıp en iyi WCSS'yi seçer."""
    sol_g, fit_g = run_single(
        g_info, matrix, K, init, g_epoch, pop_size,
        metric=metric, cluster_objective=cluster_objective,
        fitness_config=fitness_config, problem=problem,
    )
    sol_l, fit_l = run_single(
        l_info, matrix, K, init, l_epoch, pop_size,
        metric=metric, cluster_objective=cluster_objective,
        fitness_config=fitness_config, problem=problem,
    )
    if fit_l < fit_g:
        return sol_l, fit_l
    return sol_g, fit_g


def run_memetic_hybrid(base_info, matrix, K, init,
                       total_epoch, pop_size,
                       ga_inject_interval=10,
                       ga_crossover_rate=0.3,
                       ga_mutation_rate=0.1,
                       metric: str = 'pearson',
                       cluster_objective: str = 'multi',
                       fitness_config: dict | None = None,
                       problem=None):
    """
    Memetic hibrit: Sürü algoritması + GA operatör enjeksiyonu.

    Her ga_inject_interval epoch'ta:
    - Popülasyonun ga_crossover_rate kadarına crossover uygulanır
    - Popülasyonun ga_mutation_rate kadarına mutation uygulanır
    - Çeşitlilik korunur, erken yakınsama önlenir

    Literatür: "Hybrid Swarm-GA" / "Memetic Algorithm"

    base_info: ana sürü algoritması (HHO önerilen)
    """
    import numpy as np

    problem = _resolve_problem(
        matrix, K, problem=problem,
        metric=metric, cluster_objective=cluster_objective,
        fitness_config=fitness_config,
    )
    lb = np.array(problem["bounds"].lb)
    ub = np.array(problem["bounds"].ub)
    rng = np.random.default_rng(seed=42)
    dim = len(lb)

    # Başlangıç popülasyonu
    current_pop = list(init[:pop_size])
    best_sol = current_pop[0].copy()
    best_fit = float('inf')

    # Fitness hesapla
    obj_fn = problem["obj_func"]
    fits = [float(obj_fn(s)) for s in current_pop]
    best_idx = int(np.argmin(fits))
    best_fit = fits[best_idx]
    best_sol = current_pop[best_idx].copy()

    print(f"    Memetic hibrit: {total_epoch} epoch, "
          f"GA enjeksiyonu her {ga_inject_interval} epoch")

    epochs_done = 0
    round_num = 0

    while epochs_done < total_epoch:
        # Bu round'da kaç epoch çalışacak
        ep = min(ga_inject_interval, total_epoch - epochs_done)

        # Sürü algoritmasını ep epoch çalıştır
        sp = get_special_params(base_info['full_name'], ep, pop_size)
        model = base_info['class'](
            **(sp or {'epoch': ep, 'pop_size': pop_size})
        )
        try:
            model.solve(problem, starting_solutions=current_pop)
        except TypeError:
            model.solve(problem)

        # En iyiyi güncelle
        new_fit = float(model.g_best.target.fitness)
        if new_fit < best_fit:
            best_fit = new_fit
            best_sol = model.g_best.solution.copy()

        # Mevcut popülasyonu al (mealpy: Optimizer.pop, Agent.solution)
        pop = getattr(model, 'pop', None)
        if not pop:
            raise RuntimeError(
                f"{base_info['full_name']}: solve sonrası popülasyon yok; "
                'memetic hibrit bu optimizasyon sınıfı ile uyumlu değil.'
            )
        current_pop = [np.asarray(agent.solution, dtype=np.float64).copy() for agent in pop]
        fits = [float(obj_fn(s)) for s in current_pop]

        epochs_done += ep
        round_num += 1

        # GA operatör enjeksiyonu
        if epochs_done < total_epoch:
            n_cross = max(1, int(pop_size * ga_crossover_rate))
            n_mut   = max(1, int(pop_size * ga_mutation_rate))

            # Arithmetic Crossover
            cross_idx = rng.choice(pop_size, size=n_cross * 2,
                                   replace=False)
            for i in range(0, len(cross_idx) - 1, 2):
                p1 = current_pop[cross_idx[i]]
                p2 = current_pop[cross_idx[i+1]]
                alpha = rng.uniform(0.3, 0.7)
                child = np.clip(alpha * p1 + (1-alpha) * p2, lb, ub)
                child_fit = float(obj_fn(child))
                # Daha kötü ebeveynin yerine koy
                worse_idx = cross_idx[i] \
                    if fits[cross_idx[i]] > fits[cross_idx[i+1]] \
                    else cross_idx[i+1]
                if child_fit < fits[worse_idx]:
                    current_pop[worse_idx] = child
                    fits[worse_idx] = child_fit

            # Gaussian Mutation
            mut_idx = rng.choice(pop_size, size=n_mut, replace=False)
            scale = 0.05 * (ub - lb)
            for idx in mut_idx:
                mutant = np.clip(
                    current_pop[idx] + rng.normal(0, scale), lb, ub
                )
                mutant_fit = float(obj_fn(mutant))
                if mutant_fit < fits[idx]:
                    current_pop[idx] = mutant
                    fits[idx] = mutant_fit

            # En iyiyi güncelle
            round_best_idx = int(np.argmin(fits))
            if fits[round_best_idx] < best_fit:
                best_fit = fits[round_best_idx]
                best_sol = current_pop[round_best_idx].copy()

            print(f"    Round {round_num}: swarm={new_fit:.4f} "
                  f"GA_inject -> best={best_fit:.4f}")

    print(f"    Final WCSS: {best_fit:.4f}")
    return best_sol, best_fit


# ============================================================
# KAYDET
# ============================================================

def _sklearn_kmeans_init(args) -> str:
    """CLI --init değerini sklearn KMeans init= string'ine çevir."""
    v = 'kmeans++'
    if args is not None:
        v = getattr(args, 'init', 'kmeans++') or 'kmeans++'
    return 'random' if v == 'random' else 'k-means++'


def compute_memberships(X, centroids, m=2.0, max_iter=50, tol=1e-6):
    """
    Gerçek FCM iterasyonu: membership + centroid güncelleme döngüsü.
    GWO+FCM makalesiyle tam uyumlu:
        c_k = Σ(u_ik^m × x_i) / Σ(u_ik^m)
    
    Parametreler
    ------------
    X         : (n_users, n_features) kümeleme matrisi
    centroids : (K, n_features) başlangıç centroidleri
    m         : fuzzifier (genellikle 2.0)
    max_iter  : maksimum iterasyon
    tol       : centroid değişim eşiği (convergence)
    
    Döndürür
    --------
    memberships : (n_users, K) float32
    """
    centroids = np.asarray(centroids, dtype=np.float64).copy()
    X = np.asarray(X, dtype=np.float64)
    K = centroids.shape[0]

    for iteration in range(max_iter):
        old_centroids = centroids.copy()

        # Adım 1: Mesafe → Membership
        d2 = euclidean_distance_batch(X, centroids)   # (n_users, K)
        d  = np.sqrt(np.maximum(d2, 0.0))             # (n_users, K)
        memberships = _fcm_memberships_from_dist(d, m=m)  # (n_users, K)

        # Adım 2: Centroid güncelle — gerçek FCM formülü
        weights = memberships ** float(m)              # (n_users, K)
        denom   = weights.sum(axis=0)                  # (K,)
        for k in range(K):
            if denom[k] > 1e-10:
                centroids[k] = (weights[:, k:k+1] * X).sum(axis=0) / denom[k]

        # Adım 3: Convergence kontrolü
        change = float(np.linalg.norm(centroids - old_centroids))
        if change < tol:
            print(f"    FCM converge: {iteration+1} iterasyonda (change={change:.2e})")
            break
    else:
        print(f"    FCM max_iter={max_iter} tamamlandı (son change={change:.2e})")

    return memberships.astype(np.float32)

def save_assignment(assignments, gray_mask, best_sol, best_fit,
                    save_dir, extra_data=None, label=None, K=None, args=None,
                    run_id=None, seed=None, memberships=None, user_features=None):
    """
    Assignment dosyalarını kaydet.
    extra_data: LOF modunda lof_scores gibi ek veriler dict olarak geçilir.
    """
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, 'assignments.npy'),     assignments)
    if memberships is not None:
        np.save(os.path.join(save_dir, 'memberships.npy'), memberships)
    np.save(os.path.join(save_dir, 'gray_sheep_mask.npy'), gray_mask)
    np.save(os.path.join(save_dir, 'best_sol.npy'),        best_sol)
    if user_features is not None:
        np.save(os.path.join(save_dir, 'user_features.npy'), user_features)
        if args is not None and getattr(args, 'feature_extraction', None) == 'wnmf':
            np.save(
                os.path.join(save_dir, 'wnmf_user_vectors.npy'),
                user_features,
            )

    sil_euc = sil_cos = float('nan')
    if user_features is not None:
        sil_euc = compute_silhouette_wnmf(
            user_features, assignments, gray_mask, metric='euclidean',
        )
        sil_cos = compute_silhouette_wnmf(
            user_features, assignments, gray_mask, metric='cosine',
        )

    # CSV — LOF modunda lof_score sütunu da eklenir
    df_dict = {
        'user_idx'     : np.arange(len(assignments)),
        'cluster_id'   : assignments,
        'is_gray_sheep': gray_mask.astype(int),
    }
    if extra_data and 'lof_scores' in extra_data:
        np.save(os.path.join(save_dir, 'lof_scores.npy'), extra_data['lof_scores'])
        df_dict['lof_score'] = extra_data['lof_scores']
    if extra_data and extra_data.get('convergence_history'):
        pd.DataFrame(extra_data['convergence_history']).to_csv(
            os.path.join(save_dir, 'convergence_history.csv'), index=False,
        )

    pd.DataFrame(df_dict).to_csv(
        os.path.join(save_dir, 'assignment_summary.csv'), index=False
    )

    pd.DataFrame([{
        'silhouette_euclidean': sil_euc,
        'silhouette_cosine': sil_cos,
        'n_users': len(assignments),
        'n_gray_sheep': int(gray_mask.sum()),
        'K': int(assignments.max()) + 1 if len(assignments) else 0,
        'wcss': float(best_fit),
    }]).to_csv(os.path.join(save_dir, 'cluster_metrics.csv'), index=False)

    white_assign  = assignments[~gray_mask]
    K_val         = int(assignments.max()) + 1
    cluster_sizes = np.bincount(white_assign, minlength=K_val)
    active        = cluster_sizes[cluster_sizes > 0]

    threshold = extra_data.get('threshold_label', extra_data.get('threshold', '—')) if extra_data else '—'

    print(f"    OK Kaydedildi -> {save_dir}")
    print(f"      WCSS          : {best_fit:.4f}")
    print(f"      Kullanıcı     : {len(assignments)}")
    print(f"      Gray sheep    : {gray_mask.sum()} ({gray_mask.mean()*100:.1f}%)")
    print(f"      GS threshold  : {threshold:.4f}" if isinstance(threshold, float)
          else f"      GS threshold  : {threshold}")
    print(f"      Küme boyutu   : min={active.min()}, "
          f"max={active.max()}, ort={active.mean():.1f}")
    if user_features is not None and not np.isnan(sil_euc):
        print(f"      Silhouette    : {sil_euc:.4f} (euclidean, WNMF W, white users)")
        print(f"      Silhouette    : {sil_cos:.4f} (cosine, WNMF W, white users)")
    if (
        user_features is not None
        and args is not None
        and getattr(args, 'feature_extraction', None) == 'wnmf'
    ):
        print(f"      WNMF W        : wnmf_user_vectors.npy {tuple(user_features.shape)}")

    if _DB_AVAILABLE:
        # preprocessing label belirle
        prep_parts = []
        _pu = getattr(args, 'min_user_ratings', 5)
        _pi = getattr(args, 'min_item_ratings', 10)
        if _pu > 0 or _pi > 0:
            prep_parts.append(f"prune_u{_pu}_i{_pi}")
        if getattr(args, 'zscore', False):
            prep_parts.append('zscore')
        prep_parts.append(f"init_{getattr(args, 'init_mode', 'mkpp')}")
        if getattr(args, 'pca', None):
            if args.pca < 1.0:
                prep_parts.append(f'pca{int(args.pca*100)}pct')
            else:
                prep_parts.append(f'pca{int(args.pca)}')
        # WNMF latent boyutu: yeni stilde --feature-extraction wnmf --svd-components N
        # eski stilde --wnmf-features N. Klasör adıyla tutarlı tek bir wnmf{N} yaz.
        wnmf_k_for_db = None
        if getattr(args, 'feature_extraction', None) == 'wnmf':
            wnmf_k_for_db = getattr(args, 'wnmf_features', None) or getattr(args, 'svd_components', None)
        elif getattr(args, 'wnmf_features', None):
            wnmf_k_for_db = args.wnmf_features
        if wnmf_k_for_db is not None:
            prep_parts.append(f'wnmf{wnmf_k_for_db}')
            if getattr(args, 'wnmf_features', None):
                prep_parts.append(
                    f"{getattr(args, 'wnmf_init', 'inmed')}_trim"
                    f"{getattr(args, 'inmed_trim_low', 5.0):g}_{getattr(args, 'inmed_trim_high', 95.0):g}"
                )
        if getattr(args, 'train_only', False):
            prep_parts.append(
                format_train_only_folder_suffix(
                    True, getattr(args, 'eval_split', 'random'), getattr(args, 'fold', None),
                ).lstrip('_') or 'trainonly',
            )
        preprocessing = '_'.join(prep_parts) if prep_parts else 'none'

        # dataset adını belirle
        if 'ml100k' in save_dir:
            ds = 'ml100k'
        elif 'filmtrust' in save_dir:
            ds = 'filmtrust'
        else:
            ds = 'ml1m'

        lof_scores_arr = extra_data.get('lof_scores') if extra_data else None

        db_save(
            dataset=ds,
            algo=label,
            k=K,
            preprocessing=preprocessing,
            assign_suffix=assign_suffix_from_save_dir(save_dir, label) if label else None,
            wcss=float(best_fit),
            gray_count=int(gray_mask.sum()),
            gray_ratio=float(gray_mask.mean()),
            lof_threshold=float(extra_data.get('threshold', 0))
                          if extra_data else 0.0,
            n_users=len(assignments),
            cluster_min=int(active.min()),
            cluster_max=int(active.max()),
            cluster_avg=float(active.mean()),
            # best_sol_arr YOK
            assignments_arr=assignments,
            gray_mask_arr=gray_mask,
            lof_scores_arr=lof_scores_arr,
            run_id=run_id,
            seed=seed,
        )


# ============================================================
# TEK ALGORİTMA ÇALIŞTIR
# ============================================================

def _refine_centroids_with_kmeans(
    matrix: np.ndarray,
    best_sol: np.ndarray,
    K: int,
    max_iter: int = 300,
    seed: int = 42,
):
    """Meta-sezgisel centroidleri sklearn KMeans ile rafine eder.

    - init=centroids, n_init=1: HHO/HGS/AVOA gibi optimizer'ın bulduğu noktaları
      başlangıç olarak kullanır; KMeans birkaç iterasyonda yerel Lloyd optimumuna
      sürükler ve dejenere/boş kümeleri otomatik onarır.
    - Geri dönüş: (refined_sol_flat, refined_inertia, refined_labels, n_active).
    - Hata olursa orijinal best_sol'u döndürür.
    """
    try:
        from sklearn.cluster import KMeans

        n_features = matrix.shape[1]
        centroids = np.asarray(best_sol, dtype=np.float64).reshape(K, n_features)
        km = KMeans(
            n_clusters=K,
            init=centroids,
            n_init=1,
            max_iter=int(max_iter),
            random_state=int(seed),
        )
        km.fit(matrix)
        n_active = int(len(np.unique(km.labels_)))
        return (
            km.cluster_centers_.flatten().astype(best_sol.dtype, copy=False),
            float(km.inertia_),
            km.labels_.astype(np.int32, copy=False),
            n_active,
        )
    except Exception as exc:
        print(f"    KMeans refinement başarısız: {exc}")
        return None


def _repair_empty_clusters(matrix, best_sol, assignments, K):
    """Boş kümeleri, atamaları topluca ezmeden onarır.

    Meta-sezgisel atamalarının geri kalanını KORUR. Yalnızca boş kalan her küme
    için, birden fazla üyeli kümelerden kendi merkezine en uzak (en kötü oturan)
    noktayı o boş kümeye taşır ve boş kümenin merkezini o noktaya çeker. Lloyd
    yakınsamasıyla tüm atamaları yeniden hesaplamaz → algoritmalar arası farklar
    korunur.

    Dönüş: (assignments (int32), centroids (K, dim), n_moved).
    """
    X = np.asarray(matrix, dtype=np.float64)
    assignments = np.asarray(assignments, dtype=np.int32).copy()
    centroids = np.asarray(best_sol, dtype=np.float64).reshape(K, X.shape[1]).copy()
    n_moved = 0

    for _ in range(K):  # en fazla K geçiş yeter (her geçiş ≥1 boş küme doldurur)
        counts = np.bincount(assignments, minlength=K)
        empties = np.where(counts == 0)[0]
        if empties.size == 0:
            break
        if np.count_nonzero(counts > 1) == 0:
            # Onarılamaz: aktif nokta sayısı < istenen küme sayısı.
            break
        donor_idx = np.where(counts[assignments] > 1)[0]
        # Aday noktaların kendi merkezine Öklid mesafesi (en kötü oturan önce).
        d = np.linalg.norm(X[donor_idx] - centroids[assignments[donor_idx]], axis=1)
        order = donor_idx[np.argsort(-d)]
        oi = 0
        for cid in empties:
            while oi < len(order):
                p = int(order[oi])
                oi += 1
                src = int(assignments[p])
                if np.sum(assignments == src) > 1:  # bağışçıda en az 1 üye kalsın
                    assignments[p] = int(cid)
                    centroids[cid] = X[p]
                    n_moved += 1
                    break
    return assignments, centroids, n_moved


def _run_one_core(
    label,
    g_name,
    l_name,
    matrix,
    K,
    seed,
    save_dir,
    algo_map,
    use_lof,
    lof_n_neighbors,
    lof_contamination,
    baseline_epoch,
    global_epoch,
    local_epoch,
    pop_size,
    args=None,
    run_id=None,
    cluster_metric: str = 'pearson',
    init_mode: str = 'mkpp',
    disable_gray_sheep: bool = False,
):
    """Ortak gövde: algo_map ana süreçte veya worker’da bir kez oluşturulur."""
    if disable_gray_sheep:
        mode_str = 'off'
    else:
        mode_str = 'LOF' if use_lof else 'percentile'
    fitness_mode = (
        getattr(args, 'fitness', 'wcss') if args is not None else 'wcss'
    )
    cluster_objective = (
        getattr(args, 'cluster_objective', 'multi') if args is not None else 'multi'
    )
    fitness_config = (
        _fitness_config_from_args(args) if args is not None
        else {'objective': cluster_objective}
    )
    # B0_KMEANS her zaman gerçek KMeans baseline olarak çalışmalı.
    # knn_mae/latent_dev: --algo etiketi centroid aramasını yürütür; ardından kmref.
    use_centroid_opt = (
        label != 'B0_KMEANS'
        and fitness_mode in ('latent_dev', 'knn_mae', 'knn_mae_legacy')
    )
    centroid_problem = None
    fitness_evaluator = None
    search_matrix = matrix
    flat_fitness_fn = None

    if use_centroid_opt:
        opt_note = f", fitness={fitness_mode}, centroid_search={label}"
    else:
        opt_note = ""
    obj_note = f", cluster_objective={cluster_objective}" if not use_centroid_opt else ""
    if not use_centroid_opt and cluster_objective == 'multi':
        mw = fitness_config.get('mo_weights', MO_WEIGHT_PRESETS['default'])
        rep_l = float(fitness_config.get('repulsion_lambda', 0.0) or 0.0)
        obj_note = (
            f", cluster_objective={cluster_objective}, "
            f"mo=({mw[0]:g},{mw[1]:g},{mw[2]:g}), repulsion={rep_l:g}"
        )
    print(
        f"\n  [{label}] başlıyor "
        f"(gray sheep: {mode_str}, küme metrik: {cluster_metric}, init: {init_mode}"
        f"{opt_note}{obj_note})..."
    )
    t0 = time.time()
    convergence_history = None
    actual_epochs = None
    assignments = None
    w_matrix_for_save = matrix
    cluster_matrix = matrix

    if use_centroid_opt:
        from centroid_optimizer import CentroidFitnessEvaluator, load_wnmf_w_matrix

        model_path = getattr(args, 'wnmf_model_path', None) if args is not None else None
        if model_path:
            W_matrix = load_wnmf_w_matrix(model_path)
        else:
            W_matrix = np.asarray(matrix, dtype=np.float64)
            print(
                f"  W matrisi: pipeline WNMF çıktısı {W_matrix.shape}",
                flush=True,
            )
        w_matrix_for_save = W_matrix
        cluster_matrix = W_matrix
        search_matrix = W_matrix
        fitness_evaluator = CentroidFitnessEvaluator(
            W_matrix,
            K,
            train_ratings=getattr(args, 'centroid_train_ratings', None),
            val_ratings=getattr(args, 'centroid_val_ratings', None),
            n_train_sample=int(getattr(args, 'centroid_train_sample', 500) or 500),
            n_val_sample=int(getattr(args, 'centroid_val_sample', 200) or 200),
            knn_k=int(getattr(args, 'centroid_knn_k', 20) or 20),
            fitness_mode=fitness_mode,
            knn_sim_metric=str(getattr(args, 'centroid_knn_sim', 'cosine') or 'cosine'),
            min_common=int(getattr(args, 'min_common', 3) or 3),
            bias_epochs=int(getattr(args, 'centroid_bias_epochs', 5) or 5),
            use_native_predictor=fitness_mode == 'knn_mae',
            seed=seed,
        )
        centroid_problem = fitness_evaluator.make_problem()
        flat_fitness_fn = fitness_evaluator.make_flat_fitness_fn()
        print(
            f"  Centroid arama: {label} -> {fitness_evaluator.fitness_label()} "
            f"(epoch={int(getattr(args, 'centroid_iter', baseline_epoch) or baseline_epoch)}, "
            f"agents={int(getattr(args, 'centroid_agents', pop_size) or pop_size)})",
            flush=True,
        )

    if g_name == 'KMEANS':
        init = []
    else:
        init = _multi_start_init(
            search_matrix, K=K, pop_size=pop_size, seed=seed, n_restarts=10,
            metric=cluster_metric, init_mode=init_mode,
            cluster_objective=cluster_objective,
            fitness_config=fitness_config,
        )

    opt_epoch = (
        int(getattr(args, 'centroid_iter', baseline_epoch) or baseline_epoch)
        if use_centroid_opt else baseline_epoch
    )
    opt_pop = (
        int(getattr(args, 'centroid_agents', pop_size) or pop_size)
        if use_centroid_opt else pop_size
    )

    if label == 'B0_KMEANS':
        from sklearn.cluster import KMeans

        user_matrix = matrix
        b0_n_init = int(getattr(args, 'b0_n_init', 10) or 10) if args is not None else 10
        km = KMeans(n_clusters=K, n_init=b0_n_init, max_iter=500, random_state=42)
        km.fit(user_matrix)
        best_sol = km.cluster_centers_.flatten()
        best_fit = float(km.inertia_)
        assignments = km.labels_
    elif label == 'HA_AVOAHGS':
        ha_p1 = _avoahgs_p1_from_args(args)
        ha_hgs = _avoahgs_hgs_rate_from_args(args)
        print(f"    HA_AVOAHGS params: p1={ha_p1:g}, hgs_rate={ha_hgs:g}")
        if args is not None and getattr(args, 'early_stop', False):
            es_max, es_pat, es_tol, es_block, es_note = _resolve_ha_epoch_policy(
                args, search_matrix, K, opt_pop,
            )
            print(f"    HA epoch policy: {es_note}")
            best_sol, best_fit, actual_epochs, convergence_history = (
                run_ha_avoahgs_with_early_stop(
                    search_matrix, K, init, es_max, opt_pop,
                    metric=cluster_metric,
                    patience=es_pat,
                    tolerance=es_tol,
                    block_size=es_block,
                    p1=ha_p1,
                    hgs_rate=ha_hgs,
                    cluster_objective=cluster_objective,
                    fitness_config=fitness_config,
                    problem=centroid_problem,
                )
            )
            print(f"    Early-stop epoch: {actual_epochs}/{es_max}")
        else:
            from optimizers.HA_AVOAHGS import HA_AVOAHGS

            model = HA_AVOAHGS(
                epoch=opt_epoch,
                pop_size=opt_pop,
                p1=ha_p1,
                hgs_rate=ha_hgs,
            )
            problem = _resolve_problem(
                search_matrix, K, problem=centroid_problem,
                metric=cluster_metric, cluster_objective=cluster_objective,
                fitness_config=fitness_config,
            )
            try:
                model.solve(problem, starting_solutions=init[:opt_pop])
            except TypeError:
                model.solve(problem)
            best_sol = model.g_best.solution
            best_fit = float(model.g_best.target.fitness)
    elif g_name == 'KMEANS':
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize

        matrix_norm = normalize(search_matrix, norm='l2')
        kmeans = KMeans(
            n_clusters=K,
            init=_sklearn_kmeans_init(args),
            n_init=10,
            random_state=seed,
            max_iter=300,
            verbose=0
        )
        kmeans.fit(matrix_norm)
        assignments_km = kmeans.labels_

        best_sol = kmeans.cluster_centers_.flatten()
        best_fit = float(kmeans.inertia_)
        assignments = assignments_km
    elif l_name == 'GAop':
        best_sol, best_fit = run_memetic_hybrid(
            algo_map[g_name], search_matrix, K, init,
            total_epoch=global_epoch + local_epoch,
            pop_size=opt_pop,
            ga_inject_interval=10,
            ga_crossover_rate=0.3,
            ga_mutation_rate=0.1,
            metric=cluster_metric,
            cluster_objective=cluster_objective,
            fitness_config=fitness_config,
            problem=centroid_problem,
        )
    elif g_name == 'LF_HHO' or label == 'LF_HHO':
        lf_hho = LevyHHO_Clustering(
            n_agents=opt_pop,
            n_iter=opt_epoch,
            k=K,
            levy_scale=1.0,
            stagnation_tol=10,
            seed=seed,
        )
        centers = lf_hho.optimize(
            search_matrix, fitness_fn=flat_fitness_fn,
        )
        _ = lf_hho.assign(search_matrix, centers)
        best_sol = centers.flatten()
        if fitness_evaluator is not None:
            best_fit, _, _ = fitness_evaluator.evaluate_solution(best_sol)
        else:
            best_fit, _ = compute_wcss_fast(
                search_matrix, best_sol, K, metric=cluster_metric,
            )
    elif g_name == 'IWO_HHO' or label == 'IWO_HHO':
        if args is not None and getattr(args, 'early_stop', False):
            es_max = getattr(args, 'early_stop_max_epoch', EARLY_STOP_MAX_EPOCH)
            best_sol, best_fit, actual_epochs, convergence_history = (
                run_iwo_hho_with_early_stop(
                    search_matrix, K, init, es_max, opt_pop, seed,
                    metric=cluster_metric,
                    patience=getattr(args, 'early_stop_patience', EARLY_STOP_PATIENCE),
                    tolerance=getattr(args, 'early_stop_tolerance', EARLY_STOP_TOLERANCE),
                    block_size=getattr(args, 'early_stop_block', EARLY_STOP_BLOCK_SIZE),
                    cluster_objective=cluster_objective,
                    fitness_config=fitness_config,
                    problem=centroid_problem,
                )
            )
            print(f"    Early-stop epoch: {actual_epochs}/{es_max}")
        else:
            from optimizers.iwo_hho import IWO_HHO_Clustering

            problem = _resolve_problem(
                search_matrix, K, problem=centroid_problem,
                metric=cluster_metric, cluster_objective=cluster_objective,
                fitness_config=fitness_config,
            )
            iwo_hho = IWO_HHO_Clustering(
                epoch=opt_epoch,
                pop_size=opt_pop,
                seed=seed,
            )
            try:
                best_sol, best_fit = iwo_hho.solve(
                    problem, starting_solutions=init[:opt_pop],
                )
            except TypeError:
                best_sol, best_fit = iwo_hho.solve(problem)
    elif label in ('SFOA', 'SFOA_06') or g_name in ('SFOA', 'SFOA_06'):
        gp_map = {'SFOA': 0.5, 'SFOA_06': 0.6}
        sfoa_key = label if label in gp_map else g_name
        sfoa = SFOA_Clustering(
            n_agents=opt_pop,
            n_iter=opt_epoch,
            k=K,
            Gp=gp_map[sfoa_key],
            seed=seed,
        )
        centers = sfoa.optimize(search_matrix, fitness_fn=flat_fitness_fn)
        _ = sfoa.assign(search_matrix, centers)
        best_sol = centers.flatten()
        if fitness_evaluator is not None:
            best_fit, _, _ = fitness_evaluator.evaluate_solution(best_sol)
        else:
            best_fit, _ = compute_wcss_fast(
                search_matrix, best_sol, K, metric=cluster_metric,
            )
    elif l_name is None:
        if args is not None and getattr(args, 'early_stop', False):
            es_max = getattr(args, 'early_stop_max_epoch', EARLY_STOP_MAX_EPOCH)
            best_sol, best_fit, actual_epochs, convergence_history = (
                run_single_with_early_stop(
                    algo_map[g_name], search_matrix, K, init, es_max, opt_pop,
                    metric=cluster_metric,
                    patience=getattr(args, 'early_stop_patience', EARLY_STOP_PATIENCE),
                    tolerance=getattr(args, 'early_stop_tolerance', EARLY_STOP_TOLERANCE),
                    block_size=getattr(args, 'early_stop_block', EARLY_STOP_BLOCK_SIZE),
                    cluster_objective=cluster_objective,
                    fitness_config=fitness_config,
                    problem=centroid_problem,
                )
            )
            print(f"    Early-stop epoch: {actual_epochs}/{es_max}")
        else:
            best_sol, best_fit = run_single(
                algo_map[g_name], search_matrix, K, init, opt_epoch, opt_pop,
                metric=cluster_metric,
                cluster_objective=cluster_objective,
                fitness_config=fitness_config,
                problem=centroid_problem,
            )
    elif '||' in label:
        best_sol, best_fit = run_parallel_hybrid(
            algo_map[g_name], algo_map[l_name],
            search_matrix, K, init, global_epoch, local_epoch, opt_pop,
            metric=cluster_metric,
            cluster_objective=cluster_objective,
            fitness_config=fitness_config,
            problem=centroid_problem,
        )
    else:
        best_sol, best_fit = run_hybrid(
            algo_map[g_name], algo_map[l_name],
            search_matrix, K, init, global_epoch, local_epoch, opt_pop,
            metric=cluster_metric,
            cluster_objective=cluster_objective,
            fitness_config=fitness_config,
            problem=centroid_problem,
        )

    if fitness_evaluator is not None and fitness_mode == 'knn_mae':
        best_fit, _, _ = fitness_evaluator.evaluate_solution(
            best_sol, use_full_train=True,
        )
        print(
            f"    {fitness_evaluator.fitness_label()} (full train): {best_fit:.6f}",
            flush=True,
        )

    # === ATAMA + BOŞ KÜME ONARIMI ===
    # Varsayılan: meta-sezgisel optimizer'ın KENDİ atamaları korunur; yalnızca
    # boş kümeler, atamaları topluca ezmeden onarılır (_repair_empty_clusters).
    # --kmeans-refine-overwrite ile eski davranış: sklearn KMeans Lloyd ile
    # rafine + atamaları ez (algoritmalar arası farkları homojenleştirir).
    # B0_KMEANS ve dahili KMEANS zaten KMeans olduğundan onlara dokunulmaz.
    refine_enabled = bool(getattr(args, 'kmeans_refine', True))
    overwrite_mode = bool(getattr(args, 'kmeans_refine_overwrite', False))
    is_kmeans_branch = (label == 'B0_KMEANS') or (g_name == 'KMEANS')

    fcm_m = _fcm_m_from_args(args) if args is not None else 2.0
    memberships = None
    if not is_kmeans_branch:
        # 1) Meta-sezgiselin kendi atamaları (cluster_metric ile).
        if cluster_metric == 'fuzzy':
            _, assignments, memberships, best_sol = compute_fcm_objective(
                cluster_matrix, best_sol, K, m=fcm_m,
            )
        elif assignments is None:
            _, assignments = compute_wcss_fast(
                cluster_matrix, best_sol, K, metric=cluster_metric,
            )

        # 2) Onarım / refinement.
        if refine_enabled and overwrite_mode:
            max_iter = int(getattr(args, 'kmeans_refine_iter', 300))
            result = _refine_centroids_with_kmeans(
                cluster_matrix, best_sol, K, max_iter=max_iter, seed=seed,
            )
            if result is not None:
                refined_sol, refined_inertia, refined_labels, n_active = result
                print(
                    f"    KMeans refine (overwrite): aktif küme {n_active}/{K}, "
                    f"inertia={refined_inertia:.4f}"
                )
                best_sol = refined_sol
                assignments = refined_labels
                if cluster_metric == 'fuzzy':
                    _, _, memberships, _ = compute_fcm_objective(
                        cluster_matrix, best_sol, K, m=fcm_m,
                    )
        elif refine_enabled:
            assignments, repaired_centroids, n_moved = _repair_empty_clusters(
                cluster_matrix, best_sol, assignments, K,
            )
            n_active = int(len(np.unique(assignments)))
            if n_moved > 0:
                best_sol = repaired_centroids.flatten().astype(
                    np.asarray(best_sol).dtype, copy=False,
                )
                print(
                    f"    Boş küme onarımı: {n_moved} nokta taşındı, "
                    f"aktif küme {n_active}/{K} (atamalar korundu)"
                )
                if cluster_metric == 'fuzzy':
                    _, _, memberships, _ = compute_fcm_objective(
                        cluster_matrix, best_sol, K, m=fcm_m,
                    )
            else:
                print(f"    Boş küme yok: onarım gerekmedi (aktif küme {n_active}/{K})")
    elif cluster_metric == 'fuzzy':
        _, assignments, memberships, _ = compute_fcm_objective(
            cluster_matrix, best_sol, K, m=fcm_m,
        )

    # Meta-sezgisel sonrası FCM: başlangıç centroid = algo çıktısı, J_m minimize.
    # B0_KMEANS ve dahili KMEANS zaten Öklid Lloyd; FCM post-refine atlanır (kmref ile aynı mantık).
    fcm_j = None
    if getattr(args, 'fcm', False) and not is_kmeans_branch:
        fcm_iters = int(getattr(args, 'fcm_iter', 50) or 50)
        fcm_j, assignments, memberships, best_sol = compute_fcm_objective(
            cluster_matrix,
            best_sol,
            K,
            m=fcm_m,
            max_iter=fcm_iters,
            tol=1e-6,
        )
        print(
            f'    FCM post-refine ({fcm_iters} iter): J_m={fcm_j:.4f}, '
            f'aktif küme {len(np.unique(assignments))}/{K}',
        )
    elif getattr(args, 'fcm', False) and is_kmeans_branch:
        print(f'    FCM post-refine: atlandı ({label} zaten KMeans baseline)')

    if disable_gray_sheep:
        gray_mask = np.zeros(len(assignments), dtype=bool)
        extra_data = {'threshold': 0.0, 'threshold_label': 'disabled'}
    elif use_lof:
        print(f"    LOF hesaplanıyor (n_neighbors={lof_n_neighbors})...")
        gs_info   = detect_gray_sheep_lof(
            matrix, assignments, lof_n_neighbors, lof_contamination
        )
        gray_mask = gs_info['gray_sheep_mask']
        extra_data = {
            'lof_scores': gs_info['lof_scores'],
            'threshold' : gs_info['threshold'],
        }
    else:
        gs_info    = detect_gray_sheep_percentile(
            cluster_matrix, assignments, best_sol, K, metric=cluster_metric,
        )
        gray_mask  = gs_info['gray_sheep_mask']
        extra_data = {'threshold': gs_info['threshold']}

    if convergence_history is not None:
        extra_data = {
            **(extra_data or {}),
            'convergence_history': convergence_history,
            'early_stop_epochs': actual_epochs,
        }

    # DB'de WCSS kolonu her zaman gerçek kümeleme hedefini taşısın;
    # optimizer'ın iç objective'i (kompozit vb.) ile karışmasın.
    best_wcss, _ = compute_wcss_fast(
        cluster_matrix, best_sol, K, metric=cluster_metric, fcm_m=fcm_m,
    )
    if use_centroid_opt:
        fit_label = (
            'cluster_predictor_mae' if fitness_mode == 'knn_mae'
            else ('sample_mae' if fitness_mode == 'knn_mae_legacy' else 'latent_dev')
        )
        print(f"    {fit_label}    : {best_fit:.6f}  (WCSS rapor: {best_wcss:.4f})")

    if fcm_j is not None:
        best_wcss = float(fcm_j)

    save_assignment(
        assignments, gray_mask, best_sol, best_wcss,
        save_dir, extra_data, label=label, K=K, args=args,
        run_id=run_id, seed=seed, memberships=memberships,
        user_features=w_matrix_for_save,
    )
    print(f"  [{label}] tamamlandı — {time.time()-t0:.1f}s")


def _mp_run_assignment_job(job):
    """
    Windows spawn için modül düzeyinde worker — mealpy sınıflarını pickle etmemek
    için algo_map alt-süreçte get_all_algorithms_v3() ile kurulur.
    """
    (
        label,
        g_name,
        l_name,
        matrix,
        K,
        seed,
        save_dir,
        use_lof,
        lof_n_neighbors,
        lof_contamination,
        baseline_epoch,
        global_epoch,
        local_epoch,
        pop_size,
        args,
        run_id,
        cluster_metric,
        init_mode,
        disable_gray_sheep,
    ) = job
    algo_map = {a['full_name']: a for a in get_all_algorithms_v3()}
    algo_map['GAHHO.OriginalGAHHO'] = {
        'full_name': 'GAHHO.OriginalGAHHO',
        'class':     OriginalGAHHO,
    }
    algo_map['GA.EliteMultiGA'] = {
        'full_name': 'GA.EliteMultiGA',
        'class':     GA.EliteMultiGA,
    }
    algo_map['KMEANS'] = {
        'full_name': 'KMEANS',
        'class': None,
    }
    algo_map['DOA.OriginalDOA'] = {
        'full_name': 'DOA.OriginalDOA',
        'class':     OriginalDOA,
    }
    _run_one_core(
        label,
        g_name,
        l_name,
        matrix,
        K,
        seed,
        save_dir,
        algo_map,
        use_lof,
        lof_n_neighbors,
        lof_contamination,
        baseline_epoch,
        global_epoch,
        local_epoch,
        pop_size,
        args,
        run_id,
        cluster_metric,
        init_mode,
        disable_gray_sheep,
    )


def run_one(label, g_name, l_name, matrix, K, seed, save_dir, algo_map,
            use_lof=False, lof_n_neighbors=LOF_N_NEIGHBORS,
            lof_contamination=LOF_CONTAMINATION, args=None, run_id=None,
            cluster_metric: str = 'pearson', init_mode: str = 'mkpp',
            disable_gray_sheep: bool = False):
    """
    Bir algoritma için assignment üret ve kaydet.

    use_lof=True  → LOF tabanlı gray sheep (adaptif)
    use_lof=False → Sabit 80. percentile (~%20)
    """
    baseline_epoch, global_epoch, local_epoch, pop_size = _resolve_train_hyperparams(args)
    _run_one_core(
        label,
        g_name,
        l_name,
        matrix,
        K,
        seed,
        save_dir,
        algo_map,
        use_lof,
        lof_n_neighbors,
        lof_contamination,
        baseline_epoch,
        global_epoch,
        local_epoch,
        pop_size,
        args,
        run_id=run_id,
        cluster_metric=cluster_metric,
        init_mode=init_mode,
        disable_gray_sheep=disable_gray_sheep,
    )


# ============================================================
# TRAIN-ONLY VERİ YÜKLEME (wnmf_experiment eval split ile hizalı)
# ============================================================

def format_train_only_folder_suffix(train_only, eval_split, fold=None):
    """Klasör out_suffix: train-only + eval protokolü; fold 1..5 → _f{N} (ayrı atama setleri)."""
    if not train_only:
        return ''
    base = '_trainonly_official' if eval_split == 'official' else '_trainonly_rand'
    if fold is not None and 1 <= int(fold) <= 5:
        return f'{base}_f{int(fold)}'
    return base


def format_assign_suffix_from_args(args, K: int, label: str = '', g_name: str = '') -> str:
    """assign_suffix: preprocess (+ wnmf boyutu) + K (+ opsiyonel ekler)."""
    if getattr(args, 'feature_extraction', None) == 'wnmf':
        wdim = int(getattr(args, 'wnmf_features', None) or getattr(args, 'svd_components', 20))
        assign_suffix = f'_{args.preprocess}_wnmf{wdim}_k{int(K)}'
    else:
        assign_suffix = f'_{args.preprocess}_k{int(K)}'
    if getattr(args, 'cluster_objective', 'multi') == 'wcss':
        assign_suffix += '_pwcss'
    if getattr(args, 'fitness', 'wcss') == 'knn_mae':
        assign_suffix += '_knnmae'
    if getattr(args, 'l2_normalize', False):
        assign_suffix += '_l2'
    if (
        getattr(args, 'kmeans_refine', True)
        and getattr(args, 'kmeans_refine_overwrite', False)
        and label != 'B0_KMEANS'
    ):
        assign_suffix += '_kmref'
        kmi = int(getattr(args, 'kmeans_refine_iter', 300) or 300)
        if kmi != 300:
            assign_suffix += f'cap{kmi}'
    if label == 'B0_KMEANS':
        b0_ninit = int(getattr(args, 'b0_n_init', 10) or 10)
        if b0_ninit != 10:
            assign_suffix += f'_ninit{b0_ninit}'
    if getattr(args, 'fcm', False) and label != 'B0_KMEANS' and g_name != 'KMEANS':
        assign_suffix += '_fcm'
    if (
        getattr(args, 'fcm_m_suffix', False)
        and getattr(args, 'cluster_metric', '') == 'fuzzy'
    ):
        assign_suffix += format_fcm_m_folder_suffix(_fcm_m_from_args(args))
    if (
        getattr(args, 'avoahgs_param_suffix', False)
        and label == 'HA_AVOAHGS'
    ):
        assign_suffix += format_avoahgs_param_folder_suffix(
            _avoahgs_p1_from_args(args),
            _avoahgs_hgs_rate_from_args(args),
        )
    return assign_suffix


def format_out_suffix_from_args(args) -> str:
    """out_suffix: budama, metrik, init, gray sheep, train-only, wnmf epoch (dim yok)."""
    prune_suffix = format_prune_folder_suffix(
        args.min_user_ratings, args.min_item_ratings,
    )
    zscore_suffix = (
        '_colzscore' if getattr(args, 'paper_style', False)
        else ('_zscore' if args.zscore else '')
    )
    pca_suffix = (
        f'_pca{int(round(args.pca_variance * 100))}pct'
        if args.pca_variance is not None else ''
    )
    wnmf_suffix = (
        f'_{args.wnmf_init}_trim{args.inmed_trim_low:g}_{args.inmed_trim_high:g}'
        if args.wnmf_features is not None else ''
    )
    metric_suffix_map = {
        'euclidean': '_euc',
        'fuzzy': '_fuzzy',
    }
    metric_suffix = metric_suffix_map.get(args.cluster_metric, '')
    init_suffix = format_init_mode_folder_suffix(args.init_mode)
    paper_suffix = '_paper' if args.paper_mode else ''
    no_gs_suffix = '_nogs' if args.disable_gray_sheep and not args.paper_mode else ''
    train_only_suffix = format_train_only_folder_suffix(
        args.train_only, args.eval_split, args.fold,
    )
    out_suffix = (
        prune_suffix + zscore_suffix + pca_suffix + wnmf_suffix
        + metric_suffix + init_suffix + paper_suffix + no_gs_suffix
        + train_only_suffix
    )
    if (
        getattr(args, 'feature_extraction', None) == 'wnmf'
        and not getattr(args, 'legacy_wnmf_suffix', False)
    ):
        out_suffix += f'_wnmfep{args.wnmf_epochs}'
    return out_suffix


def build_folder_suffix(args, K: int, label: str = '', g_name: str = '') -> str:
    """Label hariç tam klasör soneki: out_suffix + assign_suffix."""
    return format_out_suffix_from_args(args) + format_assign_suffix_from_args(
        args, K, label=label, g_name=g_name,
    )


def _import_wnmf_loaders():
    wnmf_dir = os.path.join(_REPO_ROOT, 'wnmf')
    if wnmf_dir not in sys.path:
        sys.path.insert(0, wnmf_dir)
    from wnmf_utils import (
        load_ratings_100k,
        load_ratings_100k_all,
        load_ratings_1m,
        load_ratings_filmtrust,
        load_filmtrust_matrix,
    )
    return (
        load_ratings_100k,
        load_ratings_100k_all,
        load_ratings_1m,
        load_ratings_filmtrust,
        load_filmtrust_matrix,
    )


def _ratings_to_dense_matrix(train, n_users, n_items):
    matrix = np.zeros((n_users, n_items), dtype=np.float32)
    for row in train:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        if 0 <= u < n_users and 0 <= i < n_items:
            matrix[u, i] = r
    return matrix


def load_movielens_train_only_100k(
    eval_split='random',
    fold=None,
    random_seed=SEED,
    ratings_path=None,
):
    """
    ML-100K train rating'lerinden dense matris (test hücreleri 0).
    wnmf_experiment.py --eval-split / --fold ile aynı bölme.
    """
    load_ratings_100k, load_ratings_100k_all, _, _, _ = _import_wnmf_loaders()
    ratings_path = ratings_path or DATA_100K

    if eval_split == 'random':
        train, test = load_ratings_100k_all(
            ratings_path, random_seed=random_seed, fold=fold,
        )
        split_label = (
            f'random KFold fold {fold}/5 (seed={random_seed})'
            if fold is not None
            else f'random %20 holdout (seed={random_seed})'
        )
    elif fold is None or fold == 1:
        train, test = load_ratings_100k(DATA_100K_TRAIN, DATA_100K_TEST)
        split_label = 'official u1.base / u1.test'
    else:
        train, test = load_ratings_100k(DATA_100K_TRAIN, DATA_100K_TEST, fold=fold)
        split_label = f'official u{fold}.base / u{fold}.test'

    matrix = _ratings_to_dense_matrix(train, N_USERS_100K, N_ITEMS_100K)
    total = matrix.size
    nonzero = np.count_nonzero(matrix)
    print(f"Matrix shape (train-only, {split_label}): {matrix.shape}")
    print(f"  Train ratings: {len(train):,}  |  Test (hariç): {len(test):,}")
    print(f"  Sparsity      : {1 - nonzero / total:.3f}")
    print(
        f"  Rating range  : {matrix[matrix > 0].min():.1f} - {matrix.max():.1f}"
        if nonzero else "  Rating range  : (boş)"
    )
    return matrix


def load_movielens_train_only_1m(fold=None, random_seed=SEED, ratings_path=None):
    """ML-1M train rating'lerinden dense matris (test hücreleri 0)."""
    _, _, load_ratings_1m, _, _ = _import_wnmf_loaders()
    ratings_path = ratings_path or DATA_1M
    train, test = load_ratings_1m(ratings_path, random_seed=random_seed, fold=fold)
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    matrix = _ratings_to_dense_matrix(train, n_users, n_items)
    total = matrix.size
    nonzero = np.count_nonzero(matrix)
    fold_label = (
        f'fold {fold}/5 (seed={random_seed})'
        if fold is not None and fold != 1
        else f'%20 holdout (seed={random_seed})'
    )
    print(f"Matrix shape (train-only, {fold_label}): {matrix.shape}")
    print(f"  Train ratings: {len(train):,}  |  Test (hariç): {len(test):,}")
    print(f"  Sparsity     : {1 - nonzero / total:.3f}")
    return matrix


def load_filmtrust_train_only(random_seed=SEED, ratings_path=None, fold=None):
    """FilmTrust train rating'lerinden dense matris (test hücreleri 0)."""
    _, _, _, load_ratings_filmtrust, _ = _import_wnmf_loaders()
    ratings_path = ratings_path or DATA_FILMTRUST
    train, test = load_ratings_filmtrust(
        ratings_path, random_seed=random_seed, fold=fold,
    )
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    matrix = _ratings_to_dense_matrix(train, n_users, n_items)
    total = matrix.size
    nonzero = np.count_nonzero(matrix)
    fold_label = (
        f'fold {fold}/5 (seed={random_seed})'
        if fold is not None and fold != 1
        else f'%20 holdout (seed={random_seed})'
    )
    print(f"Matrix shape (train-only, {fold_label}): {matrix.shape}")
    print(f"  Train ratings: {len(train):,}  |  Test (hariç): {len(test):,}")
    print(f"  Sparsity     : {1 - nonzero / total:.3f}")
    return matrix


# ============================================================
# ML-1M VERİ YÜKLEME
# ============================================================

def load_movielens_1m(path):
    rows = []
    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('::')
            if len(parts) >= 3:
                rows.append((int(parts[0]), int(parts[1]), float(parts[2])))
    df = pd.DataFrame(rows, columns=['user_id', 'item_id', 'rating'])
    matrix = df.pivot_table(
        index='user_id', columns='item_id',
        values='rating', fill_value=0
    ).values.astype(np.float32)
    total   = matrix.size
    nonzero = np.count_nonzero(matrix)
    print(f"Matrix shape : {matrix.shape}")
    print(f"Sparsity     : {1 - nonzero/total:.3f}")
    return matrix


def format_prune_folder_suffix(min_user_ratings: int, min_item_ratings: int) -> str:
    """Klasör out_suffix: budama kapalıysa boş, aksi halde _pruneu{N}_i{M}."""
    if int(min_user_ratings) <= 0 and int(min_item_ratings) <= 0:
        return ''
    return f'_pruneu{int(min_user_ratings)}_i{int(min_item_ratings)}'


def format_init_mode_folder_suffix(init_mode: str) -> str:
    """Meta-sezgisel centroid başlatma (--init-mode): _imkpp veya _irand."""
    mode = (init_mode or 'mkpp').strip().lower()
    if mode == 'random':
        return '_irand'
    if mode != 'mkpp':
        raise ValueError(f"init_mode bilinmiyor: {init_mode!r} (mkpp | random)")
    return '_imkpp'


def _fcm_m_from_args(args, default: float = 2.0) -> float:
    if args is None:
        return float(default)
    return float(getattr(args, 'fcm_m', default) or default)


def format_fcm_m_folder_suffix(fcm_m: float) -> str:
    """FCM fuzzifier etiketi: m=1.5 -> _m15, m=2.0 -> _m20."""
    return f'_m{int(round(float(fcm_m) * 10))}'


HA_AVOAHGS_P1_DEFAULT = 0.4
HA_AVOAHGS_HGS_DEFAULT = 0.7


def _avoahgs_p1_from_args(args, default: float = HA_AVOAHGS_P1_DEFAULT) -> float:
    if args is None:
        return float(default)
    return float(getattr(args, 'p1', default) or default)


def _avoahgs_hgs_rate_from_args(args, default: float = HA_AVOAHGS_HGS_DEFAULT) -> float:
    if args is None:
        return float(default)
    return float(getattr(args, 'hgs_rate', default) or default)


def format_avoahgs_param_folder_suffix(p1: float, hgs_rate: float) -> str:
    """HA_AVOAHGS hiperparam etiketi: p1=0.2 hgs=0.25 -> _p20_hgs25; varsayılan (0.4, 0.7) -> ''."""
    p1 = float(p1)
    hgs_rate = float(hgs_rate)
    if (
        abs(p1 - HA_AVOAHGS_P1_DEFAULT) < 1e-9
        and abs(hgs_rate - HA_AVOAHGS_HGS_DEFAULT) < 1e-9
    ):
        return ''
    p_tag = int(round(p1 * 100))
    h_tag = int(round(hgs_rate * 100))
    return f'_p{p_tag:02d}_hgs{h_tag:02d}'


def prune_sparse_matrix(
    matrix,
    min_user_ratings=5,
    min_item_ratings=10,
    return_indices=False,
):
    """
    İteratif seyreklik budama:
    - min_user_ratings'ten az oylayan kullanıcıları kaldır
    - min_item_ratings'ten az oy alan filmleri kaldır
    Koşullar sabitlenene kadar döngü sürer.
    """
    if min_user_ratings <= 0 and min_item_ratings <= 0:
        if return_indices:
            n_u, n_i = matrix.shape
            return (
                matrix,
                np.arange(n_u, dtype=np.int64),
                np.arange(n_i, dtype=np.int64),
            )
        return matrix

    pruned = matrix
    kept_u = np.arange(matrix.shape[0], dtype=np.int64)
    kept_i = np.arange(matrix.shape[1], dtype=np.int64)
    prev_shape = None
    iteration = 0

    while prev_shape != pruned.shape:
        prev_shape = pruned.shape
        iteration += 1

        if min_user_ratings > 0:
            user_counts = np.count_nonzero(pruned, axis=1)
            keep_users = user_counts >= int(min_user_ratings)
            pruned = pruned[keep_users]
            kept_u = kept_u[keep_users]

        if min_item_ratings > 0:
            item_counts = np.count_nonzero(pruned, axis=0)
            keep_items = item_counts >= int(min_item_ratings)
            pruned = pruned[:, keep_items]
            kept_i = kept_i[keep_items]

    total = pruned.size
    nonzero = np.count_nonzero(pruned)
    sparsity = 1 - (nonzero / total if total > 0 else 0.0)
    print(
        f"  Prune tamamlandı ({iteration} iter): "
        f"shape={pruned.shape}, sparsity={sparsity:.3f}"
    )
    pruned = pruned.astype(np.float32, copy=False)
    if return_indices:
        return pruned, kept_u, kept_i
    return pruned


def impute_user_mean(matrix, eps: float = 1e-8):
    """
    Eksik (0) hücreleri kullanıcı ortalamasıyla doldur.
    GOA-k-means makalesi (Ambikesh et al.) ile aynı adım.
    """
    out = matrix.astype(np.float32, copy=True)
    rated = out > 0
    user_means = np.full(out.shape[0], np.nan, dtype=np.float32)
    for u in range(out.shape[0]):
        if rated[u].any():
            user_means[u] = float(out[u, rated[u]].mean())
    global_mean = float(out[rated].mean()) if rated.any() else 3.0
    user_means[np.isnan(user_means)] = global_mean
    for u in range(out.shape[0]):
        missing = ~rated[u]
        if missing.any():
            out[u, missing] = user_means[u]
    return out


def zscore_column_normalize(matrix, eps: float = 1e-8):
    """Sütun (film) bazlı z-score — GOA makalesi / run_goa_kmeans_paper.py."""
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    return ((matrix - mean) / (std + eps)).astype(np.float32)


def zscore_normalize(matrix):
    """
    Kullanıcı bazlı Z-score normalizasyon.
    Her kullanıcının rating'lerini normalize et:
    r_norm = (r - user_mean) / user_std

    Rating olmayan (0) hücreler normalize edilmez.
    std=0 olan kullanıcılar için sadece mean çıkar.
    """
    normalized = matrix.copy().astype(np.float32)
    for u in range(matrix.shape[0]):
        rated = matrix[u] != 0
        if rated.sum() == 0:
            continue
        mean = matrix[u, rated].mean()
        std  = matrix[u, rated].std()
        if std > 0:
            normalized[u, rated] = (matrix[u, rated] - mean) / std
        else:
            normalized[u, rated] = matrix[u, rated] - mean
    return normalized


def wnmf_feature_extract(matrix, n_components,
                         n_epochs=50, lr=0.01,
                         reg=0.01, random_seed=42,
                         init_method='inmed',
                         inmed_trim=(5.0, 95.0)):
    """
    Ham rating matrisini WNMF ile ayrıştır.
    Sadece kullanıcı latent matrisini (U) döndür.

    Uygulama: [wnmf/wnmf_model.py](wnmf_model.WNMFModel) — wnmf_experiment ile aynı
    çekirdek (tekrarlanabilirlik).

    Neden WNMF: Sıfır hücreler 'rating yok' anlamına gelir,
    standart NMF bunları sıfır rating olarak işler (hatalı).
    WNMF sadece gözlemlenen rating'leri kullanır.

    Çıktı: (n_users × n_components) dense, nonnegative U.
    """
    wnmf_dir = os.path.join(_REPO_ROOT, 'wnmf')
    if wnmf_dir not in sys.path:
        sys.path.insert(0, wnmf_dir)
    from wnmf_model import WNMFModel

    n_users, n_items = matrix.shape
    rows, cols = np.where(matrix > 0)
    if len(rows) == 0:
        raise ValueError('wnmf_feature_extract: hiç gözlemlenen rating yok')

    train_ratings = np.column_stack(
        [rows.astype(np.float32), cols.astype(np.float32), matrix[rows, cols].astype(np.float32)]
    )

    print(f"  WNMF feature extraction (WNMFModel)...")
    print(f"  Matris: {n_users}x{n_items} -> U: {n_users}x{n_components}")
    print(f"  Gözlemlenen rating: {len(rows)}, epoch: {n_epochs}")

    model = WNMFModel(
        n_users        = n_users,
        n_items        = n_items,
        latent_dim     = n_components,
        learning_rate  = lr,
        regularization = reg,
        n_epochs       = n_epochs,
        random_seed    = random_seed,
        use_bias       = True,
        use_svdpp      = False,
        init_method    = init_method,
        inmed_trim     = inmed_trim,
    )
    model.fit(train_ratings, verbose=False)
    print(f"  WNMF tamamlandı. U shape: {model.U.shape}")
    return model.U.astype(np.float32)


def compute_silhouette_wnmf(
    W: np.ndarray,
    labels: np.ndarray,
    gray_mask=None,
    metric: str = 'euclidean',
    sample: int = 500,
    random_seed: int = 42,
) -> float:
    """
    silhouette_score(wnmf_W_matrix, labels)

    W: (n_users, n_components) — kümelemede kullanılan WNMF U (L2 normalize sonrası).
    Gray sheep varsayılan olarak hariç tutulur (white-only silhouette).
    """
    from sklearn.metrics import silhouette_score

    W = np.asarray(W, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if gray_mask is not None:
        gray_mask = np.asarray(gray_mask, dtype=bool).ravel()
        if len(gray_mask) == len(labels):
            keep = ~gray_mask
            W = W[keep]
            labels = labels[keep]

    uniq = np.unique(labels)
    if len(uniq) < 2 or len(labels) < 3:
        return float('nan')

    n = len(labels)
    if n > sample:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(n, size=sample, replace=False)
        W = W[idx]
        labels = labels[idx]

    try:
        return float(silhouette_score(W, labels, metric=metric))
    except Exception:
        return float('nan')


def pca_variance_reduce(matrix, variance_ratio, random_state=42):
    """
    sklearn PCA: n_components in (0,1) → birikimli açıklanan varyans ≥ oran olacak kadar bileşen.
    """
    from sklearn.decomposition import PCA

    n_samples, n_features = matrix.shape
    pca = PCA(n_components=variance_ratio, random_state=random_state)
    out = pca.fit_transform(matrix)
    cum = float(np.sum(pca.explained_variance_ratio_))
    print(
        f"  PCA: {n_features} -> {out.shape[1]} components "
        f"(target >={variance_ratio:.0%} variance, cumulative: {cum:.4f})"
    )
    return out.astype(np.float32)


def _remap_ratings_to_pruned(
    ratings: np.ndarray,
    kept_u: np.ndarray,
    kept_i: np.ndarray,
) -> np.ndarray:
    """Orijinal (u,i) id'lerini budama sonrası satır/sütun indekslerine çevir."""
    ratings = np.asarray(ratings, dtype=np.float64)
    if ratings.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    u_map = {int(o): idx for idx, o in enumerate(np.asarray(kept_u, dtype=np.int64))}
    i_map = {int(o): idx for idx, o in enumerate(np.asarray(kept_i, dtype=np.int64))}
    rows = []
    for row in ratings:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        if u in u_map and i in i_map:
            rows.append([u_map[u], i_map[i], r])
    if not rows:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _fetch_100k_train_test_arrays(
    eval_split='random',
    fold=None,
    random_seed=SEED,
    ratings_path=None,
):
    """ML-100K train/test rating triplets (wnmf_experiment ile aynı bölme)."""
    load_ratings_100k, load_ratings_100k_all, _, _, _ = _import_wnmf_loaders()
    ratings_path = ratings_path or DATA_100K
    if eval_split == 'random':
        return load_ratings_100k_all(
            ratings_path, random_seed=random_seed, fold=fold,
        )
    if fold is None or fold == 1:
        return load_ratings_100k(DATA_100K_TRAIN, DATA_100K_TEST)
    return load_ratings_100k(DATA_100K_TRAIN, DATA_100K_TEST, fold=fold)


def prepare_matrix_for_clustering(
    matrix,
    zscore,
    pca_var,
    wnmf_k,
    preprocess='minmax',
    feature_extraction='svd',
    svd_components=20,
    min_user_ratings=5,
    min_item_ratings=10,
    wnmf_init_method='inmed',
    inmed_trim=(5.0, 95.0),
    wnmf_n_epochs=50,
    return_prune_indices=False,
    paper_style=False,
    l2_normalize=False,
):
    """Sıra: prune → z-score → PCA → WNMF (→ opsiyonel L2-normalize, cosine için)."""
    from sklearn.preprocessing import MinMaxScaler, normalize

    prune_out = prune_sparse_matrix(
        matrix,
        min_user_ratings=min_user_ratings,
        min_item_ratings=min_item_ratings,
        return_indices=return_prune_indices,
    )
    if return_prune_indices:
        matrix, kept_u, kept_i = prune_out
    else:
        matrix = prune_out
        kept_u, kept_i = None, None
    if paper_style:
        matrix = impute_user_mean(matrix)
        matrix = zscore_column_normalize(matrix)
        print("  Paper: kullanıcı-ortalama imputation + sütun z-score")
    elif zscore:
        matrix = zscore_normalize(matrix)
        print("  Z-score normalizasyon uygulandı")
        # Pearson benzerliği ile K-Means (L2/Euclidean) geometrisini hizala.
        matrix = normalize(matrix, norm='l2', axis=1).astype(np.float32, copy=False)
        print("  L2 normalization uygulandı")
    if pca_var is not None:
        matrix = pca_variance_reduce(matrix, pca_var)
    R_matrix = matrix
    if preprocess == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        R_matrix = MinMaxScaler().fit_transform(R_matrix)
    elif preprocess == 'zscore':
        from sklearn.preprocessing import StandardScaler
        R_matrix = StandardScaler().fit_transform(R_matrix)
    elif preprocess == 'maxabs':
        from sklearn.preprocessing import MaxAbsScaler
        R_matrix = MaxAbsScaler().fit_transform(R_matrix)

    if feature_extraction == 'svd':
        from sklearn.decomposition import NMF
        from sklearn.preprocessing import normalize
        nmf = NMF(
            n_components=svd_components,
            random_state=42,
            max_iter=1000,
            init='nndsvda',
        )
        X_cluster = nmf.fit_transform(R_matrix)
        print(f"NMF reconstruction error: {nmf.reconstruction_err_:.2f}")
    elif feature_extraction == 'nmf':
        from sklearn.decomposition import NMF
        from sklearn.preprocessing import normalize
        nmf = NMF(n_components=svd_components, random_state=42, max_iter=300)
        R_nmf = np.asarray(R_matrix, dtype=np.float32)
        min_val = float(np.min(R_nmf))
        if min_val < 0.0:
            R_nmf = np.maximum(R_nmf, 0.0)  # sıfırla, kaydırma
        X_cluster = nmf.fit_transform(R_nmf)
    elif feature_extraction == 'pca':
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import normalize
        pca = PCA(n_components=svd_components, random_state=42)
        X_cluster = pca.fit_transform(R_matrix)
        if not paper_style:
            X_cluster = normalize(X_cluster)
        print(f"  PCA: {R_matrix.shape[1]} -> {X_cluster.shape[1]} components (fixed)")
    elif feature_extraction == 'wnmf':
        from sklearn.preprocessing import normalize
        n_components = wnmf_k if wnmf_k is not None else svd_components
        X_cluster = wnmf_feature_extract(
            R_matrix,
            n_components=n_components,
            n_epochs=wnmf_n_epochs,
            random_seed=42,
            init_method=wnmf_init_method,
            inmed_trim=inmed_trim,
        )
    else:
        X_cluster = R_matrix

    if l2_normalize:
        # Satır-bazlı L2 normalize → euclidean mesafe = cosine mesafe.
        # Düşük-boyut WNMF'te küme ayrışmasını belirgin artırır.
        X_cluster = normalize(X_cluster)
        print(f"  L2-normalize uygulandı (cosine-eşdeğer kümeleme)")

    matrix = X_cluster.astype(np.float32, copy=False)
    if return_prune_indices:
        return matrix, kept_u, kept_i
    return matrix


# ============================================================
# DATASET ÇALIŞTIRICI
# ============================================================

def run_dataset(dataset_name, matrix, K, out_root, algo_filter=None,
                use_lof=False, lof_n_neighbors=LOF_N_NEIGHBORS,
                lof_contamination=LOF_CONTAMINATION,
                max_workers: Optional[int] = None, out_suffix: str = '',
                args=None, run_id=None, cluster_metric: str = 'pearson',
                init_mode: str = 'mkpp', disable_gray_sheep: bool = False):
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name.upper()}  |  K={K}  |  Seed={SEED}")
    mode_str = f'LOF (n={lof_n_neighbors}, cont={lof_contamination})' \
               if use_lof else 'percentile (80th)'
    if disable_gray_sheep:
        mode_str = 'off'
    print(f"Gray sheep  : {mode_str}")
    print(f"Küme metrik : {cluster_metric} (pearson | euclidean)")
    print(f"Init modu   : {init_mode}")
    print(f"{'='*60}")

    # Yeni düzen: K yalnızca assign_suffix sonunda (..._k{K}); label önünde _k{K} yok.
    # Eski düzen (args yok): K ≠ varsayılan → label_k{K} + out_suffix.
    if dataset_name == 'ml100k':
        default_k = K_100K_DEFAULT
    elif dataset_name == 'filmtrust':
        default_k = K_FILMTRUST_DEFAULT
    else:
        default_k = K_1M_DEFAULT

    skip_existing = bool(getattr(args, 'skip_existing', False)) if args is not None else False

    jobs_meta = []
    skipped_existing = []
    for label, g_name, l_name in ALGO_CONFIG:
        if algo_filter and label not in algo_filter:
            print(f"  [{label}] atlandı (filtre)")
            continue
        assign_suffix = ''
        k_suffix = ''
        if args is not None:
            assign_suffix = format_assign_suffix_from_args(args, K, label=label, g_name=g_name)
        else:
            k_suffix = '' if K == default_k else f'_k{K}'
        save_dir = os.path.join(out_root, dataset_name, f"{label}{k_suffix}{out_suffix}{assign_suffix}")

        if skip_existing and os.path.exists(os.path.join(save_dir, 'assignments.npy')):
            print(f"  [{label}] atlandı (--skip-existing) -> {save_dir}")
            skipped_existing.append((label, save_dir))
            continue

        jobs_meta.append((label, g_name, l_name, save_dir))

    if not jobs_meta:
        if skipped_existing:
            print(
                f"\n{dataset_name.upper()} — tüm algoritmalar mevcut "
                f"({len(skipped_existing)} klasör --skip-existing ile atlandı)."
            )
        else:
            print(f"\n{dataset_name.upper()} — çalıştırılacak algoritma yok.")
        return

    nw = _resolve_pool_workers(max_workers, len(jobs_meta))

    if nw == 1:
        print("Algoritma kataloğu yükleniyor (1 kez)...")
        algo_map = {a['full_name']: a for a in get_all_algorithms_v3()}
        algo_map['GAHHO.OriginalGAHHO'] = {
            'full_name': 'GAHHO.OriginalGAHHO',
            'class':     OriginalGAHHO,
        }
        algo_map['GA.EliteMultiGA'] = {
            'full_name': 'GA.EliteMultiGA',
            'class':     GA.EliteMultiGA,
        }
        algo_map['KMEANS'] = {
            'full_name': 'KMEANS',
            'class': None,
        }
        algo_map['DOA.OriginalDOA'] = {
            'full_name': 'DOA.OriginalDOA',
            'class':     OriginalDOA,
        }
        for label, g_name, l_name, save_dir in jobs_meta:
            run_one(
                label,
                g_name,
                l_name,
                matrix,
                K,
                SEED,
                save_dir,
                algo_map,
                use_lof=use_lof,
                lof_n_neighbors=lof_n_neighbors,
                lof_contamination=lof_contamination,
                args=args,
                run_id=run_id,
                cluster_metric=cluster_metric,
                init_mode=init_mode,
                disable_gray_sheep=disable_gray_sheep,
            )
    else:
        print(
            f"Paralel atama: {len(jobs_meta)} iş, en fazla {nw} süreç "
            "(CPU-yoğun optimizasyon için süreç havuzu)."
        )
        baseline_epoch, global_epoch, local_epoch, pop_size = _resolve_train_hyperparams(args)
        jobs = [
            (
                label,
                g_name,
                l_name,
                matrix,
                K,
                SEED,
                save_dir,
                use_lof,
                lof_n_neighbors,
                lof_contamination,
                baseline_epoch,
                global_epoch,
                local_epoch,
                pop_size,
                args,
                run_id,
                cluster_metric,
                init_mode,
                disable_gray_sheep,
            )
            for label, g_name, l_name, save_dir in jobs_meta
        ]
        with ProcessPoolExecutor(max_workers=nw) as pool:
            list(pool.map(_mp_run_assignment_job, jobs))

    print(f"\n{dataset_name.upper()} tamamlandi -> {os.path.join(out_root, dataset_name)}/")


# ============================================================
# CLI
# ============================================================

def parse_args():
    labels = ALGO_LABELS
    p = argparse.ArgumentParser(
        description="Assignment üretici — percentile veya LOF gray sheep"
    )
    p.add_argument(
        '--dataset', choices=['100k', '1m', 'filmtrust', 'both'], default='both',
        help="Hangi dataset (default: both; filmtrust=FilmTrust ratings.txt)"
    )
    p.add_argument(
        '--algo', nargs='+', choices=labels, default=None,
        metavar='LABEL',
        help=f"Algoritmalar (default: hepsi): {labels}"
    )
    p.add_argument(
        '--last-only', action='store_true',
        help=f"Sadece son algoritmayı çalıştır: {labels[-1]}"
    )
    p.add_argument(
        '--lof', action='store_true',
        help="LOF tabanlı gray sheep kullan (default: sabit 80. percentile)"
    )
    p.add_argument(
        '--skip-existing', action='store_true',
        help='Hedef klasörde assignments.npy varsa o algoritmayı yeniden çalıştırma, '
             'mevcut yolu yazıp atla.',
    )
    p.add_argument(
        '--no-gray-sheep', action='store_true',
        help='Gray sheep tespitini tamamen kapat (LOF/percentile yok)',
    )
    p.add_argument(
        '--zscore', action='store_true',
        help='Clustering matrisine kullanıcı bazlı Z-score normalizasyon uygula'
    )
    p.add_argument(
        '--paper-mode', action='store_true',
        help='GOA makalesi (Ambikesh et al.): imputation + sütun z-score + sabit PCA-50 '
             '(veya --feature-extraction wnmf ile WNMF-50) + euclidean + random init + '
             'gray sheep kapalı + budama yok. Eval için wnmf_experiment --paper-mode kullanın.',
    )
    p.add_argument(
        '--init-mode', choices=['mkpp', 'random'], default=None,
        help="Centroid başlangıcı: mkpp (MkMeans++) veya random. "
             "Verilmezse paper-mode'da random, diğer modlarda mkpp.",
    )
    p.add_argument(
        '--init',
        choices=['random', 'kmeans++'],
        default='kmeans++',
        help='Başlangıç merkezi seçim yöntemi (yalnızca sklearn KMeans, B0_KMEANS)',
    )
    p.add_argument(
        '--fcm', action='store_true',
        help='Meta-sezgisel centroidlerinden sonra FCM ile J_m minimize et (post-refine); '
             'memberships.npy + güncellenmiş best_sol. B0_KMEANS ve KMEANS dalları atlanır. '
             'Arama hedefi değişmez (--cluster-metric euclidean ile uyumlu).',
    )
    p.add_argument(
        '--fcm-iter', type=int, default=50, metavar='N',
        help='--fcm post-refine iterasyon sayısı (default: 50)',
    )
    p.add_argument(
        '--fcm-m', type=float, default=2.0, metavar='M',
        help='FCM fuzzifier m (--cluster-metric fuzzy). Varsayılan: 2.0; m<=1 hard assignment.',
    )
    p.add_argument(
        '--fcm-m-suffix', action='store_true',
        help='Klasör adına _m15 gibi FCM-m etiketi ekle (mevcut atamaları ezmez).',
    )
    p.add_argument(
        '--p1', type=float, default=HA_AVOAHGS_P1_DEFAULT, metavar='P',
        help=f'HA_AVOAHGS: AVOA faz-1 olasılığı (default: {HA_AVOAHGS_P1_DEFAULT})',
    )
    p.add_argument(
        '--hgs-rate', type=float, default=HA_AVOAHGS_HGS_DEFAULT, metavar='R',
        help=f'HA_AVOAHGS: HGS uygulanan en iyi birey oranı (default: {HA_AVOAHGS_HGS_DEFAULT})',
    )
    p.add_argument(
        '--avoahgs-param-suffix', action='store_true',
        help='Klasör adına _p40_hgs70 gibi HA_AVOAHGS parametre etiketi ekle '
             '(varsayılan p1/hgs için etiket eklenmez; mevcut atamaları ezmez).',
    )
    p.add_argument(
        '--no-prune', action='store_true',
        help='Kullanıcı/film budamasını kapatır (min-user ve min-item 0). '
             'Klasör adında _pruneu... eklenmez. Eşdeğer: --min-user-ratings 0 --min-item-ratings 0.',
    )
    p.add_argument(
        '--min-user-ratings', type=int, default=5, metavar='N',
        help='Veri budama: kullanıcı başına minimum rating sayısı (default: 5; 0=budama yok)',
    )
    p.add_argument(
        '--min-item-ratings', type=int, default=10, metavar='N',
        help='Veri budama: film başına minimum rating sayısı (default: 10; 0=budama yok)',
    )
    p.add_argument(
        '--wnmf-features', type=int, default=None,
        metavar='K',
        help='[DEPRECATED] WNMF latent boyutu. Kaldırılacak; yerine '
             '--feature-extraction wnmf --svd-components K kullanın. '
             'Geçici olarak verilirse otomatik olarak bu eşdeğere dönüştürülür.',
    )
    p.add_argument(
        '--wnmf-init', choices=['random', 'inmed'], default='inmed',
        help='WNMF (--feature-extraction wnmf) başlangıcı: random veya inmed (default: inmed)',
    )
    p.add_argument(
        '--wnmf-epochs', type=int, default=50, metavar='N',
        help='WNMF (--feature-extraction wnmf) epoch sayısı (default: 50)',
    )
    p.add_argument(
        '--legacy-wnmf-suffix', action='store_true',
        help='Eski klasor adi: out_suffix icine _wnmfep{N} ekleme',
    )
    p.add_argument(
        '--inmed-trim-low', type=float, default=5.0, metavar='P',
        help='INMED trimmed mean alt yüzdelik (default: 5.0)',
    )
    p.add_argument(
        '--inmed-trim-high', type=float, default=95.0, metavar='P',
        help='INMED trimmed mean üst yüzdelik (default: 95.0)',
    )
    p.add_argument(
        '--cluster-metric', choices=['auto', 'pearson', 'euclidean', 'fuzzy'],
        default='auto',
        help='Sürü kümeleme fitness: auto = WNMF kullanıldıysa euclidean, '
             'aksi halde pearson (seyrek rating uzayı). fuzzy = FCM (--fcm-m, varsayılan 2.0).',
    )
    p.add_argument(
        '--fitness',
        choices=['wcss', 'latent_dev', 'knn_mae', 'knn_mae_legacy'],
        default='wcss',
        help='Kümeleme hedefi: wcss (varsayılan), latent_dev, knn_mae (ClusterPredictor '
             'val MAE), knn_mae_legacy (hızlı Pearson kNN örneklemesi).',
    )
    p.add_argument(
        '--cluster-objective',
        choices=['multi', 'wcss'],
        default='multi',
        help='Meta-sezgisel fitness: multi (WCSS+sil+CH, varsayılan) veya wcss '
             '(saf WCSS, B0 ile aynı hedef).',
    )
    p.add_argument(
        '--mo-weights',
        type=str,
        default=None,
        metavar='PRESET|W,S,C',
        help='multi-objective ağırlıkları: default (0.5/0.25/0.25), balanced, spread, '
             'fair, wcss_heavy veya wcss,sil,ch virgülle. spread: 0.30/0.35/0.35 '
             '(sıkışmayı azaltmak için önerilir).',
    )
    p.add_argument(
        '--centroid-repulsion-lambda',
        type=float,
        default=0.0,
        metavar='L',
        help='Yakın centroid çiftlerini cezalandır (0=kapalı). Önerilen: 0.10–0.30 '
             '(--mo-weights spread ile birlikte).',
    )
    p.add_argument(
        '--centroid-repulsion-dmin',
        type=float,
        default=None,
        metavar='D',
        help='Repulsion hedef min centroid mesafesi; verilmezse otomatik '
             '(medyan_kullanıcı_mesafesi/sqrt(K)).',
    )
    p.add_argument(
        '--wnmf-model-path', type=str, default=None, metavar='PATH',
        help='--fitness latent_dev: WNMF model/W matrisi kaynağı — .pkl/.joblib '
             '(model.U / model.W / user_factors), .npy, .npz veya assignment dizini '
             '(user_features.npy, model.pkl). Zorunlu.',
    )
    p.add_argument(
        '--centroid-algo',
        choices=['MFO', 'IWO', 'HA'],
        default='MFO',
        help='Yalnızca doğrudan CentroidOptimizer.optimize() testleri için (MFO/IWO/HA). '
             'generate_assignments --fitness knn_mae/latent_dev ile centroid aramasını '
             '--algo etiketi yürütür (HA_AVOAHGS, B1_HHO, …); ardından kmref uygulanır.',
    )
    p.add_argument(
        '--centroid-agents', type=int, default=None, metavar='N',
        help='--fitness latent_dev: popülasyon boyutu (varsayılan: POP_SIZE).',
    )
    p.add_argument(
        '--centroid-iter', type=int, default=None, metavar='N',
        help='--fitness latent_dev/knn_mae: epoch sayısı (varsayılan: BASELINE_EPOCH).',
    )
    p.add_argument(
        '--baseline-epoch', type=int, default=None, metavar='N',
        help=f'Meta-algoritma epoch (varsayılan: BASELINE_EPOCH={BASELINE_EPOCH}).',
    )
    p.add_argument(
        '--pop-size', type=int, default=None, metavar='N',
        help=f'Meta-algoritma popülasyon boyutu (varsayılan: POP_SIZE={POP_SIZE}).',
    )
    p.add_argument(
        '--l2-normalize', action=argparse.BooleanOptionalAction, default=False,
        help='Kümeleme öncesi özellik matrisini satır-bazlı L2-normalize et '
             '(euclidean=cosine). Düşük-boyut WNMF için küme ayrışmasını artırır.',
    )
    p.add_argument(
        '--b0-n-init', type=int, default=10, metavar='N',
        help='B0_KMEANS sklearn KMeans n_init (varsayılan: 10). n_init=1 ile '
             'baseline zayıflatılır → meta-sezgiselin "başlatma kalitesi" '
             'avantajı ölçülebilir (makale hipotezi).',
    )
    p.add_argument(
        '--centroid-train-sample', type=int, default=500, metavar='N',
        help='--fitness knn_mae: kNN indeksi için train rating örneği (varsayılan: 500).',
    )
    p.add_argument(
        '--centroid-val-sample', type=int, default=200, metavar='N',
        help='--fitness knn_mae: fitness için val rating örneği (varsayılan: 200).',
    )
    p.add_argument(
        '--centroid-knn-k', type=int, default=20, metavar='K',
        help='--fitness knn_mae: küme-içi kNN komşu sayısı (varsayılan: 20).',
    )
    p.add_argument(
        '--centroid-knn-sim',
        choices=['cosine', 'pearson'],
        default='cosine',
        help='--fitness knn_mae: ClusterPredictor benzerlik metriği (varsayılan: cosine).',
    )
    p.add_argument(
        '--centroid-bias-epochs', type=int, default=5, metavar='N',
        help='--fitness knn_mae: küme başına bias SGD epoch (varsayılan: 5; hız için düşük tutun).',
    )
    p.add_argument(
        '--min-common', type=int, default=3, metavar='N',
        help='kNN min_support / min_common (varsayılan: 3).',
    )
    p.add_argument(
        '--save-wnmf-u', type=str, default=None, metavar='DIR',
        help='--feature-extraction wnmf iken: WNMF U matrisini DIR içine '
             'ml100k_U.npy / ml1m_U.npy olarak kaydet.',
    )
    p.add_argument(
        '--pca', type=float, default=None, metavar='VAR',
        dest='pca_variance',
        help='PCA: birikimli açıklanan varyans eşiği (0–1, örn: 0.80). '
             'DEPRECATED --wnmf-features ile birlikte kullanılamaz.',
    )
    p.add_argument(
        '--k', nargs='+', type=int, default=None, metavar='K',
        help="Küme sayısı; birden fazla değer: --k 20 30 50 (her K için ayrı klasör). "
             "Tek değer: --k 70. Verilmezse 100K=90, 1M=150. "
             "--k-100k / --k-1m ile birlikte kullanılamaz."
    )
    p.add_argument(
        '--k-100k', type=int, default=None,
        help=f"ML-100K için K (default: {K_100K_DEFAULT})"
    )
    p.add_argument(
        '--k-1m', type=int, default=None,
        help=f"ML-1M için K (default: {K_1M_DEFAULT})"
    )
    p.add_argument(
        '--k-filmtrust', type=int, default=None,
        help=f"FilmTrust için K (default: {K_FILMTRUST_DEFAULT})",
    )
    p.add_argument(
        '--n-neighbors', type=int, default=LOF_N_NEIGHBORS,
        help=f"LOF komşu sayısı (default: {LOF_N_NEIGHBORS}, sadece --lof ile)"
    )
    p.add_argument(
        '--contamination', default=str(LOF_CONTAMINATION),
        help=f"LOF contamination: 'auto' veya float (default: {LOF_CONTAMINATION})"
    )
    p.add_argument('--data-100k', default=DATA_100K)
    p.add_argument('--data-1m',   default=DATA_1M)
    p.add_argument('--data-filmtrust', default=DATA_FILMTRUST)
    p.add_argument(
        '--train-only', action='store_true',
        help='Yalnızca train rating\'leri ile matris oluştur (test hücreleri 0). '
             'wnmf_experiment.py --eval-split / --fold ile hizalıdır.',
    )
    p.add_argument(
        '--eval-split',
        choices=['official', 'random'],
        default='random',
        help='--train-only ile ML-100K bölmesi: official=u1.base; '
             'random=u.data %%20 veya KFold (default: random).',
    )
    p.add_argument(
        '--fold', type=int, default=None, metavar='N',
        help='--train-only ile holdout fold (1–5). random: KFold; official: u{N}.base/test.',
    )
    p.add_argument(
        '--jobs', type=int, default=None,
        help='Paralel algoritma süreç sayısı (varsayılan: otomatik = CPU sayısına göre; '
             '1=sıralı, 2+=belirtilen sayıda süreç, 0=CPU ile sınırlandırılmış havuz)',
    )
    p.add_argument('--preprocess',
        choices=['none','minmax','zscore','maxabs'],
        default='minmax',
        help='Ön işleme yöntemi')
    p.add_argument('--feature-extraction',
        choices=['none','svd','pca','nmf','wnmf'],
        default='svd',
        help='Boyut indirgeme yöntemi')
    p.add_argument('--svd-components', type=int, default=20,
        help='SVD/PCA/NMF bileşen sayısı')
    p.add_argument(
        '--kmeans-refine', action=argparse.BooleanOptionalAction, default=True,
        help='Centroid bulunduktan sonra boş küme onarımı. Varsayılan (açık): '
             'meta-sezgiselin kendi atamaları korunur, yalnızca boş kümeler '
             'taşıma ile onarılır (atamalar ezilmez). B0_KMEANS atlanır.',
    )
    p.add_argument(
        '--kmeans-refine-overwrite', action=argparse.BooleanOptionalAction,
        default=False,
        help='Eski davranış: sklearn KMeans Lloyd ile rafine + atamaları topluca '
             'ez (init=meta centroidler, n_init=1). Algoritmalar arası farkları '
             'homojenleştirir; karşılaştırma için varsayılan KAPALI.',
    )
    p.add_argument(
        '--kmeans-refine-iter', type=int, default=300, metavar='N',
        help='KMeans refine-overwrite için max_iter (default: 300; önerilen 100-500).'
    )
    p.add_argument(
        '--out-root', type=str, default=None, metavar='DIR',
        help='Çıktı kök klasörü (mealpy/ altında veya mutlak yol). '
             'Verilmezse --early-stop ile assignments_lof_estop / assignments_estop, '
             'aksi halde assignments_lof / assignments kullanılır.',
    )
    p.add_argument(
        '--early-stop', action='store_true',
        help='Blok bazlı convergence early-stop (mealpy tek-algo, HA_AVOAHGS, IWO_HHO)',
    )
    p.add_argument(
        '--early-stop-max-epoch', type=int, default=EARLY_STOP_MAX_EPOCH, metavar='N',
        help=f'Early-stop üst sınır epoch (default: {EARLY_STOP_MAX_EPOCH})',
    )
    p.add_argument(
        '--early-stop-patience', type=int, default=EARLY_STOP_PATIENCE, metavar='N',
        help=f'Early-stop: ardışık blok (default: {EARLY_STOP_PATIENCE} → '
             f'{EARLY_STOP_PATIENCE * EARLY_STOP_BLOCK_SIZE} epoch)',
    )
    p.add_argument(
        '--early-stop-tolerance', type=float, default=EARLY_STOP_TOLERANCE,
        help=f'Early-stop: min fitness iyileşmesi (default: {EARLY_STOP_TOLERANCE})',
    )
    p.add_argument(
        '--early-stop-block', type=int, default=EARLY_STOP_BLOCK_SIZE, metavar='N',
        help=f'Early-stop: blok başına epoch (default: {EARLY_STOP_BLOCK_SIZE})',
    )
    p.add_argument(
        '--ha-adaptive-epoch', action='store_true',
        help='HA_AVOAHGS için epoch/patience/tolerance değerlerini veri boyutu, K ve pop '
             'üzerinden adaptif hesaplar (yalnızca --early-stop ile).',
    )
    p.add_argument(
        '--ha-adaptive-max-cap', type=int, default=600, metavar='N',
        help='--ha-adaptive-epoch aktifken HA için max epoch üst sınırı (default: 600).',
    )
    p.add_argument(
        '--ha-adaptive-min-tol', type=float, default=1e-6,
        help='--ha-adaptive-epoch aktifken tolerance alt sınırı (default: 1e-6).',
    )
    args = p.parse_args()
    if args.k is not None and (
        args.k_100k is not None or args.k_1m is not None or args.k_filmtrust is not None
    ):
        p.error(
            '--k (tek veya çoklu) ile --k-100k / --k-1m / --k-filmtrust birlikte kullanılamaz'
        )
    if args.pca_variance is not None:
        if not (0 < args.pca_variance <= 1):
            p.error('--pca (0, 1] aralığında olmalı (örn. 0.80)')
    if args.pca_variance is not None and args.wnmf_features is not None:
        p.error('--pca ile DEPRECATED --wnmf-features birlikte kullanılamaz')
    # Eski API köprüsü: --wnmf-features verilmişse feature-extraction'ı zorla
    # ve --svd-components ile senkronla. Böylece klasör adında tek bir _wnmf{N}
    # yer alır ve latent boyut çakışması (wnmf_features != svd_components) imkânsızlaşır.
    if args.wnmf_features is not None:
        warnings.warn(
            '--wnmf-features is deprecated and will be removed in a future version; '
            'use --feature-extraction wnmf --svd-components N instead.',
            FutureWarning,
            stacklevel=2,
        )
        if args.feature_extraction != 'wnmf':
            print(
                f"  Not: --wnmf-features={args.wnmf_features} verildi; "
                f"--feature-extraction otomatik olarak 'wnmf' yapıldı."
            )
            args.feature_extraction = 'wnmf'
        if args.svd_components != args.wnmf_features:
            print(
                f"  Not: --svd-components={args.svd_components} -> "
                f"{args.wnmf_features} olarak güncellendi (--wnmf-features ile eşleşti)."
            )
            args.svd_components = args.wnmf_features
    if getattr(args, 'no_prune', False):
        args.min_user_ratings = 0
        args.min_item_ratings = 0
    if args.min_user_ratings < 0:
        p.error('--min-user-ratings 0 veya daha büyük olmalı')
    if args.min_item_ratings < 0:
        p.error('--min-item-ratings 0 veya daha büyük olmalı')
    if not (0 <= args.inmed_trim_low < args.inmed_trim_high <= 100):
        p.error('--inmed-trim-low ve --inmed-trim-high için 0 <= low < high <= 100 olmalı')
    if args.fold is not None and not (1 <= args.fold <= 5):
        p.error('--fold 1..5 aralığında olmalı')
    if args.fold is not None and not args.train_only:
        print('  Uyarı: --fold yalnızca --train-only ile anlamlı; yok sayılıyor.')
        args.fold = None
    if args.train_only and args.eval_split != 'random' and args.dataset in ('1m', 'both'):
        print('  Not: ML-1M için --eval-split yok sayılır (her zaman rastgele/KFold).')
    args.pca = args.pca_variance
    if args.cluster_metric == 'auto':
        args.cluster_metric = (
            'euclidean'
            if (args.wnmf_features is not None or args.feature_extraction == 'wnmf')
            else 'pearson'
        )
    if (
        args.fitness in ('latent_dev', 'knn_mae', 'knn_mae_legacy')
        and not args.wnmf_model_path
        and args.feature_extraction != 'wnmf'
        and args.wnmf_features is None
    ):
        p.error(
            '--fitness latent_dev/knn_mae için --wnmf-model-path zorunlu '
            '(veya --feature-extraction wnmf ile pipeline W matrisi kullanılır)'
        )
    if args.fitness in ('knn_mae', 'knn_mae_legacy') and not args.train_only:
        print(
            '  Uyarı: --fitness knn_mae için --train-only önerilir '
            '(val örnekleri holdout/test bölmesinden gelir).',
        )
    if args.no_gray_sheep and args.lof:
        print('  Uyarı: --no-gray-sheep aktif; --lof yok sayılıyor.')
        args.lof = False
    if args.fcm and args.cluster_metric == 'fuzzy':
        print(
            '  Uyarı: --fcm ve --cluster-metric fuzzy birlikte; arama zaten FCM J ile '
            'yapılıyor, post-refine tekrarlı olabilir. Tipik: euclidean + --fcm.',
        )
    user_init_mode = args.init_mode
    if args.paper_mode:
        if args.lof:
            print('  Uyarı: --paper-mode aktif; --lof yok sayılıyor (gray sheep kapalı).')
            args.lof = False
        args.zscore = False
        args.paper_style = True
        args.preprocess = 'none'
        if not args.no_prune:
            print('  Not: --paper-mode aktif; veri budaması kapatıldı (makale: tam matris).')
            args.min_user_ratings = 0
            args.min_item_ratings = 0
        if args.cluster_metric != 'euclidean':
            print(
                f"  Uyarı: --paper-mode aktif; cluster metric "
                f"'{args.cluster_metric}' -> 'euclidean' olarak zorlandı."
            )
        args.cluster_metric = 'euclidean'
        args.init_mode = user_init_mode or 'random'
        args.disable_gray_sheep = True
        # Makale: sabit 50 boyut (PCA veya WNMF); %95 varyans değil.
        if args.feature_extraction == 'wnmf':
            args.pca_variance = None
            args.pca = None
            if int(args.svd_components) == 20:
                args.svd_components = 50
        elif args.feature_extraction == 'pca':
            args.pca_variance = None
            args.pca = None
            if int(args.svd_components) == 20:
                args.svd_components = 50
        elif args.feature_extraction in ('none', 'svd'):
            args.feature_extraction = 'pca'
            args.pca_variance = None
            args.pca = None
            if int(args.svd_components) == 20:
                args.svd_components = 50
    else:
        args.paper_style = False
        args.init_mode = user_init_mode or 'mkpp'
        args.disable_gray_sheep = bool(args.no_gray_sheep)
    return args

def _multi_start_init(matrix, K, pop_size, seed, n_restarts=10,
                      metric: str = 'pearson', init_mode: str = 'mkpp',
                      cluster_objective: str = 'multi',
                      fitness_config: dict | None = None):
    """
    n_restarts farklı seed ile MkMeans++ / random aday üret.
    Sıralama optimizasyondaki obj_func ile yapılır (boş küme cezası dahil).
    """
    if init_mode not in ('mkpp', 'random'):
        raise ValueError(f"init_mode bilinmiyor: {init_mode}")
    candidates = []
    rng = np.random.default_rng(seed=seed)
    lb, ub = _centroid_search_bounds(matrix, K)
    fc = dict(fitness_config or {'objective': cluster_objective})
    fc.setdefault('objective', cluster_objective)
    obj_fn = make_fitness_function(matrix, K, metric=metric, **fc)
    empty_penalty = 1e6
    n_candidates = max(n_restarts * 3, pop_size + 10)
    for i in range(n_candidates):
        if init_mode == 'mkpp':
            sols = mkmeans_plus_plus_init(
                matrix, K=K, n_solutions=1, seed=seed + i * 17, metric=metric,
            )
            sol = np.clip(np.asarray(sols[0], dtype=np.float64), lb, ub)
        else:
            # K rastgele kullanıcı satırı (k-means random init); [0,5] kutusu K=30'da
            # neredeyse her zaman boş küme üretir.
            idx = rng.choice(len(matrix), size=K, replace=False)
            sol = np.clip(
                np.asarray(matrix[idx], dtype=np.float64).flatten(), lb, ub,
            )
        fit = float(obj_fn(sol))
        candidates.append((fit, sol))

    candidates.sort(key=lambda x: x[0])
    best = [np.asarray(c[1], dtype=np.float64).copy() for c in candidates[:pop_size]]
    while len(best) < pop_size:
        src = candidates[len(best) % len(candidates)][1]
        best.append(np.asarray(src, dtype=np.float64).copy())
    valid = sum(1 for f, _ in candidates if f < empty_penalty * 0.99)
    lo, hi = candidates[0][0], candidates[min(pop_size - 1, len(candidates) - 1)][0]
    print(
        f"    Init({init_mode}): {n_candidates} aday -> en iyi {pop_size} secildi  "
        f"(fitness aralığı: {lo:.1f} – {hi:.1f}, geçerli: {valid}/{len(candidates)})"
    )
    return best
# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    args = parse_args()

    db_run_id = None
    if _DB_AVAILABLE and start_run is not None:
        try:
            prep_parts = []
            if args.min_user_ratings > 0 or args.min_item_ratings > 0:
                prep_parts.append(
                    f"prune_u{args.min_user_ratings}_i{args.min_item_ratings}"
                )
            if getattr(args, 'zscore', False):
                prep_parts.append('zscore')
            prep_parts.append(f"init_{args.init_mode}")
            if getattr(args, 'pca', None):
                if args.pca < 1.0:
                    prep_parts.append(f'pca{int(args.pca * 100)}pct')
                else:
                    prep_parts.append(f'pca{int(args.pca)}')
            wnmf_k_for_db_run = None
            if getattr(args, 'feature_extraction', None) == 'wnmf':
                wnmf_k_for_db_run = getattr(args, 'wnmf_features', None) or getattr(args, 'svd_components', None)
            elif getattr(args, 'wnmf_features', None):
                wnmf_k_for_db_run = args.wnmf_features
            if wnmf_k_for_db_run is not None:
                prep_parts.append(f'wnmf{wnmf_k_for_db_run}')
                if getattr(args, 'wnmf_features', None):
                    prep_parts.append(
                        f"{args.wnmf_init}_trim{args.inmed_trim_low:g}_{args.inmed_trim_high:g}"
                    )
            if args.train_only:
                prep_parts.append(
                    format_train_only_folder_suffix(
                        True, args.eval_split, args.fold,
                    ).lstrip('_') or 'trainonly',
                )
            preprocessing_run = '_'.join(prep_parts) if prep_parts else 'none'
            db_run_id = start_run(
                command=' '.join(sys.argv),
                dataset=args.dataset,
                k=None,
                preprocessing=preprocessing_run,
                note='gray_sheep=LOF' if args.lof else 'gray_sheep=percentile',
            )
            print(f"  DB run_id: {db_run_id}")
        except Exception as exc:
            print(f"  Uyari: start_run basarisiz ({exc})", file=sys.stderr)

    selected_algos = [ALGO_CONFIG[-1][0]] if args.last_only else args.algo

    # Contamination dönüşümü
    contamination = args.contamination
    if contamination != 'auto':
        contamination = float(contamination)

    # K değerleri: --k 20 30 50 → her dataset için bu liste; yoksa dataset başına tek K
    k_multi = list(args.k) if args.k is not None else None
    if k_multi is None:
        k_100k = args.k_100k or K_100K_DEFAULT
        k_1m   = args.k_1m   or K_1M_DEFAULT
        k_filmtrust = args.k_filmtrust or K_FILMTRUST_DEFAULT

    # Çıktı klasörü — LOF ve percentile ayrı tutulur; early-stop ayrı kökte
    if args.out_root:
        out_root = (
            args.out_root if os.path.isabs(args.out_root)
            else os.path.join(BASE_DIR, args.out_root)
        )
    elif args.early_stop:
        out_root = os.path.join(
            BASE_DIR, 'results',
            'assignments_lof_estop' if args.lof else 'assignments_estop',
        )
    else:
        out_root = os.path.join(
            BASE_DIR, 'results',
            'assignments_lof' if args.lof else 'assignments',
        )
    os.makedirs(out_root, exist_ok=True)

    print("=" * 60)
    print("ASSIGNMENT ÜRETİCİ")
    print("=" * 60)
    if args.disable_gray_sheep:
        gs_banner = 'Kapalı (--no-gray-sheep/paper-mode)'
    else:
        gs_banner = 'LOF (adaptif)' if args.lof else 'Percentile (sabit %20)'
    print(f"Gray sheep  : {gs_banner}")
    print(f"Paper mode  : {'Açık' if args.paper_mode else 'Kapalı'}")
    print(f"Algoritmalar: {selected_algos or [c[0] for c in ALGO_CONFIG]}")
    print(f"Dataset     : {args.dataset}")
    if args.train_only:
        fold_note = f'fold {args.fold}' if args.fold is not None else 'holdout'
        print(
            f"Train-only  : Açık ({args.eval_split}, {fold_note}, seed={SEED})"
        )
    if k_multi is not None:
        print(f"K listesi     : {k_multi}  (her değer sırayla işlenir)")
    else:
        print(
            f"ML-100K K   : {k_100k}  |  ML-1M K: {k_1m}  |  FilmTrust K: {k_filmtrust}"
        )
    if args.lof:
        print(f"LOF         : n_neighbors={args.n_neighbors}, contamination={contamination}")
    baseline_epoch, global_epoch, local_epoch, pop_size = _resolve_train_hyperparams(args)
    print(f"Epoch       : baseline={baseline_epoch}, global={global_epoch}, local={local_epoch}")
    if args.early_stop:
        print(
            f"Early-stop  : max={args.early_stop_max_epoch}, "
            f"block={args.early_stop_block}, "
            f"patience={args.early_stop_patience}, "
            f"tol={args.early_stop_tolerance}"
        )
        if getattr(args, 'ha_adaptive_epoch', False):
            print(
                f"HA adaptif  : açık (max_cap={args.ha_adaptive_max_cap}, "
                f"min_tol={args.ha_adaptive_min_tol})"
            )
    print(f"Pop size    : {pop_size}  |  Seed: {SEED}")
    if args.kmeans_refine and getattr(args, 'kmeans_refine_overwrite', False):
        print(
            f"KMeans ref  : OVERWRITE (Lloyd, max_iter={args.kmeans_refine_iter}; "
            f"atamalar ezilir; B0_KMEANS hariç)"
        )
    elif args.kmeans_refine:
        print("KMeans ref  : Boş küme onarımı (atamalar korunur; B0_KMEANS hariç)")
    else:
        print("KMeans ref  : Kapalı (--no-kmeans-refine)")
    if getattr(args, 'fcm', False):
        print(
            f"FCM post    : Açık (meta sonrası J_m, max_iter={args.fcm_iter}; "
            f"B0_KMEANS/KMEANS atlanır)"
        )
    feat_bits = []
    if args.min_user_ratings <= 0 and args.min_item_ratings <= 0:
        feat_bits.append('Prune: kapalı')
    else:
        feat_bits.append(
            f"Prune(u>={args.min_user_ratings}, i>={args.min_item_ratings})"
        )
    if getattr(args, 'paper_style', False):
        feat_bits.append('Col-Z-score+impute (GOA paper)')
    elif args.zscore:
        feat_bits.append('Z-score')
    if args.pca_variance is not None:
        feat_bits.append(f"PCA >={args.pca_variance:.0%} var.")
    if args.wnmf_features is not None:
        feat_bits.append(
            f"WNMF k={args.wnmf_features} init={args.wnmf_init} trim=({args.inmed_trim_low:g},{args.inmed_trim_high:g})"
        )
    if feat_bits:
        print(f"Matris      : ham + " + ' + '.join(feat_bits))
    print(f"Küme metrik : {args.cluster_metric}")
    print(f"Fitness     : {args.fitness}" + (
        f" (centroid_algo={args.centroid_algo})"
        if args.fitness in ('latent_dev', 'knn_mae') else ''
    ))
    if args.fitness == 'wcss':
        print(f"Cluster obj : {args.cluster_objective} (meta-sezgisel hedef)")
        if args.cluster_objective == 'multi':
            mw = _parse_mo_weights_arg(args.mo_weights)
            print(
                f"MO weights  : wcss={mw[0]:g}, sil={mw[1]:g}, ch={mw[2]:g} "
                f"(preset={args.mo_weights or 'default'})"
            )
            if float(args.centroid_repulsion_lambda or 0.0) > 0.0:
                dmin = args.centroid_repulsion_dmin
                dtxt = f"{dmin:g}" if dmin is not None else 'auto'
                print(
                    f"Repulsion   : lambda={args.centroid_repulsion_lambda:g}, "
                    f"d_min={dtxt}"
                )
    if args.fitness in ('knn_mae', 'knn_mae_legacy'):
        print(
            f"Centroid MAE: train_sample={args.centroid_train_sample}, "
            f"val_sample={args.centroid_val_sample}, knn_k={args.centroid_knn_k}"
        )
    if args.fitness in ('latent_dev', 'knn_mae') and args.wnmf_model_path:
        print(f"W kaynağı   : {args.wnmf_model_path}")
    print(f"Çıktı kökü  : {out_root}")
    if args.jobs is None:
        print("Paralellik    : otomatik (CPU sayısına göre; --jobs 1 ile sıralı mod)")
    elif args.jobs == 1:
        print("Paralellik    : sıralı (tek süreç)")
    elif args.jobs <= 0:
        print("Paralellik    : süreç havuzu (iş ve CPU sayısına göre)")
    else:
        print(f"Paralellik    : en fazla {args.jobs} süreç")
    print("=" * 60)

    t_total = time.time()

    def _finish_db_run():
        if db_run_id is not None and _DB_AVAILABLE and finish_run is not None:
            try:
                finish_run(db_run_id)
            except Exception as exc:
                print(f"  Uyari: finish_run basarisiz ({exc})", file=sys.stderr)

    def _pool_cap():
        if args.jobs == 1:
            return 1
        if args.jobs is None or args.jobs <= 0:
            return None  # otomatik: _resolve_pool_workers CPU sayısını kullanır
        return args.jobs

    prune_suffix = format_prune_folder_suffix(
        args.min_user_ratings, args.min_item_ratings,
    )
    out_suffix = format_out_suffix_from_args(args)

    try:
        if args.dataset in ('100k', 'both'):
            raw_train_100k = raw_test_100k = None
            if args.fitness in ('knn_mae', 'knn_mae_legacy'):
                raw_train_100k, raw_test_100k = _fetch_100k_train_test_arrays(
                    eval_split=args.eval_split,
                    fold=args.fold,
                    random_seed=SEED,
                    ratings_path=args.data_100k,
                )
            if args.train_only:
                print(f"\nML-100K train-only yükleniyor (eval-split={args.eval_split}, fold={args.fold})")
                matrix_100k = load_movielens_train_only_100k(
                    eval_split=args.eval_split,
                    fold=args.fold,
                    random_seed=SEED,
                    ratings_path=args.data_100k,
                )
            else:
                print(f"\nML-100K yükleniyor: {args.data_100k}")
                matrix_100k = load_movielens(args.data_100k)
            prep_100k = prepare_matrix_for_clustering(
                matrix_100k,
                args.zscore,
                args.pca_variance,
                args.wnmf_features,
                preprocess=args.preprocess,
                feature_extraction=args.feature_extraction,
                svd_components=args.svd_components,
                min_user_ratings=args.min_user_ratings,
                min_item_ratings=args.min_item_ratings,
                wnmf_init_method=args.wnmf_init,
                inmed_trim=(args.inmed_trim_low, args.inmed_trim_high),
                wnmf_n_epochs=args.wnmf_epochs,
                return_prune_indices=(args.fitness in ('knn_mae', 'knn_mae_legacy')),
                paper_style=getattr(args, 'paper_style', False),
                l2_normalize=getattr(args, 'l2_normalize', False),
            )
            if args.fitness in ('knn_mae', 'knn_mae_legacy'):
                matrix_100k, kept_u, kept_i = prep_100k
                args.centroid_train_ratings = _remap_ratings_to_pruned(
                    raw_train_100k, kept_u, kept_i,
                )
                args.centroid_val_ratings = _remap_ratings_to_pruned(
                    raw_test_100k, kept_u, kept_i,
                )
                print(
                    f"  knn_mae ratings: train={len(args.centroid_train_ratings):,}, "
                    f"val={len(args.centroid_val_ratings):,} (prune sonrası indeksler)",
                )
            else:
                matrix_100k = prep_100k
            if args.save_wnmf_u and args.feature_extraction == 'wnmf':
                os.makedirs(args.save_wnmf_u, exist_ok=True)
                u_path = os.path.join(args.save_wnmf_u, 'ml100k_U.npy')
                np.save(u_path, matrix_100k)
                print(f"  WNMF U kaydedildi: {u_path}")
            if k_multi is not None:
                for K in k_multi:
                    run_dataset(
                        'ml100k', matrix_100k, K, out_root,
                        algo_filter=selected_algos,
                        use_lof=args.lof,
                        lof_n_neighbors=args.n_neighbors,
                        lof_contamination=contamination,
                        max_workers=_pool_cap(),
                        out_suffix=out_suffix,
                        args=args,
                        run_id=db_run_id,
                        cluster_metric=args.cluster_metric,
                        init_mode=args.init_mode,
                        disable_gray_sheep=args.disable_gray_sheep,
                    )
            else:
                run_dataset(
                    'ml100k', matrix_100k, k_100k, out_root,
                    algo_filter=selected_algos,
                    use_lof=args.lof,
                    lof_n_neighbors=args.n_neighbors,
                    lof_contamination=contamination,
                    max_workers=_pool_cap(),
                    out_suffix=out_suffix,
                    args=args,
                    run_id=db_run_id,
                    cluster_metric=args.cluster_metric,
                    init_mode=args.init_mode,
                    disable_gray_sheep=args.disable_gray_sheep,
                )

        if args.dataset in ('1m', 'both'):
            if args.train_only:
                print(f"\nML-1M train-only yükleniyor (fold={args.fold})")
                matrix_1m = load_movielens_train_only_1m(
                    fold=args.fold,
                    random_seed=SEED,
                    ratings_path=args.data_1m,
                )
            else:
                print(f"\nML-1M yükleniyor: {args.data_1m}")
                matrix_1m = load_movielens_1m(args.data_1m)
            matrix_1m = prepare_matrix_for_clustering(
                matrix_1m,
                args.zscore,
                args.pca_variance,
                args.wnmf_features,
                preprocess=args.preprocess,
                feature_extraction=args.feature_extraction,
                svd_components=args.svd_components,
                min_user_ratings=args.min_user_ratings,
                min_item_ratings=args.min_item_ratings,
                wnmf_init_method=args.wnmf_init,
                inmed_trim=(args.inmed_trim_low, args.inmed_trim_high),
                wnmf_n_epochs=args.wnmf_epochs,
                paper_style=getattr(args, 'paper_style', False),
                l2_normalize=getattr(args, 'l2_normalize', False),
            )
            if args.save_wnmf_u and args.feature_extraction == 'wnmf':
                os.makedirs(args.save_wnmf_u, exist_ok=True)
                u_path = os.path.join(args.save_wnmf_u, 'ml1m_U.npy')
                np.save(u_path, matrix_1m)
                print(f"  WNMF U kaydedildi: {u_path}")
            if k_multi is not None:
                for K in k_multi:
                    run_dataset(
                        'ml1m', matrix_1m, K, out_root,
                        algo_filter=selected_algos,
                        use_lof=args.lof,
                        lof_n_neighbors=args.n_neighbors,
                        lof_contamination=contamination,
                        max_workers=_pool_cap(),
                        out_suffix=out_suffix,
                        args=args,
                        run_id=db_run_id,
                        cluster_metric=args.cluster_metric,
                        init_mode=args.init_mode,
                        disable_gray_sheep=args.disable_gray_sheep,
                    )
            else:
                run_dataset(
                    'ml1m', matrix_1m, k_1m, out_root,
                    algo_filter=selected_algos,
                    use_lof=args.lof,
                    lof_n_neighbors=args.n_neighbors,
                    lof_contamination=contamination,
                    max_workers=_pool_cap(),
                    out_suffix=out_suffix,
                    args=args,
                    run_id=db_run_id,
                    cluster_metric=args.cluster_metric,
                    init_mode=args.init_mode,
                    disable_gray_sheep=args.disable_gray_sheep,
                )

        if args.dataset in ('filmtrust',):
            if args.fitness in ('knn_mae', 'knn_mae_legacy'):
                print(
                    '  Uyari: FilmTrust + knn_mae fitness henuz desteklenmiyor; '
                    'atlaniyor.',
                    file=sys.stderr,
                )
            elif args.train_only:
                print(
                    f"\nFilmTrust train-only yukleniyor "
                    f"(fold={args.fold}, seed={SEED})"
                )
                matrix_ft = load_filmtrust_train_only(
                    random_seed=SEED,
                    ratings_path=args.data_filmtrust,
                    fold=args.fold,
                )
            else:
                print(f"\nFilmTrust yukleniyor: {args.data_filmtrust}")
                _, _, _, _, load_filmtrust_matrix = _import_wnmf_loaders()
                matrix_ft = load_filmtrust_matrix(args.data_filmtrust)
            if args.fitness not in ('knn_mae', 'knn_mae_legacy'):
                prep_ft = prepare_matrix_for_clustering(
                    matrix_ft,
                    args.zscore,
                    args.pca_variance,
                    args.wnmf_features,
                    preprocess=args.preprocess,
                    feature_extraction=args.feature_extraction,
                    svd_components=args.svd_components,
                    min_user_ratings=args.min_user_ratings,
                    min_item_ratings=args.min_item_ratings,
                    wnmf_init_method=args.wnmf_init,
                    inmed_trim=(args.inmed_trim_low, args.inmed_trim_high),
                    wnmf_n_epochs=args.wnmf_epochs,
                    paper_style=getattr(args, 'paper_style', False),
                l2_normalize=getattr(args, 'l2_normalize', False),
                )
                matrix_ft = prep_ft
                if args.save_wnmf_u and args.feature_extraction == 'wnmf':
                    os.makedirs(args.save_wnmf_u, exist_ok=True)
                    u_path = os.path.join(args.save_wnmf_u, 'filmtrust_U.npy')
                    np.save(u_path, matrix_ft)
                    print(f"  WNMF U kaydedildi: {u_path}")
                if k_multi is not None:
                    for K in k_multi:
                        run_dataset(
                            'filmtrust', matrix_ft, K, out_root,
                            algo_filter=selected_algos,
                            use_lof=args.lof,
                            lof_n_neighbors=args.n_neighbors,
                            lof_contamination=contamination,
                            max_workers=_pool_cap(),
                            out_suffix=out_suffix,
                            args=args,
                            run_id=db_run_id,
                            cluster_metric=args.cluster_metric,
                            init_mode=args.init_mode,
                            disable_gray_sheep=args.disable_gray_sheep,
                        )
                else:
                    run_dataset(
                        'filmtrust', matrix_ft, k_filmtrust, out_root,
                        algo_filter=selected_algos,
                        use_lof=args.lof,
                        lof_n_neighbors=args.n_neighbors,
                        lof_contamination=contamination,
                        max_workers=_pool_cap(),
                        out_suffix=out_suffix,
                        args=args,
                        run_id=db_run_id,
                        cluster_metric=args.cluster_metric,
                        init_mode=args.init_mode,
                        disable_gray_sheep=args.disable_gray_sheep,
                    )

        print(f"\n{'='*60}")
        print(f"TAMAMLANDI — toplam {(time.time()-t_total)/60:.1f} dakika")
        print(f"Çıktı: {out_root}/")
        print("=" * 60)
    finally:
        _finish_db_run()