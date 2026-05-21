"""
generate_item_assignments.py
============================
Film (item) kümeleme üretici — kullanıcı kümeleme (generate_assignments.py)'nin
ITEM-AXIS muadili. Transpose matris üzerinde çalışır:
    Ham matris : (n_users × n_items)
    Item matris: (n_items × n_users)   ← clustering bu eksen üzerinde

Algoritmalar (user-side ile birebir aynı label):
    B0_KMEANS, B1_HHO, B2_HGS, B3_MFO,
    H4_MFO+HHO (sadece global MFO fazı), IWO_HHO,
    LIT_GWO, LIT_PSO, LIT_GOA, LIT_SSA, LIT_CIRCLESA,
    HA_AVOAHGS, B_AVOA
    (Label'lar user-side ile BİREBİR — fusion eşleştirmesi için.)
    NOT: Hibrit algoritmalar (H4_MFO+HHO) sadece global fazı çalıştırır;
         item clustering'de iki-fazlı hibrit yoktur.
Feature extract  : svd (sklearn.NMF), none
Gray sheep       : YOK (her zaman tüm-False maske kaydedilir — default)
Early stopping   : Destekli — generate_assignments.py'deki helper kullanılır

Çıktı:
    mealpy/results/item_assignments/{ml100k|ml1m}/{label}_..._k{K}/
        assignments.npy     ← item kümeleri (n_items,)
        best_sol.npy        ← centroid çözümü (K, n_features) düzleştirilmiş
        gray_sheep_mask.npy ← tüm-False (downstream uyumluluk için)
        assignment_summary.csv

K rehberi:
    ML-100K (≈1682 film) : K_item = 20   (default)
    ML-1M   (≈3952 film) : K_item = 50   (default)

Kullanım:
    python generate_item_assignments.py --dataset 100k --k 20 \\
        --feature-extraction svd --svd-components 50 --no-gray-sheep \\
        --algo B0_KMEANS IWO_HHO LIT_GWO HA_AVOAHGS
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(BASE_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
_OPT_DIR = os.path.join(_REPO_ROOT, 'optimizers')
if _OPT_DIR not in sys.path:
    sys.path.insert(0, _OPT_DIR)

from generate_assignments import (
    BASELINE_EPOCH,
    DATA_100K,
    DATA_1M,
    EARLY_STOP_BLOCK_SIZE,
    EARLY_STOP_MAX_EPOCH,
    EARLY_STOP_PATIENCE,
    EARLY_STOP_TOLERANCE,
    POP_SIZE,
    SEED,
    _make_problem,
    _multi_start_init,
    _resolve_pool_workers,
    _sklearn_kmeans_init,
    format_init_mode_folder_suffix,
    format_prune_folder_suffix,
    load_movielens_1m,
    prune_sparse_matrix,
    run_single,
    run_single_with_early_stop,
    run_ha_avoahgs_with_early_stop,
    run_iwo_hho_with_early_stop,
    save_assignment,
    zscore_normalize,
)
from mealpy_comparison_v2 import (
    compute_wcss_fast,
    get_all_algorithms_v3,
    get_special_params,
    load_movielens,
)

# ============================================================
# AYARLAR — minimal kapsam
# ============================================================

K_ITEM_100K_DEFAULT = 20
K_ITEM_1M_DEFAULT   = 50

# (label, g_name, l_name) — kullanıcı kümeleme ile aynı şema.
# NOT: Label'lar user-side `generate_assignments.py` ile BİREBİR aynı olmak ZORUNDA
# (wnmf_experiment.py fusion'da item klasörünü user label'ına göre arıyor).
# NOT: Hibrit algoritmalar (H4_MFO+HHO) item clustering'de sadece global (g_name) fazını çalıştırır.
ALGO_CONFIG_ITEM = [
    ('B0_KMEANS',     None,                      None),
    ('B1_HHO',        'HHO.OriginalHHO',         None),
    ('B2_HGS',        'HGS.OriginalHGS',         None),
    ('B3_MFO',        'MFO.OriginalMFO',         None),
    ('H4_MFO+HHO',    'MFO.OriginalMFO',         'HHO.OriginalHHO'),
    ('IWO_HHO',       'IWO_HHO',                 None),
    ('LIT_GWO',       'GWO.OriginalGWO',         None),
    ('LIT_PSO',       'PSO.OriginalPSO',         None),
    ('LIT_GOA',       'GOA.OriginalGOA',         None),
    ('LIT_SSA',       'SSA.OriginalSSA',         None),
    ('LIT_CIRCLESA',  'CircleSA.OriginalCircleSA', None),
    ('HA_AVOAHGS',    None,                      None),
    ('B_AVOA',        'AVOA.OriginalAVOA',       None),
]
ALGO_LABELS_ITEM = [c[0] for c in ALGO_CONFIG_ITEM]


# ============================================================
# ITEM MATRİSİ HAZIRLAMA
# ============================================================

def _prune_and_track_item_indices(matrix, min_user_ratings, min_item_ratings):
    """prune_sparse_matrix ile aynı mantık, ek olarak hangi orijinal item
    indekslerinin hayatta kaldığını da döndürür.

    Returns:
        (pruned_matrix, kept_item_indices)
        kept_item_indices: (n_kept_items,) int32 — pruned matrix'in satır j'si
                          orijinal matriste kept_item_indices[j] itemına karşılık gelir.
    """
    if min_user_ratings <= 0 and min_item_ratings <= 0:
        return matrix.astype(np.float32, copy=False), np.arange(matrix.shape[1], dtype=np.int32)

    pruned = matrix.astype(np.float32, copy=True)
    item_indices = np.arange(matrix.shape[1], dtype=np.int32)
    prev_shape = None
    iteration = 0

    while prev_shape != pruned.shape:
        prev_shape = pruned.shape
        iteration += 1

        if min_user_ratings > 0:
            user_counts = np.count_nonzero(pruned, axis=1)
            keep_users = user_counts >= int(min_user_ratings)
            pruned = pruned[keep_users]

        if min_item_ratings > 0:
            item_counts = np.count_nonzero(pruned, axis=0)
            keep_items_mask = item_counts >= int(min_item_ratings)
            pruned = pruned[:, keep_items_mask]
            item_indices = item_indices[keep_items_mask]

    total = pruned.size
    nonzero = np.count_nonzero(pruned)
    sparsity = 1 - (nonzero / total if total > 0 else 0.0)
    print(
        f"  Prune tamamlandı ({iteration} iter): "
        f"shape={pruned.shape}, sparsity={sparsity:.3f}"
    )
    return pruned.astype(np.float32, copy=False), item_indices


def prepare_item_matrix_for_clustering(
    matrix,
    zscore,
    preprocess='none',
    feature_extraction='svd',
    svd_components=50,
    min_user_ratings=5,
    min_item_ratings=10,
):
    """
    User-item matris -> item-feature matris dönüşümü.

    Sıra:
        1. prune (user-axis): seyrek kullanıcı/film at (orijinal item indeksleri takip edilir)
        2. transpose: rows = items
        3. (opsiyonel) zscore: per-row → her FİLM için bias çıkar
        4. preprocess scaler (minmax/zscore/maxabs/none)
        5. feature extraction (svd|none) — svd = sklearn NMF + L2

    Dönüş:
        (X_cluster, kept_item_indices)
        X_cluster         : (n_pruned_items × n_features) — clustering girişi
        kept_item_indices : (n_pruned_items,) int32 — hangi orijinal item indekslerinin kaldığı
    """
    from sklearn.preprocessing import normalize

    n_items_orig = matrix.shape[1]
    matrix, kept_item_indices = _prune_and_track_item_indices(
        matrix,
        min_user_ratings=min_user_ratings,
        min_item_ratings=min_item_ratings,
    )
    print(f"  Item indeks eşlemesi: {len(kept_item_indices)}/{n_items_orig} item kaldı")

    item_matrix = matrix.T.astype(np.float32, copy=False)
    print(f"  Transpose: item matrisi shape={item_matrix.shape}")

    if zscore:
        item_matrix = zscore_normalize(item_matrix)
        print("  Z-score (film-bazlı) normalizasyon uygulandı")
        item_matrix = normalize(item_matrix, norm='l2', axis=1).astype(
            np.float32, copy=False,
        )
        print("  L2 normalization uygulandı")

    R_matrix = item_matrix
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
        nmf = NMF(
            n_components=svd_components,
            random_state=42,
            max_iter=1000,
            init='nndsvda',
        )
        R_nmf = np.maximum(R_matrix, 0.0)
        X_cluster = normalize(nmf.fit_transform(R_nmf))
        print(f"  NMF reconstruction error: {nmf.reconstruction_err_:.2f}")
    elif feature_extraction == 'none':
        X_cluster = R_matrix
    else:
        raise ValueError(
            f"--feature-extraction {feature_extraction!r} desteklenmiyor; "
            "item clustering minimal kapsamda yalnız 'svd' veya 'none' kabul edilir."
        )

    return X_cluster.astype(np.float32, copy=False), kept_item_indices


# ============================================================
# ALGORİTMA ÇALIŞTIRICI (minimal)
# ============================================================

def _run_one_item(
    label,
    g_name,
    matrix,            # (n_items × n_features) — items in rows
    K,
    seed,
    save_dir,
    algo_map,
    cluster_metric: str = 'euclidean',
    init_mode: str = 'mkpp',
    args=None,
    item_indices: Optional[np.ndarray] = None,  # orijinal item indeks eşlemesi
):
    print(
        f"\n  [{label}/item] başlıyor "
        f"(metric: {cluster_metric}, init: {init_mode}, shape={matrix.shape})..."
    )
    t0 = time.time()
    convergence_history = None
    actual_epochs = None

    if label == 'B0_KMEANS':
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize

        X = normalize(matrix, norm='l2')
        km = KMeans(
            n_clusters=K,
            init=_sklearn_kmeans_init(args),
            n_init=10,
            random_state=seed,
            max_iter=500,
            verbose=0,
        )
        km.fit(X)
        best_sol = km.cluster_centers_.flatten()
        best_fit = float(km.inertia_)
        assignments = km.labels_.astype(np.int32, copy=False)
    elif g_name == 'IWO_HHO' or label == 'IWO_HHO':
        init = _multi_start_init(
            matrix, K=K, pop_size=POP_SIZE, seed=seed, n_restarts=10,
            metric=cluster_metric, init_mode=init_mode,
        )
        if args is not None and getattr(args, 'early_stop', False):
            es_max = getattr(args, 'early_stop_max_epoch', EARLY_STOP_MAX_EPOCH)
            best_sol, best_fit, actual_epochs, convergence_history = (
                run_iwo_hho_with_early_stop(
                    matrix, K, init, es_max, POP_SIZE, seed,
                    metric=cluster_metric,
                    patience=getattr(args, 'early_stop_patience', EARLY_STOP_PATIENCE),
                    tolerance=getattr(args, 'early_stop_tolerance', EARLY_STOP_TOLERANCE),
                    block_size=getattr(args, 'early_stop_block', EARLY_STOP_BLOCK_SIZE),
                )
            )
            print(f"    Early-stop epoch: {actual_epochs}/{es_max}")
        else:
            from optimizers.iwo_hho import IWO_HHO_Clustering

            problem = _make_problem(matrix, K, metric=cluster_metric)
            iwo_hho = IWO_HHO_Clustering(
                epoch=BASELINE_EPOCH,
                pop_size=POP_SIZE,
                seed=seed,
            )
            try:
                best_sol, best_fit = iwo_hho.solve(
                    problem, starting_solutions=init[:POP_SIZE],
                )
            except TypeError:
                best_sol, best_fit = iwo_hho.solve(problem)
    elif label == 'HA_AVOAHGS':
        init = _multi_start_init(
            matrix, K=K, pop_size=POP_SIZE, seed=seed, n_restarts=10,
            metric=cluster_metric, init_mode=init_mode,
        )
        if args is not None and getattr(args, 'early_stop', False):
            es_max = getattr(args, 'early_stop_max_epoch', EARLY_STOP_MAX_EPOCH)
            best_sol, best_fit, actual_epochs, convergence_history = (
                run_ha_avoahgs_with_early_stop(
                    matrix, K, init, es_max, POP_SIZE,
                    metric=cluster_metric,
                    patience=getattr(args, 'early_stop_patience', EARLY_STOP_PATIENCE),
                    tolerance=getattr(args, 'early_stop_tolerance', EARLY_STOP_TOLERANCE),
                    block_size=getattr(args, 'early_stop_block', EARLY_STOP_BLOCK_SIZE),
                )
            )
            print(f"    Early-stop epoch: {actual_epochs}/{es_max}")
        else:
            from optimizers.HA_AVOAHGS import HA_AVOAHGS

            problem = _make_problem(matrix, K, metric=cluster_metric)
            model = HA_AVOAHGS(
                epoch=BASELINE_EPOCH,
                pop_size=POP_SIZE,
                p1=0.4,
                hgs_rate=0.7,
            )
            try:
                model.solve(problem, starting_solutions=init[:POP_SIZE])
            except TypeError:
                model.solve(problem)
            best_sol = model.g_best.solution
            best_fit = float(model.g_best.target.fitness)
    else:
        if g_name not in algo_map:
            raise KeyError(
                f"Algoritma kataloğunda yok: {g_name!r} (label={label}). "
                f"ALGO_CONFIG_ITEM desteklenen algoritmalar: "
                f"{ALGO_LABELS_ITEM}."
            )
        init = _multi_start_init(
            matrix, K=K, pop_size=POP_SIZE, seed=seed, n_restarts=10,
            metric=cluster_metric, init_mode=init_mode,
        )
        if args is not None and getattr(args, 'early_stop', False):
            es_max = getattr(args, 'early_stop_max_epoch', EARLY_STOP_MAX_EPOCH)
            best_sol, best_fit, actual_epochs, convergence_history = (
                run_single_with_early_stop(
                    algo_map[g_name], matrix, K, init, es_max, POP_SIZE,
                    metric=cluster_metric,
                    patience=getattr(args, 'early_stop_patience', EARLY_STOP_PATIENCE),
                    tolerance=getattr(args, 'early_stop_tolerance', EARLY_STOP_TOLERANCE),
                    block_size=getattr(args, 'early_stop_block', EARLY_STOP_BLOCK_SIZE),
                )
            )
            print(f"    Early-stop epoch: {actual_epochs}/{es_max}")
        else:
            best_sol, best_fit = run_single(
                algo_map[g_name], matrix, K, init, BASELINE_EPOCH, POP_SIZE,
                metric=cluster_metric,
            )

    if label != 'B0_KMEANS':
        _, assignments = compute_wcss_fast(matrix, best_sol, K, metric=cluster_metric)
        assignments = assignments.astype(np.int32, copy=False)

    n_items = len(assignments)
    gray_mask = np.zeros(n_items, dtype=bool)

    best_wcss, _ = compute_wcss_fast(
        matrix, best_sol, K, metric=cluster_metric,
    )

    extra_data = {'threshold': 0.0, 'threshold_label': 'disabled'}
    if convergence_history is not None:
        extra_data['convergence_history'] = convergence_history
        extra_data['early_stop_epochs'] = actual_epochs

    save_assignment(
        assignments, gray_mask, best_sol, best_wcss,
        save_dir, extra_data, label=label, K=K, args=args,
        run_id=None, seed=seed, memberships=None,
    )
    # item_indices.npy: hangi orijinal item indekslerinin bu assignment'a karşılık geldiğini sakla.
    # wnmf_experiment.py fusion kodu bunu okuyarak assignment'ı orijinal item uzayına yayar.
    if item_indices is not None:
        np.save(os.path.join(save_dir, 'item_indices.npy'), item_indices)
        print(f"  [{label}/item] item_indices kaydedildi: {len(item_indices)} item")
    print(f"  [{label}/item] tamamlandı — {time.time()-t0:.1f}s")


def _mp_run_item_job(job):
    """Windows spawn için modül seviyesinde worker."""
    (label, g_name, matrix, K, seed, save_dir,
     cluster_metric, init_mode, args, item_indices) = job
    algo_map = _build_algo_map()
    _run_one_item(
        label, g_name, matrix, K, seed, save_dir, algo_map,
        cluster_metric=cluster_metric, init_mode=init_mode, args=args,
        item_indices=item_indices,
    )


def _build_algo_map():
    """Sadece minimal kapsam için katalog. LIT_GWO mealpy katalogundan gelir."""
    algo_map = {a['full_name']: a for a in get_all_algorithms_v3()}
    return algo_map


# ============================================================
# DATASET ÇALIŞTIRICI
# ============================================================

def run_item_dataset(dataset_name, item_matrix, K, out_root,
                    algo_filter=None,
                    max_workers: Optional[int] = None,
                    out_suffix: str = '',
                    args=None,
                    cluster_metric: str = 'euclidean',
                    init_mode: str = 'mkpp',
                    item_indices: Optional[np.ndarray] = None):
    print(f"\n{'='*60}")
    print(f"ITEM DATASET: {dataset_name.upper()}  |  K={K}  |  Seed={SEED}")
    print(f"Küme metrik : {cluster_metric}")
    print(f"Init modu   : {init_mode}")
    print(f"Item shape  : {item_matrix.shape}")
    print(f"{'='*60}")

    skip_existing = bool(getattr(args, 'skip_existing', False)) if args is not None else False

    jobs_meta = []
    skipped_existing = []
    for label, g_name, _l_name in ALGO_CONFIG_ITEM:
        if algo_filter and label not in algo_filter:
            print(f"  [{label}] atlandı (filtre)")
            continue
        assign_suffix = ''
        if args is not None:
            assign_suffix = (
                f'_{args.preprocess}_{args.feature_extraction}'
                f'{args.svd_components}_k{K}'
            )
        save_dir = os.path.join(
            out_root, dataset_name, f"{label}{out_suffix}{assign_suffix}",
        )
        if skip_existing and os.path.exists(os.path.join(save_dir, 'assignments.npy')):
            print(f"  [{label}] atlandı (--skip-existing) -> {save_dir}")
            skipped_existing.append((label, save_dir))
            continue
        jobs_meta.append((label, g_name, save_dir))

    if not jobs_meta:
        if skipped_existing:
            print(
                f"\n{dataset_name.upper()}/item — tüm algoritmalar mevcut "
                f"({len(skipped_existing)} klasör --skip-existing ile atlandı)."
            )
        else:
            print(f"\n{dataset_name.upper()}/item — çalıştırılacak algoritma yok.")
        return

    nw = _resolve_pool_workers(max_workers, len(jobs_meta))

    if nw == 1:
        print("Algoritma kataloğu yükleniyor (1 kez)...")
        algo_map = _build_algo_map()
        for label, g_name, save_dir in jobs_meta:
            _run_one_item(
                label, g_name, item_matrix, K, SEED, save_dir, algo_map,
                cluster_metric=cluster_metric, init_mode=init_mode, args=args,
                item_indices=item_indices,
            )
    else:
        print(
            f"Paralel item-atama: {len(jobs_meta)} iş, en fazla {nw} süreç."
        )
        jobs = [
            (label, g_name, item_matrix, K, SEED, save_dir,
             cluster_metric, init_mode, args, item_indices)
            for label, g_name, save_dir in jobs_meta
        ]
        with ProcessPoolExecutor(max_workers=nw) as pool:
            list(pool.map(_mp_run_item_job, jobs))

    print(
        f"\n{dataset_name.upper()}/item tamamlandı -> "
        f"{os.path.join(out_root, dataset_name)}/"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    labels = ALGO_LABELS_ITEM
    p = argparse.ArgumentParser(
        description="Item (film) kümeleme üretici — generate_assignments.py'nin "
                    "transpose-eksen muadili. "
                    f"Algoritmalar: {labels}. Gray sheep yok (default: kapalı)."
    )
    p.add_argument(
        '--dataset', choices=['100k', '1m', 'both'], default='both',
        help='Hangi dataset (default: both)',
    )
    p.add_argument(
        '--algo', nargs='+', choices=labels, default=None, metavar='LABEL',
        help=f"Algoritmalar (default: hepsi): {labels}",
    )
    p.add_argument(
        '--k', nargs='+', type=int, default=None, metavar='K',
        help=f"Film kümesi sayısı; çoklu: --k 20 30 50. "
             f"ML-100K: 20-50 önerilen (default: {K_ITEM_100K_DEFAULT}), "
             f"ML-1M: 50-100 önerilen (default: {K_ITEM_1M_DEFAULT}). "
             "--k-100k / --k-1m ile birlikte kullanılamaz.",
    )
    p.add_argument(
        '--k-100k', type=int, default=None,
        help=f"ML-100K item K (default: {K_ITEM_100K_DEFAULT})",
    )
    p.add_argument(
        '--k-1m', type=int, default=None,
        help=f"ML-1M item K (default: {K_ITEM_1M_DEFAULT})",
    )
    p.add_argument(
        '--no-gray-sheep', action='store_true',
        help='Item clustering minimal: gray sheep zaten yok. '
             'Bu flag yalnız klasör adında uyum sağlamak için kabul edilir.',
    )
    p.add_argument(
        '--zscore', action='store_true',
        help='Item bazlı Z-score (her filmin rating ortalaması/std çıkar)',
    )
    p.add_argument(
        '--init-mode', choices=['mkpp', 'random'], default='mkpp',
        help='Meta-sezgisel centroid başlangıcı (default: mkpp)',
    )
    p.add_argument(
        '--init', choices=['random', 'kmeans++'], default='kmeans++',
        help='sklearn KMeans (B0_KMEANS) init yöntemi',
    )
    p.add_argument(
        '--cluster-metric', choices=['pearson', 'euclidean'], default='euclidean',
        help='Item clustering fitness (default: euclidean — feature uzayında '
             'NMF/SVD sonrası anlamlı).',
    )
    p.add_argument(
        '--preprocess', choices=['none', 'minmax', 'zscore', 'maxabs'],
        default='none',
        help='Item matris üstüne scaler (default: none — film popülerliğini '
             'bozmamak için).',
    )
    p.add_argument(
        '--feature-extraction', choices=['none', 'svd'], default='svd',
        help='Boyut indirgeme (default: svd = sklearn NMF). Minimal kapsamda '
             'sadece bu ikisi.',
    )
    p.add_argument(
        '--svd-components', type=int, default=50,
        help='SVD/NMF bileşen sayısı (default: 50)',
    )
    p.add_argument(
        '--no-prune', action='store_true',
        help='Kullanıcı/film budamasını kapatır (min-user ve min-item 0).',
    )
    p.add_argument(
        '--min-user-ratings', type=int, default=5,
        help='Veri budama: kullanıcı başına minimum rating (default: 5)',
    )
    p.add_argument(
        '--min-item-ratings', type=int, default=10,
        help='Veri budama: film başına minimum rating (default: 10)',
    )
    p.add_argument(
        '--skip-existing', action='store_true',
        help='Hedef klasörde assignments.npy varsa atla.',
    )
    p.add_argument(
        '--jobs', type=int, default=None,
        help='Paralel algoritma süreç sayısı (None=otomatik, 1=sıralı)',
    )
    p.add_argument('--data-100k', default=DATA_100K)
    p.add_argument('--data-1m', default=DATA_1M)
    p.add_argument(
        '--early-stop', action='store_true',
        help='Mealpy single-algoritma koşularında blok bazlı erken durdurma',
    )
    p.add_argument(
        '--early-stop-max-epoch', type=int, default=EARLY_STOP_MAX_EPOCH,
        help=f'Early-stop üst sınır epoch (default: {EARLY_STOP_MAX_EPOCH})',
    )
    p.add_argument(
        '--early-stop-patience', type=int, default=EARLY_STOP_PATIENCE,
        help=f'Early-stop: ardışık blok (default: {EARLY_STOP_PATIENCE})',
    )
    p.add_argument(
        '--early-stop-tolerance', type=float, default=EARLY_STOP_TOLERANCE,
        help=f'Early-stop: min fitness iyileşmesi (default: {EARLY_STOP_TOLERANCE})',
    )
    p.add_argument(
        '--early-stop-block', type=int, default=EARLY_STOP_BLOCK_SIZE,
        help=f'Early-stop: blok başına epoch (default: {EARLY_STOP_BLOCK_SIZE})',
    )

    args = p.parse_args()

    if args.k is not None and (args.k_100k is not None or args.k_1m is not None):
        p.error('--k (tek veya çoklu) ile --k-100k / --k-1m birlikte kullanılamaz')
    if getattr(args, 'no_prune', False):
        args.min_user_ratings = 0
        args.min_item_ratings = 0
    if args.min_user_ratings < 0 or args.min_item_ratings < 0:
        p.error('--min-{user,item}-ratings 0 veya daha büyük olmalı')

    args.disable_gray_sheep = True
    return args


def main():
    args = parse_args()

    selected_algos = args.algo

    k_multi = list(args.k) if args.k is not None else None
    if k_multi is None:
        k_100k = args.k_100k or K_ITEM_100K_DEFAULT
        k_1m   = args.k_1m   or K_ITEM_1M_DEFAULT

    out_root = os.path.join(BASE_DIR, 'results', 'item_assignments')
    os.makedirs(out_root, exist_ok=True)

    prune_suffix = format_prune_folder_suffix(
        args.min_user_ratings, args.min_item_ratings,
    )
    zscore_suffix = '_zscore' if args.zscore else ''
    metric_suffix_map = {'euclidean': '_euc'}
    metric_suffix = metric_suffix_map.get(args.cluster_metric, '')
    init_suffix = format_init_mode_folder_suffix(args.init_mode)
    no_gs_suffix = '_nogs'
    out_suffix = (
        prune_suffix + zscore_suffix + metric_suffix + init_suffix + no_gs_suffix
    )

    print("=" * 60)
    print("ITEM ASSIGNMENT ÜRETİCİ")
    print("=" * 60)
    print(f"Algoritmalar : {selected_algos or ALGO_LABELS_ITEM}")
    print(f"Dataset      : {args.dataset}")
    if k_multi is not None:
        print(f"K listesi    : {k_multi}")
    else:
        print(f"ML-100K K    : {k_100k}  |  ML-1M K: {k_1m}")
    print(f"Feature ext. : {args.feature_extraction} (components={args.svd_components})")
    print(f"Preprocess   : {args.preprocess}")
    print(f"Z-score      : {args.zscore}")
    print(f"Cluster metr.: {args.cluster_metric}")
    print(f"Init mode    : {args.init_mode}")
    print(f"Çıktı kökü   : {out_root}")
    print("=" * 60)

    def _pool_cap():
        if args.jobs == 1:
            return 1
        if args.jobs is None or args.jobs <= 0:
            return None
        return args.jobs

    t_total = time.time()
    try:
        if args.dataset in ('100k', 'both'):
            print(f"\nML-100K yükleniyor: {args.data_100k}")
            raw = load_movielens(args.data_100k)
            item_matrix, item_indices = prepare_item_matrix_for_clustering(
                raw,
                zscore=args.zscore,
                preprocess=args.preprocess,
                feature_extraction=args.feature_extraction,
                svd_components=args.svd_components,
                min_user_ratings=args.min_user_ratings,
                min_item_ratings=args.min_item_ratings,
            )
            ks = k_multi if k_multi is not None else [k_100k]
            for K in ks:
                run_item_dataset(
                    'ml100k', item_matrix, K, out_root,
                    algo_filter=selected_algos,
                    max_workers=_pool_cap(),
                    out_suffix=out_suffix,
                    args=args,
                    cluster_metric=args.cluster_metric,
                    init_mode=args.init_mode,
                    item_indices=item_indices,
                )

        if args.dataset in ('1m', 'both'):
            print(f"\nML-1M yükleniyor: {args.data_1m}")
            raw = load_movielens_1m(args.data_1m)
            item_matrix, item_indices = prepare_item_matrix_for_clustering(
                raw,
                zscore=args.zscore,
                preprocess=args.preprocess,
                feature_extraction=args.feature_extraction,
                svd_components=args.svd_components,
                min_user_ratings=args.min_user_ratings,
                min_item_ratings=args.min_item_ratings,
            )
            ks = k_multi if k_multi is not None else [k_1m]
            for K in ks:
                run_item_dataset(
                    'ml1m', item_matrix, K, out_root,
                    algo_filter=selected_algos,
                    max_workers=_pool_cap(),
                    out_suffix=out_suffix,
                    args=args,
                    cluster_metric=args.cluster_metric,
                    init_mode=args.init_mode,
                    item_indices=item_indices,
                )

        print(f"\n{'='*60}")
        print(f"ITEM CLUSTERING TAMAMLANDI — {(time.time()-t_total)/60:.1f} dakika")
        print(f"Çıktı: {out_root}/")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\nKullanıcı tarafından kesildi.")


if __name__ == '__main__':
    main()
