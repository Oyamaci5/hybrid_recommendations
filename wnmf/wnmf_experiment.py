"""
wnmf_experiment.py
==================
4 algoritma için WNMF karşılaştırma deneyi.

3 senaryo:
    global       — kümeleme yok, tüm kullanıcılar tek model (ablation baseline)
    cluster_full — her küme hem U hem V öğrenir (eski versiyon, karşılaştırma için)
    cluster_sharedV — global V + küme bazlı U (önerilen versiyon)

Kullanım:
    python wnmf_experiment.py --dataset 100k
    python wnmf_experiment.py --dataset 100k --algo H4_MFO+HHO
    python wnmf_experiment.py --dataset 100k --no-global
    python wnmf_experiment.py --dataset 100k --mode sharedV   # sadece sharedV
    python wnmf_experiment.py --dataset 100k --mode all       # her iki mod
    python wnmf_experiment.py --dataset 100k --jobs 4        # küme başına 4 süreç
    python wnmf_experiment.py --dataset 1m --k 70            # .../B1_HHO_k70/ vb.
    python wnmf_experiment.py --k-100k 90 --k-1m 70        # dataset başına ayrı K
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
sys.path.insert(0, BASE_DIR)

from wnmf_model import ClusterWNMF, WNMFModel, WNMFSharedV
from wnmf_utils import (
    load_ratings_100k,
    load_ratings_1m,
    load_assignment,
    split_by_cluster,
    remap_user_ids,
    save_results,
)

# ============================================================
# AYARLAR
# ============================================================

DATA_100K_TRAIN = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-100k', 'u1.base')
DATA_100K_TEST  = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-100k', 'u1.test')
DATA_1M         = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-1m', 'ratings.dat')
ASSIGN_ROOT     = os.path.join(os.path.dirname(BASE_DIR), 'mealpy','results', 'assignments_lof')
OUT_ROOT        = os.path.join(os.path.dirname(BASE_DIR), 'results', 'wnmf')

# generate_assignments.py ile aynı: K bu değerlerden biriyse klasörde _k{K} eki yok
ASSIGN_K_DEFAULT_100K = 90
ASSIGN_K_DEFAULT_1M   = 150

LATENT_DIM       = 20
LEARNING_RATE    = 0.01
REGULARIZATION   = 0.01
N_EPOCHS_GLOBAL  = 100   # global V ve global baseline için
N_EPOCHS_CLUSTER = 50    # küme U eğitimi için (daha az veri, daha az epoch yeterli)
RANDOM_SEED      = 42

ALGO_LABELS = ['B1_HHO', 'B2_HGS', 'H1_HHO+HGS', 'H4_MFO+HHO']


def _resolved_assignment_k(
    k_explicit_dataset: Optional[int],
    k_global: Optional[int],
    default_k: int,
) -> int:
    """Hangi K ile üretilmiş assignment klasörü okunacak (generate_assignments ile uyumlu)."""
    if k_explicit_dataset is not None:
        return k_explicit_dataset
    if k_global is not None:
        return k_global
    return default_k


def _algo_assignment_dir(
    assign_root: str,
    dataset_name: str,
    label: str,
    k_used: int,
) -> str:
    """
    Örnek: k_used=70, default 90 ise → .../ml100k/B1_HHO_k70
    k_used=90, default 90 ise → .../ml100k/B1_HHO
    """
    default_k = ASSIGN_K_DEFAULT_100K if dataset_name == 'ml100k' else ASSIGN_K_DEFAULT_1M
    suffix    = '' if k_used == default_k else f'_k{k_used}'
    return os.path.join(assign_root, dataset_name, f'{label}{suffix}')


def _resolve_pool_workers(requested: Optional[int], n_tasks: int) -> int:
    """İş sayısını ve CPU’yu aşmayacak worker sayısı."""
    if n_tasks <= 0:
        return 1
    cpu = os.cpu_count() or 1
    if requested is None or requested <= 0:
        cap = cpu
    else:
        cap = requested
    return max(1, min(cap, n_tasks))


def _parallel_cluster_map(func, jobs: list, max_workers: Optional[int]):
    """Sıra korunur (ex.map). Tek iş / tek worker ise süreç havuzu kurulmaz."""
    if not jobs:
        return []
    nw = _resolve_pool_workers(max_workers, len(jobs))
    if nw == 1:
        return [func(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=nw) as pool:
        return list(pool.map(func, jobs))


def _mp_fit_predict_cluster_full(job):
    """
    Pickle edilebilir küme işi — full WNMF (U+V).
    job: (c_train, c_test, n_items, latent_dim, lr, reg, n_epochs, random_seed)
    """
    (
        c_train,
        c_test,
        n_items,
        latent_dim,
        lr,
        reg,
        n_epochs,
        random_seed,
    ) = job
    c_train_r, c_test_r, _, n_loc = remap_user_ids(c_train, c_test, n_items)
    model = WNMFModel(
        n_users        = n_loc,
        n_items        = n_items,
        latent_dim     = latent_dim,
        learning_rate  = lr,
        regularization = reg,
        n_epochs       = n_epochs,
        random_seed    = random_seed,
    )
    model.fit(c_train_r)
    if len(c_test_r) == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    pred = model.predict(
        c_test_r[:, 0].astype(np.int32),
        c_test_r[:, 1].astype(np.int32),
    )
    return c_test_r[:, 2].astype(np.float32), pred.astype(np.float32)


def _mp_fit_predict_cluster_sharedV(job):
    """
    Pickle edilebilir küme işi — sabit V, sadece U.
    job: (cid, V_global, c_train, c_test, n_items, latent_dim, lr, reg,
          n_epochs, random_seed_base)
    """
    (
        cid,
        V_global,
        c_train   ,
        c_test,
        n_items,
        latent_dim,
        lr,
        reg,
        n_epochs,
        random_seed_base,
    ) = job
    c_train_r, c_test_r, _, n_loc = remap_user_ids(c_train, c_test, n_items)
    cluster_model = ClusterWNMF(
        n_users        = n_loc,
        n_items        = n_items,
        latent_dim     = latent_dim,
        V_shared       = V_global,
        learning_rate  = lr,
        regularization = reg,
        n_epochs       = n_epochs,
        random_seed    = int(random_seed_base) + int(cid),
    )
    cluster_model.fit_cluster_U(c_train_r)
    if len(c_test_r) == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    pred = cluster_model.predict(
        c_test_r[:, 0].astype(np.int32),
        c_test_r[:, 1].astype(np.int32),
    )
    return c_test_r[:, 2].astype(np.float32), pred.astype(np.float32)


# ============================================================
# SENARYO 1: GLOBAL WNMF BASELINE
# ============================================================

def run_global_wnmf(train, test, n_items, verbose=False):
    """
    Tüm kullanıcılara tek model — ablation baseline.
    'Kümeleme eklemek ne kadar iyileştiriyor?' sorusunu cevaplar.
    """
    print("\n  [Global WNMF] başlıyor...")
    t0      = time.time()
    n_users = int(train[:, 0].max()) + 1

    model = WNMFModel(
        n_users        = n_users,
        n_items        = n_items,
        latent_dim     = LATENT_DIM,
        learning_rate  = LEARNING_RATE,
        regularization = REGULARIZATION,
        n_epochs       = N_EPOCHS_GLOBAL,
        random_seed    = RANDOM_SEED,
    )
    model.fit(train, verbose=verbose)
    mae, rmse = model.evaluate(test)

    print(f"  [Global WNMF] MAE={mae:.4f}  RMSE={rmse:.4f}  ({time.time()-t0:.1f}s)")
    return {
        'scenario'    : 'global',
        'algo_label'  : 'GLOBAL_WNMF',
        'mae'         : mae,
        'rmse'        : rmse,
        'gray_mae'    : float('nan'),
        'gray_rmse'   : float('nan'),
        'n_clusters'  : 1,
        'n_train'     : len(train),
        'n_test'      : len(test),
        'time_seconds': time.time() - t0,
    }


# ============================================================
# SENARYO 2: CLUSTER WNMF — HER KÜME U+V ÖĞRENİR
# ============================================================

def run_cluster_full(train, test, assignments, gray_mask,
                     n_items, algo_label, verbose=False,
                     max_workers: Optional[int] = None):
    """
    Her küme bağımsız U ve V öğrenir.
    Karşılaştırma için tutulur — SharedV ile farkı görmek için.
    """
    print(f"\n  [{algo_label} | Full Cluster] başlıyor...")
    t0 = time.time()

    cluster_train, gray_train = split_by_cluster(train, assignments, gray_mask)
    cluster_test,  gray_test  = split_by_cluster(test,  assignments, gray_mask)

    jobs = []
    for cid, c_train in cluster_train.items():
        c_test = cluster_test.get(cid, np.empty((0, 3), dtype=np.float32))
        if len(c_train) < 5 or len(c_test) == 0:
            continue
        jobs.append(
            (
                c_train,
                c_test,
                n_items,
                LATENT_DIM,
                LEARNING_RATE,
                REGULARIZATION,
                N_EPOCHS_CLUSTER,
                RANDOM_SEED,
            )
        )

    parts = _parallel_cluster_map(_mp_fit_predict_cluster_full, jobs, max_workers)
    true_chunks = [p[0] for p in parts if len(p[0]) > 0]
    pred_chunks = [p[1] for p in parts if len(p[1]) > 0]
    if true_chunks:
        all_true = np.concatenate(true_chunks).tolist()
        all_pred = np.concatenate(pred_chunks).tolist()
    else:
        all_true, all_pred = [], []

    mae, rmse   = _compute_metrics(all_true, all_pred)
    gray_mae, gray_rmse = _run_gray_sheep(
        gray_train, gray_test, n_items, 'full'
    )

    elapsed = time.time() - t0
    print(f"  [{algo_label} | Full] MAE={mae:.4f} RMSE={rmse:.4f} | "
          f"Gray MAE={gray_mae:.4f}  ({elapsed:.1f}s)")

    return {
        'scenario'    : 'cluster_full',
        'algo_label'  : algo_label,
        'mae'         : mae,
        'rmse'        : rmse,
        'gray_mae'    : gray_mae,
        'gray_rmse'   : gray_rmse,
        'n_clusters'  : len(cluster_train),
        'n_train'     : len(train),
        'n_test'      : len(test),
        'time_seconds': elapsed,
    }


# ============================================================
# SENARYO 3: CLUSTER WNMF — SHARED V (ÖNERİLEN)
# ============================================================

def run_cluster_sharedV(train, test, assignments, gray_mask,
                        n_items, algo_label, verbose=False,
                        max_workers: Optional[int] = None):
    """
    Global V + küme bazlı U.

    Aşama 1: Tüm train verisiyle global V öğren (~global WNMF kadar sürer)
    Aşama 2: Her küme için V sabit, sadece U eğit (çok hızlı)

    Bu yaklaşımın avantajı:
    - Film embedding'leri tüm veriden öğrenilir → kaliteli
    - Kullanıcı embedding'leri küme özelinde → uzmanlaşmış
    - Az verili kümelerde bile iyi sonuç verir
    """
    print(f"\n  [{algo_label} | SharedV] başlıyor...")
    t0      = time.time()
    n_users = int(train[:, 0].max()) + 1

    # Aşama 1: Global V öğren
    shared = WNMFSharedV(
        n_users_global  = n_users,
        n_items         = n_items,
        latent_dim      = LATENT_DIM,
        learning_rate   = LEARNING_RATE,
        regularization  = REGULARIZATION,
        n_epochs_global = N_EPOCHS_GLOBAL,
        random_seed     = RANDOM_SEED,
    )
    shared.fit_global_V(train, verbose=verbose)
    V_global = shared.V.copy()

    # Kümeleri böl
    cluster_train, gray_train = split_by_cluster(train, assignments, gray_mask)
    cluster_test,  gray_test  = split_by_cluster(test,  assignments, gray_mask)

    jobs = []
    for cid, c_train in cluster_train.items():
        c_test = cluster_test.get(cid, np.empty((0, 3), dtype=np.float32))
        if len(c_train) < 5 or len(c_test) == 0:
            continue
        jobs.append(
            (
                cid,
                V_global,
                c_train,
                c_test,
                n_items,
                LATENT_DIM,
                LEARNING_RATE,
                REGULARIZATION,
                N_EPOCHS_CLUSTER,
                RANDOM_SEED,
            )
        )

    parts = _parallel_cluster_map(_mp_fit_predict_cluster_sharedV, jobs, max_workers)
    true_chunks = [p[0] for p in parts if len(p[0]) > 0]
    pred_chunks = [p[1] for p in parts if len(p[1]) > 0]
    n_clusters_fit = len(true_chunks)
    if true_chunks:
        all_true = np.concatenate(true_chunks).tolist()
        all_pred = np.concatenate(pred_chunks).tolist()
    else:
        all_true, all_pred = [], []

    mae, rmse = _compute_metrics(all_true, all_pred)

    # Gray sheep — shared V kullanarak
    gray_mae, gray_rmse = _run_gray_sheep_sharedV(
        shared, gray_train, gray_test, n_items
    )

    elapsed = time.time() - t0
    print(f"  [{algo_label} | SharedV] MAE={mae:.4f} RMSE={rmse:.4f} | "
          f"Gray MAE={gray_mae:.4f}  ({elapsed:.1f}s)")

    return {
        'scenario'    : 'cluster_sharedV',
        'algo_label'  : algo_label,
        'mae'         : mae,
        'rmse'        : rmse,
        'gray_mae'    : gray_mae,
        'gray_rmse'   : gray_rmse,
        'n_clusters'  : n_clusters_fit,
        'n_train'     : len(train),
        'n_test'      : len(test),
        'time_seconds': elapsed,
    }


# ============================================================
# GRAY SHEEP YARDIMCILAR
# ============================================================

def _run_gray_sheep(gray_train, gray_test, n_items, mode='full'):
    """Gray sheep için bağımsız WNMFModel."""
    if len(gray_train) < 5 or len(gray_test) == 0:
        return float('nan'), float('nan')

    g_train_r, g_test_r, _, n_gs = remap_user_ids(gray_train, gray_test, n_items)

    model = WNMFModel(
        n_users        = n_gs,
        n_items        = n_items,
        latent_dim     = LATENT_DIM,
        learning_rate  = LEARNING_RATE,
        regularization = REGULARIZATION,
        n_epochs       = N_EPOCHS_CLUSTER,
        random_seed    = RANDOM_SEED,
    )
    model.fit(g_train_r)

    if len(g_test_r) == 0:
        return float('nan'), float('nan')
    return model.evaluate(g_test_r)


def _run_gray_sheep_sharedV(shared, gray_train, gray_test, n_items):
    """Gray sheep için SharedV'den türetilmiş model."""
    if len(gray_train) < 5 or len(gray_test) == 0:
        return float('nan'), float('nan')

    g_train_r, g_test_r, _, n_gs = remap_user_ids(gray_train, gray_test, n_items)

    gs_model = shared.make_cluster_model(
        n_users_cluster  = n_gs,
        n_epochs_cluster = N_EPOCHS_CLUSTER,
        random_seed      = RANDOM_SEED,
    )
    gs_model.fit_cluster_U(g_train_r)

    if len(g_test_r) == 0:
        return float('nan'), float('nan')
    return gs_model.evaluate(g_test_r)


def _compute_metrics(true_list, pred_list):
    """Toplanmış tahminlerden MAE ve RMSE hesapla."""
    if not true_list:
        return float('nan'), float('nan')
    true_arr = np.array(true_list, dtype=np.float32)
    pred_arr = np.array(pred_list, dtype=np.float32)
    errors   = true_arr - pred_arr
    return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors ** 2)))


# ============================================================
# DATASET ÇALIŞTIRICI
# ============================================================

def run_dataset(dataset_name, train, test, algo_filter=None,
                run_global=True, mode='sharedV',
                cluster_workers: Optional[int] = None,
                assign_root: Optional[str] = None,
                assignment_k: Optional[int] = None,
                assignment_k_100k: Optional[int] = None,
                assignment_k_1m: Optional[int] = None):
    """
    Bir dataset üzerinde tüm senaryoları çalıştır.

    mode parametresi:
        'sharedV' — sadece SharedV (önerilen, hızlı)
        'full'    — sadece her küme U+V (eski versiyon)
        'all'     — her ikisi de (karşılaştırma için)

    Assignment klasörü: mealpy/results/assignments_lof/{dataset}/{label} veya
    K ≠ varsayılan ise {label}_k{K} (generate_assignments ile aynı kural).
    """
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name.upper()}  |  mode={mode}")
    print(f"{'='*60}")

    root = assign_root if assign_root is not None else ASSIGN_ROOT
    dk   = ASSIGN_K_DEFAULT_100K if dataset_name == 'ml100k' else ASSIGN_K_DEFAULT_1M
    k_ds = assignment_k_100k if dataset_name == 'ml100k' else assignment_k_1m
    k_used = _resolved_assignment_k(k_ds, assignment_k, dk)
    print(f"Assignment K  : {k_used} (klasör eki: {'yok' if k_used == dk else f'_k{k_used}'})")
    print(f"Assignment kök: {root}")

    n_items = int(train[:, 1].max()) + 1
    results = []
    out_dir = os.path.join(OUT_ROOT, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # Global WNMF baseline
    if run_global:
        row = run_global_wnmf(train, test, n_items)
        row['dataset'] = dataset_name
        results.append(row)

    # Her algoritma
    for label in ALGO_LABELS:
        if algo_filter and label not in algo_filter:
            print(f"\n  [{label}] atlandı (filtre)")
            continue

        assign_dir = _algo_assignment_dir(root, dataset_name, label, k_used)
        if not os.path.exists(assign_dir):
            print(f"\n  [{label}] ATLANDI — assignment bulunamadı: {assign_dir}")
            continue

        assignments, gray_mask = load_assignment(assign_dir)

        if mode in ('full', 'all'):
            row = run_cluster_full(
                train, test, assignments, gray_mask, n_items, label,
                max_workers=cluster_workers,
            )
            row['dataset'] = dataset_name
            results.append(row)

        if mode in ('sharedV', 'all'):
            row = run_cluster_sharedV(
                train, test, assignments, gray_mask, n_items, label,
                max_workers=cluster_workers,
            )
            row['dataset'] = dataset_name
            results.append(row)

    # Kaydet ve özet yazdır
    # Not: k=14 ve k=70 gibi farklı koşuların birbirinin üstüne yazmaması için
    # çıktıyı K ve mode bazında ayrı dosyaya yazıyoruz.
    save_results(results, out_dir, f'wnmf_results_{dataset_name}_k{k_used}_{mode}.csv')
    _print_summary(results, dataset_name)

    return results


# ============================================================
# ÖZET TABLOSU
# ============================================================

def _print_summary(results, dataset_name):
    print(f"\n{'='*60}")
    print(f"ÖZET — {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"{'Algoritma':<20} {'Senaryo':<16} {'MAE':>8} {'RMSE':>8} {'GS MAE':>8}")
    print("-" * 62)

    for r in results:
        gs = r.get('gray_mae', float('nan'))
        gs_str = f"{gs:.4f}" if not (isinstance(gs, float) and np.isnan(gs)) else "  —  "
        print(
            f"{r['algo_label']:<20} "
            f"{r['scenario']:<16} "
            f"{r['mae']:>8.4f} "
            f"{r['rmse']:>8.4f} "
            f"{gs_str:>8}"
        )
    print("=" * 62)


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="WNMF karşılaştırma deneyi")
    p.add_argument('--dataset', choices=['100k', '1m', 'both'], default='both')
    p.add_argument('--algo', nargs='+', choices=ALGO_LABELS, default=None)
    p.add_argument('--no-global', action='store_true')
    p.add_argument(
        '--mode', choices=['sharedV', 'full', 'all'], default='sharedV',
        help="sharedV=önerilen, full=eski versiyon, all=ikisi birden"
    )
    p.add_argument('--epochs-global',  type=int, default=N_EPOCHS_GLOBAL)
    p.add_argument('--epochs-cluster', type=int, default=N_EPOCHS_CLUSTER)
    p.add_argument('--latent-dim',     type=int, default=LATENT_DIM)
    p.add_argument(
        '--jobs', type=int, default=None,
        help='Küme eğitiminde paralel süreç sayısı; verilmezse CPU çekirdek sayısı',
    )
    p.add_argument(
        '--k', type=int, default=None,
        dest='assignment_k',
        help='Assignment üretimindeki K; klasör {algo}_kK olur (100K varsayılan K=90, '
             '1M varsayılan K=150 ile aynıysa ek yok). --k-100k / --k-1m ile ezilebilir.',
    )
    p.add_argument(
        '--k-100k', type=int, default=None,
        dest='assignment_k_100k',
        help='Sadece ML-100K için assignment K (--k üzerine yazar)',
    )
    p.add_argument(
        '--k-1m', type=int, default=None,
        dest='assignment_k_1m',
        help='Sadece ML-1M için assignment K (--k üzerine yazar)',
    )
    p.add_argument(
        '--assign-root', type=str, default=None,
        help='Assignment kök dizini (varsayılan: mealpy/results/assignments_lof)',
    )
    return p.parse_args()


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    args             = parse_args()
    N_EPOCHS_GLOBAL  = args.epochs_global
    N_EPOCHS_CLUSTER = args.epochs_cluster
    LATENT_DIM       = args.latent_dim

    os.makedirs(OUT_ROOT, exist_ok=True)

    print("=" * 60)
    print("WNMF DENEY")
    print("=" * 60)
    print(f"Dataset       : {args.dataset}")
    print(f"Mod           : {args.mode}")
    print(f"Algoritmalar  : {args.algo or ALGO_LABELS}")
    print(f"Latent dim    : {LATENT_DIM}")
    print(f"Epoch global  : {N_EPOCHS_GLOBAL}")
    print(f"Epoch cluster : {N_EPOCHS_CLUSTER}")
    print(f"Küme işçileri : {args.jobs or 'otomatik (CPU)'}")
    print(f"Assignment K  : --k={args.assignment_k}  --k-100k={args.assignment_k_100k}  "
          f"--k-1m={args.assignment_k_1m}")
    if args.assign_root:
        print(f"Assignment kök: {args.assign_root}")
    print(f"LR            : {LEARNING_RATE}  |  Reg: {REGULARIZATION}")
    print("=" * 60)

    t_total  = time.time()
    all_rows = []
    all_tags = []

    if args.dataset in ('100k', 'both'):
        train, test = load_ratings_100k(DATA_100K_TRAIN, DATA_100K_TEST)
        k_used_100k = _resolved_assignment_k(
            args.assignment_k_100k,
            args.assignment_k,
            ASSIGN_K_DEFAULT_100K,
        )
        all_tags.append(f"ml100k_k{k_used_100k}")
        rows = run_dataset(
            'ml100k', train, test,
            algo_filter=args.algo,
            run_global=not args.no_global,
            mode=args.mode,
            cluster_workers=args.jobs,
            assign_root=args.assign_root,
            assignment_k=args.assignment_k,
            assignment_k_100k=args.assignment_k_100k,
            assignment_k_1m=args.assignment_k_1m,
        )
        all_rows.extend(rows)

    if args.dataset in ('1m', 'both'):
        train, test = load_ratings_1m(DATA_1M, random_seed=RANDOM_SEED)
        k_used_1m = _resolved_assignment_k(
            args.assignment_k_1m,
            args.assignment_k,
            ASSIGN_K_DEFAULT_1M,
        )
        all_tags.append(f"ml1m_k{k_used_1m}")
        rows = run_dataset(
            'ml1m', train, test,
            algo_filter=args.algo,
            run_global=not args.no_global,
            mode=args.mode,
            cluster_workers=args.jobs,
            assign_root=args.assign_root,
            assignment_k=args.assignment_k,
            assignment_k_100k=args.assignment_k_100k,
            assignment_k_1m=args.assignment_k_1m,
        )
        all_rows.extend(rows)

    if all_rows:
        mode_tag = f"mode-{args.mode}"
        fname = f"wnmf_all_results_{'__'.join(all_tags + [mode_tag])}.csv" if all_tags else f"wnmf_all_results_{mode_tag}.csv"
        save_results(all_rows, OUT_ROOT, fname)

    print(f"\n{'='*60}")
    print(f"TAMAMLANDI — {(time.time()-t_total)/60:.1f} dakika")
    print(f"Çıktı: {OUT_ROOT}/")
    print("=" * 60)