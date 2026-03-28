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
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from wnmf_model import WNMFModel, WNMFSharedV
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

LATENT_DIM       = 20
LEARNING_RATE    = 0.01
REGULARIZATION   = 0.01
N_EPOCHS_GLOBAL  = 100   # global V ve global baseline için
N_EPOCHS_CLUSTER = 50    # küme U eğitimi için (daha az veri, daha az epoch yeterli)
RANDOM_SEED      = 42

ALGO_LABELS = ['B1_HHO', 'B2_HGS', 'H1_HHO+HGS', 'H4_MFO+HHO']


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
                     n_items, algo_label, verbose=False):
    """
    Her küme bağımsız U ve V öğrenir.
    Karşılaştırma için tutulur — SharedV ile farkı görmek için.
    """
    print(f"\n  [{algo_label} | Full Cluster] başlıyor...")
    t0 = time.time()

    cluster_train, gray_train = split_by_cluster(train, assignments, gray_mask)
    cluster_test,  gray_test  = split_by_cluster(test,  assignments, gray_mask)

    all_true, all_pred = [], []

    for cid, c_train in cluster_train.items():
        c_test = cluster_test.get(cid, np.empty((0, 3), dtype=np.float32))
        if len(c_train) < 5 or len(c_test) == 0:
            continue

        c_train_r, c_test_r, _, n_loc = remap_user_ids(c_train, c_test, n_items)

        model = WNMFModel(
            n_users        = n_loc,
            n_items        = n_items,
            latent_dim     = LATENT_DIM,
            learning_rate  = LEARNING_RATE,
            regularization = REGULARIZATION,
            n_epochs       = N_EPOCHS_CLUSTER,
            random_seed    = RANDOM_SEED,
        )
        model.fit(c_train_r)

        if len(c_test_r) > 0:
            pred = model.predict(
                c_test_r[:, 0].astype(np.int32),
                c_test_r[:, 1].astype(np.int32),
            )
            all_true.extend(c_test_r[:, 2].tolist())
            all_pred.extend(pred.tolist())

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
                        n_items, algo_label, verbose=False):
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

    # Kümeleri böl
    cluster_train, gray_train = split_by_cluster(train, assignments, gray_mask)
    cluster_test,  gray_test  = split_by_cluster(test,  assignments, gray_mask)

    # Aşama 2: Her küme için U öğren
    all_true, all_pred = [], []
    n_clusters_fit     = 0

    for cid, c_train in cluster_train.items():
        c_test = cluster_test.get(cid, np.empty((0, 3), dtype=np.float32))
        if len(c_train) < 5 or len(c_test) == 0:
            continue

        c_train_r, c_test_r, _, n_loc = remap_user_ids(c_train, c_test, n_items)

        # V paylaşılıyor, U sıfırdan öğreniliyor
        cluster_model = shared.make_cluster_model(
            n_users_cluster  = n_loc,
            n_epochs_cluster = N_EPOCHS_CLUSTER,
            random_seed      = RANDOM_SEED + cid,
        )
        cluster_model.fit_cluster_U(c_train_r)

        if len(c_test_r) > 0:
            pred = cluster_model.predict(
                c_test_r[:, 0].astype(np.int32),
                c_test_r[:, 1].astype(np.int32),
            )
            all_true.extend(c_test_r[:, 2].tolist())
            all_pred.extend(pred.tolist())
            n_clusters_fit += 1

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
                run_global=True, mode='sharedV'):
    """
    Bir dataset üzerinde tüm senaryoları çalıştır.

    mode parametresi:
        'sharedV' — sadece SharedV (önerilen, hızlı)
        'full'    — sadece her küme U+V (eski versiyon)
        'all'     — her ikisi de (karşılaştırma için)
    """
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name.upper()}  |  mode={mode}")
    print(f"{'='*60}")

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

        assign_dir = os.path.join(ASSIGN_ROOT, dataset_name, label)
        if not os.path.exists(assign_dir):
            print(f"\n  [{label}] ATLANDI — assignment bulunamadı: {assign_dir}")
            continue

        assignments, gray_mask = load_assignment(assign_dir)

        if mode in ('full', 'all'):
            row = run_cluster_full(
                train, test, assignments, gray_mask, n_items, label
            )
            row['dataset'] = dataset_name
            results.append(row)

        if mode in ('sharedV', 'all'):
            row = run_cluster_sharedV(
                train, test, assignments, gray_mask, n_items, label
            )
            row['dataset'] = dataset_name
            results.append(row)

    # Kaydet ve özet yazdır
    save_results(results, out_dir, f'wnmf_results_{dataset_name}.csv')
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
    print(f"LR            : {LEARNING_RATE}  |  Reg: {REGULARIZATION}")
    print("=" * 60)

    t_total  = time.time()
    all_rows = []

    if args.dataset in ('100k', 'both'):
        train, test = load_ratings_100k(DATA_100K_TRAIN, DATA_100K_TEST)
        rows = run_dataset(
            'ml100k', train, test,
            algo_filter=args.algo,
            run_global=not args.no_global,
            mode=args.mode,
        )
        all_rows.extend(rows)

    if args.dataset in ('1m', 'both'):
        train, test = load_ratings_1m(DATA_1M, random_seed=RANDOM_SEED)
        rows = run_dataset(
            'ml1m', train, test,
            algo_filter=args.algo,
            run_global=not args.no_global,
            mode=args.mode,
        )
        all_rows.extend(rows)

    if all_rows:
        save_results(all_rows, OUT_ROOT, 'wnmf_all_results.csv')

    print(f"\n{'='*60}")
    print(f"TAMAMLANDI — {(time.time()-t_total)/60:.1f} dakika")
    print(f"Çıktı: {OUT_ROOT}/")
    print("=" * 60)