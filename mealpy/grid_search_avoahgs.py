"""
grid_search_avoahgs.py
======================
HA_AVOAHGS için 2-parametre grid search.
Her kombinasyon için hem WCSS hem de ClusterKNN MAE/RMSE ölçülür.

Arama uzayı (toplam 100 kombinasyon):
    p1       ∈ {0.1, 0.2, ..., 0.9, 0.95}   (AVOA faz-1 olasılığı)
    hgs_rate ∈ {0.1, 0.2, ..., 0.9, 0.95}   (HGS uygulanan en iyi birey oranı)
    → 10 × 10 = 100 kombinasyon, her biri N_REPEATS tekrar

Kullanım:
    python mealpy\\grid_search_avoahgs.py --dataset 100k --k 6
    python mealpy\\grid_search_avoahgs.py --dataset 100k --k 7 --zscore --feature-extraction none
    # Sadece WCSS (KNN atla):
    python mealpy\\grid_search_avoahgs.py --dataset 100k --k 6 --no-knn
"""

import argparse
import itertools
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(BASE_DIR)
OPT_DIR    = os.path.join(REPO_ROOT, 'optimizers')
WNMF_DIR   = os.path.join(REPO_ROOT, 'wnmf')
for p in (REPO_ROOT, BASE_DIR, OPT_DIR, WNMF_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from generate_assignments import (
    DATA_100K,
    DATA_1M,
    BASELINE_EPOCH,
    POP_SIZE,
    _make_problem,
    _multi_start_init,
    load_movielens_1m,
    prune_sparse_matrix,
    zscore_normalize,
)
from mealpy_comparison_v2 import compute_wcss_fast, load_movielens
from optimizers.HA_AVOAHGS import HA_AVOAHGS
from wnmf_utils import load_ratings_100k
from wnmf_experiment import run_cluster_knn

# ============================================================
# GRID TANIMLARI
# ============================================================
# 10 × 10 = 100 kombinasyon; N_REPEATS=1 → toplam 100 çalışma
# N_REPEATS artırılırsa her kombo tekrarlanır (daha güvenilir ama yavaş)

P1_VALUES       = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
HGS_RATE_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
N_REPEATS       = 1

# Sabit HA_AVOAHGS parametreleri (arama dışı)
P2 = 0.4; P3 = 0.6; ALPHA = 0.8; GAMA = 2.5; PUP = 0.08; LH = 10000

# ClusterKNN parametreleri
KNN_K          = 20
KNN_SIM        = 'pearson'
KNN_MIN_COMMON = 3

DATA_TRAIN = os.path.join(REPO_ROOT, 'data', 'ml-100k', 'u1.base')
DATA_TEST  = os.path.join(REPO_ROOT, 'data', 'ml-100k', 'u1.test')
N_ITEMS_100K = 1682
N_ITEMS_1M   = 3952

# ============================================================
# YARDIMCILAR
# ============================================================

def prepare_matrix(raw, zscore=False, preprocess='none',
                   feature_extraction='none', svd_components=50,
                   min_user_ratings=0, min_item_ratings=0):
    from sklearn.preprocessing import normalize
    matrix = prune_sparse_matrix(raw, min_user_ratings, min_item_ratings)
    if zscore:
        matrix = zscore_normalize(matrix)
        matrix = normalize(matrix, norm='l2').astype(np.float32)
    if preprocess == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        matrix = MinMaxScaler().fit_transform(matrix)
    elif preprocess == 'zscore':
        from sklearn.preprocessing import StandardScaler
        matrix = StandardScaler().fit_transform(matrix)
    elif preprocess == 'maxabs':
        from sklearn.preprocessing import MaxAbsScaler
        matrix = MaxAbsScaler().fit_transform(matrix)
    if feature_extraction == 'svd':
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=svd_components, random_state=42)
        matrix = normalize(svd.fit_transform(matrix)).astype(np.float32)
    elif feature_extraction == 'nmf':
        from sklearn.decomposition import NMF
        nmf = NMF(n_components=svd_components, random_state=42,
                  max_iter=1000, init='nndsvda')
        matrix = normalize(nmf.fit_transform(np.maximum(matrix, 0))).astype(np.float32)
    return matrix.astype(np.float32)


def run_avoahgs(matrix, K, p1, hgs_rate, seed,
                epoch=BASELINE_EPOCH, pop_size=POP_SIZE,
                metric='euclidean', init_mode='random'):
    """Tek HA_AVOAHGS çalışması → (assignments, wcss, elapsed_s)."""
    t0 = time.time()
    init = _multi_start_init(
        matrix, K=K, pop_size=pop_size, seed=seed, n_restarts=10,
        metric=metric, init_mode=init_mode,
    )
    problem = _make_problem(matrix, K, metric=metric)
    model = HA_AVOAHGS(
        epoch=epoch, pop_size=pop_size,
        p1=p1, p2=P2, p3=P3, alpha=ALPHA, gama=GAMA,
        PUP=PUP, LH=LH, hgs_rate=hgs_rate,
    )
    try:
        model.solve(problem, starting_solutions=init[:pop_size])
    except TypeError:
        model.solve(problem)
    best_sol = model.g_best.solution
    wcss, assignments = compute_wcss_fast(matrix, best_sol, K, metric=metric)
    assignments = assignments.astype(np.int32)
    return assignments, float(wcss), time.time() - t0


# ============================================================
# GRID SEARCH
# ============================================================

def run_grid(cluster_matrix, train, test, n_items,
             K, epoch=BASELINE_EPOCH, pop_size=POP_SIZE,
             metric='euclidean', init_mode='random',
             run_knn: bool = True,
             out_csv: Optional[str] = None):

    grid = list(itertools.product(P1_VALUES, HGS_RATE_VALUES))
    total_runs = len(grid) * N_REPEATS
    n_users = int(train[:, 0].max()) + 1

    print("=" * 70)
    print("HA_AVOAHGS GRID SEARCH  (WCSS + ClusterKNN MAE/RMSE)")
    print(f"  Kümeleme matrisi : {cluster_matrix.shape}")
    print(f"  Train/Test       : {len(train)} / {len(test)} rating")
    print(f"  K                : {K}   Epoch: {epoch}   Pop: {pop_size}")
    print(f"  Metrik           : {metric}   Init: {init_mode}")
    print(f"  p1 değerleri     : {P1_VALUES}")
    print(f"  hgs_rate değerl. : {HGS_RATE_VALUES}")
    print(f"  Tekrar/kombo     : {N_REPEATS}")
    print(f"  Toplam çalışma   : {total_runs}")
    print(f"  ClusterKNN       : {'açık (k=' + str(KNN_K) + ', sim=' + KNN_SIM + ')' if run_knn else 'kapalı (--no-knn)'}")
    print("=" * 70)

    rows = []
    run_no = 0
    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    for combo_no, (p1, hgs_rate) in enumerate(grid, 1):
        print(f"\n[Kombo {combo_no}/{len(grid)}] p1={p1:.2f}  hgs_rate={hgs_rate:.2f}")

        for rep in range(N_REPEATS):
            run_no += 1
            seed = 1000 * int(round(p1 * 100)) + int(round(hgs_rate * 100)) + rep

            # ── Adım 1: Kümeleme ──────────────────────────────────────────
            assignments, wcss, t_cluster = run_avoahgs(
                cluster_matrix, K, p1=p1, hgs_rate=hgs_rate,
                seed=seed, epoch=epoch, pop_size=pop_size,
                metric=metric, init_mode=init_mode,
            )

            mae = float('nan')
            rmse = float('nan')
            gray_mae = float('nan')
            t_knn = 0.0

            # ── Adım 2: ClusterKNN değerlendirmesi ────────────────────────
            if run_knn:
                gray_mask = np.zeros(n_users, dtype=bool)  # gray sheep yok
                memberships = None

                t_knn_start = time.time()
                result = run_cluster_knn(
                    train, test,
                    assignments, gray_mask, memberships,
                    n_items,
                    algo_label=f'p1={p1:.2f}_hgs={hgs_rate:.2f}',
                    similarity=KNN_SIM,
                    min_common=KNN_MIN_COMMON,
                    k_neighbors=KNN_K,
                    knn_mode='cluster',
                )
                t_knn = time.time() - t_knn_start
                mae      = result['mae']
                rmse     = result['rmse']
                gray_mae = result.get('gray_mae', float('nan'))

            row = {
                'p1': p1, 'hgs_rate': hgs_rate,
                'rep': rep, 'seed': seed,
                'wcss': round(wcss, 4),
                'mae': round(mae, 6) if not np.isnan(mae) else mae,
                'rmse': round(rmse, 6) if not np.isnan(rmse) else rmse,
                'gray_mae': round(gray_mae, 6) if not np.isnan(gray_mae) else gray_mae,
                't_cluster_s': round(t_cluster, 2),
                't_knn_s': round(t_knn, 2),
            }
            rows.append(row)

            print(
                f"  Rep {rep+1}/{N_REPEATS} | "
                f"WCSS={wcss:>10.2f} | MAE={mae:.4f} RMSE={rmse:.4f} | "
                f"cluster={t_cluster:.1f}s knn={t_knn:.1f}s"
                f"  [{run_no}/{total_runs}]"
            )

            # Incremental kayıt — yarıda kesilse veri korunur
            if out_csv:
                write_header = (run_no == 1)
                pd.DataFrame([row]).to_csv(
                    out_csv, mode='a', header=write_header, index=False,
                )

    # ── Özet ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)

    agg_cols = {'wcss': ['mean', 'min'], 'mae': ['mean', 'min'], 'rmse': ['mean', 'min']}
    summary = (
        df.groupby(['p1', 'hgs_rate'])
        .agg(
            wcss_mean=('wcss', 'mean'),
            wcss_best=('wcss', 'min'),
            mae_mean=('mae', 'mean'),
            mae_best=('mae', 'min'),
            rmse_mean=('rmse', 'mean'),
            rmse_best=('rmse', 'min'),
        )
        .reset_index()
        .sort_values('mae_mean')
    )

    print("\n" + "=" * 70)
    print("ÖZET — MAE'ye göre sıralı (düşük = daha iyi)")
    print("=" * 70)
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.max_rows', 110)
    pd.set_option('display.width', 130)
    print(summary.to_string(index=False))

    best = summary.iloc[0]
    print(f"\n★ EN İYİ KOMBİNASYON (MAE):")
    print(f"   p1={best['p1']:.2f}  hgs_rate={best['hgs_rate']:.2f}")
    print(f"   MAE={best['mae_mean']:.4f}  RMSE={best['rmse_mean']:.4f}  WCSS={best['wcss_mean']:.2f}")

    best_wcss = summary.sort_values('wcss_mean').iloc[0]
    print(f"\n★ EN İYİ KOMBİNASYON (WCSS):")
    print(f"   p1={best_wcss['p1']:.2f}  hgs_rate={best_wcss['hgs_rate']:.2f}")
    print(f"   WCSS={best_wcss['wcss_mean']:.2f}  MAE={best_wcss['mae_mean']:.4f}")

    total_elapsed = sum(r['t_cluster_s'] + r['t_knn_s'] for r in rows)
    print(f"\nToplam süre: {total_elapsed/60:.1f} dakika")

    if out_csv:
        summary_csv = out_csv.replace('.csv', '_summary.csv')
        summary.to_csv(summary_csv, index=False)
        print(f"\nHam sonuçlar: {out_csv}")
        print(f"Özet        : {summary_csv}")

    return summary, df


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='HA_AVOAHGS grid search — p1 × hgs_rate, WCSS + ClusterKNN MAE/RMSE'
    )
    p.add_argument('--dataset',   choices=['100k', '1m'], default='100k')
    p.add_argument('--k',         type=int, default=6)
    p.add_argument('--fold',      type=int, default=1,
                   help='ML-100K holdout fold (1-5, default: 1)')
    p.add_argument('--epoch',     type=int, default=BASELINE_EPOCH)
    p.add_argument('--pop-size',  type=int, default=POP_SIZE)
    p.add_argument('--cluster-metric', choices=['euclidean', 'pearson'], default='euclidean')
    p.add_argument('--init-mode', choices=['random', 'mkpp'], default='random')
    p.add_argument('--zscore',    action='store_true')
    p.add_argument('--preprocess', choices=['none','minmax','zscore','maxabs'], default='none')
    p.add_argument('--feature-extraction', choices=['none','svd','nmf'], default='none')
    p.add_argument('--svd-components', type=int, default=50)
    p.add_argument('--no-prune',  action='store_true')
    p.add_argument('--min-user-ratings', type=int, default=0)
    p.add_argument('--min-item-ratings', type=int, default=0)
    p.add_argument('--knn-k',     type=int, default=KNN_K,
                   help=f'ClusterKNN komşu sayısı (default: {KNN_K})')
    p.add_argument('--knn-sim',   choices=['pearson','cosine','pearson_iuf'],
                   default=KNN_SIM)
    p.add_argument('--no-knn',    action='store_true',
                   help='ClusterKNN değerlendirmesini atla (sadece WCSS)')
    p.add_argument('--n-repeats', type=int, default=N_REPEATS,
                   help=f'Tekrar/kombo (default: {N_REPEATS})')
    p.add_argument('--p1-values', nargs='+', type=float, default=None)
    p.add_argument('--hgs-rate-values', nargs='+', type=float, default=None)
    p.add_argument('--data-100k', default=DATA_100K)
    p.add_argument('--data-1m',   default=DATA_1M)
    p.add_argument('--out-csv',   type=str, default=None)
    return p.parse_args()


def main():
    global P1_VALUES, HGS_RATE_VALUES, N_REPEATS, KNN_K, KNN_SIM
    args = parse_args()

    if args.p1_values:       P1_VALUES       = args.p1_values
    if args.hgs_rate_values: HGS_RATE_VALUES = args.hgs_rate_values
    N_REPEATS = args.n_repeats
    KNN_K     = args.knn_k
    KNN_SIM   = args.knn_sim

    if args.no_prune:
        args.min_user_ratings = 0
        args.min_item_ratings = 0

    # ── Veri yükle ────────────────────────────────────────────────────────
    if args.dataset == '100k':
        print(f"ML-100K yükleniyor (fold={args.fold})...")
        raw = load_movielens(args.data_100k)
        train_path = os.path.join(REPO_ROOT, 'data', 'ml-100k', f'u{args.fold}.base')
        test_path  = os.path.join(REPO_ROOT, 'data', 'ml-100k', f'u{args.fold}.test')
        train, test = load_ratings_100k(train_path, test_path)
        n_items = N_ITEMS_100K
    else:
        print(f"ML-1M yükleniyor...")
        raw = load_movielens_1m(args.data_1m)
        # ML-1M için basit %80/%20 bölme
        from sklearn.model_selection import train_test_split
        ratings = np.array(
            [(u, i, r) for u, items in enumerate(raw) for i, r in enumerate(items) if r > 0],
            dtype=np.float32
        )
        train, test = train_test_split(ratings, test_size=0.2, random_state=42)
        n_items = N_ITEMS_1M

    # ── Kümeleme matrisi ──────────────────────────────────────────────────
    cluster_matrix = prepare_matrix(
        raw,
        zscore=args.zscore,
        preprocess=args.preprocess,
        feature_extraction=args.feature_extraction,
        svd_components=args.svd_components,
        min_user_ratings=args.min_user_ratings,
        min_item_ratings=args.min_item_ratings,
    )

    # ── Çıktı yolu ────────────────────────────────────────────────────────
    out_csv = args.out_csv
    if out_csv is None:
        fe_tag = (
            f'_{args.feature_extraction}{args.svd_components}'
            if args.feature_extraction != 'none' else ''
        )
        knn_tag = f'_knn{KNN_K}' if not args.no_knn else '_wcssonly'
        out_csv = os.path.join(
            BASE_DIR, 'results', 'grid',
            f'avoahgs_grid_{args.dataset}_k{args.k}{fe_tag}{knn_tag}.csv',
        )

    run_grid(
        cluster_matrix, train, test, n_items,
        K=args.k,
        epoch=args.epoch, pop_size=args.pop_size,
        metric=args.cluster_metric, init_mode=args.init_mode,
        run_knn=not args.no_knn,
        out_csv=out_csv,
    )


if __name__ == '__main__':
    main()
