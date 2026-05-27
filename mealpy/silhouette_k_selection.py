"""Silhouette + MAE ile optimal K seçimi (WNMF30 champion pipeline).

Silhouette: silhouette_score(wnmf_W_matrix, labels) — assignment klasöründeki
user_features.npy (WNMF U, L2 normalize) üzerinde; yoksa pipeline ile yeniden üretilir.

Örnek:
  python mealpy/silhouette_k_selection.py
  python mealpy/silhouette_k_selection.py --k 3 5 7 10 14 20 --sil-threshold 0.5
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'mealpy'))

from generate_assignments import (  # noqa: E402
    compute_silhouette_wnmf,
    load_movielens,
    prepare_matrix_for_clustering,
)

ASSIGN_ROOT = os.path.join(REPO, 'mealpy', 'results', 'assignments_lof', 'ml100k')
SUFFIX_TMPL = '_pruneu5_i10_zscore_euc_imkpp_none_wnmf30_k{k}'
RESULTS_GLOB = os.path.join(
    REPO, 'results', 'wnmf', 'ml100k', 'k{k}', 'fold5', 'run*',
    'wnmf_results_ml100k_k{k}_baselines.csv',
)

_W_CACHE: np.ndarray | None = None


def load_wnmf_W_matrix() -> np.ndarray:
    """Champion pipeline ile WNMF U (L2 normalize) — tüm K'lar için aynı W."""
    global _W_CACHE
    if _W_CACHE is not None:
        return _W_CACHE
    data_path = os.path.join(REPO, 'data', 'ml-100k', 'u.data')
    matrix = load_movielens(data_path)
    _W_CACHE = prepare_matrix_for_clustering(
        matrix,
        zscore=True,
        pca_var=None,
        wnmf_k=30,
        preprocess='none',
        feature_extraction='wnmf',
        svd_components=30,
    )
    return _W_CACHE


def assign_dir(algo: str, k: int) -> str:
    return os.path.join(ASSIGN_ROOT, f'{algo}{SUFFIX_TMPL.format(k=k)}')


def load_W_for_assignment(algo: str, k: int) -> np.ndarray:
    """Öncelik: kayıtlı user_features.npy (= kümelemede kullanılan W matrisi)."""
    uf_path = os.path.join(assign_dir(algo, k), 'user_features.npy')
    if os.path.isfile(uf_path):
        return np.load(uf_path)
    print(f'  [uyarı] {uf_path} yok; WNMF W yeniden üretiliyor.')
    return load_wnmf_W_matrix()


def load_labels(algo: str, k: int):
    d = assign_dir(algo, k)
    if not os.path.isdir(d):
        raise FileNotFoundError(d)
    assignments = np.load(os.path.join(d, 'assignments.npy'))
    gray = np.load(os.path.join(d, 'gray_sheep_mask.npy')).astype(bool)
    return assignments, gray


def compute_silhouette(W: np.ndarray, labels: np.ndarray, gray: np.ndarray,
                       metric: str = 'euclidean') -> float:
    return compute_silhouette_wnmf(W, labels, gray_mask=gray, metric=metric)


def kmeans_labels(W: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return km.fit_predict(W)


def best_mae_from_results(k: int, knn: int = 40) -> dict:
    pattern = RESULTS_GLOB.format(k=k)
    best: dict[str, dict] = {}
    for path in glob.glob(pattern):
        try:
            df = pd.read_csv(path, comment='#')
        except Exception:
            continue
        sub = df[
            (df.get('scenario') == 'cluster_knn')
            & (df.get('k_neighbors', df.get('knn', -1)) == knn)
        ]
        if sub.empty:
            continue
        run_n = int(path.replace('\\', '/').split('/run')[1].split('/')[0])
        for _, row in sub.iterrows():
            algo = str(row['algo_label'])
            mae = float(row['mae'])
            prev = best.get(algo)
            if prev is None or run_n >= prev['run_n']:
                best[algo] = {
                    'mae': mae,
                    'ndcg': float(row.get('ndcg_at_10', np.nan)),
                    'run_n': run_n,
                    'path': path,
                }
    if not best:
        return {'best_mae': np.nan, 'best_algo': None, 'ndcg': np.nan, 'by_algo': {}}
    by_algo = {a: v['mae'] for a, v in best.items()}
    winner = min(best.items(), key=lambda x: x[1]['mae'])
    return {
        'best_mae': winner[1]['mae'],
        'best_algo': winner[0],
        'ndcg': winner[1]['ndcg'],
        'by_algo': by_algo,
    }


def main():
    p = argparse.ArgumentParser(description='Silhouette + MAE ile K seçimi')
    p.add_argument('--k', nargs='+', type=int, default=[3, 5, 7, 10, 14, 20])
    p.add_argument('--sil-threshold', type=float, default=0.5)
    p.add_argument('--knn-mae', type=int, default=40,
                   help='MAE karşılaştırması için knn (expand OFF koşuları)')
    p.add_argument('--algo', nargs='+', default=['HA_AVOAHGS', 'IWO_HHO'])
    p.add_argument('--include-kmeans', action='store_true',
                   help='Aynı W üzerinde KMeans silhouette de yaz')
    args = p.parse_args()

    print('Silhouette: silhouette_score(wnmf_W_matrix, labels)')
    print('  W kaynağı: assignment/user_features.npy (yoksa pipeline ile üretilir)\n')

    rows = []
    W_ref = None
    for k in sorted(args.k):
        mae_info = best_mae_from_results(k, knn=args.knn_mae)
        row = {
            'K': k,
            'mae_best': mae_info['best_mae'],
            'mae_algo': mae_info['best_algo'],
            'ndcg': mae_info['ndcg'],
        }
        for algo in args.algo:
            d = assign_dir(algo, k)
            if not os.path.isdir(d):
                print(f'  [atla] K={k} {algo}: assignment yok -> {d}')
                row[f'sil_{algo}'] = np.nan
                row[f'silcos_{algo}'] = np.nan
                continue
            W = load_W_for_assignment(algo, k)
            if W_ref is None:
                W_ref = W
                print(f'  W shape: {W.shape}')
            labels, gray = load_labels(algo, k)
            row[f'sil_{algo}'] = compute_silhouette(W, labels, gray, metric='euclidean')
            row[f'silcos_{algo}'] = compute_silhouette(W, labels, gray, metric='cosine')
            row[f'gray_pct_{algo}'] = 100.0 * gray.mean()
            if algo in mae_info['by_algo']:
                row[f'mae_{algo}'] = mae_info['by_algo'][algo]

        if args.include_kmeans:
            W_km = W_ref if W_ref is not None else load_wnmf_W_matrix()
            km_labels = kmeans_labels(W_km, k)
            gray_zero = np.zeros(len(km_labels), dtype=bool)
            row['sil_kmeans'] = compute_silhouette(W_km, km_labels, gray_zero, metric='euclidean')
            row['silcos_kmeans'] = compute_silhouette(W_km, km_labels, gray_zero, metric='cosine')

        rows.append(row)

    df = pd.DataFrame(rows)

    sil_col = 'sil_HA_AVOAHGS' if 'sil_HA_AVOAHGS' in df.columns else f'sil_{args.algo[0]}'
    silcos_col = sil_col.replace('sil_', 'silcos_')

    print('=' * 88)
    print('Silhouette (white users, WNMF W) + MAE (expand OFF, knn=%d, fold 5)'
          % args.knn_mae)
    print('=' * 88)
    cols = ['K']
    for algo in args.algo:
        cols.extend([f'sil_{algo}', f'silcos_{algo}', f'mae_{algo}'])
    if args.include_kmeans:
        cols.extend(['sil_kmeans', 'silcos_kmeans'])
    cols.extend(['mae_best', 'ndcg'])
    present = [c for c in cols if c in df.columns]
    print(df[present].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print()
    print('Not: ML-100K WNMF30 uzayinda mutlak silhouette genelde << 0.5 (yumusak kumeler).')
    print('     Goreceli secim: en yuksek silhouette veya esik yerine top-q yuzdelik.')

    ok = df[df[sil_col] > args.sil_threshold].copy()
    if ok.empty:
        print(f'Silhouette > {args.sil_threshold} saglayan K yok ({sil_col}).')
        rel = df.loc[df[sil_col].idxmax()]
        print(f'Goreceli max silhouette (euclidean): K={int(rel["K"])} sil={rel[sil_col]:.4f} '
              f'MAE={rel["mae_best"]:.4f}')
        if silcos_col in df.columns:
            rel_c = df.loc[df[silcos_col].idxmax()]
            print(f'Goreceli max silhouette (cosine):    K={int(rel_c["K"])} sil={rel_c[silcos_col]:.4f} '
                  f'MAE={rel_c["mae_best"]:.4f}')
    else:
        k_star = int(ok.iloc[0]['K'])
        print(f'Silhouette > {args.sil_threshold} -> en kucuk K = {k_star} '
              f'(sil={ok.iloc[0][sil_col]:.3f}, MAE={ok.iloc[0]["mae_best"]:.4f})')

    valid_mae = df.dropna(subset=['mae_best'])
    if not valid_mae.empty:
        best_mae_row = valid_mae.loc[valid_mae['mae_best'].idxmin()]
        sil_v = best_mae_row.get(sil_col, float('nan'))
        print(f'En dusuk MAE -> K={int(best_mae_row["K"])} '
              f'(MAE={best_mae_row["mae_best"]:.4f}, sil={sil_v:.4f})')

    if not ok.empty:
        combo = ok.loc[ok['mae_best'].idxmin()]
        print(f'Oneri (sil>{args.sil_threshold} + min MAE) -> K={int(combo["K"])} '
              f'(sil={combo[sil_col]:.3f}, MAE={combo["mae_best"]:.4f}, NDCG={combo["ndcg"]:.4f})')

    out = os.path.join(REPO, 'results', 'silhouette_k_selection.csv')
    df.to_csv(out, index=False)
    print(f'\nKaydedildi: {out}')


if __name__ == '__main__':
    main()
