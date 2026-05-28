"""K=20 kNN sweep sonuclarini tum metriklerle karsilastir."""
import argparse
import glob
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METRIC_COLS = [
    'mae', 'rmse', 'gray_mae', 'gray_rmse', 'white_mae', 'white_rmse',
    'accuracy', 'precision_at_10', 'recall_at_10', 'f1_at_10', 'ndcg_at_10',
    'time_seconds',
]


def _latest_run_csv(k: int, fold: int, suffix_hint: str = '', min_rows: int = 1) -> str | None:
    pattern = os.path.join(
        REPO, 'results', 'wnmf', 'ml100k', f'k{k}', f'fold{fold}', 'run*',
        f'wnmf_results_ml100k_k{k}_baselines.csv',
    )
    best = None
    for path in sorted(glob.glob(pattern)):
        header = open(path, encoding='utf-8', errors='ignore').readline()
        if suffix_hint and suffix_hint not in header:
            continue
        try:
            df = pd.read_csv(path, comment='#')
        except pd.errors.EmptyDataError:
            continue
        if len(df) < min_rows:
            continue
        run_n = int(path.replace('\\', '/').split('/run')[1].split('/')[0])
        score = (len(df), run_n)
        if best is None or score > best[0]:
            best = (score, path)
    return best[1] if best else None


def _load_knn_rows(path: str, knn: int) -> pd.DataFrame:
    df = pd.read_csv(path, comment='#')
    if df.empty:
        return df
    if 'k_neighbors' in df.columns:
        df = df[df['k_neighbors'] == knn]
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=int, default=20)
    p.add_argument('--fold', type=int, default=5)
    p.add_argument('--knn', type=int, default=60)
    args = p.parse_args()

    wnmf_path = _latest_run_csv(args.k, args.fold, 'wnmf30_k20', min_rows=5)
    pca_path = _latest_run_csv(args.k, args.fold, 'pca95pct')

    frames = []
    if wnmf_path:
        df = _load_knn_rows(wnmf_path, args.knn)
        df = df.copy()
        df['feature'] = 'wnmf30'
        frames.append(df)
        print(f'WNMF30 CSV: {wnmf_path}')
    else:
        print('WNMF30 sonuc bulunamadi')

    if pca_path:
        try:
            df = _load_knn_rows(pca_path, args.knn)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        if not df.empty:
            df = df.copy()
            df['feature'] = 'pca95'
            if 'algo_label' in df.columns:
                df['algo_label'] = df['algo_label'].astype(str) + '_PCA'
            frames.append(df)
            print(f'PCA95 CSV: {pca_path}')
        else:
            print(f'PCA95 CSV bos: {pca_path}')
    else:
        print('PCA95 sonuc bulunamadi (henuz kosulmamis olabilir)')

    if not frames:
        print('Karsilastirma icin veri yok.')
        return

    all_df = pd.concat(frames, ignore_index=True)
    if 'scenario' in all_df.columns:
        all_df = all_df[all_df['scenario'].isin(['cluster_knn', 'global_svd', 'global_wnmf'])]

    sort_col = 'mae' if 'mae' in all_df.columns else all_df.columns[0]
    cols = [c for c in ['algo_label', 'feature', 'scenario'] + METRIC_COLS if c in all_df.columns]
    out = all_df[cols].sort_values(sort_col)

    print(f'\n=== K={args.k} fold={args.fold} kNN={args.knn} — tum metrikler ===')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(out.to_string(index=False))

    out_path = os.path.join(
        REPO, 'results', 'wnmf', 'ml100k', f'k{args.k}',
        f'comparison_k{args.k}_fold{args.fold}_knn{args.knn}.csv',
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f'\nKaydedildi: {out_path}')


if __name__ == '__main__':
    main()
