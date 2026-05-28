"""5-fold eval CSV'lerinden MAE/RMSE/NDCG ortalaması (train-only veya full)."""
import argparse
import glob
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-only', action='store_true')
    p.add_argument('--k', type=int, default=7)
    args = p.parse_args()

    pattern = os.path.join(
        REPO, 'results', 'wnmf', 'ml100k', f'k{args.k}', 'fold*', 'run*',
        f'wnmf_results_ml100k_k{args.k}_baselines.csv',
    )
    rows = []
    latest_by_fold = {}
    for path in sorted(glob.glob(pattern)):
        header = open(path, encoding='utf-8', errors='ignore').readline()
        is_trainonly = 'trainonly' in header
        if args.train_only and not is_trainonly:
            continue
        if not args.train_only and is_trainonly:
            continue
        df = pd.read_csv(path, comment='#')
        if df.empty:
            continue
        fold_part = path.replace('\\', '/').split('/fold')[1].split('/')[0]
        fold = int(fold_part)
        run_part = path.replace('\\', '/').split('/run')[1].split('/')[0]
        run_n = int(run_part)
        prev = latest_by_fold.get(fold)
        if prev is not None and prev['run_n'] >= run_n:
            continue
        r = df.iloc[0]
        latest_by_fold[fold] = {
            'fold': fold,
            'run_n': run_n,
            'mae': float(r['mae']),
            'rmse': float(r['rmse']),
            'ndcg': float(r.get('ndcg_at_10', np.nan)),
            'path': path,
        }

    rows = list(latest_by_fold.values())

    label = 'Train-only 5-fold' if args.train_only else '5-fold'
    if not rows:
        print(f'{label} ozet: uygun sonuc bulunamadi')
        return

    d = pd.DataFrame(rows).sort_values('fold')
    print(f'=== {label} ozet ===')
    for _, r in d.iterrows():
        print(
            f"  fold {int(r['fold'])}  "
            f"MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  NDCG={r['ndcg']:.4f}"
        )
    print(
        f"  MEAN     MAE={d['mae'].mean():.4f}  "
        f"RMSE={d['rmse'].mean():.4f}  NDCG={d['ndcg'].mean():.4f}"
    )


if __name__ == '__main__':
    main()
