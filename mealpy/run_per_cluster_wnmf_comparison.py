"""
run_per_cluster_wnmf_comparison.py
===================================
Belirli bir assignment ile tüm meta-algoritmaları per-cluster WNMF parametre
optimizasyonu için karşılaştırır.

Her algoritma K küme için (latent_dim, lr, reg, epochs) parametrelerini optimize eder.
Sonuçlar validation MAE ve test MAE olarak karşılaştırılır. Baseline (uniform params)
ile kıyaslama yapılır.

Kullanım:
    python run_per_cluster_wnmf_comparison.py \\
        --dataset 100k \\
        --assign-dir mealpy/results/assignments_lof/ml100k/B0_KMEANS_pruneu5_i10_zscore_k7 \\
        --epochs 30 --pop 30 \\
        --algo B1_HHO B2_HGS H4_MFO+HHO B3_MFO SFOA

    # Paralel algo çalıştırma:
    python run_per_cluster_wnmf_comparison.py \\
        --dataset 100k \\
        --assign-dir ... \\
        --algo B1_HHO B2_HGS H4_MFO+HHO \\
        --algo-jobs 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

_MEALPY_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_MEALPY_DIR)
_WNMF_DIR = os.path.join(_REPO_ROOT, 'wnmf')
if _MEALPY_DIR not in sys.path:
    sys.path.insert(0, _MEALPY_DIR)
if _WNMF_DIR not in sys.path:
    sys.path.insert(0, _WNMF_DIR)

from per_cluster_wnmf_optimizer import (
    PerClusterWNMFFitness,
    run_per_cluster_optimization,
    PC_VAL_SAMPLE,
    PC_LATENT_MIN, PC_LATENT_MAX,
    PC_LR_MIN, PC_LR_MAX,
    PC_REG_MIN, PC_REG_MAX,
    PC_EPOCHS_MIN, PC_EPOCHS_MAX,
    PC_GLOBAL_EPOCHS,
)
from mealpy_comparison_v2 import get_all_algorithms_v3, get_special_params

SEED = 42

# generate_assignments.py ALGO_LABELS ile uyumlu (B0_KMEANS hariç)
COMPARE_ALGOS = [
    'B1_HHO', 'B2_HGS', 'B3_MFO', 'SFOA', 'SFOA_06',
    'H1_HHO+HGS', 'H4_MFO+HHO', 'H9_QSA+CDO', 'H12_MFO+CDO',
    'H13_HHO+GAop', 'HA_AVOAHGS',
]

DEFAULT_LATENT_DIM = 20
DEFAULT_LR = 0.01
DEFAULT_REG = 0.01
DEFAULT_EPOCHS_CLUSTER = 50
DEFAULT_EPOCHS_GLOBAL = 100


def _build_algo_map():
    """Tüm mealpy algoritmalarını dict olarak döndür."""
    algo_map = {a['full_name']: a for a in get_all_algorithms_v3()}
    from mealpy.evolutionary_based import GA
    algo_map['GA.EliteMultiGA'] = {
        'full_name': 'GA.EliteMultiGA',
        'class': GA.EliteMultiGA,
    }
    return algo_map


def _algo_config(label: str):
    """generate_assignments.py ALGO_CONFIG ile eşleş."""
    from generate_assignments import ALGO_CONFIG
    for lbl, g_name, l_name in ALGO_CONFIG:
        if lbl == label:
            return lbl, g_name, l_name
    raise KeyError(f"ALGO_CONFIG içinde '{label}' bulunamadı")


def evaluate_with_params(
    train: np.ndarray,
    test: np.ndarray,
    assignments: np.ndarray,
    n_items: int,
    K: int,
    params: Dict[int, Dict[str, float]],
    gray_mask: Optional[np.ndarray] = None,
    seed: int = SEED,
    use_bias: bool = True,
) -> Tuple[float, float, List[Tuple[int, int, float, float]]]:
    """
    Optimize edilmiş per-cluster parametrelerle WNMFSharedV değerlendirmesi.
    Returns (mae, rmse, eval_rows).
    """
    from wnmf_model import WNMFModel, ClusterWNMF
    from wnmf_utils import split_by_cluster, remap_user_ids

    # En büyük latent_dim ile global V eğit
    max_ld = max(p['latent_dim'] for p in params.values())
    max_ld = max(max_ld, 5)

    model = WNMFModel(
        n_users=len(assignments),
        n_items=n_items,
        latent_dim=max_ld,
        learning_rate=DEFAULT_LR,
        regularization=DEFAULT_REG,
        n_epochs=DEFAULT_EPOCHS_GLOBAL,
        random_seed=seed,
        use_bias=use_bias,
    )

    if gray_mask is not None and np.any(gray_mask):
        user_ids = train[:, 0].astype(np.int32)
        sw = np.where(gray_mask[user_ids], 0.1, 1.0).astype(np.float32)
        model.fit(train, sample_weights=sw, verbose=False)
    else:
        model.fit(train, verbose=False)

    V_big = model.V.copy()
    mu_global = float(model.mu)
    b_i_global = model.b_i.copy()

    cluster_train, _ = split_by_cluster(train, assignments, gray_mask)
    cluster_test, _ = split_by_cluster(test, assignments, gray_mask)

    cluster_models: Dict[int, Tuple[ClusterWNMF, dict, int]] = {}

    for cid in range(K):
        p = params.get(cid)
        if p is None:
            continue

        c_train = cluster_train.get(cid)
        if c_train is None or len(c_train) < 5:
            continue

        ld = min(p['latent_dim'], max_ld)
        V_k = V_big[:, :ld].copy()

        c_train_r, _, uid_map, n_loc = remap_user_ids(
            c_train, np.empty((0, 3), dtype=np.float32), n_items,
        )

        if n_loc < 2:
            continue

        cm = ClusterWNMF(
            n_users=n_loc,
            n_items=n_items,
            latent_dim=ld,
            V_shared=V_k,
            learning_rate=p['learning_rate'],
            regularization=p['regularization'],
            n_epochs=p['n_epochs_cluster'],
            random_seed=seed + cid,
            use_bias=use_bias,
            mu=mu_global,
            b_i_global=b_i_global,
            cluster_ratings=c_train_r if use_bias else None,
        )
        cm.fit_cluster_U(c_train_r, verbose=False)
        cluster_models[cid] = (cm, uid_map, n_loc)

    # Test değerlendirmesi
    true_vals, pred_vals = [], []
    eval_rows = []

    for row in test:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        cid = int(assignments[u]) if u < len(assignments) else 0
        cm_tuple = cluster_models.get(cid)
        if cm_tuple is None:
            pred = mu_global
        else:
            cm, uid_map, n_loc = cm_tuple
            u_local = uid_map.get(u)
            if u_local is not None and 0 <= u_local < n_loc:
                b_i_val = cm.b_i[i] if use_bias and i < len(cm.b_i) else 0.0
                pred = cm.mu + cm.b_u[u_local] + b_i_val + float(np.dot(cm.U[u_local], cm.V[i]))
            else:
                pred = mu_global
        pred = float(np.clip(pred, 1.0, 5.0))
        true_vals.append(r)
        pred_vals.append(pred)
        eval_rows.append((u, i, r, pred))

    errors = np.array(true_vals) - np.array(pred_vals)
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return mae, rmse, eval_rows


def evaluate_uniform_baseline(
    train: np.ndarray,
    test: np.ndarray,
    assignments: np.ndarray,
    n_items: int,
    K: int,
    gray_mask: Optional[np.ndarray] = None,
    latent_dim: int = DEFAULT_LATENT_DIM,
    lr: float = DEFAULT_LR,
    reg: float = DEFAULT_REG,
    epochs_cluster: int = DEFAULT_EPOCHS_CLUSTER,
    epochs_global: int = DEFAULT_EPOCHS_GLOBAL,
    seed: int = SEED,
    use_bias: bool = True,
) -> Tuple[float, float]:
    """Uniform parametrelerle baseline MAE."""
    params = {
        cid: {
            'latent_dim': latent_dim,
            'learning_rate': lr,
            'regularization': reg,
            'n_epochs_cluster': epochs_cluster,
        }
        for cid in range(K)
    }
    mae, rmse, _ = evaluate_with_params(
        train, test, assignments, n_items, K, params,
        gray_mask=gray_mask, seed=seed, use_bias=use_bias,
    )
    return mae, rmse


# ============================================================
# Parallel worker (algo bazında)
# ============================================================
def _mp_optimize_algo(job):
    """Worker: bir algoritma için per-cluster optimizasyon yap."""
    (
        label, g_name, l_name,
        train, test, assignments, n_items, K,
        opt_epoch, opt_pop, gray_mask, seed, use_bias, verbose,
    ) = job

    algo_map = _build_algo_map()

    if g_name is None:
        return label, None, None, 'B0_KMEANS — atlandı'
    if l_name is None:
        # Single algo
        algo_info = algo_map.get(g_name)
    elif '||' in label:
        # Parallel hybrid — ilkini kullan (benzer sonuç)
        algo_info = algo_map.get(g_name)
    elif l_name == 'GAop':
        algo_info = algo_map.get(g_name)
    else:
        # Hybrid — global algo
        algo_info = algo_map.get(g_name)

    if algo_info is None:
        return label, None, None, f'{g_name} bulunamadı'

    try:
        best_sol, best_fit, best_params = run_per_cluster_optimization(
            matrix=None,
            K=K,
            assignments=assignments,
            algo_info=algo_info,
            algo_map=algo_map,
            train_ratings=train,
            test_ratings=test,
            n_items=n_items,
            opt_epoch=opt_epoch,
            opt_pop=opt_pop,
            gray_mask=gray_mask,
            seed=seed,
            verbose=verbose,
        )
        return label, best_fit, best_params, None
    except Exception as exc:
        import traceback
        return label, None, None, f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}'


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser(
        description='Per-cluster WNMF parameter optimization comparison'
    )
    p.add_argument('--dataset', default='100k', choices=['100k', '1m'])
    p.add_argument('--assign-dir', required=True,
                   help='Assignments klasörü (assignments.npy içermeli)')
    p.add_argument('--algo', nargs='+', default=None,
                   help=f'Karşılaştırılacak algoritmalar: {COMPARE_ALGOS}')
    p.add_argument('--epochs', type=int, default=25,
                   help='Meta-algoritma epoch sayısı (default: 25)')
    p.add_argument('--pop', type=int, default=20,
                   help='Popülasyon büyüklüğü (default: 20)')
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--algo-jobs', type=int, default=1,
                   help='Algoritmaları paralel çalıştır (0=otomatik)')
    p.add_argument('--no-bias', action='store_true')
    p.add_argument('--epochs-global', type=int, default=PC_GLOBAL_EPOCHS)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    # Assignment yükle
    from per_cluster_wnmf_optimizer import _get_wnmf_utils
    load_assignment, load_ratings_100k, load_ratings_100k_all, load_ratings_1m, _, _ = _get_wnmf_utils()

    assignments, gray_mask = load_assignment(args.assign_dir)
    K = int(assignments.max()) + 1

    # Veri yükle
    DATA_100K_TRAIN = os.path.join(_REPO_ROOT, 'data', 'ml-100k', 'u1.base')
    DATA_100K_TEST = os.path.join(_REPO_ROOT, 'data', 'ml-100k', 'u1.test')
    DATA_1M = os.path.join(_REPO_ROOT, 'data', 'ml-1m', 'ratings.dat')

    if args.dataset == '100k':
        train, test = load_ratings_100k(DATA_100K_TRAIN, DATA_100K_TEST)
        n_items = 1682
    else:
        train, test = load_ratings_1m(DATA_1M)
        n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1

    use_bias = not args.no_bias

    print('=' * 70)
    print('PER-CLUSTER WNMF PARAMETER OPTIMIZATION COMPARISON')
    print('=' * 70)
    print(f'Dataset      : {args.dataset}')
    print(f'Assign dir   : {args.assign_dir}')
    print(f'K            : {K}')
    print(f'n_users      : {len(assignments)}')
    print(f'n_items      : {n_items}')
    print(f'Train ratings: {len(train):,}')
    print(f'Test ratings : {len(test):,}')
    print(f'Global epochs: {args.epochs_global}')
    print(f'Meta epochs  : {args.epochs}')
    print(f'Meta pop     : {args.pop}')
    print(f'Bias         : {use_bias}')
    print(f'Val sample   : {PC_VAL_SAMPLE}/cluster')
    print()

    # Baseline (uniform params)
    print('--- Baseline (uniform params) ---')
    t0 = time.time()
    base_mae, base_rmse = evaluate_uniform_baseline(
        train, test, assignments, n_items, K,
        gray_mask=gray_mask,
        epochs_global=args.epochs_global,
        seed=args.seed,
        use_bias=use_bias,
    )
    print(f'  Uniform params MAE: {base_mae:.6f}  RMSE: {base_rmse:.6f}  ({time.time()-t0:.1f}s)')
    print()

    # Algoritmaları seç
    selected = args.algo if args.algo else [a for a in COMPARE_ALGOS if a != 'B0_KMEANS']

    # Normalize algo isimleri
    from generate_assignments import ALGO_CONFIG
    algo_lookup = {lbl: (lbl, g, l) for lbl, g, l in ALGO_CONFIG}

    valid_algos = []
    for label in selected:
        if label == 'B0_KMEANS':
            continue
        cfg = algo_lookup.get(label)
        if cfg is None:
            print(f'  UYARI: {label} ALGO_CONFIG içinde yok, atlanıyor.')
            continue
        valid_algos.append(cfg)

    print(f'Optimize edilecek algoritmalar: {len(valid_algos)}')
    print()

    # Run
    all_results = []

    if args.algo_jobs == 1:
        # Sequential
        for label, g_name, l_name in valid_algos:
            if g_name is None:
                continue
            algo_map = _build_algo_map()
            algo_info = algo_map.get(g_name)
            if algo_info is None:
                print(f'  [{label}] {g_name} bulunamadı, atlanıyor.')
                continue

            best_sol, best_fit, best_params = run_per_cluster_optimization(
                matrix=None, K=K, assignments=assignments,
                algo_info=algo_info, algo_map=algo_map,
                train_ratings=train, test_ratings=test,
                n_items=n_items, opt_epoch=args.epochs, opt_pop=args.pop,
                gray_mask=gray_mask, seed=args.seed, verbose=args.verbose,
            )

            # Test MAE
            test_mae, test_rmse, _ = evaluate_with_params(
                train, test, assignments, n_items, K, best_params,
                gray_mask=gray_mask, seed=args.seed, use_bias=use_bias,
            )
            all_results.append({
                'algo': label,
                'val_mae': best_fit,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'params': {str(k): v for k, v in best_params.items()},
            })
            print(f'  [{label}] Test MAE: {test_mae:.6f}  '
                  f'vs baseline: {test_mae - base_mae:+.6f}\n')
    else:
        # Parallel
        import multiprocessing
        nw = args.algo_jobs if args.algo_jobs > 0 else min(len(valid_algos), multiprocessing.cpu_count())
        print(f'Paralel mod: {len(valid_algos)} algo, {nw} worker')
        jobs = [
            (
                label, g_name, l_name,
                train, test, assignments, n_items, K,
                args.epochs, args.pop, gray_mask, args.seed, use_bias, args.verbose,
            )
            for label, g_name, l_name in valid_algos
        ]
        with ProcessPoolExecutor(max_workers=nw) as pool:
            results_raw = list(pool.map(_mp_optimize_algo, jobs))

        for label, best_fit, best_params, err_msg in results_raw:
            if err_msg:
                print(f'  [{label}] HATA: {err_msg}')
                continue
            if best_params is None:
                print(f'  [{label}] optimize edilemedi.')
                continue

            test_mae, test_rmse, _ = evaluate_with_params(
                train, test, assignments, n_items, K, best_params,
                gray_mask=gray_mask, seed=args.seed, use_bias=use_bias,
            )
            all_results.append({
                'algo': label,
                'val_mae': best_fit,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'params': {str(k): v for k, v in best_params.items()},
            })
            print(f'  [{label}] Test MAE: {test_mae:.6f}  '
                  f'vs baseline: {test_mae - base_mae:+.6f}')

    # Summary
    print()
    print('=' * 70)
    print('SONUÇLAR')
    print('=' * 70)
    print(f'{"Algoritma":<16} {"Val MAE":>10} {"Test MAE":>10} {"vs Base":>10} {"Test RMSE":>10}')
    print('-' * 60)
    print(f'{"Baseline(uniform)":<16} {"-":>10} {base_mae:>10.6f} {"0.000000":>10} {base_rmse:>10.6f}')
    for r in sorted(all_results, key=lambda x: x['test_mae']):
        diff = r['test_mae'] - base_mae
        print(
            f'{r["algo"]:<16} {r["val_mae"]:>10.6f} '
            f'{r["test_mae"]:>10.6f} {diff:>+10.6f} {r["test_rmse"]:>10.6f}'
        )


if __name__ == '__main__':
    main()
