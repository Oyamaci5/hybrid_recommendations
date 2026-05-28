"""
cluster_weight_alpha optimizasyonu (DE / mealpy).

weighted_cluster modunda membership_w = 1 / (1 + alpha * d_centroid)
Fitness: validation MAE (minimize), HA_AVOAHGS atamaları üzerinde.

Kullanım (repo kökünden):
    python mealpy/optimize_cluster_weight_alpha.py \\
        --algo HA_AVOAHGS \\
        --k 70 \\
        --assign-suffix _euc_imkpp_nogs_none_wnmf20_k70_kmref \\
        --pop-size 15 --epoch 25

En iyi alpha ile tam değerlendirme:
    python mealpy/optimize_cluster_weight_alpha.py ... --final-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mealpy import FloatVar, Problem

_MEALPY_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_MEALPY_DIR)
_WNMF_DIR = os.path.join(_REPO_ROOT, 'wnmf')
for _p in (_REPO_ROOT, _WNMF_DIR, _MEALPY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from wnmf.wnmf_utils import (  # noqa: E402
    load_assignment,
    load_centroids,
    load_ratings_100k,
    resolve_test_cluster_ids,
)
from wnmf.wnmf_experiment import (  # noqa: E402
    _build_knn_sim_index,
    _knn_predict_from_sims,
    build_item_popularity,
    run_cluster_knn,
)


@dataclass
class EvalHistoryRow:
    alpha: float
    val_mae: float
    eval_idx: int


class WeightedAlphaEvaluator:
    """weighted_cluster MAE — sim_index ve centroid mesafeleri önbellekte."""

    def __init__(
        self,
        train: np.ndarray,
        val: np.ndarray,
        assignments: np.ndarray,
        gray_mask: np.ndarray,
        centroids: np.ndarray,
        *,
        k_neighbors: int = 20,
        similarity: str = 'cosine',
        min_common: int = 3,
        assign_dir: Optional[str] = None,
        n_items: Optional[int] = None,
        verbose: bool = False,
    ):
        self.train = np.asarray(train, dtype=np.float64)
        self.val = np.asarray(val, dtype=np.float64)
        self.assignments = np.asarray(assignments, dtype=np.int64).ravel()
        self.gray_mask = np.asarray(gray_mask, dtype=bool).ravel()
        self.centroids = np.asarray(centroids, dtype=np.float64)
        self.k_neighbors = max(1, int(k_neighbors))
        self.similarity = (similarity or 'cosine').strip().lower()
        self.min_common = max(1, int(min_common))
        self.verbose = bool(verbose)

        self.n_users = len(self.assignments)
        if n_items is None:
            n_items = int(
                max(
                    self.train[:, 1].max() if len(self.train) else 0,
                    self.val[:, 1].max() if len(self.val) else 0,
                )
            ) + 1
        self.n_items = int(n_items)

        self.user_ratings: dict = {}
        for row in self.train:
            u, i, r = int(row[0]), int(row[1]), float(row[2])
            self.user_ratings.setdefault(u, {})[i] = r
        self.user_means = {
            u: float(np.mean(list(d.values())))
            for u, d in self.user_ratings.items()
        }
        self.global_mean = float(self.train[:, 2].mean()) if len(self.train) else 3.0
        self.item_popularity = build_item_popularity(user_ratings=self.user_ratings)

        white_set = {u for u in range(self.n_users) if not self.gray_mask[u]}
        self.knn_candidates = sorted(white_set)
        self.cluster_users: dict = {}
        for u in self.knn_candidates:
            cid = int(self.assignments[u])
            self.cluster_users.setdefault(cid, []).append(u)

        self.test_cluster_ids = resolve_test_cluster_ids(
            self.train,
            self.assignments,
            self.centroids,
            nearest_centroid=False,
            n_items=self.n_items,
            algo_label='alpha-opt',
            assign_dir=assign_dir,
        )

        K_cent = self.centroids.shape[0]
        self.cent_dist = np.zeros((K_cent, K_cent), dtype=np.float64)
        for a in range(K_cent):
            self.cent_dist[a] = np.linalg.norm(
                self.centroids - self.centroids[a], axis=1,
            )

        t0 = time.time()
        self.sim_index = _build_knn_sim_index(
            self.user_ratings,
            self.user_means,
            self.item_popularity,
            similarity=self.similarity,
            min_common=self.min_common,
            allowed_neighbors=set(self.knn_candidates),
        )
        if self.verbose:
            print(
                f"  [alpha-opt] sim_index hazır ({time.time() - t0:.1f}s), "
                f"val={len(self.val):,} çift",
                flush=True,
            )

        self.history: List[EvalHistoryRow] = []
        self._eval_count = 0

    def _membership_weight(self, cid: int, v: int, alpha: float) -> float:
        v_cid = int(self.assignments[v])
        if v_cid == cid:
            return 1.0
        if alpha <= 0.0:
            return 1.0
        return float(1.0 / (1.0 + alpha * self.cent_dist[cid, v_cid]))

    def predict(self, u: int, i: int, cid: int, alpha: float) -> float:
        sims: List[Tuple[float, int]] = []
        for s, v in self.sim_index.get(u, []):
            if v == u:
                continue
            if i not in self.user_ratings.get(v, {}):
                continue
            if s <= 0.0:
                continue
            eff = float(s) * self._membership_weight(cid, v, alpha)
            if eff > 0.0:
                sims.append((eff, v))
        return _knn_predict_from_sims(
            u, i, sims, self.user_ratings, self.user_means,
            self.global_mean, self.k_neighbors,
        )

    def val_mae(self, alpha: float) -> float:
        alpha = float(np.clip(alpha, 0.0, None))
        errs = []
        for row in self.val:
            u, i, r = int(row[0]), int(row[1]), float(row[2])
            cid = int(self.test_cluster_ids[u])
            pred = self.predict(u, i, cid, alpha)
            errs.append(abs(r - pred))
        mae = float(np.mean(errs)) if errs else 9999.0
        self._eval_count += 1
        self.history.append(
            EvalHistoryRow(alpha=alpha, val_mae=mae, eval_idx=self._eval_count),
        )
        if self.verbose and (self._eval_count <= 5 or self._eval_count % 10 == 0):
            print(f"    eval #{self._eval_count}: alpha={alpha:.4f} -> val_mae={mae:.4f}")
        return mae


class AlphaDEProblem(Problem):
    def __init__(self, evaluator: WeightedAlphaEvaluator, bounds):
        self.evaluator = evaluator
        super().__init__(bounds=bounds, minmax='min')

    def obj_func(self, x):
        return self.evaluator.val_mae(float(x[0]))


def _resolve_de_class():
    try:
        from mealpy.evolutionary_based.DE import OriginalDE
        return 'DE.OriginalDE', OriginalDE
    except ImportError:
        from mealpy.evolutionary_based import DE
        return 'DE', getattr(DE, 'OriginalDE', None)


def optimize_alpha_de(
    evaluator: WeightedAlphaEvaluator,
    *,
    alpha_lo: float = 0.0,
    alpha_hi: float = 5.0,
    pop_size: int = 15,
    epoch: int = 25,
    seed: int = 42,
) -> Tuple[float, float, Dict[str, Any]]:
    full_name, de_cls = _resolve_de_class()
    if de_cls is None:
        raise ImportError('mealpy DE sınıfı bulunamadı (OriginalDE)')

    bounds = [FloatVar(lb=float(alpha_lo), ub=float(alpha_hi))]
    problem = AlphaDEProblem(evaluator, bounds)

    sp: Dict[str, Any] = {'epoch': int(epoch), 'pop_size': int(pop_size)}
    try:
        from mealpy_comparison_v2 import get_special_params
        extra = get_special_params(full_name, int(epoch), int(pop_size))
        if extra:
            sp.update(extra)
    except Exception:
        pass

    model = de_cls(**sp)
    try:
        model.solve(problem, seed=int(seed))
    except TypeError:
        model.solve(problem)

    best_alpha = float(np.clip(float(model.g_best.solution[0]), alpha_lo, alpha_hi))
    best_mae = float(model.g_best.target.fitness)
    return best_alpha, best_mae, {
        'de_class': full_name,
        'pop_size': pop_size,
        'epoch': epoch,
        'seed': seed,
        'alpha_lo': alpha_lo,
        'alpha_hi': alpha_hi,
        'n_evals': evaluator._eval_count,
    }


def _assignment_dir(
    assign_root: str,
    dataset_name: str,
    algo: str,
    suffix: str,
) -> str:
    s = suffix if suffix.startswith('_') else f'_{suffix}'
    base = os.path.join(assign_root, dataset_name, f'{algo}{s}')
    if os.path.isdir(base):
        return base
    if algo != 'B0_KMEANS' and s.endswith('_kmref'):
        alt = os.path.join(assign_root, dataset_name, f'{algo}{s[:-6]}')
        if os.path.isdir(alt):
            return alt
    return base


def _train_val_split(train: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))
    n_val = max(1, int(len(train) * val_ratio))
    val = train[idx[:n_val]]
    tr = train[idx[n_val:]]
    return tr, val


def run_final_eval(
    *,
    train: np.ndarray,
    test: np.ndarray,
    assignments: np.ndarray,
    gray_mask: np.ndarray,
    assign_dir: str,
    centroids: np.ndarray,
    algo: str,
    alpha: float,
    k_neighbors: int,
    similarity: str,
    min_common: int,
    top_n: int,
    relevance_threshold: float,
) -> dict:
    return run_cluster_knn(
        train,
        test,
        assignments,
        gray_mask,
        memberships=None,
        n_items=int(max(train[:, 1].max(), test[:, 1].max())) + 1,
        algo_label=algo,
        k_neighbors=k_neighbors,
        similarity=similarity,
        min_common=min_common,
        knn_mode='weighted_cluster',
        top_n=top_n,
        relevance_threshold=relevance_threshold,
        centroids=centroids,
        nearest_centroid=False,
        assign_dir=assign_dir,
        cluster_weight_alpha=float(alpha),
        cluster_weight_base=1.0,
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description='cluster_weight_alpha DE optimizasyonu')
    p.add_argument('--dataset', default='100k', choices=['100k'])
    p.add_argument('--algo', default='HA_AVOAHGS')
    p.add_argument('--k', type=int, default=70)
    p.add_argument(
        '--assign-root',
        default=os.path.join('mealpy', 'results', 'assignments'),
    )
    p.add_argument(
        '--assign-suffix',
        default='_euc_imkpp_nogs_none_wnmf20_k70_kmref',
    )
    p.add_argument('--knn', type=int, default=20)
    p.add_argument('--similarity', default='cosine')
    p.add_argument('--min-common', type=int, default=3)
    p.add_argument('--val-ratio', type=float, default=0.15,
                   help='Train içinden validation oranı (DE fitness)')
    p.add_argument('--val-max', type=int, default=4000,
                   help='Validation üst sınırı (hız için alt örnekleme)')
    p.add_argument('--alpha-lo', type=float, default=0.0)
    p.add_argument('--alpha-hi', type=float, default=5.0)
    p.add_argument('--pop-size', type=int, default=15)
    p.add_argument('--epoch', type=int, default=25)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--top-n', type=int, default=10)
    p.add_argument('--relevance-threshold', type=float, default=4.0)
    p.add_argument('--verbose', action='store_true')
    p.add_argument(
        '--final-run',
        action='store_true',
        help='En iyi alpha ile u1.test üzerinde tam run_cluster_knn',
    )
    p.add_argument(
        '--out-dir',
        default=os.path.join('mealpy', 'results', 'alpha_de_opt'),
    )
    args = p.parse_args(argv)

    data_base = os.path.join(_REPO_ROOT, 'data', 'ml-100k')
    train_full, test = load_ratings_100k(
        os.path.join(data_base, 'u1.base'),
        os.path.join(data_base, 'u1.test'),
    )
    train_fit, val = _train_val_split(train_full, args.val_ratio, args.seed)
    if len(val) > args.val_max:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(val), size=int(args.val_max), replace=False)
        val = val[idx]

    ds_name = 'ml100k'
    assign_dir = _assignment_dir(
        os.path.join(_REPO_ROOT, args.assign_root),
        ds_name,
        args.algo,
        args.assign_suffix,
    )
    if not os.path.isdir(assign_dir):
        print(f'HATA: assignment klasörü yok: {assign_dir}', file=sys.stderr)
        return 1

    assignments, gray_mask = load_assignment(assign_dir)
    n_clusters = int(assignments.max()) + 1
    centroids = load_centroids(assign_dir, n_clusters)
    if centroids is None:
        print(f'HATA: best_sol.npy yok: {assign_dir}', file=sys.stderr)
        return 1

    print('=' * 60)
    print('cluster_weight_alpha DE optimizasyonu')
    print('=' * 60)
    print(f'Algo          : {args.algo}')
    print(f'K             : {args.k} (atama: {n_clusters} aktif küme)')
    print(f'Assignment    : {assign_dir}')
    print(f'Train (fit)   : {len(train_fit):,}')
    print(f'Val (fitness) : {len(val):,}')
    print(f'Alpha aralığı : [{args.alpha_lo}, {args.alpha_hi}]')
    print(f'DE            : pop={args.pop_size}, epoch={args.epoch}')
    print('=' * 60)

    evaluator = WeightedAlphaEvaluator(
        train_fit,
        val,
        assignments,
        gray_mask,
        centroids,
        k_neighbors=args.knn,
        similarity=args.similarity,
        min_common=args.min_common,
        assign_dir=assign_dir,
        verbose=args.verbose,
    )

    t0 = time.time()
    best_alpha, best_val_mae, meta = optimize_alpha_de(
        evaluator,
        alpha_lo=args.alpha_lo,
        alpha_hi=args.alpha_hi,
        pop_size=args.pop_size,
        epoch=args.epoch,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    # Referans: alpha=1.0 validation MAE
    ref_mae = evaluator.val_mae(1.0)

    os.makedirs(os.path.join(_REPO_ROOT, args.out_dir), exist_ok=True)
    tag = f'{args.algo}_k{args.k}_de_alpha'
    out_json = os.path.join(_REPO_ROOT, args.out_dir, f'{tag}.json')
    hist_csv = os.path.join(_REPO_ROOT, args.out_dir, f'{tag}_history.csv')

    payload = {
        'algo': args.algo,
        'k': args.k,
        'assign_suffix': args.assign_suffix,
        'assign_dir': assign_dir,
        'best_alpha': best_alpha,
        'best_val_mae': best_val_mae,
        'ref_alpha_1.0_val_mae': ref_mae,
        'de_meta': meta,
        'elapsed_s': elapsed,
        'knn': args.knn,
        'similarity': args.similarity,
        'val_n': len(val),
    }

    import pandas as pd
    pd.DataFrame(
        [{'alpha': h.alpha, 'val_mae': h.val_mae, 'eval_idx': h.eval_idx}
         for h in evaluator.history],
    ).to_csv(hist_csv, index=False)

    final_row = None
    if args.final_run:
        print(f'\n[final] u1.test, alpha={best_alpha:.4f} ...')
        final_row = run_final_eval(
            train=train_full,
            test=test,
            assignments=assignments,
            gray_mask=gray_mask,
            assign_dir=assign_dir,
            centroids=centroids,
            algo=args.algo,
            alpha=best_alpha,
            k_neighbors=args.knn,
            similarity=args.similarity,
            min_common=args.min_common,
            top_n=args.top_n,
            relevance_threshold=args.relevance_threshold,
        )
        payload['test_mae'] = final_row.get('mae')
        payload['test_rmse'] = final_row.get('rmse')
        payload['test_ndcg_at_10'] = final_row.get('ndcg_at_10')

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print('\n' + '=' * 60)
    print('SONUÇ')
    print('=' * 60)
    print(f'En iyi alpha (val) : {best_alpha:.4f}')
    print(f'Val MAE (en iyi)   : {best_val_mae:.4f}')
    print(f'Val MAE (alpha=1)  : {ref_mae:.4f}')
    print(f'DE değerlendirme   : {meta.get("n_evals")} fitness çağrısı')
    print(f'Süre               : {elapsed:.1f}s')
    print(f'Kayıt              : {out_json}')
    print(f'Geçmiş             : {hist_csv}')
    if final_row:
        print(
            f'Test MAE (alpha={best_alpha:.4f}): '
            f'{final_row["mae"]:.4f}  RMSE={final_row["rmse"]:.4f}  '
            f'NDCG@10={final_row.get("ndcg_at_10", float("nan")):.4f}',
        )
    print('=' * 60)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
