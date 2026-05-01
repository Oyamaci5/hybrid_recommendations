"""
main_clustering_cf.py (experiments)
-----------------------------------
Legacy metaheuristic Clustering-CF runner (DOA/PSO/KMeans).

For assignment-first modular pipeline, use:
  - generate_assignments.py (or your assignment source)
  - experiments.generate_recommendations
  - experiments.evaluate_assignments
  - compare/compare_baselines.py

Çalıştırma::

    python experiments/main_clustering_cf.py
    python experiments/main_clustering_cf.py --dataset 1m --K 50 --pop-size 20 --max-iter 50
    python experiments/main_clustering_cf.py --meta doa --distance pearson
    python experiments/main_clustering_cf.py --meta pso --distance euclidean --normalize user-mean

Pipeline:
  1. core/data_loader  → train_matrix, test_ratings
  2. core/fitness      → FitnessEvaluator (kullanıcı istatistikleri önden hesaplanır)
  3. optimizers/doa    → centroid optimizasyonu
  4. models/cluster_manager → kullanıcı atama
  5. core/metrics      → MAE, RMSE, Precision@10, Recall@10
  6. Sonuç tablosu
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.data_loader import load_dataset
from core.fitness import make_fitness_fn
from core.metrics import evaluate_cf
from models.baselines import KMeansBaseline, PSOBaseline
from models.cluster_manager import ClusterManager
from optimizers.doa import DOA
from preprocess.mkmeans_plus_plus import make_mkmeans_init_population

from experiments.evaluate_assignments import run as run_evaluate
from experiments.generate_recommendations import run as run_generate_recommendations

# ── Varsayılan ayarlar ────────────────────────────────────────────────────────
_ROOT = str(_PROJECT_ROOT)
_DATA_100K = os.path.join(_ROOT, "data", "ml-100k", "u.data")
_DATA_1M = os.path.join(_ROOT, "data", "ml-1m", "ratings.dat")
_RESULTS = os.path.join(_ROOT, "results", "clustering_cf")

DEFAULTS = dict(
    K=30,
    pop_size=30,
    max_iter=100,
    top_k=30,
    N=10,
    seed=42,
    distance="pearson",
    normalize="none",
    meta="doa",
)
# ──────────────────────────────────────────────────────────────────────────────


def _table(rows: list[dict]) -> None:
    """Sözlük listesini hizalanmış ASCII tablo olarak yazdırır."""
    if not rows:
        return
    headers = list(rows[0].keys())
    widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}
    sep = "+" + "+".join("-" * (widths[h] + 2) for h in headers) + "+"
    print(sep)
    print("|" + "|".join(f" {h:<{widths[h]}} " for h in headers) + "|")
    print(sep)
    for r in rows:
        print("|" + "|".join(f" {str(r[h]):<{widths[h]}} " for h in headers) + "|")
    print(sep)


def _run_assignment_mode(args: argparse.Namespace) -> None:
    """
    Bridge entrypoint for the assignment-based modular pipeline.
    """
    if not args.assignment_path:
        raise ValueError("assignment mode requires --assignment-path")
    if not args.pipeline_out_dir:
        raise ValueError("assignment mode requires --pipeline-out-dir")

    data_path = _DATA_1M if args.dataset == "1m" else _DATA_100K
    pred_dir = os.path.join(args.pipeline_out_dir, "predictions")
    eval_dir = os.path.join(args.pipeline_out_dir, "evaluation")
    os.makedirs(args.pipeline_out_dir, exist_ok=True)

    print("\n[assignment-mode] experiments.generate_recommendations")
    run_generate_recommendations(
        dataset=args.dataset,
        ratings_path=data_path,
        assignment_path=args.assignment_path,
        gray_mask_path=args.gray_mask_path,
        out_dir=pred_dir,
        K=args.K,
        top_k=args.top_k,
        n_reco=args.N,
        test_ratio=0.2,
        seed=args.seed,
        normalize=args.normalize,
    )

    print("[assignment-mode] experiments.evaluate_assignments")
    result = run_evaluate(
        ratings_path=data_path,
        prediction_dir=pred_dir,
        output_dir=eval_dir,
        test_ratio=0.2,
        seed=args.seed,
        normalize=args.normalize,
        n_reco=args.N,
        relevance_threshold=3.5,
    )
    print(f"Tamamlandı. Özet JSON: {result['json']}")
    print(f"Tamamlandı. Özet CSV: {result['csv']}")
    summary = result.get("summary", {})
    if summary:
        print("Segment özeti (MAE / RMSE):")
        for seg, vals in summary.items():
            if "MAE" in vals and "RMSE" in vals:
                print(f"  {seg}: MAE={vals['MAE']} RMSE={vals['RMSE']}")


def _run_algo(
    name: str,
    optimizer,
    fitness_eval,
    train_matrix: np.ndarray,
    clustering_matrix: np.ndarray,
    test_ratings: np.ndarray,
    K: int,
    n_items: int,
    top_k: int,
    N: int,
    results_dir: str,
    distance_metric: Literal["pearson", "euclidean"],
    seed: int,
) -> dict:
    """Tek bir algoritmayı çalıştırır, değerlendirir ve sonuç dict'i döndürür."""
    print(f"\n{'─'*56}")
    print(f"  {name}")
    print(f"{'─'*56}")

    t0 = time.time()
    init_population = None
    if name.upper() == "DOA":
        init_population = make_mkmeans_init_population(
            clustering_matrix,
            K=K,
            pop_size=getattr(optimizer, "N"),
            seed=seed,
        )
    if init_population is not None:
        flat, best_fit, curve = optimizer.optimize(
            fitness_eval, init_population=init_population
        )
    else:
        flat, best_fit, curve = optimizer.optimize(fitness_eval)
    elapsed = time.time() - t0

    centroids = flat.reshape(K, n_items)

    cm = ClusterManager(K)
    cm.fit(
        centroids,
        clustering_matrix,
        fitness_eval,
        distance_metric=distance_metric,
    )
    s = cm.summary()
    print(
        f"  Küme: min={s['min_size']}, max={s['max_size']}, "
        f"ort={s['mean_size']}, boş={s['empty']}"
    )

    print("  RS metrikleri hesaplanıyor...")
    metrics = evaluate_cf(
        test_ratings,
        train_matrix,
        cm.labels,
        centroids,
        top_k=top_k,
        N=N,
        distance_metric=distance_metric,
    )

    # Convergence curve kaydet
    os.makedirs(results_dir, exist_ok=True)
    if curve:
        np.save(os.path.join(results_dir, f"curve_{name}.npy"), np.array(curve))

    return {
        "Algoritma": name,
        "Best Fitness": round(best_fit, 2),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        f"P@{N}": metrics[f"Precision@{N}"],
        f"R@{N}": metrics[f"Recall@{N}"],
        "Süre (s)": round(elapsed, 1),
    }


def run(
    dataset: str = "100k",
    K: int = DEFAULTS["K"],
    pop_size: int = DEFAULTS["pop_size"],
    max_iter: int = DEFAULTS["max_iter"],
    top_k: int = DEFAULTS["top_k"],
    N: int = DEFAULTS["N"],
    seed: int = DEFAULTS["seed"],
    meta: str = DEFAULTS["meta"],
    distance: Literal["pearson", "euclidean"] = DEFAULTS["distance"],
    normalize: Literal["none", "user-mean"] = DEFAULTS["normalize"],
    results_dir: str = _RESULTS,
) -> list[dict]:
    """
    Tam pipeline.

    Parameters
    ----------
    dataset : {"100k", "1m"}
    K : int — küme sayısı
    pop_size : int
    max_iter : int
    top_k : int — CF komşu sayısı
    N : int — Precision@N / Recall@N
    seed : int
    meta : {"doa", "pso", "kmeans"} — çalıştırılacak metaheuristic
    distance : {"pearson", "euclidean"} — fitness ve atama metriği
    normalize : {"none", "user-mean"} — clustering giriş normalizasyonu
    results_dir : str

    Returns
    -------
    list[dict] — her algoritma için metrik satırı
    """
    # ── 1. Veri ────────────────────────────────────────────────────────
    data_path = _DATA_1M if dataset == "1m" else _DATA_100K
    print(f"\n{'='*56}")
    print(f"  Veri: MovieLens {dataset.upper()}  |  K={K}")
    print(f"{'='*56}")

    train_matrix, test_ratings, info = load_dataset(
        data_path, seed=seed, normalize=normalize
    )
    clustering_matrix = info["clustering_matrix"]
    n_users, n_items = train_matrix.shape
    print(
        f"  {n_users} kullanıcı × {n_items} film  |  "
        f"train={info['n_train']}  test={info['n_test']}  "
        f"seyreklik={info['sparsity']:.2%}"
    )
    print(f"  normalize={normalize}  distance={distance}  meta={meta}")

    # ── 2. Fitness ─────────────────────────────────────────────────────
    print("\n  FitnessEvaluator hazırlanıyor...")
    fitness_eval = make_fitness_fn(clustering_matrix, K, distance_metric=distance)
    dim = K * n_items
    print(f"  dim = {K} × {n_items} = {dim}")
    lb, ub = (-4.0, 4.0) if normalize == "user-mean" else (1.0, 5.0)

    # ── 3. Algoritmalar ────────────────────────────────────────────────
    optimizer_map = {
        "doa": DOA(
            pop_size=pop_size,
            max_iter=max_iter,
            dim=dim,
            lb=lb,
            ub=ub,
            seed=seed,
        ),
        "pso": PSOBaseline(
            pop_size=pop_size,
            max_iter=max_iter,
            dim=dim,
            lb=lb,
            ub=ub,
            seed=seed,
        ),
        "kmeans": KMeansBaseline(
            rating_matrix=clustering_matrix,
            K=K,
            seed=seed,
        ),
    }

    opt = optimizer_map.get(meta.lower())
    if opt is None:
        raise ValueError("meta must be one of: doa, pso, kmeans")

    results = []
    row = _run_algo(
        name=opt.get_name(),
        optimizer=opt,
        fitness_eval=fitness_eval,
        train_matrix=train_matrix,
        clustering_matrix=clustering_matrix,
        test_ratings=test_ratings,
        K=K,
        n_items=n_items,
        top_k=top_k,
        N=N,
        results_dir=results_dir,
        distance_metric=distance,
        seed=seed,
    )
    results.append(row)

    # ── 4. Karşılaştırma tablosu ───────────────────────────────────────
    print(f"\n{'='*56}")
    print("  SONUÇLAR")
    print(f"{'='*56}\n")
    _table(results)

    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clustering-CF with metaheuristics")
    p.add_argument(
        "--mode",
        choices=["legacy", "assignments"],
        default="legacy",
        help="legacy=doa/pso/kmeans pipeline, assignments=modular assignment pipeline",
    )
    p.add_argument("--dataset", choices=["100k", "1m"], default="100k")
    p.add_argument("--K", type=int, default=DEFAULTS["K"])
    p.add_argument("--pop-size", type=int, default=DEFAULTS["pop_size"])
    p.add_argument("--max-iter", type=int, default=DEFAULTS["max_iter"])
    p.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    p.add_argument("--N", type=int, default=DEFAULTS["N"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument(
        "--meta",
        choices=["doa", "pso", "kmeans"],
        default=DEFAULTS["meta"],
        help="Çalıştırılacak metaheuristic algoritma",
    )
    p.add_argument(
        "--distance",
        choices=["pearson", "euclidean"],
        default=DEFAULTS["distance"],
        help="Kümeleme fitness/atama mesafesi",
    )
    p.add_argument(
        "--normalize",
        choices=["none", "user-mean"],
        default=DEFAULTS["normalize"],
        help="Kümeleme girdi normalizasyonu",
    )
    p.add_argument(
        "--assignment-path",
        default=None,
        help="Mode=assignments için assignments.npy yolu",
    )
    p.add_argument(
        "--gray-mask-path",
        default=None,
        help="Mode=assignments için gray_sheep_mask.npy yolu",
    )
    p.add_argument(
        "--pipeline-out-dir",
        default=None,
        help="Mode=assignments için çıktı klasörü",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.mode == "assignments":
        _run_assignment_mode(args)
        raise SystemExit(0)
    run(
        dataset=args.dataset,
        K=args.K,
        pop_size=args.pop_size,
        max_iter=args.max_iter,
        top_k=args.top_k,
        N=args.N,
        seed=args.seed,
        meta=args.meta,
        distance=args.distance,
        normalize=args.normalize,
    )
