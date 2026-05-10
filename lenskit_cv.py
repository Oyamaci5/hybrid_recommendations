"""
Cross-validation with LensKit for MovieLens-style datasets.

Datasets:
- ml-latest-small: ratings.csv (userId, movieId, rating, timestamp) — see data/ml-latest-small/README.txt
- ml-100k: u.data (tab-separated user item rating timestamp, 1-based ids)

Splitting follows LensKit user partitioning + per-user holdout (crossfold_users + SampleN):
https://lenskit.org/stable/guide/batch.html

Run examples:
  python lenskit_cv.py --dataset ml-latest-small
  python lenskit_cv.py --dataset ml-100k --data-path data/ml-100k/u.data
  python lenskit_cv.py --dataset ml-latest-small --model als --folds 5 --holdout 5 --topn 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from lenskit.als import BiasedMFConfig, BiasedMFScorer
from lenskit.basic import PopScorer
from lenskit.batch import BatchPipelineRunner
from lenskit.data import Dataset, from_interactions_df
from lenskit.metrics import NDCG, RBP, Recall, RunAnalysis
from lenskit.pipeline import topn_pipeline
from lenskit.splitting import SampleN, crossfold_users


REPO_ROOT = Path(__file__).resolve().parent


def _load_ml_latest_small(data_dir: Path) -> Dataset:
    ratings_path = data_dir / "ratings.csv"
    if not ratings_path.is_file():
        raise FileNotFoundError(f"Missing {ratings_path}")
    ratings = pd.read_csv(ratings_path)
    return from_interactions_df(
        ratings,
        item_col="movieId",
        rating_col="rating",
        timestamp_col="timestamp",
    )


def _load_ml100k_u_data(u_data_path: Path) -> Dataset:
    if not u_data_path.is_file():
        raise FileNotFoundError(f"Missing {u_data_path}")
    ratings = pd.read_csv(
        u_data_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    return from_interactions_df(
        ratings,
        user_col="user_id",
        item_col="item_id",
        rating_col="rating",
        timestamp_col="timestamp",
    )


def load_dataset(name: str, data_path: Path | None) -> tuple[Dataset, str]:
    if name == "ml-latest-small":
        root = data_path if data_path is not None else REPO_ROOT / "data" / "ml-latest-small"
        ds = _load_ml_latest_small(Path(root))
        return ds, str(root)
    if name == "ml-100k":
        u_data = data_path if data_path is not None else REPO_ROOT / "data" / "ml-100k" / "u.data"
        ds = _load_ml100k_u_data(Path(u_data))
        return ds, str(u_data)
    raise ValueError(f"Unknown dataset: {name}")


def build_pipeline(model_name: str, topn: int, rng: int):
    if model_name == "pop":
        return topn_pipeline(PopScorer(), n=topn)
    if model_name == "als":
        cfg = BiasedMFConfig(features=32, iterations=15, reg=0.1, rng=rng)
        return topn_pipeline(BiasedMFScorer(cfg), n=topn)
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_fold(pipe, split_test, topn: int, n_jobs: int | None) -> pd.Series:
    runner = BatchPipelineRunner(n_jobs=n_jobs)
    runner.recommend(n=topn)
    outs = runner.run(pipe, split_test)
    recs = outs.output("recommendations")

    analysis = RunAnalysis()
    analysis.add_metric(NDCG(k=topn))
    analysis.add_metric(RBP())
    analysis.add_metric(Recall(k=topn))

    result = analysis.measure(recs, split_test)
    g = result.global_metrics()
    return g


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="LensKit user-wise cross-validation on MovieLens data.")
    p.add_argument(
        "--dataset",
        choices=["ml-latest-small", "ml-100k"],
        default="ml-latest-small",
        help="Which dataset layout to load.",
    )
    p.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override path: directory for ml-latest-small, or u.data file for ml-100k.",
    )
    p.add_argument("--folds", type=int, default=5, help="Number of user partitions (crossfold_users).")
    p.add_argument(
        "--holdout",
        type=int,
        default=5,
        help="Test ratings per user in each test-user group (SampleN).",
    )
    p.add_argument("--topn", type=int, default=20, help="Top-N list length for recommendations and metrics.")
    p.add_argument("--model", choices=["pop", "als"], default="pop")
    p.add_argument("--rng", type=int, default=42, help="Seed for CV shuffle and ALS.")
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel batch jobs (LensKit BatchPipelineRunner); 1 is safest on Windows.",
    )
    args = p.parse_args(argv)

    try:
        data, source = load_dataset(args.dataset, args.data_path)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        print(
            "İpucu: ml-100k için GroupLens'ten u.data indirip data/ml-100k/ altına koyun; "
            "ml-latest-small zaten ratings.csv ile gelir.",
            file=sys.stderr,
        )
        return 1

    n_users = data.user_count
    n_ratings = data.interactions().count()
    print(f"Kaynak: {source}")
    print(f"Kullanıcı: {n_users:,}, etkileşim: {n_ratings:,}")

    holdout = SampleN(args.holdout, rng=args.rng + 1)
    fold_metrics: list[pd.Series] = []

    for fold_idx, split in enumerate(crossfold_users(data, args.folds, holdout, rng=args.rng)):
        pipe = build_pipeline(args.model, args.topn, args.rng)
        pipe.train(split.train)
        metrics = evaluate_fold(pipe, split.test, args.topn, args.n_jobs if args.n_jobs > 0 else None)
        metrics.name = f"fold_{fold_idx + 1}"
        fold_metrics.append(metrics)
        print(f"Fold {fold_idx + 1}/{args.folds}: {metrics.to_dict()}")

    summary = pd.DataFrame(fold_metrics)
    print("\nOrtalama (fold üzerinden):")
    print(summary.mean(numeric_only=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
