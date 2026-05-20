"""
Cross-validation with LensKit for MovieLens-style datasets.

Datasets:
- ml-latest-small: ratings.csv (userId, movieId, rating, timestamp) — see data/ml-latest-small/README.txt
- ml-100k: u.data (tab-separated user item rating timestamp, 1-based ids)

Splitting follows LensKit user partitioning + per-user holdout (crossfold_users + SampleN):
https://lenskit.org/stable/guide/batch.html

LensKit modeller (ranking metrikleri — NDCG, Recall, RBP):
  python lenskit_cv.py --dataset ml-latest-small
  python lenskit_cv.py --dataset ml-100k --data-path data/ml-100k/u.data
  python lenskit_cv.py --dataset ml-latest-small --model als --folds 5 --holdout 5 --topn 20

Surprise modeller (rating tahmin metrikleri — MAE, RMSE):
  python lenskit_cv.py --dataset ml-100k --model svd
  python lenskit_cv.py --dataset ml-100k --model svd --n-factors 100 --n-epochs 20
  python lenskit_cv.py --dataset ml-100k --model svdpp
  python lenskit_cv.py --dataset ml-100k --model nmf --n-factors 15
  python lenskit_cv.py --dataset ml-100k --model nmf --n-factors 20 --biased
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from lenskit.als import BiasedMFConfig, BiasedMFScorer
from lenskit.basic import PopScorer
from lenskit.batch import BatchPipelineRunner
from lenskit.data import Dataset, from_interactions_df
from lenskit.metrics import NDCG, RBP, Recall, RunAnalysis
from lenskit.pipeline import topn_pipeline
from lenskit.splitting import SampleN, crossfold_users

SURPRISE_MODELS = {"svd", "svdpp", "nmf"}


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


def _load_surprise_dataset(name: str, data_path: Path | None) -> "surprise.Dataset":
    """u.data veya ratings.csv dosyasını Surprise formatına yükle."""
    try:
        from surprise import Dataset as SurpriseDataset, Reader
    except ImportError:
        print("Hata: 'surprise' kurulu değil. Kurmak için: pip install scikit-surprise", file=sys.stderr)
        sys.exit(1)

    if name == "ml-100k":
        u_data = data_path if data_path is not None else REPO_ROOT / "data" / "ml-100k" / "u.data"
        if not Path(u_data).is_file():
            raise FileNotFoundError(f"Missing {u_data}")
        reader = Reader(line_format="user item rating timestamp", sep="\t", rating_scale=(1, 5))
        return SurpriseDataset.load_from_file(str(u_data), reader=reader)

    if name == "ml-latest-small":
        ratings_path = (data_path / "ratings.csv") if data_path is not None else REPO_ROOT / "data" / "ml-latest-small" / "ratings.csv"
        if not Path(ratings_path).is_file():
            raise FileNotFoundError(f"Missing {ratings_path}")
        df = pd.read_csv(ratings_path)
        reader = Reader(rating_scale=(df["rating"].min(), df["rating"].max()))
        from surprise import Dataset as SurpriseDataset
        return SurpriseDataset.load_from_df(df[["userId", "movieId", "rating"]], reader)

    raise ValueError(f"Unknown dataset: {name}")


def run_surprise_cv(args) -> int:
    """Surprise SVD / SVD++ / NMF modellerini 5-fold CV ile çalıştır, MAE/RMSE raporla."""
    try:
        from surprise import SVD, SVDpp
        from surprise import NMF as SurpriseNMF
        from surprise.model_selection import cross_validate as surprise_cv
    except ImportError:
        print("Hata: 'surprise' kurulu değil. Kurmak için: pip install scikit-surprise", file=sys.stderr)
        return 1

    try:
        data = _load_surprise_dataset(args.dataset, args.data_path)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    n_factors = getattr(args, "n_factors", None)
    n_epochs  = getattr(args, "n_epochs",  None)
    biased    = getattr(args, "biased",    False)

    if args.model == "svd":
        kw = {}
        if n_factors is not None:
            kw["n_factors"] = n_factors
        if n_epochs is not None:
            kw["n_epochs"] = n_epochs
        algo = SVD(**kw)
        label = f"SVD(n_factors={kw.get('n_factors', 100)}, n_epochs={kw.get('n_epochs', 20)})"

    elif args.model == "svdpp":
        kw = {}
        if n_factors is not None:
            kw["n_factors"] = n_factors
        if n_epochs is not None:
            kw["n_epochs"] = n_epochs
        algo = SVDpp(**kw)
        label = f"SVD++(n_factors={kw.get('n_factors', 20)}, n_epochs={kw.get('n_epochs', 20)})"

    elif args.model == "nmf":
        kw = {"biased": biased}
        if n_factors is not None:
            kw["n_factors"] = n_factors
        if n_epochs is not None:
            kw["n_epochs"] = n_epochs
        algo = SurpriseNMF(**kw)
        label = f"NMF(n_factors={kw.get('n_factors', 15)}, biased={biased})"

    else:
        raise ValueError(f"Unknown Surprise model: {args.model}")

    print(f"Model  : {label}")
    print(f"Dataset: {args.dataset}")
    print(f"CV     : {args.folds}-fold")
    print()

    t0 = time.time()
    results = surprise_cv(
        algo,
        data,
        measures=["RMSE", "MAE"],
        cv=args.folds,
        verbose=True,
        n_jobs=1,
    )
    elapsed = time.time() - t0

    mae_vals  = results["test_mae"]
    rmse_vals = results["test_rmse"]
    fit_vals  = results["fit_time"]

    print(f"\nSonuçlar ({args.folds}-fold CV):")
    header = f"{'Model':<40} {'MAE':>8} {'RMSE':>8} {'Fit(s)':>8}"
    print(header)
    print("-" * len(header))
    for i, (mae, rmse, ft) in enumerate(zip(mae_vals, rmse_vals, fit_vals), 1):
        print(f"  Fold {i:<35} {mae:8.4f} {rmse:8.4f} {ft:8.2f}s")
    print("-" * len(header))
    print(f"  {'Ortalama':<38} {mae_vals.mean():8.4f} {rmse_vals.mean():8.4f} {fit_vals.mean():8.2f}s")
    print(f"  {'Std':<38} {mae_vals.std():8.4f} {rmse_vals.std():8.4f}")
    print(f"\nToplam süre: {elapsed:.1f}s")
    return 0


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
    p.add_argument(
        "--model",
        choices=["pop", "als", "svd", "svdpp", "nmf"],
        default="pop",
        help="pop/als → LensKit (ranking metrikleri); svd/svdpp/nmf → Surprise (MAE/RMSE)",
    )
    p.add_argument("--rng", type=int, default=42, help="Seed for CV shuffle and ALS.")
    # Surprise-özgü parametreler
    p.add_argument("--n-factors", type=int, default=None, metavar="K",
                   help="Surprise: gizli faktör sayısı (SVD=100, SVD++=20, NMF=15 varsayılan)")
    p.add_argument("--n-epochs", type=int, default=None, metavar="E",
                   help="Surprise: epoch sayısı (SVD/SVD++=20, NMF=50 varsayılan)")
    p.add_argument("--biased", action="store_true",
                   help="Surprise NMF: bias terimleri ekle (μ + b_u + b_i + q·p)")
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel batch jobs (LensKit BatchPipelineRunner); 1 is safest on Windows.",
    )
    args = p.parse_args(argv)

    if args.model in SURPRISE_MODELS:
        return run_surprise_cv(args)

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
