"""
HA_AVOAHGS official fold-1 onizleme: atama + kume dagilimi + tahmin.

  python experiments/test_ha_official_preview.py
  python experiments/test_ha_official_preview.py --k 5 --fold 1 --skip-assign
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from argparse import Namespace
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.run_euc_kmref_k_sweep import (
    ASSIGN_ROOT,
    EVAL_SPLIT,
    GEN,
    MIN_COMMON,
    SIM,
    WNMF_EPOCHS,
    assign_dir,
    cluster_stats,
    load_official_fold,
)

ALGO = "HA_AVOAHGS"
DEFAULT_K = 7
DEFAULT_FOLD = 1
DEFAULT_WNMF_DIM = 20


def run_assign(k: int, fold: int, wnmf_dim: int, jobs: int) -> int:
    cmd = [
        sys.executable, "-u", str(GEN),
        "--dataset", "100k",
        "--algo", ALGO,
        "--no-prune", "--no-gray-sheep",
        "--preprocess", "none",
        "--feature-extraction", "wnmf",
        "--svd-components", str(wnmf_dim),
        "--wnmf-epochs", str(WNMF_EPOCHS),
        "--init-mode", "mkpp",
        "--cluster-metric", "euclidean",
        "--fitness", "wcss",
        "--cluster-objective", "multi",
        "--train-only", "--eval-split", EVAL_SPLIT, "--fold", str(fold),
        "--k", str(k),
        "--kmeans-refine-overwrite",
        "--jobs", str(jobs),
    ]
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def print_cluster_distribution(
    assignments: np.ndarray, gray_mask: np.ndarray, k: int, fold: int,
) -> None:
    c = Counter(assignments.astype(int).tolist())
    sizes = sorted(c.values(), reverse=True)
    st = cluster_stats(assignments)

    print("\n" + "=" * 60)
    print(f"KUME DAGILIMI  algo={ALGO}  K={k}  (official u{fold}.base atamasi)")
    print("=" * 60)
    print(f"  Toplam kullanici     : {len(assignments)}")
    print(f"  Gray sheep (mask)    : {int(gray_mask.sum())} / {len(gray_mask)}")
    print(f"  Aktif kume sayisi    : {st['n_active_clusters']}")
    print(f"  Min / max / std size : {st['cluster_min']} / {st['cluster_max']} / {st['cluster_std']:.1f}")
    print(f"  Singleton kume       : {st['singletons']}")
    print(f"\n  {'Cluster':>8}  {'Size':>6}  {'Bar'}")
    print("  " + "-" * 50)
    max_sz = max(sizes) if sizes else 1
    for cid in sorted(c.keys()):
        sz = c[cid]
        bar = "#" * max(1, int(40 * sz / max_sz))
        print(f"  {cid:8d}  {sz:6d}  {bar}")
    print(f"\n  Size list (desc): {sizes}")


def run_predictions(
    adir: Path,
    assignments: np.ndarray,
    gray_mask: np.ndarray,
    fold: int,
    k: int,
) -> None:
    from wnmf.wnmf_experiment import (
        _align_assignment_bundle,
        _cluster_avg_predict_kwargs,
        _knn_centroid_bundle,
        _nearest_centroid_bundle,
        load_memberships,
        load_user_features,
        run_cluster_average,
        run_cluster_knn,
    )

    train, test = load_official_fold(fold)
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1
    eval_args = Namespace(similarity=SIM, min_common=MIN_COMMON)

    memberships = load_memberships(str(adir))
    uf = load_user_features(str(adir), len(assignments))
    assignments, gray_mask, memberships, uf = _align_assignment_bundle(
        assignments, gray_mask, memberships, uf,
        n_users_expected=n_users, algo_label=ALGO, assign_dir=str(adir),
    )
    st = cluster_stats(assignments)
    k_min = max(1, int(st["cluster_min"]))
    nc_avg = _nearest_centroid_bundle(None, str(adir), assignments)
    nc_knn = _knn_centroid_bundle(None, str(adir), assignments, knn_mode="cluster")
    common = dict(top_n=10, relevance_threshold=4.0, assign_dir=str(adir))

    print("\n" + "=" * 60)
    print(f"TAHMIN ASAMASI  u{fold}.base -> u{fold}.test  train={len(train)} test={len(test)}")
    print("=" * 60)

    rows = []
    for pred_name, fn in (
        ("cluster_avg", lambda: run_cluster_average(
            train, test, assignments, gray_mask, memberships, n_items, ALGO,
            **_cluster_avg_predict_kwargs(eval_args), **nc_avg, **common,
        )),
        ("cluster_avg_hard", lambda: run_cluster_average(
            train, test, assignments, gray_mask, memberships, n_items, ALGO,
            cluster_avg_hard=True, **nc_avg, **common,
        )),
    ):
        t0 = time.time()
        r = fn()
        rows.append((pred_name, 0, r, time.time() - t0))

    for knn_k in (k_min, 30):
        t0 = time.time()
        r = run_cluster_knn(
            train, test, assignments, gray_mask, memberships, n_items, ALGO,
            user_features=uf, similarity=SIM, min_common=MIN_COMMON,
            k_neighbors=knn_k, cluster_knn_backend="native",
            **nc_knn, **common,
        )
        rows.append((f"cluster_knn_native", knn_k, r, time.time() - t0))

    print(f"\n  {'Predictor':<28} {'knn_k':>5}  {'MAE':>7}  {'RMSE':>7}  {'NDCG@10':>8}  {'Cov@10':>7}  {'s':>5}")
    print("  " + "-" * 72)
    for pred, knn_k, r, sec in rows:
        print(
            f"  {pred:<28} {knn_k:5d}  {r['mae']:7.4f}  {r['rmse']:7.4f}  "
            f"{r['ndcg_at_10']:8.4f}  {r.get('coverage_at_10', float('nan')):7.4f}  {sec:5.1f}"
        )
    print(f"\n  Atama klasoru: {adir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--fold", type=int, default=DEFAULT_FOLD)
    ap.add_argument("--wnmf-dim", type=int, default=DEFAULT_WNMF_DIM)
    ap.add_argument("--skip-assign", action="store_true", help="Mevcut atamayi kullan")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    print("=" * 60)
    print("HA_AVOAHGS OFFICIAL ONIZLEME")
    print("=" * 60)
    print(f"  eval_split : {EVAL_SPLIT}")
    print(f"  fold       : {args.fold}  (atama=u{args.fold}.base, eval=u{args.fold}.base/test)")
    print(f"  K          : {args.k}")
    print(f"  preprocess : none  |  wnmf={args.wnmf_dim} ep={WNMF_EPOCHS}  |  euc + kmref")

    adir = assign_dir(ALGO, args.k, wnmf_dim=args.wnmf_dim, assign_fold=args.fold)
    if adir is None or not (adir / "assignments.npy").is_file():
        if args.skip_assign:
            sys.exit(f"Atama yok: {ASSIGN_ROOT / (ALGO + '...')}")
        print(f"\n>>> Atama uretiliyor (HA_AVOAHGS K={args.k} fold={args.fold})...", flush=True)
        rc = run_assign(args.k, args.fold, args.wnmf_dim, args.jobs)
        if rc != 0:
            sys.exit(rc)
        adir = assign_dir(ALGO, args.k, wnmf_dim=args.wnmf_dim, assign_fold=args.fold)
    else:
        print(f"\n>>> Mevcut atama kullaniliyor: {adir.name}", flush=True)

    if adir is None:
        sys.exit("Atama klasoru bulunamadi.")

    from wnmf.wnmf_experiment import load_assignment

    assignments, gray_mask = load_assignment(str(adir))
    print_cluster_distribution(assignments, gray_mask, args.k, args.fold)
    run_predictions(adir, assignments, gray_mask, args.fold, args.k)
    print("\nOnizleme tamam.")


if __name__ == "__main__":
    main()
