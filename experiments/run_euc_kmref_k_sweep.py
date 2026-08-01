"""
Strict CV5 (official) — euc / imkpp / nogs / none / WNMF20 / kmref, K=3..14

Fold F (1..5):
  Atama : uF.base (train-only) -> WNMF20 -> meta algo -> kmref
  Eval  : uF.base / uF.test + assignments from fold F

  python experiments/run_euc_kmref_k_sweep.py --phase all --cv5 --jobs 4 --overwrite
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

GEN = REPO / "mealpy" / "generate_assignments.py"
ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments" / "ml100k"
ML100K_DIR = REPO / "data" / "ml-100k"

DEFAULT_WNMF_DIM = 20
WNMF_EPOCHS = 50
EVAL_SPLIT = "official"
CV_FOLDS = [1, 2, 3, 4, 5]
K_LIST = list(range(3, 15))
KNN_K_FIXED = 30
SIM = "cosine"
MIN_COMMON = 3
PREDICTORS_PER_CELL = 8

ALGOS = ["HA_AVOAHGS", "B1_HHO", "B_AVOA", "LIT_PSO", "LIT_GWO"]

PREDICTORS = [
    "cluster_avg",
    "cluster_avg_hard",
    "cluster_knn_native (k=cluster_min, 30)",
    "cluster_knn_surprise_baseline (k=cluster_min, 30)",
    "cluster_knn_with_means (k=cluster_min, 30)",
]


def algo_label(algo: str) -> str:
    return algo


def _trainonly_suffix(assign_fold: int) -> str:
    return f"_trainonly_{EVAL_SPLIT}_f{int(assign_fold)}"


def _out_csv(wnmf_dim: int, *, cv5: bool = False, eval_fold: int | None = None) -> Path:
    if cv5:
        return REPO / "results" / f"euc_kmref_k_sweep_preds_w{wnmf_dim}_cv5_official.csv"
    if eval_fold is not None:
        return REPO / "results" / f"euc_kmref_k_sweep_preds_w{wnmf_dim}_official_f{eval_fold}.csv"
    return REPO / "results" / f"euc_kmref_k_sweep_preds_w{wnmf_dim}_official.csv"


def assign_dir(
    algo: str,
    k: int,
    *,
    wnmf_dim: int,
    assign_fold: int,
) -> Path | None:
    suf = _trainonly_suffix(assign_fold)
    candidates = [
        ASSIGN_ROOT / f"{algo}_euc_imkpp_nogs{suf}_wnmfep{WNMF_EPOCHS}_none_wnmf{wnmf_dim}_k{k}_kmref",
        ASSIGN_ROOT / f"{algo}_euc_imkpp_nogs{suf}_none_wnmf{wnmf_dim}_k{k}_kmref",
    ]
    for p in candidates:
        if (p / "assignments.npy").is_file():
            return p
    pat = f"{algo}_euc_imkpp_nogs{suf}*wnmf{wnmf_dim}_k{k}_kmref"
    for p in sorted(ASSIGN_ROOT.glob(pat)):
        if (p / "assignments.npy").is_file():
            return p
    return None


def has_assign(algo: str, k: int, *, wnmf_dim: int, assign_fold: int) -> bool:
    adir = assign_dir(algo, k, wnmf_dim=wnmf_dim, assign_fold=assign_fold)
    return adir is not None and (adir / "assignments.npy").is_file()


def count_assignments(ks: Sequence[int], wnmf_dim: int, assign_fold: int) -> int:
    return sum(
        1 for k in ks for algo in ALGOS
        if has_assign(algo, k, wnmf_dim=wnmf_dim, assign_fold=assign_fold)
    )


def load_official_fold(fold: int):
    from wnmf.wnmf_utils import load_ratings_100k

    base = str(ML100K_DIR / "u1.base")
    test = str(ML100K_DIR / "u1.test")
    return load_ratings_100k(base, test, fold=fold)


def print_protocol_summary(ks: Sequence[int], wnmf_dim: int, eval_folds: Sequence[int]) -> None:
    n_k = len(ks)
    n_fold = len(eval_folds)
    n_assign = n_k * len(ALGOS) * n_fold
    n_eval = n_assign * PREDICTORS_PER_CELL
    print("=" * 70)
    print("STRICT CV5 — OFFICIAL ML-100K")
    print("=" * 70)
    print(f"  Veri       : u{{N}}.base / u{{N}}.test  (official 5-fold)")
    print(f"  Foldlar    : {list(eval_folds)}")
    for f in eval_folds:
        print(f"    fold {f}: atama=u{f}.base  eval=u{f}.base/u{f}.test")
    print(f"  Preprocess : none  |  prune: kapali  |  gray sheep: kapali (nogs)")
    print(f"  WNMF       : dim={wnmf_dim}, epochs={WNMF_EPOCHS}, init=inmed/mkpp")
    print(f"  Kumeleme   : euclidean (euc), imkpp, wcss/multi, kmref")
    print(f"  K          : {list(ks)}")
    print(f"  Algoritmalar: {', '.join(ALGOS)}")
    print(f"  Predictors : {PREDICTORS_PER_CELL}/hucre")
    print(f"  Atama      : {n_assign} klasor ({n_k}x{len(ALGOS)}x{n_fold})")
    print(f"  Eval hedef : {n_eval} CSV satir")
    print(f"  Cikti      : {_out_csv(wnmf_dim, cv5=True).name}")
    print("=" * 70)
    for f in eval_folds:
        print(f"  fold {f} assignments: {count_assignments(ks, wnmf_dim, f)}/{n_k * len(ALGOS)}")


def cluster_stats(assignments: np.ndarray) -> dict:
    sizes = sorted(Counter(assignments.astype(int).tolist()).values())
    if not sizes:
        return {
            "n_active_clusters": 0,
            "cluster_min": 0,
            "cluster_max": 0,
            "cluster_std": float("nan"),
            "singletons": 0,
        }
    return {
        "n_active_clusters": len(sizes),
        "cluster_min": min(sizes),
        "cluster_max": max(sizes),
        "cluster_std": float(np.std(sizes)),
        "singletons": sum(1 for s in sizes if s == 1),
    }


def phase_assign(
    jobs: int,
    ks: Sequence[int],
    skip_existing: bool,
    *,
    wnmf_dim: int,
    assign_fold: int,
) -> int:
    cmd = [
        sys.executable, "-u", str(GEN),
        "--dataset", "100k",
        "--algo", *ALGOS,
        "--no-prune", "--no-gray-sheep",
        "--preprocess", "none",
        "--feature-extraction", "wnmf",
        "--svd-components", str(wnmf_dim),
        "--wnmf-epochs", str(WNMF_EPOCHS),
        "--init-mode", "mkpp",
        "--cluster-metric", "euclidean",
        "--fitness", "wcss",
        "--cluster-objective", "multi",
        "--train-only", "--eval-split", EVAL_SPLIT, "--fold", str(assign_fold),
        "--k", *[str(k) for k in ks],
        "--kmeans-refine-overwrite",
        "--jobs", str(jobs),
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def _load_done_keys(out_csv: Path, eval_fold: int, wnmf_dim: int) -> set[tuple]:
    if not out_csv.is_file():
        return set()
    df = pd.read_csv(out_csv)
    if df.empty:
        return set()
    sub = df[(df.get("fold", eval_fold) == eval_fold) & (df.get("wnmf_dim", wnmf_dim) == wnmf_dim)]
    return {
        (int(r.fold), int(r.k), str(r.algo), str(r.predictor), int(r.knn_k))
        for r in sub.itertuples()
    }


def _load_done_algo_k(out_csv: Path, eval_fold: int, wnmf_dim: int) -> set[tuple[int, str]]:
    if not out_csv.is_file():
        return set()
    df = pd.read_csv(out_csv)
    if df.empty:
        return set()
    sub = df[(df.get("fold", eval_fold) == eval_fold) & (df.get("wnmf_dim", wnmf_dim) == wnmf_dim)]
    done: set[tuple[int, str]] = set()
    for (k, algo), g in sub.groupby(["k", "algo"]):
        if len(g) >= PREDICTORS_PER_CELL:
            done.add((int(k), str(algo)))
    return done


def _append_rows(rows: List[dict], out_csv: Path) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.is_file():
        old = pd.read_csv(out_csv)
        key = ["fold", "k", "algo", "predictor", "knn_k", "wnmf_dim", "kmref"]
        merged = pd.concat([old, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=key, keep="last")
        merged = merged.sort_values(
            ["fold", "k", "algo", "predictor", "knn_k"],
        ).reset_index(drop=True)
        merged.to_csv(out_csv, index=False)
        return merged
    df.to_csv(out_csv, index=False)
    return df


def phase_eval(
    ks: Sequence[int],
    *,
    wnmf_dim: int,
    eval_fold: int,
    out_csv: Path,
    skip_existing: bool = False,
) -> pd.DataFrame:
    from wnmf.wnmf_experiment import (
        _align_assignment_bundle,
        _cluster_avg_predict_kwargs,
        _knn_centroid_bundle,
        _nearest_centroid_bundle,
        load_assignment,
        load_memberships,
        load_user_features,
        run_cluster_average,
        run_cluster_knn,
    )

    train, test = load_official_fold(eval_fold)
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1
    eval_args = Namespace(similarity=SIM, min_common=MIN_COMMON)
    done_cells = _load_done_algo_k(out_csv, eval_fold, wnmf_dim) if skip_existing else set()
    done_keys = _load_done_keys(out_csv, eval_fold, wnmf_dim) if skip_existing else set()
    if done_cells:
        print(f"  eval fold {eval_fold}: {len(done_cells)} tam hucre atlanacak", flush=True)

    all_rows: List[dict] = []
    for k in ks:
        for algo in ALGOS:
            if (k, algo) in done_cells:
                print(f"SKIP eval {algo} K={k} fold={eval_fold} (tam hucre CSV'de)", flush=True)
                continue
            batch: List[dict] = []
            adir = assign_dir(algo, k, wnmf_dim=wnmf_dim, assign_fold=eval_fold)
            if adir is None or not (adir / "assignments.npy").is_file():
                print(f"SKIP eval {algo} K={k} fold={eval_fold} (atama yok)", flush=True)
                continue

            assignments, gray_mask = load_assignment(str(adir))
            memberships = load_memberships(str(adir))
            uf = load_user_features(str(adir), len(assignments))
            assignments, gray_mask, memberships, uf = _align_assignment_bundle(
                assignments, gray_mask, memberships, uf,
                n_users_expected=n_users, algo_label=algo, assign_dir=str(adir),
            )
            st = cluster_stats(assignments)
            k_min = max(1, int(st["cluster_min"]))
            nc_avg = _nearest_centroid_bundle(None, str(adir), assignments)
            nc_knn = _knn_centroid_bundle(None, str(adir), assignments, knn_mode="cluster")
            common = dict(top_n=10, relevance_threshold=4.0, assign_dir=str(adir))

            def _base_row(predictor: str, r: dict, t0: float, *, knn_k: int = 0) -> dict:
                return {
                    "protocol": "euc_imkpp",
                    "assign_mode": "strict_cv_official",
                    "eval_split": EVAL_SPLIT,
                    "cv_split": f"u{eval_fold}.base/u{eval_fold}.test",
                    "fold": eval_fold,
                    "wnmf_dim": wnmf_dim,
                    "wnmf_epochs": WNMF_EPOCHS,
                    "kmref": True,
                    "k": k,
                    "algo": algo,
                    "label": algo_label(algo),
                    "predictor": predictor,
                    "knn_k": int(knn_k),
                    "mae": r["mae"],
                    "rmse": r["rmse"],
                    "ndcg_at_10": r["ndcg_at_10"],
                    "precision_at_10": r["precision_at_10"],
                    "recall_at_10": r["recall_at_10"],
                    "coverage_at_10": r.get("coverage_at_10", float("nan")),
                    "assign_suffix": adir.name[len(algo):],
                    "eval_seconds": round(time.time() - t0, 1),
                    **st,
                }

            for pred_name, fn in (
                ("cluster_avg", lambda: run_cluster_average(
                    train, test, assignments, gray_mask, memberships, n_items, algo,
                    **_cluster_avg_predict_kwargs(eval_args), **nc_avg, **common,
                )),
                ("cluster_avg_hard", lambda: run_cluster_average(
                    train, test, assignments, gray_mask, memberships, n_items, algo,
                    cluster_avg_hard=True, **nc_avg, **common,
                )),
            ):
                key = (eval_fold, k, algo, pred_name, 0)
                if key in done_keys:
                    print(f"  SKIP {algo} K={k} {pred_name} fold={eval_fold}", flush=True)
                    continue
                t0 = time.time()
                r = fn()
                row = _base_row(pred_name, r, t0)
                batch.append(row)
                _append_rows([row], out_csv)
                print(
                    f"  fold={eval_fold} {algo} K={k} {pred_name}: "
                    f"MAE={r['mae']:.4f} NDCG={r['ndcg_at_10']:.4f} "
                    f"Cov@10={r.get('coverage_at_10', float('nan')):.4f}",
                    flush=True,
                )

            for knn_k in (k_min, KNN_K_FIXED):
                for backend, variant, pred_base in (
                    ("native", None, "cluster_knn_native"),
                    ("surprise", "baseline", "cluster_knn_surprise_baseline"),
                    ("surprise", "withmeans", "cluster_knn_with_means"),
                ):
                    kw = dict(
                        user_features=uf,
                        similarity=SIM,
                        min_common=MIN_COMMON,
                        k_neighbors=knn_k,
                        cluster_knn_backend=backend,
                        **nc_knn,
                        **common,
                    )
                    if backend == "surprise":
                        kw["surprise_knn_variant"] = variant
                    key = (eval_fold, k, algo, pred_base, int(knn_k))
                    if key in done_keys:
                        continue
                    t0 = time.time()
                    r = run_cluster_knn(
                        train, test, assignments, gray_mask, memberships,
                        n_items, algo, **kw,
                    )
                    row = _base_row(pred_base, r, t0, knn_k=knn_k)
                    batch.append(row)
                    _append_rows([row], out_csv)
                    print(
                        f"  fold={eval_fold} {algo} K={k} {pred_base} knn_k={knn_k}: "
                        f"MAE={r['mae']:.4f} NDCG={r['ndcg_at_10']:.4f} "
                        f"Cov@10={r.get('coverage_at_10', float('nan')):.4f}",
                        flush=True,
                    )

            all_rows.extend(batch)

    if out_csv.is_file():
        return pd.read_csv(out_csv)
    return pd.DataFrame(all_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict CV5 official euc/imkpp/wnmf20/kmref")
    ap.add_argument("--phase", choices=["assign", "eval", "all"], default="all")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Atamalari yeniden uret; eval CSV sifirla",
    )
    ap.add_argument("--cv5", action="store_true")
    ap.add_argument("--fold", type=int, default=None, help="Tek fold (1-5)")
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--wnmf-dim", type=int, default=DEFAULT_WNMF_DIM)
    ap.add_argument("--k", type=int, nargs="+", default=None)
    args = ap.parse_args()
    ks = args.k if args.k else K_LIST

    if args.cv5 or args.folds:
        eval_folds = args.folds if args.folds else CV_FOLDS
        cv5 = True
    elif args.fold is not None:
        eval_folds = [args.fold]
        cv5 = False
    else:
        eval_folds = CV_FOLDS
        cv5 = True

    out_csv = _out_csv(args.wnmf_dim, cv5=cv5)
    assign_skip = args.skip_existing and not args.overwrite
    eval_skip = args.skip_existing and not args.overwrite

    if args.overwrite and out_csv.is_file() and args.phase in ("eval", "all"):
        backup = out_csv.with_suffix(".csv.bak")
        if backup.is_file():
            backup.unlink()
        out_csv.rename(backup)
        print(f"eval CSV yedek -> {backup.name}", flush=True)

    print_protocol_summary(ks, args.wnmf_dim, eval_folds)

    for eval_fold in eval_folds:
        if args.phase in ("assign", "all"):
            print(
                f"\n{'='*60}\nASSIGN fold {eval_fold} (u{eval_fold}.base train-only)\n{'='*60}",
                flush=True,
            )
            rc = phase_assign(
                args.jobs, ks, assign_skip,
                wnmf_dim=args.wnmf_dim, assign_fold=eval_fold,
            )
            if rc != 0:
                sys.exit(rc)

        if args.phase in ("eval", "all"):
            print(
                f"\n{'='*60}\nEVAL fold {eval_fold} (u{eval_fold}.base/u{eval_fold}.test)\n{'='*60}",
                flush=True,
            )
            target = out_csv if cv5 else _out_csv(args.wnmf_dim, eval_fold=eval_fold)
            df = phase_eval(
                ks,
                wnmf_dim=args.wnmf_dim,
                eval_fold=eval_fold,
                out_csv=target,
                skip_existing=eval_skip,
            )
            n_fold = (
                len(df[df["fold"] == eval_fold])
                if not df.empty and "fold" in df.columns else len(df)
            )
            print(f"  eval fold {eval_fold}: {n_fold} satir (toplam CSV {len(df)})", flush=True)

    if out_csv.is_file():
        df_all = pd.read_csv(out_csv)
        target_rows = len(ks) * len(ALGOS) * PREDICTORS_PER_CELL * len(eval_folds)
        print(f"\nBitti: {len(df_all)} / {target_rows} satir -> {out_csv}", flush=True)
        if cv5 and "fold" in df_all.columns:
            print(df_all.groupby("fold").size().to_string(), flush=True)


if __name__ == "__main__":
    main()
