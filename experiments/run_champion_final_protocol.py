"""
Champion protokol — final atama + eval + karşılaştırma tablosu.

Sweep sonuçlarından seçilen ayarlar:
  K=10, WNMF50, FCM m=1.5, fuzzy+mkpp, train-only fold=1
  Tahmin: soft=0.1, pearson, min_common=5 (MAE odaklı)

Varyantlar:
  nogs  — --no-gray-sheep  -> mealpy/results/assignments/
  lof   — --lof            -> mealpy/results/assignments_lof/

  python experiments/run_champion_final_protocol.py --phase status
  python experiments/run_champion_final_protocol.py --phase assign --jobs 4
  python experiments/run_champion_final_protocol.py --phase eval
  python experiments/run_champion_final_protocol.py --phase eval-cv5
  python experiments/run_champion_final_protocol.py --phase all --jobs 4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "wnmf"))

GEN = REPO / "mealpy" / "generate_assignments.py"
ASSIGN_NOGS = REPO / "mealpy" / "results" / "assignments" / "ml100k"
ASSIGN_LOF = REPO / "mealpy" / "results" / "assignments_lof" / "ml100k"
OUT_CSV = REPO / "results" / "champion_final_protocol_k10.csv"
OUT_MD = REPO / "results" / "champion_final_protocol_k10.md"
OUT_CSV_CV5 = REPO / "results" / "champion_final_protocol_k10_cv5.csv"

K = 10
WNMF_DIM = 50
FCM_M = 1.5
FOLD = 1
FOLDS_CV5 = [1, 2, 3, 4, 5]

ALGOS = ["HA_AVOAHGS", "B1_HHO", "B_AVOA", "LIT_PSO", "LIT_GWO"]
ALGO_LABELS = {
    "HA_AVOAHGS": "HA",
    "B1_HHO": "B1",
    "B_AVOA": "B_AVOA",
    "LIT_PSO": "LIT_PSO",
    "LIT_GWO": "LIT_GWO",
}

# Sweep-optimal predictor (MAE); NDCG profili tabloda opsiyonel sütun
PRED_MAE = Namespace(
    similarity="pearson",
    min_common=5,
    soft_membership_threshold=0.1,
)
PRED_NDCG = Namespace(
    similarity="cosine",
    min_common=4,
    soft_membership_threshold=0.1,
)


def folder_suffix(*, nogs: bool) -> str:
    gs = "_nogs" if nogs else ""
    return (
        f"_fuzzy_imkpp{gs}_trainonly_rand_wnmfep50_none_k{K}"
        f"_m{int(round(FCM_M * 10))}"
    )


def assign_dir(algo: str, *, nogs: bool) -> Path:
    root = ASSIGN_NOGS if nogs else ASSIGN_LOF
    return root / f"{algo}{folder_suffix(nogs=nogs)}"


def has_assign(algo: str, *, nogs: bool) -> bool:
    return (assign_dir(algo, nogs=nogs) / "assignments.npy").is_file()


def _gen_cmd(*, nogs: bool, jobs: int, skip_existing: bool, fold: int) -> list[str]:
    cmd = [
        sys.executable, str(GEN),
        "--dataset", "100k",
        "--algo", *ALGOS,
        "--no-prune",
        "--preprocess", "none",
        "--feature-extraction", "wnmf",
        "--svd-components", str(WNMF_DIM),
        "--wnmf-epochs", "50",
        "--init-mode", "mkpp",
        "--cluster-metric", "fuzzy",
        "--fcm-m", str(FCM_M),
        "--fcm-m-suffix",
        "--fitness", "wcss",
        "--cluster-objective", "multi",
        "--train-only", "--eval-split", "random", "--fold", str(fold),
        "--k", str(K),
        "--jobs", str(jobs),
    ]
    if nogs:
        cmd.append("--no-gray-sheep")
    else:
        cmd.append("--lof")
    if skip_existing:
        cmd.append("--skip-existing")
    return cmd


def phase_assign(jobs: int, skip_existing: bool) -> int:
    rc = 0
    for nogs, label in [(True, "nogs"), (False, "lof")]:
        cmd = _gen_cmd(
            nogs=nogs, jobs=jobs, skip_existing=skip_existing, fold=FOLD,
        )
        print(f"\n=== ASSIGN ({label}) ===", flush=True)
        print(" ".join(cmd), flush=True)
        rc = rc or subprocess.run(cmd, cwd=str(REPO)).returncode
    return rc


def _cluster_stats(labels: np.ndarray) -> dict:
    sizes = sorted(Counter(labels.astype(int).tolist()).values())
    if not sizes:
        return {}
    return {
        "n_active_clusters": len(sizes),
        "cluster_std": float(np.std(sizes)),
        "singletons": sum(1 for s in sizes if s == 1),
        "n_gray": 0,
    }


def _eval_one(
    train,
    test,
    n_items,
    n_users,
    algo: str,
    adir: Path,
    pred_args: Namespace,
) -> dict:
    from wnmf.wnmf_experiment import (
        _align_assignment_bundle,
        _cluster_avg_predict_kwargs,
        _nearest_centroid_bundle,
        load_assignment,
        load_memberships,
        run_cluster_average,
    )

    assignments, gray_mask = load_assignment(str(adir))
    memberships = load_memberships(str(adir))
    assignments, gray_mask, memberships, _ = _align_assignment_bundle(
        assignments, gray_mask, memberships, None,
        n_users_expected=n_users, algo_label=algo, assign_dir=str(adir),
    )
    nc = _nearest_centroid_bundle(None, str(adir), assignments)
    t0 = time.time()
    r = run_cluster_average(
        train, test, assignments, gray_mask, memberships, n_items, algo,
        **_cluster_avg_predict_kwargs(pred_args),
        **nc,
        top_n=10,
        relevance_threshold=4.0,
        assign_dir=str(adir),
    )
    st = _cluster_stats(assignments)
    st["n_gray"] = int(gray_mask.sum()) if gray_mask is not None else 0
    return {
        "mae": r["mae"],
        "rmse": r["rmse"],
        "ndcg_at_10": r["ndcg_at_10"],
        "precision_at_10": r["precision_at_10"],
        "recall_at_10": r["recall_at_10"],
        "gray_mae": r.get("gray_mae"),
        "white_mae": r.get("white_mae"),
        "eval_seconds": round(time.time() - t0, 1),
        **st,
    }


def _eval_baselines(train, test, k: int, wnmf_dim: int) -> list[dict]:
    from baseline_clustering import PCAKMeans, PCASOM, SOMCluster, KMeansCluster
    from wnmf.meta_dual_cf import _baseline_cluster_avg_user

    adir = assign_dir("HA_AVOAHGS", nogs=True)
    uf = adir / "user_features.npy"
    if not uf.is_file():
        uf = next(
            (assign_dir(a, nogs=True) / "user_features.npy" for a in ALGOS
             if (assign_dir(a, nogs=True) / "user_features.npy").is_file()),
            None,
        )
    if uf is None or not uf.is_file():
        print("SKIP baseline: user_features.npy yok (önce assign çalıştırın)", flush=True)
        return []

    X = np.load(uf)
    grid = max(4, int(np.ceil(np.sqrt(k * 2))))
    factories = [
        ("KMeans", lambda: KMeansCluster(n_clusters=k, random_state=42)),
        ("PCA-KMeans", lambda: PCAKMeans(n_clusters=k, pca_components=20, random_state=42)),
        ("SOM-Cluster", lambda: SOMCluster(
            n_clusters=k, grid_size=grid, n_epochs=50, random_state=42,
        )),
        ("PCA-SOM", lambda: PCASOM(
            n_clusters=k, pca_components=20, grid_size=grid, n_epochs=50, random_state=42,
        )),
    ]
    rows = []
    for name, factory in factories:
        model = factory()
        model.fit(X, verbose=False)
        labels = model.get_labels()
        m = _baseline_cluster_avg_user(train, test, labels)
        rows.append({
            "variant": "baseline",
            "algo": name,
            "label": name,
            "predictor": "cluster_avg_hard",
            "k": k,
            "wnmf_dim": wnmf_dim,
            "fcm_m": None,
            "similarity": "—",
            "min_common": "—",
            "soft_threshold": "—",
            **{k2: m[k2] for k2 in (
                "mae", "rmse", "precision_at_10", "recall_at_10", "ndcg_at_10",
            )},
            "gray_mae": float("nan"),
            "white_mae": float("nan"),
            "n_gray": 0,
            "n_active_clusters": len(set(labels.tolist())),
            "cluster_std": float(np.std(list(Counter(labels.tolist()).values()))),
            "singletons": 0,
            "eval_seconds": None,
        })
    return rows


def phase_eval(include_ndcg_profile: bool = True) -> pd.DataFrame:
    from wnmf.wnmf_experiment import RANDOM_SEED, load_ratings_100k_all

    data = str(REPO / "data" / "ml-100k" / "u.data")
    train, test = load_ratings_100k_all(data, random_seed=RANDOM_SEED, fold=FOLD)
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1

    rows: list[dict] = []

    for nogs, variant in [(True, "nogs"), (False, "lof")]:
        for algo in ALGOS:
            adir = assign_dir(algo, nogs=nogs)
            if not (adir / "assignments.npy").is_file():
                print(f"SKIP {variant}/{algo}: atama yok", flush=True)
                continue
            m = _eval_one(train, test, n_items, n_users, algo, adir, PRED_MAE)
            rows.append({
                "variant": variant,
                "algo": algo,
                "label": ALGO_LABELS.get(algo, algo),
                "predictor": "cluster_avg",
                "k": K,
                "wnmf_dim": WNMF_DIM,
                "fcm_m": FCM_M,
                "similarity": PRED_MAE.similarity,
                "min_common": PRED_MAE.min_common,
                "soft_threshold": PRED_MAE.soft_membership_threshold,
                **m,
            })
            print(
                f"  [{variant}] {algo}: MAE={m['mae']:.4f} NDCG={m['ndcg_at_10']:.4f} "
                f"gray={m['n_gray']} ({m['eval_seconds']}s)",
                flush=True,
            )
            if include_ndcg_profile and nogs and algo == "HA_AVOAHGS":
                m2 = _eval_one(train, test, n_items, n_users, algo, adir, PRED_NDCG)
                rows.append({
                    "variant": "nogs",
                    "algo": algo,
                    "label": "HA (NDCG profil)",
                    "predictor": "cluster_avg",
                    "k": K,
                    "wnmf_dim": WNMF_DIM,
                    "fcm_m": FCM_M,
                    "similarity": PRED_NDCG.similarity,
                    "min_common": PRED_NDCG.min_common,
                    "soft_threshold": PRED_NDCG.soft_membership_threshold,
                    **m2,
                })

    rows.extend(_eval_baselines(train, test, K, WNMF_DIM))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    _write_md(df)
    print(f"\n-> {OUT_CSV}", flush=True)
    print(f"-> {OUT_MD}", flush=True)
    return df


def _write_md(df: pd.DataFrame) -> None:
    meta = df[df["variant"].isin(["nogs", "lof"])].copy()
    base = df[df["variant"] == "baseline"].copy()

    lines = [
        "# Champion protokol — K=10 final tablo",
        "",
        f"Protokol: fuzzy FCM m={FCM_M}, WNMF{WNMF_DIM}, mkpp, train-only fold={FOLD}",
        f"Tahmin: soft={PRED_MAE.soft_membership_threshold}, "
        f"{PRED_MAE.similarity}, min_common={PRED_MAE.min_common}",
        "",
        "## Meta-algoritmalar (nogs vs LOF)",
        "",
        "| variant | algo | MAE | RMSE | P@10 | R@10 | NDCG | gray | white_MAE |",
        "|---------|------|-----|------|------|------|------|------|-----------|",
    ]
    for _, r in meta.iterrows():
        if "NDCG profil" in str(r.get("label", "")):
            continue
        lines.append(
            f"| {r['variant']} | {r['label']} | {r['mae']:.4f} | {r['rmse']:.4f} | "
            f"{r['precision_at_10']:.4f} | {r['recall_at_10']:.4f} | "
            f"{r['ndcg_at_10']:.4f} | {int(r.get('n_gray', 0))} | "
            f"{r.get('white_mae', float('nan')):.4f} |"
        )

    if not base.empty:
        lines.extend([
            "",
            "## Baseline (hard cluster_avg, aynı WNMF U, fold=1)",
            "",
            "| method | MAE | RMSE | P@10 | R@10 | NDCG |",
            "|--------|-----|------|------|------|------|",
        ])
        for _, r in base.iterrows():
            lines.append(
                f"| {r['label']} | {r['mae']:.4f} | {r['rmse']:.4f} | "
                f"{r['precision_at_10']:.4f} | {r['recall_at_10']:.4f} | "
                f"{r['ndcg_at_10']:.4f} |"
            )

    best_mae = meta.loc[meta["mae"].idxmin()] if not meta.empty else None
    best_ndcg = meta.loc[meta["ndcg_at_10"].idxmax()] if not meta.empty else None
    if best_mae is not None:
        lines.extend([
            "",
            f"**En iyi MAE:** {best_mae['label']} ({best_mae['variant']}) = {best_mae['mae']:.4f}",
        ])
    if best_ndcg is not None:
        lines.append(
            f"**En iyi NDCG:** {best_ndcg['label']} ({best_ndcg['variant']}) = "
            f"{best_ndcg['ndcg_at_10']:.4f}"
        )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def phase_eval_cv5(include_lof: bool = False) -> pd.DataFrame:
    from wnmf.wnmf_experiment import RANDOM_SEED, load_ratings_100k_all

    data = str(REPO / "data" / "ml-100k" / "u.data")
    rows: list[dict] = []

    for fold in FOLDS_CV5:
        train, test = load_ratings_100k_all(data, random_seed=RANDOM_SEED, fold=fold)
        n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
        n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1
        variants = [("nogs", True)]
        if include_lof:
            variants.append(("lof", False))
        for variant, nogs in variants:
            for algo in ALGOS:
                adir = assign_dir(algo, nogs=nogs)
                if not (adir / "assignments.npy").is_file():
                    print(f"SKIP fold={fold} {variant}/{algo}", flush=True)
                    continue
                m = _eval_one(train, test, n_items, n_users, algo, adir, PRED_MAE)
                rows.append({
                    "fold": fold,
                    "variant": variant,
                    "algo": algo,
                    "label": ALGO_LABELS.get(algo, algo),
                    "predictor": "cluster_avg",
                    "k": K,
                    "wnmf_dim": WNMF_DIM,
                    "fcm_m": FCM_M,
                    "similarity": PRED_MAE.similarity,
                    "min_common": PRED_MAE.min_common,
                    "soft_threshold": PRED_MAE.soft_membership_threshold,
                    "assign_suffix": folder_suffix(nogs=nogs),
                    **m,
                })
                print(
                    f"  fold={fold} {variant} {algo}: MAE={m['mae']:.4f} "
                    f"NDCG={m['ndcg_at_10']:.4f}",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    agg = df.groupby(["variant", "algo", "label"], as_index=False).agg(
        k=("k", "first"),
        wnmf_dim=("wnmf_dim", "first"),
        fcm_m=("fcm_m", "first"),
        mae=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        precision_at_10=("precision_at_10", "mean"),
        precision_at_10_std=("precision_at_10", "std"),
        recall_at_10=("recall_at_10", "mean"),
        recall_at_10_std=("recall_at_10", "std"),
        ndcg_at_10=("ndcg_at_10", "mean"),
        ndcg_at_10_std=("ndcg_at_10", "std"),
    )
    for c in ["mae", "rmse", "precision_at_10", "recall_at_10", "ndcg_at_10"]:
        agg[c] = agg[c].round(4)
        agg[f"{c}_std"] = agg[f"{c}_std"].round(4)

    OUT_CSV_CV5.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV_CV5.with_name("champion_final_protocol_k10_cv5_folds.csv"), index=False)
    agg.to_csv(OUT_CSV_CV5, index=False)
    print(f"\n-> {OUT_CSV_CV5}", flush=True)
    print(agg[["label", "mae", "mae_std", "ndcg_at_10", "ndcg_at_10_std"]].to_string(index=False), flush=True)
    return agg


def phase_status() -> None:
    print(f"Champion: K={K} WNMF{WNMF_DIM} FCM m={FCM_M}", flush=True)
    print(f"  suffix: {folder_suffix(nogs=True)}", flush=True)
    for nogs, label in [(True, "nogs"), (False, "lof")]:
        miss = [a for a in ALGOS if not has_assign(a, nogs=nogs)]
        ok = [a for a in ALGOS if has_assign(a, nogs=nogs)]
        print(f"    {label}: OK={len(ok)}/5  missing={miss or '-'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=["status", "assign", "eval", "eval-cv5", "all"],
        default="status",
    )
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--with-lof", action="store_true", help="eval-cv5: LOF varyantını da değerlendir")
    args = ap.parse_args()

    phase_status()

    if args.phase in ("assign", "all"):
        rc = phase_assign(args.jobs, args.skip_existing)
        if rc != 0:
            sys.exit(rc)
        phase_status()

    if args.phase == "eval-cv5":
        phase_eval_cv5(include_lof=args.with_lof)
    elif args.phase in ("eval", "all"):
        df = phase_eval()
        if not df.empty:
            show = df[
                ["variant", "label", "mae", "rmse", "precision_at_10", "recall_at_10", "ndcg_at_10"]
            ]
            print("\n" + show.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
