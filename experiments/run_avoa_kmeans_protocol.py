"""
AVOA-K-MEANS vs PLAIN K-means — ML-100K official 5-fold protocol.

Pipeline:
  WNMF user features → AVOA centroid search (WCSS) → Lloyd K-means (init=AVOA)
  vs sklearn KMeans baseline (B0_KMEANS).
  Prediction: CalcAvgRating (--paper-mode) veya cluster-aware WNMF (--predictor sharedV).

Phases:
  python -m experiments.run_avoa_kmeans_protocol --phase status
  python -m experiments.run_avoa_kmeans_protocol --phase assign --fold 1
  python -m experiments.run_avoa_kmeans_protocol --phase verify --fold 1
  python -m experiments.run_avoa_kmeans_protocol --phase eval --fold 1
  python -m experiments.run_avoa_kmeans_protocol --phase eval --predictor sharedV --fold 1
  python -m experiments.run_avoa_kmeans_protocol --phase aggregate
  python -m experiments.run_avoa_kmeans_protocol --phase aggregate --predictor sharedV
  python -m experiments.run_avoa_kmeans_protocol --phase all
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MEALPY = REPO / "mealpy"
GEN = MEALPY / "generate_assignments.py"
EXP = REPO / "wnmf" / "wnmf_experiment.py"
ASSIGN_ROOT = MEALPY / "results" / "assignments"
ASSIGN_ML100K = ASSIGN_ROOT / "ml100k"
PARAMS_JSON = REPO / "experiments" / "avoa_kmeans_params.json"
OUT_VERIFY = REPO / "results" / "avoa_kmeans_protocol_verify.json"

PREDICTOR_CONFIG = {
    "paper": {
        "scenario": "calc_avg_rating",
        "wnmf_mode": "baselines",
        "extra_cmd": ["--paper-mode"],
        "file_tag": "baselines",
        "label": "CalcAvgRating (paper-mode)",
    },
    "sharedV": {
        "scenario": "cluster_sharedV",
        "wnmf_mode": "sharedV",
        "extra_cmd": ["--no-cluster-avg", "--no-cluster-knn"],
        "file_tag": "sharedV",
        "label": "Cluster-aware WNMF (sharedV)",
    },
}


def out_preds_path(predictor: str) -> Path:
    suf = "" if predictor == "paper" else f"_{predictor}"
    return REPO / "results" / f"avoa_kmeans_protocol_preds{suf}.csv"


def out_cv5_path(predictor: str) -> Path:
    suf = "" if predictor == "paper" else f"_{predictor}"
    return REPO / "results" / f"avoa_kmeans_protocol_cv5{suf}.csv"


def out_report_path(predictor: str) -> Path:
    suf = "" if predictor == "paper" else f"_{predictor}"
    return REPO / "results" / f"avoa_kmeans_protocol_report{suf}.md"

CV_FOLDS = [1, 2, 3, 4, 5]
DEFAULT_K = 10
DEFAULT_LATENT = 50
DEFAULT_EPOCH = 200
DEFAULT_POP = 50
DEFAULT_WNMF_EPOCHS = 50
PROTOCOL_TAG = "avoa_kmeans_protocol"

ALGO_PLAIN = "B0_KMEANS"
ALGO_AVOA = "B_AVOA"
ALGO_LABELS = {
    ALGO_PLAIN: "PLAIN_KMEANS",
    ALGO_AVOA: "AVOA_KMEANS",
}

METRIC_COLS = [
    "mae", "rmse", "precision_at_10", "recall_at_10", "ndcg_at_10",
]


def _load_params() -> dict:
    if PARAMS_JSON.is_file():
        return json.loads(PARAMS_JSON.read_text(encoding="utf-8"))
    return {}


def folder_suffix(fold: int, k: int, latent: int, *, kmref: bool) -> str:
    """generate_assignments --legacy-wnmf-suffix ile uyumlu klasör soneki (label hariç)."""
    base = (
        f"_euc_imkpp_nogs_trainonly_official_f{int(fold)}"
        f"_none_wnmf{int(latent)}_k{int(k)}_pwcss"
    )
    return f"{base}_kmref" if kmref else base


def assign_dir(algo: str, fold: int, k: int, latent: int, *, kmref: bool) -> Path:
    suf = folder_suffix(fold, k, latent, kmref=kmref)
    return ASSIGN_ML100K / f"{algo}{suf}"


def eval_assign_suffix(fold: int, k: int, latent: int) -> str:
    """wnmf_experiment --assign-suffix (AVOA kmref; B0 otomatik strip)."""
    return folder_suffix(fold, k, latent, kmref=True)


def base_assign_args(
    fold: int,
    k: int,
    latent: int,
    epoch: int,
    pop: int,
    wnmf_epochs: int,
) -> List[str]:
    return [
        "--dataset", "100k",
        "--no-prune", "--no-gray-sheep",
        "--preprocess", "none",
        "--feature-extraction", "wnmf",
        "--svd-components", str(int(latent)),
        "--wnmf-epochs", str(int(wnmf_epochs)),
        "--legacy-wnmf-suffix",
        "--init-mode", "mkpp",
        "--cluster-metric", "euclidean",
        "--fitness", "wcss",
        "--cluster-objective", "wcss",
        "--train-only", "--eval-split", "official",
        "--fold", str(int(fold)),
        "--k", str(int(k)),
        "--baseline-epoch", str(int(epoch)),
        "--pop-size", str(int(pop)),
    ]


VARIANTS = [
    {
        "label": "PLAIN_KMEANS",
        "algo": [ALGO_PLAIN],
        "extra": [],
        "kmref": False,
    },
    {
        "label": "AVOA_KMEANS",
        "algo": [ALGO_AVOA],
        "extra": ["--kmeans-refine-overwrite", "--kmeans-refine-iter", "300"],
        "kmref": True,
    },
]


def run_cmd(cmd: List[str], *, dry_run: bool = False) -> int:
    print("\n" + "=" * 72, flush=True)
    print(" ".join(cmd), flush=True)
    print("=" * 72, flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def phase_status(
    folds: Sequence[int],
    k: int,
    latent: int,
) -> None:
    print(f"Protocol: {PROTOCOL_TAG}")
    print(f"K={k}  L={latent}  folds={list(folds)}")
    for fold in folds:
        print(f"\n--- fold {fold} ---")
        for v in VARIANTS:
            d = assign_dir(v["algo"][0], fold, k, latent, kmref=v["kmref"])
            ok = (d / "assignments.npy").is_file()
            print(f"  {v['label']:14s} {'OK' if ok else 'MISSING':7s}  {d.name}")


def phase_assign(
    folds: Sequence[int],
    k: int,
    latent: int,
    epoch: int,
    pop: int,
    wnmf_epochs: int,
    *,
    skip_existing: bool,
    dry_run: bool,
) -> int:
    for fold in folds:
        base = base_assign_args(fold, k, latent, epoch, pop, wnmf_epochs)
        for v in VARIANTS:
            cmd = [
                sys.executable, "-u", str(GEN),
                *base,
                "--algo", *v["algo"],
                *v["extra"],
            ]
            if skip_existing:
                cmd.append("--skip-existing")
            print(f"\n=== fold={fold} {v['label']} ===", flush=True)
            rc = run_cmd(cmd, dry_run=dry_run)
            if rc != 0:
                return rc
    return 0


def _import_mealpy_wcss():
    sys.path.insert(0, str(MEALPY))
    from mealpy_comparison_v2 import compute_wcss_fast  # noqa: E402
    return compute_wcss_fast


def _adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def verify_one(
    fold: int,
    k: int,
    latent: int,
) -> dict:
    compute_wcss_fast = _import_mealpy_wcss()
    from sklearn.cluster import KMeans

    d_plain = assign_dir(ALGO_PLAIN, fold, k, latent, kmref=False)
    d_avoa = assign_dir(ALGO_AVOA, fold, k, latent, kmref=True)
    row: Dict[str, Any] = {
        "fold": fold,
        "k": k,
        "latent_L": latent,
        "checks_passed": True,
        "errors": [],
    }

    for label, d, kmref in [
        ("PLAIN_KMEANS", d_plain, False),
        ("AVOA_KMEANS", d_avoa, True),
    ]:
        prefix = label.lower()
        if not d.is_dir():
            row["errors"].append(f"{label}: directory missing: {d}")
            row["checks_passed"] = False
            continue
        for fname in ("user_features.npy", "assignments.npy", "best_sol.npy"):
            if not (d / fname).is_file():
                row["errors"].append(f"{label}: missing {fname}")
                row["checks_passed"] = False

        if row["errors"]:
            continue

        X = np.load(d / "user_features.npy")
        assign = np.load(d / "assignments.npy")
        sol = np.load(d / "best_sol.npy")
        n_users, n_feat = X.shape
        row[f"{prefix}_n_users"] = int(n_users)
        row[f"{prefix}_n_features"] = int(n_feat)
        row[f"{prefix}_assign_shape"] = list(assign.shape)
        row[f"{prefix}_best_sol_shape"] = list(sol.shape)

        if n_users != 943:
            row["errors"].append(f"{label}: expected 943 users, got {n_users}")
            row["checks_passed"] = False
        if n_feat != latent:
            row["errors"].append(f"{label}: expected L={latent}, got {n_feat}")
            row["checks_passed"] = False
        if sol.shape[0] != k * n_feat:
            row["errors"].append(f"{label}: best_sol shape mismatch")
            row["checks_passed"] = False

        active = len(np.unique(assign))
        row[f"{prefix}_n_active_clusters"] = int(active)
        if active != k:
            row["errors"].append(f"{label}: active clusters {active} != K={k}")
            row["checks_passed"] = False

        wcss, _ = compute_wcss_fast(X, sol, k, metric="euclidean")
        row[f"{prefix}_wcss"] = float(wcss)

        if label == "AVOA_KMEANS":
            centroids = np.asarray(sol, dtype=np.float64).reshape(k, n_feat)
            km = KMeans(
                n_clusters=k,
                init=centroids,
                n_init=1,
                max_iter=300,
                random_state=42,
            )
            km.fit(X)
            wcss_lloyd, _ = compute_wcss_fast(
                X, km.cluster_centers_.flatten(), k, metric="euclidean",
            )
            row["avoa_wcss_after_lloyd"] = float(wcss_lloyd)
            row["avoa_km_n_iter"] = int(km.n_iter_)
            if wcss_lloyd > float(wcss) + 1e-3:
                row["errors"].append(
                    f"AVOA: Lloyd WCSS ({wcss_lloyd:.2f}) > meta WCSS ({wcss:.2f})",
                )
                row["checks_passed"] = False

            ch = d / "convergence_history.csv"
            if ch.is_file():
                hist = pd.read_csv(ch)
                if "fitness" in hist.columns and len(hist) > 1:
                    row["avoa_conv_start"] = float(hist["fitness"].iloc[0])
                    row["avoa_conv_end"] = float(hist["fitness"].iloc[-1])

    if (d_plain / "assignments.npy").is_file() and (d_avoa / "assignments.npy").is_file():
        a0 = np.load(d_plain / "assignments.npy")
        a1 = np.load(d_avoa / "assignments.npy")
        ari = _adjusted_rand_index(a0, a1)
        row["ari_plain_vs_avoa"] = float(ari)
        if ari >= 0.999:
            row["errors"].append("PLAIN vs AVOA assignments nearly identical (ARI>=0.999)")
            row["checks_passed"] = False

        w0 = row.get("plain_kmeans_wcss")
        w1 = row.get("avoa_kmeans_wcss")
        if w0 is not None and w1 is not None and w0 > 0:
            row["wcss_gain_pct_vs_plain"] = (float(w0) - float(w1)) / float(w0) * 100.0

    return row


def phase_verify(
    folds: Sequence[int],
    k: int,
    latent: int,
) -> List[dict]:
    results = [verify_one(fold, k, latent) for fold in folds]
    OUT_VERIFY.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": PROTOCOL_TAG,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "k": k,
        "latent_L": latent,
        "folds": list(folds),
        "results": results,
    }
    OUT_VERIFY.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n-> {OUT_VERIFY}")
    for r in results:
        status = "PASS" if r.get("checks_passed") else "FAIL"
        print(
            f"  fold {r['fold']}: {status}  "
            f"ARI={r.get('ari_plain_vs_avoa', float('nan')):.4f}  "
            f"errors={len(r.get('errors', []))}",
            flush=True,
        )
        for e in r.get("errors", []):
            print(f"    ! {e}", flush=True)
    return results


def _parse_eval_from_wnmf_csv(
    csv_path: Path,
    fold: int,
    k: int,
    latent: int,
    *,
    predictor: str,
) -> List[dict]:
    cfg = PREDICTOR_CONFIG[predictor]
    lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()
    cmd_line = next((ln for ln in lines if ln.startswith("# command:")), "")
    if f"--fold {fold}" not in cmd_line and f"official_f{fold}" not in cmd_line:
        return []
    data_lines = [ln for ln in lines if not ln.startswith("#")]
    if not data_lines:
        return []
    rows_out = []
    for row in csv.DictReader(data_lines):
        if row.get("scenario") != cfg["scenario"]:
            continue
        algo = row.get("algo_label", "")
        if algo not in ALGO_LABELS:
            continue
        rows_out.append({
            "protocol": PROTOCOL_TAG,
            "predictor": predictor,
            "fold": int(fold),
            "k": int(k),
            "latent_L": int(latent),
            "algo": algo,
            "variant": ALGO_LABELS[algo],
            "mae": float(row["mae"]) if row.get("mae") else float("nan"),
            "rmse": float(row["rmse"]) if row.get("rmse") else float("nan"),
            "precision_at_10": float(row["precision_at_10"])
            if row.get("precision_at_10") else float("nan"),
            "recall_at_10": float(row["recall_at_10"])
            if row.get("recall_at_10") else float("nan"),
            "ndcg_at_10": float(row["ndcg_at_10"])
            if row.get("ndcg_at_10") else float("nan"),
            "n_test": int(float(row["n_test"])) if row.get("n_test") else None,
            "result_csv": str(csv_path.relative_to(REPO)),
        })
    return rows_out


def _latest_result_csv(fold: int, k: int, *, predictor: str) -> Optional[Path]:
    cfg = PREDICTOR_CONFIG[predictor]
    base = REPO / "results" / "wnmf" / "ml100k" / f"k{k}" / f"fold{fold}"
    if not base.is_dir():
        return None
    pattern = f"wnmf_results_ml100k_k*_{cfg['file_tag']}.csv"
    candidates = sorted(
        base.rglob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in candidates:
        head = p.read_text(encoding="utf-8", errors="replace").splitlines()[:8]
        if any(f"--fold {fold}" in ln for ln in head):
            return p
    return candidates[0] if candidates else None


def phase_eval(
    folds: Sequence[int],
    k: int,
    latent: int,
    *,
    predictor: str,
    skip_existing: bool,
    dry_run: bool,
) -> int:
    cfg = PREDICTOR_CONFIG[predictor]
    out_preds = out_preds_path(predictor)
    all_rows: List[dict] = []
    print(f"Predictor: {cfg['label']} ({predictor})", flush=True)
    for fold in folds:
        suf = eval_assign_suffix(fold, k, latent)
        cmd = [
            sys.executable, "-u", str(EXP),
            "--dataset", "100k",
            "--eval-split", "official",
            "--fold", str(int(fold)),
            "--mode", cfg["wnmf_mode"],
            "--no-global",
            *cfg["extra_cmd"],
            "--k", str(int(k)),
            "--latent-dim", str(int(latent)),
            "--algo", ALGO_PLAIN, ALGO_AVOA,
            "--assign-root", str(ASSIGN_ROOT.as_posix()),
            "--assign-suffix", suf,
            "--top-n", "10",
            "--relevance-threshold", "4.0",
        ]
        if skip_existing:
            cmd.append("--skip-existing")
        print(f"\n=== eval fold={fold} predictor={predictor} ===", flush=True)
        rc = run_cmd(cmd, dry_run=dry_run)
        if rc != 0:
            return rc
        if dry_run:
            continue
        csv_path = _latest_result_csv(fold, k, predictor=predictor)
        if csv_path is None:
            print(f"  [warn] no result CSV for fold {fold}", flush=True)
            continue
        parsed = _parse_eval_from_wnmf_csv(
            csv_path, fold, k, latent, predictor=predictor,
        )
        for r in parsed:
            print(
                f"  {r['variant']:14s} MAE={r['mae']:.4f} RMSE={r['rmse']:.4f} "
                f"P@10={r['precision_at_10']:.4f} R@10={r['recall_at_10']:.4f} "
                f"NDCG@10={r['ndcg_at_10']:.4f}",
                flush=True,
            )
        all_rows.extend(parsed)

    if all_rows:
        df = pd.DataFrame(all_rows)
        out_preds.parent.mkdir(parents=True, exist_ok=True)
        if out_preds.is_file():
            old = pd.read_csv(out_preds)
            merged = pd.concat([old, df], ignore_index=True)
            key = ["fold", "k", "latent_L", "algo", "predictor"]
            if "predictor" not in merged.columns:
                merged["predictor"] = predictor
            merged = merged.drop_duplicates(subset=key, keep="last")
            merged.to_csv(out_preds, index=False)
        else:
            df.to_csv(out_preds, index=False)
        print(f"\n-> {out_preds}")
    return 0


def _pct_improve(baseline: float, challenger: float) -> float:
    if baseline == 0 or np.isnan(baseline) or np.isnan(challenger):
        return float("nan")
    return (baseline - challenger) / baseline * 100.0


def phase_aggregate(k: int, latent: int, *, predictor: str = "paper") -> pd.DataFrame:
    out_preds = out_preds_path(predictor)
    out_cv5 = out_cv5_path(predictor)
    out_report = out_report_path(predictor)
    cfg = PREDICTOR_CONFIG[predictor]

    if not out_preds.is_file():
        print(f"No preds file: {out_preds}", flush=True)
        return pd.DataFrame()

    df = pd.read_csv(out_preds)
    if "predictor" in df.columns:
        df = df[df["predictor"] == predictor]
    df = df[(df["k"] == k) & (df["latent_L"] == latent)].copy()
    if df.empty:
        print("No matching rows in preds CSV.", flush=True)
        return pd.DataFrame()

    b0 = df[df["algo"] == ALGO_PLAIN].set_index("fold")
    rows_out: List[dict] = []

    for fold in sorted(df["fold"].unique()):
        for _, r in df[df["fold"] == fold].iterrows():
            delta_mae = float("nan")
            if r["algo"] == ALGO_AVOA and fold in b0.index:
                delta_mae = float(r["mae"]) - float(b0.loc[fold, "mae"])
            rows_out.append({
                "predictor": predictor,
                "fold": int(fold),
                "variant": r["variant"],
                "algo": r["algo"],
                "k": int(k),
                "latent_L": int(latent),
                "mae": r["mae"],
                "rmse": r["rmse"],
                "precision_at_10": r["precision_at_10"],
                "recall_at_10": r["recall_at_10"],
                "ndcg_at_10": r["ndcg_at_10"],
                "delta_mae_vs_B0": delta_mae if r["algo"] == ALGO_AVOA else 0.0,
            })

    summary = df.groupby(["algo", "variant"])[METRIC_COLS].agg(["mean", "std"])
    for (algo, variant), grp in df.groupby(["algo", "variant"]):
        means = {m: float(grp[m].mean()) for m in METRIC_COLS}
        stds = {f"{m}_std": float(grp[m].std()) for m in METRIC_COLS}
        row = {
            "predictor": predictor,
            "fold": "cv5_mean",
            "variant": variant,
            "algo": algo,
            "k": int(k),
            "latent_L": int(latent),
            **means,
            **stds,
            "delta_mae_vs_B0": 0.0,
        }
        rows_out.append(row)

    if ALGO_PLAIN in df["algo"].values and ALGO_AVOA in df["algo"].values:
        plain_mean = df[df["algo"] == ALGO_PLAIN][METRIC_COLS].mean()
        avoa_mean = df[df["algo"] == ALGO_AVOA][METRIC_COLS].mean()
        delta_row = {
            "predictor": predictor,
            "fold": "cv5_delta_pct",
            "variant": "AVOA_vs_B0_improvement_pct",
            "algo": "DELTA",
            "k": int(k),
            "latent_L": int(latent),
        }
        for m in METRIC_COLS:
            delta_row[m] = _pct_improve(float(plain_mean[m]), float(avoa_mean[m]))
        delta_row["delta_mae_vs_B0"] = float(avoa_mean["mae"]) - float(plain_mean["mae"])
        rows_out.append(delta_row)

    out_df = pd.DataFrame(rows_out)
    out_cv5.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_cv5, index=False)
    print(f"\n{out_df.to_string(index=False)}")
    print(f"\n-> {out_cv5}")

    _write_report(out_df, k, latent, predictor=predictor, cfg=cfg, out_report=out_report)
    return out_df


def _write_report(
    df: pd.DataFrame,
    k: int,
    latent: int,
    *,
    predictor: str,
    cfg: dict,
    out_report: Path,
) -> None:
    delta = df[df["fold"] == "cv5_delta_pct"]
    if delta.empty:
        return
    d = delta.iloc[0]
    lines = [
        "# AVOA-K-MEANS Protocol Report",
        "",
        f"- Dataset: ML-100K official 5-fold",
        f"- K={k}, WNMF L={latent}",
        f"- Predictor: {cfg['label']}",
        f"- Baseline: PLAIN_KMEANS (B0_KMEANS)",
        f"- Challenger: AVOA_KMEANS (B_AVOA + Lloyd K-means)",
        "",
        "## CV5 Improvement vs Plain K-means (%)",
        "",
        "| Metric | Improvement % |",
        "|--------|---------------|",
    ]
    for m, label in [
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("precision_at_10", "Precision@10"),
        ("recall_at_10", "Recall@10"),
        ("ndcg_at_10", "NDCG@10"),
    ]:
        val = d.get(m, float("nan"))
        lines.append(f"| {label} | {val:.2f}% |")

    lines.extend([
        "",
        "## Tez cümlesi (şablon)",
        "",
        (
            "> AVOA-K-MEANS yönteminde, WNMF ile elde edilen kullanıcı özellik uzayında "
            "African Vulture Optimization Algorithm (AVOA) ile centroid konumları optimize edilmiş; "
            "elde edilen centroidler başlangıç değeri olarak Lloyd K-means algoritmasına verilmiştir. "
            f"Düz K-means baseline'ına kıyasla ML-100K official 5-fold protokolünde "
            f"{cfg['label']} tahmininde MAE/{d.get('mae', float('nan')):.2f}%, "
            f"RMSE/{d.get('rmse', float('nan')):.2f}%, "
            f"Precision@10/{d.get('precision_at_10', float('nan')):.2f}%, "
            f"Recall@10/{d.get('recall_at_10', float('nan')):.2f}%, "
            f"NDCG@10/{d.get('ndcg_at_10', float('nan')):.2f}% iyileşme gözlemlenmiştir."
        ),
        "",
        f"Detay: `{out_cv5_path(predictor).relative_to(REPO)}`",
    ])
    out_report.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out_report}")


def check_environment() -> List[str]:
    errors = []
    for fold in CV_FOLDS:
        base = REPO / "data" / "ml-100k" / f"u{fold}.base"
        test = REPO / "data" / "ml-100k" / f"u{fold}.test"
        if not base.is_file() or not test.is_file():
            errors.append(f"Missing official fold files: u{fold}.base / u{fold}.test")
    try:
        sys.path.insert(0, str(MEALPY))
        from mealpy.swarm_based.AVOA import OriginalAVOA  # noqa: F401
    except Exception as exc:
        errors.append(f"AVOA import failed: {exc}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="AVOA-K-MEANS vs PLAIN K-means protocol")
    ap.add_argument(
        "--phase",
        choices=["status", "assign", "verify", "eval", "aggregate", "all"],
        default="all",
    )
    ap.add_argument("--fold", nargs="+", type=int, default=None, help="subset of folds 1..5")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--latent", type=int, default=DEFAULT_LATENT)
    ap.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    ap.add_argument("--pop-size", type=int, default=DEFAULT_POP)
    ap.add_argument("--wnmf-epochs", type=int, default=DEFAULT_WNMF_EPOCHS)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--predictor",
        choices=list(PREDICTOR_CONFIG.keys()),
        default="paper",
        help="paper=CalcAvgRating; sharedV=cluster-aware WNMF",
    )
    args = ap.parse_args()

    folds = args.fold if args.fold else CV_FOLDS
    folds = [int(f) for f in folds if 1 <= int(f) <= 5]

    print("AVOA-K-MEANS Protocol")
    print(f"  Predictor: {PREDICTOR_CONFIG[args.predictor]['label']}")
    params = _load_params()
    if params:
        print(f"  Params: {PARAMS_JSON.relative_to(REPO)}")

    env_errors = check_environment()
    if env_errors:
        for e in env_errors:
            print(f"ENV ERROR: {e}", file=sys.stderr)
        if args.phase != "status":
            return 1

    if args.phase == "status":
        phase_status(folds, args.k, args.latent)
        return 0

    if args.phase in ("assign", "all"):
        rc = phase_assign(
            folds, args.k, args.latent, args.epoch, args.pop_size, args.wnmf_epochs,
            skip_existing=args.skip_existing, dry_run=args.dry_run,
        )
        if rc != 0:
            return rc

    if args.phase in ("verify", "all"):
        phase_verify(folds, args.k, args.latent)

    if args.phase in ("eval", "all"):
        # wnmf_experiment --skip-existing fold'u ayırt etmez; eval'de varsayılan kapalı.
        eval_skip = args.skip_existing if args.phase == "eval" else False
        rc = phase_eval(
            folds, args.k, args.latent,
            predictor=args.predictor,
            skip_existing=eval_skip, dry_run=args.dry_run,
        )
        if rc != 0:
            return rc

    if args.phase in ("aggregate", "all") and not args.dry_run:
        phase_aggregate(args.k, args.latent, predictor=args.predictor)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
