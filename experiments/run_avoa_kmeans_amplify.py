"""
AVOA vs B0 farkını büyütmek — çoklu müdahale pilotu (fold 1).

Sorun: WCSS fitness + CalcAvgRating/sharedV → atama farklı, MAE neredeyse aynı.

Müdahaleler (öncelik sırası):
  1. knn_mae     — centroid araması doğrudan küme-içi kNN val MAE minimize eder
  2. cluster_knn — tahmin ortalama değil, komşu tabanlı (assignment'a duyarlı)
  3. no_kmref    — Lloyd adımı atamaları homojenleştirir, kapat
  4. small_k     — K=5 ile küme ortalaması/kNN daha hassas
  5. spread_mo   — WCSS + silhouette + CH + repulsion (çeşitlilik)

Kullanım:
  python -m experiments.run_avoa_kmeans_amplify --phase all
  python -m experiments.run_avoa_kmeans_amplify --phase assign --variant knn_mae_no_kmref
  python -m experiments.run_avoa_kmeans_amplify --phase eval --variant baseline_wcss
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
GEN = REPO / "mealpy" / "generate_assignments.py"
EXP = REPO / "wnmf" / "wnmf_experiment.py"
ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments"
def out_path(predictor: str) -> Path:
    suf = "" if predictor == "cluster_knn" else f"_{predictor}"
    return REPO / "results" / f"avoa_kmeans_amplify_summary{suf}.csv"

PREDICTORS = {
    "cluster_knn": {
        "scenario_key": "cluster_knn",
        "cmd": [
            "--no-global", "--no-cluster-avg",
            "--cluster-knn-backend", "native",
            "--similarity", "cosine",
            "--knn", "30",
            "--min-common", "3",
        ],
    },
    "cluster_avg": {
        "scenario_key": "calc_avg_rating",
        "cmd": [
            "--paper-mode",
            "--no-global", "--no-cluster-knn",
        ],
    },
}

FOLD = 1
K_DEFAULT = 10
LATENT = 50
ALGOS = ["B0_KMEANS", "B_AVOA"]

BASE_ASSIGN = [
    "--dataset", "100k",
    "--no-prune", "--no-gray-sheep",
    "--preprocess", "none",
    "--feature-extraction", "wnmf",
    "--svd-components", str(LATENT),
    "--wnmf-epochs", "50",
    "--legacy-wnmf-suffix",
    "--init-mode", "mkpp",
    "--cluster-metric", "euclidean",
    "--cluster-objective", "wcss",
    "--train-only", "--eval-split", "official",
    "--fold", str(FOLD),
]

# variant_id -> config
VARIANTS: Dict[str, dict] = {
    "baseline_wcss": {
        "label": "Mevcut: WCSS + kmref + cluster_knn eval",
        "k": K_DEFAULT,
        "assign_extra": [
            "--fitness", "wcss",
            "--kmeans-refine-overwrite", "--kmeans-refine-iter", "300",
            "--baseline-epoch", "200", "--pop-size", "50",
        ],
        "suffix": (
            f"_euc_imkpp_nogs_trainonly_official_f{FOLD}"
            f"_none_wnmf{LATENT}_k{K_DEFAULT}_pwcss_kmref"
        ),
        "skip_assign": True,  # zaten mevcut
    },
    "knn_mae_no_kmref": {
        "label": "knn_mae fitness, kmref KAPALI",
        "k": K_DEFAULT,
        "assign_extra": [
            "--fitness", "knn_mae",
            "--no-kmeans-refine",
            "--cluster-objective", "wcss",
            "--baseline-epoch", "100", "--pop-size", "30",
            "--centroid-iter", "80", "--centroid-agents", "30",
            "--centroid-knn-sim", "cosine", "--centroid-knn-k", "30",
            "--centroid-train-sample", "500", "--centroid-val-sample", "300",
        ],
        "suffix": (
            f"_euc_imkpp_nogs_trainonly_official_f{FOLD}"
            f"_none_wnmf{LATENT}_k{K_DEFAULT}_pwcss_knnmae"
        ),
    },
    "knn_mae_kmref": {
        "label": "knn_mae fitness + kmref",
        "k": K_DEFAULT,
        "assign_extra": [
            "--fitness", "knn_mae",
            "--kmeans-refine-overwrite", "--kmeans-refine-iter", "300",
            "--cluster-objective", "wcss",
            "--baseline-epoch", "100", "--pop-size", "30",
            "--centroid-iter", "80", "--centroid-agents", "30",
            "--centroid-knn-sim", "cosine", "--centroid-knn-k", "30",
            "--centroid-train-sample", "500", "--centroid-val-sample", "300",
        ],
        "suffix": (
            f"_euc_imkpp_nogs_trainonly_official_f{FOLD}"
            f"_none_wnmf{LATENT}_k{K_DEFAULT}_pwcss_knnmae_kmref"
        ),
    },
    "k5_wcss_kmref": {
        "label": "K=5, WCSS + kmref",
        "k": 5,
        "assign_extra": [
            "--fitness", "wcss",
            "--kmeans-refine-overwrite", "--kmeans-refine-iter", "300",
            "--baseline-epoch", "200", "--pop-size", "50",
        ],
        "suffix": (
            f"_euc_imkpp_nogs_trainonly_official_f{FOLD}"
            f"_none_wnmf{LATENT}_k5_pwcss_kmref"
        ),
    },
    "spread_mo_no_kmref": {
        "label": "MO spread + repulsion, kmref KAPALI",
        "k": K_DEFAULT,
        "assign_extra": [
            "--fitness", "wcss",
            "--cluster-objective", "multi",
            "--mo-weights", "spread",
            "--centroid-repulsion-lambda", "0.15",
            "--no-kmeans-refine",
            "--baseline-epoch", "200", "--pop-size", "50",
        ],
        "suffix": (
            f"_euc_imkpp_nogs_trainonly_official_f{FOLD}"
            f"_none_wnmf{LATENT}_k{K_DEFAULT}_pwcss"
        ),
        "avoa_suffix": (
            f"_euc_imkpp_nogs_trainonly_official_f{FOLD}"
            f"_none_wnmf{LATENT}_k{K_DEFAULT}"
        ),
    },
}


def run_cmd(cmd: List[str]) -> int:
    print("\n" + "=" * 72)
    print(" ".join(cmd))
    print("=" * 72, flush=True)
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def assign_dir(algo: str, suffix: str) -> Path:
    return ASSIGN_ROOT / "ml100k" / f"{algo}{suffix}"


def phase_assign(variant_ids: Sequence[str], *, skip_existing: bool) -> int:
    for vid in variant_ids:
        v = VARIANTS[vid]
        if v.get("skip_assign"):
            print(f"[skip] {vid}: assignment zaten mevcut protokolde")
            continue
        k = int(v["k"])
        for algo in ALGOS:
            d = assign_dir(algo, v["suffix"])
            if skip_existing and (d / "assignments.npy").is_file():
                print(f"[skip-existing] {d.name}")
                continue
            extra = list(v["assign_extra"])
            if algo == "B0_KMEANS" and "--no-kmeans-refine" not in extra:
                # B0 her zaman düz KMeans; kmref flag'i B0'ı etkilemez
                pass
            cmd = [
                sys.executable, "-u", str(GEN),
                *BASE_ASSIGN,
                "--k", str(k),
                "--algo", algo,
                *extra,
            ]
            if skip_existing:
                cmd.append("--skip-existing")
            print(f"\n=== assign {vid} {algo} ===", flush=True)
            rc = run_cmd(cmd)
            if rc != 0:
                return rc
    return 0


def _parse_eval_csv(csv_path: Path, scenario_key: str) -> List[dict]:
    lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()
    data = [ln for ln in lines if not ln.startswith("#")]
    if not data:
        return []
    rows = []
    for row in pd.read_csv(io.StringIO("\n".join(data))).to_dict("records"):
        scen = str(row.get("scenario", ""))
        if scenario_key == "cluster_knn":
            if "cluster_knn" not in scen:
                continue
        elif scen != scenario_key:
            continue
        algo = row.get("algo_label", "")
        if algo not in ALGOS:
            continue
        rows.append({
            "algo": algo,
            "scenario": scen,
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
            "precision_at_10": float(row.get("precision_at_10", float("nan"))),
            "recall_at_10": float(row.get("recall_at_10", float("nan"))),
            "ndcg_at_10": float(row.get("ndcg_at_10", float("nan"))),
        })
    return rows


def _eval_python_pair(v: dict, *, predictor: str) -> Optional[dict]:
    """AVOA/B0 farklı suffix gerektiğinde doğrudan Python eval."""
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "wnmf"))
    from wnmf.wnmf_utils import load_ratings_100k, load_assignment
    from wnmf.wnmf_experiment import run_cluster_average, run_cluster_knn

    k = int(v["k"])
    b0_suf = v["suffix"]
    av_suf = v.get("avoa_suffix", b0_suf)
    base = REPO / "data" / "ml-100k" / f"u{FOLD}.base"
    test_p = REPO / "data" / "ml-100k" / f"u{FOLD}.test"
    train, test = load_ratings_100k(str(base), str(test_p), fold=FOLD)
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    metrics = {}
    for algo, suf, avoa_suf in [
        ("B0_KMEANS", b0_suf, None),
        ("B_AVOA", av_suf, av_suf),
    ]:
        d = _resolve_assign_dir(algo, suf, avoa_suf)
        assignments, gray = load_assignment(str(d))
        if predictor == "cluster_avg":
            row = run_cluster_average(
                train, test, assignments, gray, None, n_items, algo,
                cluster_avg_hard=True,
                cluster_avg_global_fallback=True,
                top_n=10, relevance_threshold=4.0,
            )
        else:
            row = run_cluster_knn(
                train, test, assignments, gray, None, n_items, algo,
                similarity="cosine", k_neighbors=30, min_common=3,
                cluster_knn_backend="native", top_n=10, relevance_threshold=4.0,
            )
        metrics[algo] = row

    b0, av = metrics["B0_KMEANS"], metrics["B_AVOA"]
    pred_corr = float("nan")
    try:
        pred_corr = _pred_corr(av_suf, b0_suf, k, predictor=predictor)
    except Exception:
        pass
    out = {
        "variant": v.get("_vid", ""),
        "label": v["label"],
        "k": k,
        "fold": FOLD,
        "b0_mae": float(b0["mae"]),
        "avoa_mae": float(av["mae"]),
        "delta_mae": float(av["mae"]) - float(b0["mae"]),
        "delta_mae_pct": (float(b0["mae"]) - float(av["mae"])) / float(b0["mae"]) * 100,
        "b0_ndcg": float(b0.get("ndcg_at_10", float("nan"))),
        "avoa_ndcg": float(av.get("ndcg_at_10", float("nan"))),
        "pred_corr": pred_corr,
        "predictor": predictor,
        "scenario": str(av.get("scenario", "")),
    }
    return out


def phase_eval(variant_ids: Sequence[str], *, predictor: str) -> pd.DataFrame:
    pcfg = PREDICTORS[predictor]
    all_rows = []
    for vid in variant_ids:
        v = VARIANTS[vid]
        k = int(v["k"])
        suf = v["suffix"]
        av_suf = v.get("avoa_suffix")
        if av_suf and av_suf != suf:
            print(f"\n=== eval {predictor} {vid} (python, avoa_suffix) ===", flush=True)
            v2 = {**v, "_vid": vid}
            row = _eval_python_pair(v2, predictor=predictor)
            if row:
                row["variant"] = vid
                print(
                    f"  B0 MAE={row['b0_mae']:.4f}  AVOA MAE={row['avoa_mae']:.4f}  "
                    f"dMAE={row['delta_mae']:+.4f} ({row['delta_mae_pct']:+.2f}%)  "
                    f"pred_r={row['pred_corr']:.3f}",
                    flush=True,
                )
                all_rows.append(row)
            continue
        cmd = [
            sys.executable, "-u", str(EXP),
            "--dataset", "100k",
            "--eval-split", "official",
            "--fold", str(FOLD),
            "--mode", "baselines",
            *pcfg["cmd"],
            "--k", str(k),
            "--algo", *ALGOS,
            "--assign-root", str(ASSIGN_ROOT.as_posix()),
            "--assign-suffix", suf,
            "--top-n", "10",
            "--relevance-threshold", "4.0",
        ]
        print(f"\n=== eval {predictor} {vid} ===", flush=True)
        rc = run_cmd(cmd)
        if rc != 0:
            print(f"HATA eval {vid} rc={rc}", file=sys.stderr)
            continue
        base = REPO / "results" / "wnmf" / "ml100k" / f"k{k}" / f"fold{FOLD}"
        cands = sorted(
            base.rglob("wnmf_results_ml100k_k*_baselines.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not cands:
            continue
        parsed = _parse_eval_csv(cands[0], pcfg["scenario_key"])
        b0 = next((r for r in parsed if r["algo"] == "B0_KMEANS"), None)
        av = next((r for r in parsed if r["algo"] == "B_AVOA"), None)
        if not b0 or not av:
            print(f"  [warn] B0/AVOA satırı yok: {parsed}")
            continue
        # pred correlation (hızlı)
        pred_corr = float("nan")
        try:
            pred_corr = _pred_corr(
                v.get("avoa_suffix", v["suffix"]),
                v["suffix"],
                k,
                predictor=predictor,
            )
        except Exception as exc:
            print(f"  [warn] pred_corr: {exc}")

        row = {
            "variant": vid,
            "label": v["label"],
            "k": k,
            "fold": FOLD,
            "b0_mae": b0["mae"],
            "avoa_mae": av["mae"],
            "delta_mae": av["mae"] - b0["mae"],
            "delta_mae_pct": (b0["mae"] - av["mae"]) / b0["mae"] * 100,
            "b0_ndcg": b0["ndcg_at_10"],
            "avoa_ndcg": av["ndcg_at_10"],
            "pred_corr": pred_corr,
            "predictor": predictor,
            "scenario": av["scenario"],
        }
        print(
            f"  B0 MAE={row['b0_mae']:.4f}  AVOA MAE={row['avoa_mae']:.4f}  "
            f"dMAE={row['delta_mae']:+.4f} ({row['delta_mae_pct']:+.2f}%)  "
            f"pred_r={row['pred_corr']:.3f}",
            flush=True,
        )
        all_rows.append(row)

    if all_rows:
        df = pd.DataFrame(all_rows)
        out = out_path(predictor)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\n-> {out}")
        print(df.sort_values("delta_mae_pct", ascending=False).to_string(index=False))
    return pd.DataFrame(all_rows)


def _resolve_assign_dir(algo: str, suffix: str, avoa_suffix: Optional[str] = None) -> Path:
    if algo == "B_AVOA" and avoa_suffix:
        d = assign_dir(algo, avoa_suffix)
        if d.is_dir():
            return d
    d = assign_dir(algo, suffix)
    if algo == "B0_KMEANS" and suffix.endswith("_kmref"):
        alt = suffix.replace("_kmref", "")
        d2 = assign_dir(algo, alt)
        if d2.is_dir():
            return d2
    return d


def _pred_corr(
    avoa_suffix: str,
    b0_suffix: str,
    k: int,
    *,
    predictor: str,
) -> float:
    """Tahmin korelasyonu (assignment farkının yansıması)."""
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "wnmf"))
    from wnmf.wnmf_utils import load_ratings_100k, load_assignment
    from wnmf.wnmf_experiment import run_cluster_average, run_cluster_knn

    base = REPO / "data" / "ml-100k" / f"u{FOLD}.base"
    test_p = REPO / "data" / "ml-100k" / f"u{FOLD}.test"
    train, test = load_ratings_100k(str(base), str(test_p), fold=FOLD)
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    preds = {}
    for algo in ALGOS:
        suf = b0_suffix if algo == "B0_KMEANS" else avoa_suffix
        d = _resolve_assign_dir(algo, suf, avoa_suffix if algo == "B_AVOA" else None)
        assignments, gray = load_assignment(str(d))
        if predictor == "cluster_avg":
            row = run_cluster_average(
                train, test, assignments, gray, None, n_items, algo,
                cluster_avg_hard=True,
                cluster_avg_global_fallback=True,
                top_n=10, relevance_threshold=4.0,
                return_eval_rows=True,
            )
        else:
            row = run_cluster_knn(
                train, test, assignments, gray, None, n_items, algo,
                similarity="cosine", k_neighbors=30, min_common=3,
                cluster_knn_backend="native", top_n=10, relevance_threshold=4.0,
                return_eval_rows=True,
            )
        preds[algo] = row["eval_rows"][:, 3].astype(np.float64)
    return float(np.corrcoef(preds["B0_KMEANS"], preds["B_AVOA"])[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["assign", "eval", "all"], default="all")
    ap.add_argument(
        "--variant",
        nargs="+",
        default=list(VARIANTS.keys()),
        choices=list(VARIANTS.keys()),
    )
    ap.add_argument(
        "--predictor",
        choices=list(PREDICTORS.keys()),
        default="cluster_knn",
    )
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    args = ap.parse_args()

    print("AVOA fark amplifikasyon pilotu (fold 1)")
    print(f"Eval: {args.predictor}\n")

    if args.phase in ("assign", "all"):
        rc = phase_assign(args.variant, skip_existing=args.skip_existing)
        if rc != 0:
            return rc

    if args.phase in ("eval", "all"):
        phase_eval(args.variant, predictor=args.predictor)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
