"""
IWO_HHO fold 2, K=3..24: zscore / LOF / zscore+LOF vs baseline (nogs, none, no LOF).

  python -m experiments.run_iwo_hho_fold2_zscore_lof_compare --phase all --jobs 3
  python -m experiments.run_iwo_hho_fold2_zscore_lof_compare --phase compare
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.fuzzy_official_protocol import K3_24_LIST, WNMF_DIM
from experiments.run_fuzzy_official_f1_eval import DEFAULT_SOFT

GEN = REPO / "mealpy" / "generate_assignments.py"
ESTOP = REPO / "mealpy" / "results" / "assignments_estop" / "ml100k"
LOF_ESTOP = REPO / "mealpy" / "results" / "assignments_lof_estop" / "ml100k"
BASELINE_CSV = REPO / "results" / "fuzzy_official_folds2_5_k3_24_cluster_avg_soft.csv"
OUT_CSV = REPO / "results" / "iwo_hho_fold2_zscore_lof_compare.csv"
OUT_SUMMARY = REPO / "results" / "iwo_hho_fold2_zscore_lof_compare_summary.md"
OUT_PIVOT = REPO / "results" / "iwo_hho_fold2_zscore_lof_mae_pivot.csv"

ALGO = "IWO_HHO"
FOLD = 2
SOFT = 0.1
SIM = "cosine"

VARIANTS = [
    {"variant": "baseline_nogs_none", "preprocess": "none", "lof": False, "skip_assign": True},
    {"variant": "zscore_only", "preprocess": "zscore", "lof": False, "skip_assign": False},
    {"variant": "lof_only", "preprocess": "none", "lof": True, "skip_assign": False},
    {"variant": "zscore_lof", "preprocess": "zscore", "lof": True, "skip_assign": False},
]


def assign_root(use_lof: bool) -> Path:
    return LOF_ESTOP if use_lof else ESTOP


def find_assign_dir(k: int, preprocess: str, use_lof: bool) -> Path | None:
    root = assign_root(use_lof)
    pat = (
        f"{ALGO}*trainonly_official_f{FOLD}_{preprocess}_wnmf{WNMF_DIM}_k{k}*pwcss*m15"
    )
    cands = [
        p for p in root.glob(pat)
        if (p / "assignments.npy").is_file() and (p / "memberships.npy").is_file()
    ]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def run_assign(variant: dict, *, jobs: int, skip_existing: bool) -> int:
    if variant.get("skip_assign"):
        return 0
    cmd = [
        sys.executable, "-u", str(GEN),
        "--dataset", "100k",
        "--algo", ALGO,
        "--no-prune",
        "--preprocess", variant["preprocess"],
        "--feature-extraction", "wnmf",
        "--svd-components", str(WNMF_DIM),
        "--wnmf-epochs", str(WNMF_DIM),
        "--legacy-wnmf-suffix",
        "--init-mode", "mkpp",
        "--cluster-metric", "fuzzy",
        "--fitness", "wcss",
        "--cluster-objective", "wcss",
        "--train-only", "--eval-split", "official",
        "--fold", str(FOLD),
        "--fcm-m", "1.5", "--fcm-m-suffix",
        "--k", *[str(k) for k in K3_24_LIST],
        "--jobs", str(jobs),
        "--baseline-epoch", "40", "--pop-size", "25",
        "--early-stop", "--early-stop-patience", "4",
        "--early-stop-block", "5",
    ]
    if variant["lof"]:
        cmd.append("--lof")
    else:
        cmd.append("--no-gray-sheep")
    if skip_existing:
        cmd.append("--skip-existing")
    print(f"\nASSIGN {variant['variant']}: preprocess={variant['preprocess']} lof={variant['lof']}")
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def eval_variant(variant: dict) -> list[dict]:
    from wnmf.wnmf_experiment import (
        _align_assignment_bundle,
        _cluster_avg_predict_kwargs,
        _nearest_centroid_bundle,
        load_assignment,
        load_memberships,
        run_cluster_average,
    )
    from wnmf.wnmf_utils import load_ratings_100k

    if variant["variant"] == "baseline_nogs_none":
        if not BASELINE_CSV.is_file():
            raise FileNotFoundError(BASELINE_CSV)
        df = pd.read_csv(BASELINE_CSV)
        sub = df[
            (df["fold"] == FOLD)
            & (df["algo"] == ALGO)
            & (df.get("predictor", "") == "cluster_avg_soft")
            & (df.get("similarity", "") == SIM)
        ]
        rows = []
        for _, r in sub.iterrows():
            rows.append({
                "variant": variant["variant"],
                "preprocess": "none",
                "lof": False,
                "fold": FOLD,
                "k": int(r["k"]),
                "algo": ALGO,
                "mae": float(r["mae"]),
                "rmse": float(r["rmse"]),
                "ndcg_at_10": float(r["ndcg_at_10"]),
                "precision_at_10": float(r["precision_at_10"]),
                "recall_at_10": float(r["recall_at_10"]),
                "assignment_dir": "",
                "eval_seconds": float(r.get("eval_seconds", 0)),
            })
        return rows

    base = str(REPO / "data" / "ml-100k" / "u1.base")
    test = str(REPO / "data" / "ml-100k" / "u1.test")
    train, test_arr = load_ratings_100k(base, test, fold=FOLD)
    n_items = int(max(train[:, 1].max(), test_arr[:, 1].max())) + 1
    n_users = int(max(train[:, 0].max(), test_arr[:, 0].max())) + 1
    eval_args = Namespace(similarity=SIM, min_common=3, soft_membership_threshold=SOFT)

    rows = []
    for k in K3_24_LIST:
        adir = find_assign_dir(k, variant["preprocess"], variant["lof"])
        if adir is None:
            print(f"SKIP {variant['variant']} K={k}: atama yok", flush=True)
            continue
        assignments, gray_mask = load_assignment(str(adir))
        memberships = load_memberships(str(adir))
        assignments, gray_mask, memberships, _ = _align_assignment_bundle(
            assignments, gray_mask, memberships, None,
            n_users_expected=n_users, algo_label=ALGO, assign_dir=str(adir),
        )
        nc = _nearest_centroid_bundle(None, str(adir), assignments)
        t0 = time.time()
        r = run_cluster_average(
            train, test_arr, assignments, gray_mask, memberships, n_items, ALGO,
            **_cluster_avg_predict_kwargs(eval_args),
            **nc,
            top_n=10, relevance_threshold=4.0, assign_dir=str(adir),
        )
        row = {
            "variant": variant["variant"],
            "preprocess": variant["preprocess"],
            "lof": bool(variant["lof"]),
            "fold": FOLD,
            "k": k,
            "algo": ALGO,
            "mae": float(r["mae"]),
            "rmse": float(r["rmse"]),
            "ndcg_at_10": float(r["ndcg_at_10"]),
            "precision_at_10": float(r["precision_at_10"]),
            "recall_at_10": float(r["recall_at_10"]),
            "assignment_dir": str(adir),
            "eval_seconds": round(time.time() - t0, 1),
        }
        rows.append(row)
        print(
            f"  {variant['variant']} K={k}: MAE={row['mae']:.4f} NDCG={row['ndcg_at_10']:.4f}",
            flush=True,
        )
    return rows


def phase_compare(df: pd.DataFrame) -> None:
    lines = [
        "# IWO_HHO fold 2 — zscore / LOF karsilastirma (K=3..24)",
        "",
        "Baseline: fuzzy/imkpp/nogs, preprocess=none, LOF yok (mevcut fold2 grid).",
        "LOF varyantlari: gray sheep acik (--lof; --no-gray-sheep ile birlikte kullanilmaz).",
        "",
    ]
    for v in df["variant"].unique():
        sub = df[df["variant"] == v]
        lines.append(f"## {v}")
        lines.append(f"- Ort. MAE: **{sub['mae'].mean():.4f}**")
        lines.append(f"- En iyi MAE: **{sub['mae'].min():.4f}** (K={int(sub.loc[sub['mae'].idxmin(), 'k'])})")
        lines.append(f"- Ort. NDCG@10: {sub['ndcg_at_10'].mean():.4f}")
        lines.append("")
    best_v = df.groupby("variant")["mae"].mean().idxmin()
    lines.append(f"**En dusuk ortalama MAE:** `{best_v}`")
    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")

    pivot = df.pivot_table(index="k", columns="variant", values="mae", aggfunc="first")
    pivot.to_csv(OUT_PIVOT)
    print(f"\n{OUT_SUMMARY}")
    print(f"{OUT_PIVOT}")
    print("\nOrt. MAE:")
    print(df.groupby("variant")["mae"].mean().sort_values().to_string())
    print("\nEn iyi K basina MAE (ilk 8 K):")
    print(pivot.head(8).round(4).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["assign", "eval", "compare", "all"], default="all")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument(
        "--variants",
        default="",
        help="Virgulle: lof_only,zscore_lof (bos = hepsi)",
    )
    args = ap.parse_args()
    only = {s.strip() for s in args.variants.split(",") if s.strip()}
    variants = [v for v in VARIANTS if not only or v["variant"] in only]

    if args.phase in ("assign", "all"):
        for v in variants:
            if v.get("skip_assign"):
                continue
            rc = run_assign(v, jobs=args.jobs, skip_existing=args.skip_existing)
            if rc != 0:
                sys.exit(rc)

    if args.phase in ("eval", "all", "compare"):
        all_rows: list[dict] = []
        for v in VARIANTS:
            all_rows.extend(eval_variant(v))
        out = pd.DataFrame(all_rows)
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT_CSV, index=False)
        print(f"\nCSV -> {OUT_CSV} ({len(out)} rows)")

    if args.phase in ("compare", "all"):
        phase_compare(pd.read_csv(OUT_CSV))


if __name__ == "__main__":
    main()
