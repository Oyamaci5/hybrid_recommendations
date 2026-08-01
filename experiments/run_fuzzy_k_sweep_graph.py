"""
FCM fuzzy K-sweep: assignment üret + cluster_avg eval + CSV + grafik.

Protokol: _fuzzy_imkpp_nogs_trainonly_rand_f1_none_wnmf20_k{K}  (WNMF20)

  python experiments/run_fuzzy_k_sweep_graph.py --phase assign --jobs 4
  python experiments/run_fuzzy_k_sweep_graph.py --phase eval
  python experiments/run_fuzzy_k_sweep_graph.py --phase plot
  python experiments/run_fuzzy_k_sweep_graph.py --phase all --jobs 4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from argparse import Namespace
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "wnmf"))

GEN = REPO / "mealpy" / "generate_assignments.py"
ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments"
OUT_CSV = REPO / "results" / "fuzzy_k_sweep_cluster_avg_w20.csv"
OUT_PLOT = REPO / "results" / "plots" / "fuzzy_k_sweep_cluster_avg_w20.png"

ALGOS = ["HA_AVOAHGS", "B1_HHO", "B_AVOA", "LIT_PSO", "LIT_GWO"]
K_LIST = [3, 6, 9, 10, 11, 12, 14]
WNMF_DIM = 20

ALGO_LABELS = {
    "HA_AVOAHGS": "HA",
    "B1_HHO": "B1",
    "B_AVOA": "B_AVOA",
    "LIT_PSO": "LIT_PSO",
    "LIT_GWO": "LIT_GWO",
}

PLOT_COLORS = {
    "HA": "#E63946",
    "B1": "#457B9D",
    "B_AVOA": "#2A9D8F",
    "LIT_PSO": "#E9C46A",
    "LIT_GWO": "#8338EC",
}


def suffix(k: int) -> str:
    return f"_fuzzy_imkpp_nogs_trainonly_rand_f1_none_wnmf20_k{k}"


def assign_path(algo: str, k: int) -> Path:
    return ASSIGN_ROOT / "ml100k" / f"{algo}{suffix(k)}"


def has_assign(algo: str, k: int) -> bool:
    return (assign_path(algo, k) / "assignments.npy").is_file()


def cluster_stats(algo: str, k: int) -> dict:
    p = assign_path(algo, k) / "assignments.npy"
    if not p.is_file():
        return {}
    a = np.load(p)
    sizes = sorted(Counter(a.astype(int).tolist()).values())
    if not sizes:
        return {}
    return {
        "n_active_clusters": len(sizes),
        "cluster_min": min(sizes),
        "cluster_max": max(sizes),
        "cluster_std": float(np.std(sizes)),
        "singletons": sum(1 for s in sizes if s == 1),
    }


def phase_assign(jobs: int, skip_existing: bool) -> int:
    cmd = [
        sys.executable, str(GEN),
        "--dataset", "100k",
        "--algo", *ALGOS,
        "--no-prune", "--no-gray-sheep",
        "--preprocess", "none",
        "--feature-extraction", "wnmf",
        "--svd-components", str(WNMF_DIM),
        "--init-mode", "mkpp",
        "--cluster-metric", "fuzzy",
        "--fitness", "wcss",
        "--cluster-objective", "multi",
        "--train-only", "--eval-split", "random", "--fold", "1",
        "--k", *[str(k) for k in K_LIST],
        "--jobs", str(jobs),
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    print(" ".join(cmd))
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def phase_eval() -> pd.DataFrame:
    from wnmf.wnmf_experiment import (
        RANDOM_SEED,
        _align_assignment_bundle,
        _cluster_avg_predict_kwargs,
        _nearest_centroid_bundle,
        load_assignment,
        load_memberships,
        load_ratings_100k_all,
        run_cluster_average,
    )

    data = str(REPO / "data" / "ml-100k" / "u.data")
    train, test = load_ratings_100k_all(data, random_seed=RANDOM_SEED, fold=1)
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1
    eval_args = Namespace(similarity="cosine", min_common=3)

    rows = []
    for k in K_LIST:
        for algo in ALGOS:
            adir = assign_path(algo, k)
            if not (adir / "assignments.npy").is_file():
                print(f"SKIP eval {algo} K={k} (atama yok)")
                continue
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
                **_cluster_avg_predict_kwargs(eval_args),
                **nc,
                top_n=10, relevance_threshold=4.0, assign_dir=str(adir),
            )
            st = cluster_stats(algo, k)
            rows.append({
                "protocol": "fuzzy_fcm",
                "wnmf_dim": WNMF_DIM,
                "k": k,
                "algo": algo,
                "label": ALGO_LABELS.get(algo, algo),
                "predictor": "cluster_avg",
                "mae": r["mae"],
                "rmse": r["rmse"],
                "ndcg_at_10": r["ndcg_at_10"],
                "precision_at_10": r["precision_at_10"],
                "recall_at_10": r["recall_at_10"],
                "assign_suffix": suffix(k),
                "eval_seconds": round(time.time() - t0, 1),
                **st,
            })
            print(
                f"  {algo} K={k}: MAE={r['mae']:.4f} NDCG={r['ndcg_at_10']:.4f} "
                f"cl_std={st.get('cluster_std', float('nan')):.1f}"
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if OUT_CSV.is_file():
        old = pd.read_csv(OUT_CSV)
        key = ["k", "algo", "predictor", "wnmf_dim"]
        merged = pd.concat([old, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=key, keep="last")
        merged = merged.sort_values(["k", "algo"]).reset_index(drop=True)
        merged.to_csv(OUT_CSV, index=False)
        return merged
    df.to_csv(OUT_CSV, index=False)
    return df


def _plot_k_sweep_panel(ax, df: pd.DataFrame, col: str, ylabel: str, title: str, marker: str) -> None:
    for algo in ALGOS:
        label = ALGO_LABELS.get(algo, algo)
        sub = df[df["algo"] == algo].sort_values("k")
        if sub.empty:
            continue
        c = PLOT_COLORS.get(label, None)
        ax.plot(sub["k"], sub[col], f"{marker}-", label=label, color=c, linewidth=2, markersize=7)
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_LIST)


def phase_plot() -> None:
    if not OUT_CSV.is_file():
        print(f"CSV yok: {OUT_CSV}")
        return
    import matplotlib.pyplot as plt

    df = pd.read_csv(OUT_CSV)
    df = df[df["predictor"] == "cluster_avg"].copy()
    if df.empty:
        print("Grafik için satır yok")
        return

    panels = [
        ("mae", "MAE (cluster_avg)", "MAE vs K", "o"),
        ("rmse", "RMSE", "RMSE vs K", "o"),
        ("ndcg_at_10", "NDCG@10", "NDCG@10 vs K", "s"),
        ("precision_at_10", "Precision@10", "Precision@10 vs K", "D"),
        ("recall_at_10", "Recall@10", "Recall@10 vs K", "D"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("FCM fuzzy — cluster_avg vs K (WNMF20)", fontsize=13, y=1.01)

    for ax, (col, ylabel, title, marker) in zip(axes.flat, panels):
        _plot_k_sweep_panel(ax, df, col, ylabel, title, marker)

    axes.flat[-1].axis("off")

    fig.tight_layout()
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"Grafik: {OUT_PLOT}")
    plt.close(fig)


def report_missing() -> None:
    print("Eksik atamalar:")
    for k in K_LIST:
        miss = [a for a in ALGOS if not has_assign(a, k)]
        ok = [a for a in ALGOS if has_assign(a, k)]
        print(f"  K={k:2d}: OK={len(ok)}/5  missing={miss or '-'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["assign", "eval", "plot", "status", "all"], default="status")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    report_missing()

    if args.phase in ("assign", "all"):
        rc = phase_assign(args.jobs, args.skip_existing)
        if rc != 0:
            sys.exit(rc)
        report_missing()

    if args.phase in ("eval", "all"):
        df = phase_eval()
        print(f"\n-> {OUT_CSV}  ({len(df)} satır)")
        print(df.pivot_table(index="k", columns="label", values="mae", aggfunc="first").to_string())

    if args.phase in ("plot", "all"):
        phase_plot()


if __name__ == "__main__":
    main()
