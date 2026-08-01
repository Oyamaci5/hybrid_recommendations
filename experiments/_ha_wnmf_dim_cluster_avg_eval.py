"""HA fuzzy k10: WNMF latent sweep -> cluster_avg downstream eval."""
from __future__ import annotations

import sys
import time
from argparse import Namespace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "wnmf"))

from wnmf_experiment import (  # noqa: E402
    RANDOM_SEED,
    _align_assignment_bundle,
    _cluster_avg_predict_kwargs,
    _nearest_centroid_bundle,
    load_assignment,
    load_memberships,
    load_ratings_100k_all,
    run_cluster_average,
)

ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments" / "ml100k"
DATA = REPO / "data" / "ml-100k" / "u.data"
OUT = REPO / "results" / "ha_wnmf_dim_sweep_cluster_avg_k10.csv"


def assign_suffix(dim: int) -> str:
    if dim == 20:
        return "_fuzzy_imkpp_nogs_trainonly_rand_f1_none_wnmf20_k10"
    return f"_fuzzy_imkpp_nogs_trainonly_rand_f1_wnmfep50_none_wnmf{dim}_k10"


def main() -> None:
    train, test = load_ratings_100k_all(str(DATA), random_seed=RANDOM_SEED, fold=1)
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    n_users = int(max(train[:, 0].max(), test[:, 0].max())) + 1
    eval_args = Namespace(similarity="cosine", min_common=3)

    rows = []
    for dim in [20, 25, 50, 75, 100, 125]:
        adir = ASSIGN_ROOT / f"HA_AVOAHGS{assign_suffix(dim)}"
        if not (adir / "assignments.npy").is_file():
            print(f"SKIP wnmf{dim}: missing {adir}")
            continue

        assignments, gray_mask = load_assignment(str(adir))
        memberships = load_memberships(str(adir))
        assignments, gray_mask, memberships, _ = _align_assignment_bundle(
            assignments,
            gray_mask,
            memberships,
            None,
            n_users_expected=n_users,
            algo_label="HA_AVOAHGS",
            assign_dir=str(adir),
        )
        nc = _nearest_centroid_bundle(None, str(adir), assignments)
        t0 = time.time()
        r = run_cluster_average(
            train,
            test,
            assignments,
            gray_mask,
            memberships,
            n_items,
            "HA_AVOAHGS",
            top_n=10,
            relevance_threshold=4.0,
            assign_dir=str(adir),
            **_cluster_avg_predict_kwargs(eval_args),
            **nc,
        )
        elapsed = time.time() - t0
        rows.append(
            {
                "wnmf_dim": dim,
                "mae": r["mae"],
                "rmse": r["rmse"],
                "ndcg_at_10": r["ndcg_at_10"],
                "precision_at_10": r.get("precision_at_10"),
                "time_s": round(elapsed, 1),
            }
        )
        print(
            f"wnmf{dim:3d}  MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  "
            f"NDCG@10={r['ndcg_at_10']:.4f}  ({elapsed:.1f}s)"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        f.write("wnmf_dim,mae,rmse,ndcg_at_10,precision_at_10,time_s\n")
        for row in rows:
            f.write(
                f"{row['wnmf_dim']},{row['mae']:.6f},{row['rmse']:.6f},"
                f"{row['ndcg_at_10']:.6f},{row['precision_at_10']:.6f},"
                f"{row['time_s']:.1f}\n"
            )
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
