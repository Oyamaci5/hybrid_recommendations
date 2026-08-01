"""
Thakrar (2025) — repodaki generate_assignments + wnmf_experiment ile makale replikasyonu.

Başarı kriteri: makale Fig. 2–5 eğilimleri (K=14 minimum, Alg.2 > baseline I, L sweep),
mutlak MAE'nin 0.81 altına inmesi değil. Makale metninde mutlak MAE yok; figürlerden okunmalı.

  python -m experiments.run_thakrar_repo_pipeline
  python -m experiments.run_thakrar_repo_pipeline --k-sweep
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
MEALPY = REPO / "mealpy"
WNMF = REPO / "wnmf"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(MEALPY))
sys.path.insert(0, str(WNMF))

from mealpy.evolutionary_based import GA  # noqa: E402
from mealpy_comparison_v2 import get_all_algorithms_v3  # noqa: E402
from generate_assignments import (  # noqa: E402
    ALGO_CONFIG,
    run_one,
    wnmf_feature_extract,
)
from wnmf_experiment import run_cluster_average  # noqa: E402

from experiments.run_thakrar_bigcomp_protocol import (  # noqa: E402
    RATINGS,
    SEED,
    load_latest_small,
    matrix_factorization,
    split_80_20,
)

PAPER_K_OPT = 14
PAPER_K_SWEEP = [2, 5, 8, 11, 14, 17, 19]
OUT_DIR = REPO / "results" / "thakrar_repo_pipeline"
ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments" / "ml-latest-small"


def _dense_train_matrix(train: np.ndarray, n_users: int, n_items: int) -> np.ndarray:
    m = np.zeros((n_users, n_items), dtype=np.float32)
    for u, i, r in train:
        m[int(u), int(i)] = float(r)
    return m


def _build_algo_map():
    algo_map = {a["full_name"]: a for a in get_all_algorithms_v3()}
    try:
        from ga_hho_optimizer import OriginalGAHHO  # noqa: E402
    except ImportError:
        from ga_hho import OriginalGAHHO  # noqa: E402
    algo_map["GAHHO.OriginalGAHHO"] = {
        "full_name": "GAHHO.OriginalGAHHO",
        "class": OriginalGAHHO,
    }
    algo_map["GA.EliteMultiGA"] = {
        "full_name": "GA.EliteMultiGA",
        "class": GA.EliteMultiGA,
    }
    return algo_map


def _make_gen_args(
    *,
    kmeans_overwrite: bool,
    baseline_epoch: int,
    pop_size: int,
    latent_dim: int,
    mf_epochs: int,
) -> Namespace:
    return Namespace(
        kmeans_refine=True,
        kmeans_refine_overwrite=kmeans_overwrite,
        kmeans_refine_iter=300,
        fitness="wcss",
        cluster_objective="wcss",
        feature_extraction="wnmf",
        wnmf_features=latent_dim,
        svd_components=latent_dim,
        preprocess="none",
        wnmf_epochs=mf_epochs,
        wnmf_init="inmed",
        inmed_trim_low=5.0,
        inmed_trim_high=95.0,
        legacy_wnmf_suffix=True,
        baseline_epoch=baseline_epoch,
        pop_size=pop_size,
        early_stop=False,
        fcm=False,
        centroid_iter=None,
        centroid_agents=None,
    )


def step1_features(
    train: np.ndarray,
    n_users: int,
    n_items: int,
    *,
    mode: str,
    latent_dim: int,
    mf_epochs: int,
    lr: float,
    reg: float,
) -> np.ndarray:
    """Adım 1: P matrisi — paper MF veya repo WNMF."""
    if mode == "paper_mf":
        p, _ = matrix_factorization(
            train, n_users, n_items,
            latent_dim=latent_dim, n_epochs=mf_epochs, lr=lr, reg=reg,
        )
        return p
    dense = _dense_train_matrix(train, n_users, n_items)
    return wnmf_feature_extract(
        dense,
        n_components=latent_dim,
        n_epochs=mf_epochs,
        lr=lr,
        reg=reg,
        random_seed=SEED,
    )


def step2_assignment(
    features: np.ndarray,
    k: int,
    save_dir: Path,
    args: Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    """Adım 2: generate_assignments.run_one (B1_HHO + kmref)."""
    label, g_name, l_name = ALGO_CONFIG[1]  # B1_HHO
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_one(
        label, g_name, l_name,
        features.astype(np.float32),
        k, SEED, str(save_dir),
        _build_algo_map(),
        use_lof=False,
        args=args,
        cluster_metric="euclidean",
        init_mode="mkpp",
        disable_gray_sheep=True,
    )
    assignments = np.load(save_dir / "assignments.npy")
    gray = np.load(save_dir / "gray_sheep_mask.npy")
    return assignments, gray


def step3_eval(
    train: np.ndarray,
    test: np.ndarray,
    assignments: np.ndarray,
    gray_mask: np.ndarray,
    n_items: int,
    *,
    paper_global_fallback: bool,
) -> dict:
    """Adım 3: wnmf_experiment.run_cluster_average (CalcAvgRating)."""
    row = run_cluster_average(
        train, test, assignments, gray_mask, None, n_items, "B1_HHO",
        cluster_avg_hard=True,
        cluster_avg_global_fallback=paper_global_fallback,
        similarity="pearson",
    )
    return row


def _variant_grid() -> list[dict]:
    """Önce makale-optimum, sonra iyileştirme adayları."""
    base = dict(
        k=14, latent_dim=10, mf_epochs=50, lr=0.01, reg=0.01,
        mf_mode="paper_mf", hho_epoch=100, pop_size=30,
        kmref=True, paper_global_fallback=True,
    )
    variants = [dict(base, name="paper_baseline")]
    for hho_epoch, pop in [(200, 50), (150, 40), (100, 50)]:
        variants.append(dict(
            base, name=f"hho_e{hho_epoch}_p{pop}",
            hho_epoch=hho_epoch, pop_size=pop,
        ))
    for mf_epochs in [75, 100, 150]:
        variants.append(dict(
            base, name=f"mf_T{mf_epochs}", mf_epochs=mf_epochs,
        ))
    for ld in [5, 15]:
        variants.append(dict(base, name=f"L{ld}", latent_dim=ld))
    for k in [9, 19, 12, 16]:
        if k != 14:
            variants.append(dict(base, name=f"K{k}", k=k))
    variants.append(dict(base, name="wnmf_L10", mf_mode="wnmf"))
    variants.append(dict(
        base, name="wnmf_kmref_off", mf_mode="wnmf", kmref=False,
    ))
    variants.append(dict(
        base, name="user_mean_fb", paper_global_fallback=False,
    ))
    return variants


def run_variant(
    train: np.ndarray,
    test: np.ndarray,
    n_users: int,
    n_items: int,
    cfg: dict,
) -> dict:
    t0 = time.time()
    tag = (
        f"B1_HHO_thakrar_{cfg['name']}_"
        f"L{cfg['latent_dim']}_k{cfg['k']}_"
        f"{'kmref' if cfg['kmref'] else 'noref'}"
    )
    save_dir = ASSIGN_ROOT / tag
    gen_args = _make_gen_args(
        kmeans_overwrite=cfg["kmref"],
        baseline_epoch=cfg["hho_epoch"],
        pop_size=cfg["pop_size"],
        latent_dim=cfg["latent_dim"],
        mf_epochs=cfg["mf_epochs"],
    )
    features = step1_features(
        train, n_users, n_items,
        mode=cfg["mf_mode"],
        latent_dim=cfg["latent_dim"],
        mf_epochs=cfg["mf_epochs"],
        lr=cfg["lr"],
        reg=cfg["reg"],
    )
    assignments, gray = step2_assignment(features, cfg["k"], save_dir, gen_args)
    eval_row = step3_eval(
        train, test, assignments, gray, n_items,
        paper_global_fallback=cfg["paper_global_fallback"],
    )
    return {
        "variant": cfg["name"],
        "mf_mode": cfg["mf_mode"],
        "k": cfg["k"],
        "latent_dim": cfg["latent_dim"],
        "mf_epochs": cfg["mf_epochs"],
        "hho_epoch": cfg["hho_epoch"],
        "pop_size": cfg["pop_size"],
        "kmref": cfg["kmref"],
        "paper_global_fallback": cfg["paper_global_fallback"],
        "mae": eval_row["mae"],
        "rmse": eval_row["rmse"],
        "scenario": eval_row["scenario"],
        "assign_dir": str(save_dir),
        "seconds": time.time() - t0,
    }


def _paper_trend_check(k_rows: list[dict]) -> dict:
    """Makale iddiaları: K=14 MAE minimum; K=2→K=14 ~%10 iyileşme (Fig. 2)."""
    by_k = {int(r["k"]): float(r["mae"]) for r in k_rows if not np.isnan(r.get("mae", float("nan")))}
    if PAPER_K_OPT not in by_k or 2 not in by_k:
        return {"status": "insufficient_data", "by_k": by_k}
    mae_k2, mae_k14 = by_k[2], by_k[PAPER_K_OPT]
    best_k = min(by_k, key=by_k.get)
    rel_pct = (mae_k2 - mae_k14) / mae_k2 * 100.0
    return {
        "by_k": by_k,
        "mae_k2": mae_k2,
        "mae_k14": mae_k14,
        "best_k": best_k,
        "best_mae": by_k[best_k],
        "k14_is_minimum": best_k == PAPER_K_OPT,
        "k2_to_k14_rel_pct": rel_pct,
        "paper_claims_k14_min": True,
        "paper_claims_k2_to_k14_drop_pct": 10.0,
    }


def run_k_sweep(
    train: np.ndarray,
    test: np.ndarray,
    n_users: int,
    n_items: int,
    *,
    cfg: dict | None = None,
) -> list[dict]:
    base = dict(
        latent_dim=10, mf_epochs=50, lr=0.01, reg=0.01,
        mf_mode="paper_mf", hho_epoch=100, pop_size=30,
        kmref=True, paper_global_fallback=True,
    )
    if cfg:
        base.update(cfg)
    rows = []
    for k in PAPER_K_SWEEP:
        row_cfg = dict(base, k=k, name=f"K{k}")
        print(f"K-sweep K={k} ...", flush=True)
        row = run_variant(train, test, n_users, n_items, row_cfg)
        print(f"  MAE={row['mae']:.4f} RMSE={row['rmse']:.4f}", flush=True)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-variants", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--k-sweep",
        action="store_true",
        help="Yalnızca makale K grid'i (Fig. 2) — eğilim doğrulaması",
    )
    args = ap.parse_args()

    rows_raw, _, n_users, n_items = load_latest_small(RATINGS)
    train, test = split_80_20(rows_raw)
    print(f"Repo pipeline | ml-latest-small | train={len(train):,} test={len(test):,}")
    print("Replikasyon: makale eğilimleri (Fig. 2–5), mutlak MAE eşiği yok.\n")

    if args.k_sweep:
        results = run_k_sweep(train, test, n_users, n_items)
        trend = _paper_trend_check(results)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(results)
        csv_path = OUT_DIR / "k_sweep_results.csv"
        df.to_csv(csv_path, index=False)
        summary = {"mode": "k_sweep", "paper_trend": trend}
        (OUT_DIR / "k_sweep_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8",
        )
        print(df[["k", "mae", "rmse"]].to_string(index=False))
        print(f"\n-> {csv_path}")
        if trend.get("status") != "insufficient_data":
            print(
                f"\nK=2 MAE={trend['mae_k2']:.4f}  K=14 MAE={trend['mae_k14']:.4f}  "
                f"K2→K14 rel={trend['k2_to_k14_rel_pct']:+.1f}%  "
                f"(makale ~−10%)"
            )
            print(
                f"Minimum K={trend['best_k']} (MAE={trend['best_mae']:.4f})  "
                f"K=14 minimum? {trend['k14_is_minimum']}"
            )
        return

    variants = _variant_grid()
    if args.max_variants > 0:
        variants = variants[: args.max_variants]

    results = []
    best_mae = float("inf")
    best_row = None

    for i, cfg in enumerate(variants, 1):
        print(f"[{i}/{len(variants)}] {cfg['name']} ...", flush=True)
        try:
            row = run_variant(train, test, n_users, n_items, cfg)
        except Exception as exc:
            print(f"  HATA: {exc}", flush=True)
            row = {"variant": cfg["name"], "mae": float("nan"), "error": str(exc)}
        results.append(row)
        mae = row.get("mae", float("nan"))
        if not np.isnan(mae):
            print(
                f"  MAE={mae:.4f} RMSE={row.get('rmse', float('nan')):.4f}  "
                f"scenario={row.get('scenario')}  ({row.get('seconds', 0):.0f}s)",
                flush=True,
            )
            if mae < best_mae:
                best_mae = mae
                best_row = row

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = OUT_DIR / "sweep_results.csv"
    df.to_csv(csv_path, index=False)

    k_rows = [r for r in results if r.get("variant", "").startswith("K") or r.get("variant") == "paper_baseline"]
    trend = _paper_trend_check(
        [{"k": r.get("k", PAPER_K_OPT), "mae": r.get("mae")} for r in k_rows if "k" in r]
    )
    paper_baseline = next((r for r in results if r.get("variant") == "paper_baseline"), None)

    summary = {
        "mode": "variant_grid",
        "paper_baseline_mae": paper_baseline.get("mae") if paper_baseline else None,
        "best_mae": best_mae,
        "best_variant": best_row,
        "paper_trend_partial": trend,
        "note": "Success = match paper figure trends, not MAE below arbitrary threshold.",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(df.to_string(index=False))
    print(f"\n-> {csv_path}")
    if paper_baseline:
        print(
            f"\nMakale baseline (K=14, L=10): MAE={paper_baseline['mae']:.4f}  "
            f"— figürlerden okunan referansla karşılaştırın."
        )
    if best_row and best_row.get("variant") != "paper_baseline":
        print(
            f"En düşük MAE varyant: {best_row.get('variant')} MAE={best_mae:.4f} "
            f"(makale replikasyonu için en düşük MAE hedef değil)."
        )


if __name__ == "__main__":
    main()
