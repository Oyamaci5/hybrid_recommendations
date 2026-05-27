"""
İki cluster-kNN koşusu arasında paired bootstrap CI ve Wilcoxon signed-rank testi.

Aynı test rating'leri üzerinde iki assignment/algoritma tahminlerini karşılaştırır.
Δ = Metrik(A) − Metrik(B); negatif Δ → A daha iyi.

Wilcoxon: rating bazlı paired farklar uzerinde (normallik varsayimi yok).
  MAE  → |e_a| - |e_b|
  RMSE → (e_a)^2 - (e_b)^2  (kare hata farki; RMSE ile tutarli)

Örnek (WNMF20 K=27 kNN=30, run22 ayarları):
  python mealpy/paired_bootstrap_ci.py \\
    --algo-a HA_AVOAHGS --algo-b B1_HHO \\
    --k 27 --knn 30 --fold 1 \\
    --assign-root mealpy/results/assignments \\
    --assign-suffix _euc_imkpp_nogs_none_wnmf20_k27_kmref \\
    --similarity cosine

Referansa karşı tüm algoritmalar:
  python mealpy/paired_bootstrap_ci.py \\
    --reference B0_KMEANS \\
    --algos B0_KMEANS B1_HHO IWO_HHO HA_AVOAHGS B_AVOA \\
    --k 27 --knn 30 --fold 1 \\
    --assign-suffix _euc_imkpp_nogs_none_wnmf20_k27_kmref
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import wilcoxon

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "wnmf"))

from wnmf.wnmf_experiment import (  # noqa: E402
    DATA_100K_ALL,
    DATA_100K_TEST,
    DATA_100K_TRAIN,
    RANDOM_SEED,
    _ASSIGN_KMREF_SUFFIX,
    _algo_assignment_dir,
    _assign_suffix_strip_trailing_kmref,
    run_cluster_knn,
)
from wnmf.wnmf_utils import (  # noqa: E402
    load_assignment,
    load_memberships,
    load_ratings_100k,
    load_ratings_100k_all,
)


def resolve_assign_dir(
    assign_root: str,
    label: str,
    k: int,
    assign_suffix: str,
) -> str:
    """wnmf_experiment ile uyumlu assignment klasörü (B0 kmref istisnası dahil)."""
    assign_dir = _algo_assignment_dir(
        assign_root, "ml100k", label, k, assign_suffix=assign_suffix,
    )
    if not os.path.isdir(assign_dir) and label == "B0_KMEANS":
        alt = _assign_suffix_strip_trailing_kmref(assign_suffix)
        if alt is not None:
            cand = _algo_assignment_dir(
                assign_root, "ml100k", label, k, assign_suffix=alt,
            )
            if os.path.isdir(cand):
                assign_dir = cand
    if not os.path.isdir(assign_dir):
        kmref_try = assign_dir + _ASSIGN_KMREF_SUFFIX
        if os.path.isdir(kmref_try):
            assign_dir = kmref_try
    if not os.path.isdir(assign_dir):
        raise FileNotFoundError(f"Assignment yok: {assign_dir}")
    return assign_dir


def load_split(eval_split: str, fold: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    if eval_split == "random":
        return load_ratings_100k_all(DATA_100K_ALL, random_seed=RANDOM_SEED, fold=fold)
    return load_ratings_100k(DATA_100K_TRAIN, DATA_100K_TEST, fold=fold or 1)


def predict_eval_rows(
    train: np.ndarray,
    test: np.ndarray,
    algo: str,
    assign_dir: str,
    *,
    similarity: str,
    knn: int,
    min_common: int,
) -> np.ndarray:
    """eval_rows: (n, 4) → user, item, true, pred."""
    assignments, gray_mask = load_assignment(assign_dir)
    memberships = load_memberships(assign_dir)
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    row = run_cluster_knn(
        train,
        test,
        assignments,
        gray_mask,
        memberships,
        n_items,
        algo,
        similarity=similarity,
        min_common=min_common,
        k_neighbors=knn,
        assign_dir=assign_dir,
        return_eval_rows=True,
    )
    return np.asarray(row["eval_rows"], dtype=np.float64)


def align_eval_rows(rows_a: np.ndarray, rows_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(u,i,true,pred) satırlarını ortak (u,i) anahtarına hizala."""
    key = lambda r: (int(r[0]), int(r[1]))
    map_a = {key(r): float(r[3]) for r in rows_a}
    map_b = {key(r): float(r[3]) for r in rows_b}
    common = sorted(set(map_a) & set(map_b))
    if not common:
        raise ValueError("Ortak test çifti yok.")
    trues, pa, pb = [], [], []
    true_lookup = {key(r): float(r[2]) for r in rows_a}
    for k in common:
        trues.append(true_lookup[k])
        pa.append(map_a[k])
        pb.append(map_b[k])
    return (
        np.asarray(trues, dtype=np.float64),
        np.asarray(pa, dtype=np.float64),
        np.asarray(pb, dtype=np.float64),
    )


def paired_bootstrap_delta(
    true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric: str,
    n_boot: int,
    seed: int,
    ci: float,
) -> Dict[str, float]:
    """
    Δ = metric(A) − metric(B) için paired bootstrap CI.
    MAE/RMSE: satır bazlı bootstrap (aynı indeksler A ve B'de).
    """
    true = np.asarray(true, dtype=np.float64)
    pred_a = np.asarray(pred_a, dtype=np.float64)
    pred_b = np.asarray(pred_b, dtype=np.float64)
    n = len(true)
    if n < 2:
        raise ValueError("En az 2 ortak test rating gerekli.")

    err_a = np.abs(true - pred_a)
    err_b = np.abs(true - pred_b)
    sq_a = (true - pred_a) ** 2
    sq_b = (true - pred_b) ** 2

    if metric == "mae":
        point = float(err_a.mean() - err_b.mean())
        boot_fn = lambda idx: (err_a[idx].mean() - err_b[idx].mean())
    elif metric == "rmse":
        point = float(np.sqrt(sq_a.mean()) - np.sqrt(sq_b.mean()))
        boot_fn = lambda idx: (
            np.sqrt(sq_a[idx].mean()) - np.sqrt(sq_b[idx].mean())
        )
    else:
        raise ValueError(f"metric bilinmiyor: {metric!r} (mae | rmse)")

    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[b] = boot_fn(idx)

    alpha = (100.0 - ci) / 2.0
    lo, hi = np.percentile(samples, [alpha, 100.0 - alpha])
    p_better = float(np.mean(samples < 0.0))  # A daha iyi olasılığı (Δ<0)
    return {
        "delta": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_a_better": p_better,
        "n_pairs": float(n),
        "mae_a": float(err_a.mean()),
        "mae_b": float(err_b.mean()),
    }


def paired_wilcoxon(
    true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric: str,
    alpha: float,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Paired Wilcoxon signed-rank: H0 medyan fark = 0.
    metric=mae  -> d_i = |true_i - pred_a_i| - |true_i - pred_b_i|
    metric=rmse -> d_i = (true_i - pred_a_i)^2 - (true_i - pred_b_i)^2
    """
    true = np.asarray(true, dtype=np.float64)
    pred_a = np.asarray(pred_a, dtype=np.float64)
    pred_b = np.asarray(pred_b, dtype=np.float64)

    if metric == "mae":
        diff = np.abs(true - pred_a) - np.abs(true - pred_b)
    elif metric == "rmse":
        diff = (true - pred_a) ** 2 - (true - pred_b) ** 2
    else:
        raise ValueError(f"metric bilinmiyor: {metric!r} (mae | rmse)")

    n_zero = int(np.sum(diff == 0.0))
    n_nonzero = int(len(diff) - n_zero)
    if n_nonzero < 1:
        return {
            "median_diff": float(np.median(diff)),
            "statistic": float("nan"),
            "p_value": float("nan"),
            "significant": 0.0,
            "n_nonzero": float(n_nonzero),
            "n_zero": float(n_zero),
        }

    res = wilcoxon(
        diff,
        alternative=alternative,
        zero_method="wilcox",
        method="auto",
    )
    p = float(res.pvalue)
    return {
        "median_diff": float(np.median(diff)),
        "statistic": float(res.statistic),
        "p_value": p,
        "significant": float(p < alpha),
        "n_nonzero": float(n_nonzero),
        "n_zero": float(n_zero),
    }


def _bootstrap_note(delta_lo: float, delta_hi: float, algo_a: str, algo_b: str) -> str:
    if delta_hi < 0:
        return f"{algo_a} anlamli daha iyi (bootstrap)"
    if delta_lo > 0:
        return f"{algo_b} anlamli daha iyi (bootstrap)"
    return "CI 0'i iceriyor -> belirsiz (bootstrap)"


def _wilcoxon_note(
    median_diff: float,
    p_value: float,
    alpha: float,
    algo_a: str,
    algo_b: str,
) -> str:
    if np.isnan(p_value):
        return "Wilcoxon uygulanamadi"
    if p_value >= alpha:
        return f"p>={alpha:g} -> belirsiz (Wilcoxon)"
    if median_diff < 0:
        return f"{algo_a} anlamli daha iyi (Wilcoxon)"
    if median_diff > 0:
        return f"{algo_b} anlamli daha iyi (Wilcoxon)"
    return f"p<{alpha:g} ama medyan fark=0 (Wilcoxon)"


def compare_pair(
    train: np.ndarray,
    test: np.ndarray,
    algo_a: str,
    algo_b: str,
    assign_root: str,
    k: int,
    assign_suffix: str,
    *,
    similarity: str,
    knn: int,
    min_common: int,
    n_boot: int,
    seed: int,
    ci: float,
    alpha: float,
    wilcoxon_alternative: str,
    cache: Dict[str, np.ndarray],
) -> None:
    for algo in (algo_a, algo_b):
        if algo not in cache:
            d = resolve_assign_dir(assign_root, algo, k, assign_suffix)
            print(f"  [{algo}] tahmin toplanıyor… ({d})")
            cache[algo] = predict_eval_rows(
                train, test, algo, d,
                similarity=similarity, knn=knn, min_common=min_common,
            )

    true, pa, pb = align_eval_rows(cache[algo_a], cache[algo_b])
    print(f"\nKarşılaştırma: {algo_a} vs {algo_b}  (n={len(true):,} ortak test çifti)")
    print(f"  MAE  {algo_a}: {np.abs(true - pa).mean():.4f}  |  {algo_b}: {np.abs(true - pb).mean():.4f}")
    print(f"  RMSE {algo_a}: {np.sqrt(((true - pa) ** 2).mean()):.4f}  |  "
          f"{algo_b}: {np.sqrt(((true - pb) ** 2).mean()):.4f}")
    print()
    print(f"  Delta = Metrik({algo_a}) - Metrik({algo_b})  |  {ci:.0f}% paired bootstrap CI")
    print(f"  {'Metrik':<8} {'Delta':>10} {'CI alt':>10} {'CI ust':>10} {'P(A<B)':>10}  Yorum")
    print("  " + "-" * 62)

    for metric in ("mae", "rmse"):
        stats = paired_bootstrap_delta(
            true, pa, pb, metric, n_boot=n_boot, seed=seed, ci=ci,
        )
        lo, hi = stats["ci_lo"], stats["ci_hi"]
        note = _bootstrap_note(lo, hi, algo_a, algo_b)
        print(
            f"  {metric.upper():<8} {stats['delta']:>10.4f} {lo:>10.4f} {hi:>10.4f} "
            f"{stats['p_a_better']:>9.1%}  {note}"
        )

    print()
    print(
        f"  Wilcoxon signed-rank (paired, alpha={alpha:g}, "
        f"alternative={wilcoxon_alternative})"
    )
    print(
        f"  {'Metrik':<8} {'Med.fark':>10} {'W-stat':>12} {'p-value':>12}  Yorum"
    )
    print("  " + "-" * 62)
    for metric in ("mae", "rmse"):
        wx = paired_wilcoxon(
            true, pa, pb, metric,
            alpha=alpha,
            alternative=wilcoxon_alternative,
        )
        p_str = f"{wx['p_value']:.2e}" if not np.isnan(wx["p_value"]) else "n/a"
        w_str = f"{wx['statistic']:.1f}" if not np.isnan(wx["statistic"]) else "n/a"
        note = _wilcoxon_note(
            wx["median_diff"], wx["p_value"], alpha, algo_a, algo_b,
        )
        print(
            f"  {metric.upper():<8} {wx['median_diff']:>10.4f} {w_str:>12} "
            f"{p_str:>12}  {note}"
        )
        if metric == "mae" and int(wx["n_zero"]) > 0:
            print(f"  (Wilcoxon: {int(wx['n_zero']):,} sifir farkli cift elendi)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cluster-kNN paired bootstrap CI + Wilcoxon signed-rank testi.",
    )
    p.add_argument("--algo-a", default=None, help="Birinci algoritma (A)")
    p.add_argument("--algo-b", default=None, help="İkinci algoritma (B)")
    p.add_argument(
        "--reference",
        default=None,
        help="Tüm --algos ile referans karşılaştırması (algo-a/b yerine)",
    )
    p.add_argument(
        "--algos",
        nargs="+",
        default=["B0_KMEANS", "B1_HHO", "IWO_HHO", "HA_AVOAHGS", "B_AVOA"],
    )
    p.add_argument("--k", type=int, default=27)
    p.add_argument("--knn", type=int, default=30)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--eval-split", choices=["random", "official"], default="random")
    p.add_argument(
        "--assign-root",
        default=os.path.join("mealpy", "results", "assignments"),
    )
    p.add_argument(
        "--assign-suffix",
        default="_euc_imkpp_nogs_none_wnmf20_k27_kmref",
    )
    p.add_argument("--similarity", default="cosine")
    p.add_argument("--min-common", type=int, default=3)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ci", type=float, default=95.0, help="Bootstrap guven araligi (%%)")
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Wilcoxon anlamlilik esigi (varsayilan: 0.05)",
    )
    p.add_argument(
        "--wilcoxon-alternative",
        choices=["two-sided", "less", "greater"],
        default="two-sided",
        help="Wilcoxon H1: two-sided | less (A<B) | greater (A>B)",
    )
    args = p.parse_args()

    assign_root = os.path.normpath(os.path.join(REPO, args.assign_root))
    print(f"Eval: {args.eval_split}, fold={args.fold}, K={args.k}, kNN={args.knn}")
    print(f"Suffix: {args.assign_suffix}")
    train, test = load_split(args.eval_split, args.fold)
    print()

    cache: Dict[str, np.ndarray] = {}

    if args.reference is not None:
        ref = args.reference
        others = [a for a in args.algos if a != ref]
        if ref not in args.algos:
            others = list(args.algos)
        for other in others:
            compare_pair(
                train, test, other, ref, assign_root, args.k, args.assign_suffix,
                similarity=args.similarity,
                knn=args.knn,
                min_common=args.min_common,
                n_boot=args.n_bootstrap,
                seed=args.seed,
                ci=args.ci,
                alpha=args.alpha,
                wilcoxon_alternative=args.wilcoxon_alternative,
                cache=cache,
            )
            print()
        return

    if not args.algo_a or not args.algo_b:
        p.error("--algo-a ve --algo-b verin, veya --reference kullanın")
    compare_pair(
        train, test, args.algo_a, args.algo_b, assign_root, args.k, args.assign_suffix,
        similarity=args.similarity,
        knn=args.knn,
        min_common=args.min_common,
        n_boot=args.n_bootstrap,
        seed=args.seed,
        ci=args.ci,
        alpha=args.alpha,
        wilcoxon_alternative=args.wilcoxon_alternative,
        cache=cache,
    )


if __name__ == "__main__":
    main()
