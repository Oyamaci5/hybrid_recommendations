"""
K=6, WNMF50 euclidean, official fold — B0_KMEANS vs B_AVOA vs B1_HHO.

Meta algoritmalar --early-stop ile converge edene kadar arar (yüksek max epoch).
Atamalar: kmref (--kmeans-refine-overwrite). Tahmin: cluster_avg (--paper-mode).

  python -m experiments.run_k6_wnmf50_meta_compare --phase assign
  python -m experiments.run_k6_wnmf50_meta_compare --phase analyze
  python -m experiments.run_k6_wnmf50_meta_compare --phase eval
  python -m experiments.run_k6_wnmf50_meta_compare --phase all
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO = Path(__file__).resolve().parent.parent
GEN = REPO / "mealpy" / "generate_assignments.py"
EXP = REPO / "wnmf" / "wnmf_experiment.py"

FOLD = 1
K = 6
LATENT = 50
ALGOS = ["B0_KMEANS", "B_AVOA", "B1_HHO"]

# Converge: yüksek tavan, sabırlı early-stop (epoch sınırlamadan önce durur)
ES_MAX = 2000
ES_PATIENCE = 15
ES_BLOCK = 5
ES_TOL = 1e-7
POP = 50

OUT_ANALYZE = REPO / "results" / "k6_wnmf50_meta_cluster_analysis.json"
OUT_EVAL = REPO / "results" / "k6_wnmf50_meta_cluster_avg.csv"


def suffix(*, kmref: bool) -> str:
    base = (
        f"_euc_imkpp_nogs_trainonly_official_f{FOLD}"
        f"_none_wnmf{LATENT}_k{K}_pwcss"
    )
    return f"{base}_kmref" if kmref else base


def assign_roots() -> List[Path]:
    """early-stop çıktısı assignments_estop; yoksa assignments."""
    roots = [
        REPO / "mealpy" / "results" / "assignments_estop",
        REPO / "mealpy" / "results" / "assignments",
    ]
    return [r for r in roots if r.is_dir()]


def find_assign_dir(algo: str) -> Optional[Path]:
    suf_km = suffix(kmref=True)
    suf_plain = suffix(kmref=False)
    for root in assign_roots():
        for suf in (suf_km, suf_plain):
            if algo == "B0_KMEANS" and suf.endswith("_kmref"):
                suf = suf.replace("_kmref", "")
            d = root / "ml100k" / f"{algo}{suf}"
            if (d / "assignments.npy").is_file():
                return d
    return None


def phase_assign(*, skip_existing: bool) -> int:
    cmd = [
        sys.executable, "-u", str(GEN),
        "--dataset", "100k",
        "--algo", *ALGOS,
        "--no-prune", "--no-gray-sheep",
        "--preprocess", "none",
        "--feature-extraction", "wnmf",
        "--svd-components", str(LATENT),
        "--wnmf-epochs", "50",
        "--legacy-wnmf-suffix",
        "--init-mode", "mkpp",
        "--cluster-metric", "euclidean",
        "--fitness", "wcss",
        "--cluster-objective", "wcss",
        "--train-only", "--eval-split", "official",
        "--fold", str(FOLD),
        "--k", str(K),
        "--pop-size", str(POP),
        "--kmeans-refine-overwrite",
        "--kmeans-refine-iter", "300",
        "--early-stop",
        "--early-stop-max-epoch", str(ES_MAX),
        "--early-stop-patience", str(ES_PATIENCE),
        "--early-stop-block", str(ES_BLOCK),
        "--early-stop-tolerance", str(ES_TOL),
        "--jobs", "1",
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    print("ASSIGN:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def _cluster_stats(assignments: np.ndarray) -> dict:
    ids = assignments.astype(int)
    k = int(ids.max()) + 1
    sizes = [int(np.sum(ids == c)) for c in range(k)]
    return {
        "k_active": k,
        "size_min": int(min(sizes)),
        "size_max": int(max(sizes)),
        "size_std": float(np.std(sizes)),
        "sizes": sizes,
    }


def _load_wcss_meta(d: Path) -> Optional[float]:
    meta = d / "run_meta.json"
    if meta.is_file():
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            if "wcss" in m:
                return float(m["wcss"])
        except Exception:
            pass
    sys.path.insert(0, str(REPO / "mealpy"))
    try:
        from mealpy_comparison_v2 import compute_wcss_fast
        sol = d / "best_sol.npy"
        uf = d / "user_features.npy"
        if not sol.is_file():
            uf = d / "wnmf_user_vectors.npy"
        if sol.is_file() and uf.is_file():
            X = np.load(uf)
            wcss, _ = compute_wcss_fast(
                X, np.load(sol), int(np.load(d / "assignments.npy").max()) + 1,
                metric="euclidean",
            )
            return float(wcss)
    except Exception:
        pass
    return None


def _convergence_summary(d: Path) -> dict:
    ch = d / "convergence_history.csv"
    if not ch.is_file():
        return {}
    h = pd.read_csv(ch)
    if h.empty or "fitness" not in h.columns:
        return {}
    f0 = float(h["fitness"].iloc[0])
    f1 = float(h["fitness"].iloc[-1])
    ep = int(h["epoch"].iloc[-1]) if "epoch" in h.columns else len(h) * ES_BLOCK
    return {
        "conv_epochs": ep,
        "conv_blocks": len(h),
        "fitness_init": f0,
        "fitness_final": f1,
        "fitness_gain_pct": (f0 - f1) / f0 * 100 if f0 > 1e-12 else None,
    }


def phase_analyze() -> dict:
    loaded: Dict[str, np.ndarray] = {}
    meta_rows = []
    missing = []
    for algo in ALGOS:
        d = find_assign_dir(algo)
        if d is None:
            missing.append(algo)
            continue
        a = np.load(d / "assignments.npy")
        loaded[algo] = a
        st = _cluster_stats(a)
        conv = _convergence_summary(d)
        meta_rows.append({
            "algo": algo,
            "assign_dir": str(d.relative_to(REPO)),
            "wcss": _load_wcss_meta(d),
            **st,
            **conv,
        })

    pairs = []
    algo_list = list(loaded.keys())
    for i, a1 in enumerate(algo_list):
        for a2 in algo_list[i + 1:]:
            x, y = loaded[a1], loaded[a2]
            same = float(np.mean(x == y))
            pairs.append({
                "a": a1,
                "b": a2,
                "ari": float(adjusted_rand_score(x, y)),
                "nmi": float(normalized_mutual_info_score(x, y)),
                "same_label_pct": same * 100,
            })

    payload = {
        "fold": FOLD,
        "k": K,
        "latent": LATENT,
        "metric": "euclidean",
        "early_stop": {
            "max_epoch": ES_MAX,
            "patience": ES_PATIENCE,
            "block": ES_BLOCK,
            "tol": ES_TOL,
        },
        "algos": meta_rows,
        "pairs": pairs,
        "missing": missing,
    }
    OUT_ANALYZE.parent.mkdir(parents=True, exist_ok=True)
    OUT_ANALYZE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n-> {OUT_ANALYZE}")
    for r in meta_rows:
        print(
            f"  {r['algo']}: sizes={r.get('sizes')} wcss={r.get('wcss')} "
            f"conv_ep={r.get('conv_epochs', '-')}",
            flush=True,
        )
    for p in pairs:
        print(
            f"  {p['a']} vs {p['b']}: ARI={p['ari']:.4f} "
            f"same={p['same_label_pct']:.1f}%",
            flush=True,
        )
    if missing:
        print(f"  EKSIK: {missing}", file=sys.stderr)
    return payload


def phase_eval() -> pd.DataFrame:
    root = None
    for r in assign_roots():
        if (r / "ml100k").is_dir():
            root = r
            break
    if root is None:
        print("Atama kökü yok", file=sys.stderr)
        return pd.DataFrame()

    suf = suffix(kmref=True)
    cmd = [
        sys.executable, "-u", str(EXP),
        "--dataset", "100k",
        "--eval-split", "official",
        "--fold", str(FOLD),
        "--mode", "baselines",
        "--paper-mode",
        "--no-global", "--no-cluster-knn",
        "--k", str(K),
        "--algo", *ALGOS,
        "--assign-root", str(root.as_posix()),
        "--assign-suffix", suf,
        "--top-n", "10",
        "--relevance-threshold", "4.0",
    ]
    print("EVAL:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    if rc != 0:
        return pd.DataFrame()

    base = REPO / "results" / "wnmf" / "ml100k" / f"k{K}" / f"fold{FOLD}"
    cands = sorted(
        base.rglob("wnmf_results_ml100k_k*_baselines.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    rows = []
    if cands:
        lines = cands[0].read_text(encoding="utf-8", errors="replace").splitlines()
        data = [ln for ln in lines if not ln.startswith("#")]
        for row in pd.read_csv(io.StringIO("\n".join(data))).to_dict("records"):
            if row.get("scenario") != "calc_avg_rating":
                continue
            if row.get("algo_label") not in ALGOS:
                continue
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        OUT_EVAL.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_EVAL, index=False)
        print(f"\n-> {OUT_EVAL}")
        print(df[["algo_label", "mae", "rmse", "ndcg_at_10", "precision_at_10"]].to_string(index=False))
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["assign", "analyze", "eval", "all"], default="all")
    ap.add_argument("--skip-existing", action="store_true", default=False)
    args = ap.parse_args()

    if args.phase in ("assign", "all"):
        rc = phase_assign(skip_existing=args.skip_existing)
        if rc != 0:
            return rc

    analysis = {}
    if args.phase in ("analyze", "all"):
        analysis = phase_analyze()

    if args.phase in ("eval", "all"):
        pairs = analysis.get("pairs", [])
        meaningful = any(p.get("ari", 1.0) < 0.95 for p in pairs) if pairs else True
        if not meaningful and pairs:
            print(
                "UYARI: ARI hepsi >=0.95 — küme farkı zayıf; eval yine de çalıştırılıyor.",
                file=sys.stderr,
            )
        phase_eval()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
