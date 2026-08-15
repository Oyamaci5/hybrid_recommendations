"""
PACR — Prediction-Aligned Center Refinement (onerilen yontem / makale katkisi).

Fikir: KMeans++ merkezleri geometrik gorevde (WCSS, yaklasik-NN) yenilmez (K10/P2
bulgusu). Ama NIHAI SISTEM hedefi geometrik degil: sabit-butceli havuz + kNN + MF
karisiminin dogrulugu. Lloyd bu hedefi optimize EDEMEZ (turev/kapali form yok).
PACR: B0 merkezlerinden basla (warm start) -> meta-sezgisel, merkezleri dogrudan
ic-val karisim-MAE'sine gore rafine etsin -> ayni butceyle test et.

Esitlik garantileri:
  - Havuz butcesi SABIT (B=%25, kirpmali) -> maliyet esit, tek degisken merkezler.
  - Warm start B0 -> meta, fitness'ta B0'dan kotuye gidemez.
  - Basari olcutu: test'te B0'i gecmek + algoritmalar arasi fark.

Kullanim (algo basina bir cagri; sonuclar birikir):
  python algo_selection_v2/pacr.py --algo AVOA.OriginalAVOA
  python algo_selection_v2/pacr.py --report        # tablo
Cikti: results/pacr.csv, cache: results/pacr_cache.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))
from budget_pool import budget_pools, eval_budget  # noqa: E402
from eksikler_deney import SEED, Ctx  # noqa: E402
from genre_k_deney import build_space, genre_profile  # noqa: E402
from pred_v2 import build_sims, full_metrics  # noqa: E402
from recompute_scores import discover_algorithms  # noqa: E402
from sampiyon import als_fit  # noqa: E402

K, KNN_K, BETA = 10, 20, 0.4
B_PCT = 25
NFE, POP = 300, 15
VAL_N = 2000


def get_ctx_and_cache():
    data = Path(ROOT.parent / "data" / "ml-100k")
    ctx = Ctx(data)
    X = build_space(ctx, genre_profile(ctx, data), "nmf+genre")
    S = build_sims(ctx)
    cache = RESULTS / "pacr_cache.npz"
    if cache.exists():
        z = np.load(cache)
        pm_val, pm_te = z["pm_val"], z["pm_te"]
    else:
        mf = als_fit(ctx, 80, 10)
        pm_val, pm_te = mf(ctx.vu, ctx.vi), mf(ctx.eu, ctx.ei)
        np.savez(cache, pm_val=pm_val, pm_te=pm_te)
    return ctx, X, S, pm_val, pm_te


def make_fitness(ctx, X, S, pm_val, B):
    rng = np.random.default_rng(3)
    idx = rng.choice(len(ctx.vu), min(VAL_N, len(ctx.vu)), replace=False)
    vu, vi, vr, pmv = ctx.vu[idx], ctx.vi[idx], ctx.vr[idx], pm_val[idx]
    D = X.shape[1]

    def fit(sol):
        cents = np.asarray(sol).reshape(K, D)
        pools = budget_pools(X, cents, B)
        mask = np.zeros((len(X), len(X)), bool)
        for u, pl in enumerate(pools):
            mask[u, pl] = True
        p = np.empty(len(vu))
        for j, (u, i) in enumerate(zip(vu, vi)):
            cand = ctx.raters[i]
            cand = cand[mask[u, cand]]
            pk = np.nan
            if len(cand):
                s = S[u, cand]; t = np.argsort(-s)[:KNN_K]
                s, cc = s[t], cand[t]
                w = np.abs(s).sum()
                if w > 1e-9:
                    pk = ctx.um[u] + (s * (ctx.R[cc, i] - ctx.um[cc])).sum() / w
            p[j] = pmv[j] if np.isnan(pk) else BETA * pk + (1 - BETA) * pmv[j]
        return float(np.abs(np.clip(p, 1, 5) - vr).mean())
    return fit


def run_algo(name):
    ctx, X, S, pm_val, pm_te = get_ctx_and_cache()
    B = int(len(X) * B_PCT / 100)
    km = KMeans(K, init="k-means++", n_init=10, random_state=SEED).fit(X)
    c0 = km.cluster_centers_
    fitf = make_fitness(ctx, X, S, pm_val, B)
    f0 = fitf(c0.ravel())
    print(f"[BASLANGIC] B0 val-MAE = {f0:.4f}")

    out = RESULTS / "pacr.csv"
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    if not any(r["yontem"] == "B0" for r in rows):
        pools = budget_pools(X, c0, B)
        p, fb = eval_budget(ctx, S, pm_te, pools)
        rows.append({"yontem": "B0", "val_mae": round(f0, 4), "nfe": 0,
                     **full_metrics(ctx, p, fb, pools)})
        pd.DataFrame(rows).to_csv(out, index=False)

    import mealpy
    from mealpy import FloatVar
    cls = discover_algorithms([name])[name]
    lb, ub = np.tile(X.min(0), K), np.tile(X.max(0), K)
    rng = np.random.default_rng(SEED)
    sig = 0.08 * X.std()
    starts = np.vstack([c0.ravel()] +
                       [c0.ravel() + rng.normal(0, sig, c0.size)
                        for _ in range(POP - 1)])
    starts = np.clip(starts, lb, ub)
    model = cls(epoch=1000, pop_size=POP)
    prob = {"obj_func": fitf, "bounds": FloatVar(lb=lb, ub=ub),
            "minmax": "min", "log_to": None}
    t0 = time.time()
    g = model.solve(prob, seed=SEED, termination={"max_fe": NFE},
                    starting_solutions=starts)
    cents = np.asarray(g.solution).reshape(K, X.shape[1])
    fv = float(g.target.fitness)
    print(f"[{name}] val {f0:.4f} -> {fv:.4f}  ({time.time()-t0:.0f}s)")

    pools = budget_pools(X, cents, B)
    p, fb = eval_budget(ctx, S, pm_te, pools)
    kisa = name.split(".")[0]
    rows = [r for r in pd.read_csv(out).to_dict("records")
            if r["yontem"] != kisa]
    rows.append({"yontem": kisa, "val_mae": round(fv, 4),
                 "nfe": getattr(model, "nfe_counter", NFE),
                 **full_metrics(ctx, p, fb, pools)})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows)[["yontem", "val_mae", "mae", "ndcg10", "prec10"]]
          .round(4).to_string(index=False))


def report():
    df = pd.read_csv(RESULTS / "pacr.csv")
    cols = ["yontem", "val_mae", "mae", "rmse", "ndcg10", "prec10", "rec10",
            "fallback_pct"]
    print(df[cols].sort_values("mae").round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default=None)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        run_algo(a.algo)
