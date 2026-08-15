"""
SAMPIYON KONFIGURASYON ARAMASI — V5 (kume-kNN + ALS-MF) uzerine ayar.

Asamalar (hepsi ic-val ile secilir, test SADECE final icin kullanilir):
  1) MF izgara: faktor f x reg lambda  (ALS, bias'li, ic-train)
  2) Havuz uzayi: nmf vs nmf+genre; merkez: B0 vs AVOA (denge cezali)
  3) kNN k: {20,30,40}
  4) Karisim: sabit beta izgarasi VS kullanici-uyarlamali
     beta(u) = n_u / (n_u + gamma)   (cok puanli kullanici kNN'e guvenir)
Cikti: results/sampiyon.csv  (asama satirlari + FINAL test satiri)
Kullanim: python algo_selection_v2/sampiyon.py [--resume]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))
from eksikler_deney import DIM, SEED, Ctx  # noqa: E402
from genre_k_deney import (build_space, eval_soft, genre_profile,  # noqa: E402
                           hard_fitness, soft_pools)
from pred_v2 import build_sims, full_metrics  # noqa: E402
from recompute_scores import discover_algorithms, solve_meta  # noqa: E402

K = 10


def als_fit(ctx, f, lam, iters=10, seed=42):
    mu = ctx.gmean
    bu = np.zeros(len(ctx.X)); bi = np.zeros(1682)
    rng = np.random.default_rng(seed)
    P = rng.normal(0, .1, (len(ctx.X), f)); Q = rng.normal(0, .1, (1682, f))
    ui = [np.flatnonzero(ctx.R[u] > 0) for u in range(len(ctx.X))]
    iu_ = [np.flatnonzero(ctx.R[:, i] > 0) for i in range(1682)]
    I = lam * np.eye(f)
    for _ in range(iters):
        for u in range(len(ctx.X)):
            j = ui[u]
            if not len(j):
                continue
            r = ctx.R[u, j] - mu - bi[j]
            bu[u] = (r - P[u] @ Q[j].T).sum() / (len(j) + lam)
            P[u] = np.linalg.solve(Q[j].T @ Q[j] + I, Q[j].T @ (r - bu[u]))
        for i in range(1682):
            j = iu_[i]
            if not len(j):
                continue
            r = ctx.R[j, i] - mu - bu[j]
            bi[i] = (r - (P[j] * Q[i]).sum(1)).sum() / (len(j) + lam)
            Q[i] = np.linalg.solve(P[j].T @ P[j] + I, P[j].T @ (r - bi[i]))
    return lambda us, is_: np.clip(mu + bu[us] + bi[is_] + (P[us] * Q[is_]).sum(1), 1, 5)


def knn_preds(ctx, S, pools, us, is_, knn_k):
    p = np.empty(len(us))
    for j, (u, i) in enumerate(zip(us, is_)):
        cand = ctx.raters[i]; cand = cand[np.isin(cand, pools[u])]
        if len(cand):
            s = S[u, cand]; t = np.argsort(-s)[:knn_k]
            s, cc = s[t], cand[t]
            w = np.abs(s).sum()
            if w > 1e-9:
                p[j] = ctx.um[u] + (s * (ctx.R[cc, i] - ctx.um[cc])).sum() / w
                continue
        p[j] = np.nan
    return p


def blend(ctx, pk, pm, us, beta_fixed=None, gamma=None):
    if gamma is not None:
        n = (ctx.R > 0).sum(1)[us]
        b = n / (n + gamma)
    else:
        b = np.full(len(us), beta_fixed)
    out = np.where(np.isnan(pk), pm, b * np.nan_to_num(pk) + (1 - b) * pm)
    return np.clip(out, 1, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    data = Path(ROOT.parent / "data" / "ml-100k")
    ctx = Ctx(data)
    out = RESULTS / "sampiyon.csv"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records"); done = set(old.config)

    def save(r):
        rows.append(r); pd.DataFrame(rows).to_csv(out, index=False)
        print({k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in r.items()}, flush=True)

    # ---- 1) MF izgara (ic-val MAE) ----
    grid = [(20, 5), (40, 5), (40, 10), (60, 10), (80, 10)]
    for f, lam in grid:
        cfg = f"mf_f{f}_l{lam}"
        if cfg in done:
            continue
        mf = als_fit(ctx, f, lam)
        v = float(np.abs(mf(ctx.vu, ctx.vi) - ctx.vr).mean())
        save({"config": cfg, "asama": "mf", "f": f, "lam": lam, "val_mae": v})
    dfm = pd.DataFrame(rows)
    best = dfm[dfm.asama == "mf"].sort_values("val_mae").iloc[0]
    f_b, lam_b = int(best.f), int(best.lam)
    print(f"[MF SECIMI] f={f_b} lambda={lam_b} val={best.val_mae:.4f}")
    mf = als_fit(ctx, f_b, lam_b)
    pm_val, pm_te = mf(ctx.vu, ctx.vi), mf(ctx.eu, ctx.ei)

    # ---- 2-4) uzay x merkez x k x karisim (ic-val) ----
    G = genre_profile(ctx, data)
    S = build_sims(ctx, iuf=False, sig=False)
    cls = discover_algorithms(["AVOA.OriginalAVOA"])["AVOA.OriginalAVOA"]
    cands = []
    for mode in ("nmf", "nmf+genre"):
        X = build_space(ctx, G, mode)
        km = KMeans(K, init="k-means++", n_init=10, random_state=SEED).fit(X)
        cents = {"B0": km.cluster_centers_}
        cfg = f"avoacent_{mode}"
        pos, _, _, _ = solve_meta(cls, hard_fitness(ctx, X, K, 2.5 / K),
                                  np.tile(X.min(0), K), np.tile(X.max(0), K),
                                  3000, 30, SEED, max_fe=2000)
        cents["AVOA"] = np.asarray(pos).reshape(K, X.shape[1])
        for yn, c in cents.items():
            _, pools = soft_pools(ctx, X, c)
            for kk in (20, 30, 40):
                cfg = f"val_{mode}_{yn}_k{kk}"
                if cfg in done:
                    piv = [r for r in rows if r["config"] == cfg][0]
                    cands.append((piv["val_mae"], mode, yn, kk,
                                  piv["secim"], piv["parametre"], c, pools))
                    continue
                pk_val = knn_preds(ctx, S, pools, ctx.vu, ctx.vi, kk)
                best_v, sec, par = np.inf, None, None
                for b in (0.4, 0.5, 0.6, 0.7):
                    v = float(np.abs(blend(ctx, pk_val, pm_val, ctx.vu,
                                           beta_fixed=b) - ctx.vr).mean())
                    if v < best_v:
                        best_v, sec, par = v, "sabit", b
                for g in (10, 20, 40, 80, 160):
                    v = float(np.abs(blend(ctx, pk_val, pm_val, ctx.vu,
                                           gamma=g) - ctx.vr).mean())
                    if v < best_v:
                        best_v, sec, par = v, "uyarlamali", g
                save({"config": cfg, "asama": "val", "uzay": mode, "yontem": yn,
                      "knn_k": kk, "secim": sec, "parametre": par,
                      "val_mae": best_v})
                cands.append((best_v, mode, yn, kk, sec, par, c, pools))

    # ---- FINAL: en iyi val konfig -> test (tek atis) ----
    cands.sort(key=lambda x: x[0])
    v, mode, yn, kk, sec, par, c, pools = cands[0]
    print(f"[SAMPIYON] {yn} {mode} k={kk} {sec}={par} (val {v:.4f}) -> test")
    pk_te = knn_preds(ctx, S, pools, ctx.eu, ctx.ei, kk)
    p = blend(ctx, pk_te, pm_te, ctx.eu,
              beta_fixed=par if sec == "sabit" else None,
              gamma=par if sec == "uyarlamali" else None)
    fbp = 100 * float(np.isnan(pk_te).mean())
    r = {"config": "FINAL_TEST", "asama": "final", "uzay": mode, "yontem": yn,
         "knn_k": kk, "secim": sec, "parametre": par, "f": f_b, "lam": lam_b,
         **full_metrics(ctx, p, fbp, pools)}
    save(r)


if __name__ == "__main__":
    main()
