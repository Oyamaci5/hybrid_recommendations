"""
PRED V2 — 6 makaleden alinan tekniklerle tahminci guclendirme (genre'siz).

Alinan teknikler:
  V1 significance weighting : sim *= min(ortak_puan_sayisi, 50)/50  (Breese/Herlocker)
  V2 + IUF                  : populer filmlerin benzerlige katkisi azaltilir
                              w_i = log(N/n_i)  (CS-Kmeans makalesindeki Breese formulu)
  V3 + user-item fuzyonu    : P = a*P_user_cknn + (1-a)*P_item_knn
                              (CS-Kmeans Formul 7; a ic-val'de secilir)
Degerlendirme: soft top-2 havuz, K=10, seed 42, fold 1; B0 ve AVOA (hard-fit).
Cikti: results/pred_v2.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))
from eksikler_deney import DIM, SEED, THR, TOPN, Ctx  # noqa: E402
from genre_k_deney import hard_fitness, soft_pools  # noqa: E402
from recompute_scores import discover_algorithms, solve_meta  # noqa: E402

N_I = 1682
K, KNN_K = 10, 30


def build_sims(ctx, iuf=False, sig=True):
    rated = ctx.R > 0
    w = np.log(len(ctx.X) / np.maximum(rated.sum(0), 1)) if iuf else np.ones(N_I)
    Rc = np.where(rated, ctx.R - ctx.um[:, None], 0.0) * w
    nrm = np.linalg.norm(Rc, axis=1); nrm[nrm < 1e-9] = 1.0
    S = (Rc / nrm[:, None]) @ (Rc / nrm[:, None]).T
    np.fill_diagonal(S, 0.0)
    if sig:
        co = (rated.astype(np.float32) @ rated.T.astype(np.float32))
        S = S * np.minimum(co, 50) / 50
    return S


def build_item_sim(ctx):
    rated = ctx.R > 0
    Rc = np.where(rated, ctx.R - ctx.um[:, None], 0.0)     # adjusted cosine
    nrm = np.linalg.norm(Rc, axis=0); nrm[nrm < 1e-9] = 1.0
    SI = (Rc / nrm).T @ (Rc / nrm)
    np.fill_diagonal(SI, 0.0)
    return SI


def user_pred(ctx, S, pools, u, i):
    cand = ctx.raters[i]; cand = cand[np.isin(cand, pools[u])]
    if len(cand):
        s = S[u, cand]; t = np.argsort(-s)[:KNN_K]
        s, cc = s[t], cand[t]
        w = np.abs(s).sum()
        if w > 1e-9:
            return ctx.um[u] + (s * (ctx.R[cc, i] - ctx.um[cc])).sum() / w
    return np.nan


def item_pred(ctx, SI, rated_items, u, i):
    j = rated_items[u]
    if len(j) == 0:
        return np.nan
    s = SI[i, j]; t = np.argsort(-s)[:KNN_K]
    s, jj = s[t], j[t]
    w = np.abs(s).sum()
    if w < 1e-9:
        return np.nan
    return float((s * ctx.R[u, jj]).sum() / w)


def predict_all(ctx, S, SI, pools, us, is_, alpha):
    rated_items = [np.flatnonzero(ctx.R[u] > 0) for u in range(len(ctx.X))]
    p = np.empty(len(us)); fb = 0
    for j, (u, i) in enumerate(zip(us, is_)):
        pu = user_pred(ctx, S, pools, u, i)
        pi = item_pred(ctx, SI, rated_items, u, i) if alpha < 1 else np.nan
        if np.isnan(pu) and np.isnan(pi):
            p[j] = ctx.um[u] + ctx.dev_g[i]; fb += 1
        elif np.isnan(pu):
            p[j] = pi
        elif np.isnan(pi) or alpha >= 1:
            p[j] = pu
        else:
            p[j] = alpha * pu + (1 - alpha) * pi
    return np.clip(p, 1, 5), 100 * fb / len(p)


def full_metrics(ctx, p, fb, pools):
    e = p - ctx.er
    out = {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
           "fallback_pct": round(fb, 2),
           "havuz_ort": int(np.mean([len(x) for x in pools]))}
    precs, recs, ndcgs = [], [], []
    for u in range(len(ctx.by_user)):
        idxs = np.array(ctx.by_user[u], dtype=int)
        if len(idxs) == 0:
            continue
        order = idxs[np.argsort(-p[idxs])]
        rel = (ctx.er[order] >= THR).astype(float)
        n = min(TOPN, len(order))
        precs.append(rel[:n].sum() / TOPN)
        nrel = int((ctx.er[idxs] >= THR).sum())
        if nrel:
            recs.append(rel[:n].sum() / nrel)
            disc = np.log2(np.arange(2, n + 2))
            idcg = (np.sort(rel)[::-1][:n] / disc).sum()
            ndcgs.append(float((rel[:n] / disc).sum() / idcg) if idcg else 0.0)
    out.update({"prec10": float(np.mean(precs)), "rec10": float(np.mean(recs)),
                "ndcg10": float(np.mean(ndcgs))})
    return out


def main():
    ctx = Ctx(Path(ROOT.parent / "data" / "ml-100k"))
    cls = discover_algorithms(["AVOA.OriginalAVOA"])["AVOA.OriginalAVOA"]
    cap = 2.5 / K
    km = KMeans(K, init="k-means++", n_init=10, random_state=SEED).fit(ctx.X)
    lb, ub = np.tile(ctx.X.min(0), K), np.tile(ctx.X.max(0), K)
    pos, _, _, _ = solve_meta(cls, hard_fitness(ctx, ctx.X, K, cap),
                              lb, ub, 3000, 30, SEED, max_fe=2000)
    centers = {"B0": km.cluster_centers_,
               "AVOA": np.asarray(pos).reshape(K, DIM)}
    SI = build_item_sim(ctx)
    variants = {
        "V0_cosine": build_sims(ctx, iuf=False, sig=False),
        "V1_sig": build_sims(ctx, iuf=False, sig=True),
        "V2_sig+iuf": build_sims(ctx, iuf=True, sig=True),
    }
    rows = []
    for cname, cent in centers.items():
        _, pools = soft_pools(ctx, ctx.X, cent)
        for vname, S in variants.items():
            p, fb = predict_all(ctx, S, SI, pools, ctx.eu, ctx.ei, alpha=1.0)
            r = {"yontem": cname, "varyant": vname, "alpha": 1.0,
                 **full_metrics(ctx, p, fb, pools)}
            rows.append(r); print(r["yontem"], r["varyant"],
                                  round(r["mae"], 4), round(r["ndcg10"], 4), flush=True)
        # V3: en iyi S (V2) + fuzyon; alpha ic-val'de secilir
        S = variants["V2_sig+iuf"]
        best_a, best_v = 1.0, np.inf
        for a in (0.5, 0.6, 0.7, 0.8, 0.9):
            pv, _ = predict_all(ctx, S, SI, pools, ctx.vu, ctx.vi, alpha=a)
            v = float(np.abs(pv - ctx.vr).mean())
            if v < best_v:
                best_a, best_v = a, v
        p, fb = predict_all(ctx, S, SI, pools, ctx.eu, ctx.ei, alpha=best_a)
        r = {"yontem": cname, "varyant": "V3_fuzyon", "alpha": best_a,
             **full_metrics(ctx, p, fb, pools)}
        rows.append(r); print(r["yontem"], "V3_fuzyon a=", best_a,
                              round(r["mae"], 4), round(r["ndcg10"], 4), flush=True)
        pd.DataFrame(rows).to_csv(RESULTS / "pred_v2.csv", index=False)


if __name__ == "__main__":
    main()
