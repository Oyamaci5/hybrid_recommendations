"""
TUR (GENRE) YAN BILGISI + YUKSEK K DENEYI — seed 42, fold 1, soft top-2 eval.

Uzaylar:
  nmf        : NMF-20 (mevcut)
  nmf+genre  : NMF-20 ⊕ kullanici tur profili (u.item'daki 19 tur bayragi,
               rating agirlikli ortalama; sutun olcegi NMF bloguna esitlenir)
K: {10, 14, 20, 30} — top-2 havuz sayesinde yuksek K'lar artik denenebilir
   (havuz ≈ 2N/K; K=30'da bile ~63 kullanici).

Yontemler: B0 (KMeans++), AVOA (denge cezali hard bias-MAE fitness, NFE=2000).
Cikti: results/genre_k.csv (tum metrikler + havuz_ort + boyutlar)
Kullanim: python algo_selection_v2/genre_k_deney.py [--resume]
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
from eksikler_deney import DIM, SEED, THR, TOPN, Ctx  # noqa: E402
from recompute_scores import discover_algorithms, solve_meta, sse_wcss  # noqa: E402

N_I = 1682


def genre_profile(ctx, data_dir: Path) -> np.ndarray:
    G_item = np.zeros((N_I, 19))
    with open(data_dir / "u.item", encoding="latin-1") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            G_item[int(p[0]) - 1] = [int(x) for x in p[5:24]]
    W = np.zeros((len(ctx.um), N_I))
    W[ctx.iu, ctx.ii] = ctx.ir
    gu = W @ G_item
    gu /= np.maximum(gu.sum(1, keepdims=True), 1e-9)   # kullanici tur dagilimi
    return gu


def build_space(ctx, G, mode):
    if mode == "nmf":
        return ctx.X
    s = ctx.X.std(0).mean() / max(G.std(0).mean(), 1e-9)
    return np.ascontiguousarray(np.hstack([ctx.X, G * s]))


def soft_pools(ctx, X, cents, top=2):
    d = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
    lab = d.argmin(1); near = np.argsort(d, 1)[:, :top]
    return lab, [np.flatnonzero(np.isin(lab, near[u])) for u in range(len(X))]


def eval_soft(ctx, X, cents, knn_k=30):
    lab, pools = soft_pools(ctx, X, cents)
    p = np.empty(len(ctx.eu)); fb = 0
    for j, (u, i) in enumerate(zip(ctx.eu, ctx.ei)):
        cand = ctx.raters[i]; cand = cand[np.isin(cand, pools[u])]
        if len(cand):
            s = ctx.S[u, cand]; t = np.argsort(-s)[:knn_k]
            s, cc = s[t], cand[t]
            w = np.abs(s).sum()
            if w > 1e-9:
                p[j] = ctx.um[u] + (s * (ctx.R[cc, i] - ctx.um[cc])).sum() / w
                continue
        p[j] = ctx.um[u] + ctx.dev_g[i]; fb += 1
    p = np.clip(p, 1, 5); e = p - ctx.er
    out = {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
           "fallback_pct": round(100 * fb / len(p), 2),
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
                "ndcg10": float(np.mean(ndcgs)),
                "boyutlar": str(sorted(np.bincount(lab), reverse=True)[:4])})
    return out


def hard_fitness(ctx, X, K, cap):
    def fit(sol):
        c = np.asarray(sol).reshape(K, X.shape[1])
        lab = sse_wcss(X, c)[1]
        s_ = np.zeros((K, N_I)); c_ = np.zeros((K, N_I))
        np.add.at(s_, (lab[ctx.iu], ctx.ii), ctx.ir - ctx.um[ctx.iu])
        np.add.at(c_, (lab[ctx.iu], ctx.ii), 1.0)
        d = np.where(c_[lab[ctx.vu], ctx.vi] > 0,
                     s_[lab[ctx.vu], ctx.vi] / np.maximum(c_[lab[ctx.vu], ctx.vi], 1),
                     ctx.dev_g[ctx.vi])
        mae = float(np.abs(np.clip(ctx.um[ctx.vu] + d, 1, 5) - ctx.vr).mean())
        sh = np.bincount(lab, minlength=K) / len(X)
        return mae + 10 * np.clip(sh - cap, 0, None).sum() \
            + 0.1 * (K - len(np.unique(lab)))
    return fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data" / "ml-100k"))
    ap.add_argument("--max-fe", type=int, default=2000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    data = Path(args.data_dir)

    ctx = Ctx(data)
    G = genre_profile(ctx, data)
    cls = discover_algorithms(["AVOA.OriginalAVOA"])["AVOA.OriginalAVOA"]
    out = RESULTS / "genre_k.csv"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records"); done = set(old.config)

    for mode in ("nmf", "nmf+genre"):
        X = build_space(ctx, G, mode)
        for K in (10, 14, 20, 30):
            cap = 2.5 / K
            cfg = f"B0_{mode}_K{K}"
            if cfg not in done:
                km = KMeans(K, init="k-means++", n_init=10, random_state=SEED).fit(X)
                r = {"config": cfg, "uzay": mode, "K": K, "yontem": "B0",
                     **eval_soft(ctx, X, km.cluster_centers_)}
                rows.append(r); pd.DataFrame(rows).to_csv(out, index=False)
                print(cfg, round(r["mae"], 4), round(r["ndcg10"], 4), flush=True)
            cfg = f"AVOA_{mode}_K{K}"
            if cfg not in done:
                lb, ub = np.tile(X.min(0), K), np.tile(X.max(0), K)
                pos, fv, _, _ = solve_meta(cls, hard_fitness(ctx, X, K, cap),
                                           lb, ub, 3000, 30, SEED,
                                           max_fe=args.max_fe)
                cents = np.asarray(pos).reshape(K, X.shape[1])
                r = {"config": cfg, "uzay": mode, "K": K, "yontem": "AVOA",
                     "fit_val": round(fv, 4), **eval_soft(ctx, X, cents)}
                rows.append(r); pd.DataFrame(rows).to_csv(out, index=False)
                print(cfg, round(r["mae"], 4), round(r["ndcg10"], 4), flush=True)


if __name__ == "__main__":
    main()
