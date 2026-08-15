"""
SOFT-HIZALI FITNESS DENEYI — merkez arama, degerlendirmeyle ayni yapiyi gorsun.

Onceki durum: fitness HARD atama + bias tahminci ile aranirken degerlendirme
SOFT (top-2) havuz + kNN ile yapiliyordu -> yapi uyumsuz.
Bu deney: fitness = SOFT top-2 havuz bias-MAE (ic-val) + denge cezasi.
(kNN'li fitness cok pahali; soft-bias, soft-kNN'in ucuz ve ayni-yapili vekili.)

Cikti tablosu tum metriklerle: MAE, RMSE, P@10, R@10, NDCG@10, fallback%, havuz_ort.
  - fallback% : kNN tahmini kurulamayan test tahmini orani (havuzda o filmi
    puanlayan komsu yok / benzerlik agirligi 0) -> yedek formul devreye girer
    (kullanici ort. + global film sapmasi).
  - havuz_ort : kullanici basina komsu-aday havuzunun ortalama boyutu
    (maliyet gostergesi; 943 = kumesiz tam arama).

Kullanim: python algo_selection_v2/soft_align.py [--k 10] [--max-fe 2000]
          [--dataset 100k]   (1M destegi: --dataset 1m, sonraki asama)
Cikti:   results/soft_aligned.csv
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
from eksikler_deney import DIM, SEED, THR, TOPN, Ctx  # noqa: E402
from recompute_scores import discover_algorithms, solve_meta, sse_wcss  # noqa: E402

N_I = 1682


def soft_info(ctx, cents, top=2):
    d = ((ctx.X[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
    return d.argmin(1), np.argsort(d, 1)[:, :top]


def eval_soft(ctx, cents, top=2, knn_k=30):
    lab, near = soft_info(ctx, cents, top)
    pools = [np.flatnonzero(np.isin(lab, near[u])) for u in range(len(ctx.X))]
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
            ideal = np.sort(rel)[::-1][:n]
            idcg = (ideal / disc).sum()
            ndcgs.append(float((rel[:n] / disc).sum() / idcg) if idcg else 0.0)
    out.update({"prec10": float(np.mean(precs)), "rec10": float(np.mean(recs)),
                "ndcg10": float(np.mean(ndcgs))})
    return out


def make_soft_fitness(ctx, K, cap):
    """Soft top-2 havuz bias-MAE (ic-val) + denge cezasi — tamamen vektorlu."""
    N = len(ctx.X)

    def fit(sol):
        c = np.asarray(sol).reshape(K, DIM)
        lab, near = soft_info(ctx, c, 2)
        s_ = np.zeros((K, N_I)); c_ = np.zeros((K, N_I))
        np.add.at(s_, (lab[ctx.iu], ctx.ii), ctx.ir - ctx.um[ctx.iu])
        np.add.at(c_, (lab[ctx.iu], ctx.ii), 1.0)
        a, b = near[ctx.vu, 0], near[ctx.vu, 1]
        num = s_[a, ctx.vi] + s_[b, ctx.vi]
        den = c_[a, ctx.vi] + c_[b, ctx.vi]
        d = np.where(den > 0, num / np.maximum(den, 1), ctx.dev_g[ctx.vi])
        mae = float(np.abs(np.clip(ctx.um[ctx.vu] + d, 1, 5) - ctx.vr).mean())
        sh = np.bincount(lab, minlength=K) / N
        return mae + 10 * np.clip(sh - cap, 0, None).sum() \
            + 0.1 * (K - len(np.unique(lab)))
    return fit


def make_hard_fitness(ctx, K, cap):
    N = len(ctx.X)

    def fit(sol):
        c = np.asarray(sol).reshape(K, DIM)
        lab = sse_wcss(ctx.X, c)[1]
        s_ = np.zeros((K, N_I)); c_ = np.zeros((K, N_I))
        np.add.at(s_, (lab[ctx.iu], ctx.ii), ctx.ir - ctx.um[ctx.iu])
        np.add.at(c_, (lab[ctx.iu], ctx.ii), 1.0)
        d = np.where(c_[lab[ctx.vu], ctx.vi] > 0,
                     s_[lab[ctx.vu], ctx.vi] / np.maximum(c_[lab[ctx.vu], ctx.vi], 1),
                     ctx.dev_g[ctx.vi])
        mae = float(np.abs(np.clip(ctx.um[ctx.vu] + d, 1, 5) - ctx.vr).mean())
        sh = np.bincount(lab, minlength=K) / N
        return mae + 10 * np.clip(sh - cap, 0, None).sum() \
            + 0.1 * (K - len(np.unique(lab)))
    return fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-fe", type=int, default=2000)
    ap.add_argument("--dataset", choices=["100k", "1m"], default="100k")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    if args.dataset == "1m":
        raise SystemExit("1M icin Ctx'e 1M yukleyici eklenecek (sonraki asama).")
    data = Path(args.data_dir or ROOT.parent / "data" / "ml-100k")

    ctx = Ctx(data)
    K, cap = args.k, 2.5 / args.k
    lb, ub = np.tile(ctx.X.min(0), K), np.tile(ctx.X.max(0), K)
    cls = discover_algorithms(["AVOA.OriginalAVOA"])["AVOA.OriginalAVOA"]
    rows = []

    def add(cfg, cents, extra=None):
        r = {"config": cfg, **(extra or {}), **eval_soft(ctx, cents)}
        lab = soft_info(ctx, cents)[0]
        r["boyutlar"] = str(sorted(np.bincount(lab, minlength=K), reverse=True)[:5])
        rows.append(r)
        pd.DataFrame(rows).to_csv(RESULTS / "soft_aligned.csv", index=False)
        print(cfg, {k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in r.items() if k != "config"}, flush=True)

    km = KMeans(K, init="k-means++", n_init=10, random_state=SEED).fit(ctx.X)
    add("B0_soft_eval", km.cluster_centers_)

    t0 = time.time()
    pos, fv, _, nfe = solve_meta(cls, make_hard_fitness(ctx, K, cap),
                                 lb, ub, 3000, 30, SEED, max_fe=args.max_fe)
    add("AVOA_hardfit_softeval", np.asarray(pos).reshape(K, DIM),
        {"fit_val": round(fv, 4), "nfe": nfe, "opt_s": round(time.time() - t0, 1)})

    t0 = time.time()
    pos, fv, _, nfe = solve_meta(cls, make_soft_fitness(ctx, K, cap),
                                 lb, ub, 3000, 30, SEED, max_fe=args.max_fe)
    add("AVOA_softfit_softeval", np.asarray(pos).reshape(K, DIM),
        {"fit_val": round(fv, 4), "nfe": nfe, "opt_s": round(time.time() - t0, 1)})

    print("\nReferans knn_all: MAE=0.7467 NDCG=0.8352 P@10=0.6963 R@10=0.5451")


if __name__ == "__main__":
    main()
