
"""
ABLASYON MERDIVENI — literatur kurulumundan bizim kuruluma, ADIM ADIM.

Amac: "sonucu iyilestirmek icin protokolun etrafindan dolasilmis" elestirisini
yapisal olarak cevaplamak. Her basamakta TEK BIR sey degisir ve B0 vs meta
karsilastirmasi TEKRARLANIR. Eger meta avantaji her basamakta korunuyorsa,
bulgu pipeline'a bagli DEGILDIR.

BASAMAKLAR (S0 = literatur, S6 = bizim):
  S0  ham matris + WCSS fitness + serbest atama + kume-ortalamasi   [HSC/Firefly]
  S1  + NMF-20 uzayi           (arama uzayi 1682 -> 20 boyut)
  S2  + tahmine hizali fitness (WCSS yerine ic-val MAE)
  S3  + kapasiteli onarim      (repair, regret oncelikli)
  S4  + soft top-2 havuz       (hard atama yerine)
  S5  + kume-ici kNN tahminci  (kume-ortalamasi yerine)
  S6  + kume-MF karisimi       (0.5*kNN + 0.5*MF)  = TAM SISTEM

Her basamakta: B0 (KMeans++) ve metalar (AVOA + istege bagli digerleri).
Ayrica TERS YON: bizim uzayda literatur tahmincisi (S1+cmean) zaten S1'de var.

Kullanim:
  python algo_selection_v2/ablasyon_merdiveni.py --k 40 --tum
  python algo_selection_v2/ablasyon_merdiveni.py --k 40 --seeds 42 43 44
Cikti: results/ablasyon_merdiveni.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))
import cluster_mf as cm  # noqa: E402
from eksikler_deney import Ctx  # noqa: E402
from pred_v2 import build_sims, full_metrics  # noqa: E402
from recompute_scores import discover_algorithms  # noqa: E402

N_I = 1682
ALGOS = ["AVOA.OriginalAVOA", "HGS.OriginalHGS", "HHO.OriginalHHO",
         "NGO.OriginalNGO", "GWO.OriginalGWO"]


# ---------- atama varyantlari ----------
def serbest_ata(X, C):
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    o = np.argsort(d, 1)
    return o[:, 0].astype(np.int32), o[:, :2].astype(np.int32)


def repair_ata(X, C):
    """Kapasiteli, regret oncelikli."""
    N, K = len(X), len(C)
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    oc = np.argsort(d, 1)
    en, ik = d[np.arange(N), oc[:, 0]], d[np.arange(N), oc[:, 1]]
    sira = np.argsort(-(ik - en))
    cap = int(np.ceil(N / K))
    L = -np.ones(N, np.int32); sz = np.zeros(K, np.int32)
    for u in sira:
        for c in oc[u]:
            if sz[c] < cap:
                L[u] = c; sz[c] += 1; break
    near = np.empty((N, 2), np.int32); near[:, 0] = L
    near[:, 1] = np.where(oc[:, 0] != L, oc[:, 0], oc[:, 1])
    return L, near


# ---------- tahminciler ----------
def istatistik(ctx, L, K):
    flat = L[ctx.iu].astype(np.int64) * N_I + ctx.ii
    s = np.bincount(flat, weights=ctx.ir, minlength=K * N_I)
    n = np.bincount(flat, minlength=K * N_I)
    sd = np.bincount(flat, weights=ctx.ir - ctx.um[ctx.iu], minlength=K * N_I)
    cnt = np.bincount(ctx.ii, minlength=N_I)
    im = np.where(cnt > 0, np.bincount(ctx.ii, weights=ctx.ir, minlength=N_I)
                  / np.maximum(cnt, 1), ctx.gmean)
    return s, n, sd, im


def p_cmean(ctx, L, K, st, hard=True, near=None):
    s, n, sd, im = st
    if hard:
        j = L[ctx.eu].astype(np.int64) * N_I + ctx.ei
        return np.where(n[j] > 0, s[j] / np.maximum(n[j], 1), im[ctx.ei]), \
            100 * float((n[j] == 0).mean())
    a = near[ctx.eu, 0].astype(np.int64) * N_I + ctx.ei
    b = near[ctx.eu, 1].astype(np.int64) * N_I + ctx.ei
    den = n[a] + n[b]
    return np.where(den > 0, (s[a] + s[b]) / np.maximum(den, 1), im[ctx.ei]), \
        100 * float((den == 0).mean())


def p_bias(ctx, near, K, st):
    s, n, sd, im = st
    a = near[ctx.eu, 0].astype(np.int64) * N_I + ctx.ei
    b = near[ctx.eu, 1].astype(np.int64) * N_I + ctx.ei
    den = n[a] + n[b]
    dev = np.where(den > 0, (sd[a] + sd[b]) / np.maximum(den, 1), ctx.dev_g[ctx.ei])
    return ctx.um[ctx.eu] + dev


def p_knn(ctx, S, L, near, knn_k=20):
    p = np.empty(len(ctx.eu))
    for j, (u, i) in enumerate(zip(ctx.eu, ctx.ei)):
        cand = ctx.raters[i]
        lc = L[cand]
        cand = cand[(lc == near[u, 0]) | (lc == near[u, 1])]
        pk = np.nan
        if len(cand):
            sm = S[u, cand]
            if len(cand) > knn_k:
                t = np.argpartition(-sm, knn_k)[:knn_k]
                sm, cand = sm[t], cand[t]
            w = np.abs(sm).sum()
            if w > 1e-9:
                pk = ctx.um[u] + float((sm * (ctx.R[cand, i] - ctx.um[cand])).sum() / w)
        p[j] = pk
    return p


# ---------- fitness varyantlari ----------
def fit_wcss(X, K):
    X2 = (X ** 2).sum(1)

    def f(sol):
        C = np.asarray(sol).reshape(K, X.shape[1])
        d = X2[:, None] - 2.0 * (X @ C.T) + (C ** 2).sum(1)[None, :]
        return float(np.maximum(d.min(1), 0).sum())
    return f


def fit_pred(ctx, X, K, atama):
    def f(sol):
        C = np.asarray(sol).reshape(K, X.shape[1])
        L, near = atama(X, C)
        flat = L[ctx.iu].astype(np.int64) * N_I + ctx.ii
        n = np.bincount(flat, minlength=K * N_I)
        sd = np.bincount(flat, weights=ctx.ir - ctx.um[ctx.iu], minlength=K * N_I)
        a = near[ctx.vu, 0].astype(np.int64) * N_I + ctx.vi
        b = near[ctx.vu, 1].astype(np.int64) * N_I + ctx.vi
        den = n[a] + n[b]
        dev = np.where(den > 0, (sd[a] + sd[b]) / np.maximum(den, 1), ctx.dev_g[ctx.vi])
        return float(np.abs(np.clip(ctx.um[ctx.vu] + dev, 1, 5) - ctx.vr).mean())
    return f


BASAMAKLAR = {
    "S0_literatur":  dict(uzay="ham", fitness="wcss", atama="serbest", havuz="hard",  tahmin="cmean"),
    "S1_nmf":        dict(uzay="nmf", fitness="wcss", atama="serbest", havuz="hard",  tahmin="cmean"),
    "S2_predfit":    dict(uzay="nmf", fitness="pred", atama="serbest", havuz="hard",  tahmin="cmean"),
    "S3_repair":     dict(uzay="nmf", fitness="pred", atama="repair",  havuz="hard",  tahmin="cmean"),
    "S4_soft":       dict(uzay="nmf", fitness="pred", atama="repair",  havuz="soft",  tahmin="cmean"),
    "S5_knn":        dict(uzay="nmf", fitness="pred", atama="repair",  havuz="soft",  tahmin="knn"),
    "S6_tam":        dict(uzay="nmf", fitness="pred", atama="repair",  havuz="soft",  tahmin="knn+mf"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--max-fe", type=int, default=600)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--tum", action="store_true", help="5 meta (varsayilan: AVOA)")
    ap.add_argument("--basamak", nargs="+", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    data = Path(ROOT.parent / "data" / "ml-100k")
    ctx = Ctx(data, args.fold)
    S = build_sims(ctx)
    K = args.k
    cm.K, cm.N_I = K, N_I
    X_nmf = np.ascontiguousarray(ctx.X)
    X_ham = ctx.R.copy()
    uzaylar = {"nmf": X_nmf, "ham": X_ham}

    out = RESULTS / "ablasyon_merdiveni.csv"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records")
        done = set(zip(old.basamak, old.yontem, old.K, old.seed))

    adlar = ALGOS if args.tum else ["AVOA.OriginalAVOA"]
    algos = discover_algorithms(adlar)
    from mealpy import FloatVar
    basamaklar = args.basamak or list(BASAMAKLAR)

    for bad in basamaklar:
        cfg = BASAMAKLAR[bad]
        X = uzaylar[cfg["uzay"]]
        atama = serbest_ata if cfg["atama"] == "serbest" else repair_ata
        fitf = (fit_wcss(X, K) if cfg["fitness"] == "wcss"
                else fit_pred(ctx, X, K, atama))
        for seed in args.seeds:
            km = KMeans(K, init="k-means++", n_init=5, random_state=seed).fit(X)
            merkez = {"B0": km.cluster_centers_}
            for an, cls in algos.items():
                kisa = an.split(".")[0]
                if (bad, kisa, K, seed) in done:
                    continue
                lb, ub = np.tile(X.min(0), K), np.tile(X.max(0), K)
                rng = np.random.default_rng(seed)
                c0 = km.cluster_centers_
                st_ = np.clip(np.vstack([c0.ravel()] + [
                    c0.ravel() + rng.normal(0, .08 * X.std(), c0.size)
                    for _ in range(args.pop - 1)]), lb, ub)
                g = cls(epoch=5000, pop_size=args.pop).solve(
                    {"obj_func": fitf, "bounds": FloatVar(lb=lb, ub=ub),
                     "minmax": "min", "log_to": None}, seed=seed,
                    termination={"max_fe": args.max_fe}, starting_solutions=st_)
                merkez[kisa] = np.asarray(g.solution).reshape(K, X.shape[1])
            for yn, C in merkez.items():
                if (bad, yn, K, seed) in done:
                    continue
                t0 = time.time()
                L, near = atama(X, C)
                stt = istatistik(ctx, L, K)
                hard = cfg["havuz"] == "hard"
                if cfg["tahmin"] == "cmean":
                    p, fbp = p_cmean(ctx, L, K, stt, hard, near)
                else:
                    pb = p_bias(ctx, near, K, stt)
                    pk = p_knn(ctx, S, L, near)
                    fbp = 100 * float(np.isnan(pk).mean())
                    pk = np.where(np.isnan(pk), pb, pk)
                    if cfg["tahmin"] == "knn":
                        p = pk
                    else:
                        pmf = cm.cmf_predict(cm.cluster_mf_models(ctx, L), L,
                                             ctx.eu, ctx.ei, pb)
                        p = 0.5 * pk + 0.5 * pmf
                p = np.clip(p, 1, 5)
                sz = np.bincount(L, minlength=K)
                havuz = int(np.mean([(np.isin(L, near[u])).sum()
                                     for u in range(0, len(X), 5)]))
                r = full_metrics(ctx, p, fbp, [np.arange(1)])
                r.pop("havuz_ort", None)
                rows.append({"basamak": bad, "yontem": yn, "K": K, "seed": seed,
                             "uzay": cfg["uzay"], "fitness": cfg["fitness"],
                             "atama": cfg["atama"], "havuz_tipi": cfg["havuz"],
                             "tahmin": cfg["tahmin"], "havuz": havuz,
                             "maxk": int(sz.max()), "mink": int(sz.min()),
                             "bos": int((sz == 0).sum()),
                             "sure_s": round(time.time() - t0, 1), **r})
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"{bad:13s} {yn:5s} s{seed} MAE={r['mae']:.4f} "
                      f"NDCG={r['ndcg10']:.4f} havuz={havuz} "
                      f"max/min={sz.max()}/{sz.min()}", flush=True)


if __name__ == "__main__":
    main()
