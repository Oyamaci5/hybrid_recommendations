
"""
ONARIM ONCELIK KURALI ABLASYONU — regret (pismanlik) tabanli siralama.

Mevcut kural (v1 "mesafe"): kullanicilar min_c D[u,c] artan sirada islenir.
  Sorun: alternatiflerin ne kadar kotu oldugunu goz ardi eder.

Onerilen (v2 "regret"): Regret(u) = D[u, 2.yakin] - D[u, 1.yakin]
  AZALAN sirada islenir. Kaydirilmasi pahali olan kullanici once yerlesir.
  (Capacitated assignment literaturunde standart: en yuksek pismanlik once.)

Ek varyant (v3 "regret_orani"): Regret(u) / (D[u,1.yakin] + eps)
  Olcek bagimsiz pismanlik; uzak kullanicilarin mutlak farki sismesin diye.

Olculenler: SSE (kapasiteli atamanin dogal hedefi), tahmin metrikleri,
kac kullanici yer degistirdi, yer degistirenlerin toplam ek maliyeti.

Kullanim:
  python algo_selection_v2/repair_regret.py --dataset 100k --klist 6 20 40
  python algo_selection_v2/repair_regret.py --dataset 1m --klist 6 40
Cikti: results/repair_regret.csv
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
import cluster_mf as cm  # noqa: E402
from recompute_scores import discover_algorithms  # noqa: E402


# ---------------- atama varyantlari ----------------

def _dist(X, C, X2=None):
    if X2 is None:
        X2 = (X ** 2).sum(1)
    return X2[:, None] - 2.0 * (X @ C.T) + (C ** 2).sum(1)[None, :]


def ata(X, C, kural="mesafe"):
    """Kapasiteli greedy atama; kural = mesafe | regret | regret_orani."""
    N, K = len(X), len(C)
    d = _dist(X, C)
    order_c = np.argsort(d, 1)
    en_yakin = d[np.arange(N), order_c[:, 0]]
    ikinci = d[np.arange(N), order_c[:, 1]] if K > 1 else en_yakin
    if kural == "mesafe":
        sira = np.argsort(en_yakin)                      # artan mesafe
    elif kural == "regret":
        sira = np.argsort(-(ikinci - en_yakin))           # azalan pismanlik
    else:                                                 # regret_orani
        sira = np.argsort(-((ikinci - en_yakin) / (en_yakin + 1e-9)))
    cap = int(np.ceil(N / K))
    L = -np.ones(N, np.int32); sizes = np.zeros(K, np.int32)
    for u in sira:
        for c in order_c[u]:
            if sizes[c] < cap:
                L[u] = c; sizes[c] += 1; break
    near = np.empty((N, 2), np.int32)
    near[:, 0] = L
    ilk = order_c[:, 0]
    near[:, 1] = np.where(ilk != L, ilk, order_c[:, 1])
    # teshis: SSE ve yer degistirme maliyeti
    sse = float(d[np.arange(N), L].sum())
    sse_serbest = float(en_yakin.sum())
    kaydi = L != ilk
    ek_maliyet = float((d[np.arange(N), L] - en_yakin)[kaydi].sum())
    return L, near, {"sse": sse, "sse_serbest": sse_serbest,
                     "sse_artis_pct": round(100 * (sse / sse_serbest - 1), 2),
                     "yer_degistiren": int(kaydi.sum()),
                     "yer_degistiren_pct": round(100 * kaydi.mean(), 1),
                     "ek_maliyet": ek_maliyet}


# ---------------- degerlendirme (veri setine gore) ----------------

def kur_100k(fold=1):
    from eksikler_deney import Ctx
    from genre_k_deney import build_space, genre_profile
    from pred_v2 import build_sims
    data = ROOT.parent / "data" / "ml-100k"
    ctx = Ctx(data, fold)
    X = build_space(ctx, genre_profile(ctx, data), "nmf")
    return ctx, X, build_sims(ctx), 1682


def kur_1m(fold=1):
    import ml1m_run as m
    from ctx_ml1m import Ctx1M
    ctx = Ctx1M(ROOT.parent / "data" / "ml-1m", fold)
    return ctx, m.build_space(ctx, None, "nmf"), ctx.S, 3952


def degerlendir(ctx, X, S, NI, L, near, K, knn_k=20):
    from pred_v2 import full_metrics
    cm.K, cm.N_I = K, NI
    models = cm.cluster_mf_models(ctx, L)
    flat = L[ctx.iu].astype(np.int64) * NI + ctx.ii
    n = np.bincount(flat, minlength=K * NI)
    sd = np.bincount(flat, weights=ctx.ir - ctx.um[ctx.iu], minlength=K * NI)
    a = near[ctx.eu, 0].astype(np.int64) * NI + ctx.ei
    b = near[ctx.eu, 1].astype(np.int64) * NI + ctx.ei
    den = n[a] + n[b]
    dev = np.where(den > 0, (sd[a] + sd[b]) / np.maximum(den, 1), ctx.dev_g[ctx.ei])
    p_bias = ctx.um[ctx.eu] + dev
    p_mf = cm.cmf_predict(models, L, ctx.eu, ctx.ei, p_bias)
    p_knn = np.empty(len(ctx.eu))
    for j, (u, i) in enumerate(zip(ctx.eu, ctx.ei)):
        cand = ctx.raters[i]
        lc = L[cand]
        cand = cand[(lc == near[u, 0]) | (lc == near[u, 1])]
        pk = np.nan
        if len(cand):
            s = S[u, cand]
            if len(cand) > knn_k:
                t = np.argpartition(-s, knn_k)[:knn_k]
                s, cand = s[t], cand[t]
            w = np.abs(s).sum()
            if w > 1e-9:
                pk = ctx.um[u] + float((s * (ctx.R[cand, i] - ctx.um[cand])).sum() / w)
        p_knn[j] = pk
    fb = np.isnan(p_knn)
    p = np.clip(0.5 * np.where(fb, p_bias, p_knn) + 0.5 * p_mf, 1, 5)
    r = full_metrics(ctx, p, 100 * float(fb.mean()), [np.arange(1)])
    r.pop("havuz_ort", None)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["100k", "1m"], default="100k")
    ap.add_argument("--klist", type=int, nargs="+", default=[6, 20, 40])
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-fe", type=int, default=600)
    ap.add_argument("--meta", action="store_true", help="AVOA merkezleriyle de kos")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    ctx, X, S, NI = kur_100k(args.fold) if args.dataset == "100k" else kur_1m(args.fold)
    out = RESULTS / "repair_regret.csv"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records")
        done = set(zip(old.dataset, old.merkez, old.K, old.kural))

    for K in args.klist:
        km = KMeans(K, init="k-means++", n_init=5, random_state=args.seed).fit(X)
        merkezler = {"B0": km.cluster_centers_}
        if args.meta:
            if args.dataset == "1m":
                import ml1m_run as mm
                fitf = mm.make_fitness(ctx, X, K)
                solve = mm.solve_warm
            else:
                import tabloB_plus as bp
                bp.K = K; cm.K = K
                fitf = bp.make_fitness(ctx, X)

                def solve(cls, f, Xx, KK, sd_, c0, max_fe, pop=10):
                    from mealpy import FloatVar
                    lb, ub = np.tile(Xx.min(0), KK), np.tile(Xx.max(0), KK)
                    rng = np.random.default_rng(sd_)
                    st = np.clip(np.vstack([c0.ravel()] +
                        [c0.ravel() + rng.normal(0, .08 * Xx.std(), c0.size)
                         for _ in range(pop - 1)]), lb, ub)
                    g = cls(epoch=3000, pop_size=pop).solve(
                        {"obj_func": f, "bounds": FloatVar(lb=lb, ub=ub),
                         "minmax": "min", "log_to": None}, seed=sd_,
                        termination={"max_fe": max_fe}, starting_solutions=st)
                    return np.asarray(g.solution).reshape(KK, Xx.shape[1])
            cls = discover_algorithms(["AVOA.OriginalAVOA"])["AVOA.OriginalAVOA"]
            merkezler["AVOA"] = solve(cls, fitf, X, K, args.seed,
                                      km.cluster_centers_, args.max_fe, 10)
        for mad, C in merkezler.items():
            for kural in ("mesafe", "regret", "regret_orani"):
                if (args.dataset, mad, K, kural) in done:
                    continue
                t0 = time.time()
                L, near, tesh = ata(X, C, kural)
                r = degerlendir(ctx, X, S, NI, L, near, K)
                sz = np.bincount(L, minlength=K)
                rows.append({"dataset": args.dataset, "merkez": mad, "K": K,
                             "kural": kural, **tesh,
                             "maxk": int(sz.max()), "mink": int(sz.min()),
                             "sure_s": round(time.time() - t0, 1), **r})
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"{args.dataset} {mad:4s} K={K:2d} {kural:13s} "
                      f"SSE_artis={tesh['sse_artis_pct']:5.2f}% "
                      f"kaydi={tesh['yer_degistiren_pct']:4.1f}% "
                      f"MAE={r['mae']:.4f} NDCG={r['ndcg10']:.4f}", flush=True)


if __name__ == "__main__":
    main()
