"""
TABLO B+ — (1) onarimli esit havuz + (2) warm start.

(1) REPAIR: cezali kisit yerine ONARIM. Kullanicilar merkeze uzakligina gore
    sirayla atanir; bir kume kapasiteye (ceil(N/K)) ulastiysa kullanici en yakin
    DOLU OLMAYAN kumeye gider. Kisit ihlali IMKANSIZ -> tum yontemlerin kume
    boyutlari ve havuzlari esitlenir; "fark havuz artefakti mi?" sorusu kapanir.
(2) WARM START: metalar B0(repair) merkezlerinden baslar (PACR mantigi).

Karsilastirma: B0_repair (baseline) vs {AVOA, GWO, HGS, NGO, HHO}_repair_warm.
Sistem B ile ayni: K=30, nmf+genre, kume-MF f=10, kNN k=20, beta val'de.
Pilot: --folds 1 2 --seeds 42 43 44
Cikti: results/tabloB_plus.csv
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
import cluster_mf as cm  # noqa: E402
from eksikler_deney import Ctx  # noqa: E402
from genre_k_deney import build_space, genre_profile  # noqa: E402
from pred_v2 import build_sims, full_metrics  # noqa: E402
from recompute_scores import discover_algorithms  # noqa: E402
from tam_run import BETAS, save_user_vec, user_mae_vec  # noqa: E402

K, KNN_K, F = 30, 20, 10
ALGOS = ["AVOA.OriginalAVOA", "GWO.OriginalGWO", "HGS.OriginalHGS",
         "NGO.OriginalNGO", "HHO.OriginalHHO"]


def repair_assign(X, cents):
    """Kapasiteli atama, REGRET (pismanlik) tabanli oncelik.

    Regret(u) = D[u, 2.yakin] - D[u, 1.yakin]; azalan sirada islenir.
    Alternatifi kotu olan kullanici once yerlesir; iki kumeye de yakin
    (kaydirilmasi ucuz) olanlar sona kalir. Ablasyon: repair_regret.csv
    """
    N, K_ = len(X), len(cents)
    cap = int(np.ceil(N / K_))
    d = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
    order_c = np.argsort(d, 1)
    en_yakin = d[np.arange(N), order_c[:, 0]]
    ikinci = d[np.arange(N), order_c[:, 1]] if K_ > 1 else en_yakin
    order_u = np.argsort(-(ikinci - en_yakin))       # azalan regret
    L = -np.ones(N, int); sizes = np.zeros(K_, int)
    for u in order_u:
        for c in order_c[u]:
            if sizes[c] < cap:
                L[u] = c; sizes[c] += 1; break
    # ikinci havuz kumesi: L'ye gore en yakin ikinci (kapasite serbest)
    near = np.empty((N, 2), int)
    near[:, 0] = L
    for u in range(N):
        for c in order_c[u]:
            if c != L[u]:
                near[u, 1] = c; break
    return L, near


def knn_from_pools(ctx, S, L, near, us, is_):
    p = np.empty(len(us))
    for j, (u, i) in enumerate(zip(us, is_)):
        cand = ctx.raters[i]
        cand = cand[(L[cand] == near[u, 0]) | (L[cand] == near[u, 1])]
        pk = np.nan
        if len(cand):
            s = S[u, cand]; t = np.argsort(-s)[:KNN_K]
            s, cc = s[t], cand[t]
            w = np.abs(s).sum()
            if w > 1e-9:
                pk = ctx.um[u] + (s * (ctx.R[cc, i] - ctx.um[cc])).sum() / w
        p[j] = pk
    return p


def make_fitness(ctx, X):
    """Repair atamali val-MAE (bias tahminci vekili). Ceza YOK — kisit yapisal."""
    def fit(sol):
        c = np.asarray(sol).reshape(K, X.shape[1])
        L, near = repair_assign(X, c)
        s_ = np.zeros((K, 1682)); n_ = np.zeros((K, 1682))
        np.add.at(s_, (L[ctx.iu], ctx.ii), ctx.ir - ctx.um[ctx.iu])
        np.add.at(n_, (L[ctx.iu], ctx.ii), 1.0)
        a, b = near[ctx.vu, 0], near[ctx.vu, 1]
        num = s_[a, ctx.vi] + s_[b, ctx.vi]; den = n_[a, ctx.vi] + n_[b, ctx.vi]
        dev = np.where(den > 0, num / np.maximum(den, 1), ctx.dev_g[ctx.vi])
        return float(np.abs(np.clip(ctx.um[ctx.vu] + dev, 1, 5) - ctx.vr).mean())
    return fit


def evaluate(ctx, X, S, cents):
    cm.K, cm.F = K, F
    L, near = repair_assign(X, cents)
    models = cm.cluster_mf_models(ctx, L)
    fbv = ctx.um[ctx.vu] + ctx.dev_g[ctx.vi]
    fbe = ctx.um[ctx.eu] + ctx.dev_g[ctx.ei]
    mv = cm.cmf_predict(models, L, ctx.vu, ctx.vi, fbv)
    me = cm.cmf_predict(models, L, ctx.eu, ctx.ei, fbe)
    pk_v = knn_from_pools(ctx, S, L, near, ctx.vu, ctx.vi)
    pk_e = knn_from_pools(ctx, S, L, near, ctx.eu, ctx.ei)
    best_b, best = None, np.inf
    for b in BETAS:
        pv = np.where(np.isnan(pk_v), mv, b * np.nan_to_num(pk_v) + (1 - b) * mv)
        v = float(np.abs(np.clip(pv, 1, 5) - ctx.vr).mean())
        if v < best:
            best_b, best = b, v
    p = np.clip(np.where(np.isnan(pk_e), me,
                         best_b * np.nan_to_num(pk_e) + (1 - best_b) * me), 1, 5)
    pool = int(np.mean([(np.isin(L, near[u])).sum() for u in range(len(X))]))
    m = full_metrics(ctx, p, 100 * float(np.isnan(pk_e).mean()), [np.arange(pool)])
    m["havuz"] = pool
    m["maxk"] = int(np.bincount(L, minlength=K).max())
    m["beta"] = best_b
    m["val_mae"] = round(best, 4)
    return p, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--max-fe", type=int, default=2000)
    ap.add_argument("--k", type=int, default=None,
                    help="calisma noktasi K (varsayilan 30). K=6 mutlak, K=40 fark")
    ap.add_argument("--uzay", choices=["nmf", "nmf+genre"], default="nmf",
                    help="nmf = saf CF (varsayilan), nmf+genre = tur yan bilgisi ile")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.k:
        global K
        K = args.k
        cm.K = args.k
    data = Path(ROOT.parent / "data" / "ml-100k")
    etiket = f"_K{K}" if args.k else ""
    if args.uzay == "nmf+genre":
        etiket += "_genre"
    out = RESULTS / f"tabloB_plus{etiket}.csv"
    upath = RESULTS / f"tabloB_plus{etiket}_userr.npz"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records")
        done = set(zip(old.yontem, old.fold, old.seed))

    import mealpy
    from mealpy import FloatVar
    algos = discover_algorithms(ALGOS)

    for fold in args.folds:
        ctx = Ctx(data, fold)
        X = build_space(ctx, genre_profile(ctx, data), args.uzay)
        S = build_sims(ctx)
        fitf = make_fitness(ctx, X)
        lb, ub = np.tile(X.min(0), K), np.tile(X.max(0), K)
        for seed in args.seeds:
            km = KMeans(K, init="k-means++", n_init=10, random_state=seed).fit(X)
            c0 = km.cluster_centers_
            if ("B0_repair", fold, seed) not in done:
                p, m = evaluate(ctx, X, S, c0)
                rows.append({"yontem": "B0_repair", "fold": fold, "seed": seed, **m})
                pd.DataFrame(rows).to_csv(out, index=False)
                save_user_vec(upath, f"B0_repair|f{fold}|s{seed}",
                              user_mae_vec(ctx, p))
                print(f"  B0_repair f{fold} s{seed} MAE={m['mae']:.4f} "
                      f"NDCG={m['ndcg10']:.4f} havuz={m['havuz']}", flush=True)
            rng = np.random.default_rng(seed)
            sig = 0.08 * X.std()
            starts = np.clip(np.vstack(
                [c0.ravel()] + [c0.ravel() + rng.normal(0, sig, c0.size)
                                for _ in range(14)]), lb, ub)
            for aname, cls in algos.items():
                kisa = aname.split(".")[0] + "_warm"
                if (kisa, fold, seed) in done:
                    continue
                model = cls(epoch=3000, pop_size=15)
                g = model.solve({"obj_func": fitf, "bounds": FloatVar(lb=lb, ub=ub),
                                 "minmax": "min", "log_to": None},
                                seed=seed, termination={"max_fe": args.max_fe},
                                starting_solutions=starts)
                p, m = evaluate(ctx, X, S,
                                np.asarray(g.solution).reshape(K, X.shape[1]))
                rows.append({"yontem": kisa, "fold": fold, "seed": seed, **m})
                pd.DataFrame(rows).to_csv(out, index=False)
                save_user_vec(upath, f"{kisa}|f{fold}|s{seed}", user_mae_vec(ctx, p))
                print(f"  {kisa} f{fold} s{seed} MAE={m['mae']:.4f} "
                      f"NDCG={m['ndcg10']:.4f} havuz={m['havuz']}", flush=True)


if __name__ == "__main__":
    main()
