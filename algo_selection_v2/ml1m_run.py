"""
ML-1M HEDEFLI DOGRULAMA KOSUCUSU.

ML-100K protokolunun aynisi (repair kapasite + warm start + soft top-2 +
kume-MF + kNN karisimi), yalnizca olcek parametreleri 1M'e uyarlanmis.

--exp ksweep : K = {20,40,60,100} x {B0, AVOA}, tek fold/seed -> calisma noktasi
--exp ana    : secilen K'da 6 yontem x fold x seed -> ana tablo
Cikti: results/ml1m_ksweep.csv, results/ml1m_ana.csv (+ _userr.npz)

Ornek:
  python algo_selection_v2/ml1m_run.py --exp ksweep --resume
  python algo_selection_v2/ml1m_run.py --exp ana --k 40 --folds 1 2 3 --seeds 42 43 44 45 46 --resume
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
from ctx_ml1m import N_I, N_U, Ctx1M, genre_profile_1m  # noqa: E402
from pred_v2 import full_metrics  # noqa: E402
from recompute_scores import discover_algorithms  # noqa: E402
from tabloB_plus import repair_assign  # noqa: E402
from tam_run import save_user_vec, user_mae_vec  # noqa: E402

cm.N_I = N_I                      # kume-MF'i 3952 filme uyarla
KNN_K, BETAS = 20, (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
ALGOS = ["AVOA.OriginalAVOA", "NGO.OriginalNGO", "HHO.OriginalHHO",
         "HGS.OriginalHGS", "GWO.OriginalGWO"]


def build_space(ctx, G, mode, dim=20):
    B = ctx.nmf_space(dim)
    if mode == "nmf":
        return B
    s = B.std(0).mean() / max(G.std(0).mean(), 1e-9)
    return np.ascontiguousarray(np.hstack([B, G * s]))


def fast_repair(X, cents, X2=None):
    """Kapasiteli atama, REGRET (pismanlik) tabanli oncelik.

    Regret(u) = D[u, 2.yakin] - D[u, 1.yakin]; azalan sirada islenir.
    Gerekce: alternatifi kotu olan kullanici once yerlessin; kaydirilmasi
    ucuz olanlar (iki kumeye de yakin) sona kalsin. Kapasiteli atama
    yazininda standart kural. Ablasyon: results/repair_regret.csv
    """
    N, K_ = len(X), len(cents)
    if X2 is None:
        X2 = (X ** 2).sum(1)
    d = X2[:, None] - 2.0 * (X @ cents.T) + (cents ** 2).sum(1)[None, :]
    order_c = np.argsort(d, 1)
    en_yakin = d[np.arange(N), order_c[:, 0]]
    ikinci = d[np.arange(N), order_c[:, 1]] if K_ > 1 else en_yakin
    sira = np.argsort(-(ikinci - en_yakin))          # azalan regret
    cap = int(np.ceil(N / K_))
    L = -np.ones(N, np.int32); sizes = np.zeros(K_, np.int32)
    for u in sira:
        row = order_c[u]
        for c in row:
            if sizes[c] < cap:
                L[u] = c; sizes[c] += 1; break
    near = np.empty((N, 2), np.int32)
    near[:, 0] = L
    ilk = order_c[:, 0]
    near[:, 1] = np.where(ilk != L, ilk, order_c[:, 1])
    return L, near


def make_fitness(ctx, X, K):
    """Repair atamali soft top-2 bias val-MAE. bincount ile hizlandirilmis."""
    X2 = (X ** 2).sum(1)
    dev_i = (ctx.ir - ctx.um[ctx.iu])
    vi, vu, vr = ctx.vi, ctx.vu, ctx.vr
    dg = ctx.dev_g[vi]
    um_v = ctx.um[vu]

    def fit(sol):
        c = np.asarray(sol).reshape(K, X.shape[1])
        L, near = fast_repair(X, c, X2)
        flat = L[ctx.iu].astype(np.int64) * N_I + ctx.ii
        s_ = np.bincount(flat, weights=dev_i, minlength=K * N_I)
        n_ = np.bincount(flat, minlength=K * N_I)
        a = near[vu, 0].astype(np.int64) * N_I + vi
        b = near[vu, 1].astype(np.int64) * N_I + vi
        num = s_[a] + s_[b]; den = n_[a] + n_[b]
        dev = np.where(den > 0, num / np.maximum(den, 1), dg)
        return float(np.abs(np.clip(um_v + dev, 1, 5) - vr).mean())
    return fit


def knn_pool(ctx, L, near, us, is_):
    p = np.empty(len(us))
    S = ctx.S
    for j in range(len(us)):
        u, i = us[j], is_[j]
        cand = ctx.raters[i]
        if len(cand) == 0:
            p[j] = np.nan; continue
        lc = L[cand]
        cand = cand[(lc == near[u, 0]) | (lc == near[u, 1])]
        pk = np.nan
        if len(cand):
            s = S[u, cand]
            if len(cand) > KNN_K:
                t = np.argpartition(-s, KNN_K)[:KNN_K]
                s, cand = s[t], cand[t]
            w = np.abs(s).sum()
            if w > 1e-9:
                pk = ctx.um[u] + float((s * (ctx.R[cand, i] - ctx.um[cand])).sum() / w)
        p[j] = pk
    return p


def evaluate(ctx, X, cents, K):
    L, near = fast_repair(X, cents)
    models = cm.cluster_mf_models(ctx, L)
    fbv = ctx.um[ctx.vu] + ctx.dev_g[ctx.vi]
    fbe = ctx.um[ctx.eu] + ctx.dev_g[ctx.ei]
    mv = cm.cmf_predict(models, L, ctx.vu, ctx.vi, fbv)
    me = cm.cmf_predict(models, L, ctx.eu, ctx.ei, fbe)
    pk_v = knn_pool(ctx, L, near, ctx.vu, ctx.vi)
    pk_e = knn_pool(ctx, L, near, ctx.eu, ctx.ei)
    best_b, best = None, np.inf
    for b in BETAS:
        pv = np.where(np.isnan(pk_v), mv, b * np.nan_to_num(pk_v) + (1 - b) * mv)
        v = float(np.abs(np.clip(pv, 1, 5) - ctx.vr).mean())
        if v < best:
            best_b, best = b, v
    p = np.clip(np.where(np.isnan(pk_e), me,
                         best_b * np.nan_to_num(pk_e) + (1 - best_b) * me), 1, 5)
    pool = int(np.mean([(np.isin(L, near[u])).sum()
                        for u in range(0, N_U, 10)]))     # 1/10 orneklem (hiz)
    m = full_metrics(ctx, p, 100 * float(np.isnan(pk_e).mean()), [np.arange(pool)])
    m.update({"havuz": pool, "maxk": int(np.bincount(L, minlength=K).max()),
              "beta": best_b, "val_mae": round(best, 4)})
    return p, m


def solve_warm(cls, fitf, X, K, seed, c0, max_fe, pop=15):
    from mealpy import FloatVar
    lb, ub = np.tile(X.min(0), K), np.tile(X.max(0), K)
    rng = np.random.default_rng(seed)
    starts = np.clip(np.vstack(
        [c0.ravel()] + [c0.ravel() + rng.normal(0, 0.08 * X.std(), c0.size)
                        for _ in range(pop - 1)]), lb, ub)
    g = cls(epoch=5000, pop_size=pop).solve(
        {"obj_func": fitf, "bounds": FloatVar(lb=lb, ub=ub), "minmax": "min",
         "log_to": None},
        seed=seed, termination={"max_fe": max_fe}, starting_solutions=starts)
    return np.asarray(g.solution).reshape(K, X.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["ksweep", "ana"], required=True)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--klist", type=int, nargs="+",
                    default=[6, 10, 20, 40, 60])
    ap.add_argument("--folds", type=int, nargs="+", default=[1])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--uzay", choices=["nmf", "nmf+genre"], default="nmf")
    ap.add_argument("--max-fe", type=int, default=1200)
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data" / "ml-1m"))
    ap.add_argument("--tum", action="store_true",
                    help="ksweep'te TUM algoritmalari kos (varsayilan: yalniz AVOA)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    data = Path(args.data_dir)

    out = RESULTS / f"ml1m_{args.exp}.csv"
    upath = RESULTS / f"ml1m_{args.exp}_userr.npz"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records")
        done = set(zip(old.yontem, old.K, old.fold, old.seed))

    sec = ALGOS if args.tum else ["AVOA.OriginalAVOA"]
    algos = discover_algorithms(ALGOS if args.exp == "ana" else sec)
    ks = args.klist if args.exp == "ksweep" else [args.k]

    for fold in args.folds:
        t0 = time.time()
        ctx = Ctx1M(data, fold)
        G = genre_profile_1m(ctx, data) if args.uzay != "nmf" else None
        X = build_space(ctx, G, args.uzay)
        print(f"  baglam+uzay {time.time()-t0:.0f}s", flush=True)
        for K in ks:
            cm.K = K
            fitf = make_fitness(ctx, X, K)
            for seed in args.seeds:
                km = KMeans(K, init="k-means++", n_init=5,
                            random_state=seed).fit(X)
                c0 = km.cluster_centers_
                todo = [("B0_repair", None)] + [(a.split(".")[0] + "_warm", a)
                                                for a in algos]
                for yn, aname in todo:
                    if (yn, K, fold, seed) in done:
                        continue
                    t1 = time.time()
                    cent = c0 if aname is None else solve_warm(
                        algos[aname], fitf, X, K, seed, c0, args.max_fe)
                    p, m = evaluate(ctx, X, cent, K)
                    rows.append({"yontem": yn, "K": K, "fold": fold,
                                 "seed": seed, "sure_s": round(time.time()-t1, 1),
                                 **m})
                    pd.DataFrame(rows).to_csv(out, index=False)
                    save_user_vec(upath, f"{yn}|K{K}|f{fold}|s{seed}",
                                  user_mae_vec_1m(ctx, p))
                    print(f"  K={K} {yn:11s} f{fold} s{seed} MAE={m['mae']:.4f} "
                          f"NDCG={m['ndcg10']:.4f} havuz={m['havuz']} "
                          f"({m['sure_s'] if 'sure_s' in m else round(time.time()-t1)}s)",
                          flush=True)


def user_mae_vec_1m(ctx, p):
    v = np.full(N_U, np.nan)
    for u in range(N_U):
        idx = ctx.by_user[u]
        if idx:
            i = np.array(idx)
            v[u] = float(np.abs(p[i] - ctx.er[i]).mean())
    return v


if __name__ == "__main__":
    main()
