"""
AVOA HIPERPARAMETRE + ARAMA BUTCESI AYARI (adil protokolle).

AVOA parametreleri (mealpy): p1, p2, p3, alpha, gama.
Ek arama ayarlari: pop_size, warm-start sapmasi sigma, NFE.

ADALET KURALI: secim IC-VAL fitness'i ile yapilir (test'e bakilmaz). Kazanan
"arama butcesi" ayarlari (pop/sigma/NFE) TUM rakiplere uygulanir; algoritmaya
ozgu parametreler (p1..gama) yalnizca AVOA'da anlamli oldugundan, rakiplerin
kendi parametreleri de ayni protokolle ayarlanmali (--exp rakip).

--exp bud    : pop_size x sigma x NFE izgarasi (AVOA, ic-val)
--exp avoa   : p1/p2/p3/alpha/gama izgarasi (en iyi butce ile)
--exp rakip  : HGS(PUP) ve HHO/GWO/NGO (parametresiz) ayni butcede
Cikti: results/avoa_tune.csv
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
import tabloB_plus as bp  # noqa: E402
from eksikler_deney import Ctx  # noqa: E402
from genre_k_deney import build_space, genre_profile  # noqa: E402
from pred_v2 import build_sims  # noqa: E402
from recompute_scores import discover_algorithms  # noqa: E402

OUT = RESULTS / "avoa_tune.csv"
K = 6   # calisma noktasi A (mutlak); B icin --k 40


def solve(cls, fitf, X, K_, seed, pop, sigma, nfe, c0, **kw):
    from mealpy import FloatVar
    lb, ub = np.tile(X.min(0), K_), np.tile(X.max(0), K_)
    rng = np.random.default_rng(seed)
    starts = np.clip(np.vstack(
        [c0.ravel()] + [c0.ravel() + rng.normal(0, sigma * X.std(), c0.size)
                        for _ in range(pop - 1)]), lb, ub)
    g = cls(epoch=5000, pop_size=pop, **kw).solve(
        {"obj_func": fitf, "bounds": FloatVar(lb=lb, ub=ub), "minmax": "min",
         "log_to": None},
        seed=seed, termination={"max_fe": nfe}, starting_solutions=starts)
    return np.asarray(g.solution).reshape(K_, X.shape[1]), float(g.target.fitness)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["bud", "avoa", "rakip"], required=True)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pop", type=int, default=15)
    ap.add_argument("--sigma", type=float, default=0.08)
    ap.add_argument("--nfe", type=int, default=1200)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    data = Path(ROOT.parent / "data" / "ml-100k")
    ctx = Ctx(data, 1)
    X = build_space(ctx, genre_profile(ctx, data), "nmf+genre")
    S = build_sims(ctx)
    K_ = args.k
    bp.K = K_; cm.K = K_
    fitf = bp.make_fitness(ctx, X)
    km = KMeans(K_, init="k-means++", n_init=10, random_state=args.seed).fit(X)
    c0 = km.cluster_centers_

    rows, done = [], set()
    if args.resume and OUT.exists() and OUT.stat().st_size > 10:
        old = pd.read_csv(OUT); rows = old.to_dict("records"); done = set(old.config)

    def add(cfg, cls, kw, pop, sigma, nfe):
        if cfg in done:
            return
        cent, fv = solve(cls, fitf, X, K_, args.seed, pop, sigma, nfe, c0, **kw)
        _, m = bp.evaluate(ctx, X, S, cent)
        rows.append({"config": cfg, "K": K_, "val_fit": round(fv, 5),
                     "pop": pop, "sigma": sigma, "nfe": nfe,
                     "mae": m["mae"], "ndcg10": m["ndcg10"], "havuz": m["havuz"]})
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"{cfg:34s} val={fv:.5f}  MAE={m['mae']:.4f} "
              f"NDCG={m['ndcg10']:.4f}", flush=True)

    A = discover_algorithms(["AVOA.OriginalAVOA"])["AVOA.OriginalAVOA"]

    if args.exp == "bud":
        for pop in (10, 15, 30):
            for sig in (0.04, 0.08, 0.16):
                add(f"bud_pop{pop}_sig{sig}", A, {}, pop, sig, args.nfe)
        for nfe in (600, 2400):
            add(f"bud_nfe{nfe}", A, {}, args.pop, args.sigma, nfe)

    elif args.exp == "avoa":
        base = dict(p1=0.6, p2=0.4, p3=0.6, alpha=0.8, gama=2.5)
        for name, vals in [("p1", (0.3, 0.6, 0.9)), ("p2", (0.2, 0.4, 0.7)),
                           ("p3", (0.3, 0.6, 0.9)), ("alpha", (0.5, 0.8, 0.95)),
                           ("gama", (1.5, 2.5, 3.5))]:
            for v in vals:
                kw = dict(base); kw[name] = v
                add(f"avoa_{name}={v}", A, kw, args.pop, args.sigma, args.nfe)

    else:  # rakip
        for nm in ("HGS.OriginalHGS", "HHO.OriginalHHO", "GWO.OriginalGWO",
                   "NGO.OriginalNGO"):
            cls = discover_algorithms([nm])[nm]
            kisa = nm.split(".")[0]
            if kisa == "HGS":
                for pup in (0.04, 0.08, 0.16):
                    add(f"HGS_PUP={pup}", cls, {"PUP": pup}, args.pop,
                        args.sigma, args.nfe)
            else:
                add(f"{kisa}_varsayilan", cls, {}, args.pop, args.sigma, args.nfe)


if __name__ == "__main__":
    main()
