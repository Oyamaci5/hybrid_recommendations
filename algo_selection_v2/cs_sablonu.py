
"""
CS-KMEANS SABLONU + AVOA IKAMESI (literaturun kendi algoritmik yapisi).

CS-Kmeans makalesinin sozde kodu (Sekil 2) su dongudur:
  1. Merkezleri metasezgisel operatorle guncelle       (Levy ucusu, Formul 1)
  2. r ~ U(0,1) < Pa ise bir merkezi AT ve rastgele yenile   (terk etme)
  3. Ornekleri en yakin merkeze ata; merkezleri ORTALAMA ile yenile  (Lloyd!)
  4. Yeni uygunluk G, eskisi G1'den iyiyse KABUL, degilse eskiyi koru (elitizm)

Bizim yontemimizden uc farki:
  (a) Lloyd adimi dongu ICINDE  -> memetik / Lamarckian
  (b) terk etme (Pa) mekanizmasi -> cesitlilik
  (c) elitist kabul               -> monoton iyilesme

Kollar:
  CS_levy     : orijinal (Levy ucusu operatoru)
  CS_avoa     : ayni sablon, operator AVOA guncellemesi   <- "ikame"
  CS_avoa_noL : ayni sablon, Lloyd adimi KAPALI (katkisini izole eder)
  BIZIM       : mevcut yontem (warm start + kapasiteli, Lloyd yok)

Tum kollar AYNI degerlendirmeyle olculur (repair + soft top-2 + kNN/MF).
Fitness: --fit wcss (literaturun hedefi) veya pred (bizim hedef).

Kullanim: python algo_selection_v2/cs_sablonu.py --k 6 --fit pred --iters 60
Cikti: results/cs_sablonu.csv
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
import tabloB_plus as bp  # noqa: E402
from eksikler_deney import Ctx  # noqa: E402
from genre_k_deney import build_space, genre_profile  # noqa: E402
from pred_v2 import build_sims  # noqa: E402


# ---------------- operatorler ----------------
def levy(boyut, rng, beta=1.5):
    """Mantegna Levy ucusu."""
    sg = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
          / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sg, boyut); v = rng.normal(0, 1, boyut)
    return u / np.abs(v) ** (1 / beta)


def op_levy(P, best, rng, alpha=0.01):
    """CS: x_i^{t+1} = x_i^t + alpha * Levy"""
    return P + alpha * levy(P.shape, rng) * (P - best)


def op_avoa(P, best, ikinci, rng, t, T, p1=0.6, p2=0.4, alpha=0.8, gama=2.5):
    """AVOA guncelleme kuralinin sadelestirilmis hali (kesif/somuru fazlari)."""
    n, d = P.shape
    F = (2 * rng.random(n) + 1) * (1 - t / T) * (2 * rng.random(n) - 1)
    R = np.where(rng.random(n) < alpha, 0, 1)[:, None]
    hedef = R * best + (1 - R) * ikinci
    yeni = np.empty_like(P)
    for i in range(n):
        f = F[i]
        if abs(f) >= 1:                       # kesif
            if rng.random() < p1:
                yeni[i] = hedef[i] - (abs(2 * rng.random() * hedef[i] - P[i]) * f)
            else:
                yeni[i] = hedef[i] - f + rng.random() * (
                    (P.max(0) - P.min(0)) * rng.random() + P.min(0))
        else:                                  # somuru
            if rng.random() < p2:
                d1 = hedef[i] - P[i]
                yeni[i] = hedef[i] * (rng.random() * P[i] / (2 * np.pi)) * np.cos(P[i]) \
                    + hedef[i] * (rng.random() * P[i] / (2 * np.pi)) * np.sin(P[i]) \
                    + 0.0 * d1
            else:
                A = best - (abs(best - P[i]) * f) / 2
                B = ikinci - (abs(ikinci - P[i]) * f) / 2
                yeni[i] = (A + B) / 2
    return yeni


# ---------------- CS sablonu ----------------
def cs_sablon(X, K, fitf, rng, iters=60, pop=10, Pa=0.25, operator="levy",
              lloyd=True, c0=None):
    """Makalenin dongusu: operator -> terk etme -> (Lloyd) -> elitist kabul."""
    D = X.shape[1]
    if c0 is None:
        P = np.stack([X[rng.choice(len(X), K, replace=False)].ravel()
                      for _ in range(pop)])
    else:
        P = np.stack([c0.ravel()] + [c0.ravel() + rng.normal(0, .08 * X.std(), c0.size)
                                     for _ in range(pop - 1)])
    F = np.array([fitf(p) for p in P])
    nfe = pop
    for t in range(iters):
        b1, b2 = np.argsort(F)[:2]
        best, ikinci = P[b1], P[b2]
        # 1) operator
        if operator == "levy":
            Y = op_levy(P, best, rng)
        else:
            Y = op_avoa(P.reshape(pop, -1), best, ikinci, rng, t, iters)
        # 2) terk etme (Pa)
        mask = rng.random(pop) < Pa
        for i in np.flatnonzero(mask):
            c = rng.integers(K)
            Y[i].reshape(K, D)[c] = X[rng.integers(len(X))]
        # 3) Lloyd adimi (makalenin Adim 4'u)
        if lloyd:
            for i in range(pop):
                C = Y[i].reshape(K, D)
                d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
                L = d.argmin(1)
                for c in range(K):
                    if (L == c).any():
                        C[c] = X[L == c].mean(0)
                Y[i] = C.ravel()
        # 4) elitist kabul
        Fy = np.array([fitf(y) for y in Y]); nfe += pop
        iyi = Fy < F
        P[iyi], F[iyi] = Y[iyi], Fy[iyi]
    return P[np.argmin(F)].reshape(K, D), float(F.min()), nfe


def wcss_fit(X, K):
    X2 = (X ** 2).sum(1)

    def f(sol):
        C = np.asarray(sol).reshape(K, X.shape[1])
        d = X2[:, None] - 2.0 * (X @ C.T) + (C ** 2).sum(1)[None, :]
        return float(np.maximum(d.min(1), 0).sum())
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--fit", choices=["wcss", "pred"], default="pred")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--pa", type=float, default=0.25)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    data = Path(ROOT.parent / "data" / "ml-100k")
    ctx = Ctx(data, args.fold)
    X = build_space(ctx, genre_profile(ctx, data), "nmf")
    S = build_sims(ctx)
    K = args.k
    bp.K = K; cm.K = K
    fitf = wcss_fit(X, K) if args.fit == "wcss" else bp.make_fitness(ctx, X)
    km = KMeans(K, init="k-means++", n_init=10, random_state=args.seed).fit(X)
    c0 = km.cluster_centers_

    out = RESULTS / "cs_sablonu.csv"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records")
        done = set(zip(old.kol, old.K, old.fitness, old.seed))

    kollar = [
        ("CS_levy",      dict(operator="levy", lloyd=True,  c0=None)),
        ("CS_avoa",      dict(operator="avoa", lloyd=True,  c0=None)),
        ("CS_avoa_warm", dict(operator="avoa", lloyd=True,  c0=c0)),
        ("CS_avoa_noL",  dict(operator="avoa", lloyd=False, c0=c0)),
    ]
    for ad, kw in kollar:
        if (ad, K, args.fit, args.seed) in done:
            continue
        rng = np.random.default_rng(args.seed)
        t0 = time.time()
        C, fv, nfe = cs_sablon(X, K, fitf, rng, args.iters, args.pop, args.pa, **kw)
        _, m = bp.evaluate(ctx, X, S, C)
        rows.append({"kol": ad, "K": K, "fitness": args.fit, "seed": args.seed,
                     "iters": args.iters, "nfe": nfe, "fit_val": round(fv, 5),
                     "sure_s": round(time.time() - t0, 1), **m})
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{ad:13s} nfe={nfe:5d} fit={fv:.5f} MAE={m['mae']:.4f} "
              f"NDCG={m['ndcg10']:.4f} ({rows[-1]['sure_s']}s)", flush=True)

    # referans: bizim yontem (ayni NFE)
    if ("BIZIM", K, args.fit, args.seed) not in done:
        from mealpy import FloatVar
        from recompute_scores import discover_algorithms
        cls = discover_algorithms(["AVOA.OriginalAVOA"])["AVOA.OriginalAVOA"]
        nfe_hedef = args.iters * args.pop * 2
        lb, ub = np.tile(X.min(0), K), np.tile(X.max(0), K)
        rng = np.random.default_rng(args.seed)
        st = np.clip(np.vstack([c0.ravel()] + [
            c0.ravel() + rng.normal(0, .08 * X.std(), c0.size)
            for _ in range(args.pop - 1)]), lb, ub)
        t0 = time.time()
        g = cls(epoch=5000, pop_size=args.pop).solve(
            {"obj_func": fitf, "bounds": FloatVar(lb=lb, ub=ub), "minmax": "min",
             "log_to": None}, seed=args.seed,
            termination={"max_fe": nfe_hedef}, starting_solutions=st)
        _, m = bp.evaluate(ctx, X, S, np.asarray(g.solution).reshape(K, X.shape[1]))
        rows.append({"kol": "BIZIM", "K": K, "fitness": args.fit,
                     "seed": args.seed, "iters": args.iters, "nfe": nfe_hedef,
                     "fit_val": round(float(g.target.fitness), 5),
                     "sure_s": round(time.time() - t0, 1), **m})
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{'BIZIM':13s} nfe={nfe_hedef:5d} MAE={m['mae']:.4f} "
              f"NDCG={m['ndcg10']:.4f}", flush=True)


if __name__ == "__main__":
    main()
