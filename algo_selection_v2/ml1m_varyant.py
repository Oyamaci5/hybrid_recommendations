"""
ML-1M VARYANT DENEYLERI — uc iyilestirme adayi, EK AYAR YAPMADAN.

V1 olcekli butce : NFE ve pop_size K ile olceklenir (arama uzayi K*D boyutlu,
                   sabit butce buyuk K'da yetersiz kaliyor olabilir).
                   nfe = taban * (K/20), pop = 10 + K//10  (kaba, ayarsiz kural)
V2 memetik       : AVOA + her N nesilde en iyi coozume 1 Lloyd adimi
                   (merkez = kume ortalamasi). Klasik memetik yerel arama.
V3 hibrit        : AVOA yarim butce -> en iyi cozumden HGS ile devam
                   (toplam NFE ayni; AVOA'nin kesif, HGS'nin somuru fazi)

Referans: duz AVOA (mevcut protokol, ayni toplam NFE).
Kullanim: python algo_selection_v2/ml1m_varyant.py --klist 20 40 60 [--max-fe 600]
Cikti: results/ml1m_varyant.csv
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
import ml1m_run as m  # noqa: E402
from ctx_ml1m import Ctx1M  # noqa: E402
from recompute_scores import discover_algorithms  # noqa: E402


def lloyd_step(X, cents):
    """Tek Lloyd adimi: kapasiteli atama -> kume ortalamasi."""
    L, _ = m.fast_repair(X, cents)
    K = len(cents)
    yeni = np.vstack([X[L == c].mean(0) if (L == c).any() else cents[c]
                      for c in range(K)])
    return yeni


def solve_memetik(cls, fitf, X, K, seed, c0, max_fe, pop=10, adim=3):
    """AVOA + periyodik Lloyd yerel aramasi (butce parcalara bolunur)."""
    cent = c0
    kalan = max_fe
    pay = max_fe // adim
    for _ in range(adim):
        b = min(pay, kalan)
        if b <= 0:
            break
        cent = m.solve_warm(cls, fitf, X, K, seed, cent, max_fe=b, pop=pop)
        cent = lloyd_step(X, cent)          # yerel arama
        kalan -= b
    return cent


def solve_hibrit(cls_a, cls_b, fitf, X, K, seed, c0, max_fe, pop=10):
    """Yari butce AVOA (kesif) -> yari butce HGS (somuru), warm devir."""
    ara = m.solve_warm(cls_a, fitf, X, K, seed, c0, max_fe=max_fe // 2, pop=pop)
    return m.solve_warm(cls_b, fitf, X, K, seed + 1, ara,
                        max_fe=max_fe - max_fe // 2, pop=pop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klist", type=int, nargs="+", default=[20, 40, 60])
    ap.add_argument("--max-fe", type=int, default=600)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data" / "ml-1m"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    ctx = Ctx1M(Path(args.data_dir), args.fold)
    X = m.build_space(ctx, None, "nmf")
    al = discover_algorithms(["AVOA.OriginalAVOA", "HGS.OriginalHGS"])
    A, H = al["AVOA.OriginalAVOA"], al["HGS.OriginalHGS"]

    out = RESULTS / "ml1m_varyant.csv"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records")
        done = set(zip(old.varyant, old.K))

    for K in args.klist:
        cm.K = K
        fitf = m.make_fitness(ctx, X, K)
        km = KMeans(K, init="k-means++", n_init=3, random_state=args.seed).fit(X)
        c0 = km.cluster_centers_
        nfe_olc = int(args.max_fe * K / 20)          # V1 kurali
        pop_olc = 10 + K // 10
        plan = [
            ("AVOA_duz", lambda: m.solve_warm(A, fitf, X, K, args.seed, c0,
                                              max_fe=args.max_fe, pop=10),
             args.max_fe, 10),
            ("V1_olcekli", lambda: m.solve_warm(A, fitf, X, K, args.seed, c0,
                                                max_fe=nfe_olc, pop=pop_olc),
             nfe_olc, pop_olc),
            ("V2_memetik", lambda: solve_memetik(A, fitf, X, K, args.seed, c0,
                                                 args.max_fe, pop=10),
             args.max_fe, 10),
            ("V3_hibrit", lambda: solve_hibrit(A, H, fitf, X, K, args.seed, c0,
                                               args.max_fe, pop=10),
             args.max_fe, 10),
        ]
        for ad, fn, nfe, pop in plan:
            if (ad, K) in done:
                continue
            t0 = time.time()
            cent = fn()
            p, mm = m.evaluate(ctx, X, cent, K)
            rows.append({"varyant": ad, "K": K, "nfe": nfe, "pop": pop,
                         "sure_s": round(time.time() - t0, 1),
                         "mae": mm["mae"], "rmse": mm["rmse"],
                         "ndcg10": mm["ndcg10"], "prec10": mm["prec10"],
                         "havuz": mm["havuz"], "beta": mm["beta"]})
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"K={K:2d} {ad:11s} nfe={nfe:5d} pop={pop:2d} "
                  f"MAE={mm['mae']:.4f} NDCG={mm['ndcg10']:.4f} "
                  f"({rows[-1]['sure_s']}s)", flush=True)


if __name__ == "__main__":
    main()
