
"""
ML-1M KOMSU GERI CAGIRMA (recall@k) OLCUMU.

Soru: Gercek en-benzer k komsunun kaci kumeleme havuzunda kaliyor?
Kumeleme = yaklasik en yakin komsu (ANN) indeksi; bu metrik onun kalitesidir.

Olculen: her kullanici icin S[u,:] uzerinden gercek top-k komsu -> havuzda
kalma orani. Orneklem: hiz icin kullanicilarin rastgele bir alt kumesi
(--ornek, varsayilan 1500; tam olcum icin --ornek 6040).

Kullanim:
  python algo_selection_v2/ml1m_recall.py --klist 6 20 40 --tum
  (--tum: 5 algoritma; yoksa yalniz AVOA)
Cikti: results/ml1m_recall.csv
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
from ctx_ml1m import N_U, Ctx1M  # noqa: E402
from recompute_scores import discover_algorithms  # noqa: E402


def recall_olc(S, L, near, ornek_idx, topk=20):
    """Gercek top-k komsunun havuzda kalma orani (orneklem uzerinden)."""
    rec = np.empty(len(ornek_idx))
    for j, u in enumerate(ornek_idx):
        gercek = np.argpartition(-S[u], topk)[:topk]
        rec[j] = ((L[gercek] == near[u, 0]) | (L[gercek] == near[u, 1])).mean()
    return float(rec.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klist", type=int, nargs="+", default=[6, 20, 40])
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--ornek", type=int, default=1500)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-fe", type=int, default=400)
    ap.add_argument("--tum", action="store_true", help="5 algoritma kos")
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data" / "ml-1m"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    ctx = Ctx1M(Path(args.data_dir), args.fold)
    X = m.build_space(ctx, None, "nmf")
    rng = np.random.default_rng(7)
    ornek = (np.arange(N_U) if args.ornek >= N_U
             else rng.choice(N_U, args.ornek, replace=False))
    print(f"orneklem: {len(ornek)} kullanici, top-{args.topk} komsu", flush=True)

    adlar = m.ALGOS if args.tum else ["AVOA.OriginalAVOA"]
    algos = discover_algorithms(adlar)
    out = RESULTS / "ml1m_recall.csv"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records")
        done = set(zip(old.yontem, old.K))

    for K in args.klist:
        cm.K = K
        km = KMeans(K, init="k-means++", n_init=3, random_state=args.seed).fit(X)
        c0 = km.cluster_centers_
        merkez = {"B0": c0}
        fitf = m.make_fitness(ctx, X, K)
        for aname, cls in algos.items():
            kisa = aname.split(".")[0]
            if (kisa, K) in done:
                continue
            merkez[kisa] = m.solve_warm(cls, fitf, X, K, args.seed, c0,
                                        max_fe=args.max_fe, pop=10)
        for ad, cent in merkez.items():
            if (ad, K) in done:
                continue
            t0 = time.time()
            L, near = m.fast_repair(X, cent)
            r = recall_olc(ctx.S, L, near, ornek, args.topk)
            havuz = int(np.mean([(np.isin(L, near[u])).sum()
                                 for u in ornek[:300]]))
            rows.append({"yontem": ad, "K": K, "topk": args.topk,
                         "recall": round(r, 4), "recall_pct": round(100 * r, 1),
                         "havuz": havuz,
                         "havuz_pct": round(100 * havuz / N_U, 1),
                         "sure_s": round(time.time() - t0, 1)})
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"K={K:2d} {ad:5s} recall@{args.topk}={100*r:5.1f}%  "
                  f"havuz={havuz} (%{100*havuz/N_U:.0f})", flush=True)


if __name__ == "__main__":
    main()
