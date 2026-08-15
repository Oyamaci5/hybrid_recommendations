
"""
BUYUK K + KUME-ORTALAMASI DENEYI — literaturun kurulumunu birebir taklit.

Soru: HSC (K=70), Firefly (K=90) gibi calismalar buyuk K + kume-ortalamasi
kullaniyor. Bu kurulumda sonuc ne oluyor ve kume dagilimlari nasil?

Kosullar: repair YOK (literaturdeki gibi serbest Voronoi), tahminci = kume-film
ortalamasi (fallback: film ort. -> kullanici ort. -> global ort.)
K in {70, 90, 120} + kiyas icin bizim tahminci (kNN+MF).

Cikti: results/ml1m_buyukK.csv  (kume boyut dagilimi dahil)
Kullanim: python algo_selection_v2/ml1m_buyukK_cmean.py --klist 70 90 120
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
import ml1m_run as m  # noqa: E402
from ctx_ml1m import N_I, N_U, Ctx1M  # noqa: E402
from ml1m_tahminci2 import free_assign  # noqa: E402
from pred_v2 import full_metrics  # noqa: E402
from recompute_scores import discover_algorithms  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klist", type=int, nargs="+", default=[70, 90, 120])
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-fe", type=int, default=400)
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data" / "ml-1m"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    ctx = Ctx1M(Path(args.data_dir), args.fold)
    X = m.build_space(ctx, None, "nmf")
    im = np.bincount(ctx.ii, weights=ctx.ir, minlength=N_I) / np.maximum(
        np.bincount(ctx.ii, minlength=N_I), 1)
    im = np.where(np.bincount(ctx.ii, minlength=N_I) > 0, im, ctx.gmean)
    cls = discover_algorithms(["AVOA.OriginalAVOA"])["AVOA.OriginalAVOA"]

    out = RESULTS / "ml1m_buyukK.csv"
    rows, done = [], set()
    if args.resume and out.exists() and out.stat().st_size > 10:
        old = pd.read_csv(out); rows = old.to_dict("records")
        done = set(zip(old.yontem, old.K, old.tahminci, old.mod))

    for K in args.klist:
        cm.K = K
        km = KMeans(K, init="k-means++", n_init=3, random_state=args.seed).fit(X)
        c0 = km.cluster_centers_
        merkezler = {"B0": c0}
        # AVOA (kisitsiz fitness — literatur kurulumu)
        pos = m.solve_warm(cls, m.make_fitness(ctx, X, K), X, K, args.seed, c0,
                           max_fe=args.max_fe, pop=10)
        merkezler["AVOA"] = pos

        for ad, cent in merkezler.items():
            for mod in ("repairSIZ", "repairLI"):
                L, near = (free_assign(X, cent) if mod == "repairSIZ"
                           else m.fast_repair(X, cent))
                sz = np.bincount(L, minlength=K)
                bos = int((sz == 0).sum())
                # --- literatur tahmincisi: kume-film ortalamasi ---
                flat = L[ctx.iu].astype(np.int64) * N_I + ctx.ii
                s = np.bincount(flat, weights=ctx.ir, minlength=K * N_I)
                n = np.bincount(flat, minlength=K * N_I)
                j = L[ctx.eu].astype(np.int64) * N_I + ctx.ei
                p_cm = np.where(n[j] > 0, s[j] / np.maximum(n[j], 1), im[ctx.ei])
                fb_cm = 100 * float((n[j] == 0).mean())
                r = full_metrics(ctx, np.clip(p_cm, 1, 5), fb_cm, [np.arange(1)])
                r.pop("havuz_ort", None)
                rows.append({"yontem": ad, "K": K, "tahminci": "cmean",
                             "mod": mod, "maxk": int(sz.max()),
                             "mink": int(sz.min()), "bos_kume": bos,
                             "ort_kume": int(sz.mean()),
                             "std_kume": int(sz.std()), **r})
                print(f"K={K:3d} {ad:4s} {mod:9s} cmean  MAE={r['mae']:.4f} "
                      f"NDCG={r['ndcg10']:.4f} | kume: max={sz.max()} "
                      f"min={sz.min()} bos={bos} ort={int(sz.mean())}±{int(sz.std())}",
                      flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)
            # --- bizim tahminci (yalniz repairLI) ---
            _, mm = m.evaluate(ctx, X, cent, K)
            rows.append({"yontem": ad, "K": K, "tahminci": "kNN+MF",
                         "mod": "repairLI", "maxk": mm["maxk"], **mm})
            print(f"K={K:3d} {ad:4s} repairLI  kNN+MF MAE={mm['mae']:.4f} "
                  f"NDCG={mm['ndcg10']:.4f} havuz={mm['havuz']}", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
