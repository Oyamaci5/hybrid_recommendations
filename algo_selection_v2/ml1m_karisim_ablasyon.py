"""
ML-1M KARISIM ABLASYONU — hangi bilesenler birlikte?

Bilesenler: cmean (kume-film ort.), bias (kull.ort + kume sapmasi),
            cknn (kume-ici kNN), cmf (kume-basina ALS-MF)

Denenen kombinasyonlar (agirliklar ic-dogrulamada secilir):
  tekli    : cmean | bias | cknn | cmf
  ikili    : cmf+cknn | cmf+cmean | cmf+bias | cknn+cmean
  uclu     : cmf+cknn+bias | cmf+cknn+cmean
  sabit    : cmf+cknn duz ortalama (0.5/0.5, ayarsiz referans)

Iki atama modu: repairLI ve repairSIZ.
Kullanim: python algo_selection_v2/ml1m_karisim_ablasyon.py --k 40
Cikti: results/ml1m_karisim.csv
"""
from __future__ import annotations

import argparse
import itertools
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
from ctx_ml1m import N_I, Ctx1M  # noqa: E402
from ml1m_tahminci2 import free_assign  # noqa: E402
from pred_v2 import full_metrics  # noqa: E402

AGIRLIK = np.arange(0, 1.01, 0.1)


def bilesenler(ctx, X, cent, K, mod):
    L, near = (free_assign(X, cent) if mod == "repairSIZ"
               else m.fast_repair(X, cent))
    flat = L[ctx.iu].astype(np.int64) * N_I + ctx.ii
    s = np.bincount(flat, weights=ctx.ir, minlength=K * N_I)
    n = np.bincount(flat, minlength=K * N_I)
    sd = np.bincount(flat, weights=ctx.ir - ctx.um[ctx.iu], minlength=K * N_I)
    im = np.bincount(ctx.ii, weights=ctx.ir, minlength=N_I) / np.maximum(
        np.bincount(ctx.ii, minlength=N_I), 1)

    def cmean(us, is_):
        j = L[us].astype(np.int64) * N_I + is_
        return np.where(n[j] > 0, s[j] / np.maximum(n[j], 1), im[is_])

    def bias(us, is_):
        a = near[us, 0].astype(np.int64) * N_I + is_
        b = near[us, 1].astype(np.int64) * N_I + is_
        den = n[a] + n[b]
        dev = np.where(den > 0, (sd[a] + sd[b]) / np.maximum(den, 1),
                       ctx.dev_g[is_])
        return ctx.um[us] + dev

    cm.K = K
    models = cm.cluster_mf_models(ctx, L)
    out = {}
    for et, (us, is_) in (("val", (ctx.vu, ctx.vi)), ("test", (ctx.eu, ctx.ei))):
        b_ = bias(us, is_)
        k_ = m.knn_pool(ctx, L, near, us, is_)
        out[et] = {"cmean": cmean(us, is_), "bias": b_,
                   "cknn": np.where(np.isnan(k_), b_, np.nan_to_num(k_)),
                   "cmf": cm.cmf_predict(models, L, us, is_, b_)}
    return out, 100 * float(np.isnan(m.knn_pool(ctx, L, near, ctx.eu[:1],
                                                ctx.ei[:1])).mean())


def en_iyi_agirlik(parc_v, vr, adlar):
    """Izgara: 2 bilesende w, 3 bilesende (w1,w2)."""
    best = (None, np.inf)
    if len(adlar) == 1:
        return (1.0,), float(np.abs(np.clip(parc_v[adlar[0]], 1, 5) - vr).mean())
    if len(adlar) == 2:
        for w in AGIRLIK:
            p = w * parc_v[adlar[0]] + (1 - w) * parc_v[adlar[1]]
            e = float(np.abs(np.clip(p, 1, 5) - vr).mean())
            if e < best[1]:
                best = ((round(w, 1), round(1 - w, 1)), e)
        return best
    for w1 in AGIRLIK:
        for w2 in AGIRLIK:
            if w1 + w2 > 1:
                continue
            p = (w1 * parc_v[adlar[0]] + w2 * parc_v[adlar[1]]
                 + (1 - w1 - w2) * parc_v[adlar[2]])
            e = float(np.abs(np.clip(p, 1, 5) - vr).mean())
            if e < best[1]:
                best = ((round(w1, 1), round(w2, 1),
                         round(1 - w1 - w2, 1)), e)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data" / "ml-1m"))
    args = ap.parse_args()
    ctx = Ctx1M(Path(args.data_dir), args.fold)
    X = m.build_space(ctx, None, "nmf")
    K = args.k
    km = KMeans(K, init="k-means++", n_init=3, random_state=args.seed).fit(X)
    cent = km.cluster_centers_

    kombin = [("cmean",), ("bias",), ("cknn",), ("cmf",),
              ("cmf", "cknn"), ("cmf", "cmean"), ("cmf", "bias"),
              ("cknn", "cmean"),
              ("cmf", "cknn", "bias"), ("cmf", "cknn", "cmean")]
    rows = []
    for mod in ("repairLI", "repairSIZ"):
        parc, fbp = bilesenler(ctx, X, cent, K, mod)
        for adlar in kombin:
            w, val = en_iyi_agirlik(parc["val"], ctx.vr, list(adlar))
            pt = sum(wi * parc["test"][a] for wi, a in zip(w, adlar))
            pt = np.clip(pt, 1, 5)
            r = full_metrics(ctx, pt, fbp, [np.arange(1)])
            r.pop("havuz_ort", None)
            rows.append({"mod": mod, "kombin": "+".join(adlar),
                         "agirlik": str(w), "val_mae": round(val, 4), **r})
            print(f"{mod:9s} {'+'.join(adlar):18s} w={str(w):18s} "
                  f"MAE={r['mae']:.4f} RMSE={r['rmse']:.4f} "
                  f"NDCG={r['ndcg10']:.4f}", flush=True)
        # ayarsiz duz ortalama referansi
        pt = np.clip(0.5 * parc["test"]["cmf"] + 0.5 * parc["test"]["cknn"], 1, 5)
        r = full_metrics(ctx, pt, fbp, [np.arange(1)])
        r.pop("havuz_ort", None)
        rows.append({"mod": mod, "kombin": "cmf+cknn (duz 0.5)",
                     "agirlik": "(0.5,0.5)", "val_mae": np.nan, **r})
        print(f"{mod:9s} {'cmf+cknn duz 0.5':18s} MAE={r['mae']:.4f} "
              f"NDCG={r['ndcg10']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(RESULTS / "ml1m_karisim.csv", index=False)


if __name__ == "__main__":
    main()
