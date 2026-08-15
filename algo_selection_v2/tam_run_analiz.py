"""
TAM RUN ANALIZI — ozet + Friedman/Holm + kullanici-bazli Wilcoxon + grafikler.

Kullanim (tam_run bittikten sonra):
  python algo_selection_v2/tam_run_analiz.py --table B
  python algo_selection_v2/tam_run_analiz.py --table A
Cikti: konsol ozeti + results/plots/tamrun_{A,B}_*.png + results/tamrun_{A,B}_ozet.csv
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"


def holm(pairs):
    pairs = sorted(pairs, key=lambda x: x[1])
    k = len(pairs)
    out = []
    for i, (name, p, d) in enumerate(pairs):
        out.append((name, min(p * (k - i), 1.0), d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=None,
                    help="A veya B (tam_run_*.csv) — ya da --file kullanin")
    ap.add_argument("--file", default=None,
                    help="dogrudan CSV adi, orn: tabloB_plus_K6.csv")
    args = ap.parse_args()
    if args.file:
        path = RESULTS / args.file
        args.table = Path(args.file).stem
        upath_override = RESULTS / (Path(args.file).stem + "_userr.npz")
    else:
        path = RESULTS / f"tam_run_{args.table}.csv"
        upath_override = None
    df = pd.read_csv(path)
    df = df[df.yontem != "KNN_ALL"]

    # --- 1) Ozet ---
    g = df.groupby("yontem").agg(
        mae=("mae", "mean"), mae_std=("mae", "std"),
        rmse=("rmse", "mean"), ndcg=("ndcg10", "mean"),
        ndcg_std=("ndcg10", "std"), prec=("prec10", "mean"),
        rec=("rec10", "mean"), havuz=("havuz", "mean"),
        n=("mae", "count")).round(4).sort_values("mae")
    g["f1_10"] = (2 * g.prec * g.rec / (g.prec + g.rec)).round(4)
    if "maxk" in df:
        g = g.join(df.groupby("yontem")["maxk"].mean().round(0))
    print("=== OZET (fold x seed ortalamasi) ===")
    print(g.to_string())
    g.to_csv(RESULTS / f"tamrun_{args.table}_ozet.csv")

    # --- 2) Friedman (hucre = fold x seed) + Holm'lu ikili Wilcoxon (B0'a karsi) ---
    piv = df.pivot_table(index=["fold", "seed"], columns="yontem", values="mae")
    piv = piv.dropna()
    if piv.shape[1] >= 3:
        st, p = friedmanchisquare(*[piv[c] for c in piv.columns])
        print(f"\nFriedman (MAE, {len(piv)} hucre x {piv.shape[1]} yontem): "
              f"chi2={st:.1f}, p={p:.2e}")
    else:
        print(f"\n(2 yontem -> Friedman atlandi; dogrudan Wilcoxon, "
              f"{len(piv)} hucre)")
    baz = next((c for c in df.yontem.unique() if str(c).startswith("B0")), None)
    for metrik, iyi_yon in (("mae", "kucuk"), ("ndcg10", "buyuk")):
        pv2 = df.pivot_table(index=["fold", "seed"], columns="yontem",
                             values=metrik).dropna()
        if baz is None or baz not in pv2:
            continue
        pairs = []
        for m in pv2.columns:
            if m == baz:
                continue
            s, p_ = wilcoxon(pv2[m], pv2[baz])
            pairs.append((m, p_, float((pv2[m] - pv2[baz]).mean())))
        print(f"\n{baz}'a karsi Holm duzeltmeli Wilcoxon — {metrik.upper()} "
              f"(fold x seed hucreleri):")
        for name, ph, d in holm(pairs):
            meta_iyi = (d < 0) if iyi_yon == "kucuk" else (d > 0)
            yon = "META IYI" if meta_iyi else "B0 iyi"
            print(f"  {name:8s} fark={d:+.4f}  holm_p={ph:.4f}  "
                  f"{'ANLAMLI' if ph < 0.05 else 'fark yok'} ({yon})")

    # --- 3) Kullanici-bazli Wilcoxon (once fold/seed uzerinden kullanici ort.) ---
    upath = upath_override or (RESULTS / f"tam_run_userr_{args.table}.npz")
    if upath.exists():
        z = np.load(upath)
        met = {}
        for key in z.files:
            y = key.split("|")[0]
            met.setdefault(y, []).append(z[key])
        umean = {y: np.nanmean(np.vstack(v), axis=0) for y, v in met.items()}
        ubaz = next((c for c in umean if str(c).startswith("B0")), None)
        if ubaz:
            print(f"\nKullanici-bazli Wilcoxon (n~943, {ubaz}'a karsi):")
            pairs = []
            for y, v in umean.items():
                if y in (ubaz, "KNN_ALL"):
                    continue
                mask = np.isfinite(v) & np.isfinite(umean[ubaz])
                s, pv = wilcoxon(v[mask], umean[ubaz][mask])
                pairs.append((y, pv, float((v[mask] - umean[ubaz][mask]).mean())))
            for name, ph, d in holm(pairs):
                yon = "META IYI" if d < 0 else "B0 iyi"
                print(f"  {name:8s} fark={d:+.4f}  holm_p={ph:.4g}  "
                      f"{'ANLAMLI' if ph < 0.05 else 'fark yok'} ({yon})")

    # --- 4) Grafikler ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS.mkdir(exist_ok=True)
    order = g.index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    df.boxplot(column="mae", by="yontem", ax=axes[0], grid=False)
    axes[0].set_title("MAE dagilimi (fold x seed)"); axes[0].set_xlabel("")
    df.boxplot(column="ndcg10", by="yontem", ax=axes[1], grid=False)
    axes[1].set_title("NDCG@10 dagilimi"); axes[1].set_xlabel("")
    for y, gg in df.groupby("yontem"):
        axes[2].scatter(gg.havuz, gg.mae, label=y, s=14, alpha=.6)
    axes[2].set_xlabel("havuz"); axes[2].set_ylabel("MAE")
    axes[2].set_title("Dogruluk-maliyet"); axes[2].legend(fontsize=7)
    fig.suptitle(f"Tam run — Tablo {args.table}")
    fig.tight_layout()
    fp = PLOTS / f"tamrun_{args.table}.png"
    fig.savefig(fp, dpi=150)
    print(f"\nGrafik -> {fp}")


if __name__ == "__main__":
    main()
