"""
SAF META karsilastirmasi — Lloyd YOK, KMeans++ YOK.

Soru: "Rafinasyon olmadan algoritmalarin kendi arama gucu nasil siralaniyor?"

Tasarim:
  - Meta sonuclari per_run.csv'den okunur (wcss_meta = kmref ONCESI ham cikti).
  - B0 baseline'lari Lloyd/++ icermez:
      B0_RAND_CENT : K rastgele veri noktasi merkez, optimizasyon yok (taban cizgisi)
      B0_RAND_SEARCH: ayni butce (epoch*pop aday) rastgele arama, en iyisi tutulur
        - adaylar veri noktalarindan secilir (140-D uniform arama umutsuz oldugu icin
          bu, rastgele aramanin GUCLU halidir -> gecilmesi anlamli esik)
  - Referans cizgisi olarak KMeans++ degeri sadece tabloya not edilir, yarisa girmez.

Adalet notu: butce epoch*pop ile esitlendi; mealpy algoritmalari epoch basina farkli
sayida degerlendirme yapabilir. Final turda butce NFE (fonksiyon degerlendirme sayisi)
ile esitlenmeli — rapora bakiniz.

Kullanim: python algo_selection_v2/pure_meta_compare.py [--budget 1500] [--seeds 42 43 44 45 46]
Cikti: results/pure_meta_summary.csv, results/pure_meta_friedman.txt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))
from recompute_scores import (build_features, cluster_metrics,  # noqa: E402
                              load_ml100k_train, sse_wcss)


def rand_cent(X, K, rng):
    idx = rng.choice(len(X), K, replace=False)
    return X[idx]


def random_search(X, K, budget, rng):
    best_w, best_c = np.inf, None
    for _ in range(budget):
        c = rand_cent(X, K, rng)
        w, _ = sse_wcss(X, c)
        if w < best_w:
            best_w, best_c = w, c
    return best_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data" / "ml-100k"))
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--budget", type=int, default=1500, help="epoch*pop esdegeri")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = ap.parse_args()

    train = load_ml100k_train(Path(args.data_dir), 1)
    X = build_features(train, "nmf", 20, None)

    # --- 1) Lloyd'suz baseline'lar ---
    rows = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        t0 = time.time()
        c = rand_cent(X, args.k, rng)
        w, lab = sse_wcss(X, c)
        rows.append({"algorithm": "B0_RAND_CENT", "seed": seed, "wcss_meta": w,
                     "time_s": round(time.time() - t0, 2),
                     **{f"{k}_meta": v for k, v in cluster_metrics(X, lab).items()}})
        t0 = time.time()
        c = random_search(X, args.k, args.budget, rng)
        w, lab = sse_wcss(X, c)
        rows.append({"algorithm": "B0_RAND_SEARCH", "seed": seed, "wcss_meta": w,
                     "time_s": round(time.time() - t0, 2),
                     **{f"{k}_meta": v for k, v in cluster_metrics(X, lab).items()}})
        print(f"seed {seed}: RAND_CENT / RAND_SEARCH tamam")
    base = pd.DataFrame(rows)

    # --- 2) Meta sonuclari (temiz cekirdek: verilen seed'ler) ---
    pr = pd.read_csv(RESULTS / "per_run.csv")
    meta = pr[(~pr.algorithm.str.startswith("B0")) & (pr.seed.isin(args.seeds))]
    keep = ["algorithm", "seed", "wcss_meta", "silhouette_meta",
            "davies_bouldin_meta", "empty_clusters_meta", "time_s"]
    meta = meta[[c for c in keep if c in meta.columns]]
    allr = pd.concat([meta, base], ignore_index=True)

    # --- 3) Ozet + rank + Friedman ---
    g = allr.groupby("algorithm").agg(
        wcss_mean=("wcss_meta", "mean"), wcss_std=("wcss_meta", "std"),
        sil_mean=("silhouette_meta", "mean"),
        db_mean=("davies_bouldin_meta", "mean"),
        empty_mean=("empty_clusters_meta", "mean"),
        time_mean=("time_s", "mean"), n=("seed", "count")).reset_index()
    piv = allr.pivot_table(index="seed", columns="algorithm", values="wcss_meta")
    n_nan = int(piv.isna().sum().sum())
    assert n_nan == 0, f"pivot'ta {n_nan} eksik hucre — seed filtresi kacak veriyor!"
    g = g.merge(piv.rank(axis=1).mean(0).rename("avg_rank"),
                left_on="algorithm", right_index=True).sort_values("avg_rank")
    g.to_csv(RESULTS / "pure_meta_summary.csv", index=False)

    stat, p = friedmanchisquare(*[piv[c].values for c in piv.columns])
    kpp = pr[pr.algorithm == "B0_KMEANS++"]
    kpp = kpp[kpp.seed.isin(args.seeds)].wcss_kmref.mean()
    txt = (f"Friedman (saf meta WCSS, {piv.shape[0]} seed x {piv.shape[1]} yontem): "
           f"chi2={stat:.1f}, p={p:.3e}\n"
           f"Referans (yarista degil): KMeans++ n_init=10 = {kpp:.2f}\n")
    (RESULTS / "pure_meta_friedman.txt").write_text(txt, encoding="utf-8")

    print("\n=== SAF META SIRALAMASI (Lloyd/++ yok) ===")
    print(g.round(3).to_string(index=False))
    print("\n" + txt)
    rs = g[g.algorithm == "B0_RAND_SEARCH"]
    if len(rs):
        beat = g[(g.avg_rank < float(rs.avg_rank.iloc[0]))
                 & (~g.algorithm.str.startswith("B0"))]
        print(f"[ESIK] Esit butceli rastgele aramayi gecen meta: {len(beat)} adet")
        print("  -> Gecemeyen bir meta, 'arama' yapmiyor demektir (eleme kriteri).")


if __name__ == "__main__":
    main()
