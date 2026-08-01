"""Denge (balance) manzarası: B0 vs AVOA vs HHO + PCA-space K-means.

Reframe: katkı 'daha ayrışmış' değil 'daha DENGELİ / dejenere dev-kümeden kaçan'.
Bu script kimin nerede dengeli/çökük olduğunu metriklerle gösterir:
  - max_share : en büyük küme / N  (1'e yakın = çöküş)
  - entropy   : normalize Shannon (1 = tam dengeli, 0 = tek küme)
  - gini      : küme boyutu Gini (0 = dengeli, 1 = dengesiz)
  - n_degen   : <2% N olan küme sayısı (singleton/uç)
Çıktı: results/meta_ablation/ablation_balance.csv + konsol.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "wnmf"))
from run_k_meta_ablation import catalog, assign_dir  # noqa: E402
from wnmf_utils import load_ratings_100k  # noqa: E402

OUT = REPO / "results" / "meta_ablation" / "ablation_balance.csv"


def balance(a):
    a = a.astype(int)
    n = len(a)
    sizes = np.bincount(a)
    sizes = sizes[sizes > 0]
    K = len(sizes)
    p = sizes / n
    entropy = float(-(p * np.log(p)).sum() / np.log(K)) if K > 1 else 0.0
    s = np.sort(sizes)
    gini = float((2 * np.arange(1, K + 1) - K - 1).dot(s) / (K * s.sum())) if K > 1 else 0.0
    return {
        "k_active": int(K),
        "max_share": round(float(sizes.max() / n), 3),
        "min_size": int(sizes.min()),
        "entropy": round(entropy, 3),
        "gini": round(gini, 3),
        "n_degen": int((sizes < 0.02 * n).sum()),
    }


def main():
    rows = []
    # 1) Ablation hücreleri (WNMF50)
    cat = catalog()
    for cid, cell in cat.items():
        for algo in cell.algos:
            f = assign_dir(algo, cell) / "assignments.npy"
            if not f.is_file():
                continue
            b = balance(np.load(f))
            b.update({"source": "wnmf50", "cell": cid, "algo": algo,
                      "k": cell.k, "kmref": cell.kmref})
            rows.append(b)
    # 2) PCA-space K-means (çöküş kanıtı)
    base = glob.glob(str(REPO / "data" / "**" / "u1.base"), recursive=True)[0]
    train, _ = load_ratings_100k(base, base.replace("u1.base", "u1.test"), 1)
    nu = int(train[:, 0].max()) + 1; ni = int(train[:, 1].max()) + 1
    R = np.zeros((nu, ni), np.float32)
    R[train[:, 0].astype(int), train[:, 1].astype(int)] = train[:, 2]
    mask = R > 0
    um = np.where(mask.sum(1) > 0, R.sum(1) / np.maximum(mask.sum(1), 1), 0)
    Rc = np.where(mask, R - um[:, None], 0).astype(np.float32)
    for nc in (10, 20):
        X = PCA(n_components=nc, random_state=42).fit_transform(Rc)
        for k in (6, 10):
            a = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).labels_
            b = balance(a)
            b.update({"source": f"pca{nc}", "cell": f"pca{nc}_k{k}",
                      "algo": "B0_KMEANS", "k": k, "kmref": "-"})
            rows.append(b)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"-> {OUT}\n")

    # WNMF50: B0 vs meta denge karşılaştırması (overwrite hücreleri = paper-faithful)
    print("WNMF50 — denge (entropy↑ dengeli, max_share↓ iyi, n_degen=dejenere küme):")
    print(f"{'cell':18s} {'algo':10s} {'maxshare':>8} {'minsz':>5} {'entropy':>7} {'gini':>5} {'degen':>5}")
    for cid in cat:
        g = df[(df.source == 'wnmf50') & (df.cell == cid)]
        for _, r in g.iterrows():
            print(f"{cid:18s} {r['algo']:10s} {r['max_share']:>8} {r['min_size']:>5} "
                  f"{r['entropy']:>7} {r['gini']:>5} {r['n_degen']:>5}")
        if len(g):
            print("")

    print("PCA-space K-means (çöküş kanıtı — K-means burada dejenere):")
    for _, r in df[df.source.str.startswith('pca')].iterrows():
        print(f"  {r['cell']:12s} max_share={r['max_share']} min_size={r['min_size']} "
              f"entropy={r['entropy']} n_degen={r['n_degen']}")


if __name__ == "__main__":
    main()
