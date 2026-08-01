"""Ablation hücreleri için KÜME KALİTESİ metrikleri (MAE değil).

WNMF50 W uzayında her (hücre, algo) için: silhouette↑, Davies-Bouldin↓,
Calinski-Harabasz↑, WCSS↓, küme-boyut std. Amaç: AVOA/HHO'nun B0'ı GERÇEKTEN ve
anlamlı geçtiği ekseni veriyle bulmak (savunulacak "temel").

Çıktı: results/meta_ablation/ablation_cluster_quality.csv + konsol özet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))
from run_k_meta_ablation import catalog, assign_dir  # noqa: E402

OUT = REPO / "results" / "meta_ablation" / "ablation_cluster_quality.csv"


def _load_W(d: Path):
    for name in ("user_features.npy", "wnmf_user_vectors.npy"):
        f = d / name
        if f.is_file():
            return np.load(f)
    return None


def _wcss(W, a):
    a = a.astype(int)
    total = 0.0
    for c in np.unique(a):
        members = W[a == c]
        if len(members) == 0:
            continue
        centroid = members.mean(axis=0)
        total += float(((members - centroid) ** 2).sum())
    return total


def metrics(W, a):
    a = a.astype(int)
    k = len(np.unique(a))
    out = {"k_active": int(k), "wcss": round(_wcss(W, a), 2),
           "size_std": round(float(np.std(np.bincount(a))), 1)}
    if k >= 2 and k < len(a):
        out["silhouette"] = round(float(silhouette_score(W, a, metric="euclidean")), 4)
        out["davies_bouldin"] = round(float(davies_bouldin_score(W, a)), 4)
        out["calinski_harabasz"] = round(float(calinski_harabasz_score(W, a)), 1)
    else:
        out["silhouette"] = out["davies_bouldin"] = out["calinski_harabasz"] = float("nan")
    return out


def main():
    cat = catalog()
    rows = []
    for cid, cell in cat.items():
        for algo in cell.algos:
            d = assign_dir(algo, cell)
            a_f = d / "assignments.npy"
            if not a_f.is_file():
                continue
            W = _load_W(d)
            if W is None:
                continue
            a = np.load(a_f)
            m = metrics(W, a)
            m.update({"cell": cid, "algo": algo, "k": cell.k, "init": cell.init,
                      "fitness": cell.fitness, "kmref": cell.kmref,
                      "b0_n_init": cell.b0_n_init})
            rows.append(m)
    df = pd.DataFrame(rows)
    cols = ["cell", "algo", "k", "init", "fitness", "kmref",
            "silhouette", "davies_bouldin", "calinski_harabasz", "wcss",
            "size_std", "k_active"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(OUT, index=False)
    print(f"-> {OUT}\n")

    # Hücre bazında B0 vs en iyi meta — hangi metrikte meta kazanıyor?
    print("Küme kalitesi (sil↑ DB↓ CH↑ WCSS↓). 'meta-kazanç' = en iyi meta B0'a kıyasla.")
    print(f"{'cell':20s} {'algo':10s} {'sil':>7} {'DB':>7} {'CH':>8} {'WCSS':>9} {'szstd':>6}")
    for cid in cat:
        g = df[df.cell == cid]
        if g.empty:
            continue
        for _, r in g.iterrows():
            print(f"{cid:20s} {r['algo']:10s} {r['silhouette']:>7} {r['davies_bouldin']:>7} "
                  f"{r['calinski_harabasz']:>8} {r['wcss']:>9} {r['size_std']:>6}")
        print("")

    # Özet: kaç hücrede meta B0'ı geçti (her metrikte)
    print("=== Meta (AVOA/HHO) B0'ı geçti mi? (hücre sayısı) ===")
    better = {"silhouette": 0, "davies_bouldin": 0, "calinski_harabasz": 0, "wcss": 0}
    total = 0
    for cid in cat:
        g = df[df.cell == cid]
        b0 = g[g.algo == "B0_KMEANS"]
        meta = g[g.algo.isin(["B_AVOA", "B1_HHO"])]
        if b0.empty or meta.empty:
            continue
        total += 1
        b0 = b0.iloc[0]
        better["silhouette"] += int(meta["silhouette"].max() > b0["silhouette"])
        better["davies_bouldin"] += int(meta["davies_bouldin"].min() < b0["davies_bouldin"])
        better["calinski_harabasz"] += int(meta["calinski_harabasz"].max() > b0["calinski_harabasz"])
        better["wcss"] += int(meta["wcss"].min() < b0["wcss"])
    for k, v in better.items():
        print(f"  {k:18s}: {v}/{total} hücrede meta daha iyi")


if __name__ == "__main__":
    main()
