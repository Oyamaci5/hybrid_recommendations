"""Temsil taraması: hangi kullanıcı-uzayı GERÇEKTEN kümelenebilir? (silhouette↑)

WNMF50'de silhouette≈0 → belirgin küme yok. Burada farklı temsilleri B0 KMeans ile
deneyip silhouette/DB/CH'yi karşılaştırırız. Amaç: silhouette'i anlamlı (>0) yapan
uzayı bulmak; orada hem küme kalitesi hem algoritma farkları büyür.

Temsiller:
  - WNMF dim 10/20/30/50  (results/meta_ablation/wnmf_u_d*/ml100k_U.npy, wnmf_u/)
  - PCA(ham profil)       (kullanıcı-merkezli, 0-dolgu) -> 10/20/50 bileşen
  - raw                   (ham 943x1682 profil, 0-dolgu)

Çıktı: results/meta_ablation/representation_sweep.csv + konsol.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "wnmf"))
from wnmf_utils import load_ratings_100k  # noqa: E402

OUT = REPO / "results" / "meta_ablation" / "representation_sweep.csv"
KS = [4, 6, 8, 10, 14]


def build_raw(train, n_users, n_items, center=True):
    R = np.zeros((n_users, n_items), dtype=np.float32)
    R[train[:, 0].astype(int), train[:, 1].astype(int)] = train[:, 2]
    if center:
        # kullanıcı ortalamasını yalnız gözlenen hücrelerden çıkar
        mask = R > 0
        usum = R.sum(1); ucnt = mask.sum(1)
        umean = np.where(ucnt > 0, usum / np.maximum(ucnt, 1), 0)
        R = np.where(mask, R - umean[:, None], 0.0).astype(np.float32)
    return R


def reps():
    base = glob.glob(str(REPO / "data" / "**" / "u1.base"), recursive=True)[0]
    train, _ = load_ratings_100k(base, base.replace("u1.base", "u1.test"), 1)
    n_users = int(train[:, 0].max()) + 1
    n_items = int(train[:, 1].max()) + 1
    out = {}
    # WNMF dumps
    for d, path in [
        (10, REPO / "results/meta_ablation/wnmf_u_d10/ml100k_U.npy"),
        (20, REPO / "results/meta_ablation/wnmf_u_d20/ml100k_U.npy"),
        (30, REPO / "results/meta_ablation/wnmf_u_d30/ml100k_U.npy"),
        (50, REPO / "results/meta_ablation/wnmf_u/ml100k_U.npy"),
    ]:
        if Path(path).is_file():
            out[f"wnmf{d}"] = np.load(path)
    # PCA(raw centered)
    Rc = build_raw(train, n_users, n_items, center=True)
    for nc in (10, 20, 50):
        out[f"pca{nc}"] = PCA(n_components=nc, random_state=42).fit_transform(Rc)
    out["raw_centered"] = Rc
    return out


def main():
    rows = []
    for name, X in reps().items():
        for k in KS:
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
            a = km.labels_
            if len(np.unique(a)) < 2:
                continue
            rows.append({
                "rep": name, "dim": X.shape[1], "k": k,
                "silhouette": round(float(silhouette_score(X, a)), 4),
                "davies_bouldin": round(float(davies_bouldin_score(X, a)), 3),
                "calinski_harabasz": round(float(calinski_harabasz_score(X, a)), 1),
                "size_std": round(float(np.std(np.bincount(a))), 1),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"-> {OUT}\n")
    # En iyi silhouette'ler
    print("En yüksek silhouette (gerçek küme yapısı):")
    print(df.sort_values("silhouette", ascending=False).head(12).to_string(index=False))
    print("\nTemsil başına en iyi silhouette:")
    best = df.loc[df.groupby("rep")["silhouette"].idxmax()].sort_values(
        "silhouette", ascending=False)
    print(best[["rep", "k", "silhouette", "davies_bouldin", "calinski_harabasz"]].to_string(index=False))


if __name__ == "__main__":
    main()
