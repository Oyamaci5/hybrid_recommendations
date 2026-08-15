"""
E6 kontrolu — ozellik uzayi kararindan once kumelenebilirlik testi.

Soru: "Meta-sezgisel secimine gecmeden once hangi uzayda kume yapisi var?"
  - Hopkins istatistigi: ~0.5 = rastgele (kumelenemez), >0.7 = kumelenebilir.
  - KMeans++ K-sweep: her K icin silhouette / Davies-Bouldin / WCSS.
    Silhouette hicbir K'da > 0.05 degilse o uzayda meta secimi anlamsizdir.

Uzaylar: raw, user z-score raw, SVD-{10,20,50}, NMF-20, (opsiyonel --extra-npy WNMF).
Tum metrikler ayni uzayda, Oklid ile hesaplanir.

Kullanim (repo kokunden):
  python algo_selection_v2/space_check.py
  python algo_selection_v2/space_check.py --extra-npy path/to/wnmf_features.npy

Cikti: algo_selection_v2/results/space_check.csv + konsol onerisi
NOT: Yuksek boyutlu seyrek ham matriste Hopkins yapay olarak 1'e yaklasir;
karar silhouette ile birlikte verilmelidir.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def load_train(data_dir: Path, fold: int) -> np.ndarray:
    df = pd.read_csv(data_dir / f"u{fold}.base", sep="\t",
                     names=["u", "i", "r", "t"])
    mat = np.zeros((943, 1682))
    mat[df["u"] - 1, df["i"] - 1] = df["r"]
    return mat


def hopkins(X: np.ndarray, n: int = 200, seed: int = 42) -> float:
    """H ~ 0.5 rastgele; H -> 1 kumelenebilir."""
    rng = np.random.default_rng(seed)
    n = min(n, len(X) - 1)
    idx = rng.choice(len(X), n, replace=False)
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    # gercek noktalardan en yakin komsuya (kendisi haric)
    w = nn.kneighbors(X[idx], return_distance=True)[0][:, 1]
    # ayni sinirlar icinde uniform rastgele noktalardan en yakin gercek noktaya
    U = rng.uniform(X.min(0), X.max(0), size=(n, X.shape[1]))
    u = nn.kneighbors(U, n_neighbors=1, return_distance=True)[0][:, 0]
    return float(u.sum() / (u.sum() + w.sum()))


def build_spaces(train: np.ndarray, extra_npy: str | None) -> dict[str, np.ndarray]:
    spaces: dict[str, np.ndarray] = {"raw": train.copy()}
    z = train.copy()
    mask = z != 0
    mu = np.where(mask.sum(1) > 0, z.sum(1) / np.maximum(mask.sum(1), 1), 0)
    sd = np.array([z[i, mask[i]].std() if mask[i].sum() > 1 else 1.0
                   for i in range(len(z))])
    sd[sd < 1e-9] = 1.0
    z[mask] = ((z - mu[:, None]) / sd[:, None])[mask]
    spaces["raw_zscore"] = z
    for d in (10, 20, 50):
        spaces[f"svd{d}"] = TruncatedSVD(d, random_state=42).fit_transform(train)
    spaces["nmf20"] = NMF(20, init="nndsvda", max_iter=400,
                          random_state=42).fit_transform(train)
    if extra_npy:
        X = np.load(extra_npy)
        assert X.shape[0] == train.shape[0]
        spaces[f"npy_{Path(extra_npy).stem}"] = np.asarray(X, dtype=np.float64)
    return spaces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO / "data" / "ml-100k"))
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--k-list", type=int, nargs="+",
                    default=[3, 5, 7, 10, 14, 18, 24, 30])
    ap.add_argument("--extra-npy", default=None)
    ap.add_argument("--seeds", type=int, default=3,
                    help="KMeans tekrari (std icin)")
    ap.add_argument("--spaces", nargs="+", default=None,
                    help="sadece bu uzaylari kos (orn: svd20 nmf20)")
    ap.add_argument("--append", action="store_true",
                    help="space_check.csv'ye ekle (parcali kosu)")
    args = ap.parse_args()

    train = load_train(Path(args.data_dir), args.fold)
    spaces = build_spaces(train, args.extra_npy)
    if args.spaces:
        spaces = {k: v for k, v in spaces.items() if k in args.spaces}

    rows = []
    for name, X in spaces.items():
        H = hopkins(X)
        print(f"\n[{name}] shape={X.shape}  Hopkins={H:.3f}"
              + ("  (dikkat: yuksek boyutta sisirilmis olabilir)"
                 if X.shape[1] > 100 else ""))
        for K in args.k_list:
            sils, dbs, wcss = [], [], []
            for s in range(args.seeds):
                km = KMeans(K, init="k-means++", n_init=10,
                            random_state=42 + s).fit(X)
                lab = km.labels_
                if len(np.unique(lab)) < 2:
                    continue
                sils.append(silhouette_score(X, lab, metric="euclidean"))
                dbs.append(davies_bouldin_score(X, lab))
                wcss.append(km.inertia_)
            row = {"space": name, "dim": X.shape[1], "hopkins": round(H, 3),
                   "K": K,
                   "sil_mean": np.mean(sils), "sil_std": np.std(sils),
                   "db_mean": np.mean(dbs), "wcss_mean": np.mean(wcss)}
            rows.append(row)
            print(f"  K={K:>2}  sil={row['sil_mean']:.3f}±{row['sil_std']:.3f}"
                  f"  DB={row['db_mean']:.2f}  WCSS={row['wcss_mean']:.0f}")

    df = pd.DataFrame(rows)
    out = RESULTS / "space_check.csv"
    if args.append and out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
        df = df.drop_duplicates(["space", "K"], keep="last")
    df.to_csv(out, index=False)

    print("\n=== KARAR TABLOSU (uzay basina en iyi K, silhouette'e gore) ===")
    best = df.loc[df.groupby("space")["sil_mean"].idxmax()]
    best = best.sort_values("sil_mean", ascending=False)
    print(best[["space", "dim", "hopkins", "K", "sil_mean", "db_mean"]]
          .to_string(index=False))

    top = best.iloc[0]
    verdict = ("KULLANILABILIR" if top.sil_mean > 0.05 else
               "ZAYIF — meta secimine gecmeden uzay yeniden dusunulmeli")
    print(f"\n[ONERI] {top.space} (K={int(top.K)}, sil={top.sil_mean:.3f}) -> {verdict}")
    print("Not: WNMF uzayinizi da kiyasa katmak icin --extra-npy verin.")


if __name__ == "__main__":
    main()
