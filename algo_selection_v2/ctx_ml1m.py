"""
ML-1M BAGLAM SINIFI — Ctx ile ayni arayuz, olcek icin optimize.

Farklar (ML-100K'ya gore):
  - 6040 kullanici x 3952 film, ~1M puan (resmi fold YOK -> rastgele %90/10, tekrar=fold)
  - Benzerlik matrisi float32 (6040^2 x 4B = 146 MB; float64 olsaydi 292 MB)
  - raters listesi liste-of-array (3952 film) — bellek dostu
  - Tur profili movies.dat'tan (18 tur, ML-100K'daki 19'dan 'unknown' yok)

Arayuz (tabloB_plus/cluster_mf/pred_v2 bunlari bekliyor):
  ctx.R, ctx.um, ctx.dev_g, ctx.gmean, ctx.X(disaridan), ctx.S,
  ctx.iu/ii/ir (ic-train), ctx.vu/vi/vr (ic-val), ctx.eu/ei/er (test),
  ctx.raters, ctx.by_user

Kullanim:
  from ctx_ml1m import Ctx1M, genre_profile_1m
  ctx = Ctx1M(Path('data/ml-1m'), fold=1)      # fold = tekrar no (1..5)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

N_U, N_I = 6040, 3952
GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
          "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


class Ctx1M:
    """ML-1M baglami. fold = rastgele %90/10 bolme tekrari (1..5)."""

    def __init__(self, data_dir: Path, fold: int = 1, dim: int = 20,
                 val_pay: float = 0.1, dtype=np.float32):
        df = pd.read_csv(data_dir / "ratings.dat", sep="::", engine="python",
                         names=["u", "i", "r", "t"], encoding="latin-1")
        assert df.r.between(1, 5).all(), "Rating 1-5 disinda!"
        assert not df.duplicated(["u", "i"]).any(), "Duplicate (u,i)!"
        u = df.u.values - 1; i = df.i.values - 1; r = df.r.values.astype(np.float64)

        # --- test bolmesi: fold'a gore rastgele %10 (seed = fold) ---
        rng = np.random.default_rng(1000 + fold)
        te_mask = np.zeros(len(r), bool)
        te_mask[rng.choice(len(r), len(r) // 10, replace=False)] = True
        self.eu, self.ei, self.er = u[te_mask], i[te_mask], r[te_mask]
        tu, ti, tr = u[~te_mask], i[~te_mask], r[~te_mask]

        # --- ic-val (%10 of train), seed sabit (7) — 100K ile ayni mantik ---
        rng2 = np.random.default_rng(7)
        v_mask = np.zeros(len(tr), bool)
        v_mask[rng2.choice(len(tr), int(len(tr) * val_pay), replace=False)] = True
        self.iu, self.ii, self.ir = tu[~v_mask], ti[~v_mask], tr[~v_mask]
        self.vu, self.vi, self.vr = tu[v_mask], ti[v_mask], tr[v_mask]

        # --- ic-train matrisi ve istatistikler ---
        self.R = np.zeros((N_U, N_I), dtype=dtype)
        self.R[self.iu, self.ii] = self.ir
        rated = self.R > 0
        self.gmean = float(self.ir.mean())
        cnt_u = rated.sum(1)
        self.um = np.where(cnt_u > 0, self.R.sum(1) / np.maximum(cnt_u, 1),
                           self.gmean).astype(np.float64)
        self.dev_g = np.zeros(N_I)
        np.add.at(self.dev_g, self.ii, self.ir - self.um[self.iu])
        self.dev_g /= np.maximum(np.bincount(self.ii, minlength=N_I), 1)

        # --- benzerlik (ortalama-merkezli cosine), float32 ---
        Rc = np.where(rated, self.R - self.um[:, None].astype(dtype), 0).astype(dtype)
        nrm = np.linalg.norm(Rc, axis=1); nrm[nrm < 1e-9] = 1.0
        Rn = Rc / nrm[:, None]
        self.S = (Rn @ Rn.T).astype(dtype)
        np.fill_diagonal(self.S, 0.0)

        self.raters = [np.flatnonzero(rated[:, x]) for x in range(N_I)]
        self.by_user = [[] for _ in range(N_U)]
        for j, uu in enumerate(self.eu):
            self.by_user[uu].append(j)

        print(f"[ML-1M fold {fold}] train={len(self.iu)} val={len(self.vu)} "
              f"test={len(self.eu)} | sparsity={1 - len(tr)/(N_U*N_I):.4f} "
              f"| S={self.S.nbytes/1e6:.0f} MB", flush=True)

    def nmf_space(self, dim: int = 20, seed: int = 42) -> np.ndarray:
        """NMF ozellik uzayi (ic-train'den)."""
        X = NMF(dim, init="nndsvda", max_iter=200,
                random_state=seed).fit_transform(self.R.astype(np.float64))
        return np.ascontiguousarray(X)


def genre_profile_1m(ctx, data_dir: Path) -> np.ndarray:
    """Kullanici tur profili: puanla agirlikli 18-boyutlu tur dagilimi."""
    G_item = np.zeros((N_I, len(GENRES)))
    idx = {g: k for k, g in enumerate(GENRES)}
    with open(data_dir / "movies.dat", encoding="latin-1") as f:
        for line in f:
            p = line.rstrip("\n").split("::")
            if len(p) < 3:
                continue
            mid = int(p[0]) - 1
            for g in p[2].split("|"):
                if g in idx:
                    G_item[mid, idx[g]] = 1
    W = np.zeros((N_U, N_I))
    W[ctx.iu, ctx.ii] = ctx.ir
    gu = W @ G_item
    gu /= np.maximum(gu.sum(1, keepdims=True), 1e-9)
    return gu


if __name__ == "__main__":
    import sys
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "data/ml-1m")
    c = Ctx1M(d, fold=1)
    X = c.nmf_space(20)
    print("NMF uzayi:", X.shape, "| kullanici basina ort. puan:",
          int((c.R > 0).sum(1).mean()))
