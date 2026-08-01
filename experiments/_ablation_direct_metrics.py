"""Ablation hücreleri için DOĞRUDAN calc_avg_rating MAE/RMSE + pairwise ARI.

wnmf_experiment'in --assign-suffix klasör-eşleme gariplikleri (_ninit, _kmrefcap)
olmadan, assignments.npy'den Thakrar Alg.6'yı birebir hesaplar. base B0 üzerinde
MAE=0.8289/RMSE=1.0473 ile doğrulandı (wnmf_experiment ile aynı).

Çıktı: results/meta_ablation/ablation_direct.csv + konsol özet.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "wnmf"))

from run_k_meta_ablation import catalog, assign_dir, FOLD  # noqa: E402
from wnmf_utils import load_ratings_100k  # noqa: E402

OUT = REPO / "results" / "meta_ablation" / "ablation_direct.csv"


def _load_data():
    base = glob.glob(str(REPO / "data" / "**" / "u1.base"), recursive=True)[0]
    test = base.replace("u1.base", "u1.test")
    train, te = load_ratings_100k(base, test, FOLD)
    return train, te


def _build(train, te):
    n_users = int(max(train[:, 0].max(), te[:, 0].max())) + 1
    n_items = int(max(train[:, 1].max(), te[:, 1].max())) + 1
    gmean = float(train[:, 2].mean())
    usum = np.zeros(n_users); ucnt = np.zeros(n_users)
    np.add.at(usum, train[:, 0].astype(int), train[:, 2])
    np.add.at(ucnt, train[:, 0].astype(int), 1.0)
    umean = np.where(ucnt > 0, usum / np.maximum(ucnt, 1), gmean)
    return n_users, n_items, gmean, ucnt, umean


def calc_avg(a, train, te, n_items, gmean, ucnt, umean):
    """Thakrar Alg.6: küme-içi sert ortalama → kullanıcı ort. → global ort, clip[1,5]."""
    a = a.astype(int)
    K = a.max() + 1
    csum = np.zeros((K, n_items)); ccnt = np.zeros((K, n_items))
    cu = a[train[:, 0].astype(int)]
    np.add.at(csum, (cu, train[:, 1].astype(int)), train[:, 2])
    np.add.at(ccnt, (cu, train[:, 1].astype(int)), 1.0)
    pr = np.empty(len(te))
    for n, (u, i, r) in enumerate(te):
        u = int(u); i = int(i); cid = a[u]
        if ccnt[cid, i] > 0:
            pr[n] = csum[cid, i] / ccnt[cid, i]
        elif ucnt[u] > 0:
            pr[n] = umean[u]
        else:
            pr[n] = gmean
    pr = np.clip(pr, 1.0, 5.0)
    err = te[:, 2] - pr
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))


def main():
    train, te = _load_data()
    n_users, n_items, gmean, ucnt, umean = _build(train, te)
    cat = catalog()
    rows = []
    labels_by_cell = {}
    for cid, cell in cat.items():
        labels_by_cell[cid] = {}
        for algo in cell.algos:
            f = assign_dir(algo, cell) / "assignments.npy"
            if not f.is_file():
                continue
            a = np.load(f)
            labels_by_cell[cid][algo] = a
            mae, rmse = calc_avg(a, train, te, n_items, gmean, ucnt, umean)
            sizes = np.bincount(a.astype(int))
            rows.append({
                "cell": cid, "algo": algo, "k": cell.k, "init": cell.init,
                "fitness": cell.fitness, "kmref": cell.kmref,
                "b0_n_init": cell.b0_n_init, "mae": round(mae, 4),
                "rmse": round(rmse, 4), "size_min": int(sizes.min()),
                "size_max": int(sizes.max()), "k_active": int((sizes > 0).sum()),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"-> {OUT}\n")

    # Hücre içi MAE spread + B0'a göre meta kazanç
    print(f"{'cell':20s} {'k':>2} {'init':6s} {'fit':7s} {'kmref':9s} "
          f"{'B0':>7} {'AVOA':>7} {'HHO':>7} {'spread':>7} {'bestd%':>8}")
    for cid in cat:
        g = df[df.cell == cid]
        if g.empty:
            continue
        m = {r["algo"]: r["mae"] for _, r in g.iterrows()}
        b0 = m.get("B0_KMEANS"); av = m.get("B_AVOA"); hh = m.get("B1_HHO")
        vals = [v for v in (b0, av, hh) if v is not None]
        spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
        cfg = cat[cid]
        delta = ""
        if b0 and (av or hh):
            best_meta = min(v for v in (av, hh) if v is not None)
            delta = f"{(best_meta - b0) / b0 * 100:+.2f}"
        print(f"{cid:20s} {cfg.k:>2} {cfg.init:6s} {cfg.fitness:7s} {cfg.kmref:9s} "
              f"{(b0 or float('nan')):7.4f} {(av or float('nan')):7.4f} "
              f"{(hh or float('nan')):7.4f} {spread:7.4f} {delta:>8}")

    # Pairwise ARI
    print("\nPairwise ARI (algoritma atamaları arası benzerlik; düşük = daha farklı):")
    for cid, labs in labels_by_cell.items():
        algos = list(labs)
        pw = []
        for i in range(len(algos)):
            for j in range(i + 1, len(algos)):
                ari = adjusted_rand_score(labs[algos[i]], labs[algos[j]])
                pw.append(f"{algos[i][:5]}~{algos[j][:5]}={ari:.2f}")
        if pw:
            print(f"  {cid:20s} {'  '.join(pw)}")


if __name__ == "__main__":
    main()
