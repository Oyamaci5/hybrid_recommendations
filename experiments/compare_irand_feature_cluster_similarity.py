"""
_euc_irand_nogs_trainonly_rand_f1_none × svd5/10, wnmf5/10 × K=3,7,14,27,70
assignment küme benzerliği: ARI, NMI, label agreement, centroid L2 (Hungarian).

  python experiments/compare_irand_feature_cluster_similarity.py
  python experiments/compare_irand_feature_cluster_similarity.py --csv results/irand_feature_cluster_similarity.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "mealpy"))

from compare_cluster_structure import (  # noqa: E402
    _common_mask,
    _load_centroids,
    _load_labels_and_gray,
    _structure_stats,
    mean_centroid_distance,
)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ASSIGN_ROOT = os.path.join(REPO, "mealpy", "results", "assignments", "ml100k")
DEFAULT_ALGOS = ["B0_KMEANS", "B1_HHO", "HA_AVOAHGS", "IWO_HHO"]
FEATURES: Tuple[Tuple[str, int], ...] = (
    ("svd", 5),
    ("svd", 10),
    ("wnmf", 5),
    ("wnmf", 10),
)
DEFAULT_K = [3, 7, 14, 27, 70]


def suffix(feat: str, dim: int, k: int) -> str:
    return f"_euc_irand_nogs_trainonly_rand_f1_none_{feat}{int(dim)}_k{int(k)}"


def config_id(feat: str, dim: int, k: int, algo: str) -> str:
    return f"{algo}|{feat}{dim}|k{k}"


def load_one(
    algo: str, feat: str, dim: int, k: int,
) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], str]]:
    suf = suffix(feat, dim, k)
    d = os.path.join(ASSIGN_ROOT, f"{algo}{suf}")
    fp = os.path.join(d, "assignments.npy")
    if not os.path.isfile(fp):
        return None
    a, g = _load_labels_and_gray(fp)
    c = _load_centroids(fp, k)
    return a, g, c, config_id(feat, dim, k, algo)


def compare_pair(
    la: np.ndarray,
    ga: Optional[np.ndarray],
    lb: np.ndarray,
    gb: Optional[np.ndarray],
    ca: Optional[np.ndarray],
    cb: Optional[np.ndarray],
) -> dict:
    mask = _common_mask(ga, gb, len(la))
    n_ok = int(mask.sum())
    out = {"n_users": n_ok, "label_agreement": np.nan, "ari": np.nan, "nmi": np.nan, "centroid_l2_mean": np.nan}
    if n_ok < 2:
        return out
    out["label_agreement"] = float((la[mask] == lb[mask]).mean())
    out["ari"] = float(adjusted_rand_score(la[mask], lb[mask]))
    out["nmi"] = float(normalized_mutual_info_score(la[mask], lb[mask], average_method="arithmetic"))
    if ca is not None and cb is not None and ca.shape == cb.shape:
        out["centroid_l2_mean"] = mean_centroid_distance(ca, cb)
    return out


def build_index(algos: List[str]) -> Dict[str, dict]:
    idx: Dict[str, dict] = {}
    for feat, dim in FEATURES:
        for k in DEFAULT_K:
            for algo in algos:
                out = load_one(algo, feat, dim, k)
                if out is None:
                    continue
                a, g, c, cid = out
                idx[cid] = {
                    "algo": algo,
                    "feature": feat,
                    "latent_dim": dim,
                    "k": k,
                    "labels": a,
                    "gray": g,
                    "centroids": c,
                    "suffix": suffix(feat, dim, k),
                }
    return idx


def pairwise_rows(idx: Dict[str, dict], *, same_k: bool, same_feat: bool, same_algo: bool) -> List[dict]:
    rows: List[dict] = []
    keys = sorted(idx.keys())
    for ka, kb in combinations(keys, 2):
        da, db = idx[ka], idx[kb]
        if same_k and da["k"] != db["k"]:
            continue
        if same_feat and (da["feature"], da["latent_dim"]) != (db["feature"], db["latent_dim"]):
            continue
        if same_algo and da["algo"] != db["algo"]:
            continue
        m = compare_pair(
            da["labels"], da["gray"], db["labels"], db["gray"],
            da["centroids"], db["centroids"],
        )
        rows.append({
            "comparison": (
                "same_k_feat" if (same_k and same_feat and not same_algo)
                else "same_k_algo" if (same_k and same_algo and not same_feat)
                else "same_k_cross" if same_k
                else "all"
            ),
            "k": da["k"],
            "algo_a": da["algo"],
            "feature_a": f"{da['feature']}{da['latent_dim']}",
            "algo_b": db["algo"],
            "feature_b": f"{db['feature']}{db['latent_dim']}",
            "config_a": ka,
            "config_b": kb,
            **m,
        })
    return rows


def ref_vs_b0_rows(idx: Dict[str, dict], reference: str = "B0_KMEANS") -> List[dict]:
    rows: List[dict] = []
    b0_keys = [k for k, v in idx.items() if v["algo"] == reference]
    for ref_key in b0_keys:
        ref = idx[ref_key]
        for cid, other in idx.items():
            if cid == ref_key:
                continue
            if other["k"] != ref["k"] or (other["feature"], other["latent_dim"]) != (ref["feature"], ref["latent_dim"]):
                continue
            m = compare_pair(
                ref["labels"], ref["gray"], other["labels"], other["gray"],
                ref["centroids"], other["centroids"],
            )
            rows.append({
                "k": ref["k"],
                "feature": f"{ref['feature']}{ref['latent_dim']}",
                "algo": other["algo"],
                "reference": reference,
                **m,
            })
    return rows


def structure_rows(idx: Dict[str, dict]) -> List[dict]:
    rows: List[dict] = []
    for cid, v in sorted(idx.items()):
        st = _structure_stats(v["labels"], v["k"])
        rows.append({
            "config": cid,
            "algo": v["algo"],
            "feature": f"{v['feature']}{v['latent_dim']}",
            "k": v["k"],
            "n_active": int(st["n_active"]),
            "n_empty": int(st["n_empty"]),
            "size_cv": st["size_cv"],
            "gini_sizes": st["gini_sizes"],
        })
    return rows


def print_k_summary(df_feat: pd.DataFrame, k: int) -> None:
    sub = df_feat[df_feat["k"] == k]
    if sub.empty:
        return
    print(f"\n--- K={k}: aynı özellik/latent, farklı algoritma (B0 referans ARI) ---")
    ref = sub[sub["algo"] == "B0_KMEANS"]
    if not ref.empty:
        print(ref[["feature", "algo", "ari", "nmi", "label_agreement"]].to_string(index=False))
    pivot = sub.pivot_table(index="feature", columns="algo", values="ari", aggfunc="first")
    print("\nARI tablosu (satır=özellik, sütun=algo):")
    print(pivot.round(4).to_string())

    print(f"\n--- K={k}: aynı algo, farklı özellik (B0, çiftler arası ARI ort.) ---")
    b0 = sub[sub["algo"] == "B0_KMEANS"]
    feats = sorted(b0["feature"].unique())
    if len(feats) >= 2:
        for fa, fb in combinations(feats, 2):
            ra = b0[b0["feature"] == fa]
            rb = b0[b0["feature"] == fb]
            if not ra.empty and not rb.empty:
                print(f"  B0 {fa} vs {fb}: ARI={ra['ari'].iloc[0]:.4f}  NMI={ra['nmi'].iloc[0]:.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algos", nargs="+", default=DEFAULT_ALGOS)
    p.add_argument("--csv", default=None, help="Tüm çiftleri tek CSV")
    p.add_argument("--reference", default="B0_KMEANS")
    args = p.parse_args()

    idx = build_index(args.algos)
    print(f"Yüklenen konfigürasyon: {len(idx)} / {len(args.algos)*len(FEATURES)*len(DEFAULT_K)}")
    if not idx:
        print("Assignment bulunamadı.")
        return

    st_df = pd.DataFrame(structure_rows(idx))
    print("\nKüme yapısı özeti (ilk 12):")
    print(st_df.head(12).to_string(index=False))

    # Aynı K + aynı feature: algoritmalar arası
    algo_pairs = pairwise_rows(idx, same_k=True, same_feat=True, same_algo=False)
    df_algo = pd.DataFrame(algo_pairs)

    # Aynı K + aynı algo: özellikler arası
    feat_pairs = pairwise_rows(idx, same_k=True, same_feat=False, same_algo=True)
    df_feat = pd.DataFrame(feat_pairs)

    ref_rows = ref_vs_b0_rows(idx, reference=args.reference)
    df_ref = pd.DataFrame(ref_rows)

    print("\n" + "=" * 72)
    print("ÖZET: Meta vs B0 (aynı K ve özellik) — ortalama ARI / NMI")
    if not df_ref.empty:
        summ = df_ref.groupby(["feature", "algo"]).agg(
            ari_mean=("ari", "mean"),
            nmi_mean=("nmi", "mean"),
            n_k=("k", "count"),
        ).reset_index()
        print(summ.round(4).to_string(index=False))

    print("\n" + "=" * 72)
    print("ÖZET: Özellik değişimi (aynı algo+K) — ortalama ARI")
    if not df_feat.empty:
        summ_f = df_feat.groupby(["algo_a", "k"]).agg(
            ari_mean=("ari", "mean"),
            nmi_mean=("nmi", "mean"),
            n_pairs=("ari", "count"),
        ).reset_index().rename(columns={"algo_a": "algo"})
        print(summ_f.round(4).head(20).to_string(index=False))

    for k in DEFAULT_K:
        if not df_ref.empty:
            sub = df_ref[df_ref["k"] == k]
            if sub.empty:
                continue
            print(f"\n--- K={k}: {args.reference} vs meta (ARI) ---")
            print(
                sub.pivot_table(index="feature", columns="algo", values="ari", aggfunc="first")
                .round(4)
                .to_string()
            )
        if not df_feat.empty:
            subf = df_feat[(df_feat["k"] == k) & (df_feat["algo_a"] == "B0_KMEANS")].copy()
            if len(subf):
                print(f"\n--- K={k}: B0 özellik çiftleri (ARI) ---")
                for _, r in subf.iterrows():
                    print(
                        f"  {r['feature_a']} vs {r['feature_b']}: "
                        f"ARI={r['ari']:.4f}  NMI={r['nmi']:.4f}  agree={r['label_agreement']:.1%}"
                    )

    if args.csv:
        out = os.path.normpath(args.csv)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        df_algo["pair_type"] = "algo_same_feat_k"
        df_feat["pair_type"] = "feat_same_algo_k"
        pd.concat([df_algo, df_feat], ignore_index=True).to_csv(out, index=False)
        st_df.to_csv(out.replace(".csv", "_structure.csv"), index=False)
        df_ref.to_csv(out.replace(".csv", "_vs_b0.csv"), index=False)
        print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
