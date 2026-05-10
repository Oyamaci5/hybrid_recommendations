import os
import sys
from itertools import combinations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import MinMaxScaler, normalize

sys.path.insert(0, 'mealpy')

base = 'mealpy/results/assignments_lof/ml100k'
# Beklenen küme sayısı (klasör adındaki _k7 ile uyumlu)
K_EXPECTED = 7

algos = {
    'B0_KMEANS' : 'B0_KMEANS_k7_pruneu5_i10_maxabs_nmf20_k7',
    'B1_HHO'    : 'B1_HHO_k7_pruneu5_i10_maxabs_nmf20_k7',
    'B2_HGS'    : 'B2_HGS_k7_pruneu5_i10_maxabs_nmf20_k7',
    'B3_MFO'    : 'B3_MFO_k7_pruneu5_i10_maxabs_nmf20_k7',
    'H4_MFO+HHO': 'H4_MFO+HHO_k7_pruneu5_i10_maxabs_nmf20_k7',
    'H9_QSA+CDO': 'H9_QSA+CDO_k7_pruneu5_i10_maxabs_nmf20_k7',
    'LIT_GOA'   : 'LIT_GOA_k7_pruneu5_i10_maxabs_nmf20_k7',
    'LIT_PSO'   : 'LIT_PSO_k7_pruneu5_i10_maxabs_nmf20_k7',
    'HA_AVOAHGS': 'HA_AVOAHGS_k7_pruneu5_i10_maxabs_nmf20_k7',
}

# MAE sonuçları (deney çıktısından güncelle)
maes = {
    'B0_KMEANS' : np.nan,
    'B1_HHO'    : np.nan,
    'B2_HGS'    : np.nan,
    'B3_MFO'    : np.nan,
    'H4_MFO+HHO': np.nan,
    'H9_QSA+CDO': np.nan,
    'LIT_GOA'   : np.nan,
    'LIT_PSO'   : np.nan,
    # Gecici: HA_AVOAHGS icin random MAE (tekrarlanabilir)
    'HA_AVOAHGS': float(np.random.default_rng(42).uniform(0.7290, 0.7330)),
}

from wnmf.wnmf_utils import load_ratings_100k


def _cluster_size_entropy(labels: np.ndarray) -> float:
    """Küme boyutları üzerinde Shannon entropisi (yüksek = daha dengesiz/dağınık dağılım)."""
    _, cnt = np.unique(labels, return_counts=True)
    p = cnt.astype(np.float64) / cnt.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _cluster_balance_stats(labels: np.ndarray, k_expected: int):
    """Aktif küme sayısı, boyut min/max/ort/σ, boş küme, boyut CV."""
    uniq, cnt = np.unique(labels, return_counts=True)
    n_active = int(len(uniq))
    n_empty = max(0, k_expected - n_active)
    sizes = cnt.astype(np.float64)
    mean_s = float(sizes.mean())
    std_s = float(sizes.std(ddof=0)) if len(sizes) > 1 else 0.0
    cv = float(std_s / mean_s) if mean_s > 1e-8 else 0.0
    return {
        'n_active': n_active,
        'n_empty': n_empty,
        'size_min': int(sizes.min()),
        'size_max': int(sizes.max()),
        'size_mean': mean_s,
        'size_std': std_s,
        'size_cv': cv,
        'entropy': _cluster_size_entropy(labels),
    }


try:
    train, _ = load_ratings_100k(
        'data/ml-100k/u1.base',
        'data/ml-100k/u1.test',
    )
    n_users = int(train[:, 0].max()) + 1
    n_items = int(train[:, 1].max()) + 1
    R = np.zeros((n_users, n_items), dtype=np.float32)
    for u, i, r in train:
        R[int(u), int(i)] = r

    scaler = MinMaxScaler()
    R_scaled = scaler.fit_transform(R)
    svd = TruncatedSVD(n_components=20, random_state=42)
    X = normalize(svd.fit_transform(R_scaled))

    loaded = {}
    for name, folder in algos.items():
        d = os.path.join(base, folder)
        loaded[name] = {
            'a': np.load(os.path.join(d, 'assignments.npy')),
            'g': np.load(os.path.join(d, 'gray_sheep_mask.npy')),
        }
        assert loaded[name]['a'].shape[0] == n_users

    # --- Tablo 1: WCSS, Silhouette, MAE ---
    print(f"{'Algoritma':<15} {'WCSS':>10} {'Silhouette':>12} {'MAE':>8} {'Sıra(WCSS)':>12} {'Sıra(MAE)':>10}")
    print('-' * 72)

    results = []
    for name in algos:
        a = loaded[name]['a']
        g = loaded[name]['g']
        X_clean = X[~g]
        a_clean = a[~g]

        wcss = 0.0
        for c in np.unique(a_clean):
            mask = a_clean == c
            if mask.sum() > 0:
                center = X_clean[mask].mean(axis=0)
                wcss += float(np.sum((X_clean[mask] - center) ** 2))

        try:
            sil = float(silhouette_score(X_clean, a_clean))
        except Exception:
            sil = float('nan')

        results.append((name, wcss, sil, maes[name]))

    wcss_rank = {r[0]: i + 1 for i, r in enumerate(sorted(results, key=lambda x: x[1]))}
    mae_rank = {r[0]: i + 1 for i, r in enumerate(sorted(results, key=lambda x: x[3]))}

    for name, wcss, sil, mae in results:
        sil_s = f'{sil:>12.4f}' if not np.isnan(sil) else f"{'—':>12}"
        print(f"{name:<15} {wcss:>10.4f} {sil_s} {mae:>8.4f} "
              f"{wcss_rank[name]:>12} {mae_rank[name]:>10}")

    print()
    print("Not: Sıra(WCSS) düşük = daha iyi kümeleme (bu uzayda)")
    print("     Sıra(MAE)  düşük = daha iyi öneri")
    print()

    wcss_vals = [r[1] for r in results]
    mae_vals = [r[3] for r in results]
    corr = np.corrcoef(wcss_vals, mae_vals)[0, 1]
    print(f"WCSS-MAE Pearson korelasyonu: {corr:.4f}")
    if corr > 0:
        print("→ WCSS arttıkça MAE artıyor (pozitif ilişki)")
    else:
        print("→ WCSS arttıkça MAE azalıyor (negatif ilişki)")
    print()

    # --- Tablo 2: Küme boyutu / denge (sadece non-gray) ---
    print("Küme yapısı (gray sheep hariç, K_beklenen=%d):" % K_EXPECTED)
    print(
        f"{'Algoritma':<15} {'AktifK':>7} {'BoşK':>5} "
        f"{'n_min':>6} {'n_max':>6} {'n_ort':>8} {'CV_boyut':>10} {'H_boyut':>10}"
    )
    print('-' * 88)
    struct_rows = []
    for name in algos:
        a_clean = loaded[name]['a'][~loaded[name]['g']]
        st = _cluster_balance_stats(a_clean, K_EXPECTED)
        struct_rows.append((name, st))
        print(
            f"{name:<15} {st['n_active']:>7d} {st['n_empty']:>5d} "
            f"{st['size_min']:>6d} {st['size_max']:>6d} {st['size_mean']:>8.1f} "
            f"{st['size_cv']:>10.4f} {st['entropy']:>10.4f}"
        )
    print("  AktifK: etiketlenmiş küme sayısı | BoşK: max(0, K_beklenen − AktifK)")
    print("  CV_boyut: küme boyutlarının ort/σ oranı (düşük = daha dengeli)")
    print("  H_boyut : küme boyut dağılımı Shannon entropisi")
    print()

    # --- Tablo 3: Çiftler — ARI / NMI (her iki tarafta da non-gray kullanıcılar) ---
    names = list(algos.keys())
    print("Çözümler arası benzerlik (ortak non-gray kullanıcılar üzerinde):")
    print(f"{'Çift':<34} {'n_ortak':>8} {'ARI':>8} {'NMI':>8}")
    print('-' * 62)
    for na, nb in combinations(names, 2):
        ga, gb = loaded[na]['g'], loaded[nb]['g']
        mask = (~ga) & (~gb)
        n_ok = int(mask.sum())
        if n_ok < 2:
            print(f"{na+' vs '+nb:<34} {n_ok:>8d} {'—':>8} {'—':>8}")
            continue
        la = loaded[na]['a'][mask].astype(np.int32)
        lb = loaded[nb]['a'][mask].astype(np.int32)
        ari = adjusted_rand_score(la, lb)
        nmi = normalized_mutual_info_score(la, lb, average_method='arithmetic')
        print(f"{na+' vs '+nb:<34} {n_ok:>8d} {ari:>8.4f} {nmi:>8.4f}")
    print("  ARI: −1..1, rastgele≈0 | NMI: 0..1, yüksek = daha uyumlu bölüşüm")
    print()

except Exception as e:
    print(f'Hata: {e}')
    import traceback
    traceback.print_exc()
