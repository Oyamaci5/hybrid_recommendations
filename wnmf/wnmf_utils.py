"""
wnmf_utils.py
=============
WNMF deneyleri için yardımcı fonksiyonlar.

İçerik:
    load_ratings_100k()    — ML-100K train/test yükle
    load_ratings_1m()      — ML-1M train/test yükle
    load_assignment()      — assignments.npy + gray_sheep_mask.npy yükle
    load_centroids()         — best_sol.npy centroid matrisi (opsiyonel)
    nearest_centroid_assignments() — train profiline göre en yakın küme
    split_by_cluster()     — rating'leri kümelere böl
    remap_user_ids()       — küme içi kullanıcı indekslerini sıfırdan başlat
    save_results()         — sonuçları CSV olarak kaydet (isteğe bağlı komut satırı başlığı)
    save_dataframe_csv()   — DataFrame’i CSV’ye yaz (isteğe bağlı komut satırı başlığı)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from typing import Dict, List, Tuple, Optional, Any


# ============================================================
# VERİ YÜKLEME
# ============================================================

def load_ratings_100k(base_path: str, test_path: str, fold: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    ML-100K formatında train ve test rating'lerini yükle.

    Dosya formatı: user_id  item_id  rating  timestamp (tab ile ayrılmış)
    Döndürülen array'lerde user_id ve item_id 0-indexed.

    Parametreler
    ------------
    base_path : str — eğitim dosyası (örn: data/ml-100k/u1.base)
    test_path : str — test dosyası   (örn: data/ml-100k/u1.test)
    fold      : int — 1 ise base_path/test_path olduğu gibi kullanılır;
                      2–5 ise aynı dizinde u{fold}.base / u{fold}.test okunur.

    Döndürür
    --------
    (train_ratings, test_ratings) : her biri shape (n, 3) array
        sütunlar: [user_id, item_id, rating]
    """
    if fold != 1:
        data_dir = os.path.dirname(os.path.abspath(base_path))
        base_path = os.path.join(data_dir, f'u{fold}.base')
        test_path = os.path.join(data_dir, f'u{fold}.test')

    def _read(path):
        df = pd.read_csv(
            path, sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            usecols=['user_id', 'item_id', 'rating'],
        )
        df['user_id'] -= 1   # 1-indexed → 0-indexed
        df['item_id'] -= 1
        return df[['user_id', 'item_id', 'rating']].values.astype(np.float32)

    train = _read(base_path)
    test  = _read(test_path)

    print(f"ML-100K yüklendi")
    print(f"  Train: {len(train):,} rating")
    print(f"  Test : {len(test):,} rating")
    print(f"  Kullanıcı: {int(train[:,0].max())+1}, Film: {int(train[:,1].max())+1}")

    return train, test


def load_ratings_100k_all(ratings_path: str, test_ratio: float = 0.2,
                          random_seed: int = 42,
                          fold: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    ML-100K u.data dosyasından tüm rating'leri yükle ve train/test'e böl.

    fold None veya 1: train_test_split ile test_ratio (varsayılan %%20).
    fold=2..5: KFold(n_splits=5, shuffle=True, random_state=random_seed).

    Döndürülen array'lerde user_id ve item_id 0-indexed (943×1682 ile uyumlu).
    """
    if fold is not None and not (1 <= fold <= 5):
        raise ValueError(
            f"load_ratings_100k_all: fold None veya 1..5 olmalı, gelen: {fold}"
        )

    df = pd.read_csv(
        ratings_path, sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        usecols=['user_id', 'item_id', 'rating'],
    )
    df['user_id'] -= 1
    df['item_id'] -= 1

    if fold is None or fold == 1:
        train_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=random_seed,
            shuffle=True,
        )
        split_label = f'rastgele %%{int(round(test_ratio * 100))} (seed={random_seed})'
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        splits = list(kf.split(df))
        train_idx, test_idx = splits[fold - 1]
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        split_label = f'KFold fold {fold}/5 (seed={random_seed})'

    train = train_df[['user_id', 'item_id', 'rating']].values.astype(np.float32)
    test = test_df[['user_id', 'item_id', 'rating']].values.astype(np.float32)

    print(f"ML-100K yüklendi ({split_label})")
    print(f"  Kaynak: {ratings_path}")
    print(f"  Train: {len(train):,} rating")
    print(f"  Test : {len(test):,} rating")
    print(f"  Kullanıcı: {int(max(train[:, 0].max(), test[:, 0].max())) + 1}, "
          f"Film: {int(max(train[:, 1].max(), test[:, 1].max())) + 1}")

    return train, test


def load_ratings_1m(ratings_path: str, test_ratio: float = 0.2,
                    random_seed: int = 42, fold: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    ML-1M formatında rating'leri yükle ve train/test'e böl.

    ML-1M'in hazır fold'ları yok.
    fold None veya 1: train_test_split ile test_ratio (varsayılan %20), random_state=random_seed.
    fold=2..5: KFold(n_splits=5, shuffle=True, random_state=random_seed) parçası splits[fold-1].
    Dosya formatı: UserID::MovieID::Rating::Timestamp

    Parametreler
    ------------
    ratings_path : str   — ratings.dat dosya yolu
    test_ratio   : float — holdout için test oranı; KFold fold'larında yok sayılır
    random_seed  : int   — holdout ve KFold için random_state
    fold         : int veya None — None/1: holdout; 2..5: KFold parçası

    Döndürür
    --------
    (train_ratings, test_ratings)
    """
    if fold is not None and not (1 <= fold <= 5):
        raise ValueError(f"load_ratings_1m: fold None veya 1..5 olmalı, gelen: {fold}")

    rows = []
    with open(ratings_path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('::')
            if len(parts) >= 3:
                rows.append((
                    int(parts[0]) - 1,   # 0-indexed
                    int(parts[1]) - 1,
                    float(parts[2]),
                ))

    df = pd.DataFrame(rows, columns=['user_id', 'item_id', 'rating'])

    if fold is None or fold == 1:
        train_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=random_seed,
            shuffle=True,
        )
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        splits = list(kf.split(df))
        train_idx, test_idx = splits[fold - 1]
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

    combo = pd.concat([train_df, test_df], axis=0)

    # User IDs: ML-1M has contiguous 1-indexed users (1..6040); after the -1
    # subtraction above they are 0..6039.  No remapping is applied so that
    # ratings[:, 0] == i  ↔  assignments[i] (position in the pivot-table
    # produced by generate_assignments.py).  Guard against accidental gaps.
    unique_users = np.sort(combo['user_id'].unique())
    if unique_users[-1] - unique_users[0] + 1 != len(unique_users):
        raise ValueError(
            f"load_ratings_1m: non-contiguous user IDs detected "
            f"(expected {unique_users[-1] - unique_users[0] + 1} IDs, "
            f"found {len(unique_users)}).  User remapping would misalign "
            "with assignments.npy — update the pipeline if you subset the data."
        )

    # Item IDs: ML-1M movie IDs have gaps (1..3952 with ~246 missing entries).
    # Compress to 0..n_unique_items-1 so the WNMF V matrix is the right size.
    unique_items = np.sort(combo['item_id'].unique())
    i_map = {int(it): j for j, it in enumerate(unique_items)}

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['item_id'] = train_df['item_id'].map(i_map)
    test_df['item_id'] = test_df['item_id'].map(i_map)

    train = train_df[['user_id', 'item_id', 'rating']].values.astype(np.float32)
    test = test_df[['user_id', 'item_id', 'rating']].values.astype(np.float32)

    print(f"ML-1M yüklendi")
    print(f"  Train: {len(train):,} rating")
    print(f"  Test : {len(test):,} rating")
    print(f"  Kullanıcı: {int(train[:,0].max())+1}, Film: {int(train[:,1].max())+1}")

    return train, test


# ============================================================
# ASSIGNMENT YÜKLEME
# ============================================================

def load_assignment(assignment_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    generate_assignments.py'nin ürettiği dosyaları yükle.

    Parametreler
    ------------
    assignment_dir : str — results/assignments/ml100k/B1_HHO/ gibi bir yol

    Döndürür
    --------
    (assignments, gray_sheep_mask)
        assignments     : shape (n_users,) — her kullanıcının küme ID'si
        gray_sheep_mask : shape (n_users,) — True = gray sheep
    """
    assignments = np.load(os.path.join(assignment_dir, 'assignments.npy'))
    gray_mask   = np.load(os.path.join(assignment_dir, 'gray_sheep_mask.npy'))

    if len(gray_mask) != len(assignments):
        raise ValueError(
            f"load_assignment: assignments.npy ({len(assignments)}) ve "
            f"gray_sheep_mask.npy ({len(gray_mask)}) uzunlukları eşleşmiyor."
        )

    n_gray    = gray_mask.sum()
    n_users   = len(assignments)
    n_clusters = len(np.unique(assignments[~gray_mask]))
    a_min, a_max = int(assignments.min()), int(assignments.max())

    print(f"Assignment yüklendi: {assignment_dir}")
    print(f"  cluster_id  : min={a_min}, max={a_max}, len={n_users}")
    print(f"  Kullanıcı   : {n_users}")
    print(f"  Gray sheep  : {n_gray} ({n_gray/n_users*100:.1f}%)")
    print(f"  Aktif küme  : {n_clusters}")

    return assignments, gray_mask


def load_memberships(assignment_dir: str) -> Optional[np.ndarray]:
    """
    Soft üyelik matrisi (memberships.npy) varsa yükle.
    Yoksa None döner; eski assignment çıktılarıyla geriye uyum korunur.
    """
    path = os.path.join(assignment_dir, 'memberships.npy')
    if not os.path.exists(path):
        return None
    memberships = np.load(path)
    if memberships.ndim != 2:
        raise ValueError(
            f"load_memberships: memberships.npy 2D olmalı, gelen shape={memberships.shape}"
        )
    print(f"  memberships yüklendi: {memberships.shape}")
    return memberships


def load_centroids(assignment_dir: str, n_clusters: int) -> Optional[np.ndarray]:
    """
    generate_assignments.py'nin kaydettiği best_sol.npy → (K, n_features) centroid matrisi.
    Dosya yoksa veya boyut uyuşmazsa None.
    """
    path = os.path.join(assignment_dir, 'best_sol.npy')
    if not os.path.isfile(path):
        return None
    best_sol = np.load(path)
    if n_clusters < 1 or best_sol.size % n_clusters != 0:
        return None
    n_features = best_sol.size // n_clusters
    return np.asarray(best_sol, dtype=np.float64).reshape(n_clusters, n_features)


def build_user_profile_matrix(
    train: np.ndarray,
    n_users: int,
    n_features: int,
) -> np.ndarray:
    """Train rating'lerinden kullanıcı profil matrisi (eksik=0). Shape: (n_users, n_features)."""
    profiles = np.zeros((n_users, n_features), dtype=np.float64)
    for row in train:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        if 0 <= u < n_users and 0 <= i < n_features:
            profiles[u, i] = r
    return profiles


def _euclidean_distance_batch(users: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    users = np.asarray(users, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    u2 = (users ** 2).sum(axis=1, keepdims=True)
    c2 = (centroids ** 2).sum(axis=1, keepdims=True).T
    cross = users @ centroids.T
    return np.maximum(u2 + c2 - 2.0 * cross, 0.0)


def _pearson_distance_batch(users: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    user_mask = (users != 0).astype(np.float64)
    cent_mask = (centroids != 0).astype(np.float64)
    u_count = np.maximum(user_mask.sum(axis=1, keepdims=True), 1.0)
    c_count = np.maximum(cent_mask.sum(axis=1, keepdims=True), 1.0)
    u_mean = (users * user_mask).sum(axis=1, keepdims=True) / u_count
    c_mean = (centroids * cent_mask).sum(axis=1, keepdims=True) / c_count
    u_centered = np.where(user_mask > 0, users - u_mean, 0.0)
    c_centered = np.where(cent_mask > 0, centroids - c_mean, 0.0)
    u_norm = np.maximum(np.linalg.norm(u_centered, axis=1, keepdims=True), 1.0)
    c_norm = np.maximum(np.linalg.norm(c_centered, axis=1, keepdims=True), 1.0)
    corr = (u_centered / u_norm) @ (c_centered / c_norm).T
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
    corr = np.clip(corr, -1.0, 1.0)
    return 1.0 - corr


def nearest_centroid_assignments(
    user_profiles: np.ndarray,
    centroids: np.ndarray,
    metric: str = 'euclidean',
) -> np.ndarray:
    """Her kullanıcı için en yakın centroid kümesini döndür. Shape: (n_users,)."""
    metric = (metric or 'euclidean').strip().lower()
    if metric == 'pearson':
        dist = _pearson_distance_batch(user_profiles, centroids)
    else:
        dist = _euclidean_distance_batch(user_profiles, centroids)
    return np.argmin(dist, axis=1).astype(np.int32)


def resolve_test_cluster_ids(
    train: np.ndarray,
    assignments: np.ndarray,
    centroids: Optional[np.ndarray],
    nearest_centroid: bool,
    n_items: int,
    centroid_metric: str = 'euclidean',
    algo_label: str = '',
) -> np.ndarray:
    """
    Test tahmininde kullanılacak küme ID'leri.
    nearest_centroid=False → offline assignments; True → train profiline göre en yakın centroid.
    """
    if not nearest_centroid:
        return assignments
    if centroids is None:
        print(
            f"  [{algo_label}] uyarı: --nearest-centroid aktif ama best_sol.npy yok; "
            f"offline assignment kullanılıyor.",
            flush=True,
        )
        return assignments

    n_users = len(assignments)
    n_features = centroids.shape[1]
    if n_features != n_items:
        print(
            f"  [{algo_label}] uyarı: centroid boyutu ({n_features}) != n_items ({n_items}); "
            f"en yakın centroid atlanıyor (offline assignment kullanılıyor).",
            flush=True,
        )
        return assignments

    profiles = build_user_profile_matrix(train, n_users, n_features)
    test_ids = nearest_centroid_assignments(profiles, centroids, metric=centroid_metric)
    changed = int(np.sum(test_ids != assignments))
    print(
        f"  [{algo_label}] nearest-centroid: {changed}/{n_users} kullanıcı "
        f"offline atamadan farklı küme seçti ({centroid_metric})",
        flush=True,
    )
    return test_ids


# ============================================================
# KÜMELERİ AYIR
# ============================================================

def split_by_cluster(
    ratings: np.ndarray,
    assignments: np.ndarray,
    gray_mask: np.ndarray,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Rating'leri küme assignment'larına göre böl.

    Önemli: WNMF her küme için bağımsız U ve V matrisi öğrenir.
    Bu fonksiyon her kümenin rating'lerini ayrı array'e koyar.

    Parametreler
    ------------
    ratings     : shape (n_ratings, 3) — tüm rating'ler [user_id, item_id, rating]
    assignments : shape (n_users,)     — her kullanıcının küme ID'si
    gray_mask   : shape (n_users,)     — True = gray sheep

    Döndürür
    --------
    cluster_ratings : dict {cluster_id → np.ndarray shape (m, 3)}
        Her kümedeki kullanıcıların rating'leri.
        user_id'ler orijinal (global) indekslerdir.
    gray_ratings    : np.ndarray shape (m, 3)
        Gray sheep kullanıcıların rating'leri.
    """
    user_ids = ratings[:, 0].astype(np.int32)

    # Gray sheep rating'leri
    gray_user_set  = set(np.where(gray_mask)[0])
    gray_mask_r    = np.isin(user_ids, list(gray_user_set))
    gray_ratings   = ratings[gray_mask_r]

    # Normal kullanıcıların küme bazlı rating'leri
    normal_ratings = ratings[~gray_mask_r]
    normal_users   = normal_ratings[:, 0].astype(np.int32)

    cluster_ratings: Dict[int, np.ndarray] = {}
    for cid in np.unique(assignments[~gray_mask]):
        cid = int(cid)
        cluster_user_set = set(np.where((assignments == cid) & (~gray_mask))[0])
        mask = np.isin(normal_users, list(cluster_user_set))
        if mask.sum() > 0:
            cluster_ratings[cid] = normal_ratings[mask]

    n_empty = sum(1 for v in cluster_ratings.values() if len(v) == 0)
    print(f"Küme bölme tamamlandı:")
    print(f"  Normal küme : {len(cluster_ratings)} ({n_empty} boş)")
    print(f"  Gray sheep  : {len(gray_ratings)} rating")

    return cluster_ratings, gray_ratings


def remap_user_ids(
    train: np.ndarray,
    test: np.ndarray,
    n_items: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], int]:
    """
    Bir kümenin kullanıcı ID'lerini 0'dan başlayacak şekilde yeniden numaralandır.

    Neden gerekli: WNMFModel U matrisini (n_users, latent_dim) boyutunda tutar.
    Kümedeki kullanıcılar global indekslerle gelir (örn: 47, 253, 891...).
    Bunları 0, 1, 2, ... olarak yeniden numaralandırmak gerekir.

    Parametreler
    ------------
    train   : shape (n, 3) — kümenin train rating'leri
    test    : shape (m, 3) — kümenin test rating'leri
    n_items : int          — toplam film sayısı (değişmez)

    Döndürür
    --------
    train_r     : yeniden numaralandırılmış train
    test_r      : yeniden numaralandırılmış test
    uid_map     : {eski_id → yeni_id} sözlüğü
    n_users_loc : kümedeki kullanıcı sayısı
    """
    # Train'deki tüm kullanıcıları topla
    all_users   = np.unique(train[:, 0].astype(np.int32))
    uid_map     = {int(u): i for i, u in enumerate(all_users)}
    n_users_loc = len(all_users)

    def remap(arr):
        arr2 = arr.copy()
        for old, new in uid_map.items():
            arr2[arr[:, 0] == old, 0] = new
        return arr2

    train_r = remap(train)

    # Test'te train'de olmayan kullanıcı varsa filtrele
    test_mask = np.isin(test[:, 0].astype(np.int32), list(uid_map.keys()))
    test_r    = remap(test[test_mask])

    return train_r, test_r, uid_map, n_users_loc


# ============================================================
# SONUÇ KAYDETME
# ============================================================

def save_dataframe_csv(
    df          : pd.DataFrame,
    path        : str,
    run_command : Optional[str] = None,
    index       : bool = False,
) -> None:
    """
    DataFrame’i CSV olarak yazar; run_command verilirse ilk satır:
    # command: <çalıştırılan komut>
    (pandas: pd.read_csv(..., comment='#') ile okunabilir)
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        if run_command:
            f.write(f"# command: {run_command}\n")
        df.to_csv(f, index=index)


def save_results(
    results      : list,
    save_path    : str,
    filename     : str = 'wnmf_results.csv',
    run_command  : Optional[str] = None,
):
    """
    Deney sonuçlarını CSV olarak kaydet.

    Parametreler
    ------------
    results      : list of dict — her satır bir sonuç
    save_path    : str          — kayıt klasörü
    filename     : str          — dosya adı
    run_command  : çalıştırılan komut satırı (dosya başına yorum olarak yazılır)
    """
    os.makedirs(save_path, exist_ok=True)
    df   = pd.DataFrame(results)
    path = os.path.join(save_path, filename)
    save_dataframe_csv(df, path, run_command=run_command)
    print(f"Sonuçlar kaydedildi: {path}")
    return df


def append_rows_to_accum_csv(rows: List[Dict[str, Any]], path: str) -> None:
    """
    Deney satırlarını tek bir CSV'de biriktirir (aynı dosyaya tekrar tekrar yazım).

    Var olan dosyayla sütun birleşimi yapılır; dosya yoksa oluşturulur.
    Başlık satırı # ile başlamaz (append ile uyum için).
    """
    if not rows:
        return
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        new_df.to_csv(path, index=False, encoding='utf-8')
        print(f"Birikim CSV oluşturuldu: {path}  (+{len(new_df)} satır)")
        return
    try:
        old_df = pd.read_csv(path, encoding='utf-8', comment='#')
    except pd.errors.EmptyDataError:
        new_df.to_csv(path, index=False, encoding='utf-8')
        print(f"Birikim CSV yazıldı: {path}  (+{len(new_df)} satır)")
        return
    all_cols = list(dict.fromkeys(list(old_df.columns) + list(new_df.columns)))
    old_df = old_df.reindex(columns=all_cols)
    new_df = new_df.reindex(columns=all_cols)
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined.to_csv(path, index=False, encoding='utf-8')
    print(f"Birikim CSV güncellendi: {path}  (+{len(new_df)} satır, toplam {len(combined)})")
