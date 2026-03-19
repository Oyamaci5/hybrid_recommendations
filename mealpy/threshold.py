import numpy as np
import pandas as pd

# Fonksiyonu buraya koy
def load_movielens(path):
    df = pd.read_csv(
        path, sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    matrix = df.pivot_table(
        index='user_id',
        columns='item_id',
        values='rating',
        fill_value=0
    ).values.astype(np.float32)
    return matrix

# Yolu kendi klasörüne göre ayarla
full_matrix = load_movielens('data/movielens_100k/u.data')

print(f"Matrix shape: {full_matrix.shape}")
print(f"Toplam kullanıcı: {full_matrix.shape[0]}")

# Threshold testi
print("\nThreshold → Gray Sheep oranı:")
print("-" * 40)

for t in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    # Her kullanıcının dolu rating oranı
    user_activity = np.count_nonzero(full_matrix, axis=1) / full_matrix.shape[1]
    
    # Düşük aktiviteli = az rating yapmış = gray sheep adayı
    gray = (user_activity < (1 - t)).sum()
    n_users = full_matrix.shape[0]
    print(f"  t={t:.2f} → Gray sheep: {gray:4d} / {n_users} = {gray/n_users:.1%}")

# Kullanıcı aktivite dağılımı
print(f"\nKullanıcı aktivite istatistikleri:")
user_activity = np.count_nonzero(full_matrix, axis=1) / full_matrix.shape[1]
print(f"  Min  : {user_activity.min():.4f}")
print(f"  Max  : {user_activity.max():.4f}")
print(f"  Ortalama: {user_activity.mean():.4f}")
print(f"  Medyan  : {np.median(user_activity):.4f}")
print(f"  %25  : {np.percentile(user_activity, 25):.4f}")
print(f"  %75  : {np.percentile(user_activity, 75):.4f}")