# test_ml1m_size.py
import numpy as np
import pandas as pd
import time
import os
from mealpy_comparison_v2 import make_fitness_function, mkmeans_plus_plus_init

print("ML-1M yükleniyor...")
path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'ml-1m', 'ratings.dat')
print("Yol:", path)
print("Dosya var mı:", os.path.exists(path))

rows = []
with open(path, 'r', encoding='latin-1') as f:
    for line in f:
        parts = line.strip().split('::')
        if len(parts) >= 3:
            rows.append((int(parts[0]), int(parts[1]), float(parts[2])))

df = pd.DataFrame(rows, columns=['user_id', 'item_id', 'rating'])
matrix = df.pivot_table(index='user_id', columns='item_id',
                        values='rating', fill_value=0).values.astype(np.float32)
print(f"Matrix: {matrix.shape}")
print(f"Boyut : {matrix.nbytes / 1e6:.1f} MB")

K = 70
print(f"\n1 init çözümü üretiliyor (K={K})...")
t0 = time.time()
init = mkmeans_plus_plus_init(matrix, K=K, n_solutions=1, seed=42)
t1 = time.time()
print(f"1 init çözümü: {t1-t0:.2f}s")
print(f"30 çözüm tahmini: {(t1-t0)*30/60:.1f} dakika")

print(f"\n1 fitness çağrısı test ediliyor...")
fitness_fn = make_fitness_function(matrix, K)
t0 = time.time()
fitness_fn(init[0])
t1 = time.time()
print(f"1 fitness çağrısı: {t1-t0:.2f}s")
print(f"50 epoch × 30 pop = 1500 çağrı tahmini: {(t1-t0)*1500/60:.1f} dakika/algoritma")
print(f"25 algoritma toplam tahmini: {(t1-t0)*1500*25/3600:.1f} saat")