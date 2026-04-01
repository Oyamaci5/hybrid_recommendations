# test_phase2_rank.py
import pandas as pd
import numpy as np

df = pd.read_csv('results/phase2_complete.csv')
print(f"Toplam: {len(df)}, Başarılı: {df['success'].sum()}")
ranked = df.sort_values('wcss')
print(f"Toplam başarılı: {len(ranked)}")
print(f"\n{'Sıra':<4} {'Algoritma':<35} {'WCSS':>8} {'DB':>6} {'Sil':>7} {'Süre':>7}")
print("-" * 70)
for i, (_, row) in enumerate(ranked.iterrows(), 1):
    print(f"{i:<4} {row['algorithm']:<35} {row['wcss']:>8.2f} "
          f"{row['davies_bouldin']:>6.3f} {row['silhouette']:>7.4f} "
          f"{row['time_seconds']:>6.1f}s")

# suc = df[df['success'] == True].copy()

# def norm_min(col):
#     r = suc[col].max() - suc[col].min()
#     if r == 0: return pd.Series(0.5, index=suc.index)
#     return 1 - (suc[col] - suc[col].min()) / r

# def norm_max(col):
#     r = suc[col].max() - suc[col].min()
#     if r == 0: return pd.Series(0.5, index=suc.index)
#     return (suc[col] - suc[col].min()) / r

# suc['score_wcss'] = norm_min('wcss')
# suc['score_db']   = norm_min('davies_bouldin')
# suc['score_time'] = norm_min('time_seconds')
# suc['score_gray'] = norm_min('gray_sheep_ratio')
# suc['score_sil']  = norm_max('silhouette')
# suc['composite']  = (0.35*suc['score_wcss'] + 0.15*suc['score_sil'] +
#                      0.30*suc['score_db']   + 0.15*suc['score_gray'] +
#                      0.05*suc['score_time'])

# ranked = suc.sort_values('composite', ascending=False)
# print(f"\n{'Sıra':<4} {'Algoritma':<35} {'WCSS':>8} {'DB':>6} {'Sil':>7} {'Score':>7}")
# print("-" * 70)
# for i, (_, row) in enumerate(ranked.head(25).iterrows(), 1):
#     print(f"{i:<4} {row['algorithm']:<35} {row['wcss']:>8.2f} "
#           f"{row['davies_bouldin']:>6.3f} {row['silhouette']:>7.4f} "
#           f"{row['composite']:>7.3f}")