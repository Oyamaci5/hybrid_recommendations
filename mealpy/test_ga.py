# test_combined_phases.py
import pandas as pd
import os

phase3_root = 'results/phase3'
runs = sorted([d for d in os.listdir(phase3_root)
               if os.path.isdir(os.path.join(phase3_root, d)) and d.isdigit()])
latest = runs[-1]

# Phase 2 sonuçları
path_p2 = os.path.join(phase3_root, latest, 'ml100k-phase2', 'phase2_complete.csv')
# Phase 3 ML-100K sonuçları
path_p3 = os.path.join(phase3_root, latest, 'ml-100k-phase3', 'phase3_complete.csv')

df2 = pd.read_csv(path_p2)
df3 = pd.read_csv(path_p3)

suc2 = df2[df2['success']==True][['algorithm','wcss','davies_bouldin','silhouette','time_seconds']].copy()
suc3 = df3[df3['success']==True][['algorithm','wcss','davies_bouldin','silhouette','time_seconds']].copy()

suc2.columns = ['algorithm','wcss_p2','db_p2','sil_p2','time_p2']
suc3.columns = ['algorithm','wcss_p3','db_p3','sil_p3','time_p3']

merged = suc2.merge(suc3, on='algorithm', how='outer')
merged = merged.sort_values('wcss_p2')

print(f"Phase 2: {len(suc2)} algoritma")
print(f"Phase 3: {len(suc3)} algoritma")
print(f"\n{'Algoritma':<35} {'WCSS_P2':>8} {'DB_P2':>6} {'WCSS_P3':>8} {'DB_P3':>6}")
print("-"*70)
for _, row in merged.iterrows():
    p2 = f"{row['wcss_p2']:>8.2f} {row['db_p2']:>6.3f}" if pd.notna(row.get('wcss_p2')) else f"{'—':>8} {'—':>6}"
    p3 = f"{row['wcss_p3']:>8.2f} {row['db_p3']:>6.3f}" if pd.notna(row.get('wcss_p3')) else f"{'—':>8} {'—':>6}"
    print(f"{row['algorithm']:<35} {p2} {p3}")