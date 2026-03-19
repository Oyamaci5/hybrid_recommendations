import pandas as pd

df2 = pd.read_csv('mealpy/results/phase2_success.csv')
df2_all = pd.read_csv('mealpy/results/phase2_complete.csv')

print(f"Aşama 2 toplam: {len(df2_all)}")
print(f"Başarılı: {len(df2)}")
print()
print("TÜM BAŞARILI SONUÇLAR (silhouette sıralı):")
cols = ['algorithm','wcss','silhouette','davies_bouldin','gray_sheep_ratio','time_seconds']
print(df2[cols].sort_values('silhouette', ascending=False).to_string())
print()
print("SSA, GWO, HHO, WOA, PSO sonuçları:")
hedef = ['SSA.OriginalSSA','GWO.OriginalGWO','HHO.OriginalHHO',
         'WOA.OriginalWOA','PSO.OriginalPSO']
print(df2[df2['algorithm'].isin(hedef)][cols].to_string())