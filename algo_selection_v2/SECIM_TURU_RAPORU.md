# Seçim Turu Raporu — 2026-08-02

Protokol: ML-100K fold 1 (yalnız train), NMF-20 uzayı, K=7, SSE-WCSS fitness,
kmref AÇIK, 22 algoritma × 5 seed (epoch=50, pop=30) + KMeans++/random baseline.
Dosyalar: `results/per_run.csv`, `results/summary.csv`, `results/space_check.csv`, `results/budget_test.csv`.

## 1. Uzay kararı (E6 kontrolü)

| Uzay | Hopkins | En iyi K | Silhouette |
|---|---|---|---|
| **nmf20 (seçildi)** | **0.916** | 7 | **0.291** (DB=1.405) |
| svd10 | 0.815 | 3 | 0.317 (K=3 CF için işlevsiz) |
| svd20 | 0.809 | 3 | 0.275 |
| raw_zscore | 0.848 | 10 | 0.229 |

Eski karşılaştırmadaki uzayda silhouette ≈ −0.03 idi → H8 hatası doğrulandı:
**düşük iyileşmelerin ana nedeni uzay seçimiydi.** NMF-20'de yapı var (sil +0.29).
Kendi WNMF uzayınız `--extra-npy` ile mutlaka kıyasa katılmalı (NMF-20 onun vekili).

## 2. Seçim turu sonucu (Friedman χ²=64.3, p=0.00001 → sıralama anlamlı)

| Sıra | Algoritma | WCSS (kmref) | std | avg rank |
|---|---|---|---|---|
| 1 | **B0_KMEANS++ (n_init=10)** | 223.49 | 0.63 | 1.4 |
| 2 | SMA | 224.47 | 2.04 | 3.6 |
| 3 | MPA | 228.49 | 5.30 | 6.4 |
| 4 | NGO | 229.46 | 9.48 | 7.0 |
| 5 | HGS | 231.18 | 7.50 | 9.0 |
| 6 | AVOA | 230.02 | 2.53 | 9.2 |
| … | HHO | 236.25 | 12.04 | 12.4 |
| son | SA / PSO / SquirrelSA | 260–270 | 14–31 | 19–21 |

**Seçim kapısı bulgusu:** epoch=50 bütçesinde hiçbir meta KMeans++'ı geçemedi.
Bütçe testi (epoch=100, pop=50, 3 seed): SMA 223.06 ve MPA 222.98 ile B0
seviyesine ulaştı/geçti; AVOA ulaşabiliyor ama kararsız (223.1–231.9);
HHO geride (228–241).

**Metrik çelişkisi bulgusu (tez için değerli):** WCSS'te en kötü algoritmalar
(SA, PSO, GA) silhouette/DB'de en iyi çıkıyor — kötü optimizasyon, kaba (K≈3
benzeri) dejenere yapılara düşüyor ve silhouette bunu ödüllendiriyor.
Tek metrikle seçim yapılamaz; WCSS + Sil + DB birlikte raporlanmalı.

## 3. Yorum ve sonraki karar

1. Saf WCSS hedefinde meta-init'in KMeans++'a katkısı ≈ 0 (ancak eşitliyor).
   Eski CF deneylerindeki %0.05'lik iyileşmeler bununla tutarlı — sorun
   algoritmada değil, **hedef fonksiyonda**. Makale düzeyi iyileşme için yol:
   tahmine hizalı fitness (knn_mae, roadmap Faz 1) veya bulanık/soft atama.
2. Kısa liste (final tur adayları): **SMA, MPA, NGO, HGS, AVOA** + literatür
   kıyası için **HHO**. AVOA mealpy'de mevcut (`AVOA.OriginalAVOA`) — özel
   implementasyona gerek yok; özgünlük cümlesi ("film önerisinde AVOA'lı
   kümeleme yok") hâlâ geçerli.
3. BWO / DOA / SFOA mealpy 3.0.3'te yok; gerekiyorsa repo'daki özel
   implementasyonlar (`mealpy/doa_optimizer.py`, `sfoa_optimizer.py`) aynı
   protokole sarılabilir.

## 4. Final tur komutu (kendi makinenizde, ~30–60 dk)

```bash
# algos.txt'yi kısa listeyle daraltın: SMA, MPA, NGO, HGS, AVOA, HHO
python algo_selection_v2/recompute_scores.py \
  --feature-mode nmf --dim 20 --k 7 \
  --runs 30 --epoch 100 --pop 50 --resume
# ardından WNMF uzayıyla tekrar: --features <wnmf_features.npy>
```

Final turdan sonra: kısa listenin ilk 2–3'ü Thakrar pipeline'ına init olarak
verilip CV5 CF metrikleriyle (MAE/RMSE/NDCG, kullanıcı-bazlı Wilcoxon) tez
tablosu üretilecek.
