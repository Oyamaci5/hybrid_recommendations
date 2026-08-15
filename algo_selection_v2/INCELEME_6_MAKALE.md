# 6 Makale İncelemesi — Alınanlar, Eksikler, Bizim Konum

Kaynak PDF'ler kullanıcı tarafından yüklendi; metinler çıkarılıp yöntem/protokol/
sonuç bölümleri tarandı. Test sonuçları: `results/pred_v2.csv` (seed 42, fold 1).

## Makale karnesi

| Makale | Yöntem | Raporlanan | Protokol | Ana eksikler |
|---|---|---|---|---|
| GWO+FCM (Katarya 2018) | GWO ile FCM merkez opt. | MAE **0.68** | **70:30** tek split | Resmi fold yok; tek koşu; seed/istatistik yok; küme boyut dağılımı yok; 0.68 ayarlı SVD'den (0.736) iyi — protokol şüphesi |
| HSC Sparrow (Top-N) | Sparrow + WCSS fitness | MAE/SD/RMSE/t | 80:20 tek split | Öneri = "kümedeki en yüksek ortalamalı filmler" → **küme içi kişiselleştirme yok**; WCSS fitness (bizim K1: hedef uyumsuzluğu); tek koşu |
| HHO-K-means | HHO+k-means kıyas | MAE **0.50–0.56** | belirsiz | k-means baseline'ı 0.728 (bizimkiyle uyumlu) ama HHO'ya %28 iyileşme yazıyor — dürüst SOTA'nın çok altında, büyük protokol şüphesi; metrik tanımları verilmemiş |
| Harmony+FCM+HF | HS ile FCM (K=4, m=1.5), CF+CBF α=0.7 karışım | MAE 0.7011, RMSE 0.8974 | belirsiz | K=4 çok kaba; genre'ye bağımlı; tek koşu; split raporsuz. **Artısı: α-karışım fikri** |
| Firefly+collab (Kumar&Prabhu) | Firefly + kümeleme | MAE **0.76–0.80** (K'ya göre) | 80:20 | Tek koşu, istatistik yok. Not: 0.6 değil — bizim bandımız |
| CS-Kmeans (Cuckoo) | CS+k-means; **IUF benzerlik + user/item füzyonu** (P=m·Pu+n·Pi) | grafik, net sayı yok | belirsiz | Split/istatistik yok. **Artısı: füzyon + IUF fikirleri** |

Ortak eksikler (tezin eleştiri bölümü): resmi fold kullanmama, tek koşu/tek seed,
anlamlılık testi yokluğu, küme boyut dağılımı raporlamama (K1 dejenerasyon riski),
güçlü baseline (KMeans++ n_init=10, ayarlı kNN/SVD) yokluğu.

## Alınan teknikler ve TEST sonuçları (genre'siz, K=10, soft top-2)

| Varyant | Kaynak | AVOA MAE | Sonuç |
|---|---|---|---|
| V0 merkezli cosine kNN (mevcut) | — | 0.7726 | referans |
| V1 + significance weighting | Breese/Herlocker | 0.7722 | nötr |
| V2 + IUF | CS-Kmeans | 0.7783 | **zarar** |
| V3 + user-item füzyonu (α=0.9) | CS-Kmeans F.7 | 0.7796 | nötr/zarar |
| **V5 cknn + ALS-MF karışımı (β=0.5)** | Harmony'nin α-karışım fikri, genre'siz | **0.7440** | **★ tavanı geçti** |

V5 detay: ALS-MF (20 faktör, bias'lı, iç-train) tek başına 0.7652; küme-kNN tek
başına 0.7726; **β=0.5 karışım 0.7440 MAE / 0.9475 RMSE / P@10 0.6991 /
NDCG@10 0.8372** → kümesiz tam kNN'i (0.7467/0.8352) tüm metriklerde geçiyor.
Neden: kNN yerel komşuluk sinyali, MF global düşük-rank sinyali taşıyor —
hatalar ilişkisiz, karışım ikisini de düzeltiyor. Literatürdeki 6 makalenin
hiçbiri MF karışımı kullanmıyor → **bizim ayırt edici katkımız**.

## Bizim konum (fold 1, seed 42, resmi split, sızıntısız)

| Sistem | MAE | RMSE | P@10 | NDCG@10 |
|---|---|---|---|---|
| Firefly makalesi bandı | 0.76–0.80 | — | — | — |
| Katarya 2016 (PSO-FCM) | 0.75 | — | — | — |
| Kümesiz tam kNN (tavan) | 0.7467 | 0.9556 | 0.6963 | 0.8352 |
| Ayarlı SVD (Surprise, 5-fold) | 0.736 | — | — | — |
| **AVOA küme-kNN + MF (β=.5)** | **0.7440** | **0.9475** | **0.6991** | **0.8372** |

Şüpheli protokollü 0.68/0.56 iddiaları hariç, **resmi split'te raporlayan tüm
küme tabanlı sistemlerin önündeyiz** ve bunu 1/4 komşu havuzuyla yapıyoruz.
