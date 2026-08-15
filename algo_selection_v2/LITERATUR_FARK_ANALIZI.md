# Literatür Farkı Analizi — "MAE neden 0.85, makaleler neden 0.70?"

Deney: `predictor_upgrade.py`, fold 1, seed 42, K=7, NMF-20.
Veri: `results/predictor_upgrade.csv`.

## 1. Cevap: eksik kümelemede değil, TAHMİNCİ katmanındaydı

Aynı kümeleme, üç tahminci (test MAE):

| Tahminci | B0_KMEANS++ | GWO (pred-fit) | AVOA (pred-fit) |
|---|---|---|---|
| P0 küme-film ort. (eski pilot) | 0.8485 | 0.8363 | 0.8361 |
| P1 bias (kull. ort + küme sapması) | 0.7883 | 0.7723 | 0.7747 |
| P2 küme-içi kNN (cosine, k=30) | 0.7877 | 0.7611 | **0.7559** |

Kümesiz referanslar: item_mean 0.829, bias_global 0.769, **kNN(tümü) 0.747**.

- 0.85→0.76 düşüşün tamamı tahminci yükseltmesinden geldi. Literatürün 0.73–0.75
  bandı kNN/MF tahmincilerinden geliyor; kümeleme algoritmasından değil.
- **Protokol doğrulaması:** bizim kNN(tümü)=0.747, Surprise kütüphanesinin yayın
  benchmark'ı KNNWithMeans=0.750 ile birebir uyuşuyor → pipeline artık literatür
  kalitesinde ölçüyor. (SVD referansı: 0.736.)

## 2. "Algoritmalar arası fark neden yoktu?" — tahminci doygunluğu

Küme-film ortalaması kaba bir tahminci: üyelik değişimlerini ortalamanın içinde
eritiyor; bu yüzden tüm algoritmalar 0.84–0.85'e sıkışıyordu. Tahminci keskinleşince
fark ortaya çıktı:

- **AVOA (pred-fit + cknn) 0.7559 vs B0_KMEANS++ (cknn) 0.7877 → %4.0 iyileşme.**
- GWO 0.7611 → %3.4. AVOA–GWO arası %0.7 → algoritmalar arası fark da artık ölçülür.
- Fallback: AVOA %0.8 vs B0 %2.9 → MAE'ye göre şekillenen kümeler kapsamayı da iyileştiriyor.

## 3. Makalelerin 0.68–0.75'i nasıl okunmalı

| Kaynak | MAE | Not |
|---|---|---|
| Katarya & Verma 2016 (KM-PSO-FCM) | 0.75 | "mevcut 0.78'den %3.5 iyi" iddiası |
| Katarya & Verma 2017 (K-means+Cuckoo) | 0.68 | Ayrıntılı split protokolü makalede belirsiz; tek koşu |
| Surprise SVD (5-fold CV, dürüst) | 0.736 | Ayarlı matris çarpanlarına ayırma |
| Surprise KNNWithMeans | 0.750 | Bizim knn_all ile uyumlu |
| **Bizim AVOA pred-fit + cknn (fold1, s42)** | **0.756** | Resmi split, sızıntısız, tekrarlanabilir |

Kritik gözlem: 0.68'lik küme+kNN sonucu, ayarlı SVD'den (0.736) bile iyi olurdu —
kümeleme komşu uzayını DARALTIR, tam kNN'den (0.747) iyi olması yapısal olarak
beklenmez. Bu değerler ya farklı/raporlanmamış split, ya tek koşu en-iyi-seed, ya da
seçilmiş alt küme ile açıklanır. Tezde bu tablo "sonuçlar neden birebir
karşılaştırılamaz" bölümünün kendisi olur; bizim savunulabilir hedef bandımız
0.74–0.76 ve şu an içindeyiz.

## 4. Kalan gerçek eksikler (0.756 → 0.73 bandına inmek için)

1. **Fitness–tahminci hizalaması:** fitness şu an bias-MAE, değerlendirme cknn —
   fitness da cknn-MAE olursa (pahalı ama mümkün) kazanç artar.
2. **NFE:** pred-fit yalnız 2.000 NFE; 5–10k ile doyum eğrisi.
3. **kNN ayarı:** k=30 sabit; k∈{20,40,60} + shrinkage + Pearson denenmedi.
4. **Soft/fuzzy atama:** sınır kullanıcılar iki kümeden tahmin alabilir
   (Katarya'nın FCM kullanmasının gerçek nedeni muhtemelen bu).
5. **CV5 + 30 seed + kullanıcı-bazlı Wilcoxon:** iddiaları mühürlemek için.

## Kaynaklar

- Katarya & Verma 2016: https://link.springer.com/article/10.1007/s11042-016-3481-4
- Katarya & Verma 2017 (Cuckoo): https://www.sciencedirect.com/science/article/pii/S1110866516300470
- Surprise benchmark: https://surpriselib.com/ ve https://github.com/NicolasHug/Surprise
