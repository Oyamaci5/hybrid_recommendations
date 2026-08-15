
# A Listesi Sonuçları (İş 1–3 tamamlandı)

---

## İŞ 1 — ML-1M K=6 mühürlendi (3 fold × 5 seed = 15 hücre)

| Yöntem | MAE | ±std | RMSE | NDCG@10 | P@10 | Sıra |
|---|---|---|---|---|---|---|
| **AVOA** | **0.6887** | 0.0018 | 0.8791 | 0.8867 | 0.5323 | 2.60 |
| **HHO** | **0.6887** | 0.0016 | **0.8789** | 0.8869 | **0.5326** | **2.47** |
| NGO | 0.6889 | 0.0015 | 0.8795 | **0.8871** | 0.5325 | 2.53 |
| HGS | 0.6893 | 0.0013 | 0.8794 | 0.8862 | 0.5321 | 3.67 |
| GWO | 0.6902 | 0.0013 | 0.8806 | 0.8851 | 0.5320 | 4.87 |
| B0 | 0.6903 | 0.0017 | 0.8805 | 0.8829 | 0.5312 | 4.87 |

**Friedman χ²=28.2, p=3.4e-05.** B0'a karşı Holm: AVOA (p=0.002), HHO (p=0.011),
NGO (p=0.008) **anlamlı**; HGS (p=0.071) ve GWO (p=0.561) değil.

**Yorum:**
- Bol bütçe noktasında (havuz %33) bile 3 meta B0'ı anlamlı geçiyor — ama fark
  binde 1.6 (K=40'ta binde 16.6, yani **10 kat küçük**). Fark hunisi bir kez daha.
- **AVOA'nın üstünlüğü burada kayboluyor:** HHO/NGO ile istatistiksel olarak
  ayrışmıyor (hepsi 0.6887–0.6889). Sıralamada HHO 1.
- **RMSE 0.8789** — SVD referansını (0.8761) neredeyse yakaladık; MAE'de
  (0.6887 vs 0.6909) zaten öndeyiz. Literatür kıyas tablosu için en güçlü satır.
- GWO yine B0'dan ayrışamıyor (üçüncü veri noktası; artık kalıcı bir bulgu).

**Tez için:** K=6 tablosu "mutlak performans + literatür kıyası", K=40 tablosu
"algoritma seçimi önemlidir" iddiasını taşır. İkisi birlikte fark hunisini gösterir.

---

## İŞ 2 — kNN komşu sayısı (k) taraması

**K=6 (havuz 2013):**

| k | B0 MAE | AVOA MAE | Fark | AVOA NDCG |
|---|---|---|---|---|
| 5 | 0.6964 | 0.6934 | −0.0030 | 0.8818 |
| 10 | 0.6907 | 0.6879 | −0.0029 | 0.8875 |
| **20** | **0.6890** | **0.6862** | −0.0028 | 0.8893 |
| 30 | 0.6892 | 0.6865 | −0.0027 | **0.8897** |
| 50 | 0.6900 | 0.6876 | −0.0024 | 0.8892 |
| 70 | 0.6907 | 0.6884 | −0.0023 | 0.8896 |

**K=40 (havuz 302):**

| k | B0 MAE | AVOA MAE | Fark |
|---|---|---|---|
| 5 | 0.7242 | 0.7139 | −0.0103 |
| 10 | 0.7203 | 0.7099 | −0.0104 |
| **20** | **0.7201** | **0.7097** | **−0.0104** |
| 30 | 0.7207 | 0.7104 | −0.0103 |
| 50 | 0.7218 | 0.7119 | −0.0099 |
| 70 | 0.7225 | 0.7129 | −0.0096 |

**Üç bulgu:**
1. **k=20 her iki çalışma noktasında da optimum** (K=6'da 0.6862, K=40'ta 0.7097).
   Katarya'nın (2016) 15–20 optimum bulgusu bizim verimizde doğrulandı.
2. **Eğri çok yayvan:** k=10–50 arası fark binde 2'nin altında. Yani k seçimi
   kritik değil → hiperparametre hassasiyeti düşük (K19 bulgusuyla tutarlı).
3. **AVOA'nın üstünlüğü k'dan bağımsız:** fark tüm k değerlerinde neredeyse sabit
   (K=40'ta −0.0096 ile −0.0104 arası). Yani sonuç "şanslı k seçimi"nden gelmiyor.

---

## İŞ 3 — Kapsama / çeşitlilik metrikleri (ML-1M, K=40)

| Yöntem | MAE | NDCG@10 | Katalog kapsama | Farklı film | Gini | Novelty | Fallback% |
|---|---|---|---|---|---|---|---|
| AVOA | **0.7069** | **0.8797** | 0.661 | 2612 | 0.708 | 10.30 | **0.29** |
| HHO | 0.7162 | 0.8690 | 0.673 | 2661 | 0.699 | 10.33 | 1.40 |
| HGS | 0.7178 | 0.8677 | 0.673 | 2660 | 0.699 | 10.33 | 1.31 |
| NGO | 0.7193 | 0.8654 | 0.674 | 2663 | 0.698 | 10.33 | 1.36 |
| B0 | 0.7216 | 0.8661 | 0.680 | 2688 | 0.698 | 10.33 | 1.71 |
| GWO | 0.7222 | 0.8650 | **0.684** | **2701** | 0.699 | 10.33 | 1.72 |

**Bulgular:**
1. **Doğruluk–kapsama ödünleşimi var:** AVOA en iyi MAE/NDCG'yi verirken en düşük
   katalog kapsamasına sahip (%66.1 vs GWO %68.4). Yani daha isabetli ama biraz
   daha dar bir film yelpazesi öneriyor. Fark küçük (76 film, %2.9) ama gerçek.
2. **Fallback dramatik düşük: %0.29 vs B0 %1.71** (6 kat az). AVOA'nın kümeleri
   test filmlerini çok daha iyi örtüyor — komşu-recall bulgusuyla tutarlı.
3. Novelty ve Gini pratikte aynı (10.30 vs 10.33; 0.708 vs 0.698) → AVOA
   "popüler filmlere kaçarak" kazanmıyor. Bu önemli bir sağlamlık kontrolü.

**Tez için dürüst cümle:** *"Önerilen yöntem doğruluk ve sıralama kalitesinde
üstünken katalog kapsamasında %2.9 gerilemektedir; bu, kümelerin daha keskin
tanımlanmasının doğal sonucudur ve çeşitlilik öncelikli senaryolarda ödünleşim
olarak değerlendirilmelidir."*

---

## Güncellenen durum

| Veri | Nokta | En iyi | MAE | RMSE | NDCG | Mühür |
|---|---|---|---|---|---|---|
| ML-100K | K=6 | AVOA | 0.7373 | 0.9412 | 0.8393 | ✓ 50 hücre |
| ML-100K | K=40 | AVOA | 0.7709 | — | 0.8082 | ✓ 50 hücre |
| **ML-1M** | **K=6** | **AVOA/HHO** | **0.6887** | **0.8789** | **0.8869** | **✓ 15 hücre** |
| ML-1M | K=40 | AVOA | 0.7068 | 0.9062 | 0.8783 | ✓ 15 hücre |

Referanslar: SVD-MBCF 0.6909/0.8761 · kümesiz kNN (ML-100K) 0.7467 ·
Katarya-2016 0.75 · Firefly 0.76–0.80

## Kalan işler
- İş 4: varyant denemeleri (`ml1m_varyant.csv` henüz boş — koşu sürüyor olabilir)
- İş 5: tahminci ablasyonu çok-seed
- İş 6: ML-1M cold-start katmanlı analiz
