# ML-1M Öncesi Eksikler Envanteri (ML-100K durumu)

Tarih: bu oturum. Amaç: ML-1M'e geçmeden ML-100K tarafında neyin mühürlü,
neyin tek-seed pilot olduğunu netleştirmek.

## A) MÜHÜRLÜ (5 fold × 10 seed = 50 hücre) ✓

| Sonuç | Dosya | Durum |
|---|---|---|
| Tablo B — serbest protokol, K=30 | `tam_run_B.csv` (300 satır) | ✓ 5/5 meta anlamlı (Friedman p=6e-37) |
| **Tablo B+ — repair, K=6** | `tabloB_plus_K6.csv` (299 satır) | ✓ **AVOA 0.7391 / B0 0.7469; Friedman p=5e-20** |

**K=6 mühürlü sonuç (yeni):**

| Yöntem | MAE | NDCG@10 | F1@10 | Havuz |
|---|---|---|---|---|
| **AVOA** | **0.7391** ±0.008 | **0.8382** | **0.6324** | 315 |
| GWO | 0.7441 | 0.8372 | 0.6311 | 314 |
| HHO | 0.7450 | 0.8362 | 0.6303 | 314 |
| HGS | 0.7455 | 0.8350 | 0.6296 | 315 |
| NGO | 0.7458 | 0.8353 | 0.6298 | 315 |
| B0 | 0.7469 | 0.8335 | 0.6288 | 315 |

- Hücre-bazlı Wilcoxon: **5/5 meta hem MAE hem NDCG'de anlamlı** (Holm p≤0.013).
- Kullanıcı-bazlı (n=943): AVOA p=1.3e-07, GWO p=0.001, NGO p=0.012 anlamlı;
  HHO/HGS anlamsız → **AVOA'nın üstünlüğü kullanıcı düzeyinde de sağlam**.
- AVOA farkı diğer metaların ~3 katı (−0.0078 vs −0.0012…−0.0029).

## B) EKSİK — ML-1M'den ÖNCE tamamlanmalı ⚠

| # | Eksik | Neden gerekli | Maliyet |
|---|---|---|---|
| E1 | **Tablo A tamamlanmamış** (`tam_run_A.csv` 20/105 satır, sadece fold 1) | Mutlak performans + literatür kıyası tablosu; KNN_ALL referansı da eksik | ~45 dk |
| E2 | **K=40 mühürlemesi yok** (`tabloB_plus_K40.csv` yok) | İkinci çalışma noktası; "fark maksimum" iddiası tek seed'e dayanıyor | ~2.5 sa |
| E3 | Klasik baseline'lar tek seed (`klasik_baselines.csv`, fold1 s42) | SOM/PCA-kmeans kıyası makale tablosuna girecek → çoklu seed şart | ~1 sa |
| E4 | K taraması tek seed (`k_tarama_repair.csv`) | "K=6 ve K=40 seçimi" gerekçesi; en az 3 seed × 2 fold | ~1.5 sa |
| E5 | Cold-start katmanları tek fold | Katmanlı sonuç 5 fold'a yayılmalı (özellikle cold katmanı n=126) | ~20 dk (mevcut koşulardan türetilebilir) |
| E6 | Hibrit fitness (K16) tam koşusu yok | "Hibrit üç metrikte en iyi" iddiası 1 seed | ~2 sa (opsiyonel) |

**Öncelik sırası: E1 → E2 → E3/E4 → (E5) → (E6).**
E1 ve E2 olmadan makale ana tabloları eksik kalır; E3/E4 hakem sorularına karşı.

## C) TAMAM — tekrar gerekmez ✓

- Algoritma eleme (22 algo, NFE-eşit, 30 seed) — `final_run.csv`, `pure_meta_summary.csv`
- Fitness ablasyonu (WCSS / pred-MAE / NDCG / hibrit) — pilot yeterli, yön belli
- Hiperparametre ayarı — ayarlamama kararı gerekçeli (val↔test r=−0.50)
- Kısıt sertliği ablasyonu (1.0×/1.5×/3.0×/serbest) — K17/K20
- Yer değiştirme analizi (%100 telafi) — K21
- Atipik/LOF yönlendirme deneyi — K23
- Protokol savunması + literatür dayanağı — `PROTOKOL_SAVUNMASI.md`

## D) ML-1M için gereken teknik hazırlık

1. `Ctx` sınıfına 1M yükleyici (6040×3952); resmi fold yok → rastgele %90/10 × 5 tekrar.
2. Bellek: benzerlik matrisi 6040² float64 ≈ 292 MB — tek seferlik, kabul edilebilir.
   Gerekirse float32'ye düşür (146 MB).
3. Küme-MF: K=6'da küme başına ~1000 kullanıcı × 3952 film → ALS süresi artar;
   f=10 ile ~3-4 kat yavaş. K taramasını 1M'de yeniden yapmak gerekir
   (havuz = 2N/K olduğundan K=6 artık 2000 kişilik havuz demek — çok pahalı;
   1M için K≈30-60 bandı mantıklı).
4. Tür bilgisi: ML-1M'de `movies.dat` içinde tür var, aynı profil çıkarılabilir.
5. Beklenti: 1M'de havuz kısıtı daha bağlayıcı → **algoritma farkının büyümesi**
   beklenir (K18'deki "havuz küçüldükçe fark büyür" eğilimi).

## E) ML-1M'de koşulacak minimum set (öneri)

Tam replikasyon değil, **hedefli doğrulama**:
1. K taraması (K = 10, 20, 40, 60) × B0 + AVOA, 3 seed × 2 fold → çalışma noktası
2. Seçilen K'da 6 yöntem × 5 seed × 3 fold → ana tablo + istatistik
3. Cold-start katmanlı analiz (1M'de cold kullanıcı sayısı çok daha fazla)
4. (Ops.) klasik baseline'lar aynı noktada

Bu, "yöntem küçük veriye özgü değil" iddiasını kanıtlamaya yeter.
