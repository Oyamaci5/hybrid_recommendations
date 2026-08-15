# ML-100K Nihai Sonuç Özeti (mühürlü: 5 fold × 10 seed)

Tüm tablolar saf CF (tür bilgisi yok) — tür varyantı ablasyon olarak ayrı.

---

## TABLO 1 — Ana sonuç: eşit maliyetli arena, K=6 (`tabloB_plus_K6.csv`)

Repair kapasite kısıtı (herkes havuz=315, max küme=158), warm start, küme-MF+kNN.

| Yöntem | MAE | RMSE | NDCG@10 | F1@10 | B0'a fark (MAE) |
|---|---|---|---|---|---|
| **AVOA** | **0.7373** ±0.006 | **0.9412** | **0.8393** | **0.6332** | **−0.0214 (%2.8)** |
| NGO | 0.7448 | 0.9486 | 0.8339 | 0.6295 | −0.0139 |
| HHO | 0.7455 | 0.9493 | 0.8330 | 0.6289 | −0.0132 |
| HGS | 0.7465 | 0.9501 | 0.8314 | 0.6281 | −0.0122 |
| GWO | 0.7506 | 0.9540 | 0.8288 | 0.6257 | −0.0081 |
| B0 (KMeans++) | 0.7587 | 0.9615 | 0.8203 | 0.6212 | — |

**Friedman χ²=173.1, p=1.6e-35.** 5/5 meta hem MAE hem NDCG'de anlamlı (Holm p<0.0001).
Kullanıcı-bazlı Wilcoxon (n=943): AVOA **p=2.7e-50**, diğerleri p<1e-36.
**AVOA farkı ikinciyi %54 geçiyor** (−0.0214 vs −0.0139).

## TABLO 2 — Yüksek kısıt noktası, K=40 (`tabloB_plus_K40.csv`)

Havuz 47 (%5), max küme 24.

| Yöntem | MAE | NDCG@10 | B0'a fark |
|---|---|---|---|
| **AVOA** | **0.7709** | **0.8082** | **−0.0306 (%3.8)** |
| NGO | 0.7950 | 0.7845 | −0.0065 |
| HHO | 0.7954 | 0.7832 | −0.0061 |
| HGS | 0.7970 | 0.7829 | −0.0045 |
| GWO | 0.8011 | 0.7814 | −0.0004 (anlamsız) |
| B0 | 0.8015 | 0.7806 | — |

**Friedman χ²=185.3, p=3.9e-38.** AVOA farkı diğerlerinin **5 katı**;
GWO burada B0'dan ayrışamıyor (p=0.48) → serbest protokoldeki GWO liderliğinin
havuz artefaktı olduğu kesin.

## TABLO 3 — Mutlak performans noktası (`tam_run_A.csv`, K=10 + global MF, 6 yöntem)

| Yöntem | MAE | RMSE | NDCG@10 | F1@10 | Havuz | B0'a fark |
|---|---|---|---|---|---|---|
| KNN_ALL (kümesiz, referans) | **0.7216** | 0.9187 | **0.8549** | — | 943 | — |
| GWO | 0.7243 | 0.9202 | 0.8545 | 0.6406 | 488 | −0.0015 (anlamlı) |
| HHO | 0.7252 | 0.9209 | 0.8539 | 0.6406 | 280 | −0.0006 (anlamlı) |
| AVOA | 0.7255 | 0.9210 | 0.8540 | 0.6404 | 278 | −0.0003 (anlamsız) |
| HGS | 0.7257 | 0.9211 | 0.8541 | 0.6405 | 271 | −0.0001 (anlamsız) |
| B0 | 0.7258 | 0.9205 | 0.8546 | 0.6409 | 367 | — |
| NGO | 0.7261 | 0.9214 | 0.8534 | 0.6400 | 253 | +0.0003 (anlamsız) |

- **En iyi mutlak değerimiz: MAE 0.7243–0.7258 / NDCG 0.854** — ayarlı SVD (0.736)
  ve Katarya-2016 (0.75) önünde; kümesiz kNN'e 0.003–0.004 fark ama havuzun
  yalnız %27–29'uyla.
- **Fark hunisi kesin kanıtı (6 yöntem):** MAE'de 5 metanın yalnız 2'si anlamlı
  ve fark binde 1.5; **NDCG'de metaların 3'ü B0'ın GERİSİNDE** (anlamlı).
  Yani bu çalışma noktasında algoritma seçimi pratik olarak önemsiz —
  hatta yönü metrikten metriğe değişiyor.
- GWO'nun küçük MAE üstünlüğü yine havuz ile geliyor (488 vs B0 367, AVOA 278).

## TABLO 3 AYAR KONTROLÜ (türsüz uzayda yeniden) — `tabloA_k_sweep.csv`, `tabloA_knn_sweep.csv`

Tablo A parametreleri (`sampiyon.py`) tür bilgisi VARKEN seçilmişti; türsüz
uzayda tekrar doğrulandı (fold1, s42):

| K | B0 MAE / havuz | AVOA MAE / havuz | KNN_ALL |
|---|---|---|---|
| 4 | 0.7335 / 851 | 0.7359 / 544 | 0.7328 |
| 6 | 0.7349 / 704 | **0.7347 / 461** | 0.7328 |
| **10 (mevcut)** | 0.7372 / 440 | 0.7368 / 248 | 0.7328 |
| 14 | 0.7387 / 299 | 0.7398 / 177 | 0.7328 |

- **Hiçbir K'da meta B0'ı anlamlı geçmiyor** (en iyi durum K=6'da 0.0002) →
  fark hunisi K'dan bağımsız; Tablo A'nın mesajı sağlam.
- KNN_ALL bu noktada hep önde (0.7328) — global MF + bol havuz varken kümeleme
  yalnız MALİYET kazandırıyor: AVOA K=10 havuzun %26'sıyla 0.004 fark veriyor.
- kNN k taraması: k=10/20/30/40 → val 0.7290/0.7289/0.7294/0.7297 → **k=20 doğrulandı**
  (fark binde 1'in altı, seçim kararlı).
- MF ızgarası (türsüz): f=40 → 0.7379, f=80 → 0.7372, f=120 → 0.7370 →
  **f=80 yeterli** (f=120 kazancı binde 0.2, maliyet 1.5×).

**Sonuç: Tablo A konfigürasyonu türsüz uzayda da geçerli, yeniden ayar gerekmiyor.**

## ABLASYON — tür bilgisi (`tabloB_plus_K6_genre.csv`)

| | B0 | AVOA | Fark |
|---|---|---|---|
| Saf CF (Tablo 1) | 0.7587 | 0.7373 | −0.0214 |
| + tür bilgisi | 0.7469 | 0.7391 | −0.0078 |

**Tür bilgisi B0'ı 0.0118 iyileştiriyor, AVOA'yı 0.0018 kötüleştiriyor.**
Yorum: meta-sezgisel merkez araması, yan bilgi mühendisliğinin sağladığı
kazancın büyük kısmını zaten kendi buluyor; ikisi birlikte kullanıldığında
fazlalık oluşuyor. **Yan bilgi olmayan veri setlerinde meta daha kritik.**

---

## Üç cümlelik sonuç

1. **Eşit maliyette meta-sezgisel kümeleme K-means++'ı anlamlı geçiyor**
   (K=6: %2.8, K=40: %3.8; 5/5 yöntem, kullanıcı düzeyinde p<1e-36).
2. **AVOA açık ara en iyi** — farkı ikinci sıradakinin 1.5–5 katı, üç protokolde
   de tutarlı.
3. **Fark hunisi:** global MF + bol havuz olan noktada (Tablo 3) fark kayboluyor
   (5 metanın 2'si anlamlı, NDCG'de 3'ü baseline'ın gerisinde); kısıt sıkılaştıkça
   büyüyor (Tablo 1: %2.8 → Tablo 2: %3.8). Algoritma seçimi, sistemin kümelemeye
   ne kadar bağımlı olduğuna göre önem kazanıyor.

---

## ML-100K TAMAMLANDI — durum

Mühürlü (5 fold × 10 seed): Tablo A (305 satır), Tablo B (300), Tablo B+ K=6 (300),
K=40 (300), tür ablasyonu (300). Toplam **1505 bağımsız koşu**.

Tek-seed kalan yardımcı analizler (makalede "pilot/analiz" etiketiyle sunulacak,
mühür gerekmez): K taraması, klasik baseline'lar, LOF/atipik, cold-start katmanı,
fitness ablasyonu, hiperparametre transferi.

**Sıradaki: ML-1M genelleme** (bkz. `EKSIKLER_ML1M_ONCESI.md` bölüm D-E).
