# Danışman Raporu — Önerilen Yöntem ve Mevcut Durum

## 1. Önerilen yöntem: PACR (Prediction-Aligned Center Refinement)

Küme tabanlı CF'de merkezler klasik olarak K-means'le (WCSS hedefi) bulunur.
Gösterdik ki (i) WCSS hedefinde KMeans++ yenilmez, (ii) ama nihai sistemin hedefi
WCSS değil, tahmin doğruluğudur ve Lloyd bu hedefi optimize edemez (kapalı form yok).

**PACR:** dengeli K-means merkezlerinden başla (warm start) → meta-sezgisel,
merkezleri doğrudan iç-validasyon karışım-MAE'sine göre rafine etsin (eşit küme
boyutu cezasıyla) → aynı komşu-havuz bütçesiyle test et. Sistem zinciri:
NMF-20+tür profili uzayı → PACR merkezleri → soft top-2 havuz → küme-içi kNN (k=20)
→ ALS-MF (f=80) ile β=0.4 karışım.

### Sonuç (ML-100K resmi fold 1, sızıntısız, seed 42) — `results/pacr_p3.csv`

| Yöntem | Test MAE | NDCG@10 | P@10 | Havuz |
|---|---|---|---|---|
| Dengeli K-means (baseline) | 0.7408 | 0.8404 | 0.6991 | 197 |
| PACR-AVOA | 0.7404 | **0.8470** | 0.7065 | 194 |
| PACR-HHO | 0.7404 | 0.8445 | — | 187 |
| PACR-HGS | 0.7395 | 0.8447 | — | 188 |
| PACR-GWO | 0.7397 | 0.8446 | — | 198 |
| **PACR-NGO** | **0.7391** | 0.8457 | — | 187 |

- **5/5 meta-sezgisel, baseline'ı hem MAE hem NDCG'de geçti** — eşit veya daha
  küçük havuzla (maliyet kontrollü).
- Algoritmalar arası sıralama tutarlı farklılaşıyor → kümeleme kalitesi
  algoritmaya bağlı (tez sorusunun cevabı).
- Mutlak konum: MAE 0.739 bandı; ayrıca kırpmalı bütçeli varyantla 0.7369
  (`budget_pool.csv`) — resmi split'te raporlayan küme tabanlı literatürün önünde
  (Katarya 0.75, Firefly 0.76–0.80), ayarlı SVD'ye (0.736) 0.001–0.003 mesafede.

## 2. Literatürden daha iyi yaptıklarımız

1. **Resmi 5-fold split + sızıntısız protokol** (fitness iç-val'de; test tek atış).
   İncelenen 6 makalenin hiçbirinde ikisi birden yok (70:30/80:20 tek split).
2. **NFE-eşit bütçe:** algoritmalar epoch değil fonksiyon-çağrısı bütçesiyle
   yarışıyor (aynı epoch'ta 5 kat fark ölçtük). Literatürde hiç yapılmıyor.
3. **Güçlü baseline:** KMeans++ n_init=10 + dengeli varyantı + kümesiz tam kNN
   üst sınırı. Makaleler zayıf random-init k-means'e karşı raporluyor; onların
   +%6–8'lik iyileşmesini zayıf baseline'a karşı biz de üretiyoruz, güçlüsüne
   karşı kaybolduğunu gösteriyoruz (kritik replikasyon).
4. **Küme boyut dağılımı raporu:** kısıtsız doğruluk-fitness'ının "tek dev küme"
   hilesine çöktüğünü gösterdik; boyut raporlamayan makale sonuçları bu tuzağa açık.
5. **Maliyet ekseni (komşu havuzu %):** doğruluk-maliyet cephesi grafiğiyle kıyas;
   literatür yalnız doğruluk raporluyor, havuz büyüklüğü gizli serbestlik.
6. **Çok metrik:** MAE/RMSE + P@10/R@10/NDCG@10 + fallback; ve istatistik planı
   (Friedman+Holm, kullanıcı-bazlı Wilcoxon).

## 3. İncelenen makalelerin eksikleri (özet; detay: INCELEME_6_MAKALE.md)

| Eksik | Kimde |
|---|---|
| Resmi fold yok, tek split | 6/6 |
| Tek koşu, seed/istatistik yok | 6/6 |
| Küme boyut dağılımı raporsuz | 6/6 |
| Güçlü baseline yok | 6/6 |
| Şüpheli düşük MAE (0.68 / 0.50–0.56) | GWO-FCM, HHO-K-means |
| Küme içi kişiselleştirmesiz öneri | HSC (Sparrow) |

## 4. Katkı cümleleri (tez/makale)

1. PACR: Lloyd'un optimize edemediği tahmine-hizalı hedefte, dengeli kısıt altında
   meta-sezgisel merkez rafinasyonu — 5 farklı algoritmayla baseline üstü sonuç.
2. Kritik replikasyon: meta-kümeleme iyileşme iddialarının baseline gücüne ve
   havuz bütçesine bağlılığının deneysel kanıtı (üç adalet protokolü).
3. Dejenerasyon analizi: kısıtsız doğruluk-fitness'ının kümelemeyi yok ettiğinin
   gösterimi ve denge kısıtının gerekliliği.
4. Küme-kNN + MF karışımı: kümesiz kNN üst sınırını tüm metriklerde aşan,
   maliyeti 1/3–1/4'e indiren hibrit (literatürdeki 6 çalışmanın hiçbirinde yok).

## 5. Sıradaki adım (onay bekliyor)

"Gerçek koşu": PACR-{NGO, AVOA, HGS} + dengeli-B0 + kümesiz kNN referansı,
10 seed × 5 resmi fold, kullanıcı-bazlı Wilcoxon + Friedman/Holm → makale ana
tablosu. Süre ~2–4 saat (sizin makinede, komut hazır).
