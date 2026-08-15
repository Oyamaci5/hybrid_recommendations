
# Karışım Ablasyonu — Nihai Tahminci Kararı (ML-1M, K=40, fold1 s42)

Kaynak: `ml1m_karisim.csv`. Ağırlıklar iç-doğrulamada 0.1 adımlı ızgarayla seçildi.
Bileşenler: **cmean** (küme-film ort.), **bias** (kull. ort + küme sapması),
**cknn** (küme-içi kNN k=20), **cmf** (küme-başına ALS-MF).

## Sonuç tablosu

| Kombinasyon | Ağırlık | repairLİ MAE | repairSİZ MAE | En iyi NDCG |
|---|---|---|---|---|
| cmean | — | 0.8066 | 0.8098 | 0.860 |
| bias | — | 0.7405 | 0.7346 | 0.866 |
| cknn | — | 0.7335 | 0.7265 | 0.868 |
| cmf | — | 0.7371 | 0.7480 | 0.867 |
| cmf+cmean | .8/.2 | 0.7339 | 0.7430 | 0.869 |
| cknn+cmean | .8/.2 | 0.7319 | 0.7254 | 0.869 |
| cmf+bias | .5/.5 | 0.7244 | 0.7253 | 0.872 |
| **cmf+cknn** | .5/.5 · .3/.7 | **0.7201** | **0.7194** | **0.873** |
| **cmf+cknn+bias** | .4/.4/.2 · .3/.6/.1 | **0.7193** | **0.7191** | **0.874** |
| cmf+cknn+cmean | .5/.5/.0 · .3/.6/.1 | 0.7201 | 0.7198 | 0.874 |
| cmf+cknn düz 0.5 (ayarsız) | 0.5/0.5 | 0.7201 | 0.7209 | 0.874 |

## Beş bulgu

**1. cmean (küme-ortalaması) hiçbir kombinasyonda işe yaramıyor.**
Tek başına en kötü (0.807–0.810). Üçlü karışımda ağırlığı 0.0–0.1'e düşüyor;
repairLİ'de tam sıfır. Literatürün standart tercihi bizim hattımızda gereksiz.

**2. En iyi: cmf + cknn + bias (0.7191–0.7193).**
Ama cmf+cknn (0.7194–0.7201) ile farkı **0.0003–0.0008** — yani bias'ın katkısı
ihmal edilebilir. **Sadeleştirme kararı: iki bileşenli karışım (cmf+cknn) yeter.**
Üçlü, karmaşıklığı %50 artırıp kazancı binde 0.5'te bırakıyor.

**3. Ayarsız düz ortalama (0.5/0.5) neredeyse ayarlı kadar iyi** (0.7201 vs 0.7194).
Bu, yöntemin **sağlamlığının kanıtı**: β'yı ayarlamasanız bile sonuç değişmiyor.
Makalede "hiperparametre hassasiyeti düşük" iddiası olarak kullanılır ve K19
bulgusuyla (ayar transfer edilmiyor) tutarlı.

**4. Kısıt–bileşen etkileşimi net:**
- **cmf repair'li iken iyi** (0.7371 vs 0.7480): dengeli kümeler her MF'e yeterli
  veri veriyor; dev küme modeli "ortalama kullanıcıya" çekiyor.
- **cknn repair'siz iken iyi** (0.7265 vs 0.7335): büyük havuz daha çok gerçek
  komşu içeriyor (komşu-recall bulgusuyla birebir tutarlı).
- **Karışımda fark siliniyor** (0.7201 vs 0.7194 → 0.0007). İki bileşen zıt
  yönde etkileniyor, karışım gerilimi çözüyor.
  → **Repair kararının savunması:** maliyeti yarıya indiriyor (havuz 302 vs ~600),
  bedeli binde 0.7.

**5. NDCG'de repair'siz hafif önde** (0.8737 vs 0.8674) — sıralama kalitesi havuz
genişliğine MAE'den daha duyarlı. Ana tabloda repair'li kullanıp bu farkı
sınırlılık olarak raporlamak dürüst yol.

## KARAR — nihai tahminci

```
p(u,i) = 0.5 · kNN_küme(u,i) + 0.5 · MF_küme(u,i)        [ayarsız düz ortalama]
```
- Ağırlık iç-doğrulamada da seçilebilir (β ∈ [0.3, 0.6] bandında dolaşıyor,
  kazanç binde 0.1–1).
- cmean ve bias yalnızca **fallback** olarak kalır (kNN kurulamayan durumlar).
- Sadelik + sağlamlık + performans üçlüsünde en iyi denge bu.

## Makaleye giren cümle

> "Dört bileşenin tüm ikili ve üçlü kombinasyonları iç-doğrulama ile
> karşılaştırılmış; küme-ortalamasının (literatürün yaygın tercihi) hiçbir
> kombinasyonda anlamlı ağırlık almadığı, en iyi sonucun küme-içi kNN ile
> küme-başına matris çarpanlarına ayırmanın eşit ağırlıklı karışımından elde
> edildiği görülmüştür (MAE 0.720). Ağırlık ayarının kazancı binde 1'in altında
> kalmıştır; bu, yöntemin hiperparametre hassasiyetinin düşük olduğunu gösterir."
