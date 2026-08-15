
# HSC Replikasyonu — Neden Aynı Sonuçları Alamıyoruz?

HSC (Symmetry 2022, 14, 793) protokolünü birebir kurduk (`hsc_replikasyon.py`):
%80/%20 bölme, ham 943×1682 matris, [1,5] rastgele merkezler, Öklid, WCSS fitness,
serbest atama, küme-ortalaması tahmini, K=70.

| | HSC bildirimi | Bizim replikasyon |
|---|---|---|
| MAE (ML-100K, K=70) | **0.685** | **0.858** |
| RMSE | 1.220 | 1.089 |
| Precision | 0.602 | 0.558 |
| Küme dağılımı | raporlanmamış | max 184, **min 1** |

Fark %25. Aşağıda, kanıtlarla desteklenen dört açıklama adayı — hiçbiri
"hata" iddiası değil, **raporlanan bilgiyle sonuçların tutarsızlığı**.

---

## 1. MAE tanımı standart değil

Makalenin Denklem (7)'si:

> MAE = |p_ij − t_ij| / **M**, *"M veri setindeki film sayısıdır"*

Standart MAE, **tahmin sayısına** bölünür (ML-100K'da 20.000 test puanı).
M = 1682 (film sayısı) ile bölmek farklı bir ölçek verir. Payda 20.000 yerine
1682 olursa değerler ~12 kat büyür; kullanıcı başına normalize edilirse çok
küçülür. Hangi yorumla olursa olsun **bizim hesapladığımız MAE ile aynı şey
değil** — dolayısıyla sayılar doğrudan karşılaştırılamaz.

## 2. RMSE/MAE oranı iç tutarsız

Aynı tablodan (Tablo 5, ML-100K):

| Yöntem | MAE | RMSE | RMSE/MAE |
|---|---|---|---|
| NMF | 0.758 | 0.963 | **1.27** |
| SVD | 0.737 | 0.934 | **1.27** |
| Firefly | 0.695 | 1.229 | **1.77** |
| Cuckoo | 0.697 | 1.231 | 1.77 |
| Whale | 0.691 | 1.228 | 1.78 |
| **Sparrow (önerilen)** | **0.685** | **1.220** | **1.78** |

1–5 ölçeğinde bu oran tipik olarak 1.25–1.35 arasındadır (bizim tüm
ölçümlerimizde 1.27–1.33). NMF/SVD satırları bu bandın içinde; **sürü tabanlı
satırların hepsi 1.78** — yani o satırlarda MAE ile RMSE aynı tahminlerden
hesaplanmamış görünüyor. Sürü yöntemlerinin "düşük MAE"si bu yüzden şüpheli:
aynı tahminler RMSE'de en kötü performansı veriyor.

## 3. MAE'nin K ile monoton düşmesi — sızıntı imzası

Tablo 4 (ML-100K): K=10 → 0.785, K=20 → 0.776, ..., **K=70 → 0.695**.
Recall 0.329 → 0.604, Precision 0.311 → 0.552 (hepsi monoton iyileşiyor).

Bu davranış küme-ortalaması tahmincisiyle **beklenmez**: K büyüdükçe küme
küçülür, ortalama daha az puandan hesaplanır, gürültü artar. Bizim ölçümümüz
(ML-1M, aynı tahminci): K=6 → 0.807, K=40 → 0.807 civarı, K büyüdükçe
**kötüleşiyor**; ML-100K replikasyonunda K=70'te 0.858.

Monoton iyileşmenin bilinen tek mekanizması: **kümeleme, test puanlarını da
içeren tam matriste yapılmışsa.** K büyüdükçe kümeler tekilleşir (bizim
replikasyonda min küme = 1 kişi), kullanıcının kendi puanı küme ortalamasına
hâkim olur ve "tahmin" kendi kendini tahmin etmeye yaklaşır. Makale, kümelemenin
yalnız eğitim matrisinde yapıldığını **açıkça belirtmiyor**: *"The rating matrix
containing the rating data is clustered into K clusters."*

## 4. Arama uzayı, optimizasyonun anlamlı çalışamayacağı büyüklükte

HSC'de her merkez **film sayısı kadar** boyuta sahip:

| | Arama uzayı boyutu |
|---|---|
| HSC (K=70, ML-100K) | 70 × 1682 = **117.740** |
| HSC (K=70, ML-1M) | 70 × 3952 = **276.640** |
| Bizim (K=70, NMF-20) | 70 × 20 = **1.400** |

Makale popülasyon ve iterasyon sayısını vermiyor. 100 sparrow × 100 iterasyon
= 10.000 değerlendirme olsa bile, 117.740 boyutlu uzayda boyut başına 0.08
örneklem düşer — bu, optimizasyonun rastgele aramadan ayırt edilemeyeceği bir
rejimdir. Nitekim bizim replikasyonumuzda SSA, K-means++ merkezlerinden daha
iyi bir WCSS bulamıyor.

---

## Sonuç: bu bir "kusur avı" değil, metodolojik gerekçe

Replikasyon başarısızlığının kaynağı bizim tarafımızda olabilir mi? Kontrol ettik:
- Bölme oranı, matris boyutu, doldurma (0), mesafe (Öklid), fitness (WCSS),
  tahminci (küme-ortalaması), K=70 — hepsi makaledeki gibi.
- Kendi hattımızın doğruluğu bağımsız olarak doğrulanmış durumda (kümesiz
  kNN'imiz 0.7467, Surprise'ın yayınlanmış 0.750 değeriyle uyumlu).

Dolayısıyla makul sonuç: **HSC'nin bildirdiği sayılar, makalede tarif edilen
protokolden üretilemiyor.** Bu, tezimizin üç metodolojik kararını doğrudan
gerekçelendiriyor:

1. **Küme boyut dağılımı raporlanmalı** — min küme = 1 kişi olduğunda
   "küme-ortalaması" kavramı anlamını yitiriyor; okuyucu bunu göremiyor.
2. **Sızıntı bariyerleri açıkça yazılmalı** — hangi matrisin hangi aşamada
   kullanıldığı belirsizse, K ile monoton iyileşme gibi anomaliler
   açıklanamaz kalıyor.
3. **Arama uzayı boyutu ve bütçe (NFE) raporlanmalı** — ham matriste arama,
   meta-sezgiseli fiilen devre dışı bırakıyor; bu yüzden biz NMF ile 20 boyuta
   iniyoruz (84–198 kat küçük uzay).

## Makaleye giren cümle (ölçülü dil)

> "HSC'nin bildirdiği sonuçlar, makalede tarif edilen protokolle yeniden
> üretilememiştir (bildirilen MAE 0.685; replikasyon 0.858). Bildirilen MAE
> tanımının standarttan farklı olması (payda: film sayısı), sürü tabanlı
> satırlarda RMSE/MAE oranının diğer satırlardan belirgin sapması (1.78 vs 1.27)
> ve hata metriğinin küme sayısıyla monoton azalması, sonuçların doğrudan
> karşılaştırılmasını güçleştirmektedir. Bu gözlemler, çalışmamızda küme boyut
> dağılımının, veri bölme bariyerlerinin ve arama uzayı boyutunun açıkça
> raporlanması kararının gerekçesini oluşturmaktadır."

## Ek: aynı analiz Firefly makalesi için de geçerli

Firefly makalesi (K=90) HSC ile **aynı formülleri, aynı tablo yapısını ve aynı
SD değerlerini** (0.113–0.123) kullanıyor. Tablo 5'teki Firefly satırı
(MAE 0.695, RMSE 1.229) ile Firefly makalesinin kendi sonuçları arasında
tutarlılık kontrolü yapılmalı — iki çalışma büyük olasılıkla ortak bir şablondan
türemiş.
