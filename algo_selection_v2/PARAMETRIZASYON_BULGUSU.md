
# Merkez Parametrizasyonu Neden Gerekli — Deneysel Kanıt ★★★

Soru: "Daha iyi kümeleme merkez bulmaktan mı geçiyor? Atamayı doğrudan
meta-sezgiselle yapsak ne olur? Repair'siz ne olur?"

Deney: ML-100K, K=6, fold 1, seed 42, aynı NFE (1200), aynı tahminci
(0.5·kNN + 0.5·küme-MF). `dogrudan_atama.csv`

---

## A) Doğrudan atama araması (merkez YOK, etiket vektörü aranıyor)

Kodlama: her kullanıcıya sürekli bir sayı x_u ∈ [0,K), etiket = ⌊x_u⌋.
Arama boyutu **943** (merkez aramada 6×20 = 120).

| Yöntem | MAE | NDCG@10 | Havuz | Max/Min küme |
|---|---|---|---|---|
| B0 (referans) | 0.7489 | 0.8402 | 715 | 534 / 2 |
| **AVOA** | 0.7299 | 0.8498 | 905 | **915 / 0** |
| **HHO** | 0.7303 | 0.8550 | **942** | **942 / 0** |
| GWO | 0.7489 | 0.8402 | 715 | 534 / 2 |
| HGS | 0.7489 | 0.8423 | 690 | 517 / 12 |
| NGO | 0.7489 | 0.8402 | 645 | 521 / 13 |

**Bulgu 1 — Dejenerasyon anında geliyor.** HHO 943 kullanıcının **942'sini tek
kümeye** koydu (havuz 942 = tüm veri seti); AVOA 915'ini. Yani "iyi MAE"leri
(0.730) kümelemeyi tamamen iptal etmekten geliyor — kümesiz kNN'i taklit ediyorlar.
Fitness doğru optimize edildi, ama hedefin küresel optimumu "kümeleme yapma".

**Bulgu 2 — Diğer üçü hiç hareket edemedi.** GWO/HGS/NGO B0 ile aynı sonucu
veriyor: 943 boyutlu arama uzayı, 1200 NFE bütçesiyle taranamayacak kadar büyük.
Yani doğrudan atama araması ya dejenere oluyor ya da hiç öğrenemiyor.

**Bulgu 3 — Merkez parametrizasyonu bir düzenlileştiricidir.** Merkez üzerinden
arama, çözümü "geometrik olarak tutarlı" olmaya zorlar: bir kullanıcıyı kümeye
almak, o kümenin merkezine yakın tüm kullanıcıları da almak demektir. Doğrudan
etiket aramasında bu kısıt yok, dolayısıyla hiçbir yapı korunmuyor.

## B) Repair'siz merkez araması (serbest Voronoi, fitness'ta da repair yok)

| Yöntem | MAE | NDCG@10 | Havuz | Max/Min küme |
|---|---|---|---|---|
| B0 | 0.7489 | 0.8402 | 715 | 534 / 2 |
| **AVOA** | **0.7305** | **0.8541** | **937** | **941 / 0** |
| GWO | 0.7452 | 0.8421 | 694 | 548 / 2 |
| NGO | 0.7509 | 0.8374 | 423 | 315 / 2 |
| HHO | 0.7521 | 0.8365 | 381 | 320 / 9 |
| HGS | 0.7529 | 0.8362 | 396 | 231 / 6 |

**Bulgu 4 — AVOA repair'siz de dejenere oluyor** (941/0, havuz 937 = %99).
En düşük MAE'yi veriyor ama kümeleme yapmıyor. Diğerleri kümelemeyi koruyor
(havuz 381–694) ama MAE'leri kötü.

**Bulgu 5 — Repair'siz sıralama anlamsız:** MAE'ye göre AVOA 1., ama o
kümelemeyi iptal etmiş; HGS son, ama en dengeli kümelemeyi yapmış (231/6).
Yani **kısıt olmadan MAE ile algoritma kıyaslamak, "kim kümelemekten en çok
kaçındı" yarışına dönüşüyor.**

---

## Karşılaştırmalı özet (K=6, aynı bütçe, aynı tahminci)

| Protokol | AVOA MAE | AVOA havuzu | Kümeleme korunuyor mu? |
|---|---|---|---|
| Doğrudan atama | 0.7299 | 905 (%96) | ✗ dejenere |
| Merkez, repair'siz | 0.7305 | 937 (%99) | ✗ dejenere |
| **Merkez, repair'li** | **0.7450** | **315 (%33)** | **✓ dengeli (158/153)** |

Repair'li protokolde MAE 0.015 daha yüksek — **ama havuz 3 kat küçük ve
kümeleme gerçekten yapılıyor.** Diğer ikisinde ölçülen şey kümeleme değil,
"kümelemeden kaçma becerisi".

## Tez için sonuç cümleleri

1. *"Kümeleme kalitesi merkez konumlarıyla parametrize edilmelidir; doğrudan
   atama araması hem arama uzayını N boyuta çıkarır hem de yapısal bir
   düzenlileştirici içermediğinden dejenere çözümlere yakınsar (943 kullanıcının
   942'si tek kümede)."*
2. *"Kapasite kısıtı olmadan doğruluk temelli uygunluk fonksiyonu, kümelemenin
   iptalini ödüllendirir; bu nedenle kısıt, adil karşılaştırmanın yanı sıra
   problemin iyi tanımlı kalması için de gereklidir."*
3. *"Kısıtsız protokollerde gözlenen düşük hata değerleri, kümeleme başarısının
   değil kümelemeden kaçınmanın sonucudur; küme boyut dağılımı raporlanmadığında
   bu ayrım görünmez."*

Bu üç cümle, incelenen literatürün küme boyut dağılımı raporlamaması eleştirimizin
en güçlü dayanağıdır.
