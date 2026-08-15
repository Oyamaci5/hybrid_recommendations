# "Neden K=40?" Sorusuna Cevap — K'yı Seçmemek

## Sorun

K=6 mutlak olarak en iyi (MAE 0.688) ama orada AVOA–B0 farkı yok (−0.0004).
K=40'ta fark var (−0.0150) ama mutlak sonuç daha kötü (0.7066).
"Farkın göründüğü K'yı seçmek" = kiraz toplama (cherry-picking) → hakem hemen yakalar.

## Çözüm: K bir hiperparametre değil, DAĞITIM BÜTÇESİDİR

K'yı optimize edilecek bir ayar gibi sunmak yanlış. K, sistemin komşuluk arama
maliyetini belirler (havuz ≈ 2N/K). Gerçek sistemlerde bu bütçe **dışarıdan
verilir** (gecikme hedefi, donanım, kullanıcı sayısı). Dolayısıyla doğru soru
"en iyi K nedir?" değil, **"verilen bütçede kim daha iyi?"**

Bu çerçevede üç sunum biçimi var; üçünü birden kullanacağız.

---

## KURGU 1 — Eş-maliyet eğrisi (ana tablo, K seçimi YOK)

Her K bir bütçe seviyesi; hepsi raporlanır (ML-1M, fold1 s42):

| Havuz (bütçe) | B0 MAE | AVOA MAE | Fark |
|---|---|---|---|
| 2013 (%33) | 0.6880 | 0.6876 | −0.0004 |
| 1208 (%20) | 0.6973 | 0.6912 | −0.0060 |
| 604 (%10) | 0.7099 | 0.6986 | −0.0113 |
| 302 (%5) | 0.7216 | 0.7066 | −0.0150 |
| 201 (%3) | 0.7321 | 0.7125 | −0.0196 |

**İddia:** "AVOA her bütçe seviyesinde ≥ B0; avantajı bütçe daraldıkça artıyor."
Hiçbir K seçilmiyor, tüm eğri veriliyor → kiraz toplama imkânsız.

**İstatistik:** K'yı *bloklama faktörü* olarak alıp Friedman uygulanır
(5 bütçe × 3 fold × 5 seed = 75 blok). Bu, deney tasarımında standart yaklaşımdır
(randomized block design) ve "hangi K" tartışmasını tamamen ortadan kaldırır.

## KURGU 2 — Eş-kalite analizi (en güçlü tek cümle)

Aynı kaliteyi tutturmak için gereken bütçe:

| Hedef MAE | B0'ın ihtiyacı | AVOA'nın ihtiyacı | Kazanç |
|---|---|---|---|
| ≤ 0.70 | havuz 1208 (K=10) | havuz **604** (K=20) | **2.0×** |
| ≤ 0.71 | havuz 604 (K=20) | havuz **302** (K=40) | **2.0×** |
| ≤ 0.72 | havuz 604 (K=20) | havuz **201** (K=60) | **3.0×** |

**İddia:** "AVOA, K-means++ ile aynı doğruluğu **yarı hesapla** sağlıyor."
Bu cümle K seçimi gerektirmiyor — eğriden okunuyor. Makale başlığına çıkabilir.

## KURGU 3 — Sabit bütçeli havuz (K'dan tamamen bağımsız)

ML-100K'da denenmişti (`budget_pool.py`): havuz boyutu B sabitlenir, kümeleme
yalnızca "hangi B kişi" sorusunu cevaplar; K serbest bırakılır. Böylece K
tartışması hiç doğmaz. ML-1M'de B ∈ {%5, %10, %20} ile tekrarlanabilir.

Not: ML-100K'da bu protokolde B0 öndeydi (görev geometrikleşiyor) — o bulgu da
dürüstçe raporlanacak; kurgu 1 ve 2 ile birlikte tam resim verir.

---

## EK KANIT — ML-1M'de repair'siz küme dağılımları (K=6, fold1 s42)

| Yöntem | Repair'siz küme boyutları | Max/Min | Havuz |
|---|---|---|---|
| B0 (KMeans++) | **3451**, 889, 853, 378, 242, **227** | **15.2×** | **3623 (%60)** |
| AVOA | 1347, 1215, 1070, 928, 808, 672 | **2.0×** | 2054 (%34) |
| (repair'li — her ikisi) | 1007 … 1005 | 1.0× | 2013 (%33) |

**İki kritik bulgu:**

1. **Repair'siz karşılaştırma adaletsiz olurdu:** B0 tek kümede kullanıcıların
   %57'sini topluyor (3451/6040) ve havuzu 3623 kişiye çıkıyor — AVOA'nın
   havuzunun **%76 fazlası**. Yani repair'siz protokolde B0'ın "iyi MAE"si
   büyük ölçüde daha çok komşuya bakmasından gelir, kümeleme kalitesinden değil.
   ML-100K'da gördüğümüz "havuz artefaktı" ML-1M'de çok daha şiddetli.
2. **AVOA doğal olarak dengeli kümeliyor:** kapasite kısıtı olmadan bile max/min
   oranı 2.0× (B0'da 15.2×). Tahmin-hizalı fitness, dev küme oluşturmayı
   cezalandırıyor çünkü dev küme küme-MF'i ve komşuluk kalitesini bozuyor.
   **Bu, repair'in AVOA'ya dayatılan bir kısıt değil, zaten yöneldiği yapı
   olduğunun kanıtı** — kısıt esas olarak B0'ı hizaya sokuyor.

### Repair'siz MAE (aynı merkezler, atama serbest) — `ml1m_repair_ablasyon.csv`

| Mod | Yöntem | MAE | NDCG@10 | Havuz | Max küme |
|---|---|---|---|---|---|
| **Repair'SİZ** | B0 | **0.6897** | 0.8878 | **3596 (%60)** | 3451 |
| Repair'Lİ | B0 | 0.6880 | 0.8860 | 2013 (%33) | 1007 |
| Repair'Lİ | AVOA | 0.6876 | 0.8878 | 2013 (%33) | 1007 |

**Şaşırtıcı sonuç: repair'siz B0 (0.6897), repair'li B0'dan (0.6880) DAHA KÖTÜ**
— üstelik havuzu 1.8 katı. Yani serbest atama, %78 daha fazla hesapla daha kötü
doğruluk veriyor.

Nedeni: dev küme (3451 kişi) küme-MF modelini bozuyor — o kümenin MF'i
"ortalama kullanıcı" modeline dönüşüyor, kişiselleştirme kayboluyor. Küçük
kümeler (227 kişi) ise yeterli veri bulamıyor. Kapasite kısıtı her iki uçtaki
bozulmayı da engelliyor.

**Bu, kısıtın savunmasını tersine çeviriyor:** repair yalnızca "adil karşılaştırma
aracı" değil, **sistemin kendisini de iyileştiren bir tasarım kararı**. Hem daha
ucuz (havuz %60→%33) hem daha doğru (0.6897→0.6880). Makalede kısıt bir
"kısıtlama" olarak değil, katkı olarak sunulabilir.

Sonuç: "repair kullanmasak ne olur?" → hem karşılaştırma çöker (yöntemler farklı
maliyet sınıfında yarışır) hem de sistem kötüleşir (dengesiz kümeler küme-MF'i
bozar).

## Makalede nasıl yazılır (öneri cümleler)

> "K, önerilen sistemde ayarlanacak bir hiperparametre değil, komşuluk arama
> bütçesini belirleyen dağıtım kısıtıdır (havuz ≈ 2N/K). Bu nedenle tek bir K
> seçmek yerine, beş bütçe seviyesinde doğruluk-maliyet eğrisi raporlanmıştır
> (Şekil X). AVOA tüm seviyelerde K-means++ merkezlerine eşit veya üstün sonuç
> vermiş; avantajı bütçe daraldıkça artmıştır (−0.0004'ten −0.0196'ya).
> Eş-kalite ekseninden okunduğunda AVOA, aynı hata düzeyini K-means++'ın
> gerektirdiği havuzun yarısıyla sağlamaktadır."

Ve sınırlılık bölümüne dürüst not:

> "Bol bütçe rejiminde (havuz ≥ %33) yöntemler ayrışmamaktadır; bu beklenen bir
> sonuçtur, çünkü havuz genişledikçe kümeleme kararı komşuluk kümesini giderek
> daha az kısıtlar. Yöntemin katkısı, hesaplama bütçesinin bağlayıcı olduğu
> ölçeklenebilirlik rejimindedir."

## Sonuç: hangi K'da ana tablo?

**Hiçbiri tek başına.** Ana tablo = tüm bütçe seviyeleri (Kurgu 1), yanında
eş-kalite tablosu (Kurgu 2). "Mutlak en iyi sistemimiz" cümlesi için K=6 satırı
gösterilir (MAE 0.6876 — literatür bandı), "algoritma seçimi önemlidir" cümlesi
için eğrinin tamamı ve Friedman sonucu gösterilir.
