# Protokol Savunması — Kapasite Kısıtı ve Soft Top-2 Havuz

Bu belge, hakem/danışman sorularına karşı iki tasarım kararının literatür
dayanağını ve deneysel gerekçesini toplar.

---

## 1. Kapasite kısıtı (repair) — icat değil, yerleşik alan

Kapasiteli/dengeli kümeleme, kümeleme literatüründe kendi adı olan bir alt daldır:

| Kaynak | Katkı | Bizimle ilişki |
|---|---|---|
| **Bradley, Bennett & Demiriz (2000)**, *Constrained K-Means Clustering*, MSR-TR-2000-65 | K-means'e "her küme en az m nokta içersin" kısıtı; boş/kırıntı kümeleri engeller; min-cost-flow ile çözer | Bizim K1 dejenerasyon bulgumuzun klasik karşılığı; alt-sınır kısıtı |
| **Malinen & Fränti (2014)**, *Balanced K-Means for Clustering*, S+SSPR | Eşit boyutlu kümeler; atama fazı Macar algoritmasıyla optimal çözülür (O(n³)) | Bizim repair'in optimal (ama pahalı) versiyonu |
| **Malinen, Järviö & Fränti (2025)**, *Fixed-sized clusters k-Means*, arXiv:2501.16113 | Sabit boyutlu küme varyantı, ölçeklenebilir | Güncel alan kanıtı |
| **Capacitated Clustering Problem (CCP)** yazını | Küme kapasitesi aşılamaz; en kısa atama yoluna göre gruplama | Bizim repair'in tam tanımı |
| Wikipedia: *Balanced clustering* | Alanın standart tanımı | Terminoloji referansı |

**Sonuç:** "kullanıcıyı kapasite dolduğu için ikinci en yakın kümeye atamak"
literatürde tanımlı bir işlemdir (greedy capacitated assignment). Bizim
yaptığımız, bu bilinen kısıtı **öneri sistemi bağlamına taşımak** ve
karşılaştırma protokolü olarak kullanmak — özgün olan bu birleşim.

Ek meşruiyet: dengeli kümeleme öneri servisinde **ölçeklenebilirlik garantisi**
demektir (küme başına sorgu maliyeti sabitlenir). Dev küme = yavaş sorgu,
kırıntı küme = kullanılamaz öneri.

## 2. Soft top-2 havuz — "multi-clustering" ailesinin kısıtlı hali

| Kaynak | Katkı | Bizimle ilişki |
|---|---|---|
| **Kużelewska (2018)**, *CF Recommender Systems Based on k-means Multi-clustering*, ICCS | Tek kümeleme yerine çoklu kümeleme; kullanıcı birden çok kümeye ait olabilir; doğruluk kaybını önler | Bizim top-2 havuzumuzun kavramsal atası |
| Kużelewska (2020), *Effect of Dataset Size...multi-clustering* | Çoklu küme komşuluk modellemesi ölçekle birlikte incelenir | Ölçek argümanı |
| Kużelewska (2021), *Quality of Recommendations and Cold-Start... Multi-clusters* | Çoklu küme cold-start'ı iyileştirir | Fallback azalması bulgumuzla uyumlu |
| **Overlapping co-clustering** (arXiv:1604.02071), CCCF, BinRec | Kullanıcı/öğe birden çok alt gruba ait; sonuçlar birleştirilir | Örtüşen küme ailesi |

**Literatürün ortak tespiti:** kümeleme hız kazandırır ama doğruluğu düşürür;
**örtüşme (multi/overlapping clustering) bu kaybı telafi eder.** Bizim soft top-2
havuzumuz bunun en yalın, maliyeti tam kontrol edilebilir biçimi: kullanıcı
kendi kümesi + en yakın ikinci kümeden komşu seçer, havuz boyutu ≈ 2N/K.

## 3. Neden özellikle top-2? (deneysel gerekçe)

| Havuz | K=10, AVOA | Kanıt dosyası |
|---|---|---|
| top-1 (hard) | MAE 0.7906, fallback %4.1 | `eksikler_soft.csv` |
| **top-2 (soft)** | **MAE 0.7726, fallback %1.4** | aynı |
| top-3+ | havuz ≈3N/K → maliyet artışı; sınırda kümesiz kNN'e yakınsar | K18 (K=2 satırı: fark → 0) |

- top-1 → top-2 geçişi MAE'yi −0.018 düşürüp fallback'i 3 kat azaltıyor.
- Kapasite kısıtı altında top-2 **zorunlu telafi**: yer değiştiren kullanıcıların
  **%100'ünün** en yakın kümesi havuzda kalıyor (K21 ölçümü) — kısıt komşuluk
  kaybına yol açmıyor.
- top-3 ve üzeri havuzu şişirir; K18 kanıtladı ki havuz büyüdükçe algoritma
  farkı erir (K=2'de fark sıfır). Yani top-2, "telafi et ama kıyası bozma"
  noktasıdır.

## 4. Kısıt yanlış mı? — üç katmanlı cevap

1. **Doğruluk açısından:** kısıt tek başına MAE'yi ~0.015 kötüleştiriyor
   (K20: serbest 0.7502 vs repair 0.7654, B0). Ancak serbest protokoldeki
   kazancın kaynağı yöntem değil, havuzun %33→%43-50 büyümesi.
2. **Kullanıcı açısından:** zarar telafi ediliyor (K21: %100 en-yakın-küme
   havuzda kalıyor). Kaybedilen tek şey küme-MF modelinin birincil sahipliği.
3. **Bilimsel açıdan:** kısıtsız protokolde algoritma farkı 10 kat eriyor ve
   dejenerasyon geri geliyor (en büyük küme %40). Yani kısıt, ölçülmek istenen
   şeyi ölçülebilir kılan koşuldur.

**Makale formülasyonu:** ana tablo repair'li (adil kıyas), ablasyon tablosu
serbest (gerçekçi üst sınır), ikisi birlikte doğruluk-maliyet cephesi olarak
sunulur. Kısıt bir sınırlama değil, **kontrollü deney tasarımıdır.**

## Kaynaklar

- Bradley, Bennett, Demiriz — Constrained K-Means Clustering:
  https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2000-65.pdf
- Malinen & Fränti — Balanced K-Means for Clustering:
  https://link.springer.com/chapter/10.1007/978-3-662-44415-3_4
- Malinen, Järviö, Fränti — Fixed-sized clusters k-Means: https://arxiv.org/abs/2501.16113
- Balanced clustering (tanım): https://en.wikipedia.org/wiki/Balanced_clustering
- Kużelewska — CF RS Based on k-means Multi-clustering:
  https://link.springer.com/chapter/10.1007/978-3-319-91446-6_30
- Kużelewska — Multi-clusters & cold-start:
  https://link.springer.com/chapter/10.1007/978-3-030-77964-1_6
- Multi-clustering & dataset size: https://pmc.ncbi.nlm.nih.gov/articles/PMC7304038/
- Overlapping co-clustering: https://arxiv.org/pdf/1604.02071
