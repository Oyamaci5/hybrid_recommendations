
# Literatürdeki "k" Değerleri — Düzeltme

Önceki notta bu değerleri kNN komşu sayısı sanmıştım. PDF bağlamlarını
kontrol edince çoğunun **küme sayısı (K)** olduğu görüldü. Doğrusu:

| Makale | Değer | Neyin k'sı? | Kanıt (metinden) | Tahmin yöntemi |
|---|---|---|---|---|
| GOA-k-means (2024) | **K=3** | küme sayısı | "GOA-k-means (k=3)", "effect of number of search agents on RMSE" | belirtilmemiş |
| HSC Sparrow | **K=70** | küme sayısı | "sparrow search based on **cluster size**", "comparison with other algorithms (k=70)" | **küme-ortalaması** ("kümedeki en yüksek ortalama puanlı filmler önerilir") |
| Firefly + collab | **K=90** | küme sayısı | "comparison with other algorithms (k=90) k-means" | küme tabanlı |
| **Katarya & Verma (2016)** | **kNN k = 5–60** | **gerçek komşu sayısı** | "neighbourhood size varies from 5 to 60 in an increment of 5", "number of neighbours **in clusters**" | **küme-içi kNN** |

## Sonuçlar

**1. Bizim K değerlerimiz literatürle uyumlu.** HSC K=70, Firefly K=90, GOA K=3
kullanıyor; biz K=6–60 aralığını tarıyoruz. Yani küme sayısı seçimimiz
literatürün içinde ve üstelik **tek bir K'ya sabitlenmiyoruz** (onlar sabitliyor).

**2. kNN komşu sayısı için tek referans Katarya (2016).** 5'ten 60'a tarayıp
**15–20'de optimum** bulmuş, 60'a kadar stabil kalıyor demiş. Bizim k=20
seçimimiz tam bu bandın içinde → literatürle doğrulanmış.

**3. En kritik bulgu — tahmin yöntemi ayrımı:**
- HSC (K=70) **küme-ortalaması** ile tahmin ediyor: "kümedeki en yüksek ortalama
  puanlı filmler önerilir". Kullanıcının kendi eğilimi hiç kullanılmıyor.
- Katarya küme-içi **kNN** kullanıyor — bizim hattımıza en yakın çalışma.
- Bizim ablasyonumuz (ML-1M, K=40): küme-ortalaması 0.807 vs küme-içi kNN 0.734
  → **%9 fark**. Yani HSC hattı yapısal olarak dezavantajlı.

**4. Karşılaştırma tablosuna not düşülecek:** "Bazı çalışmalarda bildirilen k
değeri küme sayısını (K), bazılarında komşu sayısını ifade etmektedir; bu ayrım
makalelerde her zaman açık değildir." — bu, literatür karşılaştırmalarının neden
zor olduğuna dair somut bir örnek ve bizim iki parametreyi ayrı ayrı raporlama
gerekçemiz.

## Bizim gösterimimiz (karışıklığı önlemek için)

| Sembol | Anlam | Bizim değerimiz |
|---|---|---|
| **K** | küme sayısı (= bütçe: havuz ≈ 2N/K) | 6 / 40 (iki çalışma noktası) |
| **k** | kNN komşu sayısı | 20 (Katarya'nın 15–20 optimumuyla uyumlu) |

Tezde bu iki sembol ilk kullanımda açıkça tanımlanacak.
