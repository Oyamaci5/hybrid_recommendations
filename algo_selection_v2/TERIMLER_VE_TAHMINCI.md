# Terimler Sözlüğü ve Tahminci Seçimi Gerekçesi

Bu belge iki soruyu net Türkçeyle cevaplar: (1) akıştaki terimler ne demek,
(2) neden küme-ortalaması değil küme-içi kNN + küme-MF kullanıyoruz.

---

## BÖLÜM 1 — Terimler

### Veri bölmesi: neden üç parça?

Elimizdeki tüm puanlar üç parçaya ayrılıyor:

```
TÜM PUANLAR (ML-1M: 1.000.209 puan)
│
├── TEST  (%10 ≈ 100.000 puan)  →  Sonuç raporlamak için. Modele HİÇ gösterilmez.
│                                   Sadece en sonda, bir kez ölçüm yapılır.
└── EĞİTİM (%90 ≈ 900.000 puan)
    │
    ├── İÇ-EĞİTİM (%90'ın %90'ı ≈ 810.000)  → Modelin öğrendiği veri:
    │                                          NMF uzayı, benzerlik matrisi,
    │                                          küme-MF modelleri buradan kurulur.
    └── İÇ-DOĞRULAMA (%90'ın %10'u ≈ 90.000) → Karar verme verisi:
                                               merkez arama (fitness) ve β seçimi
                                               burada ölçülür.
```

**Neden "iç" kelimesi?** Çünkü bu ikinci bölme, eğitim setinin *içinde* yapılıyor.
Alternatif Türkçe adlandırma (tezde kullanılabilir): **öğrenme kümesi** ve
**geliştirme kümesi** (İng. *development set*).

**Neden bu ayrım şart?** Meta-sezgisel algoritma binlerce aday merkez deniyor ve
en iyisini seçiyor. Bu seçimi test verisinde yapsaydık, test verisine göre
ayarlanmış bir sonuç raporlamış olurduk — literatürdeki şüpheli düşük MAE
değerlerinin (0.50-0.68) muhtemel kaynağı budur. Bizde test verisi hiçbir karar
aşamasına girmiyor.

**Neden %10 test?** ML-1M'in resmi bölmesi yok; literatürde standart uygulama
%90/%10 (bazı çalışmalarda %80/%20). Biz %90/%10 seçtik ve rastgele bölmeyi
5 farklı tohumla (fold) tekrarlıyoruz.

### "Bütçe" (NFE) nedir?

**NFE = Number of Function Evaluations** = uygunluk (fitness) fonksiyonunun kaç
kez çağrıldığı. Meta-sezgisel algoritmalar rastgele adaylar üretip her birini
"ne kadar iyi?" diye ölçer; bu ölçüm bir NFE'dir.

Türkçe karşılık: **arama bütçesi** ya da **değerlendirme sayısı**.

**Neden epoch değil NFE ile ölçüyoruz?** Algoritmalar aynı epoch'ta farklı
sayıda değerlendirme yapabiliyor (biz ölçtük: aynı epoch'ta SMA, GWO'dan 5 kat
fazla çağrı yapıyordu). Epoch'u eşitlemek, birine 5 kat fazla arama hakkı vermek
demek. NFE'yi eşitlemek adil karşılaştırmanın koşulu.

### Diğer terimler

| Terim | Anlamı |
|---|---|
| **Havuz** | Bir kullanıcı için komşu adaylarının listesi. Kullanıcının kümesi + en yakın ikinci küme (≈ 2N/K kişi). kNN komşularını buradan seçer. |
| **Kapasiteli onarım (repair)** | Hiçbir küme ⌈N/K⌉ kişiden fazlasını alamaz; dolan kümeye gelen kullanıcı sıradaki en yakın kümeye gider. |
| **Warm start (sıcak başlangıç)** | Meta-sezgiselin başlangıç popülasyonunu rastgele değil, K-means++ çözümünden türetmek. |
| **Fallback (yedek tahmin)** | Havuzda o filmi puanlamış komşu yoksa devreye giren basit formül: kullanıcı ortalaması + filmin genel sapması. |
| **β (beta)** | Karışım ağırlığı: `tahmin = β·kNN + (1−β)·MF`. İç-doğrulamada seçilir. |

---

## BÖLÜM 2 — ALS-MF nasıl tahmin yapıyor? (kNN'den farkı)

**kNN (komşuluk tabanlı):** "Sana en benzeyen 20 kişi bu filme kaç vermiş?"
Doğrudan başka kullanıcıların puanlarına bakar.

**MF (matris çarpanlarına ayırma):** Kimseye bakmaz; her kullanıcıya ve her filme
birer **gizli vektör** öğrenir (biz f=10 boyut kullanıyoruz).

```
tahmin(u,i) = μ  +  b_u  +  b_i  +  p_u · q_i
              ↑     ↑      ↑        ↑
           genel  kullanıcı film   zevk uyumu
        ortalama  eğilimi  eğilimi (vektör iç çarpımı)
```

- `b_u`: "Bu kullanıcı genelde cömert mi, cimri mi?"
- `b_i`: "Bu film genelde beğeniliyor mu?"
- `p_u · q_i`: "Bu kullanıcının zevk profili bu filmin özellikleriyle uyuşuyor mu?"

**ALS (Alternating Least Squares)** bu vektörleri *öğretme* yöntemidir: film
vektörlerini sabitleyip kullanıcı vektörlerini kapalı formülle çöz, sonra tersini
yap, 6-10 tur tekrarla.

**Küme-başına MF ne demek?** Her kümenin kendi MF modeli var; c kümesinin modeli
yalnız o kümedeki kullanıcıların puanlarıyla eğitiliyor. Böylece tahminin *iki*
bileşeni de (kNN ve MF) kümelemeden etkileniyor — bu, algoritma farkının tahmine
yansımasını sağlayan tasarım kararı (global MF kullansaydık farkı ezerdi, ölçtük).

**Neden ikisi birlikte?** Farklı hata yapıyorlar: kNN yerel komşuluk sinyali
taşır, MF küresel örüntüyü. Hataları ilişkisiz olduğu için karışımları ikisinden
de iyi:

| Tahminci | MAE (ML-100K, AVOA kümeleri) |
|---|---|
| Yalnız kNN | 0.7726 |
| Yalnız MF | 0.7652 |
| **Karışım (β=0.5)** | **0.7440** |

---

## BÖLÜM 3 — Neden küme-ortalaması değil? (literatürden ayrıldığımız nokta)

İncelediğimiz makalelerin çoğu **küme-ortalaması** kullanıyor: "kullanıcının
kümesindeki herkesin bu filme verdiği ortalama puan". Biz de öyle başladık ve
üç tahminciyi aynı kümeler üzerinde karşılaştırdık (`predictor_upgrade.csv`):

| Tahminci | B0 kümeleri | AVOA kümeleri |
|---|---|---|
| **Küme-ortalaması** (literatür) | 0.8485 | 0.8361 |
| Bias (kullanıcı ort. + küme sapması) | 0.7883 | 0.7747 |
| **Küme-içi kNN (k=20)** | **0.7877** | **0.7559** |

**Küme-ortalaması %11 daha kötü.** Üstelik düz *item-mean* (0.829) bile ondan iyi
— yani küme-ortalaması, kullanıcının kendi eğilimini (cömert/cimri) hiç
kullanmadığı için kaba kalıyor.

**Bu bulgu bizim için önemli çünkü:**
1. Literatürdeki "kümeleme doğruluğu düşürür" tespitinin bir kısmı, kümeleme
   yönteminden değil **zayıf tahminciden** geliyor.
2. Kaba tahminci algoritma farkını da eziyor: küme-ortalamasıyla tüm
   metasezgiseller 0.84-0.85 bandında sıkışıyordu; kNN'e geçince ayrıştılar.
   Yani "hangi meta daha iyi?" sorusu ancak yeterince keskin bir tahminciyle
   sorulabilir.

**Sonuç:** küme-ortalaması "daha mantıklı" görünse de (basit, hızlı), veriye göre
belirgin şekilde kötü. Ama tezde **ablasyon tablosu olarak mutlaka yer alacak** —
hem literatürle kıyas noktası hem de yukarıdaki iki bulgunun kanıtı.

### Sıradaki adım (öneri)
ML-1M'de aynı tahminci ablasyonunu tekrarlamak (küme-ort. / bias / kNN / karışım),
böylece "tahminci seçimi" bulgusu iki veri setinde de gösterilmiş olur.
Maliyet: tek fold/seed, ~5 dk.
