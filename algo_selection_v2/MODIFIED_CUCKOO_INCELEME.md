
# "Data Filtering for a Modified Cuckoo Search" (ICWR 2021) — İnceleme

Haghgu, Hasheminejad & Azmi, Alzahra Üniversitesi. **Bizim yapıya en yakın makale**
— ama önemli bir sızıntı sorunu içeriyor. İkisini de aşağıda.

---

## 1. Bizim yapıyla ÖRTÜŞEN yönler (öğrenilecek/alıntılanacak)

| Özellik | Onlarda | Bizde | Değerlendirme |
|---|---|---|---|
| **Optimizasyon merkezleri değil ATAMAYI değiştiriyor** | *"moving some items into better clusters"* — kullanıcıları daha iyi kümeye taşıyor | biz merkez arıyoruz | **Farklı yaklaşım.** Biz doğrudan atama aramasını denedik ve dejenere oldu (`PARAMETRIZASYON_BULGUSU.md`); onlarınki "K-means sonrası düzeltme" — ara bir yol |
| **Küme boyutuna müdahale ediyor** | *"combined the clusters with less than 60 users"* — 60 kişiden az kümeleri birleştiriyor | kapasite tavanı (üst sınır) | **En yakın akraba!** Onlar alt sınır, biz üst sınır koyuyoruz. Küme boyutunu bir tasarım değişkeni olarak gören **tek makale** |
| **Küme boyutu eşiği raporlanmış** | 65 küme, <60 kullanıcılı olanlar birleştirilmiş | maxk/mink/boş küme tablosu | Kısmi raporlama — diğer 15 makaleden iyi, bizden zayıf |
| **Turnuva seçimi** | rastgele seçim yerine turnuva → erken yakınsamayı engelliyor | elitist kabul denedik (küçük fayda) | Aynı amaç, farklı mekanizma |
| **K seçimi: elbow + WCSS farkı** | *"elbow eğrisi göremedik"*, fark eşiği 70 | doğruluk-maliyet cephesi | Bizimki daha savunulabilir; onlar da dirsek bulamadıklarını dürüstçe yazmış |
| **10-fold CV** | var | 5 fold × 10 seed | İyi; bu makale çoklu koşu yapan az sayıdakinden |
| **Metrik seti** | MAE, RMSE, P, R, F-measure | + NDCG, kapsama, havuz | Bizimki daha geniş |

## 2. KRİTİK SORUN — veri filtreleme sızıntısı

Makalenin başlığındaki "yenilik" şu:

> *"We filtered all the movies with **ratings above 4 of 5** to recommend new
> movies to users with high ratings for a more qualified prediction."*

**Yani puanı 4'ün üstünde olan filmler seçilip veri seti onlarla kuruluyor.**
Bu, hedef değişkene göre filtreleme — **hedef sızıntısı (target leakage)**:

1. Tahmin edilecek şey puandır; puana göre örnek seçmek, tahmin problemini
   kolaylaştırır (varyans yapay olarak daralır).
2. MAE 81 → 60 (%26 iyileşme) ve Precision 21 → 68 (3.2 kat) sıçramasının
   büyük kısmı bundan gelir; algoritmadan değil.
3. Sonuçlar filtrelenmemiş veriyle karşılaştırılamaz — bizim 0.71–0.74
   değerlerimizle onların "0.60"ı **aynı problemi ölçmüyor**.

Bir de tablo tuhaflığı: MAE/RMSE **yüzde olarak** verilmiş (60, 79) — 1-5
ölçeğinde 0.60/0.79'a karşılık geliyor olmalı ama bu net değil; K-means
baseline'ları 81/92 (=0.81/0.92) makul, dolayısıyla yorum bu yönde.

## 3. Sonuç: bizim yapıya uygun mu?

**Kısmen — ve iki farklı şekilde değerli:**

**(a) Metodolojik akraba olarak (olumlu):** Küme boyutunu bir tasarım değişkeni
olarak ele alan, bunu raporlayan ve müdahale eden **tek makale**. Bizim kapasite
kısıtımızın literatürdeki en yakın dayanağı. Tezde şöyle kullanılmalı:

> "Haghgu ve ark. (2021), 60 kullanıcıdan az üyeye sahip kümeleri birleştirerek
> küme boyutuna müdahale eden ender çalışmalardan biridir; bu, küme boyut
> dağılımının tahmin kalitesini etkilediğine dair literatürdeki dolaylı bir
> kabuldür. Çalışmamız bu müdahaleyi alt sınır yerine üst sınır (kapasite
> tavanı) olarak formüle etmekte ve dağılımı tüm tablolarda raporlamaktadır."

**(b) Kritik replikasyon örneği olarak (uyarı):** Puana göre filtreleme,
"iyileşme" iddialarının protokol tercihlerinden nasıl doğabileceğinin **en net
örneği**. HSC'de bunu ancak dolaylı kanıtlarla göstermiştik; burada makale
kendisi açıkça yazıyor. Bu, kritik replikasyon bölümümüzün en güçlü alıntısı.

## 4. Denenebilecek tek fikir: "atama düzeltme" (post-hoc refinement)

Onların asıl mekanizması — *"K-means sonrası bazı kullanıcıları daha iyi kümeye
taşımak"* — bizde yok ve denenebilir:

```
1. B0 veya meta ile merkezleri bul
2. Kapasiteli atama yap
3. DÜZELTME: rastgele bir kullanıcı seç; başka bir kümeye taşımak
   iç-val MAE'yi düşürüyorsa taşı (kapasite kontrolüyle)
4. NFE bütçesi bitene kadar tekrarla
```

Bu, merkez aramasından **bağımsız bir ikinci optimizasyon katmanı** olur ve
bizim "doğrudan atama araması dejenere oluyor" bulgumuzla çelişmez — çünkü
kapasite kısıtı ve iyi bir başlangıç noktası var.

**Öneri:** düşük öncelikli deneme. Beklenti: küçük kazanç (atama zaten
merkezlerden türüyor), ama "literatürden aldığımız üçüncü fikri de denedik"
demek için değerli. Karar sizin.

## 5. Kaynakçaya eklenecek künye

Z. Haghgu, S. M. H. Hasheminejad, R. Azmi, "A Novel Data Filtering for a
Modified Cuckoo Search Based Movie Recommender," *2021 7th International
Conference on Web Research (ICWR)*, pp. 243–247, doi: 10.1109/ICWR51868.2021.9443116.
