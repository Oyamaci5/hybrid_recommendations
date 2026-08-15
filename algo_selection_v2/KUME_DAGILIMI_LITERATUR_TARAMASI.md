
# Sistematik Tarama: Literatürde Küme Boyut Dağılımı Raporlanıyor mu?

**Yöntem.** Elimizdeki 15 makalenin tam metni, şu terimler için tarandı:
`cluster size`, `size of the/each cluster`, `users per cluster`, `users in cluster`,
`number of users/members in`, `cluster distribution`, `empty cluster`,
`smallest/largest cluster`.

**Sonuç: 15 makalenin HİÇBİRİNDE küme üye sayısı dağılımı raporlanmıyor.**

---

## Tarama sonuçları (eşleşme sayısı ve bağlamı)

| Makale | Eşleşme | Bağlam — ne kastediliyor? |
|---|---|---|
| HSC (Sparrow) | 6 | *"Ni are the number of users in the cluster"* → yalnız **formül notasyonu**. "Table 4 ... based on different cluster size" → burada **cluster size = K (küme sayısı)**, üye sayısı değil |
| HHO-K-means | 4 | *"across all cluster sizes"* → **K taraması** |
| Firefly | 4 | *"where Ni is the number of users in cluster i"* → **formül notasyonu**; "Table 4. Performance based on cluster size" → **K** |
| GOA | 4 | *"accuracy of the algorithms across a range of cluster sizes"* → **K** |
| ABC (Movie RS) | 2 | *"changes in the number of users"* → **veri seti büyüklüğü**, küme değil |
| Crow Search | 1 | atama matrisi tanımı |
| Katarya PSO | 1 | komşuluk boyutu (kNN k) |
| GWO-FCM, Dipper-GWO, Harmony-FCM, CS-Kmeans, IWO, MOHHO, WOA-MapReduce, MOO survey | 0 | hiç geçmiyor |

**Kritik ayrım:** literatürde "cluster size" ifadesi neredeyse her yerde
**küme sayısını (K)** kastediyor; "bir kümede kaç kullanıcı var" anlamında
kullanılmıyor. `Ni` sembolü tanımlanıyor ama **değeri hiçbir tabloda verilmiyor**.

---

## Bu neden gerekli? — Üç somut gerekçe (kendi ölçümlerimizle)

### 1. Küme-ortalaması tahmininin güvenilirliği doğrudan üye sayısına bağlı

"Kümedeki kullanıcıların ortalaması" ifadesi, kümede 3706 kişi varken ile
6 kişi varken tamamen farklı şeyler ifade eder. Ölçtük (ML-1M, K=70, serbest atama):

| Yöntem | En büyük küme | En küçük küme | Boş küme | Ortalama ± std |
|---|---|---|---|---|
| B0 | 836 | 6 | 0 | 86 ± 114 |
| AVOA | **3706** | **0** | **59** | 86 ± **501** |

Aynı "K=70" başlığı altında iki tamamen farklı sistem var. Standart sapma 501 —
yani "ortalama küme 86 kişilik" ifadesi anlamsız.

### 2. Raporlanmadığında dejenerasyon görünmez kalıyor

ML-1M, K=120, serbest atama, AVOA: **105 küme boş**, tek kümede 3893 kullanıcı
(%64). Sistem "120 kümeli" diye tanıtılıyor ama etkin küme sayısı 15.
**Ve bu dejenere yapı en iyi küme-ortalaması MAE'sini veriyor** (0.7884 vs
dürüst kurulumda 0.8312).

→ Okuyucu, iyileşmenin daha iyi kümelemeden mi yoksa kümelemenin çökmesinden mi
geldiğini ayırt edemiyor.

### 3. "K arttıkça MAE düşüyor" anomalisi ancak dağılımla açıklanabilir

HSC Tablo 4: K=10 → 0.785, K=70 → 0.695 (monoton iyileşme).
Bizim dürüst replikasyonumuz: K=70 → 0.817, K=120 → 0.831 (monoton **kötüleşme**).

Küme-ortalaması tahmincisiyle teorik beklenti kötüleşmedir (küme küçülür,
ortalama gürültülenir). HSC'nin ters yöndeki sonucu, ancak küme dağılımı
bilinirse yorumlanabilir — ama raporlanmamış.

---

## Literatürden destekleyici alıntılar (kısıt/dengeleme gerekliliği)

Küme boyut dengesizliği, kümeleme literatüründe **bilinen bir sorun** olarak
ele alınıyor; öneri sistemlerinde ihmal ediliyor:

- **Bradley, Bennett & Demiriz (2000)**, *Constrained K-Means Clustering*:
  K-means'e minimum küme boyutu kısıtı eklemelerinin gerekçesi doğrudan
  *"boş veya çok az noktalı kümelerle sonuçlanan yerel çözümlerden kaçınmak"*.
- **Malinen & Fränti (2014)**, *Balanced K-Means*: eşit boyutlu kümelerin
  atama fazını Macar algoritmasıyla optimal çözüyorlar.
- **Capacitated Clustering Problem** yazını: kapasite aşımı yapısal olarak
  engelleniyor.

Yani "küme boyutu önemlidir" fikri kümeleme alanında yerleşik; **öneri
sistemleri alanında bu köprü kurulmamış.** Bizim katkımız tam olarak bu köprü.

---

## Tez/makale için hazır formülasyon

> "İncelenen 15 çalışmanın hiçbirinde küme üye sayısı dağılımı raporlanmamıştır.
> 'Küme boyutu' ifadesi bu çalışmaların çoğunda küme sayısını (K) belirtmekte,
> bir kümedeki kullanıcı sayısını ifade etmemektedir; `Ni` gibi semboller
> formüllerde tanımlanmakta ancak değerleri verilmemektedir. Oysa küme-ortalaması
> temelli tahminde bu bilgi belirleyicidir: deneylerimizde aynı K değeri altında
> en büyük kümenin 3706, en küçüğünün 0 kullanıcı içerdiği (59 boş küme)
> yapılandırmalar gözlenmiş; bu dejenere yapılandırmalar dengeli olanlardan daha
> düşük hata değerleri üretmiştir. Dolayısıyla küme boyut dağılımının
> raporlanması, sonuçların yorumlanabilmesi için gereklidir ve bu çalışmada
> tüm tablolarda maksimum/minimum/boş küme sayıları verilmiştir."

## Bizim raporlama standardımız (tez tablolarında zorunlu sütunlar)

| Sütun | Anlamı |
|---|---|
| `maxk` | En büyük kümedeki kullanıcı sayısı |
| `mink` | En küçük kümedeki kullanıcı sayısı |
| `bos_kume` | Boş küme sayısı |
| `ort_kume ± std` | Ortalama küme büyüklüğü ve dağılımı |
| `havuz` | Kullanıcı başına komşu adayı sayısı (maliyet) |
| `fallback_pct` | Tahmin kurulamayan durum oranı |

Bu altı sütun, bir okuyucunun sonucu bağımsız yorumlayabilmesi için yeterlidir
ve önerdiğimiz raporlama standardını oluşturur.
