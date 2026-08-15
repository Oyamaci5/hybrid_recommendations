
# Kapasiteli Onarım (Repair) — Nasıl Çalışıyor, Neden Eşit?

---

## 1. Algoritma (kod: `ml1m_run.fast_repair`)

```
GİRDİ : X (kullanıcı özellik matrisi, N×d), C (merkezler, K×d)
ÇIKTI : L (her kullanıcının kümesi), near (havuz için en yakın 2 küme)

1: cap ← ⌈N / K⌉                          # kapasite tavanı
2: D[u,c] ← ‖X_u − C_c‖²                  # tüm uzaklıklar (açılım formülüyle)
3: sizes[c] ← 0                            # küme doluluk sayaçları
4: for u ∈ SIRALA(kullanıcılar, artan min_c D[u,c]):     ← ÖNCELİK KURALI
5:     for c ∈ SIRALA(kümeler, artan D[u,c]):            ← TERCİH SIRASI
6:         if sizes[c] < cap:
7:             L[u] ← c ;  sizes[c] ← sizes[c] + 1 ;  break
8: near[u,0] ← L[u]                        # atandığı küme
9: near[u,1] ← en yakın DİĞER küme         # havuzun ikinci parçası
```

**İki kritik satır:**

- **Satır 4 — öncelik kuralı:** kullanıcılar, en yakın merkeze olan uzaklıklarına
  göre **artan** sırada işlenir. Yani "kümesinden emin olan" (merkeze çok yakın)
  kullanıcılar önce yerleşir; sınırda kalanlar sona bırakılır. Bu, kapasite
  dolduğunda kaydırılacak kişilerin **zaten kararsız olanlar** olmasını sağlar.
- **Satır 6 — kapasite kontrolü:** küme doluysa kullanıcı sıradaki en yakın
  kümeye gider. Merkez konumlarına dokunulmaz; yalnızca atama kısıtlanır.

**Sonuç dağılımı:** ML-1M K=6 için cap = ⌈6040/6⌉ = 1007 →
küme boyutları [1007, 1007, 1007, 1007, 1007, 1005]. Yani "eşit" derken
en fazla 1–2 kişilik sapma kastediliyor.

**Karmaşıklık:** O(N·K log K) — 6040 kullanıcı × 40 küme için milisaniyeler.
(Optimal çözüm Macar algoritmasıyla O(N³) olurdu; Malinen & Fränti 2014 bunu
kullanıyor ama 6040 kullanıcı için pratik değil. Bizimki *greedy sezgisel*.)

---

## 2. Eşitliğin amacı — üç ayrı gerekçe

### Amaç A: Eşit hesaplama maliyeti (asıl amaç)

Komşu havuzu = atanan küme + en yakın ikinci küme ≈ **2N/K kişi**.
Kümeler eşitse bu sayı herkes için aynı; eşit değilse kullanıcıdan kullanıcıya
ve yöntemden yönteme değişir.

Ölçtük (ML-1M, K=6, kısıtsız):

| Yöntem | Havuz | En büyük küme |
|---|---|---|
| B0 (KMeans++) | 3623 (%60) | 3451 |
| AVOA | 2054 (%34) | 1347 |
| **repair'li (her ikisi)** | **2013 (%33)** | **1007** |

Kısıtsız halde B0, AVOA'dan **%76 daha fazla** komşuya bakıyor. Bu durumda MAE
karşılaştırması "kim daha iyi kümeledi"yi değil "kim daha çok hesap yaptı"yı
ölçer. Eşit kapasite, tek değişkeni **merkez kalitesi** yapıyor.

### Amaç B: Dejenerasyonu engellemek

Kısıtsız doğruluk hedefinin küresel optimumu "kümelemeyi iptal etmek"tir.
Ölçümler:
- ML-1M K=120, serbest: **105 küme boş**, tek kümede 3893 kullanıcı (%64)
- Doğrudan atama araması: HHO 943'ün **942'sini** tek kümeye koydu

Kapasite tavanı bunu **yapısal olarak imkânsız** kılıyor (ceza terimi gibi
"caydırma" değil, doğrudan engelleme).

### Amaç C: Küme-başına MF'in çalışabilmesi

Her kümenin kendi matris çarpanlarına ayırma modeli var. Dev kümede model
"ortalama kullanıcıya" yakınsıyor (kişiselleştirme kaybı), 6 kişilik kümede
öğrenecek veri yok. Ölçtük (ML-1M, K=6):

| | Küme-MF tek başına MAE |
|---|---|
| repair'li (dengeli) | **0.7371** |
| repair'siz (dengesiz) | 0.7480 |

Yani kısıt, sistemin kendisini de iyileştiriyor — sadece ölçüm aracı değil.

---

## 3. "Tam eşitlik" şart mı? — Hayır, ama en temiz nokta

Kapasiteyi gevşetip test ettik (ML-100K, K=30):

| Kapasite | B0 MAE | AVOA MAE | Fark | AVOA havuzu |
|---|---|---|---|---|
| 1.0× (tam eşit) | 0.7976 | 0.7712 | −0.0264 | 63 |
| 1.5× | 0.7941 | 0.7656 | −0.0285 | 93 |
| 3.0× | 0.7941 | 0.7575 | −0.0366 | 188 |
| ∞ (serbest) | 0.7759 | 0.7694 | −0.0065 | 132 |

**AVOA her sertlik seviyesinde kazanıyor** → bulgu kısıt seçimine bağlı değil.
Gevşek kısıtta fark büyüyor **ama havuz da büyüyor**, yani kazancın ne kadarının
merkez kalitesinden ne kadarının ek maliyetten geldiği ayrışamıyor.

**Bu yüzden 1.0× ana tablo, diğerleri ablasyon.** Tam eşitlik "en iyi sistem"
değil, **en yorumlanabilir ölçüm noktası**.

---

## 4. Kullanıcı açısından maliyeti (dürüstlük kontrolü)

| K | Yer değiştiren kullanıcı | Top-2 havuzunda en yakın kümesi kalan |
|---|---|---|
| 6 | 247/943 (%26) | **%100** |
| 40 | 374/943 (%40) | **%100** |

Kapasite yüzünden ikinci kümeye atanan kullanıcıların **hepsi**, soft top-2
havuz sayesinde kendi en yakın kümesini komşu havuzunda buluyor. Kaybettikleri
tek şey, o kümenin küme-MF modelinin *birincil* sahibi olmak.

→ Kısıt, kullanıcıyı komşularından etmiyor; yalnızca hangi MF modelinin
"ev sahibi" olduğunu değiştiriyor.

---

## 5. Tek paragraflık özet (teze girecek hali)

> Önerilen protokolde her küme en fazla ⌈N/K⌉ kullanıcı içerebilir. Atama,
> kullanıcıların en yakın merkeze uzaklıklarına göre artan sırada, açgözlü
> (greedy) bir kapasiteli atama sezgiseliyle yapılır: merkeze en yakın
> kullanıcılar önce yerleştirilir, kapasitesi dolan kümeye yönlenen kullanıcı
> sıradaki en yakın kümeye atanır. Bu kısıtın üç işlevi vardır: (i) tüm
> yöntemlerde komşu havuzu boyutunu (≈2N/K) eşitleyerek karşılaştırmayı tek
> değişkenli (merkez kalitesi) hale getirmek, (ii) doğruluk temelli uygunluk
> fonksiyonunun kümelemeyi dejenere etmesini yapısal olarak engellemek,
> (iii) küme-başına matris çarpanlarına ayırma modellerinin yeterli ve dengeli
> veriyle eğitilmesini sağlamak. Kısıtın kullanıcı düzeyindeki maliyeti, soft
> top-2 havuz tarafından tamamen telafi edilmektedir: yer değiştiren
> kullanıcıların tamamı en yakın kümelerini komşu havuzunda korumaktadır.
