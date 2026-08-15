
# Büyük K + Küme-Ortalaması Deneyi — Literatür Kurulumunun Anatomisi

ML-1M, fold 1, seed 42. Literatürün kurulumu (K=70/90/120 + küme-ortalaması)
ile bizim kurulumumuz aynı merkezler üzerinde karşılaştırıldı.
Kaynak: `ml1m_buyukK.csv`

---

## Ana tablo

| K | Yöntem | Atama | Tahminci | MAE | NDCG | Küme: max / min / boş |
|---|---|---|---|---|---|---|
| 70 | B0 | serbest | cmean | 0.8249 | 0.8522 | 836 / 6 / 0 |
| 70 | B0 | repair | cmean | 0.8167 | 0.8558 | 87 / 38 / 0 |
| 70 | **AVOA** | **serbest** | cmean | **0.7843** | 0.8710 | **3706 / 0 / 59** |
| 70 | AVOA | repair | cmean | 0.8228 | 0.8488 | 87 / 37 / 0 |
| 70 | B0 | repair | **kNN+MF** | 0.7340 | 0.8564 | — |
| 70 | **AVOA** | repair | **kNN+MF** | **0.7124** | **0.8755** | — |
| 120 | B0 | repair | cmean | 0.8312 | 0.8474 | 51 / 8 / 0 |
| 120 | **AVOA** | **serbest** | cmean | **0.7884** | 0.8722 | **3893 / 0 / 105** |
| 120 | AVOA | repair | kNN+MF | **0.7215** | 0.8703 | — |

---

## BULGU 1 — Kısıtsız optimizasyon kümeleri yok ediyor (K büyüdükçe daha çok)

AVOA serbest atamada:
- K=70: en büyük küme **3706 kişi** (%61), **59 küme tamamen boş** → etkin küme
  sayısı 11
- K=90: max 3366, **78 boş** → etkin 12
- K=120: max 3893, **105 boş** → etkin 15

Yani "K=120 ile kümeledik" denilen sistemde gerçekte ~15 küme var ve biri
kullanıcıların üçte ikisini barındırıyor. **Ve bu dejenere yapı en iyi
cmean MAE'sini veriyor** (0.784–0.788), çünkü dev küme ortalaması = global
ortalamaya yakın, yani kararlı bir tahmin.

→ **HSC bilmecesinin mekanizması bu.** Küme boyut dağılımı raporlanmadığında,
"K arttıkça MAE düşüyor" sonucu kümeleme başarısı gibi görünüyor; oysa
optimizasyon K arttıkça daha çok kümeyi boşaltıp tek dev kümeye yaklaşabiliyor.

## BULGU 2 — Bizim replikasyonda MAE, K ile KÖTÜLEŞİYOR (HSC'nin tersi)

| K | B0 repair cmean | B0 serbest cmean |
|---|---|---|
| 70 | 0.8167 | 0.8249 |
| 90 | 0.8233 | 0.8287 |
| 120 | 0.8312 | 0.8407 |

HSC Tablo 4'te K=10→70 arasında MAE 0.785→0.695 **iyileşiyordu**. Bizde
K=70→120 arasında 0.817→0.831 **kötüleşiyor** — teorik beklentiyle uyumlu
(küme küçüldükçe ortalama gürültülenir). Bu, HSC'nin monoton iyileşmesinin
protokol dışı bir kaynaktan geldiği hipotezini güçlendiriyor.

## BULGU 3 — Meta-sezgiselin avantajı TAHMİNCİYE BAĞLI (dürüst uyarı)

Kapasiteli atama + küme-ortalaması ile:

| K | B0 cmean | AVOA cmean | Kim iyi? |
|---|---|---|---|
| 70 | 0.8167 | 0.8228 | **B0** |
| 90 | 0.8233 | 0.8320 | **B0** |
| 120 | 0.8312 | 0.8363 | **B0** |

Ama kNN+MF ile:

| K | B0 | AVOA | Kim iyi? |
|---|---|---|---|
| 70 | 0.7340 | **0.7124** | AVOA (−0.022) |
| 90 | 0.7400 | **0.7179** | AVOA (−0.022) |
| 120 | 0.7472 | **0.7215** | AVOA (−0.026) |

**Neden:** bizim uygunluk fonksiyonumuz soft top-2 bias tahmincisini kullanıyor;
bu, kNN+MF'e yakın, hard küme-ortalamasına uzak. Meta-sezgisel neyi optimize
ederse onda kazanıyor.

**Tez için iki sonuç:**
1. *Dürüstlük:* "AVOA her koşulda daha iyi" diyemeyiz; **fitness–tahminci
   hizalaması** şart. Bu, sınırlılıklar bölümüne yazılacak.
2. *Literatür eleştirisi güçleniyor:* küme-ortalaması kullanan çalışmalar
   WCSS'i optimize ediyor — yani ne fitness tahminciyle hizalı, ne de tahminci
   yeterince keskin. İki kat uyumsuzluk.

## BULGU 4 — Bizim kurulum her K'da açık ara önde

| K | En iyi literatür kurulumu (cmean) | Bizim (kNN+MF) | Fark |
|---|---|---|---|
| 70 | 0.7843 (AVOA, dejenere) / 0.8167 (B0, dürüst) | **0.7124** | %9–13 |
| 90 | 0.7853 / 0.8233 | **0.7179** | %9–13 |
| 120 | 0.7884 / 0.8312 | **0.7215** | %8–13 |

Dejenere çözüm dahil bile bizim kurulum %9 önde; dürüst kurulumla (repair'li
B0 cmean) fark %13.

## BULGU 5 — NDCG'de dejenerasyon ödüllendiriliyor

AVOA-serbest-cmean, dejenere olmasına rağmen NDCG'de yüksek (0.871–0.874).
Sebep: tek dev kümede sıralama fiilen **popülerlik sıralamasına** dönüşüyor ve
NDCG popüler-ağırlıklı test dağılımında bunu ödüllendiriyor. Bizim yöntemimiz
benzer NDCG'yi (0.8703–0.8755) **gerçek kümelerle** elde ediyor.

→ Bu, "NDCG tek başına yeterli değil, küme dağılımıyla birlikte okunmalı"
argümanımızın kanıtı.

---

## Özet: literatür kurulumunun üç yapısal sorunu (deneyle gösterildi)

| Sorun | Kanıt |
|---|---|
| Kısıtsız atama → dejenerasyon | K=120'de 105 boş küme, tek kümede %64 kullanıcı |
| Küme-ortalaması tahminci → %13 kayıp | 0.8312 vs 0.7215 (aynı merkezler) |
| WCSS fitness → tahminciyle hizasız | AVOA cmean'de B0'ı geçemiyor |

Bizim kurulumumuz üçünü de adresliyor: kapasite kısıtı, kNN+MF karışımı,
tahmine hizalı fitness.
