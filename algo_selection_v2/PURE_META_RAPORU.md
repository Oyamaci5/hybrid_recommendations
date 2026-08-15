# Saf Meta Karşılaştırması — Lloyd/KMeans++ YOK (nmf20, K=7)

Protokol: fold 1 train, NMF-20, K=7, SSE-WCSS, 22 meta × 5 seed (epoch=50, pop=30).
B0'lar Lloyd'suz: **B0_RAND_CENT** (K rastgele veri noktası, arama yok) ve
**B0_RAND_SEARCH** (eşit bütçe: 1500 rastgele aday, en iyisi).
Dosyalar: `results/pure_meta_summary.csv`, `results/pure_meta_friedman.txt`.

## Sonuç tablosu (özet)

| Grup | Algoritmalar | WCSS |
|---|---|---|
| Rastgele aramayı GEÇEN (gerçek arama) | **SMA 258, GWO 274, AGTO 284, AVOA 285, NGO 290, HBA 297, INFO 300** | < 318 |
| Eşik: B0_RAND_SEARCH | — | **318** |
| Rastgele aramanın ALTINDA | MFO, HHO, WOA, AEO, HGS, GTO, OOA, MPA (320–344) | 318–345 |
| Arama yok çizgisi: B0_RAND_CENT | — | 385 |
| Çöken (boş kümeli, dejenere) | GA, SquirrelSA, PSO, DE, CircleSA, SA, BeesA | 456–2075 |

Friedman χ²=107.7, **p=6.4e-13** → sıralama istatistiksel olarak sağlam.
Referans (yarış dışı): KMeans++ n_init=10 = 223.5.

## Ana bulgular

1. **Lloyd kalkınca algoritma farkları patladı:** kmref'li tabloda uçlar arası fark
   ~46 puandı; burada 258→2075 (8 kat). Algoritmaları gerçekten ayrıştıran deney bu.
   Tezin "algoritma karşılaştırma" bölümü bu protokolle yazılmalı.

2. **Eleme kriteri çalıştı:** 22 metanın yalnızca 7'si eşit bütçeli rastgele aramadan
   iyi. **HHO ve MPA dahil 8 meta rastgele aramayı geçemiyor** — bu bütçede
   "arama" yapmıyorlar. (HHO'nun literatür değeri kalır ama performans iddiası kalmaz.)

3. **MPA tersine döndü — iki deney farklı yetenek ölçüyor:** kmref'li tabloda MPA
   3.'ydü, burada 14. → MPA iyi havza buluyor ama içini kazamıyor; Lloyd onu
   kurtarıyordu. SMA/AVOA/NGO iki tabloda da üstte → tutarlı adaylar.

4. **Silhouette/DB tuzağı bir kez daha:** çöken algoritmalar (BeesA sil=0.78,
   DB=0.39!) en iyi iç metrikleri veriyor çünkü boş kümeli, K≈3'e dejenere
   çözümler üretiyorlar. İç metrikler TEK BAŞINA kullanılamaz; boş küme sayısı +
   WCSS ile birlikte okunmalı. (empty_mean sütunu tabloda.)

5. **KMeans++ farkı:** en iyi saf meta bile (SMA 258) KMeans++'tan %15 geride.
   Saf meta protokolü "hangi algoritma daha iyi arayıcı" sorusunun cevabı;
   üretim pipeline'ı için değil.

## Adalet uyarısı (final tur için zorunlu düzeltme)

SMA'nın koşu süresi 3.9s, diğerlerinin ~0.7s — aynı epoch×pop'ta bazı algoritmalar
**daha fazla fonksiyon değerlendirmesi (NFE)** yapıyor. SMA'nın birinciliği kısmen
bütçe avantajı olabilir. Final turda bütçe NFE ile eşitlenmeli
(mealpy `model.nfe_counter` / history üzerinden) — epoch ile değil.

## Güncel kısa liste (iki track birleşimi)

| Aday | Saf meta | kmref'li | Not |
|---|---|---|---|
| SMA | 1 | 2 | İki track'te de lider; NFE kontrolü şart |
| AVOA | 4 | 6 | Tutarlı + özgünlük cümlesi (film önerisinde ilk) |
| GWO | 2 | 14 | Saf aramada güçlü + Katarya 2018 literatür kıyası |
| NGO | 5 | 4 | Tutarlı, sessiz güçlü |
| AGTO | 3 | 13 | Saf aramada güçlü |
| HHO | eşik altı | 12 | Yalnız literatür kıyası için tabloda tutulur |

## Sıradaki adım

Final tur bu 6 aday ile: NFE-eşitlenmiş bütçe, 30 seed, hem saf hem kmref'li
metriklerle. Ardından ilk 2–3 aday CF pipeline'ına (fitness=knn_mae varyantı dahil).
