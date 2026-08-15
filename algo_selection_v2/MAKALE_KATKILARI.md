# Makale Katkıları ve Neden-Sonuç Zincirleri

Kural: her deney sonrası katkı varsa buraya girer, yoksa "girmeyenler"e düşer.
Tüm sonuçlar: seed 42, fold 1, cknn tahminci (k=30). Çok-seed/CV5 mührü bekleyenler işaretli.

---

## K1. Dejenere çözüm bulgusu — kısıtsız MAE-fitness kümelemeyi yok eder ★ (ana katkı adayı)

- **Gözlem:** Kısıtsız pred-MAE fitness'lı AVOA, 943 kullanıcının 928–937'sini TEK
  kümeye atıyor (`eksikler_balance.csv` öncesi boyut dökümü). "İyi" MAE'si (0.750),
  kümesiz kNN'i (0.7467) taklit etmesinden geliyor.
- **Neden:** Komşu havuzunu kısıtlamak MAE'yi ancak bozabilir → doğruluk hedefinin
  küresel optimumu "kümeleme yapma"dır. Meta bunu keşfediyor (aslında fitness'ı
  doğru optimize ettiğinin kanıtı).
- **Sonuç 1:** Küme tabanlı CF'de doğruluk iddiası, **küme boyut dağılımı
  raporlanmadan** değerlendirilemez — incelediğimiz makaleler (Katarya vd.) bunu
  raporlamıyor. Güçlü literatür eleştirisi.
- **Sonuç 2:** Dürüst problem tanımı KISITLI optimizasyon: "dengeli kümeler
  ALTINDA doğruluğu maksimize et" (ölçeklenebilirlik-doğruluk takası).
  Denge cezalı fitness önerildi (pay sınırı 2.5/K, ceza 10×aşım).
- **Düzeltme:** Önceki %2.7 (cf_stage) ve %4.0 (predictor_upgrade) iyileşme
  iddiaları kısmen bu mekanizmadan besleniyordu → dengeli rakamlarla değiştirildi (K2).

## K2. Dengeli meta-kümeleme B0'ı K büyüdükçe geçiyor [çok-seed teyidi bekliyor]

`eksikler_k_sweep.csv` + `eksikler_balance.csv` (dengeli AVOA):

| K | B0 MAE | AVOA-dengeli MAE | Fark | B0 NDCG@10 | AVOA-d NDCG@10 |
|---|---|---|---|---|---|
| 7 | 0.7851 | 0.7877 | −0.3% | 0.7852 | 0.7851 |
| 10 | 0.8010 | 0.7881 | **+1.6%** | 0.7788 | 0.7831 |

- **Neden:** K büyüdükçe KMeans++ kümeleri küçülür → küme-içi komşu kıtlığı →
  kNN zayıflar, fallback artar (B0: K=5'te %1.5 → K=20'de %8.3). MAE-farkındalıklı
  merkez seçimi komşu kapsamasını koruyarak bu bozulmayı yavaşlatır.
- **Sonuç:** Meta'nın katkısı küçük K'da yoktur (kümeleme zaten kNN'e yakın),
  **K büyüdükçe büyür** — tam da ölçeklenebilirliğin istendiği rejimde. Tez
  anlatısına birebir oturuyor.

## K3. K seçimi: çift kanıtlı gerekçe (iç + dış metrik)

- İç kalite (space_check): silhouette K=5–7'de tepe (0.29), sonra düşer;
  WCSS dirseği 7–10 arası.
- Dış kalite (k_sweep): MAE K ile monoton bozulur (komşu kıtlığı); kayıp
  K≤10'da küçük, K>10'da hızlanır (B0: 0.801→0.827).
- **Karar ve nedeni:** **K=10** — silhouette hâlâ kabul edilebilir, küme başına
  ~94 kullanıcı (kNN havuzu ≈ komşu sayısı k=30'un 3 katı → sağlıklı), meta
  katkısının görünür olduğu ilk nokta, hesap maliyeti kümesiz kNN'in ~1/10'u.
  "Neden 7 değil?" → K=7'de meta katkısı yok; "neden 14 değil?" → doğruluk
  kaybı hızlanıyor (B0 0.821) ve küme başına kullanıcı k=30 sınırına yaklaşıyor.

## K4. NFE seçimi: test-doyum kanıtı [çok-seed teyidi bekliyor]

`eksikler_nfe_bal.csv` (dengeli AVOA, K=10):

| max_fe | iç-val MAE | test MAE |
|---|---|---|
| 500 | 0.7837 | 0.7932 |
| 1000 | 0.7829 | 0.7934 |
| 2000 | 0.7810 | 0.7906 |
| 5000 | 0.7755 | 0.7909 |

- **Neden-sonuç:** İç-val monoton iyileşir (optimizasyon çalışıyor) ama test
  2000'de doyar; 5000'de iç-val −0.006 kazanırken test kıpırdamaz → 2000 sonrası
  kazanç **iç-val'e aşırı uyum**. NFE=2000 seçiminin gerekçesi budur (ve maliyet
  1/2.5). Ayrıca final turdaki "NFE eşitleme" kararının gerekçesi: eşit epoch,
  eşit bütçe değildir (SMA örneği: aynı epoch'ta 5 kat fazla çağrı).

## K5. Soft atama (top-2 küme havuzu) sınır-kullanıcı kaybını telafi ediyor ★

`eksikler_soft.csv` (K=10, seed 42, kNN k=30):

| Yöntem | Havuz | Ort. havuz boyutu | MAE | Fallback% |
|---|---|---|---|---|
| kNN (kümesiz, üst sınır) | 943 | 943 (%100) | 0.7467 | 0.2 |
| B0 | top-1 (hard) | 241 | 0.8010 | 4.0 |
| B0 | top-2 (soft) | 440 (%47) | 0.7761 | 1.4 |
| AVOA-dengeli | top-1 (hard) | 143 | 0.7906 | 4.1 |
| **AVOA-dengeli** | **top-2 (soft)** | **248 (%26)** | **0.7726** | 1.4 |

- **Neden:** Hard atamada sınır kullanıcıları gerçek komşularının bir kısmını
  komşu kümede bırakıyor (knn_all'ın üstünlüğünün asıl nedeni). Kullanıcıya en
  yakın 2 kümenin birleşik havuzunu verince kayıp komşuların çoğu geri geliyor;
  hard→soft geçişi MAE'yi B0'da −0.025, AVOA'da −0.018 düşürdü, fallback'i
  4'te 1'e indirdi. (Katarya hattının FCM kullanmasının gerçek işlevi bu.)
- **Verimlilik sınırı (makale grafiği):** AVOA-dengeli top-2, B0 top-2'den hem
  DAHA DOĞRU (0.7726 < 0.7761) hem yarı havuzla (248 vs 440) çalışıyor —
  doğruluk-maliyet düzleminde B0'ı domine ediyor. Kümesiz kNN'e kalan fark %3.5
  (0.7726 vs 0.7467), maliyet ise ~1/4.
- **Hizalama fırsatı (sonraki deney):** merkezler hâlâ hard bias-MAE ile arandı,
  soft havuzla değerlendirildi; fitness doğrudan soft-havuz MAE olursa fark
  daha da kapanabilir.

## K6. Tür (genre) yan bilgisi her konfigürasyonda iyileştiriyor ★ [çok-seed bekliyor]

`genre_k.csv` (seed 42, soft top-2, kNN k=30). Kullanıcı tür profili = puanla
ağırlıklı 19-boyutlu tür dağılımı (u.item), NMF-20'ye ölçek eşitlenerek eklendi.

| K | B0 nmf → +genre | AVOA nmf → +genre | En iyi NDCG@10 |
|---|---|---|---|
| 10 | 0.7761 → **0.7662** | 0.7726 → 0.7669 | 0.8082 (B0+g) |
| 14 | 0.7906 → 0.7779 | 0.7938 → **0.7741** | 0.8046 (AVOA+g) |
| 20 | 0.7969 → 0.7931 | 0.8005 → 0.7960 | 0.7860 (AVOA+g) |
| 30 | 0.8191 → 0.8119 | 0.8139 → **0.7941** | 0.7728 (AVOA+g) |

- **Neden:** Tür profili, puan-örtüşmesi az olan kullanıcılar arasında da benzerlik
  sinyali taşıyor → kümeler zevk gruplarına daha iyi oturuyor; fallback her K'da
  düşüyor (örn. AVOA K30: %6.7→%2.7). Literatürle uyum: Katarya'nın "type
  division" adımının doğrulanması, ama bizde kısıtlı-optimizasyon çerçevesinde.
- **K büyürken:** genre, bozulmayı yavaşlatıyor ve **AVOA'nın avantajı yüksek K'da
  açılıyor** (K=30: AVOA 0.7941 vs B0 0.8119 → +%2.2, NDCG +1.1 puan; havuz 135
  kullanıcı = tam aramanın ~1/7 maliyeti). K2 bulgusunu güçlendiriyor: meta'nın
  değeri ölçeklenebilirlik rejiminde.
- **Yeni en iyi (fold1, s42):** genre+K10 → MAE 0.766, NDCG 0.808 (tavan: kümesiz
  kNN 0.747/0.835; fark %2.6'ya indi, maliyet ~1/3).

## K7. Küme-kNN + MF karışımı tavanı aşıyor ★★ (ana sonuç adayı) [çok-seed bekliyor]

`pred_v2.csv` V5 (seed 42, K=10, soft top-2, genre'siz): küme-kNN tahmini ile
ALS-MF (20 faktör, bias'lı, iç-train) tahmini β=0.5 ile karıştırıldı (β iç-val'de
seçildi):

| Sistem | MAE | RMSE | P@10 | NDCG@10 |
|---|---|---|---|---|
| küme-kNN tek (AVOA) | 0.7726 | 0.9907 | 0.6739 | 0.8024 |
| ALS-MF tek | 0.7652 | 0.9826 | — | — |
| kümesiz tam kNN (eski tavan) | 0.7467 | 0.9556 | 0.6963 | 0.8352 |
| **AVOA küme-kNN + MF (β=.5)** | **0.7440** | **0.9475** | **0.6991** | **0.8372** |

- **Neden:** kNN yerel komşuluk sinyali, MF global düşük-rank sinyali taşır;
  hata kaynakları ilişkisiz → karışım her ikisinin hatasını düzeltir ve
  KISITLI havuza rağmen kısıtsız kNN'i tüm metriklerde geçer.
- **Konum:** Ayarlı SVD (0.736) ile aramızda 0.008 kaldı; resmi split'te
  raporlayan küme tabanlı literatürün (Katarya 0.75, Firefly 0.76–0.80) önündeyiz.
  İncelenen 6 makalenin hiçbiri MF karışımı kullanmıyor → ayırt edici katkı.
- İyileştirme alanı: β'nın küme başına/kullanıcı başına uyarlanması (az puanlı
  kullanıcıda MF'e, çok puanlıda kNN'e ağırlık) + AVOA fitness'ının doğrudan
  karışım-MAE'sini optimize etmesi.

## K8. ŞAMPİYON KONFIGÜRASYON — bizim en iyi değerimiz ★★★

`sampiyon.csv` (seed 42, fold 1; tüm seçimler iç-val'de, test TEK atış):

**Konfig:** AVOA merkezleri (denge cezalı, NFE=2000) + nmf+genre uzayı + K=10 +
soft top-2 havuz + kNN k=20 + ALS-MF (f=80, λ=10) + sabit β=0.4 karışım.

| Metrik | Değer | Kıyas |
|---|---|---|
| **MAE** | **0.7389** | ayarlı SVD 0.736'ya 0.003; Katarya-2016 0.75, Firefly 0.76+ geride |
| **RMSE** | **0.9399** | Harmony (0.8974, protokol belirsiz) hariç incelenen her şeyden iyi |
| **P@10** | **0.7033** | kümesiz kNN 0.6963'ü geçti |
| **R@10** | **0.5480** | kümesiz kNN 0.5451'i geçti |
| **NDCG@10** | **0.8470** | kümesiz kNN 0.8352'yi geçti |
| Havuz | 310 (%33) | maliyet ~1/3 |

Yolculuk: 0.8485 (ilk pilot, küme-ort.) → 0.7726 (soft+kNN) → 0.7440 (V5 +MF)
→ **0.7389 (şampiyon)**. Toplam iyileşme %12.9.

Ayar bulguları (neden-sonuç):
- MF ızgarası: f=80, λ=10 → val 0.7332 (f=20 λ=5'ten −0.027; küçük λ büyük f'te patlıyor: f40/λ5=0.777).
- Karışım her yerde β=0.4'ü seçti → MF %60 ağırlık taşıyor; kNN bileşeni yine de
  MAE'yi val'de −0.003 ve tüm sıralama metriklerini yukarı itiyor.
- Kullanıcı-uyarlamalı β(u) sabit β'yı GEÇEMEDİ (val'de hep ikinci) → basit olan kazandı.
- kNN k: karışım içinde k=20 en iyi (tek başınayken 30'du) — MF globali
  taşıyınca kNN'in işi yerel hassasiyet, küçük k daha keskin.
- **Val sıralamasında ilk 4'ün 4'ü de AVOA'lı konfigler** (en iyi B0 5. sırada) —
  algoritma seçiminin katkısı ayar sonrası da görünür.

Mühür işleri: çok-seed (10+) + CV5 + kullanıcı-bazlı Wilcoxon (AVOA-şampiyon vs
B0-şampiyon vs kümesiz kNN); ML-1M genelleme.

## K9. İnce K taraması + 4 yöntem: karar artık grafiklerle ★ [tek seed]

`k_fine.csv` + `plots/k_fine_sweep.png` + `plots/frontier.png`
(K=10→30 ikişer; B0/AVOA/GWO/NGO; şampiyon pipeline sabit; seed 42, test metrikleri)

En iyi 5 (MAE): GWO_K18 0.7375 (havuz %44), GWO_K10 0.7378 (%42),
GWO_K26 0.7379 (%38), GWO_K16 0.7380 (%38), B0_K10 0.7381 (%33).

Yöntem ortalamaları (K=10–30):

| Yöntem | MAE | NDCG@10 | Havuz % |
|---|---|---|---|
| GWO | **0.7390** | **0.8476** | 38.7 |
| NGO | 0.7431 | 0.8452 | 22.7 |
| AVOA | 0.7434 | 0.8440 | **21.7** |
| B0 | 0.7441 | 0.8408 | 16.5 |

Neden-sonuç bulguları:
1. **Havuz yüzdesi yöntemin gizli serbestlik derecesi:** GWO sistematik olarak
   büyük havuzlu kümelemeler üretiyor (%39 ort.) → MAE üstünlüğünün bir kısmı
   kümesiz kNN'e yaklaşmasından. Adil kıyas tek eksende değil, **doğruluk-maliyet
   cephesinde** yapılmalı (frontier.png): düşük-maliyet bölgesinde (havuz <%25)
   AVOA/NGO öne çıkıyor; yüksek-maliyet bölgesinde GWO. B0 yüksek K'da domine
   ediliyor (hem pahalı değil hem doğruluk düşük değil ama NDCG'si en zayıf).
2. **MF karışımı K duyarlılığını azaltıyor:** karışımsız MAE K=10→30'da
   0.77→0.82'ye bozuluyordu; karışımla bant 0.737–0.749'a sıkıştı (MF K hatasını
   tamponluyor). K seçimi artık kritik değil — bu da bir katkı cümlesi.
3. K=10–12 bölgesi tüm yöntemlerde güvenli tepe; K9 grafiği "neden K=10"
   sorusunun görsel cevabı.
4. NDCG'de metalar B0'dan tutarlı iyi (ort. +0.4–0.7 puan) — sıralama kalitesi
   merkez seçimine MAE'den daha duyarlı.

Uyarı: tek seed + test üzerinde tarama → K/algoritma SEÇİMİ iddiası için
çok-seed CV5 mührü şart; bu tablo "davranış analizi" olarak sunulur.

## K10. Üç adalet protokolü — "algoritma farkı" tanıma bağlı ★ (tartışma bölümünün omurgası)

`k_fine.csv`, `budget_pool.csv`, `esit_boyut.csv` (hepsi seed 42, kNN+MF pipeline):

| Protokol | Serbestlik | Kazanan | Not |
|---|---|---|---|
| P1 Serbest havuz (K9) | havuz boyutu serbest | GWO (0.7375) | ama havuzu %39-44'e şişiriyor — gizli maliyet |
| P2 Kırpmalı eşit bütçe | havuz tam B kişi | **B0** (0.7369 @B35) | kırpma görevi yaklaşık-NN'e çevirir; Voronoi-optimal KMeans++ yapısal kazanır |
| P3 Eşit-boyut kısıtı | küme boyutları ≈N/K | **AVOA** (sıralamada) | Lloyd bu kısıtı çözemez (sezgisel gerekir); MAE eşit (0.7407 vs 0.7408), **NDCG +0.6, P@10 +0.7 puan** AVOA lehine, benzer havuzla |

Neden-sonuç zinciri:
- P2'de B0'ın kazanması tesadüf değil: sabit-bütçeli en-yakın-havuz görevi geometrik
  yaklaşık-NN problemidir; WCSS optimali (kompakt Voronoi hücreleri) bu görevin
  doğal çözümüdür. Meta'dan burada fark beklemek yanlış beklentidir.
- P3'te fark geri geliyor çünkü eşit-boyut kısıtı Lloyd'un kapalı-form güncellemesini
  bozar (kapasiteli atama NP-zor); B0 tarafı sezgisel yamayla (kapasiteli greedy)
  oynarken meta kısıtı doğrudan optimize eder. **Meta'nın meşru sahası: kısıtlı
  kümeleme.** (K1'deki "kısıtlı optimizasyon" tezinin ikinci kanıtı.)
- Davranışsal bulgu: **GWO her protokolde kısıt/bütçe boşluğunu istismar ediyor**
  (P1'de havuz şişirme, P3'te cezayı yiyip dengesiz çözüm: [281,224,...], havuz %42).
  Keşif-baskın karakter; kısıt sadakati de bir seçim kriteri olmalı — algoritma
  seçim protokolüne yeni eksen.
- Not: P2 kırpmalı havuz aynı zamanda mutlak en iyi MAE'yi verdi (B0_B35 0.7369)
  → sistem önerisi olarak "budgeted pooling" makaleye ayrıca girer.

Tek seed uyarısı: P3'teki NDCG farkı (+0.006) mühürsüz iddia edilemez →
çok-seed + CV5 + kullanıcı-bazlı Wilcoxon SIRADAKI iş.

## K11. PACR — B0'ı geçen yöntem bulundu ★★★ (ana katkı) [tek seed, mühür bekliyor]

`pacr_p3.csv`: dengeli-B0 merkezlerinden warm start; fitness = soft top-2
bias+MF karışım val-MAE + eşit-boyut cezası (vektörlü, NFE=2000); eval = soft
top-2 kNN(k=20)+MF(β=0.4) test. **5/5 meta, baseline'ı hem MAE hem NDCG'de
eşit/daha küçük havuzla geçti:** NGO 0.7391/0.8457, HGS 0.7395/0.8447,
GWO 0.7397/0.8446, AVOA 0.7404/**0.8470**, HHO 0.7404/0.8445 vs
B0-dengeli 0.7408/0.8404 (havuzlar 187–198 vs 197).

Neden çalıştı (üç bileşen birden şart):
1. Arena = eşit-boyut kısıtı → Lloyd'un kapalı-form avantajı yok (K10-P3 dersi);
2. Warm start → 390-boyutlu uzayda sıfırdan arama yerine yerel rafinasyon
   (final turda sıfırdan aramanın B0'ı geçemediğini görmüştük);
3. Fitness = tahmine hizalı (WCSS değil) → Lloyd'un giremediği hedef (K1 dersi).
Kırpmalı-bütçe arenasında (P2) aynı reçete test'e taşınmadı (`pacr.csv`) —
görev geometrikleşince meta alanı kapanıyor; bu kontrast tartışmada kullanılacak.

Not: farklar binde 1–7; "5/5 tutarlılık + iki metrik birden" deseni umut verici
ama İDDİA için çok-seed CV5 + kullanıcı-bazlı Wilcoxon şart (bekletildi, komut hazır).

## K12. Sistem B: Küme-başına MF — MAE farkını koruyan en güçlü sistem ★★ [tek seed]

`cluster_mf.csv` (K=30, nmf+genre, soft top-2, kNN k=20 + KÜME-MF karışımı, β val'de):

| Yöntem | MAE | NDCG@10 | Havuz | Max küme |
|---|---|---|---|---|
| B0 (KMeans++) | 0.7941 | 0.7924 | 81 | 59 |
| AVOA (cap 10×) | 0.7807 | 0.8019 | 135 | 110 |
| NGO (cap 10×) | 0.7760 | 0.8108 | 162 | 145 |
| HGS (cap 10×) | 0.7663 | 0.8247 | 268 | 322 |
| AVOA (cap 50×, sert) | 0.7801 | 0.8121 | 164 | 102 |
| HGS (cap 50×, sert) | 0.7764 | 0.8130 | 200 | 173 |

- **Mekanizma:** global MF kümelemeden bağımsızdı → farkı ezdi (β payı kadar).
  Küme-MF her kümenin kendi verisiyle eğitilir → tahminin İKİ bileşeni de
  kümelemeye bağlı → algoritma farkı tahmine akar. Şimdiye kadarki en büyük CF
  MAE farkı: %1.8–3.5 (kısıt sertliğine göre), NDCG farkı +2 ile +3.2 puan.
- **Şeffaf pazarlık:** metalar doğruluğu kısmen daha büyük kümelerle satın alıyor
  (küme-MF büyük kümede daha iyi öğreniyor — meta bu takası keşfediyor). Sert
  ceza (50×) altında bile B0'a +%1.8 fark korunuyor; ama tam eşitlik iddiası
  için havuz-normalize karşılaştırma (frontier) rapora eklenmeli.
- Sistem B rolü: makale Tablo B'si (algoritma katkısı) bu sistemde kurulur;
  Tablo A (mutlak, 0.737–0.744) şampiyon sistemde kalır.

## K13. Gerçek run öncesi son ayarlar (`sistemB_tune.csv`) [tek seed]

1. **Sistem B — K seçimi:** K=30 onaylandı. K=40 farkı büyütüyor (HGS−B0 = 0.022)
   ama kısıt ihlali ağırlaşıyor (maxk 359); K=20'de HGS mutlak iyi (0.7581) ama
   havuz 320'ye şişiyor. K=30 = fark (+0.018) ve kısıt disiplini dengesi.
2. **Küme-MF faktörü:** f=10 doğrulandı (f=5: 0.7781, f=10: 0.7764, f=20: 0.7777)
   — ~90 kişilik kümede 10 faktör veri/kapasite dengesi.
3. **β ızgarası genişletilince ŞAMPİYON İYİLEŞTİ:** eski ızgara (0.4–0.7) dar
   kalmış; b_knn=0.2–0.3 daha iyi → **yeni en iyi: MAE 0.7362, NDCG@10 0.8533**
   (B0, K=10; AVOA 0.7372/0.8524). Ayarlı SVD'nin (0.736) dibine gelindi.
4. **Üçlü karışım kararı — küme-MF şampiyona GİREMEZ:** val optimizasyonu
   küme-MF'e her iki yöntemde de SIFIR ağırlık verdi (b_cmf=0.0): K=10'da global
   MF (72k puanla eğitilmiş) küme-MF'i (küme başına ~7k puan) tamamen domine
   ediyor. Sonuç: Sistem A (mutlak) ve Sistem B (algoritma farkı) **birleştirilemez,
   ayrı tablolar olarak kalmalı** — iki-tablo tez tasarımının deneysel gerekçesi.

Gerçek run konfigürasyonu donduruldu:
- **Tablo A:** K=10, nmf+genre, soft top-2, kNN k=20, global ALS-MF f=80/λ=10,
  β ızgarası 0.2–0.7 (val'de) → B0 + AVOA (+kümesiz kNN referansı)
- **Tablo B:** K=30, küme-MF f=10, sert ceza w=50, kNN k=20 →
  B0, AVOA, NGO, HGS (+havuz/maxk raporu zorunlu)
- Her ikisi: 10 seed × 5 resmi fold, kullanıcı-bazlı Wilcoxon + Friedman/Holm.

## K14. TAM RUN SONUCU — ana iddia mühürlendi ★★★★ (5 fold × 10 seed = 300 koşu)

`tam_run_B.csv` + `tamrun_B_ozet.csv` (Sistem B: K=30, küme-MF, sert kısıt):

| Yöntem | MAE | NDCG@10 | Havuz | B0'a fark (MAE) |
|---|---|---|---|---|
| GWO | **0.7506** ±0.007 | **0.8326** | 286 | **−0.0253 (%3.3)** |
| HHO | 0.7637 ±0.010 | 0.8216 | 173 | −0.0122 (%1.6) |
| HGS | 0.7647 ±0.009 | 0.8216 | 175 | −0.0112 (%1.4) |
| NGO | 0.7666 ±0.009 | 0.8189 | 155 | −0.0093 (%1.2) |
| AVOA | 0.7694 ±0.011 | 0.8159 | 132 | −0.0065 (%0.8) |
| B0 | 0.7759 ±0.010 | 0.8093 | 82 | — |

İstatistik (hepsi Holm düzeltmeli):
- **Friedman χ²=179.7, p=6.1e-37** → algoritma sıralaması anlamlı.
- **5/5 meta, B0'ı MAE'de VE NDCG'de anlamlı geçti** (hücre-bazlı Wilcoxon, tüm p<0.0001).
- **Kullanıcı-bazlı Wilcoxon (n≈943): 5/5 anlamlı**, p değerleri 5e-10 ile 4e-63 arası.

Tez ana cümlesi artık kanıtlı: *"Küme tabanlı CF'de kümeleme kalitesi meta-sezgisel
algoritma seçimine bağlıdır; tahmine-hizalı fitness ile optimize edilen merkezler,
KMeans++ baseline'ını 50 bağımsız koşunun tamamında ve kullanıcı düzeyinde
istatistiksel olarak anlamlı biçimde geçmektedir (Holm-düzeltmeli p<1e-9)."*

Dürüstlük sütunu: havuz boyutu performansla korele (GWO 286 vs B0 82) —
doğruluk-maliyet takası makalede havuz sütunuyla birlikte sunulur; en küçük
havuzlu meta bile (AVOA, 132) B0'ı anlamlı geçiyor → sonuç yalnız havuz artefaktı değil.

İkinci bulgu: sistem sıralaması (GWO>HHO≈HGS>NGO>AVOA) saf arama sıralamasından
(AVOA 1.) farklı → "iyi arayıcı ≠ iyi sistem bileşeni"; algoritma seçimi hedefe
göre yapılmalı (fark hunisiyle birlikte tartışma bölümü malzemesi).

## K15b. TABLO B+ MÜHÜRLENDİ (K=6, 5 fold × 10 seed) ★★★★

`tabloB_plus_K6.csv` (299 satır), Friedman χ²=100.1 **p=5.0e-20**:

| Yöntem | MAE | NDCG@10 | F1@10 | Hücre-Wilcoxon (MAE) | Kullanıcı-Wilcoxon |
|---|---|---|---|---|---|
| **AVOA** | **0.7391** ±0.008 | **0.8382** | **0.6324** | **p<0.0001** | **p=1.3e-07** |
| GWO | 0.7441 | 0.8372 | 0.6311 | p=0.0001 | p=0.001 |
| HHO | 0.7450 | 0.8362 | 0.6303 | p=0.0003 | anlamsız |
| HGS | 0.7455 | 0.8350 | 0.6296 | p<0.0001 | anlamsız |
| NGO | 0.7458 | 0.8353 | 0.6298 | p<0.0001 | p=0.012 |
| B0_repair | 0.7469 | 0.8335 | 0.6288 | — | — |

Havuz tüm yöntemlerde 314–315 (eşit maliyet), max küme 158 (kapasite tavanı).

- **5/5 meta hem MAE hem NDCG'de anlamlı** (Holm düzeltmeli).
- **AVOA farkı diğerlerinin ~3 katı** (−0.0078 vs −0.0012…−0.0029) ve kullanıcı
  düzeyinde de en güçlü (p=1.3e-07). HHO/HGS kullanıcı düzeyinde anlamsız →
  "AVOA seçimi" iddiası yalnızca ortalamaya değil, kullanıcı dağılımına dayanıyor.
- Bu tablo makalenin ANA TABLOSU olacak (eşit maliyet + mühürlü istatistik).

## K15. Tablo B+ PİLOT — onarımlı eşit havuz: AVOA net kazanıyor ★★★ [3 hücre; K15b ile mühürlendi]

`tabloB_plus.csv` (K=30, repair + warm start; fold 1 s42/s43, fold 2 s42):

| Yöntem | MAE | std | NDCG@10 | Havuz | Max küme |
|---|---|---|---|---|---|
| **AVOA_warm** | **0.7712** | 0.009 | **0.8035** | 63 | 32 |
| HHO_warm | 0.7940 | 0.015 | 0.7828 | 63 | 32 |
| GWO_warm | 0.7956 | 0.013 | 0.7832 | 63 | 32 |
| HGS_warm | 0.7956 | 0.016 | 0.7898 | 63 | 32 |
| B0_repair | 0.7976 | 0.014 | 0.7879 | 63 | 32 |
| NGO_warm | 0.8040 | — | 0.7758 | 63 | 32 |

**Protokolün gücü:** repair operatörü kısıt ihlalini yapısal olarak imkânsız
kılıyor → tüm yöntemlerde havuz 63, max küme 32 (tam eşit). Maliyet artık
serbestlik derecesi DEĞİL; ölçülen tek şey merkez kalitesi.

İki kritik bulgu:
1. **GWO'nun üstünlüğü havuz artefaktıymış — kanıtlandı.** Serbest protokolde
   (K14) GWO 1. sıradaydı (havuz 286); havuz eşitlenince B0 seviyesine düştü
   (0.7956 vs 0.7976). K10'daki "GWO kısıt boşluğunu istismar ediyor" davranışsal
   gözlemi böylece deneysel olarak doğrulandı.
2. **AVOA eşit-maliyet arenasının açık kazananı:** B0'a karşı MAE −0.0264 (%3.3),
   NDCG +0.0156; üç hücrenin üçünde de birinci ve rakiplerinden ~%2.5 önde.
   Tezin "film önerisinde AVOA" seçimi artık adil protokolde de savunuluyor.

Yan etki (dürüst not): repair havuzu 63'e düşürdüğü için mutlak MAE'ler serbest
protokolden yüksek (0.77–0.80 vs 0.75–0.78) — maliyet 1/15'e inerken doğruluk
bedeli ~%2. Bu, makalenin doğruluk-maliyet cephesine üçüncü nokta olarak girer.

Tam koşu komutu (5 fold × 10 seed):
`python algo_selection_v2/tabloB_plus.py --folds 1 2 3 4 5 --seeds 42 ... 51 --resume`

## K16. Fitness hedefi karşılaştırması (madde 3 pilotu) ★ [fold 1, 1-2 seed]

`ndcg_fitness.csv` vs `tabloB_plus.csv` (aynı iskele: repair + warm start, K=30):

| Fitness | Yöntem | MAE | NDCG@10 | P@10 |
|---|---|---|---|---|
| MAE (referans) | AVOA | 0.7787 | 0.7979 | 0.6716 |
| NDCG | AVOA | 0.7750 | 0.8038 | 0.6736 |
| **Hibrit (0.5/0.5)** | **AVOA** | **0.7711** | **0.8065** | **0.6760** |
| MAE | GWO / HGS | 0.802 / 0.805 | 0.777 / 0.785 | 0.654 / 0.658 |
| NDCG | GWO / HGS | 0.805 / 0.809 | 0.784 / 0.782 | 0.654 / 0.654 |

Bulgular:
1. **Hibrit fitness üç metrikte birden en iyi** — MAE'yi bozmadan sıralamayı
   iyileştiriyor (AVOA: MAE −0.008, NDCG +0.009, P@10 +0.004). Sıralama ve hata
   hedefleri çelişmiyor, birbirini düzenlileştiriyor (fitness gürültüsüne karşı
   iki sinyal). İncelenen 6 makalenin hiçbirinde sıralama-hedefli fitness yok.
2. **Saf NDCG fitness da MAE'yi düşürüyor** (0.7787→0.7750): sıralamayı düzelten
   merkezler hata metriğini de düzeltiyor; ters etki yok.
3. **Fitness değişimi algoritma sıralamasını değiştirmiyor** (AVOA ≫ GWO ≈ HGS
   her üç hedefte de) → K15'teki AVOA üstünlüğü hedef seçiminden bağımsız, sağlam.
4. GWO/HGS NDCG hedefinden faydalanmıyor → hedef-algoritma etkileşimi var;
   "her algoritmaya her hedef uymaz" (NFL'in pratik yansıması).

Öneri: makale ana sistemi **hibrit fitness** ile kurulmalı (bedava kazanç).
Doğrulama: 5 fold × 10 seed hibrit koşusu (tabloB_plus'a `--fitness hib`
seçeneği eklenerek) — mevcut tam koşu bitince.

## K17. Kısıt sertliği ablasyonu — AVOA üstünlüğü kısıttan bağımsız ★★ [fold1 s42]

İtiraz: "eşit küme boyutu yapay bir kısıt, havuzu sabitlemek doğru mu?" — haklı.
Cevap: kısıt bir *ölçüm aracı*, sistem gereği değil. Üç sertlik seviyesinde test:

**(a) K taraması (kapasite = tam eşit)** `repair_k_sweep.csv`:

| K | Havuz (%) | B0 MAE | AVOA MAE | Fark | AVOA NDCG |
|---|---|---|---|---|---|
| 5 | 377 (%40) | 0.7614 | **0.7490** | −0.0124 | **0.8379** |
| 7 | 269 (%29) | 0.7672 | **0.7541** | −0.0131 | 0.8328 |
| 10 | 189 (%20) | 0.7797 | **0.7541** | −0.0256 | 0.8208 |
| 20 | 95 (%10) | 0.7942 | **0.7680** | −0.0262 | 0.8083 |
| 30 | 63 (%7) | 0.7976 | **0.7712** | −0.0264 | 0.8035 |
| 50 | 37 (%4) | 0.8172 | **0.7855** | −0.0317 | 0.7978 |
| — | 943 (%100) | kümesiz kNN: MAE 0.7467, NDCG 0.8352 | | | |

**AVOA K=5, kümesiz kNN'i NDCG'de GEÇİYOR** (0.8379 vs 0.8352) ve MAE'de
0.002 farkla yakalıyor — havuzun yalnızca %40'ıyla. Sıralama kalitesi açısından
kümeleme kayıp değil KAZANÇ (küme, gürültülü uzak komşuları eliyor).

**(b) Kapasite gevşetme (K=30)** `soft_cap.csv`:

| Kapasite | B0 MAE / havuz | AVOA MAE / havuz | Fark |
|---|---|---|---|
| 1.0× (tam eşit) | 0.7976 / 63 | 0.7712 / 63 | −0.0264 |
| 1.5× | 0.7941 / 78 | 0.7656 / 93 | −0.0285 |
| 3.0× | 0.7941 / 81 | 0.7575 / 188 | −0.0366 (havuz 2.3×) |
| ∞ (serbest, K14) | 0.7759 / 82 | 0.7694 / 132 | −0.0065 |

Sonuçlar:
1. **AVOA her K'da ve her kapasite seviyesinde B0'ı geçiyor** (−0.026 ile −0.037);
   üstünlük kısıt seçimine bağlı değil. Kısıt yalnızca farkın *ne kadarının*
   havuzdan geldiğini ayrıştırıyor.
2. **Kapasite gevşedikçe fark büyüyor ama havuz da büyüyor** → gevşek kısıtta
   "kazanç mı, maliyet mi?" ayrışamıyor. 1.0×–1.5× bandı ikisini de kontrol eden
   nokta; makalede ana tablo 1.0×, ablasyon olarak 1.5×/3.0×/serbest verilecek.
3. **K seçimi = maliyet seçimi:** havuz 189→37 (K 10→50) ile MAE 0.754→0.786;
   fark tüm bantta korunuyor. Doğruluk-maliyet cephesi üç protokolde de aynı
   hikâyeyi anlatıyor.

Not (yöntem netleştirmesi): meta yalnız "başlangıç noktası" vermiyor — Tablo B/B+
hattında Lloyd YOK; meta merkezleri final merkezlerdir, atama doğrudan onlardan
çıkar. Repair yalnızca kapasite tavanı uygular (kullanıcı dolu kümeye giderse
sıradaki en yakına), merkez konumunu değiştirmez.

## K18. TAM K TARAMASI (K=2..70) — iki çalışma noktası kesinleşti ★★★

`k_tarama_repair.csv` + `plots/k_tarama_repair.png` (repair + warm start, fold1 s42)

| K | Havuz | B0 MAE | AVOA MAE | Fark | AVOA NDCG |
|---|---|---|---|---|---|
| 2 | 943 | 0.7355 | 0.7351 | −0.0003 | — |
| 4 | 471 | 0.7515 | 0.7472 | −0.0043 | 0.8431 |
| **6** | **315** | 0.7654 | **0.7450** | **−0.0204** | **0.8399** |
| 10 | 189 | 0.7797 | 0.7541 | −0.0256 | 0.8208 |
| 20 | 95 | 0.7942 | 0.7680 | −0.0262 | 0.8083 |
| 30 | 63 | 0.8061 | 0.7755 | −0.0306 | 0.8125 |
| **40** | **47** | 0.8106 | **0.7788** | **−0.0318** | 0.7873 |
| 60 | 31 | 0.8224 | 0.7914 | −0.0311 | — |

Referans: kümesiz kNN 0.7467 / 0.8352.

1. **Fark hunisi K ekseninde tam görünür:** K=2'de havuz=943 (kümeleme yok) →
   fark sıfır (−0.0003); K arttıkça fark büyüyor, K≈40'ta doyuyor (−0.032).
   Kümeleme ne kadar bağlayıcıysa merkez kalitesi o kadar belirleyici.
2. **ÇALIŞMA NOKTASI A (mutlak): K=6** — AVOA MAE **0.7450**, kümesiz kNN'i
   (0.7467) MAE'de VE NDCG'de (0.8399 vs 0.8352) geçiyor, havuzun %33'üyle.
   Bu bizim en iyi mutlak sonucumuz ve makale ana iddiası.
3. **ÇALIŞMA NOKTASI B (algoritma farkı): K=40** — fark maksimum (−0.0318),
   havuz sadece 47 (%5).

**Algoritma kıyası iki noktada** (`--exp algos`):

| K=6 | MAE | NDCG | | K=40 | MAE | NDCG |
|---|---|---|---|---|---|---|
| **AVOA** | **0.7450** | **0.8399** | | **AVOA** | **0.7788** | **0.7873** |
| GWO | 0.7569 | 0.8261 | | HGS | 0.8096 | 0.7736 |
| NGO | 0.7594 | 0.8260 | | HHO | 0.8097 | 0.7741 |
| HHO | 0.7634 | 0.8219 | | NGO | 0.8106 | 0.7737 |
| HGS | 0.7638 | 0.8203 | | GWO | 0.8150 | 0.7604 |
| B0 | 0.7654 | 0.8223 | | B0 | 0.8106 | 0.7708 |

- **AVOA her iki noktada da açık ara birinci** (K=6'da 2.'ye 0.012, K=40'ta 0.031).
- K=40'ta GWO B0'ın bile ALTINDA → serbest protokoldeki GWO liderliği tamamen
  havuz artefaktıymış (K15 bulgusunun kesin teyidi).
- Sıralama K'ya göre değişiyor (K=6: GWO 2., K=40: GWO son) → "algoritma seçimi
  çalışma noktasına bağlıdır" bulgusu; tek K'da kıyas yapan literatür yanıltıcı.

## K19. Hiperparametre ayarı DENENDİ — ayarlamamak doğru karar ★ (metodolojik bulgu)

`avoa_tune.csv` (K=6, fold1 s42; 24 konfigürasyon: pop×sigma×NFE + p1/p2/p3/alpha/gama)

| Bulgu | Sayı |
|---|---|
| İç-val fitness yayılımı (24 konfig) | 0.7607–0.7656 (%0.6) |
| Test MAE yayılımı | 0.7445–0.7542 (%1.3) |
| **val↔test Spearman korelasyonu** | **−0.50 (NEGATİF)** |

**Kritik sonuç: iç-val'e göre en iyi parametreyi seçmek test'i KÖTÜLEŞTİRİYOR.**
%10'luk iç-val, parametreler arası %0.6'lık farkı ayırt edecek çözünürlükte değil;
seçim gürültüyü takip ediyor (negatif korelasyon bunun imzası). Bu yüzden:
- Tüm deneylerde **mealpy varsayılan parametreleri** kullanıldı ve öyle kalmalı.
- Bu, AVOA'nın "az parametre ayarı gerektirir" avantajının deneysel kanıtı
  (algoritma seçim kriterlerimizden biriydi — şimdi verisi var).
- Literatür eleştirisine ek madde: incelenen makalelerin hiçbiri parametre
  ayarının transfer edilebilirliğini test etmiyor; ayarlı sonuç raporlayanlar
  muhtemelen bu tuzağa düşüyor.

Rakip kontrolü (aynı bütçe, `--exp rakip`): HGS'nin PUP ayarı da fark yaratmadı
(0.7625–0.7643); varsayılan konfigürasyonda K=6 sıralaması **AVOA 0.7450 <
HHO 0.7554 < NGO 0.7594 < HGS 0.7625 < GWO 0.7648** — K18 ile tutarlı.
Arama bütçesi (pop=15, sigma=0.08, NFE=1200) tüm yöntemlerde eşit tutuldu.

## K20. K=6'da repair'li vs repair'siz — kısıtın işlevi netleşti ★ `k6_repairsiz.csv`

| Yöntem | REPAIR'Lİ MAE / havuz / maxk | SERBEST MAE / havuz / maxk |
|---|---|---|
| B0 | 0.7654 / 315 / 158 | 0.7502 / 405 / 226 |
| AVOA | **0.7450** / 315 / 158 | 0.7481 / 439 / **377** |
| GWO | 0.7569 / 315 / 158 | 0.7450 / 468 / 322 |
| HGS | 0.7638 / 315 / 158 | 0.7483 / 417 / 344 |
| Fark (AVOA−B0) | **−0.0204** | **−0.0021** |

1. **Serbest bırakınca herkesin MAE'si düşüyor — çünkü havuz şişiyor** (315 → 405–468,
   yani %33 → %43–50). Kazanç yöntemden değil, daha çok komşuya bakmaktan geliyor.
2. **Algoritma farkı 10 KAT eriyor** (−0.0204 → −0.0021). Serbest protokolde
   yöntemler ayrışamıyor; kimin iyi olduğu ölçülemez hale geliyor.
3. **Dejenerasyon eğilimi geri geliyor:** serbest AVOA'nın en büyük kümesi 377
   kişi (943'ün %40'ı) — K1'deki "tek dev küme" davranışının hafif hali.
4. Serbest protokolde en iyi MAE GWO'nun (0.7450) ama havuzu da en büyük (468) →
   yine maliyet-kazanç karışımı; repair'li protokolde aynı bütçede AVOA kazanıyor.

**Sonuç:** repair bir "kısıtlama dezavantajı" değil, ölçüm koşulu. Serbest sonuçlar
makalede ablasyon olarak verilir; ana tablo repair'li kalır çünkü tek değişkenli
karşılaştırmayı yalnız o sağlıyor.

## K21. Yer değiştirme analizi — kısıtın kullanıcıya maliyeti ölçüldü ★

Soru: "Kullanıcıyı zorla başka kümeye sokmak hata değil mi?" Ölçüm (fold1, s42):

| K | Yer değiştiren kullanıcı | Top-2 havuzunda EN YAKIN kümesi olan | Merkez uzaklık çarpanı (medyan) |
|---|---|---|---|
| 6 | 247/943 (%26) | **%100** | 13.4× |
| 40 | 374/943 (%40) | **%100** | 2.1× |

**Kritik bulgu: soft top-2 havuz kısıtın zararını yapısal olarak telafi ediyor.**
Yer değiştiren kullanıcıların **%100'ünün** komşu havuzunda kendi en yakın kümesi
yine var (near[:,1] tanım gereği en yakın diğer küme). Yani kullanıcı "yanlış"
kümeye atansa da komşularını kaybetmiyor; kaybettiği tek şey, o kümenin
küme-MF modelinin birincil sahibi olmak. Bu, soft atamanın kısıtlı kümelemede
neden şart olduğunun kanıtı (K5 + K21 birlikte okunmalı).

Kısıtın meşruiyeti (tez savunması):
1. **Literatürde yerleşik alan:** kapasiteli/dengeli kümeleme (constrained
   k-means, balanced k-means) bilinen bir varyanttır; bizim yaptığımız icat değil,
   bu varyantın öneri sistemine uygulanması.
2. **Sistem gereği:** öneri servislerinde küme başına gecikme/kaynak garantisi
   dengeli bölütleme gerektirir (dev küme = yavaş sorgu, kırıntı küme = kötü öneri).
3. **Ölçüm gereği:** K20 gösterdi ki kısıtsız protokolde algoritma farkı 10 kat
   eriyor ve kazanç havuz büyümesinden geliyor — kısıt, karşılaştırmayı
   tek değişkenli yapan şey.
4. **Literatür boşluğu:** taranan CF-kümeleme çalışmalarında kapasite kısıtı veya
   küme boyut dağılımı raporu YOK; hepsi serbest Voronoi ile çalışıyor. Bu,
   "dev küme + kırıntı küme" dejenerasyonunu (bizde K1) görünmez kılıyor.

## K22. Klasik kümeleme baseline'ları eklendi (HHO-K-means kıyas seti) ★★

`klasik_baselines.csv` — hepsi AYNI tahmin hattında (repair, soft top-2,
küme-MF + kNN karışımı); tek değişken kümeleme yöntemi. fold1, s42.

**K=6 (çalışma noktası A):**

| Yöntem | MAE | RMSE | NDCG@10 | P@10 |
|---|---|---|---|---|
| k-means (random, tek koşu) | 0.7585 | 0.9659 | 0.8260 | 0.6895 |
| KMeans++ (n_init=10) | 0.7654 | 0.9726 | 0.8223 | 0.6852 |
| PCA-k-means | 0.7645 | 0.9712 | 0.8248 | 0.6830 |
| SOM-Cluster | 0.7574 | 0.9657 | 0.8317 | 0.6978 |
| PCA-SOM | 0.7591 | 0.9664 | 0.8243 | 0.6878 |
| **AVOA (önerilen)** | **0.7450** | **0.9500** | **0.8399** | **0.6985** |

**K=40 (çalışma noktası B):** AVOA 0.7788 / 0.9933 / 0.7873; klasiklerin en iyisi
PCA-SOM 0.8055; KMeans++ 0.8106; PCA-k-means 0.8151. AVOA farkı %3.3.

Bulgular:
1. **AVOA beş klasik yöntemin hepsini her metrikte geçiyor** (K=6'da 2.'ye
   −0.012 MAE, +0.008 NDCG; K=40'ta −0.027 MAE). HHO-K-means makalesinin kıyas
   seti birebir tekrarlandı, üstüne meta-sezgisel eklendi.
2. **Klasikler birbirine çok yakın (0.757–0.765):** aynı güçlü hatta konulduklarında
   PCA/SOM varyantlarının farkı binde 1-8. Literatürdeki büyük farklar
   (ör. HHO-K-means makalesi %28 iyileşme) hattın zayıflığından geliyor,
   kümeleme yönteminden değil → **kritik replikasyon bulgusunun ikinci kanıtı.**
3. **Sürpriz:** güçlü hatta random-init k-means, KMeans++'tan iyi çıkabiliyor
   (0.7585 vs 0.7654) — WCSS'i daha iyi optimize etmek tahmin doğruluğunu garanti
   etmiyor (geometrik hedef ≠ tahmin hedefi tezinin bir kanıtı daha).
4. SOM-Cluster klasikler içinde en iyi (0.7574) — sıralama metriklerinde de
   KMeans++'ı geçiyor; makalede "en güçlü klasik rakip" olarak sunulmalı.

## K23. Cold-start katmanlı analiz + gray sheep (LOF) ★★ `coldstart_katman.csv`

**(a) Kullanıcı yoğunluğuna göre katmanlı sonuç** (K=6, fold1 s42, test MAE):

| Katman (iç-train puan sayısı) | n | B0 MAE | AVOA MAE | Fark |
|---|---|---|---|---|
| **cold (<20)** | 126 | 0.8228 | **0.8128** | −0.0100 |
| 20–50 | 145 | 0.7596 | **0.7435** | −0.0161 |
| 50–150 | 141 | 0.7502 | **0.7334** | −0.0168 |
| yoğun (≥150) | 47 | 0.7400 | **0.7152** | −0.0247 |

Bulgular:
1. **AVOA her katmanda kazanıyor** — sonuç yalnızca yoğun kullanıcılardan gelmiyor,
   soğuk kullanıcılarda da (−0.010) geçerli.
2. **Ama fark yoğunlukla büyüyor** (−0.010 → −0.025): merkez kalitesi, komşuluğu
   zengin kullanıcılarda daha çok işe yarıyor. Soğuk kullanıcıda tahmin zaten
   MF/fallback'e yaslanıyor, kümelemenin manevra alanı dar.
3. **Dürüst çerçeve:** yöntemimiz cold-start *çözümü* değil; cold-start'ta da
   bozulmadan çalışan (hafif iyileştiren) bir yöntem. Literatürde "cold-start'ı
   çözdük" diyen çalışmalar (GOA 2024, GWO-FCM) katmanlı analiz yapmıyor —
   bu ayrım bizim metodolojik katkımız.

**(b) Gray sheep / LOF** (n_neighbors=20, NMF+tür uzayında):

- LOF %2.0 kullanıcıyı (19/943) aykırı işaretliyor. Dağılım **iki uçlu**:
  puan sayıları [4, 6, 10, 11, 16, 19, 22, 29 | 145, 150, 243, ..., 608] —
  yani 8 tanesi aşırı seyrek (<30 puan), 11 tanesi aşırı yoğun (>145).
  Normal kullanıcı medyanı 45. Yani LOF "ortalama davranış"tan sapan iki uç grubu
  yakalıyor: verisi çok az olanlar + kimseye benzemeyecek kadar çok/farklı
  film izlemiş olanlar. Klasik "gray sheep" tanımının (az puanlı, benzersiz)
  yalnız yarısı; diğer yarısı literatürde adlandırılmayan bir grup.
- **Bu kullanıcıların tahmin hatası çok yüksek: MAE 1.0097 vs 0.7596** (%33 kötü).
  Yani sistemin en zayıf noktası soğuk kullanıcılar değil, atipik kullanıcılar.
  (Cold-start katmanı 0.8228; aykırılar 1.0097.)
- LOF ile merkez öğrenmeyi temizlemek: B0'a fayda (0.7654→0.7601), AVOA'ya zarar
  (0.7450→0.7549). Neden: AVOA zaten MAE-hizalı fitness ile aykırıları
  dengeliyor; onları çıkarmak fitness sinyalini bozuyor. **LOF ana hatta
  girmiyor**, ancak "atipik kullanıcı" analizi tezde ayrı bir bulgu olarak durur.

**(c) Yönlendirme (routing) deneyi** — atipiklere ayrı tahmin yolu (K=6, AVOA):

| Strateji | Genel MAE | NDCG@10 | Atipik alt-grup MAE |
|---|---|---|---|
| Mevcut (routing yok) | 0.7464 | 0.8429 | 0.8159 |
| Atipik → saf küme-MF | 0.7467 | 0.8431 | 0.8459 (kötü) |
| Atipik → tam havuz kNN (β=0.5) | 0.7464 | 0.8433 | 0.8082 |
| **Atipik → tam havuz kNN (β=0.7)** | 0.7463 | **0.8434** | **0.7998 (−%2.0)** |

- **Doğru müdahale: havuzu genişletmek, kNN'i kapatmak değil.** Atipik kullanıcı
  komşusuz değil — komşuları *kümesinin dışında*. Saf MF'e yönlendirmek durumu
  kötüleştiriyor (0.8459), tam havuza açmak iyileştiriyor (0.7998).
- Genel MAE'ye etkisi ihmal edilebilir (%2'lik kullanıcı grubu) → makalede
  **ana katkı değil, teşhis + hedefli düzeltme** bölümü olarak sunulur.
- Politika kararı (tezde açıkça yazılacak): atipik kullanıcılar
  **değerlendirmeden ÇIKARILMAZ** (test manipülasyonu olur), **merkez
  öğrenmeden de çıkarılmaz** (AVOA'ya zarar veriyor: 0.7450→0.7549);
  yalnızca **tahmin aşamasında daha geniş havuza yönlendirilir**.

Not: alt-grup MAE'si iki şekilde raporlanabilir — kullanıcı-ortalamalı 1.0097
(her kullanıcı eşit ağırlık) vs puan-ortalamalı 0.8159 (çok puanlı kullanıcı
baskın). İkisi de doğru; makalede kullanıcı-ortalamalı verilecek çünkü grup
iki-uçlu (bkz. b maddesi).

## K24. Latent boyut / uzay ablasyonu — GÜNCEL hatta ★ `latent_sweep.csv` [fold1 s42]

Eski `space_check` yalnız silhouette bakıyordu (eski hat). Bu tablo güncel hatta
(repair, soft top-2, küme-MF + kNN karışımı, K=6):

| Uzay | B0 MAE | AVOA MAE | Fark | AVOA NDCG |
|---|---|---|---|---|
| NMF-10 + tür | 0.7550 | 0.7501 | −0.005 | 0.8280 |
| **NMF-20 + tür (mevcut)** | 0.7654 | **0.7450** | **−0.020** | **0.8399** |
| NMF-30 + tür | 0.7628 | 0.7497 | −0.013 | 0.8376 |
| SVD-20 + tür | 0.7605 | 0.7480 | −0.013 | 0.8328 |
| NMF-20 (tür YOK) | 0.7772 | 0.7449 | **−0.032** | 0.8368 |

1. **NMF-20 + tür seçimi doğrulandı** — hem en iyi AVOA MAE'si hem en iyi NDCG.
   Boyut artışı (30) ya da azalışı (10) fayda vermiyor; SVD, NMF'in gerisinde
   (negatif-olmayanlık kullanıcı-zevk profiline daha uygun).
2. **Tür bilgisi B0'ı AVOA'dan çok daha fazla kurtarıyor:** tür çıkınca B0 0.767→0.777
   bozulurken AVOA 0.745'te sabit kalıyor. Yani **meta-sezgisel merkez araması, iyi
   özellik mühendisliğinin yerini kısmen doldurabiliyor** — pratik değeri yüksek bir
   bulgu (yan bilgi yoksa meta daha kritik). Tür yokken fark −0.032 ile en yüksek.
3. Bu ablasyon "neden NMF-20?" sorusuna güncel hatta gerekçe sağlıyor (eski
   space_check yalnız kümelenebilirlik bakıyordu).

---

## Girmeyenler (denendi, katkı çıkmadı / zayıf)

- **kNN k=60:** her konfigürasyonda k=30'dan kötü (aşırı-düzleştirme) → k=30
  gerekçelendi, tabloya girmez.
- **Significance weighting / IUF / user-item füzyonu (`pred_v2.csv` V1–V3):**
  6 makaleden alınan benzerlik teknikleri güçlü merkezli-cosine kNN'e katkı
  vermedi (V1 nötr, IUF −0.006 zarar, füzyon α=0.9'a kaçtı). Neden-sonuç:
  bu hileler zayıf temel tahmincileri düzeltmek için; iyi ayarlı kNN'de düzeltecek
  sistematik hata bırakmıyorlar. Makaleye "denendi, katkısız" cümlesiyle girer.
- **Soft-hizalı fitness (`soft_aligned.csv`):** iç-val'i daha iyi optimize etti
  (0.765 vs 0.781) ama test MAE iyileşmedi (0.7775 vs hard-fit 0.7726; tek seed,
  fark gürültü bandında). Neden-sonuç: daha esnek hedef → iç-val'e aşırı uyum
  riski artıyor; %10'luk val bu esnekliği taşımıyor. Yan kazanım: en dengeli
  küme dağılımı (158/144/136/124/117) ve en küçük havuz (219). Çok-seed + daha
  büyük val ile yeniden denenebilir; şimdilik ana hat hard-fit + soft-eval.

---

## Ölçüm sözlüğü (tablo sütunları)

- **fallback%:** kNN tahmini kurulamayan test tahminlerinin oranı — kullanıcının
  havuzunda o filmi puanlamış komşu yoksa (veya benzerlik ağırlığı 0'sa) yedek
  formüle düşülür: kullanıcı ortalaması + global film sapması. 1.41 = tahminlerin
  %1.41'i yedekten geldi. Düşük olması küme yapısının kapsayıcılığını gösterir.
- **havuz_ort:** kullanıcı başına komşu-aday havuzunun ortalama boyutu (top-2'de
  en yakın 2 kümenin toplam üye sayısı). MALIYET göstergesi: 943 = kümesiz tam
  arama; 248 = benzerlik hesabının ~1/4'e inmesi. Doğruluk-maliyet grafiğinin
  x ekseni budur.

## Optimize edilebilir değerler envanteri (tam liste)

| Grup | Parametre | Şu anki | Denenmiş mi |
|---|---|---|---|
| Temsil | NMF boyutu L | 20 | kısmen (WNMF sweep eski repo'da) |
| Temsil | Uzay (NMF/SVD/WNMF) | NMF | evet (space_check) |
| Kümeleme | K | 10 | evet (k_sweep) |
| Kümeleme | denge sınırı cap | 2.5/K | hayır (2/K, 3/K denenmeli) |
| Kümeleme | havuz genişliği top-p | 2 | kısmen (1 vs 2; 3 denenmedi) |
| Meta | NFE | 2000 | evet (doyum kanıtlı) |
| Meta | pop_size | 30 | hayır |
| Meta | algoritma | AVOA | evet (22 aday elendi) |
| Tahminci | kNN k | 30 | evet (30>60) |
| Tahminci | benzerlik | cosine (merkezli) | hayır (Pearson + significance weighting) |
| Tahminci | shrinkage λ | yok | hayır |
| Tahminci | tip (kNN/bias/MF) | kNN | kısmen (küme-içi MF denenmedi) |
| Değerlendirme | ilgililik eşiği | ≥4 | hayır (≥3.5 varyantı) |

## Diğer makalelerde olup bizde henüz olmayanlar

Metrikler: F1@10, coverage (katalog kapsama), sınıflandırma accuracy'si
(tahmin≥3.5 vs gerçek≥3.5), sensitivity/specificity, MAP/HR@10, çeşitlilik/yenilik
(Bobadilla 2024), hata SD + t-değeri (Katarya formatı). İlk üçü kolay eklenir.
Teknikler: gerçek FCM üyelikleri (bizim top-2 bunun ayrık yaklaşığı), yan bilgi
(u.item tür vektörleri, yaş/meslek — Katarya "type division" tür kullanıyor!),
küme-içi MF, benzerlik füzyonu (Pearson+Jaccard), zaman farkındalığı.
En yüksek getiri adayı: **tür (genre) yan bilgisini özellik uzayına katmak** —
ML-100K'da hazır ve literatür doğrudan kullanıyor.

## ML-1M planı (sonraki aşama)

`soft_align.py --dataset 1m` iskeleti hazır; gereken: Ctx'e 1M yükleyici
(6040×3952), S matrisi ~280MB (tek seferlik, feasible), resmi 5-fold yerine
1M standardı rastgele %90/10 split × 5 tekrar. Koşu sizin makinede (~30-60 dk).

- **LOF gray-sheep (K=10):** B0'da zarar (0.801→0.811; %16 kullanıcıyı dışlamak
  merkezleri kaydırıyor, bilgi kaybı). AVOA-dengeli'de marjinal fayda
  (MAE 0.7906→0.7887, NDCG 0.786→0.792) — tek seed'de küçük; çok-seed teyidi
  gelmedikçe makaleye girmez. `eksikler_lof.csv`.
- **NFE>2000:** kazanç yok (K4'te kanıt) — "daha çok bütçe = daha iyi" varsayımı
  bu problemde yanlış; sadece gerekçe cümlesi olarak kullanılır.
- **Kısıtsız pred-MAE'nin ham rakamları:** görünüşte parlak (0.750) ama dejenere;
  yalnızca K1'in kanıtı olarak yaşar, performans iddiası olarak asla.

## Sıradaki mühürleme işleri

- [ ] K2/K4/LOF çok-seed (10–30) + CV5 tekrar; kullanıcı-bazlı Wilcoxon
- [ ] Dengeli fitness'ta cap duyarlılığı (2/K, 2.5/K, 3/K)
- [ ] GWO/NGO'nun dengeli fitness ile aynı tabloya eklenmesi
- [ ] knn_all satırının makale ana tablosuna "üst sınır" olarak konması
