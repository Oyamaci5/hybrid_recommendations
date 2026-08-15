# Denetim Raporu + Adım Adım Kontrol Listesi

Kod incelemesi sonucu: `mealpy-algorithms-comparision.py`, `centroid_optimizer.py`,
`ranked_scores.csv`, phase1–3 CSV'leri, 20-algoritma log'u.

---

## A) DOĞRU YAPILANLAR ✓

| # | Ne | Neden doğru |
|---|---|---|
| D1 | ML-100K resmi 5-fold (u1–u5) protokolü | Literatürle birebir kıyas için tek geçerli split |
| D2 | `thakrar/` temiz pipeline (meta-init → K-means → MF) | Literatürün kullandığı yapı tam olarak bu; init=hho ile MAE 0.825→0.814 farkı ölçülebilmiş |
| D3 | Test seçimi: Wilcoxon + Friedman | Metasezgisel literatürünün standardı (Derrac 2011) |
| D4 | `paired_bootstrap_ci.py` | CV5'in n=5 sorununu çözen doğru araç |
| D5 | `--train-only` bayrağı (fitness'ı train'de hesaplama) | Veri sızıntısını engeller |
| D6 | Boş küme cezası fitness'ta | Dejenere çözümleri engeller |
| D7 | Çoklu K sweep + yakınsama eğrisi toplama | Seçim savunması için gerekli malzeme |
| D8 | SQLite ile deney takibi | Yeniden üretilebilirlik altyapısı |

## B) HATALI / GÜVENİLMEZ ✗  (eski karşılaştırma sonuçlarını geçersiz kılanlar)

| # | Hata | Kanıt | Etki |
|---|---|---|---|
| H1 | "WCSS" aslında **Pearson mesafesi toplamı** (varsayılan `metric='pearson'`) | `compute_wcss_fast` | Literatürdeki WCSS (kare Oklid SSE) ile karşılaştırılamaz; k-means'in optimize ettiği şey de bu değil |
| H2 | Üç metrik **üç farklı uzayda**: WCSS=pearson, Silhouette=pearson-precomputed 300 örneklem, DB=ham matris Oklid | `_compute_metrics` | Metrikler birbiriyle ve kendi WCSS'iyle tutarsız; sıralamalar anlamsız |
| H3 | Silhouette örneklemi **seed'siz** `np.random.choice` | `_compute_metrics` | Aynı koşu iki kez farklı skor verir — tekrar üretilemez |
| H4 | **Tek koşu / algoritma** (seed tekrarı yok) | 20-algo log | WCSS farkı %1.5 bandında (674–685); bu fark gürültüden ayırt edilemez. 18/20 algoritmanın 684.7'de eşitlenmesi arama uzayının çökmesi demek |
| H5 | Gray sheep oranı **her algoritmada tam %20** | log + `ranked_scores.csv` | Eşik quantile ile kurulmuş → %20 bir bulgu değil, tanım gereği; skor bileşeni olarak kullanılamaz (`score_gray=0` hepsi) |
| H6 | Composite skor **süreyi kaliteyle karıştırıyor** | `ranked_scores.csv` (score_time ağırlıklı) | EPC.DevEPC'nin 1. olması hız etkisi; kalite sıralaması değil |
| H7 | Fazlar arası WCSS ölçekleri uyumsuz (ph3≈26, log≈680, ranked≈130–150) | phase3 vs log vs ranked | Farklı matris/örneklem → dosyalar arası hiçbir kıyas geçerli değil |
| H8 | Silhouette tüm algoritmalarda ≈ **−0.03** | log | O uzayda küme yapısı YOK; hangi meta olursa olsun anlamlı kümeleme çıkmaz. Uzay/K seçimi metadan önce çözülmeli |

## C) EKSİK / YAPILMASI GEREKEN □

| # | Eksik | Çözüm |
|---|---|---|
| E1 | Seed tekrarı (≥5 seçim, 30 final) + mean±std + rank | `recompute_scores.py --runs 5` (yazıldı) |
| E2 | Aynı uzayda SSE-WCSS + Sil + DB | `recompute_scores.py` (yazıldı) |
| E3 | KMeans++ (n_init=10) baseline'ı aynı tabloda | Yazıldı — **seçim kapısı**: meta bunu geçemiyorsa katkısı sıfır |
| E4 | kmref öncesi/sonrası WCSS ayrımı | Yazıldı (`wcss_meta` vs `wcss_kmref` + `kmref_gain_pct`) |
| E5 | Friedman + (anlamlıysa) Holm düzeltmeli post-hoc | Friedman yazıldı; post-hoc, anlamlılık çıkarsa eklenecek |
| E6 | Küme yapısı ön kontrolü: Hopkins istatistiği + K için silhouette>0 kontrolü | Sonraki adım — H8 çözülmeden meta seçimi yapılmamalı |
| E7 | AVOA'nın mealpy'de olup olmadığının netleştirilmesi; yoksa özel AVOA'nın aynı protokole sarılması | `algos.txt` uyarı satırı; koşuda görülecek |
| E8 | CF aşaması iyileştirme %'sinin literatür formatında raporu (baseline MAE → yöntem MAE, %) | Algoritma seçimi bitince |

---

## D) SORULARINIZIN CEVAPLARI

**"kmref açık olması gerekmiyor mu?" — EVET, açık olmalı.**
Literatürdeki akış tam sizin dediğiniz gibi: meta-sezgisel merkezleri optimize eder →
bu merkezler K-means'e (Lloyd) başlangıç verilir → atama K-means'ten çıkar
(Katarya PSO/GWO: merkez optimizasyonu → FCM/K-means; Thakrar Alg.2: meta-init → K-means;
GOA makalesi: GOA → k-means). Önceki "kmref kapalı" kararı algoritmalar arası farkı
büyütmek içindi ama literatür protokolünden sapıyordu. **Yeni kural:**
- Ana protokol: kmref AÇIK (literatürle kıyaslanabilir).
- Meta'nın katkısı iki şekilde ölçülür: (1) kmref sonrası WCSS'in KMeans++'a karşı durumu,
  (2) Lloyd'un kaç iterasyonda yakınsadığı (iyi init = az iterasyon) → script ikisini de kaydediyor.

**"Sonuçlar doğru mu?" — Eski karşılaştırma sonuçları güvenilir DEĞİL** (H1–H8).
Özellikle: %20 gray sheep bulgu değil; WCSS'ler Pearson toplamı; tek koşu; silhouette −0.03
her yerde. Algoritma seçimine sıfırdan, temiz protokolle başlamak doğru karar.

**"İyileşme neden %0.05 gibi düşük, makalelerde %5+?"** — Üç neden birden:
1. H8: küme yapısı olmayan uzayda hiçbir init farkı tahmine yansımaz.
2. kmref kapalıyken meta atamaları homojen; açıkken de Lloyd her init'i benzer yerel
   minimuma taşıyorsa uzay kötü demektir (bunu `kmref_gain_pct` gösterecek).
3. Makalelerin çoğu zayıf baseline'a karşı raporlar (random-init K-means, tek fold).
   Sizin baseline'ınız güçlü (KMeans++ n_init=10 + CV5). Bu farkı tezde açıkça yazmak
   gerekir; ama önce 1–2'yi düzeltip gerçek iyileşme payını görmeliyiz.

**"AVOA şart mı?" — Hayır.** Seçim hunisi sonucu ne çıkarsa o. HHO'nun kıyas makalesi
olması (repo literatüründe HHO tabanlı öneri çalışması + Thakrar init=hho sonucu zaten var)
onu güçlü aday yapar. AVOA'nın avantajı "film önerisinde AVOA'lı kümeleme yok" özgünlük
cümlesi; HHO'nun avantajı doğrudan kıyaslanabilirlik. İkisi de meşru — veri karar versin.

---

## E) HER ADIM SONRASI KONTROL EDİLECEKLER (pipeline kontrol kapıları)

**Adım 1 — Veri yükleme** (script otomatik kontrol ediyor):
- [ ] shape 943×1682, 80.000 train / 20.000 test satır
- [ ] rating ∈ [1,5], duplicate yok, train∩test = ∅
- [ ] sparsity ≈ 0.937 raporlandı

**Adım 2 — Özellik uzayı:**
- [ ] Sadece train ile fit (test satırı girmedi)
- [ ] NaN/inf yok; varyansı 0 boyut yok
- [ ] Matris hash'i loglandı → tüm algoritmalar AYNI hash'i kullanıyor
- [ ] Hopkins istatistiği > 0.7 (kümelenebilirlik var mı?) — yoksa dur, uzayı değiştir
- [ ] KMeans++ ile K sweep: en iyi K'da silhouette > 0 mı? ≤0 ise meta seçimine geçme

**Adım 3 — Meta optimizasyon (her koşu):**
- [ ] Best-so-far eğrisi monoton azalıyor (script assert ediyor)
- [ ] WCSS < K=1 global SSE (script referansı basıyor)
- [ ] Boş küme sayısı = 0 (değilse ceza mekanizması gözden geçirilir)
- [ ] Aynı seed → aynı sonuç (tekrar üretilebilirlik: 1 algoritmayla test et)
- [ ] Seed'ler arası std, algoritmalar arası farktan KÜÇÜK mü? Değilse fark gürültü

**Adım 4 — kmref (Lloyd rafine):**
- [ ] wcss_kmref ≤ wcss_meta (script assert ediyor)
- [ ] Meta-init'li Lloyd iterasyon sayısı < random-init'li Lloyd (iyi init kanıtı)
- [ ] SEÇİM KAPISI: en az bir meta, KMeans++ (n_init=10) ortalamasını geçiyor mu?

**Adım 5 — Küme kalite metrikleri:**
- [ ] WCSS, Sil, DB hepsi aynı uzay + Oklid (script garanti ediyor)
- [ ] Sıralama Sil ve DB'de de tutarlı mı? (WCSS'te iyi ama Sil'de kötü → şüphe)

**Adım 6 — CF değerlendirme (seçim bittikten sonra):**
- [ ] Tahminler clip(1,5)
- [ ] Fallback zinciri tanımlı + fallback % raporlandı (<%25 hedef; %24'ünüz açıklanmalı)
- [ ] 5 fold ayrı ayrı + ortalama; tek fold ile sonuç iddia edilmedi
- [ ] Anlamlılık: kullanıcı-bazlı Wilcoxon (n=943) veya paired bootstrap; CV5 üstü Wilcoxon YASAK (n=5 → min p=0.0625)
- [ ] İyileşme % + güven aralığı birlikte raporlandı

---

## F) SIRADAKI ADIMLAR (onaylarsanız)

1. `recompute_scores.py --runs 5` seçim turu (25 algo × 5 seed, SVD-20 uzayı) → ilk eleme
2. Hopkins + K-sweep silhouette kontrolü (E6) → uzay kararı (SVD mi WNMF mi, hangi K)
3. Sağ kalan 5–6 algo × 30 seed final + Friedman/Holm → algoritma seçimi kesinleşir
4. Seçilen 2–3 algo Thakrar pipeline'a init olarak → CV5 CF metrikleri (kmref AÇIK)
5. Kullanıcı-bazlı anlamlılık testleri → literatür formatında iyileşme tablosu
