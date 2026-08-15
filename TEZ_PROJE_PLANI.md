# Lisans Tezi Proje Planı — Meta-Sezgisel Destekli Kümeleme Tabanlı CF Öneri Sistemi

**Hedef:** 2 hafta içinde hocaya sunulabilir, uçtan uca savunulabilir bir yapı.
**Tez cümlesi (taslak):** "ML-100K resmi 5-fold protokolünde, WNMF kullanıcı özellik uzayında AVOA ile optimize edilen küme merkezleri, düz K-means'e ve literatürdeki meta-sezgisel alternatiflere (PSO, GWO, HHO) kıyasla küme tabanlı CF tahmin doğruluğunu iyileştirmektedir."

---

## 1. Uçtan Uca Pipeline (6 Aşama)

```
[1] Veri Yükleme → [2] Ön İşleme → [3] Özellik Çıkarımı (WNMF)
→ [4] Meta-Sezgisel Kümeleme (AVOA) → [5] Küme Tabanlı Tahmin → [6] Değerlendirme + İstatistik
```

Her aşama tek bir config dosyasından (JSON/YAML) kontrol edilmeli; şu an aynı parametreler onlarca script'e dağılmış durumda (bkz. Bölüm 8).

---

## 2. Veri Yükleme — Kontrol Listesi

Dataset: **ML-100K resmi 5-fold (u1–u5)** ana protokol; ML-1M yalnızca genelleme kanıtı için tek konfigürasyonla.

Dikkat edilecekler:

- **Resmi split kullan** (u1.base/u1.test) — rastgele split değil. Literatürle (Thakrar, Katarya) karşılaştırılabilirlik bunun üzerine kurulu.
- **Rating ölçeği 1–5**; tahminler `clip(1,5)` yapılmalı (MAE'yi yapay şişirmemek için).
- **ID yeniden indeksleme**: user/item ID'leri 1-tabanlı; 0-tabanlı diziye çevirirken off-by-one kontrolü.
- **Duplicate kontrolü**: aynı (user,item) çifti birden fazla satırda olmamalı.
- **Sparsity raporu**: ML-100K ≈ %93.7 seyreklik — tezde tablo olarak verilmeli.
- **Test'te olup train'de olmayan kullanıcı/film**: fallback stratejisi (global ortalama → kullanıcı ortalaması → küme ortalaması) açıkça tanımlanmalı ve **fallback oranı raporlanmalı**. Thakrar koşularınızda `global_fallback_pct ≈ %24` — bu sayı sunumda mutlaka açıklanmalı, yoksa hoca sorar.
- **Veri sızıntısı (leakage)**: WNMF/normalizasyon/kümeleme **yalnızca train fold** üzerinde fit edilmeli. `--train-only` bayrağınız doğru yaklaşım; tüm final koşularda zorunlu olsun.
- **Seed sabitleme**: numpy + mealpy seed'leri config'de; her koşu yeniden üretilebilir olmalı.

---

## 3. Ön İşleme Kararları (Final Konfigürasyon)

Deneyleriniz zaten şunu gösterdi; final'de tek konfigürasyona sabitleyin:

| Adım | Karar | Gerekçe |
|---|---|---|
| Normalizasyon | user z-score veya none (deney sonuçlarınızdan iyi olan) | zscore_lof karşılaştırmanız mevcut |
| Gray sheep | LOF (opsiyonel ablasyon olarak) | tez katkısı olarak anlatılabilir |
| Özellik çıkarımı | WNMF, L=20–50 (k'ya göre en iyi) | `ha_wnmf_dim_sweep` sonuçlarınız var |
| K seçimi | K=10 (ana), K sweep grafiği ek | `fuzzy_official_k3_24` sonuçlarınız var |
| kmeans-refine (kmref) | **Kapalı** (final meta karşılaştırmasında) | kmref atamaları homojenleştirip meta farkını siliyor (kendi bulgunuz, META_MAE_FITNESS_ROADMAP) |

**Kritik tasarım kararı — fitness fonksiyonu:** Kendi teşhisiniz doğru: WCSS fitness'ı ile holdout MAE hedefi uyumsuz (algoritmalar WCSS'te yakınsıyor, tahminler r≈0.8–0.9 korelasyonlu). İki seçenek:

- **A (önerilen, dürüst anlatım):** Fitness = WCSS kalır; tez "AVOA, K-means init problemi olarak WCSS'i minimize eder" der; MAE iyileşmesinin küçük (%0.05–0.3) olması literatürle tutarlı biçimde raporlanır. Thakrar protokolü tam bunu yapıyor ve init=hho/mkpp farkını gösteriyor (MAE 0.825→0.814). Az risk, 2 haftaya sığar.
- **B (iddialı):** Fitness = `knn_mae` (roadmap Faz 1). Fark büyür ama koşu süresi çok uzar; 2 haftada tüm algoritmalar için CV5 yetişmeyebilir. Ancak **tek pilot koşu** (fold 1, AVOA vs B0) "gelecek çalışma + ön sonuç" olarak sunuma eklenebilir.

Önerim: **A ana omurga + B'den tek pilot sonuç.**

---

## 4. Algoritma Seçim Protokolü (Evaluation Dışı Kriterler)

AVOA seçimini yalnızca MAE ile değil, şu 6 kriterle savunun (sunumda tablo yapın):

1. **Ortalama sıralama (rank)**: `meta_algo_rank_summary.csv` — B_AVOA avg_rank_MAE=3.625 ile birinci, NDCG/Precision'da da ilk 3'te. Tek metrikte değil, tüm metriklerde tutarlı.
2. **Yakınsama davranışı**: iterasyon-fitness eğrileri (`compare_convergence`, `fold2_..._convergence.csv` mevcut). AVOA'nın erken yakınsama + geç iterasyonda iyileşme dengesi gösterilmeli.
3. **Kararlılık (stability)**: aynı konfigürasyonda çoklu seed → fitness std. Düşük std = güvenilir. (Eksikse: 5 seed × final config, ~1 gün koşu.)
4. **Çalışma süresi**: fitness değerlendirme başına süre; AVOA vs HHO (218s) vs HGS (121s) tablonuz var.
5. **Keşif/sömürü dengesi + parametre sayısı**: AVOA'nın az kontrol parametresi (P1,P2,P3) ayar yükünü azaltır — pratik avantaj olarak anlatın.
6. **No Free Lunch gerekçesi**: "en iyi evrensel algoritma yoktur, bu problem sınıfı (sürekli centroid optimizasyonu, 93% seyrek uzay) için deneysel seçim yaptık" — hocaya karşı en sağlam çerçeve.

Rakip set (final): **B0_KMEANS (baseline), AVOA, HHO, PSO, GWO** (+ isterseniz HA_AVOAHGS hibritiniz "önerilen yöntem varyantı" olarak). PSO ve GWO literatür karşılığı olduğu için zorunlu (Katarya makaleleri).

---

## 5. İstatistiksel Testler — Mevcut Durum Kritiği ⚠

Mevcut: Wilcoxon signed-rank (ikili) + Friedman (çoklu), α=0.05. **Test seçimi doğru, uygulama detayları düzeltilmeli:**

1. **CV5 üzerinde Wilcoxon geçersiz**: n=5 fold ile two-sided Wilcoxon'un ulaşabileceği en küçük p = 0.0625 → **α=0.05'te anlamlılık matematiksel olarak imkânsız**. 5 fold ortalamasıyla "anlamlı fark" iddia etmeyin. Çözümler:
   - **Kullanıcı bazında eşleştirilmiş test**: her kullanıcının MAE'si üzerinden Wilcoxon (n=943) — güçlü ve literatürde kabul görür.
   - **Paired bootstrap CI**: `paired_bootstrap_ci.py` zaten var — tahmin çiftleri üzerinden %95 CI raporlayın; CI sıfırı içermiyorsa fark anlamlı.
   - Alternatif: 5 fold × 6 seed = 30 örneklemle fold-seed düzeyinde test.
2. **Benchmark fonksiyonları (comparison_algorithms) için**: standart = **30 bağımsız koşu** / fonksiyon. Koşu sayınızı kontrol edin; 30'dan azsa tamamlayın. Friedman sonrası **post-hoc Holm düzeltmesi** ekleyin (çoklu karşılaştırma hatası için) — `statistical_analysis.py`'de şu an yok.
3. **Etki büyüklüğü**: p-değeri yanına MAE farkının % iyileşmesi + CI koyun. %0.05'lik iyileşme "istatistiksel olarak anlamlı ama pratik olarak küçük" diye dürüstçe çerçevelenmeli — bu tezi zayıflatmaz, güçlendirir.
4. **Metodoloji atıfları** (teze mutlaka): Demšar (2006) "Statistical Comparisons of Classifiers over Multiple Data Sets" (Friedman+post-hoc standardı); Derrac et al. (2011) (metasezgisel karşılaştırmalarda parametrik olmayan testler rehberi).

---

## 6. Final Deney Planı

| # | Deney | Amaç | Durum |
|---|---|---|---|
| E1 | B0 vs AVOA vs HHO vs PSO vs GWO, K=10, CV5, kmref yok | Ana sonuç tablosu (MAE/RMSE/P@10/R@10/NDCG@10) | Çoğu koşu mevcut, eksikler tamamlanacak |
| E2 | K sweep (K=3..24) AVOA | K seçimi gerekçesi grafiği | Mevcut (`fuzzy_official_k3_24`) |
| E3 | Yakınsama eğrileri (5 algo, fold 2) | Algoritma seçimi savunması | Mevcut, yeniden çizilecek |
| E4 | Ablasyon: init modu (random/mkpp/avoa) Thakrar pipeline | Meta katkısını izole eder | Mevcut (`thakrar_bigcomp_protocol.csv`) |
| E5 | Benchmark fonksiyonları + Wilcoxon/Friedman+Holm | AVOA'nın genel gücü | Mevcut, Holm eklenecek |
| E6 | İstatistik: kullanıcı-bazlı Wilcoxon + paired bootstrap | Anlamlılık | **Yeni — öncelik** |
| E7 | (Ops.) knn_mae fitness pilotu, fold 1 | Gelecek çalışma teaser'ı | Roadmap Faz 1 |

---

## 7. Literatür Örtüşmesi — Hangi Makaleler?

**Ana referans (protokol birebir):** Thakrar pipeline'ının kaynak makalesi (repo'da "Thakrar Algoritma 2 / BigComp" olarak geçen çalışma — MF + K-means + meta-init). Sizin MAE≈0.814–0.825 aralığınız bu protokolle doğrudan karşılaştırılır. Tezde "aynı protokol, aynı veri, aynı metrik" cümlesi kurabildiğiniz tek makale bu → **birincil örtüşme makalesi**.

**Yöntem ailesi (kümeleme + meta-sezgisel CF, ML-100K MAE karşılaştırılabilir):**
1. Katarya & Verma (2016), *KM-PSO-FCM*, Multimedia Tools and Applications — PSO'lu küme merkezi optimizasyonu; sizin PSO baseline'ınızın literatür karşılığı.
2. Katarya (2018), *GWO+FCM*, Neural Computing and Applications — GWO baseline karşılığı.
3. Ganesh et al. (2024), *GOA-K-means film önerici*, Multimedia Tools and Applications — en güncel "meta-sezgisel + k-means + film" makalesi; giriş/motivasyon için ideal.
4. Abdollahzadeh et al. (2021), *AVOA orijinal makalesi*, Computers & Industrial Engineering — algoritma tanımı için zorunlu atıf.

Bunların 1–3'ü literature.csv'nizde zaten var. MAE değerleriniz (0.77–0.82 bandı) bu aile ile aynı mertebede — örtüşme iddiası savunulabilir. Dikkat: farklı ön işleme/split kullanan makalelerle **sayı sayıya kıyas yapmayın**; "aynı mertebe + aynı yönde iyileşme" deyin.

**Ek araştırma gereken konular (1–2 saatlik tarama yeter):**
- "AVOA + recommender system" araması: AVOA'yı CF kümelemede kullanan yayın var mı? Yoksa bu **tezin özgünlük cümlesi** olur; varsa karşılaştırma referansınız olur. (Öncelik: yüksek)
- Thakrar makalesinin tam künyesini (yazar, yıl, konferans) netleştirip kaynakçaya doğru girmek.
- Demšar 2006 + Derrac 2011 (Bölüm 5'teki istatistik atıfları).

---

## 8. Kod Düzenleme Planı — `tez/` Paketi

Mevcut repo deney arkeolojisi halinde (100+ script). Hocaya sunulacak temiz paket, **`thakrar/` modülünün yapısı örnek alınarak** ayrı klasörde toplanmalı:

```
tez/
├── config.py          # tüm parametreler tek yerde (dataset, K, L, seed, algo)
├── data.py            # ML-100K/1M yükleme + kontroller (Bölüm 2 checklist'i kod olarak)
├── preprocess.py      # normalizasyon, LOF, pruning
├── features.py        # WNMF özellik çıkarımı (train-only fit)
├── optimizers.py      # AVOA, HHO, PSO, GWO sarmalayıcıları (mealpy üstü, tek arayüz)
├── clustering.py      # centroid fitness (WCSS / knn_mae), atama, B0 K-means
├── predict.py         # cluster-avg / cluster-kNN tahmin + fallback zinciri
├── evaluate.py        # MAE, RMSE, P@10, R@10, NDCG@10, fallback %
├── stats.py           # kullanıcı-bazlı Wilcoxon, paired bootstrap, Friedman+Holm
├── run_experiment.py  # E1–E7 tek komutla: python -m tez.run_experiment --exp E1
└── results/           # sadece final CSV + figürler
```

Kaynak eşleşmesi: `data.py` ← `thakrar/data.py`; `optimizers.py` ← `optimizers/` + `mealpy/generate_assignments.py`; `predict.py` ← `wnmf/cluster_predictor.py`; `stats.py` ← `comparison_algorithms/statistical_analysis.py` + `mealpy/paired_bootstrap_ci.py`. Eski scriptler silinmez, dokunulmaz — sadece bu paket sunulur.

---

## 9. İki Haftalık Takvim

| Gün | İş |
|---|---|
| 1–2 | `tez/` paketi iskeleti + data/preprocess/features taşıma; config sabitleme (Bölüm 3 kararları) |
| 3–4 | optimizers + clustering + predict taşıma; E1 eksik koşuları başlat (arka planda) |
| 5 | `stats.py`: kullanıcı-bazlı Wilcoxon + bootstrap + Friedman/Holm (Bölüm 5 düzeltmeleri) |
| 6–7 | E1 tamamla, E5'e Holm ekle, E6 istatistikleri üret |
| 8 | Literatür: AVOA+RS taraması, Thakrar künyesi, Demšar/Derrac okuma notu |
| 9 | Figürler: yakınsama eğrileri, K sweep, ana sonuç tablosu, pipeline diyagramı |
| 10 | (Ops.) E7 knn_mae pilotu fold 1 |
| 11–12 | Sunum + kısa rapor: motivasyon → pipeline → algoritma seçimi (6 kriter) → sonuçlar → istatistik → literatür örtüşmesi → gelecek çalışma |
| 13 | Doğrulama: sayıların CSV↔slayt tutarlılığı, seed'le bir koşunun yeniden üretimi |
| 14 | Tampon + prova |

---

## 10. Riskler ve Tavsiyeler

- **Küçük iyileşme problemi**: AVOA-KMeans CV5 iyileşmesi %0.05–0.29. Bunu gizlemeyin; "WCSS fitness'ının sınırı" olarak analiz edin (kendi roadmap bulgunuz) — bu analiz tezin en güçlü tartışma bölümü olur.
- **kmref tuzağı**: final meta karşılaştırmalarında kmeans-refine kapalı olduğundan emin olun; açıksa algoritma farkı görünmez.
- **Fallback %24**: sunumda mutlaka bir slayt; "test kullanıcısı hiçbir kümeye güvenle atanamazsa..." açıklaması hazır olsun.
- **Koşu süresi**: E1 eksikleri hemen (gün 3) başlatılmalı; CV5 × 5 algo uzun sürer.
- **Kapsam dondurma**: yeni algoritma/yeni K/yeni dataset eklemeyi bugün itibarıyla durdurun. 2 haftalık hedef sunulabilirlik, keşif değil.

## 11. Hocaya Teslim Edilecekler

1. Sunum (≈15 slayt) — Bölüm 9'daki akış
2. `tez/` temiz kod paketi + README (tek komutla E1 yeniden üretimi)
3. Ana sonuç tablosu + 4 figür (pipeline, yakınsama, K sweep, iyileşme % + CI)
4. Kısa metodoloji raporu (bu planın 2–7 bölümleri genişletilmiş hali — tez bölüm taslağı olarak da kullanılır)
