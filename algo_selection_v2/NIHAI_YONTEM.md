
# NİHAİ YÖNTEM — Her Kararın Gerekçesi ve Kanıtı

**Yöntem adı:** CAMC-CF — *Capacity-Aware Metaheuristic Clustering for
Collaborative Filtering* (Kapasite-Farkındalıklı Metasezgisel Kümelemeli CF)

Bu belge, sistemdeki **her tasarım kararını** şu formatta verir:
*ne yaptık → alternatifi neydi → neden bu → kanıt dosyası.*

---

## PİPELİNE ÖZETİ

```
Puanlar R
  └─ [1] Bölme: test %10 | öğrenme %81 | geliştirme %9
       └─ [2] Özellik uzayı: NMF-20 (öğrenme setinden)
            └─ [3] Başlangıç: KMeans++ merkezleri C₀
                 └─ [4] Meta-sezgisel arama (AVOA, NFE=600–1200)
                      │    fitness = geliştirme MAE'si (kapasiteli atama ile)
                      └─ [5] Kapasiteli onarım: |küme| ≤ ⌈N/K⌉
                           └─ [6] Soft top-2 havuz (≈2N/K kişi)
                                └─ [7] Tahmin: 0.5·küme-kNN(k=20) + 0.5·küme-MF(f=10)
                                     └─ [8] Ölçüm: MAE/RMSE/P/R/F1/NDCG + havuz + kapsama
```

---

## KARAR 1 — Veri bölmesi: üç parçalı

| | |
|---|---|
| **Ne** | Test %10 (hiç dokunulmaz) · öğrenme %81 · geliştirme %9 |
| **Alternatif** | Tek bölme (%90/%10); fitness'ı test'te ölçmek |
| **Neden** | Meta-sezgisel binlerce aday deneyip en iyisini seçiyor. Bu seçim test'te yapılsaydı sonuç test'e ayarlanmış olurdu. Geliştirme seti bu sızıntıyı keser. |
| **Kanıt** | Literatürdeki şüpheli düşük MAE'ler (0.50–0.68) çoğunlukla bölme protokolü raporlanmadan veriliyor (`INCELEME_6_MAKALE.md`) |

## KARAR 2 — Özellik uzayı: NMF-20 (tür bilgisi YOK)

| | |
|---|---|
| **Ne** | Ham puan matrisi → NMF, 20 boyut, yalnız öğrenme setinden |
| **Alternatif** | Ham matris · SVD-20 · NMF-10/30 · +tür (genre) bilgisi |
| **Neden** | NMF-20 hem en iyi MAE hem en iyi NDCG; SVD gerisinde (negatif-olmayanlık zevk profiline uygun). Tür bilgisi B0'ı iyileştiriyor ama AVOA'yı değiştirmiyor → meta zaten o bilgiyi kendi buluyor. Saf CF kalmak literatür kıyasını da adil yapıyor. |
| **Kanıt** | `latent_sweep.csv` (NMF-20 en iyi), `tabloB_plus_K6_genre.csv` (tür ablasyonu) |

## KARAR 3 — Arama uzayı: MERKEZLER (atama değil)

| | |
|---|---|
| **Ne** | Meta-sezgisel K×D boyutlu merkez vektörü arıyor |
| **Alternatif** | Doğrudan atama araması (N boyutlu etiket vektörü) |
| **Neden** | (a) Uzay 120–800 boyut vs 943; (b) merkez parametrizasyonu **yapısal düzenlileştirici** — geometrik tutarlılığı zorunlu kılıyor; (c) yeni kullanıcıya uygulanabilir kural verir. |
| **Kanıt** | `dogrudan_atama.csv`: doğrudan atamada HHO 943 kullanıcının **942'sini tek kümeye** koydu; GWO/HGS/NGO hiç hareket edemedi. **NFE'yi 5 katına çıkarmak değiştirmedi** (`PARAMETRIZASYON_BULGUSU.md`) |

## KARAR 4 — Sıcak başlangıç (warm start)

| | |
|---|---|
| **Ne** | Popülasyon KMeans++ çözümünden türetiliyor (C₀ + gürültü) |
| **Alternatif** | Rastgele başlangıç |
| **Neden** | Sıfırdan arayan meta-sezgisel K-means++'ı geçemiyordu; sıcak başlangıçla 5/5 algoritma geçti. Yöntem böylece *memetik* oluyor: deterministik sezgisel + metasezgisel rafinasyon. |
| **Kanıt** | `MAKALE_KATKILARI.md` K11/K15 |

## KARAR 5 — Uygunluk fonksiyonu: geliştirme MAE'si (WCSS değil)

| | |
|---|---|
| **Ne** | fitness = kapasiteli atama + soft havuz + bias tahminci ile geliştirme MAE'si |
| **Alternatif** | WCSS (literatürün standardı) · NDCG · hibrit |
| **Neden** | WCSS'te K-means++ yenilmez (kapalı-form çözüyor) ve tahmine yansımıyor. Tahmine hizalı hedefte meta-sezgisel anlamlı fark yaratıyor. NDCG/hibrit denendi: hibrit marjinal iyi, ek karmaşıklık değmez. |
| **Kanıt** | `final_run.csv` (WCSS'te 5/5 meta B0'ın altında), `ndcg_fitness.csv` (fitness ablasyonu) |

## KARAR 6 — Kapasite kısıtı (repair)

| | |
|---|---|
| **Ne** | Hiçbir küme ⌈N/K⌉ kişiyi aşamaz; dolan kümeye gelen sıradaki en yakına gider |
| **Alternatif** | Serbest Voronoi · ceza terimi · esnek kapasite (1.5×, 3×) |
| **Neden** | Üç sebep: (1) **adalet** — kısıtsızken yöntemler farklı maliyet sınıflarında yarışıyor (B0 havuzu %60, AVOA %34); (2) **iyi tanımlılık** — kısıtsız MAE hedefinin optimumu "kümelemeyi iptal et"; (3) **performans** — dev küme küme-MF'i bozuyor, repair'li B0 repair'sizden iyi (0.6880 vs 0.6897, ML-1M). |
| **Kanıt** | `K_SECIMI_SAVUNMASI.md`, `ml1m_repair_ablasyon.csv`, `dogrudan_atama.csv` |
| **Literatür** | Bradley 2000 (constrained k-means), Malinen & Fränti 2014 (balanced k-means) |

## KARAR 7 — Soft top-2 havuz

| | |
|---|---|
| **Ne** | Komşu adayları = kullanıcının kümesi + en yakın ikinci küme (≈2N/K) |
| **Alternatif** | Hard (top-1) · top-3+ |
| **Neden** | Kapasite kısıtı %26–40 kullanıcıyı ikinci kümeye itiyor; top-2 bunu **%100 telafi ediyor** (yer değiştirenlerin hepsi en yakın kümesini havuzda buluyor). top-1'e göre MAE −0.018, fallback 3 kat az. top-3 havuzu şişirip kıyası bozar. |
| **Kanıt** | `eksikler_soft.csv`, K21 yer değiştirme analizi |
| **Literatür** | Kużelewska 2018 (multi-clustering) ailesinin kısıtlı hali |

## KARAR 8 — Tahminci: 0.5·kNN + 0.5·küme-MF

| | |
|---|---|
| **Ne** | Küme-içi kNN (k=20, merkezli cosine) ile küme-başına ALS-MF (f=10) eşit ağırlıklı karışım |
| **Alternatif** | Küme-ortalaması (literatür) · bias · tek bileşen · üçlü karışım · ayarlı β |
| **Neden** | Küme-ortalaması %9–11 kötü **ve algoritma farkını eziyor**. Karışım tek bileşenlerin ikisini de geçiyor. Üçlü karışımın kazancı binde 0.5 → sadelik tercih edildi. **Ayarsız 0.5/0.5, ayarlıya eşit** → hiperparametre sömürüsü yok. |
| **Kanıt** | `ml1m_karisim.csv`, `predictor_upgrade.csv` |
| **Literatür** | Dacrema 2019: iyi ayarlı kNN birçok modern yöntemi geçiyor; Katarya 2016 k=15–20 optimum |

## KARAR 9 — Algoritma seçimi: AVOA

| | |
|---|---|
| **Ne** | 22 aday → adil eleme → AVOA |
| **Alternatif** | GWO/HHO/HGS/NGO/PSO/SMA vb. |
| **Neden** | ML-1M ana tabloda **15/15 hücrede birinci**, diğer 4 metayı da anlamlı geçiyor (Holm p ≤ 0.0003), kullanıcı düzeyinde p=2.6e-193. Ayrıca komşu-recall'de B0'ın 2 katı. Özgünlük: film önerisinde AVOA'lı kümeleme literatürde yok. |
| **Kanıt** | `ml1m_ana.csv`, `KOMSU_RECALL_BULGUSU.md`, `pure_meta_summary.csv` |

## KARAR 10 — Bütçe (NFE) ve K

| | |
|---|---|
| **Ne** | NFE tüm algoritmalarda eşit (600–1200); K seçilmiyor, tüm bütçe eğrisi raporlanıyor |
| **Alternatif** | Epoch eşitlemek · tek K seçmek |
| **Neden** | Aynı epoch'ta algoritmalar 5 kata kadar farklı NFE harcıyor → epoch eşitlemek adil değil. K bir hiperparametre değil **dağıtım bütçesi** (havuz≈2N/K); tek K seçmek kiraz toplama olur. |
| **Kanıt** | `K_SECIMI_SAVUNMASI.md`, `ml1m_ksweep.csv` |

## KARAR 11 — Hiperparametre ayarı YAPILMIYOR

| | |
|---|---|
| **Ne** | mealpy varsayılan parametreleri, ayarsız |
| **Alternatif** | AVOA p1/p2/p3/alpha/gama ızgarası |
| **Neden** | 24 konfigürasyon denendi: geliştirme–test korelasyonu **−0.50 (negatif)** → ayar test'i kötüleştiriyor. Ayrıca AVOA'nın "az parametre" avantajının kanıtı. |
| **Kanıt** | `avoa_tune.csv`, K19 |

---

## MEVCUT SONUÇLAR (mühürlü)

| Veri | Nokta | En iyi | MAE | NDCG@10 | Havuz |
|---|---|---|---|---|---|
| ML-100K | K=6 | AVOA | 0.7373 | 0.8393 | 315 (%33) |
| ML-100K | K=40 | AVOA | 0.7709 | 0.8082 | 47 (%5) |
| ML-1M | K=6 | HHO/AVOA | 0.686/0.688 | 0.888 | 2013 (%33) |
| **ML-1M** | **K=40** | **AVOA** | **0.7068** | **0.8783** | **302 (%5)** |

İstatistik: ML-100K Friedman p=1.6e-35 · ML-1M p=1.2e-12 · AVOA her ikisinde
tüm rakipleri anlamlı geçiyor.

---

# EKSİKLER VE İYİLEŞTİRME ALANLARI

## A) Tamamlanması gereken koşular (öncelik sırasıyla)

| # | İş | Komut | Süre |
|---|---|---|---|
| 1 | **ML-1M K=6 mühürleme** (mutlak performans tablosu tek fold) | `ml1m_run.py --exp ana --k 6 --folds 1 2 3 --seeds 42..46` | ~3 sa |
| 2 | **kNN k taraması** (k=20 seçimini kendi verimizde doğrula) | `ml1m_knnk_sweep.py --k 40` ve `--k 6` | ~40 dk |
| 3 | **E1 kapsama metrikleri** (coverage/gini/novelty) | `e1_coverage.py --k 40` | ~20 dk |
| 4 | **Varyant denemeleri** (ölçekli bütçe / memetik / hibrit) | `ml1m_varyant.py --klist 20 40 60` | ~1 sa |
| 5 | ML-1M tahminci ablasyonu çok-seed'e taşıma | mevcut script, seed döngüsü | ~1 sa |
| 6 | ML-1M cold-start katmanlı analiz | ML-100K scripti uyarlanacak | ~30 dk |

## B) Yöntemsel iyileştirme fikirleri (denenmemiş)

| Fikir | Gerekçe | Risk |
|---|---|---|
| **Fitness'a komşu-recall terimi** | Şu an merkezler MAE üzerinden dolaylı aranıyor; recall doğrudan kümelemenin işi (K: AVOA %60.6 vs B0 %28) | Ek hiperparametre (ağırlık) |
| **Uyarlamalı kapasite** | Yoğun bölgelerde daha büyük küme izni (cap'i yoğunluğa göre) | Kısıt saflığı bozulur |
| **Atipik kullanıcı yönlendirmesi** | LOF aykırıları için geniş havuz; alt-grupta −%2 kazanç ölçüldü | Genel etkisi binde 0.1 |
| **RMSE-hedefli β** | SVD referansına karşı RMSE'de gerideyiz (0.906 vs 0.876) | Tek satır, düşük risk |
| **Küme-MF'e komşuluk düzenlileştirmesi** | Küçük kümelerde MF zayıf; graph-reg NMF fikri | Karmaşıklık artışı |

## C) Yazım/sunum eksikleri

- [ ] Akış diyagramının görsel hali (Mermaid → PNG/PDF)
- [ ] Yakınsama eğrileri grafiği (fitness vs NFE, 5 algoritma)
- [ ] Doğruluk–maliyet cephesi grafiği (ML-1M için, ML-100K'da var)
- [ ] Komşu-recall grafiği (ML-1M'de ölçülmedi — ML-100K'da var)
- [ ] Literatür karşılaştırma tablosunun son hali (K/k ayrımı notuyla)
- [ ] Sınırlılıklar bölümü: bol bütçede fark yok, RMSE'de SVD gerisinde,
      tek veri ailesi (MovieLens), zaman-farkındalıklı bölme yok

## D) Bilinçli kapsam dışı (gerekçeli)

Derin öğrenme baseline'ları · ML-10M/20M · zaman bazlı bölme · yeni
metasezgiseller · çok amaçlı optimizasyon → hepsi "gelecek çalışma" notu olarak
yazılacak, deney yapılmayacak.
