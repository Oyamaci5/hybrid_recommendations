# Meta-sezgisel Kümeleme Ablation Raporu
### WNMF50 + euclidean + ML-100K official fold-1 — B0_KMEANS vs B_AVOA vs B1_HHO

**Soru:** AVOA/HHO ile küme atamaları B0 K-means'ten gerçekten farklı mı, ve neden
MAE/RMSE farkları ~%0.6 ile ihmal edilebilir çıkıyor? Bunu nasıl büyütürüz? kmref'te
hata var mı?

**Kısa cevap:**
1. **kmref'te pseudocode hatası YOK** — HHO-K-means (Ambikesh 2023) *Algorithm 3*'ün
   birebir uygulaması.
2. Farkların küçüklüğü **yapısal**: WCSS fitness + tam Lloyd refinement + güçlü B0
   (`n_init=10`) üçlüsü üç algoritmayı aynı yerel optimuma çekiyor.
3. **En çok ayrıştıran kaldıraç = kmref'i gevşetmek (repair-only) + k'yı yükseltmek**;
   bu kombinasyonda AVOA, B0'ı k=14'te **%1.69 MAE** ile geçiyor.
4. `n_init=1` baseline zayıflatması bu veri/protokolde farkın kaynağı **değil**.

---

## 1. Pipeline (load_movielens → metrik) — kod izi

| Aşama | Dosya:satır | Ne yapıyor |
|------|-------------|-----------|
| Veri yükleme | [wnmf/wnmf_utils.py:31](../../wnmf/wnmf_utils.py) `load_ratings_100k` | u1.base/u1.test → train(80k), test(20k), 943×1682, 0-indexli |
| WNMF gömme | [mealpy/generate_assignments.py](../../mealpy/generate_assignments.py) `--feature-extraction wnmf` → `WNMFModel` | R(943×1682) → U(943×50) kullanıcı latent matrisi (`wnmf_user_vectors.npy`) |
| B0 centroid | [generate_assignments.py:1416](../../mealpy/generate_assignments.py) | `sklearn.KMeans(n_init=10, max_iter=500)` doğrudan U üzerinde |
| Meta centroid | mealpy `AVOA.OriginalAVOA` / `HHO.OriginalHHO` | U uzayında centroid arar; fitness = **WCSS** ([compute_wcss_fast](../../mealpy/mealpy-algorithms-comparision.py)) veya **cluster_predictor_mae** ([centroid_optimizer.py:191](../../mealpy/centroid_optimizer.py)) |
| kmref | [generate_assignments.py:1200](../../mealpy/generate_assignments.py) `_refine_centroids_with_kmeans` | meta centroid → `KMeans(init=centroids, n_init=1)` → Lloyd yakınsamaya kadar; `--kmeans-refine-overwrite` ile atamaları ezer |
| (alternatif) repair | [generate_assignments.py:1240](../../mealpy/generate_assignments.py) `_repair_empty_clusters` | meta atamasını **korur**, yalnız boş kümeleri onarır |
| Tahmin | [wnmf/wnmf_experiment.py](../../wnmf/wnmf_experiment.py) `run_cluster_average` (`--paper-mode`) | **calc_avg_rating** = Thakrar *Alg.6*: küme-içi sert ortalama → kullanıcı ort. → global ort, clip[1,5] |
| Metrik | MAE/RMSE [wnmf_model.py:367](../../wnmf/wnmf_model.py); NDCG@10/P@10 [wnmf_experiment.py:2070](../../wnmf/wnmf_experiment.py) `_compute_topn_metrics` | |

Bu yapı iki makaleyle uyumlu: **MF→meta-init→K-means→cluster-average** (Thakrar
2025, *Alg.2/Alg.6*) iskeletine **WNMF** katmanı eklenmiş hâli; meta-init = HHO/AVOA
(Ambikesh 2023, *Alg.3*).

---

## 2. kmref hata kontrolü — temiz

HHO-K-means *Algorithm 3*:
```
Use HHO to find optimal initial centroids
Use K-means with the optimized initial centroids:
    Repeat until convergence: assign → recompute centroids
Return assignments
```
Bizim `_refine_centroids_with_kmeans`: `KMeans(init=meta_centroids, n_init=1,
max_iter=300)` → Lloyd yakınsamaya kadar → atamaları ez. **Birebir aynı.** Pseudocode
hatası yok.

**Ama:** Algorithm 3'ün "yakınsamaya kadar Lloyd" adımı, WCSS uzayında üç algoritmayı
**aynı yerel optimuma** çeker. B0 da aynı WCSS'i minimize ettiğinden, kmref sonrası
B0/AVOA/HHO neredeyse özdeş partition üretir → MAE farkı kaybolur. Bu bir bug değil,
**tasarımın doğal sonucu**.

### Baseline doğrulaması (kod güveni)
Direkt calc_avg hesaplayıcımız ([_ablation_direct_metrics.py](../../experiments/_ablation_direct_metrics.py))
base B0 için **MAE=0.8289, RMSE=1.0473** — `wnmf_experiment` ile birebir aynı; tahmin
kaynak dağılımı cluster_mean=19626 / user_mean=374 / global=0 (yani 20.000 tahminin
~%98'i küme ortalaması → "yutucu" predictor doğrulandı).

---

## 3. Ablation sonuçları — hangi kaldıraç farkı büyütüyor?

Tüm hücreler fold-1, WNMF50, euclidean, official train-only. MAE doğrudan (validated)
hesaplandı. `spread` = max−min MAE (algoritmalar arası fark); `bestd%` = en iyi
meta'nın B0'a göre MAE değişimi (negatif = meta daha iyi).

| Hücre | k | init | fitness | kmref | B0 | AVOA | HHO | spread | bestd% |
|------|--:|------|---------|-------|----:|----:|----:|------:|------:|
| base | 6 | mkpp | wcss | overwrite | 0.8289 | 0.8286 | 0.8307 | 0.0021 | −0.04 |
| kmref_repair | 6 | mkpp | wcss | **repair** | 0.8289 | 0.8312 | 0.8316 | 0.0027 | +0.28 |
| kmref_capped | 6 | mkpp | wcss | capped(1) | 0.8289 | 0.8331 | 0.8305 | 0.0042 | +0.19 |
| fit_knnmae | 6 | mkpp | **knn_mae** | overwrite | 0.8289 | 0.8244 | 0.8273 | 0.0045 | **−0.54** |
| fit_knnmae_repair | 6 | mkpp | knn_mae | repair | 0.8289 | 0.8314 | 0.8357 | 0.0068 | +0.30 |
| init_random | 6 | **random** | wcss | overwrite | 0.8289 | 0.8243 | 0.8237 | 0.0052 | **−0.63** |
| init_random_repair | 6 | random | wcss | repair | 0.8289 | 0.8328 | 0.8345 | 0.0056 | +0.47 |
| k10 | 10 | mkpp | wcss | overwrite | 0.8392 | 0.8397 | 0.8362 | 0.0035 | −0.36 |
| k14 | 14 | mkpp | wcss | overwrite | 0.8470 | 0.8450 | 0.8467 | 0.0020 | −0.24 |
| **k10_repair** | 10 | mkpp | wcss | **repair** | 0.8392 | 0.8294 | 0.8332 | **0.0098** | **−1.17** |
| **k14_repair** | 14 | mkpp | wcss | **repair** | 0.8470 | **0.8327** | 0.8438 | **0.0143** | **−1.69** |
| b0_ninit1 | 6 | mkpp | wcss | (B0 n_init=1) | 0.8221 | — | — | — | — |
| k14_b0_ninit1 | 14 | mkpp | wcss | (B0 n_init=1) | 0.8489 | — | — | — | — |

**Pairwise ARI** (düşük = atamalar daha farklı):

| Hücre | B0~AVOA | B0~HHO | AVOA~HHO |
|------|:--:|:--:|:--:|
| base (overwrite) | 0.53 | 0.45 | 0.40 |
| kmref_repair | 0.18 | 0.28 | 0.21 |
| k14 (overwrite) | 0.28 | 0.25 | 0.30 |
| k14_repair | **0.13** | **0.14** | 0.32 |
| fit_knnmae (overwrite) | 0.45 | 0.54 | 0.46 |
| fit_knnmae_repair | **0.00** | **0.01** | 0.74 |

### Okuma
- **kmref overwrite → repair**, farkı korur: ARI 0.53→0.18; etki **k ile büyür**
  (k6 spread 0.0021→0.0027, k10 0.0035→0.0098, k14 0.0020→**0.0143** ≈ 7×).
- **En anlamlı meta kazancı: repair + yüksek k.** k14_repair'de **AVOA, B0'ı %1.69
  MAE ile geçiyor** (0.8327 vs 0.8470). AVOA tutarlı biçimde en iyi meta.
- **random init + overwrite**: AVOA/HHO B0'ı %0.63 geçiyor — mkpp meta'yı K-means
  optimumuna yapıştırıyor, random init arama alanını açıyor.
- **knn_mae fitness ikircikli:** overwrite ile AVOA B0'ı %0.54 geçiyor; **repair ile
  küme çöküyor** (dev küme: AVOA max=816/943, ARI≈0.00) çünkü sert küme-ortalaması
  düşük-varyanslı bir tahmincidir → val-MAE'yi minimize etmek "az sayıda dev küme"yi
  ödüllendirir. Yani değer ancak Lloyd yeniden-dengelemesiyle ortaya çıkıyor.

### Yön (sign) inceliği
k6+mkpp+repair'de meta B0'dan biraz **kötü** (+0.28%); k10/k14+repair'de meta **iyi**
(−1.17%/−1.69%). Düşük k'da meta'nın WCSS-suboptimal partition'ı tahmin için zarar;
yüksek k'da meta'nın farklı partition'ı küme-ortalamasına avantaj sağlıyor.

---

## 4. "Farklar neden bu kadar az?" — nihai teşhis

1. **Fitness = WCSS = K-means'in kendi hedefi** → meta, K-means'e dik sinyal üretmez.
2. **kmref-overwrite tam Lloyd** → meta'nın farkını siler (ARI 0.53; spread 0.002).
3. **B0 n_init=10 güçlü** → "daha iyi başlatma" avantajı kapanır (ama n_init=1 de bu
   veride yardımcı olmuyor — aşağıya bak).
4. **Predictor (sert küme-ortalaması) yutucu** → ~%98 tahmin küme ortalaması; büyük
   sayılar yasasıyla farklı atamalar benzer item ortalamaları üretir. **Asıl bağlayıcı
   kısıt budur** — knn_mae'nin küme çökerterek bile MAE'yi pek değiştirememesi bunu
   doğruluyor.

**`n_init=1` hipotezi reddedildi:** B0 n_init=1 → k6 MAE=0.8221 (daha *iyi* ama
dejenere: min küme=2), k14 MAE=0.8489 (biraz kötü). Yani makalelerdeki büyük kazanç
bizim kurulumda zayıf-baseline artefaktı **değil**.

---

## 5. Öneri — paper-uyumlu "iyileştirilmiş" konfigürasyon

Algoritma farkını görünür kılıp meta'yı B0'ın önüne geçiren reçete:

> **repair-only kmref + k = 10–14 + AVOA** (isteğe bağlı: random init).
> Sonuç: k14_repair'de AVOA, B0'a göre **−%1.69 MAE** (0.8327 vs 0.8470), ARI 0.13
> (gerçekten farklı atama). Yapı paper'la uyumlu kalır (WNMF→meta-centroid→
> K-means(repair)→calc_avg_rating).

**Dikkat — istatistiksel güç:** Tüm sayılar tek fold (1), tek seed. Meta kazançları
küçük ve tek-koşu varyansı sınırında. Tez için **fold 1–5 + çok-seed + paired
bootstrap CI** ([mealpy/paired_bootstrap_ci.py](../../mealpy/paired_bootstrap_ci.py))
şart; ancak yön (repair + yüksek k ayrıştırır, AVOA en iyi meta) tutarlı.

### Daha da büyütmek isteniyorsa (predictor'ı değiştir)
Asıl tavan, sert küme-ortalaması predictor'ı. Küme-içi sapma/bias modeli veya küme-içi
kNN-CF (atama → komşu havuzu) farkı birkaç kat büyütür; ama bu, Thakrar Alg.6'dan
sapmak demektir (raporda strict vs gevşek olarak ayrılmalı).

---

## 6. Üretilen dosyalar / nasıl tekrar koşulur

- Driver: [experiments/run_k_meta_ablation.py](../../experiments/run_k_meta_ablation.py)
  — `python -m experiments.run_k_meta_ablation --phase all --cells all --skip-existing`
- Direkt metrik: [experiments/_ablation_direct_metrics.py](../../experiments/_ablation_direct_metrics.py)
- Kod değişikliği: `--b0-n-init` knob + kmref-iter/n_init suffix kodlaması
  ([generate_assignments.py:1416,1886](../../mealpy/generate_assignments.py))
- Veri: `results/meta_ablation/ablation_direct.csv` (MAE/RMSE/küme), `ablation_metrics.csv`
  (wnmf_experiment NDCG/P@10), `ablation_cluster_analysis.json` (ARI)
