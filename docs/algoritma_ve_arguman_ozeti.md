# Hybrid Recommendations: Arguman ve Algoritma Ozeti

Bu dokuman, projede su ana kadar kullandigimiz temel argumanlari, algoritmalari, hesaplama metriklerini ve bunlarin kodda nerede kullanildigini tek yerde toplar.

Kapsam:
- `mealpy/generate_assignments.py`
- `mealpy/mealpy-algorithms-comparision.py`
- `wnmf/wnmf_experiment.py`
- `assignment_db.py`
- `docs/ml100k_6step_scenario.md`

---

## 1) Uctan Uca Akis (Kisa Ozet)

1. Veri yuklenir (`ml-100k` veya `ml-1m`)
2. Clustering matrisi hazirlanir (sira vardir): `prune -> (opsiyonel) zscore -> (opsiyonel) PCA veya (opsiyonel) WNMF U`
3. Meta-sezgisel algoritmalar ile assignment (kume atama) uretilir
4. Gray sheep tespiti yapilir (percentile veya LOF)
5. Assignment dosyalari kaydedilir (`assignments.npy`, `gray_sheep_mask.npy`, vs.)
6. WNMF deney asamasinda bu assignment'lar yuklenerek:
   - `cluster_avg`
   - `cluster_knn`
   - `cluster_full`
   - `cluster_sharedV`
   - (opsiyonel) `global`
   senaryolari kosulur

Onemli not:
- Bu adimlar pipeline sirasini gosterir.
- `prune` her durumda once calisir.
- `zscore` aciksa `prune` sonrasi uygulanir.
- `PCA` ile `WNMF` ayni kosuda birlikte acilmaz (`--pca` + `--wnmf-features` yasak).
- Yani pratikte iki ayrik dal vardir: `... -> PCA` veya `... -> WNMF`.
- Komut ornekleri icin: `docs/preprocess_kullanim_ornekleri.md`

---

## 2) Assignment Uretiminde Kullanilan Algoritmalar

Kaynak: `mealpy/generate_assignments.py` icindeki `ALGO_CONFIG`.

- `B0_KMEANS` -> klasik KMeans baseline
- `B1_HHO` -> Harris Hawks Optimization
- `B2_HGS` -> Hunger Games Search
- `B3_MFO` -> Moth Flame Optimization
- `H1_HHO+HGS` -> sirali hibrit (global + local)
- `H4_MFO+HHO` -> sirali hibrit
- `H5_GAHHO` -> GA operatorlu HHO varyanti
- `H5_EliteGA+HHO` -> Elite GA + HHO hibriti
- `H9_QSA+CDO` -> QSA + CDO hibriti
- `H12_MFO+CDO` -> MFO + CDO hibriti
- `H13_HHO+GAop` -> memetic yaklasim (HHO + periyodik GA enjeksiyonu)
- `LIT_GOA` -> literatur tabanli GOA
- `LIT_GWO` -> literatur tabanli GWO
- `LIT_SSA` -> literatur tabanli SSA

Not:
- Kumeleme fitness'i `make_fitness_function(...)` ile olusturulur.
- Tum meta-sezgisel yontemler ayni problem tanimini (centroid optimizasyonu) minimize eder.

---

## 3) Assignment Asamasi Argumanlari (`generate_assignments.py`)

### 3.1 Veri ve algoritma secimi
- `--dataset {100k,1m,both}`
- `--algo LABEL [LABEL ...]`
- `--last-only`

### 3.2 Gray sheep modu
- `--lof` -> LOF tabanli adaptif tespit
- `--no-gray-sheep` -> gray sheep tespitini tamamen kapatir
- `--n-neighbors` -> LOF komsu sayisi
- `--contamination` -> `auto` veya oran

LOF verilmezse:
- sabit percentile tabanli tespit (80. percentile, yaklasik %20 gray sheep)
- `%20` istemiyorsan `--no-gray-sheep` kullan.

### 3.3 On-isleme / ozellik cikarma
- `--min-user-ratings` (vars. 5)
- `--min-item-ratings` (vars. 10)
- `--zscore`
- `--pca` (0,1] varyans esigi
- `--paper-mode` (zscore+pca+euclidean+random init preset, gray sheep kapali)
- `--wnmf-features K`
- `--wnmf-init {random,inmed}`
- `--inmed-trim-low`, `--inmed-trim-high`
- `--cluster-metric {auto,pearson,euclidean}`

Kural:
- `--pca` ve `--wnmf-features` birlikte kullanilamaz.
- `--paper-mode` ve `--wnmf-features` birlikte kullanilamaz.
- `--cluster-metric auto`:
  - `wnmf-features` varsa `euclidean`
  - yoksa `pearson`
- `--paper-mode`:
  - `zscore` zorunlu acik
  - `pca` varsayilan `0.95` (istenirse `--pca` ile degistirilebilir)
  - `cluster-metric` `euclidean` olarak zorlanir
  - init modu `random`
  - gray sheep kapali

### 3.4 K ve paralellik
- `--k` (tek veya coklu)
- `--k-100k`, `--k-1m` (dataset bazli ayri K)
- `--jobs`

### 3.5 Cikti ve isimlendirme
- `--save-wnmf-u DIR`
- preprocess suffix mantigi:
  - `_pruneu5_i10`
  - `_zscore`
  - `_pca80pct`
  - `_wnmf20_inmed_trim5_95`

Ornek:
- `_pruneu5_i10_zscore_wnmf20_inmed_trim5_95`

---

## 4) WNMF Deney Asamasi Argumanlari (`wnmf_experiment.py`)

### 4.1 Temel secimler
- `--dataset {100k,1m,both}`
- `--algo LABEL [LABEL ...]`
- `--mode {baselines,sharedV,full,all}`
- `--no-global`

### 4.2 Assignment kaynagi
- `--assign-root`
- `--assign-suffix`
- `--k` (tek/coklu)
- `--k-100k`, `--k-1m`

### 4.3 Model hiperparametreleri
- `--latent-dim`
- `--lr`
- `--reg`
- `--epochs-global`
- `--epochs-cluster`
- `--epochs-grid` (cartesian kombinasyon)

### 4.4 Paralellik
- `--jobs` -> kume ici surec sayisi
- `--algo-jobs` -> algoritma bazli paralellik

Not:
- `--algo-jobs > 1` olunca asiri CPU yuklenmesini onlemek icin kume-ici paralellik 1'e cekilir.

### 4.5 Senaryo detaylari
- `--weighted-v`
- `--no-bias`
- `--cluster-bias`
- `--svdpp`
- `--compare-mf-svdpp`

### 4.6 Baseline kontrolu
- `--no-cluster-avg`
- `--no-cluster-knn`
- `--similarity {pearson,cosine}`
- `--knn`
- `--min-common`
- `--top-n`
- `--relevance-threshold`

---

## 5) Formuller ve Metrikler

## 5.1 Clustering mesafeleri ve fitness

Kaynak: `mealpy/mealpy-algorithms-comparision.py`

- Euclidean squared distance:
  - `d(u,c)^2 = ||u-c||^2 = ||u||^2 + ||c||^2 - 2 u.c`
- Pearson distance:
  - `distance = 1 - corr(u,c)`
  - seyrek veri icin non-zero odakli merkezleme kullanilir

WCSS / objective:
- Her kullanici en yakin centroid'e atanir
- `WCSS = sum_i min_k distance(x_i, c_k)`
- Meta-sezgisel algoritmalar bu degeri minimize eder (`minmax="min"`).

## 5.2 MkMeans++ baslangic mantigi

- Ilk merkez rastgele secilir
- Sonraki merkezler mevcut merkezlere uzakliga gore olasiliksal secilir
- `probs ~ distance^2`
- Populasyon icin coklu cozum uretilir

## 5.3 Gray sheep tespiti

### Percentile modu
- Kullanicinin atandigi merkezden uzakligi hesaplanir
- Esik verilmezse 80. percentile esik olur
- `gray_mask = user_distance > threshold`

### LOF modu
Ozellik vektoru (kullanici bazli) 4 bilesen:
1. ortalama rating
2. normalize rating sayisi
3. rating std
4. kume icine gore fark (`|r_ui - cluster_mean_i|` ortalamasi)

Sonra:
- Ozellikler kolon bazli z-score edilir
- `LocalOutlierFactor` ile outlier tespiti yapilir
- `prediction == -1` olanlar gray sheep

## 5.4 Kume ici tahmin (ClusterKNN)

Kaynak: `wnmf/wnmf_experiment.py` -> `run_cluster_knn`

Pearson similarity:
- `sim(u,v) = sum((r_ui-mean_u)(r_vi-mean_v)) / sqrt(sum(...) * sum(...))`

Cosine similarity:
- `sim(u,v) = sum(r_ui*r_vi) / (||r_u|| ||r_v||)`

Mean-centered tahmin:
- `pred(u,i) = mean_u + sum(sim(u,v)*(r_vi-mean_v)) / sum(|sim(u,v)|)`

Kurallar:
- Sadece ayni kumedeki komsular
- En fazla `k_neighbors` komsu
- Ortak item sayisi `< min_common` ise benzerlik 0 kabul
- Son tahmin `[1,5]` araligina clip edilir

## 5.5 WNMF senaryolari

### Global WNMF
- Tum kullanicilar tek model
- Ablation baseline

### Cluster Full
- Her kume icin bagimsiz `U` ve `V` ogrenilir

### Cluster SharedV (onerilen)
- Asama 1: tum train ile global `V` ogren
- Asama 2: her kumede `V` sabit, sadece `U` ogren

Avantaj:
- item embedding'leri tum veriden geldiği icin daha stabil
- kume-ozel kullanici faktorleriyle uzmanlasma saglar

## 5.6 Degerlendirme metrikleri

MAE:
- `MAE = mean(|y_true - y_pred|)`

RMSE:
- `RMSE = sqrt(mean((y_true - y_pred)^2))`

Top-N (cluster_avg):
- Precision@N, Recall@N, F1@N
- relevant: `rating >= relevance_threshold`

Ek raporlar:
- `gray_mae`, `gray_rmse`
- `white_mae`, `white_rmse`
- cluster bazli MAE dagilimi (std/min/max)

---

## 6) "Su Ana Kadar" En Cok Gecen Calisma Pattern'leri

Mevcut dosya adlari ve senaryo notlarina gore en sik gorulen pattern:

- Dataset:
  - agirlikla `ml100k`
- Gray sheep:
  - genelde `--lof`
- On-isleme:
  - `--min-user-ratings 5`
  - `--min-item-ratings 10`
  - siklikla `--zscore`
  - siklikla `--wnmf-features` (10/20/50 gibi)
  - `--wnmf-init inmed`, `trim 5-95`
- K denemeleri:
  - `10, 15, 30, 50` gibi coklu K taramalari
- WNMF asamasi:
  - `cluster_knn` ve `cluster_sharedV` odagi
  - `similarity=pearson`, `knn=30`, `min_common=3`

---

## 7) Dosya - Sorumluluk Haritasi

- `mealpy/generate_assignments.py`
  - CLI argumanlari
  - preprocess pipeline
  - assignment uretimi
  - LOF / percentile gray sheep
  - assignment kaydi

- `mealpy/mealpy-algorithms-comparision.py`
  - mesafe fonksiyonlari
  - WCSS/fitness
  - mkmeans++ init
  - algorithm catalog ve ranking mantigi

- `wnmf/wnmf_experiment.py`
  - assignment yukleme
  - global/full/sharedV senaryolari
  - cluster_avg ve cluster_knn baseline
  - tum MAE/RMSE ve top-N metrikleri

- `assignment_db.py`
  - run/assignment/wnmf sonucu sqlite kaydi

---

## 8) Hizli Komut Sablonlari

### Assignment uretimi (ML-100K, LOF + zscore + WNMF)

```bash
python mealpy/generate_assignments.py --dataset 100k --lof --min-user-ratings 5 --min-item-ratings 10 --zscore --wnmf-features 20 --wnmf-init inmed --inmed-trim-low 5 --inmed-trim-high 95
```

### WNMF deneyi (sharedV + cluster_knn baseline)

```bash
python wnmf/wnmf_experiment.py --dataset 100k --mode sharedV --assign-root mealpy/results/assignments_lof --assign-suffix _pruneu5_i10_zscore_wnmf20_inmed_trim5_95 --similarity pearson --knn 30 --min-common 3
```

---

## 9) Run Kronolojisi (Terminal Logundan Otomatik)

Kaynak:
- terminal kaydi: `terminals/5.txt`
- Bu bolum yalnizca logda gercekten gorunen komut/surec ciktilarindan uretilmistir.

Not:
- Ilk assignment kosusunun komut satiri log kesitinde gorunmuyor; ancak ciktilardan ayni kosunun basariyla tamamlandigi net.

### 9.1 Zaman Cizelgesi (Tablo)

| Sira | Tip | Komut | Durum | MAE | RMSE | Cikti klasoru / not |
|---|---|---|---|---|
| 1 | Assignment | _(komut satiri logda yok)_ | Basarili | - | - | `mealpy/results/assignments_lof/ml100k/` altina `k30 + _pruneu5_i10_wnmf20_inmed_trim5_95` assignment'lari yazildi; logda `TAMAMLANDI — toplam 1.9 dakika` |
| 2 | Deneme | `--dataset 100k --mode baselines --no-global --no-cluster-knn --k 30 ...` | Hatali | - | - | PowerShell parser: `Missing expression after unary operator '--'` (komut `python` olmadan girilmis) |
| 3 | Deneme | `python .\wnmf\wnmf_experiment.py ... --no-cluster-knn ...` | Hatali | - | - | `unrecognized arguments: --no-cluster-knn` |
| 4 | WNMF baseline | `python .\wnmf\wnmf_experiment.py --dataset 100k --mode baselines --no-global --k 30 --assign-root mealpy/results/assignments_lof --assign-suffix _pruneu5_i10_wnmf20_inmed_trim5_95` | Basarili | 0.8614* | 1.0980* | `results/wnmf/ml100k/k30/run62` (yalnizca `cluster_avg`, 11 algoritma), sure `~0.1 dk` |
| 5 | WNMF baseline | `python wnmf/wnmf_experiment.py --dataset 100k --mode baselines --k 30 --assign-root mealpy/results/assignments_lof --assign-suffix _pruneu5_i10_wnmf20_inmed_trim5_95 --knn 30 --min-common 3` | Basarili | 0.7793** | 1.0111** | `results/wnmf/ml100k/k30/run63` (`global + cluster_avg + cluster_knn`), sure `~3.1 dk` |

### 9.2 Run63 Kisa Ozet (logdan)

- `GLOBAL_WNMF`: `MAE=0.7793`, `RMSE=1.0111`
- `cluster_knn` tarafinda en iyi MAE (logdaki ozet satirlarina gore): `LIT_GOA = 0.8079`
- KNN ayarlari: `similarity=pearson`, `k=30`, `min_common=3`

Tablodaki MAE/RMSE notasyonu:
- `*` : run62'de tek bir "overall" satir olmadigi icin en iyi `cluster_avg` satiri yazildi (`LIT_GOA`).
- `**`: run63'te `GLOBAL_WNMF` satiri yazildi.

### 9.3 Hata Notu (CLI uyumsuzlugu)

- Log, `--no-cluster-knn` bayraginin parser tarafinda tanimli olmadigini gosteriyor.
- Buna karsin `--no-cluster-avg` calisiyor.
- Yani su anki calisan pratik:
  - sadece `cluster_avg` icin: `--no-global` ile kosu (run62 gibi)
  - `cluster_knn` kapatma secenegi mevcut parser'da yok.


