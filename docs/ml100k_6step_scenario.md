# ML-100K Uctan Uca 6-Adim Senaryosu

Bu senaryo su akisi calistirir:
- veri budama (`min_user_ratings>=5`, `min_item_ratings>=10`)
- kullanici bazli Z-score
- INMED (`trimmed mean` %5-%95) ile WNMF latent ozellik cikarma
- tum meta-sezgisel algoritmalarla assignment uretimi
- LOF tabanli gray sheep izolasyonu
- kume ici mean-centered kNN tahmin

## 1) Assignment uretimi (Adim 1-5)

```bash
python mealpy/generate_assignments.py \
  --dataset 100k \
  --lof \
  --min-user-ratings 5 \
  --min-item-ratings 10 \
  --zscore \
  --wnmf-features 20 \
  --wnmf-init inmed \
  --inmed-trim-low 5 \
  --inmed-trim-high 95
```

Paper mode (literatur preset, assignment-only):

```bash
python mealpy/generate_assignments.py \
  --dataset 100k \
  --paper-mode \
  --algo LIT_GOA \
  --k 30
```

Paper mode notlari:
- z-score otomatik acilir.
- PCA otomatik olarak `0.95` olur (`--pca` ile override edilebilir).
- `cluster-metric` otomatik `euclidean` olur.
- centroid init `random` olur.
- gray sheep tespiti kapatilir (LOF/percentile uretilmez).

Notlar:
- `--algo` verilmezse tum algoritmalar calisir.
- `--cluster-metric auto` varsayilaninda, `--wnmf-features` kullanildiginda `euclidean` secilir.
- Cikti klasor son eki:
  - `_pruneu5_i10_zscore_wnmf20_inmed_trim5_95`

## 2) Tahmin asamasi (Adim 6)

```bash
python wnmf/wnmf_experiment.py \
  --dataset 100k \
  --mode baselines \
  --assign-root mealpy/results/assignments_lof \
  --assign-suffix _pruneu5_i10_zscore_wnmf20_inmed_trim5_95 \
  --similarity pearson \
  --knn 30 \
  --min-common 3
```

Notlar:
- `cluster_knn` senaryosu mean-centered CF formulunu kullanir.
- Istersen `--mode sharedV` ile baselines + sharedV WNMF beraber calistirilabilir.

## 3) Dogrulama kontrol listesi

- `mealpy/results/assignments_lof/ml100k/*_pruneu5_i10_zscore_wnmf20_inmed_trim5_95/` klasorleri olusmus olmali.
- Her algoritma klasorunde su dosyalar olmali:
  - `assignments.npy`
  - `gray_sheep_mask.npy`
  - `lof_scores.npy`
  - `assignment_summary.csv`
- `wnmf/results/...` altindaki sonuc CSV'lerinde en az `cluster_knn` satirlari bulunmali.
- MAE/RMSE degerleri `gray` ve `white` ayrimiyla raporlaniyor olmali.

## 4) Uctan uca teknik akis (dosya + fonksiyon zinciri)

Bu bolum, "hangi dosya calisiyor, hangi fonksiyona gidiyor" sorusunu adim adim verir.

### Adim 1 - Entry point: assignment uretimi

- Calisan dosya: `mealpy/generate_assignments.py`
- Baslangic:
  - `if __name__ == '__main__'`
  - `parse_args()`
- Veri yukleme:
  - ML-100K: `load_movielens(...)`
  - ML-1M: `load_movielens_1m(...)`

### Adim 2 - Clustering girdisi hazirlama (preprocess)

- Fonksiyon: `prepare_matrix_for_clustering(...)`
- Sirayla:
  1. `prune_sparse_matrix(...)`
  2. Opsiyonel `zscore_normalize(...)`
  3. Opsiyonel `pca_variance_reduce(...)`
  4. Opsiyonel `wnmf_feature_extract(...)`

### Adim 3 - WNMF latent ozellik cikarimi (opsiyonel)

- Fonksiyon: `wnmf_feature_extract(...)` (`mealpy/generate_assignments.py`)
- Cagri zinciri:
  1. `WNMFModel(...)` (`wnmf/wnmf_model.py`)
  2. `model.fit(train_ratings, ...)`
  3. `fit()` icinde, egitim dongusunden once:
     - `if init_method == "inmed": _initialize_inmed_factors(...)`
  4. Epoch dongusu: `for epoch in range(self.n_epochs): ...`
  5. Cikti: `model.U.astype(np.float32)`
- Format:
  - `U` boyutu: `(n_users, k)`
  - Tip: `float32`
  - Bu matris, meta-sezgisel clustering girdisi olur.

### Adim 4 - Meta-sezgisel assignment optimizasyonu

- Fonksiyon: `run_dataset(...)` -> her algoritma icin `run_one(...)` -> `_run_one_core(...)`
- Yardimci fonksiyonlar (`mealpy/mealpy_comparison_v2.py`):
  - `mkmeans_plus_plus_init(...)` (baslangic populasyonu)
  - `make_fitness_function(...)` (amac fonksiyonu)
  - `compute_wcss_fast(...)` (fitness/assignment hesaplari)
- Algoritma kosumu:
  - Tek algoritma: `run_single(...)`
  - Hibrit: `run_hybrid(...)`
  - Paralel hibrit: `run_parallel_hybrid(...)`
  - Ozel yollar: `run_kmeans_baseline(...)`, `run_gahho(...)`, `run_elitega_hho(...)`

### Adim 5 - Gray sheep tespiti ve assignment kaydi

- Gray sheep:
  - LOF acik: `detect_gray_sheep_lof(...)`
  - LOF kapali: `detect_gray_sheep_percentile(...)`
- Kayit:
  - `save_assignment(...)`
  - Uretilen dosyalar:
    - `assignments.npy`
    - `gray_sheep_mask.npy`
    - `best_sol.npy`
    - `lof_scores.npy` (LOF varsa)
    - `assignment_summary.csv`

### Adim 6 - WNMF deney/evaluasyon asamasi

- Calisan dosya: `wnmf/wnmf_experiment.py`
- Baslangic:
  - `if __name__ == '__main__'`
  - `parse_args()`
- Veri yukleme (`wnmf/wnmf_utils.py`):
  - `load_ratings_100k(...)`
  - `load_ratings_1m(...)`
- Ana kosucu:
  - `run_dataset(...)`

### Adim 7 - Assignment yukle ve senaryolari calistir

- `wnmf/wnmf_experiment.py::run_dataset(...)` icinde:
  1. Assignment klasoru bulunur (`_algo_assignment_dir(...)`)
  2. `load_assignment(assign_dir)` (`wnmf/wnmf_utils.py`)
     - `assignments.npy`
     - `gray_sheep_mask.npy`
  3. Senaryo calismalari:
     - `run_cluster_knn(...)` (baseline)
     - Opsiyonel `run_cluster_average(...)`
     - Mode'a gore:
       - `run_cluster_full(...)`
       - `run_cluster_sharedV(...)`
     - Opsiyonel global:
       - `run_global_wnmf(...)`

### Adim 8 - SharedV detay yolu (mode=sharedV/all)

- `run_cluster_sharedV(...)` icinde:
  1. `WNMFSharedV(...)` olusturulur
  2. `fit_global_V(train, ...)` ile global `V` ogrenilir
  3. `split_by_cluster(...)` ile train/test cluster bazli bolunur
  4. Cluster worker'lari (`_mp_fit_predict_cluster_sharedV`) calisir
  5. MAE/RMSE + gray/white metrikleri raporlanir

### Adim 9 - Sonuc dosyalari

- Assignment ciktilari:
  - `mealpy/results/assignments_lof/{dataset}/{algo...}/...`
- WNMF sonuc ciktilari:
  - `results/wnmf/{dataset}/k{K}/run{N}/wnmf_results_{dataset}_k{K}_{mode}.csv`
  - `mode=all` ise ek split CSV dosyalari da yazilir.
