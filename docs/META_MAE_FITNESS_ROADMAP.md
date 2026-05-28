# Meta-algoritma farkını gösterme — yol haritası

## Sorun özeti

- Meta atamaları WCSS’te birbirine yakın; **downstream cluster kNN** tahminleri de çok korelasyonlu (r ≈ 0.79–0.90).
- `kmref` (KMeans Lloyd) atamaları homojenleştiriyor.
- Fitness (WCSS) ≠ ölçtüğünüz metrik (holdout MAE).

## Faz 1 — Fitness = küme-içi kNN MAE (şimdi)

**Amaç:** Meta-algo centroid ararken doğrudan `ClusterPredictor` val MAE’sini minimize etsin.

### Adımlar

1. **Assignment üret (pilot, tek K, tek algo)**
   ```bash
   python mealpy/generate_assignments.py --dataset 100k --k 30 \
     --algo B1_HHO --no-gray-sheep --no-prune \
     --preprocess none --feature-extraction wnmf --svd-components 20 \
     --init-mode mkpp --no-kmeans-refine \
     --fitness knn_mae --train-only --fold 1 \
     --centroid-knn-sim cosine --centroid-knn-k 30 \
     --centroid-val-sample 500 --centroid-iter 30 --centroid-agents 20
   ```
   - `--fitness knn_mae` → `CentroidOptimizer` + `ClusterPredictor` (cosine, küme bias SGD).
   - `--no-kmeans-refine` → kmref kapalı (meta farkı korunur).
   - `--train-only` + fold 1 → val = holdout test (fitness ile eval uyumlu).

2. **B0 baseline (aynı fitness ile adil karşılaştırma)**
   ```bash
   python mealpy/generate_assignments.py --dataset 100k --k 30 \
     --algo B0_KMEANS ... (aynı bayraklar, fitness knn_mae)
   ```

3. **Holdout değerlendirme (native kNN)**
   ```bash
   python wnmf/wnmf_experiment.py --dataset 100k --eval-split random --fold 1 \
     --mode baselines --no-global --no-cluster-avg \
     --cluster-knn-backend native --lambda-shrink 25 \
     --lambda-shrink-de \
     --similarity cosine --knn 30 \
     --k 30 --algo B0_KMEANS B1_HHO \
     --assign-root mealpy/results/assignments \
     --assign-suffix _euc_imkpp_nogs_none_wnmf20_k30
   ```
   (kmref yok → suffix’te `_kmref` olmamalı)

4. **Anlamlılık**
   - `python mealpy/diagnose_algo_pred_correlation.py --k 30 --wnmf-dim 20`
   - `python mealpy/paired_bootstrap_ci.py` (B0 vs meta MAE farkı)

### Beklenti

- Meta’lar **farklı assignment** üretir → tahmin korelasyonu düşer, MAE spread artar.
- Optimizasyon yavaştır (her fitness = fit + sim + val); pilot için küçük `--centroid-iter` / tek algo kullanın.

---

## Faz 2 — WNMF 40, kmref yok, tam grid (sonraki)

| Parametre | Değer |
|-----------|--------|
| WNMF | `--svd-components 40` |
| kmref | `--no-kmeans-refine` |
| Fitness | `--fitness knn_mae` |
| K | 5, 30, 70 |
| Algo | B0, B1_HHO, B_AVOA, HA_AVOAHGS, IWO_HHO |
| Eval | `wnmf_experiment` native, kNN 5/30/70 |

```bash
# Örnek assignment suffix: _euc_imkpp_nogs_none_wnmf40_k30  (_kmref YOK)

python mealpy/generate_assignments.py --dataset 100k --k 30 \
  --algo B1_HHO B_AVOA HA_AVOAHGS IWO_HHO B0_KMEANS \
  --no-gray-sheep --no-prune --preprocess none \
  --feature-extraction wnmf --svd-components 40 \
  --init-mode mkpp --no-kmeans-refine \
  --fitness knn_mae --train-only --fold 1 \
  --centroid-knn-sim cosine --centroid-knn-k 30
```

---

## Faz 3 — İsteğe bağlı iyileştirmeler

- **Tam train MAE fitness** (örneklemesiz, yavaş): `--centroid-val-sample` = tüm holdout.
- **Meta swarm + MAE:** `H4_MFO+HHO` vb. için `make_fitness_function` yerine iki aşamalı (önce centroid opt, sonra ince ayar).
- **kmref A/B:** aynı MAE-fitness assignment’da kmref açık/kapalı çift koşu.

---

## Karar matrisi

| Seçenek | Artı | Eksi |
|---------|------|------|
| WCSS + kmref (eski) | Hızlı, stabil | Meta ≈ B0 downstream |
| knn_mae + kmref | Orta | kmref yine sönümler |
| **knn_mae + no kmref** | Assignment farkı | Yavaş üretim |
| WNMF40 | Daha zengin uzay | Eski wnmf20 ile karşılaştırma ayrı koşu |

---

## Dosya referansları

Tam parametre tablosu, suffix anatomisi ve hizalama kontrol listesi: **`docs/PARAMETRELER_EXPERIMENT_ASSIGNMENT.md`**.

| Bileşen | Dosya |
|---------|--------|
| MAE fitness (centroid arama) | `mealpy/centroid_optimizer.py` → `cluster_predictor_mae` |
| Assignment üretimi | `mealpy/generate_assignments.py` → `--fitness knn_mae` |
| Downstream tahmin | `wnmf/cluster_predictor.py` |
| Eval | `wnmf/wnmf_experiment.py` → `--cluster-knn-backend native` |
