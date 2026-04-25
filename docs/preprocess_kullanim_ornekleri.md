# Preprocess Kullanim Ornekleri

Bu dosya, `generate_assignments.py` icindeki preprocess adimlarinin
"sirali mi / ayri mi" sorusuna net cevap ve birebir komut ornekleri verir.

## 1) Mantik: Sirali pipeline + secimli dal

Kod akisi (`prepare_matrix_for_clustering`):
1. `prune_sparse_matrix(...)` (her zaman)
2. `zscore_normalize(...)` (sadece `--zscore` varsa)
3. `pca_variance_reduce(...)` (sadece `--pca` varsa)
4. `wnmf_feature_extract(...)` (sadece `--wnmf-features` varsa)

CLI kurali:
- `--pca` ve `--wnmf-features` birlikte verilemez.

Bu nedenle pratik akis:
- `prune -> (opsiyonel) zscore -> PCA`
veya
- `prune -> (opsiyonel) zscore -> WNMF`

---

## 2) Komut Ornekleri

## A) Sadece prune

```bash
python mealpy/generate_assignments.py --dataset 100k --min-user-ratings 5 --min-item-ratings 10
```

Akis:
- `prune`

## B) prune + zscore

```bash
python mealpy/generate_assignments.py --dataset 100k --min-user-ratings 5 --min-item-ratings 10 --zscore
```

Akis:
- `prune -> zscore`

## C) prune + zscore + PCA

```bash
python mealpy/generate_assignments.py --dataset 100k --min-user-ratings 5 --min-item-ratings 10 --zscore --pca 0.80
```

Akis:
- `prune -> zscore -> PCA`

## D) prune + zscore + WNMF (sik kullanim)

```bash
python mealpy/generate_assignments.py --dataset 100k --lof --min-user-ratings 5 --min-item-ratings 10 --zscore --wnmf-features 20 --wnmf-init inmed --inmed-trim-low 5 --inmed-trim-high 95
```

Akis:
- `prune -> zscore -> WNMF(U)`

## E) prune + WNMF (zscore olmadan)

```bash
python mealpy/generate_assignments.py --dataset 100k --lof --min-user-ratings 5 --min-item-ratings 10 --wnmf-features 20 --wnmf-init inmed --inmed-trim-low 5 --inmed-trim-high 95
```

Akis:
- `prune -> WNMF(U)`

---

## 3) Gecersiz Kombinasyon Ornegi

Asagidaki komut gecersizdir:

```bash
python mealpy/generate_assignments.py --dataset 100k --pca 0.80 --wnmf-features 20
```

Sebep:
- parser kontrolu: `--pca` ile `--wnmf-features` birlikte kullanilamaz.
