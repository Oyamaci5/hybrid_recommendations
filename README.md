# hybrid_recommendations

MovieLens tabanlı öneri ve kümeleme deneyleri: meta-sezgisel optimizasyon (**mealpy**), kullanıcı küme atamaları, ağırlıklı NMF (**WNMF**) ve deney sonuçlarının SQLite ile izlenmesi.

## Kurulum

```bash
cd hybrid_recommendations
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

## Veri

- **ML-100K:** `data/ml-100k/` (ör. `u.data`, `u1.base` / `u1.test`)
- **ML-1M:** `data/ml-1m/ratings.dat`

Veri setleri [MovieLens](https://grouplens.org/datasets/movielens/) kaynağından indirilip bu yapıya yerleştirilmelidir.

## Önemli dizinler

| Dizin / dosya | Açıklama |
|---------------|----------|
| `mealpy/` | Küme atama üretimi (`generate_assignments.py`), hibrit testler, `mealpy_comparison_v2` |
| `methods/`, `models/` | WNMF deney koşucusu (`methods/wnmf_experiment.py`) ve WNMF modelleri (`models/wnmf.py`) |
| `optimizers/` | COA, HHO, PSO vb. sarmalayıcılar |
| `core/`, `models/`, `methods/`, `experiments/` | MF / deney altyapısı |
| `assignment_db.py` | Ortak SQLite API (`results/assignment_experiments.sqlite`) |

## Tipik komutlar

**Küme atamaları (LOF gray sheep örneği):**

```bash
cd mealpy
python generate_assignments.py --lof --dataset both --algo B1_HHO --k 30
```

**WNMF deneyi:**

```bash
cd hybrid_recommendations
python methods/wnmf_experiment.py --dataset 100k --k 30 --mode sharedV
```

**Deney veritabanı (repo kökünden):**

```bash
cd hybrid_recommendations
python assignment_db.py runs
python assignment_db.py run <id>
```

## Bağımlılıklar

`requirements.txt` içinde: **numpy**, **pandas**, **scipy**, **scikit-learn**, **matplotlib**, **mealpy**. İsterseniz yalnızca kullandığınız alt projeye göre ortamı sadeleştirebilirsiniz (ör. WNMF için `matplotlib` şart değildir).

## Lisans

Proje içi lisans dosyası yoksa depo sahibinin belirttiği koşullar geçerlidir.


DBeaver (veya benzeri) ile bağlanma
Yeni bağlantı → SQLite seçin.
Path / Database file: tam dosya yolu, örneğin Windows’ta
d:\hybrid_recommendations\results\assignment_experiments.sqlite
Kullanıcı/şifre/port yok (SQLite dosya tabanlı).
Bağlanmadan önce dosyanın gerçekten var olduğundan emin olun; yoksa önce script’i bir kez çalıştırıp DB’yi oluşturun.
Tablolar: runs, assignments, wnmf_results. BLOB sütunlar (ör. assignments_npy) ikili veri; DBeaver’da ham byte olarak görürsünüz.