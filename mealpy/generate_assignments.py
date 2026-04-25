"""
generate_assignments.py
=======================
ALGO_CONFIG listesindeki algoritmalar için (başta B1/B2/H1/H4, sonra B3, hibritler, LIT_*)
ML-100K ve ML-1M üzerinde tek run çalıştırır, assignment kaydeder.

Gray sheep tespiti:
    Varsayılan : Sabit 80. percentile (~%20 gray sheep)
    --lof flag  : LOF tabanlı adaptif threshold (önerilen)

Çıktı yapısı:
    --lof verilmezse : results/assignments/
    --lof verilirse  : results/assignments_lof/

    Her ikisinde de:
    ├── ml100k/
    │   ├── B1_HHO/              ← K=90 (default)
    │   ├── B1_HHO_k70/          ← K=70 (--k 70)
    │   └── H4_MFO+HHO/
    └── ml1m/
        └── ...

Kullanım:
    python generate_assignments.py                            # varsayılan
    python generate_assignments.py --lof                      # LOF gray sheep
    python generate_assignments.py --dataset 100k --algo H4_MFO+HHO
    python generate_assignments.py --last-only               # sadece son algoritma
    python generate_assignments.py --lof --k 70               # LOF + K=70
    python generate_assignments.py --lof --k 20 30 50 90     # aynı koşuda birden fazla K
    python generate_assignments.py --k-100k 70 --k-1m 120     # ayrı K (--k çoklu ile birlikte kullanılmaz)
    python generate_assignments.py --lof --n-neighbors 15 --contamination 0.1
    python generate_assignments.py --jobs 4              # algoritmaları paralel süreçte
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(BASE_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, BASE_DIR)
_OPT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'optimizers')
if _OPT_DIR not in sys.path:
    sys.path.insert(0, _OPT_DIR)

try:
    from assignment_db import (
        finish_run,
        init_db,
        save_assignment as db_save,
        start_run,
    )
    init_db()
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False
    finish_run = None
    start_run = None

from mealpy_comparison_v2 import (
    load_movielens,
    mkmeans_plus_plus_init,
    make_fitness_function,
    compute_wcss_fast,
    compute_fcm_objective,
    detect_gray_sheep,
    get_all_algorithms_v3,
    get_special_params,
)
from core.fitness import calculate_clustering_fitness
from mealpy import FloatVar
from mealpy.evolutionary_based import GA
try:
    from ga_hho_optimizer import OriginalGAHHO  # pyright: ignore[reportMissingImports]
except ImportError:
    from ga_hho import OriginalGAHHO  # pyright: ignore[reportMissingImports]
from de_hho import DE_HHO  # pyright: ignore[reportMissingImports]

# ============================================================
# AYARLAR
# ============================================================

DATA_100K = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-100k', 'u.data')
DATA_1M   = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-1m',   'ratings.dat')

K_100K_DEFAULT = 30
K_1M_DEFAULT   = 70

GLOBAL_EPOCH   = 20
LOCAL_EPOCH    = 15
BASELINE_EPOCH = 100
POP_SIZE       = 15
SEED           = 42

LOF_N_NEIGHBORS   = 20
LOF_CONTAMINATION = 'auto'

ALGO_CONFIG = [
    ('B0_KMEANS',  'KMEANS', None),
    ('B1_HHO',     'HHO.OriginalHHO', None),
    ('B2_HGS',     'HGS.OriginalHGS', None),
    ('H1_HHO+HGS', 'HHO.OriginalHHO', 'HGS.OriginalHGS'),
    ('H4_MFO+HHO', 'MFO.OriginalMFO', 'HHO.OriginalHHO'),
    ('B3_MFO',     'MFO.OriginalMFO', None),
    ('H9_QSA+CDO',  'QSA.OriginalQSA',  'CDO.OriginalCDO'),
    ('H12_MFO+CDO', 'MFO.OriginalMFO',  'CDO.OriginalCDO'),
    ('H13_HHO+GAop', 'HHO.OriginalHHO', 'GAop'),
    ('H5_GAHHO',   'GAHHO.OriginalGAHHO', None),
    ('DE_HHO', 'DE_HHO', None),
    ('H5_EliteGA+HHO', 'GA.EliteMultiGA', 'HHO.OriginalHHO'),
    ('LIT_GOA', 'GOA.OriginalGOA', None),
    ('LIT_GWO', 'GWO.OriginalGWO', None),
    ('LIT_SSA', 'SSA.DevSSA', None),
]

ALGO_LABELS = [c[0] for c in ALGO_CONFIG]

# hybrid_test / rapor ile aynı etiketler; Wilcoxon çiftleri (klasör adı = ilk sütun)
WILCOXON_PAIRS = [
    ('H5_EliteGA+HHO', 'H4_MFO+HHO', 'GA global vs MFO global'),
    ('H5_EliteGA+HHO', 'B1_HHO',     'H5 vs HHO tek'),
    ('H5_EliteGA+HHO', 'B3_MFO',     'H5 vs MFO tek'),
]


def _resolve_pool_workers(requested: Optional[int], n_tasks: int) -> int:
    if n_tasks <= 0:
        return 1
    cpu = os.cpu_count() or 1
    if requested is None or requested <= 0:
        cap = cpu
    else:
        cap = requested
    return max(1, min(cap, n_tasks))


# ============================================================
# GRAY SHEEP TESPİTİ — İKİ MOD
# ============================================================

def detect_gray_sheep_percentile(matrix, assignments, solution, K,
                                 metric: str = 'pearson'):
    """
    Sabit 80. percentile ile gray sheep tespiti.
    Her zaman ~%20 gray sheep üretir.
    Orijinal detect_gray_sheep fonksiyonunu çağırır.
    """
    return detect_gray_sheep(matrix, assignments, solution, K, metric=metric)


def detect_gray_sheep_lof(matrix, assignments, n_neighbors=LOF_N_NEIGHBORS,
                          contamination=LOF_CONTAMINATION):
    """
    LOF tabanlı adaptif gray sheep tespiti.

    Sabit percentile yerine LOF kullanmanın avantajı:
    - Threshold veriden otomatik belirlenir
    - Gray sheep sayısı veri yapısına göre değişir
    - Jüri sorusuna net metodolojik cevap verilebilir:
      'Neden %20?' değil, 'LOF skoru eşiği aştığında gray sheep'

    Parametreler
    ------------
    matrix        : (n_users, n_items) rating matrisi
    assignments   : (n_users,) küme atamaları
    n_neighbors   : LOF komşu sayısı
    contamination : 'auto' veya float (0-0.5)
    """
    from sklearn.neighbors import LocalOutlierFactor

    n_users  = len(matrix)
    features = _build_lof_features(matrix, assignments)

    lof = LocalOutlierFactor(
        n_neighbors   = min(n_neighbors, n_users - 1),
        contamination = contamination,
        metric        = 'euclidean',
        novelty       = False,
    )
    lof.fit(features)

    lof_scores  = -lof.negative_outlier_factor_
    predictions = lof.fit_predict(features)
    gray_mask   = predictions == -1

    if gray_mask.sum() > 0 and (~gray_mask).sum() > 0:
        threshold = float(
            (lof_scores[gray_mask].min() + lof_scores[~gray_mask].max()) / 2
        )
    else:
        threshold = float(lof_scores.mean())

    return {
        'gray_sheep_mask' : gray_mask,
        'lof_scores'      : lof_scores,
        'threshold'       : threshold,
        'gray_sheep_count': int(gray_mask.sum()),
        'gray_sheep_ratio': float(gray_mask.mean()),
    }


def _build_lof_features(matrix, assignments):
    """
    LOF için 4 özellik: ort. rating, rating sayısı, std, küme içi fark.
    Doğrudan seyrek matrisi kullanmak yerine anlamlı özetler kullanılır.
    """
    n_users  = len(matrix)
    features = np.zeros((n_users, 4), dtype=np.float32)

    for u in range(n_users):
        rated   = matrix[u][matrix[u] > 0]
        n_rated = len(rated)

        if n_rated == 0:
            features[u] = [0, 0, 0, 1]
            continue

        avg_r = float(rated.mean())
        std_r = float(rated.std()) if n_rated > 1 else 0.0
        cnt_r = float(n_rated)

        cid          = int(assignments[u])
        cluster_mask = assignments == cid
        if cluster_mask.sum() > 1:
            c_mat   = matrix[cluster_mask]
            c_means = c_mat[:, matrix[u] > 0].mean(axis=0)
            diff    = float(np.abs(rated - c_means).mean())
        else:
            diff = 1.0

        features[u] = [avg_r, cnt_r / matrix.shape[1], std_r, diff]

    for col in range(features.shape[1]):
        s = features[:, col].std()
        if s > 0:
            features[:, col] = (features[:, col] - features[:, col].mean()) / s

    return features


# ============================================================
# ALGORİTMA ÇALIŞTIRICILAR
# ============================================================

def _centroid_value_upper_bound(matrix: np.ndarray) -> float:
    """Rating veya latent uzayında merkez bileşenleri için üst sınır (alt 0)."""
    return float(max(5.0, float(np.max(matrix)) * 1.5 + 1e-6))


def _make_problem(matrix, K, metric: str = 'pearson'):
    n_items = matrix.shape[1]
    ub = _centroid_value_upper_bound(matrix)
    return {
        "obj_func"       : make_fitness_function(matrix, K, metric=metric),
        "bounds"         : FloatVar(
                               lb=[0.0] * (K * n_items),
                               ub=[ub] * (K * n_items),
                               name="centroids"
                           ),
        "minmax"         : "min",
        "log_to"         : None,
        "save_population": False,
    }


def _make_problem_de_hho(matrix, K):
    """
    DE_HHO için objective: Pearson-distance tabanlı clustering fitness.
    """
    n_items = matrix.shape[1]
    ub = _centroid_value_upper_bound(matrix)

    def _obj(solution):
        centroids = np.asarray(solution, dtype=np.float32).reshape((K, n_items))
        return calculate_clustering_fitness(centroids, matrix, K)

    return {
        "obj_func": _obj,
        "bounds": FloatVar(
            lb=[0.0] * (K * n_items),
            ub=[ub] * (K * n_items),
            name="centroids",
        ),
        "minmax": "min",
        "log_to": None,
        "save_population": False,
    }


def run_single(algo_info, matrix, K, init, epoch, pop_size, metric: str = 'pearson'):
    if algo_info['full_name'] == 'DE_HHO':
        problem = _make_problem_de_hho(matrix, K)
    else:
        problem = _make_problem(matrix, K, metric=metric)
    sp      = get_special_params(algo_info['full_name'], epoch, pop_size)
    model   = algo_info['class'](**(sp or {'epoch': epoch, 'pop_size': pop_size}))
    try:
        model.solve(problem, starting_solutions=init[:pop_size])
    except TypeError:
        model.solve(problem)
    best_fit = float(model.g_best.target.fitness)
    print(f"    {algo_info['full_name'].split('.')[0]} WCSS: {best_fit:.4f}")
    return model.g_best.solution, best_fit


def run_hybrid(g_info, l_info, matrix, K, init, g_epoch, l_epoch, pop_size,
               metric: str = 'pearson'):
    problem = _make_problem(matrix, K, metric=metric)

    sp_g    = get_special_params(g_info['full_name'], g_epoch, pop_size)
    g_model = g_info['class'](**(sp_g or {'epoch': g_epoch, 'pop_size': pop_size}))
    try:
        g_model.solve(problem, starting_solutions=init[:pop_size])
    except TypeError:
        g_model.solve(problem)
    best_g_sol = g_model.g_best.solution
    best_g_fit = float(g_model.g_best.target.fitness)
    print(f"    Global ({g_info['full_name'].split('.')[0]}) WCSS: {best_g_fit:.4f}")

    sp_l    = get_special_params(l_info['full_name'], l_epoch, pop_size)
    l_model = l_info['class'](**(sp_l or {'epoch': l_epoch, 'pop_size': pop_size}))
    rng = np.random.default_rng(seed=42)
    lb  = np.array(problem["bounds"].lb)
    ub  = np.array(problem["bounds"].ub)
    search_range = ub - lb           # [0, 5] → range=5
    noise_scale  = 0.02 * search_range   # çözüm uzayının %2'si

    local_start = []
    for i in range(pop_size):
        if i == 0:
            local_start.append(best_g_sol.copy())   # en iyi nokta korunsun
        else:
            noise = rng.normal(0, noise_scale)
            noisy = np.clip(best_g_sol + noise, lb, ub)
            local_start.append(noisy)
    try:
        l_model.solve(problem, starting_solutions=local_start)
    except TypeError:
        l_model.solve(problem)
    best_l_fit = float(l_model.g_best.target.fitness)
    print(f"    Local  ({l_info['full_name'].split('.')[0]}) WCSS: {best_l_fit:.4f}")

    if best_l_fit < best_g_fit:
        best_sol, best_fit, improved = l_model.g_best.solution, best_l_fit, True
    else:
        best_sol, best_fit, improved = best_g_sol, best_g_fit, False
    print(f"    Lokal iyilestirdi: {improved}  ->  Final WCSS: {best_fit:.4f}")
    return best_sol, best_fit


def run_parallel_hybrid(g_info, l_info, matrix, K, init, g_epoch, l_epoch, pop_size,
                        metric: str = 'pearson'):
    """Etikette '||' geçen hibrit: iki algoritmayı ayrı çalıştırıp en iyi WCSS'yi seçer."""
    sol_g, fit_g = run_single(g_info, matrix, K, init, g_epoch, pop_size, metric=metric)
    sol_l, fit_l = run_single(l_info, matrix, K, init, l_epoch, pop_size, metric=metric)
    if fit_l < fit_g:
        return sol_l, fit_l
    return sol_g, fit_g


def run_memetic_hybrid(base_info, matrix, K, init,
                       total_epoch, pop_size,
                       ga_inject_interval=10,
                       ga_crossover_rate=0.3,
                       ga_mutation_rate=0.1,
                       metric: str = 'pearson'):
    """
    Memetic hibrit: Sürü algoritması + GA operatör enjeksiyonu.

    Her ga_inject_interval epoch'ta:
    - Popülasyonun ga_crossover_rate kadarına crossover uygulanır
    - Popülasyonun ga_mutation_rate kadarına mutation uygulanır
    - Çeşitlilik korunur, erken yakınsama önlenir

    Literatür: "Hybrid Swarm-GA" / "Memetic Algorithm"

    base_info: ana sürü algoritması (HHO önerilen)
    """
    import numpy as np

    problem = _make_problem(matrix, K, metric=metric)
    lb = np.array(problem["bounds"].lb)
    ub = np.array(problem["bounds"].ub)
    rng = np.random.default_rng(seed=42)
    dim = len(lb)

    # Başlangıç popülasyonu
    current_pop = list(init[:pop_size])
    best_sol = current_pop[0].copy()
    best_fit = float('inf')

    # Fitness hesapla
    obj_fn = problem["obj_func"]
    fits = [float(obj_fn(s)) for s in current_pop]
    best_idx = int(np.argmin(fits))
    best_fit = fits[best_idx]
    best_sol = current_pop[best_idx].copy()

    print(f"    Memetic hibrit: {total_epoch} epoch, "
          f"GA enjeksiyonu her {ga_inject_interval} epoch")

    epochs_done = 0
    round_num = 0

    while epochs_done < total_epoch:
        # Bu round'da kaç epoch çalışacak
        ep = min(ga_inject_interval, total_epoch - epochs_done)

        # Sürü algoritmasını ep epoch çalıştır
        sp = get_special_params(base_info['full_name'], ep, pop_size)
        model = base_info['class'](
            **(sp or {'epoch': ep, 'pop_size': pop_size})
        )
        try:
            model.solve(problem, starting_solutions=current_pop)
        except TypeError:
            model.solve(problem)

        # En iyiyi güncelle
        new_fit = float(model.g_best.target.fitness)
        if new_fit < best_fit:
            best_fit = new_fit
            best_sol = model.g_best.solution.copy()

        # Mevcut popülasyonu al (mealpy: Optimizer.pop, Agent.solution)
        pop = getattr(model, 'pop', None)
        if not pop:
            raise RuntimeError(
                f"{base_info['full_name']}: solve sonrası popülasyon yok; "
                'memetic hibrit bu optimizasyon sınıfı ile uyumlu değil.'
            )
        current_pop = [np.asarray(agent.solution, dtype=np.float64).copy() for agent in pop]
        fits = [float(obj_fn(s)) for s in current_pop]

        epochs_done += ep
        round_num += 1

        # GA operatör enjeksiyonu
        if epochs_done < total_epoch:
            n_cross = max(1, int(pop_size * ga_crossover_rate))
            n_mut   = max(1, int(pop_size * ga_mutation_rate))

            # Arithmetic Crossover
            cross_idx = rng.choice(pop_size, size=n_cross * 2,
                                   replace=False)
            for i in range(0, len(cross_idx) - 1, 2):
                p1 = current_pop[cross_idx[i]]
                p2 = current_pop[cross_idx[i+1]]
                alpha = rng.uniform(0.3, 0.7)
                child = np.clip(alpha * p1 + (1-alpha) * p2, lb, ub)
                child_fit = float(obj_fn(child))
                # Daha kötü ebeveynin yerine koy
                worse_idx = cross_idx[i] \
                    if fits[cross_idx[i]] > fits[cross_idx[i+1]] \
                    else cross_idx[i+1]
                if child_fit < fits[worse_idx]:
                    current_pop[worse_idx] = child
                    fits[worse_idx] = child_fit

            # Gaussian Mutation
            mut_idx = rng.choice(pop_size, size=n_mut, replace=False)
            scale = 0.05 * (ub - lb)
            for idx in mut_idx:
                mutant = np.clip(
                    current_pop[idx] + rng.normal(0, scale), lb, ub
                )
                mutant_fit = float(obj_fn(mutant))
                if mutant_fit < fits[idx]:
                    current_pop[idx] = mutant
                    fits[idx] = mutant_fit

            # En iyiyi güncelle
            round_best_idx = int(np.argmin(fits))
            if fits[round_best_idx] < best_fit:
                best_fit = fits[round_best_idx]
                best_sol = current_pop[round_best_idx].copy()

            print(f"    Round {round_num}: swarm={new_fit:.4f} "
                  f"GA_inject → best={best_fit:.4f}")

    print(f"    Final WCSS: {best_fit:.4f}")
    return best_sol, best_fit


# ============================================================
# KAYDET
# ============================================================

def save_assignment(assignments, gray_mask, best_sol, best_fit,
                    save_dir, extra_data=None, label=None, K=None, args=None,
                    run_id=None, seed=None, memberships=None):
    """
    Assignment dosyalarını kaydet.
    extra_data: LOF modunda lof_scores gibi ek veriler dict olarak geçilir.
    """
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, 'assignments.npy'),     assignments)
    if memberships is not None:
        np.save(os.path.join(save_dir, 'memberships.npy'), memberships)
    np.save(os.path.join(save_dir, 'gray_sheep_mask.npy'), gray_mask)
    np.save(os.path.join(save_dir, 'best_sol.npy'),        best_sol)

    # CSV — LOF modunda lof_score sütunu da eklenir
    df_dict = {
        'user_idx'     : np.arange(len(assignments)),
        'cluster_id'   : assignments,
        'is_gray_sheep': gray_mask.astype(int),
    }
    if extra_data and 'lof_scores' in extra_data:
        np.save(os.path.join(save_dir, 'lof_scores.npy'), extra_data['lof_scores'])
        df_dict['lof_score'] = extra_data['lof_scores']

    pd.DataFrame(df_dict).to_csv(
        os.path.join(save_dir, 'assignment_summary.csv'), index=False
    )

    white_assign  = assignments[~gray_mask]
    K_val         = int(assignments.max()) + 1
    cluster_sizes = np.bincount(white_assign, minlength=K_val)
    active        = cluster_sizes[cluster_sizes > 0]

    threshold = extra_data.get('threshold', '—') if extra_data else '—'

    print(f"    OK Kaydedildi -> {save_dir}")
    print(f"      WCSS          : {best_fit:.4f}")
    print(f"      Kullanıcı     : {len(assignments)}")
    print(f"      Gray sheep    : {gray_mask.sum()} ({gray_mask.mean()*100:.1f}%)")
    print(f"      GS threshold  : {threshold:.4f}" if isinstance(threshold, float)
          else f"      GS threshold  : {threshold}")
    print(f"      Küme boyutu   : min={active.min()}, "
          f"max={active.max()}, ort={active.mean():.1f}")

    if _DB_AVAILABLE:
        # preprocessing label belirle
        prep_parts = []
        prep_parts.append(
            f"prune_u{getattr(args, 'min_user_ratings', 5)}_i{getattr(args, 'min_item_ratings', 10)}"
        )
        if getattr(args, 'zscore', False):
            prep_parts.append('zscore')
        if getattr(args, 'pca', None):
            if args.pca < 1.0:
                prep_parts.append(f'pca{int(args.pca*100)}pct')
            else:
                prep_parts.append(f'pca{int(args.pca)}')
        if getattr(args, 'wnmf_features', None):
            prep_parts.append(f'wnmf{args.wnmf_features}')
            prep_parts.append(
                f"{getattr(args, 'wnmf_init', 'inmed')}_trim"
                f"{getattr(args, 'inmed_trim_low', 5.0):g}_{getattr(args, 'inmed_trim_high', 95.0):g}"
            )
        preprocessing = '_'.join(prep_parts) if prep_parts else 'none'

        # dataset adını belirle
        ds = 'ml100k' if 'ml100k' in save_dir else 'ml1m'

        lof_scores_arr = extra_data.get('lof_scores') if extra_data else None

        db_save(
            dataset=ds,
            algo=label,
            k=K,
            preprocessing=preprocessing,
            wcss=float(best_fit),
            gray_count=int(gray_mask.sum()),
            gray_ratio=float(gray_mask.mean()),
            lof_threshold=float(extra_data.get('threshold', 0))
                          if extra_data else 0.0,
            n_users=len(assignments),
            cluster_min=int(active.min()),
            cluster_max=int(active.max()),
            cluster_avg=float(active.mean()),
            # best_sol_arr YOK
            assignments_arr=assignments,
            gray_mask_arr=gray_mask,
            lof_scores_arr=lof_scores_arr,
            run_id=run_id,
            seed=seed,
        )


# ============================================================
# TEK ALGORİTMA ÇALIŞTIR
# ============================================================

def _run_one_core(
    label,
    g_name,
    l_name,
    matrix,
    K,
    seed,
    save_dir,
    algo_map,
    use_lof,
    lof_n_neighbors,
    lof_contamination,
    baseline_epoch,
    global_epoch,
    local_epoch,
    pop_size,
    args=None,
    run_id=None,
    cluster_metric: str = 'pearson',
):
    """Ortak gövde: algo_map ana süreçte veya worker’da bir kez oluşturulur."""
    mode_str = 'LOF' if use_lof else 'percentile'
    print(f"\n  [{label}] başlıyor (gray sheep: {mode_str}, küme metrik: {cluster_metric})...")
    t0   = time.time()
    #init = mkmeans_plus_plus_init(matrix, K=K, n_solutions=50, seed=seed)
    init = _multi_start_init(
        matrix, K=K, pop_size=POP_SIZE, seed=seed, n_restarts=10,
        metric=cluster_metric,
    )

    if g_name == 'KMEANS':
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize

        matrix_norm = normalize(matrix, norm='l2')
        kmeans = KMeans(
            n_clusters=K,
            init='k-means++',
            n_init=10,
            random_state=seed,
            max_iter=300,
            verbose=0
        )
        kmeans.fit(matrix_norm)
        assignments_km = kmeans.labels_

        # best_sol: centroidleri orijinal rating uzayına döndür
        # centroids shape: (K, n_items)
        best_sol = kmeans.cluster_centers_.flatten()
        best_fit = float(kmeans.inertia_)
        assignments = assignments_km
    elif l_name == 'GAop':
        best_sol, best_fit = run_memetic_hybrid(
            algo_map[g_name], matrix, K, init,
            total_epoch=global_epoch + local_epoch,
            pop_size=pop_size,
            ga_inject_interval=10,
            ga_crossover_rate=0.3,
            ga_mutation_rate=0.1,
            metric=cluster_metric,
        )
    elif l_name is None:
        best_sol, best_fit = run_single(
            algo_map[g_name], matrix, K, init, baseline_epoch, pop_size,
            metric=cluster_metric,
        )
    elif '||' in label:
        best_sol, best_fit = run_parallel_hybrid(
            algo_map[g_name], algo_map[l_name],
            matrix, K, init, global_epoch, local_epoch, pop_size,
            metric=cluster_metric,
        )
    else:
        best_sol, best_fit = run_hybrid(
            algo_map[g_name], algo_map[l_name],
            matrix, K, init, global_epoch, local_epoch, pop_size,
            metric=cluster_metric,
        )

    memberships = None
    if g_name != 'KMEANS':
        _, assignments = compute_wcss_fast(
            matrix, best_sol, K, metric=cluster_metric,
        )
        if cluster_metric == 'fuzzy':
            _, assignments, memberships = compute_fcm_objective(
                matrix, best_sol, K, m=2.0,
            )
    elif cluster_metric == 'fuzzy':
        _, assignments, memberships = compute_fcm_objective(
            matrix, best_sol, K, m=2.0,
        )

    if use_lof:
        print(f"    LOF hesaplanıyor (n_neighbors={lof_n_neighbors})...")
        gs_info   = detect_gray_sheep_lof(
            matrix, assignments, lof_n_neighbors, lof_contamination
        )
        gray_mask = gs_info['gray_sheep_mask']
        extra_data = {
            'lof_scores': gs_info['lof_scores'],
            'threshold' : gs_info['threshold'],
        }
    else:
        gs_info    = detect_gray_sheep_percentile(
            matrix, assignments, best_sol, K, metric=cluster_metric,
        )
        gray_mask  = gs_info['gray_sheep_mask']
        extra_data = {'threshold': gs_info['threshold']}

    # DB'de WCSS kolonu her zaman gerçek kümeleme hedefini taşısın;
    # optimizer'ın iç objective'i (kompozit vb.) ile karışmasın.
    best_wcss, _ = compute_wcss_fast(
        matrix, best_sol, K, metric=cluster_metric,
    )

    save_assignment(
        assignments, gray_mask, best_sol, best_wcss,
        save_dir, extra_data, label=label, K=K, args=args,
        run_id=run_id, seed=seed, memberships=memberships,
    )
    print(f"  [{label}] tamamlandı — {time.time()-t0:.1f}s")


def _mp_run_assignment_job(job):
    """
    Windows spawn için modül düzeyinde worker — mealpy sınıflarını pickle etmemek
    için algo_map alt-süreçte get_all_algorithms_v3() ile kurulur.
    """
    (
        label,
        g_name,
        l_name,
        matrix,
        K,
        seed,
        save_dir,
        use_lof,
        lof_n_neighbors,
        lof_contamination,
        baseline_epoch,
        global_epoch,
        local_epoch,
        pop_size,
        args,
        run_id,
        cluster_metric,
    ) = job
    algo_map = {a['full_name']: a for a in get_all_algorithms_v3()}
    algo_map['GAHHO.OriginalGAHHO'] = {
        'full_name': 'GAHHO.OriginalGAHHO',
        'class':     OriginalGAHHO,
    }
    algo_map['DE_HHO'] = {
        'full_name': 'DE_HHO',
        'class':     DE_HHO,
    }
    algo_map['GA.EliteMultiGA'] = {
        'full_name': 'GA.EliteMultiGA',
        'class':     GA.EliteMultiGA,
    }
    algo_map['KMEANS'] = {
        'full_name': 'KMEANS',
        'class': None,
    }
    _run_one_core(
        label,
        g_name,
        l_name,
        matrix,
        K,
        seed,
        save_dir,
        algo_map,
        use_lof,
        lof_n_neighbors,
        lof_contamination,
        baseline_epoch,
        global_epoch,
        local_epoch,
        pop_size,
        args,
        run_id,
        cluster_metric,
    )


def run_one(label, g_name, l_name, matrix, K, seed, save_dir, algo_map,
            use_lof=False, lof_n_neighbors=LOF_N_NEIGHBORS,
            lof_contamination=LOF_CONTAMINATION, args=None, run_id=None,
            cluster_metric: str = 'pearson'):
    """
    Bir algoritma için assignment üret ve kaydet.

    use_lof=True  → LOF tabanlı gray sheep (adaptif)
    use_lof=False → Sabit 80. percentile (~%20)
    """
    _run_one_core(
        label,
        g_name,
        l_name,
        matrix,
        K,
        seed,
        save_dir,
        algo_map,
        use_lof,
        lof_n_neighbors,
        lof_contamination,
        BASELINE_EPOCH,
        GLOBAL_EPOCH,
        LOCAL_EPOCH,
        POP_SIZE,
        args,
        run_id=run_id,
        cluster_metric=cluster_metric,
    )


# ============================================================
# ML-1M VERİ YÜKLEME
# ============================================================

def load_movielens_1m(path):
    rows = []
    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('::')
            if len(parts) >= 3:
                rows.append((int(parts[0]), int(parts[1]), float(parts[2])))
    df = pd.DataFrame(rows, columns=['user_id', 'item_id', 'rating'])
    matrix = df.pivot_table(
        index='user_id', columns='item_id',
        values='rating', fill_value=0
    ).values.astype(np.float32)
    total   = matrix.size
    nonzero = np.count_nonzero(matrix)
    print(f"Matrix shape : {matrix.shape}")
    print(f"Sparsity     : {1 - nonzero/total:.3f}")
    return matrix


def prune_sparse_matrix(matrix, min_user_ratings=5, min_item_ratings=10):
    """
    İteratif seyreklik budama:
    - min_user_ratings'ten az oylayan kullanıcıları kaldır
    - min_item_ratings'ten az oy alan filmleri kaldır
    Koşullar sabitlenene kadar döngü sürer.
    """
    if min_user_ratings <= 0 and min_item_ratings <= 0:
        return matrix

    pruned = matrix
    prev_shape = None
    iteration = 0

    while prev_shape != pruned.shape:
        prev_shape = pruned.shape
        iteration += 1

        if min_user_ratings > 0:
            user_counts = np.count_nonzero(pruned, axis=1)
            keep_users = user_counts >= int(min_user_ratings)
            pruned = pruned[keep_users]

        if min_item_ratings > 0:
            item_counts = np.count_nonzero(pruned, axis=0)
            keep_items = item_counts >= int(min_item_ratings)
            pruned = pruned[:, keep_items]

    total = pruned.size
    nonzero = np.count_nonzero(pruned)
    sparsity = 1 - (nonzero / total if total > 0 else 0.0)
    print(
        f"  Prune tamamlandı ({iteration} iter): "
        f"shape={pruned.shape}, sparsity={sparsity:.3f}"
    )
    return pruned.astype(np.float32, copy=False)


def zscore_normalize(matrix):
    """
    Kullanıcı bazlı Z-score normalizasyon.
    Her kullanıcının rating'lerini normalize et:
    r_norm = (r - user_mean) / user_std

    Rating olmayan (0) hücreler normalize edilmez.
    std=0 olan kullanıcılar için sadece mean çıkar.
    """
    normalized = matrix.copy().astype(np.float32)
    for u in range(matrix.shape[0]):
        rated = matrix[u] != 0
        if rated.sum() == 0:
            continue
        mean = matrix[u, rated].mean()
        std  = matrix[u, rated].std()
        if std > 0:
            normalized[u, rated] = (matrix[u, rated] - mean) / std
        else:
            normalized[u, rated] = matrix[u, rated] - mean
    return normalized


def wnmf_feature_extract(matrix, n_components,
                         n_epochs=50, lr=0.01,
                         reg=0.01, random_seed=42,
                         init_method='inmed',
                         inmed_trim=(5.0, 95.0)):
    """
    Ham rating matrisini WNMF ile ayrıştır.
    Sadece kullanıcı latent matrisini (U) döndür.

    Uygulama: [wnmf/wnmf_model.py](wnmf_model.WNMFModel) — wnmf_experiment ile aynı
    çekirdek (tekrarlanabilirlik).

    Neden WNMF: Sıfır hücreler 'rating yok' anlamına gelir,
    standart NMF bunları sıfır rating olarak işler (hatalı).
    WNMF sadece gözlemlenen rating'leri kullanır.

    Çıktı: (n_users × n_components) dense, nonnegative U.
    """
    wnmf_dir = os.path.join(_REPO_ROOT, 'wnmf')
    if wnmf_dir not in sys.path:
        sys.path.insert(0, wnmf_dir)
    from wnmf_model import WNMFModel

    n_users, n_items = matrix.shape
    rows, cols = np.where(matrix > 0)
    if len(rows) == 0:
        raise ValueError('wnmf_feature_extract: hiç gözlemlenen rating yok')

    train_ratings = np.column_stack(
        [rows.astype(np.float32), cols.astype(np.float32), matrix[rows, cols].astype(np.float32)]
    )

    print(f"  WNMF feature extraction (WNMFModel)...")
    print(f"  Matris: {n_users}x{n_items} -> U: {n_users}x{n_components}")
    print(f"  Gözlemlenen rating: {len(rows)}, epoch: {n_epochs}")

    model = WNMFModel(
        n_users        = n_users,
        n_items        = n_items,
        latent_dim     = n_components,
        learning_rate  = lr,
        regularization = reg,
        n_epochs       = n_epochs,
        random_seed    = random_seed,
        use_bias       = True,
        use_svdpp      = False,
        init_method    = init_method,
        inmed_trim     = inmed_trim,
    )
    model.fit(train_ratings, verbose=False)
    print(f"  WNMF tamamlandı. U shape: {model.U.shape}")
    return model.U.astype(np.float32)


def pca_variance_reduce(matrix, variance_ratio, random_state=42):
    """
    sklearn PCA: n_components in (0,1) → birikimli açıklanan varyans ≥ oran olacak kadar bileşen.
    """
    from sklearn.decomposition import PCA

    n_samples, n_features = matrix.shape
    pca = PCA(n_components=variance_ratio, random_state=random_state)
    out = pca.fit_transform(matrix)
    cum = float(np.sum(pca.explained_variance_ratio_))
    print(
        f"  PCA: {n_features} -> {out.shape[1]} components "
        f"(target >={variance_ratio:.0%} variance, cumulative: {cum:.4f})"
    )
    return out.astype(np.float32)


def prepare_matrix_for_clustering(
    matrix,
    zscore,
    pca_var,
    wnmf_k,
    min_user_ratings=5,
    min_item_ratings=10,
    wnmf_init_method='inmed',
    inmed_trim=(5.0, 95.0),
):
    """Sıra: prune → z-score → PCA → WNMF (--pca ile --wnmf-features birlikte CLI’de yasak)."""
    from sklearn.preprocessing import normalize

    matrix = prune_sparse_matrix(
        matrix,
        min_user_ratings=min_user_ratings,
        min_item_ratings=min_item_ratings,
    )
    if zscore:
        matrix = zscore_normalize(matrix)
        print("  Z-score normalizasyon uygulandı")
        # Pearson benzerliği ile K-Means (L2/Euclidean) geometrisini hizala.
        matrix = normalize(matrix, norm='l2', axis=1).astype(np.float32, copy=False)
        print("  L2 normalization uygulandı (axis=1)")
    if pca_var is not None:
        matrix = pca_variance_reduce(matrix, pca_var)
    if wnmf_k is not None:
        matrix = wnmf_feature_extract(
            matrix,
            n_components=wnmf_k,
            n_epochs=50,
            random_seed=42,
            init_method=wnmf_init_method,
            inmed_trim=inmed_trim,
        )
    return matrix


# ============================================================
# DATASET ÇALIŞTIRICI
# ============================================================

def run_dataset(dataset_name, matrix, K, out_root, algo_filter=None,
                use_lof=False, lof_n_neighbors=LOF_N_NEIGHBORS,
                lof_contamination=LOF_CONTAMINATION,
                max_workers: Optional[int] = None, out_suffix: str = '',
                args=None, run_id=None, cluster_metric: str = 'pearson'):
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name.upper()}  |  K={K}  |  Seed={SEED}")
    mode_str = f'LOF (n={lof_n_neighbors}, cont={lof_contamination})' \
               if use_lof else 'percentile (80th)'
    print(f"Gray sheep  : {mode_str}")
    print(f"Küme metrik : {cluster_metric} (pearson | euclidean)")
    print(f"{'='*60}")

    # K=default için suffix yok, farklı K için _k{K} eklenir
    default_k = K_100K_DEFAULT if dataset_name == 'ml100k' else K_1M_DEFAULT
    k_suffix  = '' if K == default_k else f'_k{K}'

    jobs_meta = []
    for label, g_name, l_name in ALGO_CONFIG:
        if algo_filter and label not in algo_filter:
            print(f"  [{label}] atlandı (filtre)")
            continue
        save_dir = os.path.join(out_root, dataset_name, f"{label}{k_suffix}{out_suffix}")
        jobs_meta.append((label, g_name, l_name, save_dir))

    if not jobs_meta:
        print(f"\n{dataset_name.upper()} — çalıştırılacak algoritma yok.")
        return

    nw = _resolve_pool_workers(max_workers, len(jobs_meta))

    if nw == 1:
        print("Algoritma kataloğu yükleniyor (1 kez)...")
        algo_map = {a['full_name']: a for a in get_all_algorithms_v3()}
        algo_map['GAHHO.OriginalGAHHO'] = {
            'full_name': 'GAHHO.OriginalGAHHO',
            'class':     OriginalGAHHO,
        }
        algo_map['DE_HHO'] = {
            'full_name': 'DE_HHO',
            'class':     DE_HHO,
        }
        algo_map['GA.EliteMultiGA'] = {
            'full_name': 'GA.EliteMultiGA',
            'class':     GA.EliteMultiGA,
        }
        algo_map['KMEANS'] = {
            'full_name': 'KMEANS',
            'class': None,
        }
        for label, g_name, l_name, save_dir in jobs_meta:
            run_one(
                label,
                g_name,
                l_name,
                matrix,
                K,
                SEED,
                save_dir,
                algo_map,
                use_lof=use_lof,
                lof_n_neighbors=lof_n_neighbors,
                lof_contamination=lof_contamination,
                args=args,
                run_id=run_id,
                cluster_metric=cluster_metric,
            )
    else:
        print(
            f"Paralel atama: {len(jobs_meta)} iş, en fazla {nw} süreç "
            "(CPU-yoğun optimizasyon için süreç havuzu)."
        )
        jobs = [
            (
                label,
                g_name,
                l_name,
                matrix,
                K,
                SEED,
                save_dir,
                use_lof,
                lof_n_neighbors,
                lof_contamination,
                BASELINE_EPOCH,
                GLOBAL_EPOCH,
                LOCAL_EPOCH,
                POP_SIZE,
                args,
                run_id,
                cluster_metric,
            )
            for label, g_name, l_name, save_dir in jobs_meta
        ]
        with ProcessPoolExecutor(max_workers=nw) as pool:
            list(pool.map(_mp_run_assignment_job, jobs))

    print(f"\n{dataset_name.upper()} tamamlandi -> {os.path.join(out_root, dataset_name)}/")


# ============================================================
# CLI
# ============================================================

def parse_args():
    labels = ALGO_LABELS
    p = argparse.ArgumentParser(
        description="Assignment üretici — percentile veya LOF gray sheep"
    )
    p.add_argument(
        '--dataset', choices=['100k', '1m', 'both'], default='both',
        help="Hangi dataset (default: both)"
    )
    p.add_argument(
        '--algo', nargs='+', choices=labels, default=None,
        metavar='LABEL',
        help=f"Algoritmalar (default: hepsi): {labels}"
    )
    p.add_argument(
        '--last-only', action='store_true',
        help=f"Sadece son algoritmayı çalıştır: {labels[-1]}"
    )
    p.add_argument(
        '--lof', action='store_true',
        help="LOF tabanlı gray sheep kullan (default: sabit 80. percentile)"
    )
    p.add_argument(
        '--zscore', action='store_true',
        help='Clustering matrisine kullanıcı bazlı Z-score normalizasyon uygula'
    )
    p.add_argument(
        '--min-user-ratings', type=int, default=5, metavar='N',
        help='Veri budama: kullanıcı başına minimum rating sayısı (default: 5)',
    )
    p.add_argument(
        '--min-item-ratings', type=int, default=10, metavar='N',
        help='Veri budama: film başına minimum rating sayısı (default: 10)',
    )
    p.add_argument(
        '--wnmf-features', type=int, default=None,
        metavar='K',
        help='WNMF ile K boyutlu kullanıcı latent matris çıkar. '
             'Bu matris metasezgisel clustering girişi olur. '
             'Öneri için ham rating kullanılır. (örn: --wnmf-features 20)'
    )
    p.add_argument(
        '--wnmf-init', choices=['random', 'inmed'], default='inmed',
        help='--wnmf-features için WNMF başlangıcı: random veya inmed (default: inmed)',
    )
    p.add_argument(
        '--inmed-trim-low', type=float, default=5.0, metavar='P',
        help='INMED trimmed mean alt yüzdelik (default: 5.0)',
    )
    p.add_argument(
        '--inmed-trim-high', type=float, default=95.0, metavar='P',
        help='INMED trimmed mean üst yüzdelik (default: 95.0)',
    )
    p.add_argument(
        '--cluster-metric', choices=['auto', 'pearson', 'euclidean', 'fuzzy'],
        default='auto',
        help='Sürü kümeleme fitness: auto = WNMF kullanıldıysa euclidean, '
             'aksi halde pearson (seyrek rating uzayı). fuzzy = FCM (m=2.0).',
    )
    p.add_argument(
        '--save-wnmf-u', type=str, default=None, metavar='DIR',
        help='--wnmf-features ile: U matrisini DIR içine ml100k_U.npy / ml1m_U.npy olarak kaydet.',
    )
    p.add_argument(
        '--pca', type=float, default=None, metavar='VAR',
        dest='pca_variance',
        help='PCA: birikimli açıklanan varyans eşiği (0–1, örn: 0.80). '
             '--wnmf-features ile birlikte kullanılamaz.',
    )
    p.add_argument(
        '--k', nargs='+', type=int, default=None, metavar='K',
        help="Küme sayısı; birden fazla değer: --k 20 30 50 (her K için ayrı klasör). "
             "Tek değer: --k 70. Verilmezse 100K=90, 1M=150. "
             "--k-100k / --k-1m ile birlikte kullanılamaz."
    )
    p.add_argument(
        '--k-100k', type=int, default=None,
        help=f"ML-100K için K (default: {K_100K_DEFAULT})"
    )
    p.add_argument(
        '--k-1m', type=int, default=None,
        help=f"ML-1M için K (default: {K_1M_DEFAULT})"
    )
    p.add_argument(
        '--n-neighbors', type=int, default=LOF_N_NEIGHBORS,
        help=f"LOF komşu sayısı (default: {LOF_N_NEIGHBORS}, sadece --lof ile)"
    )
    p.add_argument(
        '--contamination', default=str(LOF_CONTAMINATION),
        help=f"LOF contamination: 'auto' veya float (default: {LOF_CONTAMINATION})"
    )
    p.add_argument('--data-100k', default=DATA_100K)
    p.add_argument('--data-1m',   default=DATA_1M)
    p.add_argument(
        '--jobs', type=int, default=None,
        help='Paralel algoritma süreç sayısı (varsayılan: otomatik = CPU sayısına göre; '
             '1=sıralı, 2+=belirtilen sayıda süreç, 0=CPU ile sınırlandırılmış havuz)',
    )
    args = p.parse_args()
    if args.k is not None and (args.k_100k is not None or args.k_1m is not None):
        p.error('--k (tek veya çoklu) ile --k-100k / --k-1m birlikte kullanılamaz')
    if args.pca_variance is not None:
        if not (0 < args.pca_variance <= 1):
            p.error('--pca (0, 1] aralığında olmalı (örn. 0.80)')
    if args.pca_variance is not None and args.wnmf_features is not None:
        p.error('--pca ile --wnmf-features birlikte kullanılamaz')
    if args.min_user_ratings < 0:
        p.error('--min-user-ratings 0 veya daha büyük olmalı')
    if args.min_item_ratings < 0:
        p.error('--min-item-ratings 0 veya daha büyük olmalı')
    if not (0 <= args.inmed_trim_low < args.inmed_trim_high <= 100):
        p.error('--inmed-trim-low ve --inmed-trim-high için 0 <= low < high <= 100 olmalı')
    args.pca = args.pca_variance
    if args.cluster_metric == 'auto':
        args.cluster_metric = 'euclidean' if args.wnmf_features else 'pearson'
    return args

def _multi_start_init(matrix, K, pop_size, seed, n_restarts=10,
                      metric: str = 'pearson'):
    """
    n_restarts farklı seed ile MkMeans++ çalıştır.
    Her seferinde 1 çözüm üret, WCSS hesapla.
    En iyi pop_size kadar çözümü döndür.
    """
    candidates = []
    for i in range(n_restarts * 3):   # fazla üret, en iyileri seç
        sols = mkmeans_plus_plus_init(
            matrix, K=K, n_solutions=1, seed=seed + i * 17, metric=metric,
        )
        sol = sols[0]
        wcss, _ = compute_wcss_fast(matrix, sol, K, metric=metric)
        candidates.append((wcss, sol))

    # WCSS'e göre sırala, en iyi pop_size tanesini al
    candidates.sort(key=lambda x: x[0])
    best = [c[1] for c in candidates[:pop_size + 10]]
    print(f"    Init: {n_restarts*3} aday -> en iyi {pop_size} secildi  "
          f"(WCSS aralığı: {candidates[0][0]:.1f} – {candidates[pop_size-1][0]:.1f})")
    return best
# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    args = parse_args()

    db_run_id = None
    if _DB_AVAILABLE and start_run is not None:
        try:
            prep_parts = []
            prep_parts.append(f"prune_u{args.min_user_ratings}_i{args.min_item_ratings}")
            if getattr(args, 'zscore', False):
                prep_parts.append('zscore')
            if getattr(args, 'pca', None):
                if args.pca < 1.0:
                    prep_parts.append(f'pca{int(args.pca * 100)}pct')
                else:
                    prep_parts.append(f'pca{int(args.pca)}')
            if getattr(args, 'wnmf_features', None):
                prep_parts.append(f'wnmf{args.wnmf_features}')
                prep_parts.append(
                    f"{args.wnmf_init}_trim{args.inmed_trim_low:g}_{args.inmed_trim_high:g}"
                )
            preprocessing_run = '_'.join(prep_parts) if prep_parts else 'none'
            db_run_id = start_run(
                command=' '.join(sys.argv),
                dataset=args.dataset,
                k=None,
                preprocessing=preprocessing_run,
                note='gray_sheep=LOF' if args.lof else 'gray_sheep=percentile',
            )
            print(f"  DB run_id: {db_run_id}")
        except Exception as exc:
            print(f"  Uyari: start_run basarisiz ({exc})", file=sys.stderr)

    selected_algos = [ALGO_CONFIG[-1][0]] if args.last_only else args.algo

    # Contamination dönüşümü
    contamination = args.contamination
    if contamination != 'auto':
        contamination = float(contamination)

    # K değerleri: --k 20 30 50 → her dataset için bu liste; yoksa dataset başına tek K
    k_multi = list(args.k) if args.k is not None else None
    if k_multi is None:
        k_100k = args.k_100k or K_100K_DEFAULT
        k_1m   = args.k_1m   or K_1M_DEFAULT

    # Çıktı klasörü — LOF ve percentile ayrı tutulur
    out_root = os.path.join(
        BASE_DIR, 'results',
        'assignments_lof' if args.lof else 'assignments'
    )
    os.makedirs(out_root, exist_ok=True)

    print("=" * 60)
    print("ASSIGNMENT ÜRETİCİ")
    print("=" * 60)
    print(f"Gray sheep  : {'LOF (adaptif)' if args.lof else 'Percentile (sabit %20)'}")
    print(f"Algoritmalar: {selected_algos or [c[0] for c in ALGO_CONFIG]}")
    print(f"Dataset     : {args.dataset}")
    if k_multi is not None:
        print(f"K listesi     : {k_multi}  (her değer sırayla işlenir)")
    else:
        print(f"ML-100K K   : {k_100k}  |  ML-1M K: {k_1m}")
    if args.lof:
        print(f"LOF         : n_neighbors={args.n_neighbors}, contamination={contamination}")
    print(f"Epoch       : baseline={BASELINE_EPOCH}, global={GLOBAL_EPOCH}, local={LOCAL_EPOCH}")
    print(f"Pop size    : {POP_SIZE}  |  Seed: {SEED}")
    feat_bits = []
    feat_bits.append(f"Prune(u>={args.min_user_ratings}, i>={args.min_item_ratings})")
    if args.zscore:
        feat_bits.append('Z-score')
    if args.pca_variance is not None:
        feat_bits.append(f"PCA >={args.pca_variance:.0%} var.")
    if args.wnmf_features is not None:
        feat_bits.append(
            f"WNMF k={args.wnmf_features} init={args.wnmf_init} trim=({args.inmed_trim_low:g},{args.inmed_trim_high:g})"
        )
    if feat_bits:
        print(f"Matris      : ham + " + ' + '.join(feat_bits))
    print(f"Küme metrik : {args.cluster_metric}")
    print(f"Çıktı kökü  : {out_root}")
    if args.jobs is None:
        print("Paralellik    : otomatik (CPU sayısına göre; --jobs 1 ile sıralı mod)")
    elif args.jobs == 1:
        print("Paralellik    : sıralı (tek süreç)")
    elif args.jobs <= 0:
        print("Paralellik    : süreç havuzu (iş ve CPU sayısına göre)")
    else:
        print(f"Paralellik    : en fazla {args.jobs} süreç")
    print("=" * 60)

    t_total = time.time()

    def _finish_db_run():
        if db_run_id is not None and _DB_AVAILABLE and finish_run is not None:
            try:
                finish_run(db_run_id)
            except Exception as exc:
                print(f"  Uyari: finish_run basarisiz ({exc})", file=sys.stderr)

    def _pool_cap():
        if args.jobs == 1:
            return 1
        if args.jobs is None or args.jobs <= 0:
            return None  # otomatik: _resolve_pool_workers CPU sayısını kullanır
        return args.jobs

    prune_suffix = f"_pruneu{args.min_user_ratings}_i{args.min_item_ratings}"
    zscore_suffix = '_zscore' if args.zscore else ''
    pca_suffix = (
        f'_pca{int(round(args.pca_variance * 100))}pct'
        if args.pca_variance is not None else ''
    )
    wnmf_suffix = (
        f'_wnmf{args.wnmf_features}_{args.wnmf_init}_trim{args.inmed_trim_low:g}_{args.inmed_trim_high:g}'
        if args.wnmf_features is not None else ''
    )
    metric_suffix_map = {
        'euclidean': '_euc',
        'fuzzy': '_fuzzy',
    }
    metric_suffix = metric_suffix_map.get(args.cluster_metric, '')
    out_suffix = prune_suffix + zscore_suffix + pca_suffix + wnmf_suffix + metric_suffix

    try:
        if args.dataset in ('100k', 'both'):
            print(f"\nML-100K yükleniyor: {args.data_100k}")
            matrix_100k = load_movielens(args.data_100k)
            matrix_100k = prepare_matrix_for_clustering(
                matrix_100k,
                args.zscore,
                args.pca_variance,
                args.wnmf_features,
                min_user_ratings=args.min_user_ratings,
                min_item_ratings=args.min_item_ratings,
                wnmf_init_method=args.wnmf_init,
                inmed_trim=(args.inmed_trim_low, args.inmed_trim_high),
            )
            if args.save_wnmf_u and args.wnmf_features:
                os.makedirs(args.save_wnmf_u, exist_ok=True)
                u_path = os.path.join(args.save_wnmf_u, 'ml100k_U.npy')
                np.save(u_path, matrix_100k)
                print(f"  WNMF U kaydedildi: {u_path}")
            if k_multi is not None:
                for K in k_multi:
                    run_dataset(
                        'ml100k', matrix_100k, K, out_root,
                        algo_filter=selected_algos,
                        use_lof=args.lof,
                        lof_n_neighbors=args.n_neighbors,
                        lof_contamination=contamination,
                        max_workers=_pool_cap(),
                        out_suffix=out_suffix,
                        args=args,
                        run_id=db_run_id,
                        cluster_metric=args.cluster_metric,
                    )
            else:
                run_dataset(
                    'ml100k', matrix_100k, k_100k, out_root,
                    algo_filter=selected_algos,
                    use_lof=args.lof,
                    lof_n_neighbors=args.n_neighbors,
                    lof_contamination=contamination,
                    max_workers=_pool_cap(),
                    out_suffix=out_suffix,
                    args=args,
                    run_id=db_run_id,
                    cluster_metric=args.cluster_metric,
                )

        if args.dataset in ('1m', 'both'):
            print(f"\nML-1M yükleniyor: {args.data_1m}")
            matrix_1m = load_movielens_1m(args.data_1m)
            matrix_1m = prepare_matrix_for_clustering(
                matrix_1m,
                args.zscore,
                args.pca_variance,
                args.wnmf_features,
                min_user_ratings=args.min_user_ratings,
                min_item_ratings=args.min_item_ratings,
                wnmf_init_method=args.wnmf_init,
                inmed_trim=(args.inmed_trim_low, args.inmed_trim_high),
            )
            if args.save_wnmf_u and args.wnmf_features:
                os.makedirs(args.save_wnmf_u, exist_ok=True)
                u_path = os.path.join(args.save_wnmf_u, 'ml1m_U.npy')
                np.save(u_path, matrix_1m)
                print(f"  WNMF U kaydedildi: {u_path}")
            if k_multi is not None:
                for K in k_multi:
                    run_dataset(
                        'ml1m', matrix_1m, K, out_root,
                        algo_filter=selected_algos,
                        use_lof=args.lof,
                        lof_n_neighbors=args.n_neighbors,
                        lof_contamination=contamination,
                        max_workers=_pool_cap(),
                        out_suffix=out_suffix,
                        args=args,
                        run_id=db_run_id,
                        cluster_metric=args.cluster_metric,
                    )
            else:
                run_dataset(
                    'ml1m', matrix_1m, k_1m, out_root,
                    algo_filter=selected_algos,
                    use_lof=args.lof,
                    lof_n_neighbors=args.n_neighbors,
                    lof_contamination=contamination,
                    max_workers=_pool_cap(),
                    out_suffix=out_suffix,
                    args=args,
                    run_id=db_run_id,
                    cluster_metric=args.cluster_metric,
                )

        print(f"\n{'='*60}")
        print(f"TAMAMLANDI — toplam {(time.time()-t_total)/60:.1f} dakika")
        print(f"Çıktı: {out_root}/")
        print("=" * 60)
    finally:
        _finish_db_run()