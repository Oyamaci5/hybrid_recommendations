"""
Mealpy Algoritma Karşılaştırma Sistemi
MovieLens 100K - Gray Sheep Tespiti + Kümeleme Kalitesi
Mealpy 3.x uyumlu

Düzeltmeler:
  v2: _solve_worker_v3'e silhouette eklendi
      PHASE3_MUST'a OOA eklendi (ENCR global bileşeni)
      RUN_INFO.txt'e tarih + config bilgisi eklendi
  v3: MUST_ALL_STAGES — SFOA.OriginalSFOA tüm aşamalarda zorunlu (2–3. seçimde elenmez)
  v4: 3 aşama → 2 aşama (eski Aşama 1 kaldırıldı); eleme=run_phase(2), final=run_phase(3);
      K1,K2=60,90; kayıt yine results/phase3/<run_no>/
  v5: Eleme time_limit 180s, init n_solutions=30; final havuz = eleme top-10 (PHASE3_MUST yok);
      davranış analizi K=60; davranışta tüm Aşama 3 katılımcıları (top-10 kohort)
  v6: Silhouette: beyaz koyun alt kümesinde max 300 örnek, Pearson precomputed mesafe matrisi
  v7: Ana koşu: run_phase için parallel_workers=4, time_limit=None (paralel havuz)
  v8: run_phase n_runs + seed 0..n-1 ortalama; Aşama 2 kohort WCSS10∪DB5; davranış/hibrit tam ML-100K
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import time
import warnings
import os
import multiprocessing as mp

warnings.filterwarnings('ignore')

# ============================================================
# KLASÖR AYARI
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def next_phase3_run_dir(base_results_dir=None):
    """
    Her çalıştırmayı üst üste yazmadan saklamak için:
    results/phase3/1, results/phase3/2, ... sıradaki klasörü oluşturur.
    """
    base = base_results_dir if base_results_dir is not None else RESULTS_DIR
    phase3_root = os.path.join(base, 'phase3')
    os.makedirs(phase3_root, exist_ok=True)
    max_n = 0
    for name in os.listdir(phase3_root):
        path = os.path.join(phase3_root, name)
        if os.path.isdir(path) and name.isdigit():
            try:
                max_n = max(max_n, int(name))
            except ValueError:
                pass
    next_n = max_n + 1
    run_dir = os.path.join(phase3_root, str(next_n))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# ============================================================
# BÖLÜM 1: VERİ YÜKLEME
# ============================================================

def load_movielens(path):
    df = pd.read_csv(
        path, sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    matrix = df.pivot_table(
        index='user_id',
        columns='item_id',
        values='rating',
        fill_value=0
    ).values.astype(np.float32)

    total   = matrix.size
    nonzero = np.count_nonzero(matrix)

    print(f"Matrix shape     : {matrix.shape}")
    print(f"Sparsity         : {1 - nonzero/total:.3f}")
    print(f"Rating range     : {matrix[matrix>0].min():.1f} - {matrix.max():.1f}")
    return matrix


def load_movielens_1m(path):
    df = pd.read_csv(
        path, sep='::',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        engine='python',
        encoding='latin-1',
    )
    matrix = df.pivot_table(
        index='user_id',
        columns='item_id',
        values='rating',
        fill_value=0,
    ).values.astype(np.float32)

    total = matrix.size
    nonzero = np.count_nonzero(matrix)
    print(f"Matrix shape     : {matrix.shape}")
    print(f"Sparsity         : {1 - nonzero/total:.3f}")
    print(f"Rating range     : {matrix[matrix>0].min():.1f} - {matrix.max():.1f}")
    return matrix


def sample_matrix(matrix, n_users=200, n_items=200, seed=42):
    np.random.seed(seed)

    user_activity  = np.count_nonzero(matrix, axis=1)
    top_users      = np.argsort(user_activity)[-n_users:]
    item_popularity = np.count_nonzero(matrix, axis=0)
    top_items      = np.argsort(item_popularity)[-n_items:]

    sub = matrix[np.ix_(top_users, top_items)]
    print(f"Alt matris       : {sub.shape}")
    print(f"Alt sparsity     : {1 - np.count_nonzero(sub)/sub.size:.3f}")
    return sub


# ============================================================
# BÖLÜM 2: FITNESS FONKSİYONU
# ============================================================

def pearson_distance_batch(users, centroids):
    """
    Vektörize Pearson korelasyon mesafesi.
    CF'de 0 = eksik veri; sadece non-zero değerler üzerinden ortalama alınır.
    """
    user_mask = (users    != 0).astype(np.float32)
    cent_mask = (centroids != 0).astype(np.float32)

    u_count = user_mask.sum(axis=1, keepdims=True)
    c_count = cent_mask.sum(axis=1, keepdims=True)
    u_count = np.where(u_count == 0, 1, u_count)
    c_count = np.where(c_count == 0, 1, c_count)

    u_mean = (users    * user_mask).sum(axis=1, keepdims=True) / u_count
    c_mean = (centroids * cent_mask).sum(axis=1, keepdims=True) / c_count

    # Sadece izlenen filmlerde merkezleme; izlenmeyenler 0 kalır
    u_centered = np.where(user_mask > 0, users    - u_mean, 0)
    c_centered = np.where(cent_mask > 0, centroids - c_mean, 0)

    u_norm = np.linalg.norm(u_centered, axis=1, keepdims=True)
    c_norm = np.linalg.norm(c_centered, axis=1, keepdims=True)
    u_norm = np.where(u_norm == 0, 1, u_norm)
    c_norm = np.where(c_norm == 0, 1, c_norm)

    corr = u_centered / u_norm @ (c_centered / c_norm).T
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
    corr = np.clip(corr, -1, 1)
    return 1 - corr


def euclidean_distance_batch(users, centroids):
    """
    Squared Euclidean distances: users (n, d), centroids (K, d) -> (n, K).
    ||u - c||^2 = ||u||^2 + ||c||^2 - 2 u·c
    """
    users = np.asarray(users, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    u2 = (users ** 2).sum(axis=1, keepdims=True)
    c2 = (centroids ** 2).sum(axis=1, keepdims=True).T
    cross = users @ centroids.T
    d2 = u2 + c2 - 2.0 * cross
    return np.maximum(d2, 0.0)


def _fcm_memberships_from_dist(dist_matrix, m: float = 2.0, eps: float = 1e-12):
    """
    FCM üyelikleri: u_ij = 1 / sum_k (d_ij / d_ik)^(2/(m-1)).
    dist_matrix shape: (n_users, K), değerler d_ij (non-negative).
    """
    dist = np.asarray(dist_matrix, dtype=np.float64)
    if dist.ndim != 2:
        raise ValueError(f"dist_matrix 2D olmalı, gelen shape={dist.shape}")

    n_users, n_clusters = dist.shape
    memberships = np.zeros((n_users, n_clusters), dtype=np.float64)
    power = 2.0 / (float(m) - 1.0)

    for i in range(n_users):
        row = np.maximum(dist[i], 0.0)
        zero_mask = row <= eps
        if np.any(zero_mask):
            # d_ij = 0 için üyelik, sıfır mesafeli kümeler arasında eşit paylaşılır.
            zc = int(np.sum(zero_mask))
            memberships[i, zero_mask] = 1.0 / zc
            continue

        ratio = (row[:, None] / row[None, :]) ** power
        denom = np.sum(ratio, axis=1)
        memberships[i] = 1.0 / np.maximum(denom, eps)

    return memberships


def compute_fcm_objective(matrix, solution, K, m: float = 2.0):
    """
    FCM amaç fonksiyonu:
        J = sum_i sum_j (u_ij^m) * (d_ij^2)
    Burada d_ij Öklid mesafesi, d_ij^2 için squared Euclidean kullanılır.
    """
    centroids = solution.reshape(K, matrix.shape[1])
    d2 = euclidean_distance_batch(matrix, centroids)   # squared Euclidean
    d = np.sqrt(np.maximum(d2, 0.0))
    memberships = _fcm_memberships_from_dist(d, m=m)
    j_val = float(np.sum((memberships ** float(m)) * d2))
    hard_assignments = np.argmax(memberships, axis=1).astype(np.int32)
    return j_val, hard_assignments, memberships


def compute_wcss_fast(matrix, solution, K, metric='pearson'):
    """
    metric='pearson' : rating/latent için Pearson mesafesi (1-corr) toplamı.
    metric='euclidean' : k-means tarzı SSE — atanan merkeze squared L2 toplamı.
    """
    centroids = solution.reshape(K, matrix.shape[1])
    if metric == 'fuzzy':
        j_val, hard_assignments, _ = compute_fcm_objective(matrix, solution, K, m=2.0)
        return float(j_val), hard_assignments
    if metric == 'euclidean':
        dist_matrix = euclidean_distance_batch(matrix, centroids)
    elif metric == 'pearson':
        dist_matrix = pearson_distance_batch(matrix, centroids)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    assignments = np.argmin(dist_matrix, axis=1)
    min_distances = dist_matrix[np.arange(len(matrix)), assignments]
    return float(min_distances.sum()), assignments


def make_fitness_function(matrix, K, metric='pearson'):
    baseline = None
    eps = 1e-12

    def _sample_indices(n_rows, max_n=300):
        if n_rows <= max_n:
            return np.arange(n_rows)
        return np.random.choice(n_rows, max_n, replace=False)

    def _multiobjective_score(solution):
        nonlocal baseline
        wcss, assignments = compute_wcss_fast(matrix, solution, K, metric=metric)

        idx = _sample_indices(len(matrix), max_n=300)
        sub_x = matrix[idx]
        sub_y = assignments[idx]

        sil = -1.0
        ch = 1.0
        uniq = np.unique(sub_y)
        if len(uniq) >= 2 and len(uniq) < len(sub_y):
            try:
                sil = float(silhouette_score(sub_x, sub_y, metric='cosine'))
            except Exception:
                sil = -1.0
            try:
                ch = float(calinski_harabasz_score(sub_x, sub_y))
            except Exception:
                ch = 1.0

        obj_wcss = float(max(wcss, 0.0))
        obj_sil = float(1.0 - sil)
        obj_ch = float(1.0 / max(ch, eps))

        if baseline is None:
            baseline = {
                'wcss': max(obj_wcss, eps),
                'sil': max(obj_sil, eps),
                'ch': max(obj_ch, eps),
            }

        norm_wcss = obj_wcss / baseline['wcss']
        norm_sil = obj_sil / baseline['sil']
        norm_ch = obj_ch / baseline['ch']
        fitness_base = float(0.50 * norm_wcss + 0.25 * norm_sil + 0.25 * norm_ch)
        active_clusters = len(np.unique(assignments))
        missing_ratio = max(0.0, 1.0 - active_clusters / K)
        penalty = fitness_base * missing_ratio * 2.0
        return float(fitness_base + penalty)

    def fitness(solution):
        return _multiobjective_score(solution)
    return fitness


# ============================================================
# BÖLÜM 3: MKMEANS++ BAŞLANGIÇ POPÜLASYONU
# ============================================================

def mkmeans_plus_plus_init(matrix, K, n_solutions=30, seed=42, metric='pearson'):
    np.random.seed(seed)
    n_users, _ = matrix.shape
    solutions  = []

    def _pairwise_dists(rows_idx):
        sub = matrix[rows_idx]
        if metric == 'euclidean':
            return euclidean_distance_batch(matrix, sub)
        return pearson_distance_batch(matrix, sub)

    for _ in range(n_solutions):
        idx = [np.random.randint(0, n_users)]
        for k in range(1, K):
            dists = _pairwise_dists(idx).min(axis=1)
            dists = np.maximum(dists, 0)
            probs = dists ** 2
            total = probs.sum()
            probs = probs / total if total > 0 else np.ones(n_users) / n_users
            idx.append(np.random.choice(n_users, p=probs))
        solutions.append(matrix[idx].flatten())

    print(f"MkMeans++ ile {n_solutions} başlangıç çözümü oluşturuldu ({metric})")
    return solutions


# ============================================================
# BÖLÜM 4: GRAY SHEEP TESPİTİ
# ============================================================

def detect_gray_sheep(matrix, assignments, solution, K, threshold=None,
                      metric='pearson'):
    centroids      = solution.reshape(K, matrix.shape[1])
    n_users        = len(matrix)
    user_distances = np.zeros(n_users)

    if metric == 'euclidean':
        for i in range(n_users):
            user_distances[i] = float(
                np.sum((matrix[i].astype(np.float64) - centroids[assignments[i]]) ** 2)
            )
    else:
        for i in range(n_users):
            user     = matrix[i]
            centroid = centroids[assignments[i]]
            mask     = (user != 0) | (centroid != 0)

            if mask.sum() < 2:
                user_distances[i] = 1.0
                continue

            u_f, c_f = user[mask], centroid[mask]
            if np.std(u_f) == 0 or np.std(c_f) == 0:
                user_distances[i] = 1.0
            else:
                corr = float(np.nan_to_num(np.corrcoef(u_f, c_f)[0, 1], nan=0.0))
                user_distances[i] = 1 - np.clip(corr, -1, 1)

    if threshold is None:
        threshold = np.percentile(user_distances, 80)

    mask = user_distances > threshold
    return {
        'gray_sheep_mask'  : mask,
        'user_distances'   : user_distances,
        'threshold'        : threshold,
        'gray_sheep_count' : int(mask.sum()),
        'gray_sheep_ratio' : float(mask.mean()),
    }


# ============================================================
# BÖLÜM 5: ALGORİTMA LİSTESİ
# ============================================================

BLACKLIST = {
    'LSHADEcnEpSin.OriginalLSHADEcnEpSin',
    'PSS.OriginalPSS', 'ACOR.OriginalACOR', 'FFA.OriginalFFA',
    'ALO.OriginalALO', 'HCO.OriginalHCO', 'SPBO.OriginalSPBO',
    'GSKA.OriginalGSKA', 'MA.OriginalMA', 'ESO.OriginalESO',
    'BFO.OriginalBFO', 'DMOA.OriginalDMOA',
    'HS.OriginalHS', 'EPC.DevEPC',
}


def get_special_params(algo_name, epoch, pop_size):
    special = {
        'BCO.OriginalBCO'  : {'epoch': epoch, 'pop_size': pop_size, 'n_chemotaxis': 3},
        'IWO.OriginalIWO'  : {'epoch': epoch, 'pop_size': pop_size, 'seed_max': 5},
        'BSO.OriginalBSO'  : {'epoch': epoch, 'pop_size': pop_size, 'm_clusters': 3},
        'CHIO.OriginalCHIO': {'epoch': epoch, 'pop_size': pop_size, 'max_age': 2},
        'SARO.OriginalSARO': {'epoch': epoch, 'pop_size': pop_size, 'mu': 5},
        'CEM.OriginalCEM'  : {'epoch': epoch, 'pop_size': pop_size, 'n_best': 5},
        'BSA.OriginalBSA'  : {'epoch': epoch, 'pop_size': pop_size, 'ff': 5},
        'EHO.OriginalEHO'  : {'epoch': epoch, 'pop_size': pop_size, 'n_clans': 3},
        'GA.EliteMultiGA': {
            'epoch': epoch, 'pop_size': pop_size,
            'pc': 0.9, 'pm': 0.05,
            'crossover': 'arithmetic', 'mutation': 'swap',
            'elite_best': 0.1, 'elite_worst': 0.3,
        },
        'GAHHO.OriginalGAHHO': {'epoch': epoch, 'pop_size': pop_size, 'pc': 0.9, 'pm': 0.05, 'crossover': 'arithmetic', 'mutation': 'swap', 'mutation_multipoints': True},
        'SSA.DevSSA': {
            'epoch': epoch, 'pop_size': pop_size,
            'ST': 0.8, 'PD': 0.2, 'SD': 0.1
        },
    }
    return special.get(
        algo_name, {'epoch': epoch, 'pop_size': pop_size}
    )


def get_all_algorithms_v3():
    import mealpy, inspect, pkgutil
    algo_list = []
    for _, modname, _ in pkgutil.walk_packages(
        path=mealpy.__path__, prefix=mealpy.__name__ + '.', onerror=lambda x: None
    ):
        try:
            module = __import__(modname, fromlist="dummy")
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (any(name.startswith(p) for p in ('Original', 'Dev', 'Improved', 'CL_', 'Swarm', 'Levy', 'AIW_', 'Elite')) and
                        hasattr(obj, 'solve') and hasattr(obj, 'generate_population')):
                    full_name = f"{modname.split('.')[-1]}.{name}"
                    if full_name not in BLACKLIST:
                        algo_list.append({
                            'full_name': full_name,
                            'module': modname,
                            'class_name': name,
                            'class': obj,
                        })
        except Exception:
            pass

    seen, unique = set(), []
    for a in algo_list:
        if a['full_name'] not in seen:
            seen.add(a['full_name'])
            unique.append(a)
    print(f"Toplam {len(unique)} algoritma (blacklist hariç)")
    return unique


# ============================================================
# BÖLÜM 6: TEK ALGORİTMA ÇALIŞTIR
# ============================================================

def _compute_metrics(matrix, best_sol, K):
    """Ortak metrik hesaplama: WCSS, gray sheep, silhouette, DB."""
    _, assignments = compute_wcss_fast(matrix, best_sol, K)
    gs_info = detect_gray_sheep(matrix, assignments, best_sol, K)

    white_mask   = ~gs_info['gray_sheep_mask']
    white_matrix = matrix[white_mask]
    white_assign = assignments[white_mask]

    sil, db = -1.0, 999.0
    if len(np.unique(white_assign)) >= 2 and len(white_matrix) > 10:
        try:
            n_sample   = min(300, len(white_matrix))
            idx        = np.random.choice(len(white_matrix), n_sample, replace=False)
            sub_mat    = white_matrix[idx]
            sub_assign = white_assign[idx]
            dist_mat   = pearson_distance_batch(sub_mat, sub_mat)
            dist_mat   = np.clip(dist_mat, 0, 2)
            np.fill_diagonal(dist_mat, 0)
            sil = silhouette_score(dist_mat, sub_assign, metric='precomputed')
            db = davies_bouldin_score(white_matrix, white_assign)
        except Exception:
            pass

    return {
        'silhouette'          : float(sil),
        'davies_bouldin'      : float(db),
        'gray_sheep_count'    : int(gs_info['gray_sheep_count']),
        'gray_sheep_ratio'    : float(gs_info['gray_sheep_ratio']),
        'gray_sheep_threshold': float(gs_info['threshold']),
    }


def _aggregate_phase_runs(rows, algo_name, n_runs):
    """n_runs tekrarın sonuçlarını birleştirir; tümü başarılı değilse hata döner."""
    if len(rows) != n_runs:
        return {
            'algorithm': algo_name, 'wcss': None, 'silhouette': None,
            'davies_bouldin': None, 'gray_sheep_count': None,
            'gray_sheep_ratio': None, 'gray_sheep_threshold': None,
            'time_seconds': None, 'success': False,
            'error': f'n_runs uyuşmazlığı: {len(rows)} != {n_runs}',
        }
    if not all(r.get('success') for r in rows):
        errs = [str(r.get('error') or '') for r in rows if not r.get('success')]
        return {
            'algorithm': algo_name, 'wcss': None, 'silhouette': None,
            'davies_bouldin': None, 'gray_sheep_count': None,
            'gray_sheep_ratio': None, 'gray_sheep_threshold': None,
            'time_seconds': float(np.mean([r.get('time_seconds') or 0 for r in rows])),
            'success': False, 'error': '; '.join(e for e in errs if e) or 'partial failure',
            'n_runs': n_runs,
        }

    def _mean(key):
        return float(np.mean([float(r[key]) for r in rows]))

    return {
        'algorithm': algo_name,
        'wcss': _mean('wcss'),
        'silhouette': float(np.nanmean([r['silhouette'] for r in rows])),
        'davies_bouldin': _mean('davies_bouldin'),
        'gray_sheep_count': int(round(_mean('gray_sheep_count'))),
        'gray_sheep_ratio': _mean('gray_sheep_ratio'),
        'gray_sheep_threshold': _mean('gray_sheep_threshold'),
        'time_seconds': _mean('time_seconds'),
        'success': True,
        'error': None,
        'n_runs': n_runs,
    }


def _run_algo_v3_serialized(algo_module, class_name, algo_name,
                            matrix, K, initial_solutions,
                            epoch, pop_size, rng_seed=None):
    """
    Tek algoritma çözümü (module/class string ile) — timeout worker ve
    ProcessPoolExecutor görevleri için ortak gövde; sonuç sözlüğü döner.
    """
    if rng_seed is not None:
        np.random.seed(int(rng_seed))
    start = time.time()
    try:
        from mealpy import FloatVar
        import importlib

        n_items = matrix.shape[1]
        bounds = FloatVar(lb=[0.0] * (K * n_items),
                          ub=[5.0] * (K * n_items), name="centroids")
        problem = {
            "obj_func"       : make_fitness_function(matrix, K),
            "bounds"         : bounds,
            "minmax"         : "min",
            "log_to"         : None,
            "save_population": False,
        }

        module = importlib.import_module(algo_module)
        AlgoClass = getattr(module, class_name)
        special = get_special_params(algo_name, epoch, pop_size)
        model = AlgoClass(**(special or {'epoch': epoch, 'pop_size': pop_size}))

        try:
            model.solve(problem, starting_solutions=initial_solutions[:pop_size])
        except TypeError:
            model.solve(problem)

        best_sol = model.g_best.solution
        best_fit, _ = compute_wcss_fast(matrix, best_sol, K, metric='pearson')
        metrics = _compute_metrics(matrix, best_sol, K)

        return {
            'algorithm'   : algo_name,
            'wcss'        : float(best_fit),
            'success'     : True,
            'error'       : None,
            'time_seconds': time.time() - start,
            **metrics,
        }
    except Exception as e:
        return {
            'algorithm'           : algo_name,
            'wcss'                : None,
            'silhouette'          : None,
            'davies_bouldin'      : None,
            'gray_sheep_count'    : None,
            'gray_sheep_ratio'    : None,
            'gray_sheep_threshold': None,
            'time_seconds'        : time.time() - start,
            'success'             : False,
            'error'               : str(e),
        }


def _solve_worker_v3(algo_module, class_name, algo_name,
                     matrix, K, initial_solutions,
                     epoch, pop_size, out_queue, rng_seed=None):
    """Multiprocessing worker (timeout dalı)."""
    out_queue.put(_run_algo_v3_serialized(
        algo_module, class_name, algo_name, matrix, K,
        initial_solutions, epoch, pop_size, rng_seed=rng_seed,
    ))


def run_algorithm_v3(algo_info, matrix, K,
                     initial_solutions, epoch, pop_size,
                     time_limit=None):
    start     = time.time()
    algo_name = algo_info['full_name']

    try:
        AlgoClass = algo_info['class']
        special   = get_special_params(algo_name, epoch, pop_size)
        model     = AlgoClass(**(special or {'epoch': epoch, 'pop_size': pop_size}))

        # ── Multiprocess yolu (timeout varsa) ──────────────
        if time_limit is not None:
            ctx  = mp.get_context("spawn")
            q    = ctx.Queue()
            proc = ctx.Process(
                target=_solve_worker_v3,
                args=(algo_info['module'], algo_info['class_name'],
                      algo_name, matrix, K,
                      initial_solutions, epoch, pop_size, q),
            )
            proc.daemon = True
            proc.start()
            proc.join(timeout=time_limit)

            if proc.is_alive():
                proc.terminate(); proc.join()
                return {
                    'algorithm': algo_name,
                    'wcss': None, 'silhouette': None, 'davies_bouldin': None,
                    'gray_sheep_count': None, 'gray_sheep_ratio': None,
                    'gray_sheep_threshold': None,
                    'time_seconds': time_limit, 'success': False,
                    'error': f'TIMEOUT ({time_limit}s aşıldı)',
                }

            try:
                return q.get(timeout=5)
            except Exception:
                return {
                    'algorithm': algo_name,
                    'wcss': None, 'silhouette': None, 'davies_bouldin': None,
                    'gray_sheep_count': None, 'gray_sheep_ratio': None,
                    'gray_sheep_threshold': None,
                    'time_seconds': time_limit, 'success': False,
                    'error': 'NO_RESULT_FROM_WORKER',
                }

        # ── Direkt çalıştır (timeout yok) ──────────────────
        from mealpy import FloatVar

        n_items = matrix.shape[1]
        problem = {
            "obj_func"       : make_fitness_function(matrix, K),
            "bounds"         : FloatVar(lb=[0.0] * (K * n_items),
                                        ub=[5.0] * (K * n_items), name="centroids"),
            "minmax"         : "min",
            "log_to"         : None,
            "save_population": False,
        }

        try:
            model.solve(problem, starting_solutions=initial_solutions[:pop_size])
        except TypeError:
            model.solve(problem)

        best_sol = model.g_best.solution
        best_fit, _ = compute_wcss_fast(matrix, best_sol, K, metric='pearson')
        metrics  = _compute_metrics(matrix, best_sol, K)

        return {
            'algorithm'  : algo_name,
            'wcss'       : float(best_fit),
            'success'    : True,
            'error'      : None,
            'time_seconds': time.time() - start,
            **metrics,
        }

    except Exception as e:
        return {
            'algorithm'           : algo_name,
            'wcss'                : None,
            'silhouette'          : None,
            'davies_bouldin'      : None,
            'gray_sheep_count'    : None,
            'gray_sheep_ratio'    : None,
            'gray_sheep_threshold': None,
            'time_seconds'        : time.time() - start,
            'success'             : False,
            'error'               : str(e),
        }


# ============================================================
# BÖLÜM 6b: DAVRANIŞ ANALİZİ
# ============================================================

def _convergence_speed(history):
    if len(history) < 2:
        return 0
    total = history[0] - history[-1]
    if total == 0:
        return len(history)
    target = history[0] - 0.8 * total
    for i, v in enumerate(history):
        if v <= target:
            return i + 1
    return len(history)


def _exploration_ratio(history):
    n = len(history)
    if n < 10:
        return 0
    late_start = int(n * 0.7)
    late = history[late_start] - history[-1]
    total = history[0] - history[-1]
    return (late / total) if total != 0 else 0


def run_algorithm_with_history(algo_info, matrix, K,
                                initial_solutions, epoch, pop_size):
    from mealpy import FloatVar
    n_items  = matrix.shape[1]
    problem  = {
        "obj_func"       : make_fitness_function(matrix, K),
        "bounds"         : FloatVar(lb=[0.0] * (K * n_items),
                                    ub=[5.0] * (K * n_items)),
        "minmax"         : "min",
        "log_to"         : None,
        "save_population": False,
    }
    algo_name = algo_info['full_name']
    special   = get_special_params(algo_name, epoch, pop_size)
    model     = algo_info['class'](**(special or {'epoch': epoch, 'pop_size': pop_size}))

    solve_start = time.time()
    try:
        model.solve(problem, starting_solutions=initial_solutions[:pop_size])
    except TypeError:
        model.solve(problem)
    execution_time_sec = time.time() - solve_start

    history    = model.history.list_global_best_fit
    conv_speed = _convergence_speed(history)
    expl_ratio = _exploration_ratio(history)
    best_sol = model.g_best.solution
    _, labels = compute_wcss_fast(matrix, best_sol, K)

    ch_score = np.nan
    unique_labels = np.unique(labels)
    if len(unique_labels) >= 2 and len(unique_labels) < len(matrix):
        try:
            ch_score = float(calinski_harabasz_score(matrix, labels))
        except Exception:
            ch_score = np.nan

    if conv_speed <= epoch * 0.3 and expl_ratio < 0.1:
        category = 'LOCAL'
    elif conv_speed >= epoch * 0.6 and expl_ratio >= 0.2:
        category = 'GLOBAL'
    else:
        category = 'BALANCED'

    final_wcss, _ = compute_wcss_fast(matrix, best_sol, K, metric='pearson')

    return {
        'algorithm'      : algo_name,
        'final_wcss'     : float(final_wcss),
        'history'        : history,
        'convergence_speed': conv_speed,
        'exploration_ratio': expl_ratio,
        'execution_time_sec': float(execution_time_sec),
        'ch_score': ch_score,
        'category'       : category,
    }


def run_behavior_analysis(algo_list, matrix, K,
                           initial_solutions, epoch, pop_size,
                           save_path=RESULTS_DIR):
    print(f"\n{'='*60}")
    print(f"DAVRANIŞ ANALİZİ: {len(algo_list)} algoritma")
    print(f"{'='*60}")

    results = []
    for i, algo in enumerate(algo_list):
        print(f"[{i+1}/{len(algo_list)}] {algo['full_name']}...", end=' ', flush=True)
        try:
            r = run_algorithm_with_history(algo, matrix, K,
                                            initial_solutions, epoch, pop_size)
            results.append(r)
            print(f"✓ WCSS={r['final_wcss']:.1f} | "
                  f"ConvSpeed={r['convergence_speed']} | "
                  f"ExplRatio={r['exploration_ratio']:.2f} | "
                  f"Time={r['execution_time_sec']:.2f}s | "
                  f"CH={r['ch_score']:.2f} | "
                  f"Kategori={r['category']}")
        except Exception as e:
            print(f"✗ {e}")

    local_algos   = [r for r in results if r['category'] == 'LOCAL']
    global_algos  = [r for r in results if r['category'] == 'GLOBAL']
    balanced_algos = [r for r in results if r['category'] == 'BALANCED']

    print(f"\nKATEGORİ ÖZET:")
    print(f"  LOCAL   : {len(local_algos)}")
    print(f"  GLOBAL  : {len(global_algos)}")
    print(f"  BALANCED: {len(balanced_algos)}")

    for label, lst in [("LOCAL", local_algos), ("GLOBAL", global_algos)]:
        if lst:
            print(f"\n{label} algoritmalar:")
            for r in sorted(lst, key=lambda x: x['final_wcss']):
                print(f"  {r['algorithm']:<35} WCSS={r['final_wcss']:.1f}")

    pd.DataFrame([{
        'algorithm'        : r['algorithm'],
        'final_wcss'       : r['final_wcss'],
        'convergence_speed': r['convergence_speed'],
        'exploration_ratio': r['exploration_ratio'],
        'execution_time_sec': r['execution_time_sec'],
        'ch_score'         : r['ch_score'],
        'category'         : r['category'],
    } for r in results]).to_csv(
        os.path.join(save_path, 'behavior_analysis.csv'), index=False
    )
    return results, local_algos, global_algos


# ============================================================
# BÖLÜM 6c: HİBRİT OPTİMİZER (Sequential)
# ============================================================

class HybridOptimizer:
    """
    Sıralı hibrit: önce global tarama, sonra lokal ince ayar.
    ENCR switch kodu (phase4_encr.py) bu sınıfın yerini alacak.
    """

    def __init__(self, global_algo_class, local_algo_class,
                 global_algo_name, local_algo_name,
                 global_epoch, local_epoch, pop_size):
        self.global_algo_class = global_algo_class
        self.local_algo_class  = local_algo_class
        self.global_algo_name  = global_algo_name
        self.local_algo_name   = local_algo_name
        self.global_epoch = global_epoch
        self.local_epoch  = local_epoch
        self.pop_size     = pop_size
        self.g_best       = None
        self.global_history = []
        self.local_history  = []

    def solve(self, problem, initial_solutions=None):
        print(f"    [Global] {self.global_algo_name} çalışıyor...")
        special_g = get_special_params(self.global_algo_name,
                                        self.global_epoch, self.pop_size)
        global_model = self.global_algo_class(
            **(special_g or {'epoch': self.global_epoch, 'pop_size': self.pop_size})
        )
        try:
            global_model.solve(
                problem,
                starting_solutions=(initial_solutions[:self.pop_size]
                                    if initial_solutions else None)
            )
        except TypeError:
            global_model.solve(problem)

        best_global_sol = global_model.g_best.solution
        best_global_fit = global_model.g_best.target.fitness
        self.global_history = global_model.history.list_global_best_fit
        print(f"    [Global] En iyi WCSS: {best_global_fit:.2f}")

        print(f"    [Lokal]  {self.local_algo_name} çalışıyor...")
        special_l = get_special_params(self.local_algo_name,
                                        self.local_epoch, self.pop_size)
        local_model = self.local_algo_class(
            **(special_l or {'epoch': self.local_epoch, 'pop_size': self.pop_size})
        )
        local_start = [best_global_sol.copy() for _ in range(self.pop_size)]
        try:
            local_model.solve(problem, starting_solutions=local_start)
        except TypeError:
            local_model.solve(problem)

        best_local_fit = local_model.g_best.target.fitness
        self.local_history = local_model.history.list_global_best_fit

        if best_local_fit < best_global_fit:
            self.g_best = local_model.g_best
            print(f"    [Lokal]  İyileştirdi: {best_local_fit:.2f}")
        else:
            self.g_best = global_model.g_best
            print(f"    [Lokal]  Global daha iyi kaldı: {best_global_fit:.2f}")


def run_hybrid_algorithm(global_algo_info, local_algo_info,
                          matrix, K, initial_solutions,
                          global_epoch, local_epoch, pop_size):
    start       = time.time()
    hybrid_name = (f"Hybrid_{global_algo_info['full_name'].split('.')[0]}"
                   f"_{local_algo_info['full_name'].split('.')[0]}")
    try:
        from mealpy import FloatVar
        n_items = matrix.shape[1]
        problem = {
            "obj_func"       : make_fitness_function(matrix, K),
            "bounds"         : FloatVar(lb=[0.0] * (K * n_items),
                                        ub=[5.0] * (K * n_items)),
            "minmax"         : "min",
            "log_to"         : None,
            "save_population": False,
        }

        hybrid = HybridOptimizer(
            global_algo_class=global_algo_info['class'],
            local_algo_class=local_algo_info['class'],
            global_algo_name=global_algo_info['full_name'],
            local_algo_name=local_algo_info['full_name'],
            global_epoch=global_epoch,
            local_epoch=local_epoch,
            pop_size=pop_size,
        )
        hybrid.solve(problem, initial_solutions=initial_solutions)

        best_sol = hybrid.g_best.solution
        best_fit, _ = compute_wcss_fast(matrix, best_sol, K, metric='pearson')
        metrics  = _compute_metrics(matrix, best_sol, K)

        return {
            'algorithm'       : hybrid_name,
            'wcss'            : float(best_fit),
            'success'         : True,
            'error'           : None,
            'time_seconds'    : time.time() - start,
            'global_component': global_algo_info['full_name'],
            'local_component' : local_algo_info['full_name'],
            **metrics,
        }
    except Exception as e:
        return {
            'algorithm'           : hybrid_name,
            'wcss'                : None,
            'silhouette'          : None,
            'davies_bouldin'      : None,
            'gray_sheep_count'    : None,
            'gray_sheep_ratio'    : None,
            'gray_sheep_threshold': None,
            'time_seconds'        : time.time() - start,
            'success'             : False,
            'error'               : str(e),
            'global_component'    : global_algo_info['full_name'],
            'local_component'     : local_algo_info['full_name'],
        }


# ============================================================
# BÖLÜM 7: AŞAMA ÇALIŞTIR
# ============================================================

def run_phase(phase_num, algo_list, matrix, K,
              initial_solutions, epoch, pop_size,
              time_limit=None, save_path=RESULTS_DIR,
              parallel_workers=1, use_init=True, n_runs=1):

    init_for_run = initial_solutions if use_init else None
    nr = max(1, int(n_runs))
    os.makedirs(save_path, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"AŞAMA {phase_num}: {len(algo_list)} algoritma")
    print(f"Matrix: {matrix.shape} | K={K} | epoch={epoch} | pop={pop_size}")
    if nr > 1:
        print(f"n_runs={nr} (seed=0..{nr - 1}, WCSS/Sil/DB ve diğer metrikler ortalama)")
    pw = int(parallel_workers) if parallel_workers else 1
    if time_limit is not None and pw > 1:
        print("Uyarı: time_limit ile paralel çalıştırma desteklenmiyor; sıralı mod.")
        pw = 1
    if pw > 1:
        print(f"Paralel işçi sayısı: {pw} (ProcessPoolExecutor)")
    print(f"{'='*60}")

    results = []

    if pw > 1:
        import sys
        from concurrent.futures import ProcessPoolExecutor

        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)
        import mp_phase_worker as _mpw

        tasks = [
            (
                a['module'], a['class_name'], a['full_name'],
                matrix, K, init_for_run, epoch, pop_size, nr,
            )
            for a in algo_list
        ]
        with ProcessPoolExecutor(
            max_workers=pw,
            initializer=_mpw.pool_init,
            initargs=(BASE_DIR,),
        ) as ex:
            results = list(ex.map(_mpw.run_algo_v3_task, tasks))

        for i, result in enumerate(results):
            algo_name = result.get('algorithm', '?')
            print(f"\n[{i+1}/{len(results)}] {algo_name}...", end=' ', flush=True)
            if result['success']:
                sil = result.get('silhouette', float('nan'))
                sil_str = f"{sil:.3f}" if not np.isnan(sil) else "n/a"
                print(f"✓ WCSS={result['wcss']:.1f} | Sil={sil_str} | "
                      f"DB={result['davies_bouldin']:.3f} | "
                      f"GS={result['gray_sheep_ratio']:.1%} | "
                      f"{result['time_seconds']:.1f}s")
            else:
                print(f"✗ {result['error']}")
    else:
        for i, algo in enumerate(algo_list):
            print(f"\n[{i+1}/{len(algo_list)}] {algo['full_name']}...", end=' ', flush=True)
            if nr == 1:
                result = run_algorithm_v3(algo, matrix, K,
                                          init_for_run, epoch, pop_size,
                                          time_limit=time_limit)
            else:
                run_rows = []
                for run_index in range(nr):
                    np.random.seed(run_index)
                    run_rows.append(
                        run_algorithm_v3(algo, matrix, K,
                                         init_for_run, epoch, pop_size,
                                         time_limit=time_limit)
                    )
                result = _aggregate_phase_runs(
                    run_rows, algo['full_name'], nr)
            results.append(result)

            if result['success']:
                sil = result.get('silhouette', float('nan'))
                sil_str = f"{sil:.3f}" if not np.isnan(sil) else "n/a"
                print(f"✓ WCSS={result['wcss']:.1f} | Sil={sil_str} | "
                      f"DB={result['davies_bouldin']:.3f} | "
                      f"GS={result['gray_sheep_ratio']:.1%} | "
                      f"{result['time_seconds']:.1f}s")
            else:
                print(f"✗ {result['error']}")

            if (i + 1) % 10 == 0:
                pd.DataFrame(results).to_csv(
                    f'{save_path}/phase{phase_num}_partial.csv', index=False)

    df = pd.DataFrame(results)
    df.to_csv(f'{save_path}/phase{phase_num}_complete.csv', index=False)
    df[df['success'] == True ].to_csv(f'{save_path}/phase{phase_num}_success.csv', index=False)
    df[df['success'] == False].to_csv(f'{save_path}/phase{phase_num}_failed.csv',  index=False)

    suc = (df['success'] == True).sum()
    fail = (df['success'] == False).sum()
    print(f"\n{'='*60}")
    print(f"AŞAMA {phase_num} ÖZET: Başarılı={suc} | Başarısız={fail}")
    if fail > 0:
        print("\nBAŞARISIZ:")
        for _, row in df[df['success'] == False].iterrows():
            print(f"  {row['algorithm']:<35} → {row['error']}")
    return df


# ============================================================
# BÖLÜM 8: SIRALAMA
# ============================================================

def rank_and_filter(df, top_n=20, save_path=None):
    successful = df[df['success'] == True].copy()
    print(f"\nBaşarılı: {len(successful)} / {len(df)}")

    if len(successful) == 0:
        print("Hiç başarılı algoritma yok!")
        return successful

    def norm_min(col):
        r = successful[col].max() - successful[col].min()
        if r == 0:
            return pd.Series(0.5, index=successful.index)
        return 1 - (successful[col] - successful[col].min()) / r

    def norm_max(col):
        r = successful[col].max() - successful[col].min()
        if r == 0:
            return pd.Series(0.5, index=successful.index)
        return (successful[col] - successful[col].min()) / r

    successful['score_wcss'] = norm_min('wcss')
    successful['score_db']   = norm_min('davies_bouldin')
    successful['score_time'] = norm_min('time_seconds')
    successful['score_gray'] = norm_min('gray_sheep_ratio')

    if 'silhouette' in successful.columns and successful['silhouette'].notna().any():
        successful['score_sil'] = norm_max('silhouette')
    else:
        successful['score_sil'] = pd.Series(0.5, index=successful.index)

    W_WCSS, W_SIL, W_DB, W_GRAY, W_TIME = 0.35, 0.15, 0.30, 0.15, 0.05

    successful['composite'] = (
        W_WCSS * successful['score_wcss'] +
        W_SIL  * successful['score_sil']  +
        W_DB   * successful['score_db']   +
        W_GRAY * successful['score_gray'] +
        W_TIME * successful['score_time']
    )

    ranked = successful.sort_values('composite', ascending=False)

    print(f"\nTOP {min(top_n, len(ranked))} ALGORİTMA:")
    print(f"{'Algoritma':<35} {'WCSS':>8} {'Sil':>6} {'DB':>6} "
          f"{'GS%':>6} {'Süre':>6} {'Score':>7}")
    print(f"{'Ağırlık':<35} {'%35':>8} {'%15':>6} {'%30':>6} "
          f"{'%15':>6} {'%5':>6} {'':>7}")
    print("-" * 88)

    for _, row in ranked.head(top_n).iterrows():
        sil_v = row.get('silhouette', float('nan'))
        sil_s = f"{sil_v:6.3f}" if not (isinstance(sil_v, float) and np.isnan(sil_v)) else "   n/a"
        print(f"{row['algorithm']:<35} {row['wcss']:>8.1f} {sil_s} "
              f"{row['davies_bouldin']:>6.3f} {row.get('gray_sheep_ratio', 0):>5.1%} "
              f"{row['time_seconds']:>6.1f} {row['composite']:>7.3f}")

    out_dir = save_path if save_path is not None else RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    cols = ['algorithm', 'composite', 'score_wcss', 'score_sil', 'score_db',
            'score_gray', 'score_time', 'wcss', 'silhouette', 'davies_bouldin',
            'gray_sheep_ratio', 'time_seconds']
    ranked[[c for c in cols if c in ranked.columns]].to_csv(
        os.path.join(out_dir, 'ranked_scores.csv'), index=False)

    return ranked.head(top_n)


# ============================================================
# BÖLÜM 9: ANA ÇALIŞTIRMA
# ============================================================

if __name__ == "__main__":

    RUN_START = time.strftime('%Y-%m-%d %H:%M:%S')

    print("=" * 60)
    print("AŞAMALI ALGORİTMA TARAMA SİSTEMİ")
    print("Mealpy 3.x | MovieLens 100K | Gray Sheep + Hibrit")
    print("=" * 60)

    DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-100k', 'u.data')
    DATA_PATH_1M = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-1m', 'ratings.dat')
    full_matrix = load_movielens(DATA_PATH)
    all_algos   = get_all_algorithms_v3()

    K1, K2 = 50, 70

    # Tüm aşama çıktıları aynı çalıştırma kökünde (results/phase3/<n>/); alt klasörler aşama/dataset
    phase3_root = os.path.join(RESULTS_DIR, 'phase3')
    run_dirs = [int(d) for d in os.listdir(phase3_root) if d.isdigit()]
    if not run_dirs:
        raise RuntimeError("results/phase3 altında mevcut çalıştırma klasörü bulunamadı.")
    phase3_run_dir = os.path.join(phase3_root, str(max(run_dirs)))
    run_n = os.path.basename(phase3_run_dir)

    ml100k_phase2_dir = os.path.join(phase3_run_dir, 'ml100k-phase2')
    ml1m_phase2_dir = os.path.join(phase3_run_dir, 'ml1m-phase2')
    ml100k_phase3_dir = os.path.join(phase3_run_dir, 'ml-100k-phase3')
    ml1m_phase3_dir = os.path.join(phase3_run_dir, 'ml-1m-phase3')
    os.makedirs(ml100k_phase3_dir, exist_ok=True)
    os.makedirs(ml1m_phase3_dir, exist_ok=True)

    # ── Aşama 2 (mevcut sonuçlardan okuma, ML-100K) ───────────
    print(f"\n>>> AŞAMA 2 SKIP (mevcut dosyadan okunuyor)  (phase3/{run_n}/ml100k-phase2)")
    df_elim = pd.read_csv(os.path.join(phase3_run_dir, 'ml100k-phase2', 'phase2_complete.csv'))
    top25_elim = rank_and_filter(df_elim, top_n=25)
    top25_names = set(top25_elim['algorithm'].tolist())
    algos_phase3 = [a for a in all_algos if a['full_name'] in top25_names]

    # ── Aşama 3 (ML-100K tam matris) ───────────────────────
    print(f"\n>>> AŞAMA 3 (FİNAL, ML-100K) BAŞLIYOR  (phase3/{run_n}/ml-100k-phase3)")
    df3_100k = run_phase(3, algos_phase3, full_matrix, K2, None,
                         epoch=50, pop_size=30, time_limit=None,
                         parallel_workers=4, save_path=ml100k_phase3_dir,
                         use_init=False, n_runs=5)
    final = rank_and_filter(df3_100k, top_n=10, save_path=ml100k_phase3_dir)

    # ── Aşama 3 (ML-1M tam matris, aynı algos_phase3) ───────
    print(f"\n>>> AŞAMA 3 (FİNAL, ML-1M) BAŞLIYOR  (phase3/{run_n}/ml-1m-phase3)")
    full_matrix_1m = load_movielens_1m(DATA_PATH_1M)

    df3_1m = run_phase(3, algos_phase3, full_matrix_1m, K2, None,
                       epoch=70, pop_size=50, time_limit=None,
                       parallel_workers=4, save_path=ml1m_phase3_dir,
                       use_init=False, n_runs=5)
    rank_and_filter(df3_1m, top_n=10, save_path=ml1m_phase3_dir)

    # ── Davranış Analizi (tam ML-100K, algos_phase3) ────────
    print("\n>>> DAVRANIŞ ANALİZİ BAŞLIYOR (ML-100K tam matris, algos_phase3)")
    behavior_results, local_algos, global_algos = run_behavior_analysis(
        algos_phase3, full_matrix, K2,
        initial_solutions=None, epoch=50, pop_size=20,
        save_path=ml100k_phase3_dir,
    )

    # ── Hibrit Test (yalnız ML-100K) ───────────────────────
    print("\n>>> HİBRİT ALGORİTMA TESTİ BAŞLIYOR (ML-100K)")

    if len(global_algos) > 0:
        best_global_r = sorted(global_algos, key=lambda x: x['final_wcss'])[0]
        global_src    = "GLOBAL kategorisinden"
    else:
        best_global_r = sorted(behavior_results,
                               key=lambda x: x['exploration_ratio'],
                               reverse=True)[0]
        global_src = "en yüksek exploration_ratio (fallback)"

    if len(local_algos) > 0:
        best_local_r = sorted(local_algos, key=lambda x: x['final_wcss'])[0]
        local_src    = "LOCAL kategorisinden"
    else:
        sa_cands = [r for r in behavior_results
                    if 'SA' in r['algorithm'] and 'CSA' not in r['algorithm']]
        if sa_cands:
            best_local_r = sorted(sa_cands, key=lambda x: x['convergence_speed'])[0]
            local_src    = "SA (hızlı yakınsama, lokal bileşen)"
        else:
            best_local_r = sorted(behavior_results,
                                  key=lambda x: x['convergence_speed'])[0]
            local_src = "en hızlı yakınsayan (fallback)"

    if best_global_r['algorithm'] == best_local_r['algorithm']:
        for cand in sorted(behavior_results, key=lambda x: x['convergence_speed']):
            if cand['algorithm'] != best_global_r['algorithm']:
                best_local_r = cand
                local_src   += " (çakışma önlendi)"
                break

    global_info = next((a for a in all_algos
                        if a['full_name'] == best_global_r['algorithm']), None)
    local_info  = next((a for a in all_algos
                        if a['full_name'] == best_local_r['algorithm']), None)

    if global_info is None or local_info is None:
        print("  UYARI: Bileşen bulunamadı, hibrit atlandı.")
        hybrid_row = {
            'algorithm': 'Hybrid_SKIPPED', 'wcss': None, 'silhouette': None,
            'davies_bouldin': None, 'gray_sheep_count': None,
            'gray_sheep_ratio': None, 'gray_sheep_threshold': None,
            'time_seconds': None, 'success': False,
            'error': 'SKIP: bileşen all_algos içinde yok',
            'global_component': None, 'local_component': None,
        }
    else:
        print(f"\n  Global : {global_info['full_name']}  [{global_src}]")
        print(f"  Lokal  : {local_info['full_name']}  [{local_src}]")

        hybrid_row = run_hybrid_algorithm(
            global_info, local_info,
            full_matrix, K2, None,
            global_epoch=30, local_epoch=20, pop_size=30,
        )
        if hybrid_row['success']:
            print(f"\n  WCSS={hybrid_row['wcss']:.1f} | "
                  f"Sil={hybrid_row['silhouette']:.3f} | "
                  f"DB={hybrid_row['davies_bouldin']:.3f} | "
                  f"GS={hybrid_row['gray_sheep_ratio']:.1%} | "
                  f"{hybrid_row['time_seconds']:.1f}s")
        else:
            print(f"  HATA: {hybrid_row['error']}")

    pd.DataFrame([hybrid_row]).to_csv(
        os.path.join(ml100k_phase3_dir, 'hybrid_result.csv'), index=False)

    # ── Final özet ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEZİN İÇİN ÖNERİLEN ALGORİTMALAR:")
    print("=" * 60)
    for i, (_, row) in enumerate(final.iterrows()):
        print(f"\n{i+1}. {row['algorithm']}")
        print(f"   Composite : {row['composite']:.3f}")
        print(f"   WCSS      : {row['wcss']:.1f}")
        print(f"   DB        : {row['davies_bouldin']:.3f}")
        print(f"   GS Oranı  : {row['gray_sheep_ratio']:.1%}")
        print(f"   Süre      : {row['time_seconds']:.1f}s")

    final.to_csv(os.path.join(ml100k_phase3_dir, 'FINAL_RECOMMENDATIONS.csv'), index=False)

    # DÜZELTİLDİ: RUN_INFO.txt artık tarih + config içeriyor
    algo_names_p3 = [a['full_name'] for a in algos_phase3]
    manifest = f"""Phase 3 çalıştırması #{run_n} (eleme + çift final: ML-100K / ML-1M)
Tarih         : {RUN_START}
Kök klasör    : results/phase3/{run_n}/
Alt klasörler : ml100k-phase2 | ml1m-phase2 | ml-100k-phase3 | ml-1m-phase3

── Aşama 2 (eleme, ML-100K örnek) ─────────────
K             : {K1}
sample        : 200×200
use_init      : False
n_runs        : 20 (seed 0..19, ortalama metrik)
epoch         : 50
pop_size      : 30
time_limit    : yok (ProcessPoolExecutor, paralel)
parallel_workers: 4
kohort        : WCSS en iyi 10 ∪ Davies-Bouldin en iyi 5 (tekrarsız) → algos_phase3
ML-100K kayıt : …/{run_n}/ml100k-phase2/
ml1m-phase2/  : boş klasör (adlandırma tutarlılığı)

── Aşama 3 (final, ML-100K) ────────────────────
K             : {K2}
use_init      : False
n_runs        : 5
epoch         : 50
pop_size      : 30
time_limit    : yok (ProcessPoolExecutor, paralel)
parallel_workers: 4
matrix        : {full_matrix.shape}
sparsity      : {1 - np.count_nonzero(full_matrix)/full_matrix.size:.3f}
data_path     : {DATA_PATH}
kayıt         : …/{run_n}/ml-100k-phase3/
rank_and_filter: top_n=10

── Aşama 3 (final, ML-1M) ───────────────────────
K             : {K2}
use_init      : False
n_runs        : 5
epoch         : 50
pop_size      : 30
time_limit    : yok (ProcessPoolExecutor, paralel)
parallel_workers: 4
matrix        : {full_matrix_1m.shape}
sparsity      : {1 - np.count_nonzero(full_matrix_1m)/full_matrix_1m.size:.3f}
data_path     : {DATA_PATH_1M}
kayıt         : …/{run_n}/ml-1m-phase3/
rank_and_filter: top_n=10
katılımcılar  : algos_phase3 ({len(algos_phase3)})

── Davranış analizi (ML-100K tam matris) ───────
K             : {K2}
matrix        : {full_matrix.shape} (tam ML-100K)
kayıt         : …/{run_n}/ml-100k-phase3/
katılımcılar  : algos_phase3 ({len(algos_phase3)})

── Aşama 3 Algoritmaları ({len(algos_phase3)}) ─────────────────
{chr(10).join('  ' + n for n in algo_names_p3)}

── Hibrit (yalnız ML-100K) ─────────────────────
global : {global_info['full_name'] if global_info else 'SKIP'}  [{global_src}]
local  : {local_info['full_name']  if local_info  else 'SKIP'}  [{local_src}]

── Dosyalar ─────────────────────────────────────
  ml100k-phase2/phase2_*.csv       Aşama 2 eleme (ML-100K örnek)
  ml1m-phase2/                     (bu koşuda Aşama 2 yok; klasör oluşturulur)
  ml-100k-phase3/phase3_*.csv      Aşama 3 tam ML-100K
  ml-100k-phase3/ranked_scores.csv composite top-10 (ML-100K)
  ml-100k-phase3/FINAL_RECOMMENDATIONS.csv
  ml-100k-phase3/behavior_analysis.csv
  ml-100k-phase3/hybrid_result.csv
  ml-1m-phase3/phase3_*.csv        Aşama 3 tam ML-1M
  ml-1m-phase3/ranked_scores.csv     composite top-10 (ML-1M)
  RUN_INFO.txt                     bu dosya (kök)
"""
    with open(os.path.join(phase3_run_dir, 'RUN_INFO.txt'), 'w', encoding='utf-8') as f:
        f.write(manifest)

    print(f"\n── Kayıt: results/phase3/{run_n}/ ──")
    print("   ml100k-phase2 | ml1m-phase2 | ml-100k-phase3 | ml-1m-phase3")
    print("   RUN_INFO.txt (kök) | ML-100K: FINAL / behavior / hybrid → ml-100k-phase3/")