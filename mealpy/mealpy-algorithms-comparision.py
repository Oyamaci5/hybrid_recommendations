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
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
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


def compute_wcss_fast(matrix, solution, K):
    centroids    = solution.reshape(K, matrix.shape[1])
    dist_matrix  = pearson_distance_batch(matrix, centroids)
    assignments  = np.argmin(dist_matrix, axis=1)
    min_distances = dist_matrix[np.arange(len(matrix)), assignments]
    return float(min_distances.sum()), assignments


def make_fitness_function(matrix, K):
    def fitness(solution):
        wcss, _ = compute_wcss_fast(matrix, solution, K)
        return float(wcss)
    return fitness


# ============================================================
# BÖLÜM 3: MKMEANS++ BAŞLANGIÇ POPÜLASYONU
# ============================================================

def mkmeans_plus_plus_init(matrix, K, n_solutions=30, seed=42):
    np.random.seed(seed)
    n_users, _ = matrix.shape
    solutions  = []

    for _ in range(n_solutions):
        idx = [np.random.randint(0, n_users)]
        for k in range(1, K):
            dists = pearson_distance_batch(matrix, matrix[idx]).min(axis=1)
            dists = np.maximum(dists, 0)
            probs = dists ** 2
            total = probs.sum()
            probs = probs / total if total > 0 else np.ones(n_users) / n_users
            idx.append(np.random.choice(n_users, p=probs))
        solutions.append(matrix[idx].flatten())

    print(f"MkMeans++ ile {n_solutions} başlangıç çözümü oluşturuldu")
    return solutions


# ============================================================
# BÖLÜM 4: GRAY SHEEP TESPİTİ
# ============================================================

def detect_gray_sheep(matrix, assignments, solution, K, threshold=None):
    centroids      = solution.reshape(K, matrix.shape[1])
    n_users        = len(matrix)
    user_distances = np.zeros(n_users)

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
    'BFO.OriginalBFO', 'DMOA.OriginalDMOA', 'SSA.OriginalSSA',
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
    }
    return special.get(algo_name, None)


def get_all_algorithms_v3():
    import mealpy, inspect, pkgutil
    algo_list = []
    for _, modname, _ in pkgutil.walk_packages(
        path=mealpy.__path__, prefix=mealpy.__name__ + '.', onerror=lambda x: None
    ):
        try:
            module = __import__(modname, fromlist="dummy")
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (name.startswith('Original') and
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


def _run_algo_v3_serialized(algo_module, class_name, algo_name,
                            matrix, K, initial_solutions,
                            epoch, pop_size):
    """
    Tek algoritma çözümü (module/class string ile) — timeout worker ve
    ProcessPoolExecutor görevleri için ortak gövde; sonuç sözlüğü döner.
    """
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
        best_fit = model.g_best.target.fitness
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
                     epoch, pop_size, out_queue):
    """Multiprocessing worker (timeout dalı)."""
    out_queue.put(_run_algo_v3_serialized(
        algo_module, class_name, algo_name, matrix, K,
        initial_solutions, epoch, pop_size,
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
        best_fit = model.g_best.target.fitness
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

    try:
        model.solve(problem, starting_solutions=initial_solutions[:pop_size])
    except TypeError:
        model.solve(problem)

    history    = model.history.list_global_best_fit
    conv_speed = _convergence_speed(history)
    expl_ratio = _exploration_ratio(history)

    if conv_speed <= epoch * 0.3 and expl_ratio < 0.1:
        category = 'LOCAL'
    elif conv_speed >= epoch * 0.6 and expl_ratio >= 0.2:
        category = 'GLOBAL'
    else:
        category = 'BALANCED'

    return {
        'algorithm'      : algo_name,
        'final_wcss'     : float(model.g_best.target.fitness),
        'history'        : history,
        'convergence_speed': conv_speed,
        'exploration_ratio': expl_ratio,
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
        best_fit = hybrid.g_best.target.fitness
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
              parallel_workers=1):

    os.makedirs(save_path, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"AŞAMA {phase_num}: {len(algo_list)} algoritma")
    print(f"Matrix: {matrix.shape} | K={K} | epoch={epoch} | pop={pop_size}")
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
                matrix, K, initial_solutions, epoch, pop_size,
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
            result = run_algorithm_v3(algo, matrix, K,
                                      initial_solutions, epoch, pop_size,
                                      time_limit=time_limit)
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
    full_matrix = load_movielens(DATA_PATH)
    all_algos   = get_all_algorithms_v3()

    K1, K2 = 60, 90
    K_BEH = 60  # davranış analizi (küçük örnek matris 200×300)

    # Tüm aşama çıktıları aynı çalıştırma klasöründe (results/phase3/<n>/)
    phase3_run_dir = next_phase3_run_dir()
    run_n = os.path.basename(phase3_run_dir)

    # ── Aşama 2 (eleme, örnek matris) ─────────────────────────
    print(f"\n>>> AŞAMA 2 (ELEME) BAŞLIYOR  (phase3/{run_n})")
    matrix_eleme = sample_matrix(full_matrix, n_users=300, n_items=400)
    init_eleme   = mkmeans_plus_plus_init(matrix_eleme, K=K1, n_solutions=30)

    df_elim = run_phase(2, all_algos, matrix_eleme, K1, init_eleme,
                        epoch=30, pop_size=20, time_limit=180,
                        save_path=phase3_run_dir)
    top10_elim = rank_and_filter(df_elim, top_n=10)
    top10_names = set(top10_elim['algorithm'].tolist())
    algos_phase3 = [a for a in all_algos if a['full_name'] in top10_names]

    # ── Aşama 3 (tam matris) ─────────────────────────────────
    print(f"\n>>> AŞAMA 3 (FİNAL) BAŞLIYOR  (phase3/{run_n})")
    init_full = mkmeans_plus_plus_init(full_matrix, K=K2, n_solutions=50)

    df3 = run_phase(3, algos_phase3, full_matrix, K2, init_full,
                    epoch=50, pop_size=30, time_limit=900,
                    save_path=phase3_run_dir)
    final = rank_and_filter(df3, top_n=5, save_path=phase3_run_dir)

    # ── Davranış Analizi ─────────────────────────────────────
    print("\n>>> DAVRANIŞ ANALİZİ BAŞLIYOR")
    matrix_beh = sample_matrix(full_matrix, n_users=200, n_items=300)
    init_beh   = mkmeans_plus_plus_init(matrix_beh, K=K_BEH, n_solutions=20)

    behavior_results, local_algos, global_algos = run_behavior_analysis(
        algos_phase3, matrix_beh, K=K_BEH,
        initial_solutions=init_beh, epoch=50, pop_size=20,
        save_path=phase3_run_dir,
    )

    # ── Hibrit Test ───────────────────────────────────────────
    print("\n>>> HİBRİT ALGORİTMA TESTİ BAŞLIYOR")

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
            full_matrix, K2, init_full,
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
        os.path.join(phase3_run_dir, 'hybrid_result.csv'), index=False)

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

    final.to_csv(os.path.join(phase3_run_dir, 'FINAL_RECOMMENDATIONS.csv'), index=False)

    # DÜZELTİLDİ: RUN_INFO.txt artık tarih + config içeriyor
    algo_names_p3 = [a['full_name'] for a in algos_phase3]
    manifest = f"""Phase 3 çalıştırması #{run_n} (2 aşama: eleme + final)
Tarih         : {RUN_START}
Klasör        : results/phase3/{run_n}/

── Aşama 2 (eleme) ──────────────────────────────
K             : {K1}
sample        : 300×400 kullanıcı/film alt kümesi
init_solutions: 30
epoch         : 30
pop_size      : 20
time_limit    : 180s
final havuz   : elemeden top-10 (zorunlu algoritma yok)

── Aşama 3 (final) ───────────────────────────────
K             : {K2}
epoch         : 50
pop_size      : 30
time_limit    : 900s
matrix        : {full_matrix.shape}
sparsity      : {1 - np.count_nonzero(full_matrix)/full_matrix.size:.3f}
data_path     : {DATA_PATH}

── Davranış analizi ─────────────────────────────
K             : {K_BEH}
sample        : 200×300
katılımcılar  : Aşama 3'e giren top-10 kohort ({len(algos_phase3)})

── Aşama 3 Algoritmaları ({len(algos_phase3)}) ─────────────────
{chr(10).join('  ' + n for n in algo_names_p3)}

── Hibrit ───────────────────────────────────────
global : {global_info['full_name'] if global_info else 'SKIP'}  [{global_src}]
local  : {local_info['full_name']  if local_info  else 'SKIP'}  [{local_src}]

── Dosyalar ─────────────────────────────────────
  phase2_*.csv               Aşama 2 (eleme) ham/partial/success/failed
  phase3_complete.csv          ham Aşama 3 sonuçları
  phase3_success.csv           başarılı algoritmalar
  phase3_failed.csv            başarısız / timeout
  ranked_scores.csv            composite skor (final sıralama)
  FINAL_RECOMMENDATIONS.csv    top-5 özet
  behavior_analysis.csv        davranış kategorileri
  hybrid_result.csv            hibrit metrikler
  RUN_INFO.txt                 bu dosya
"""
    with open(os.path.join(phase3_run_dir, 'RUN_INFO.txt'), 'w', encoding='utf-8') as f:
        f.write(manifest)

    print(f"\n── Kayıt: results/phase3/{run_n}/ ──")
    print("   phase2_*.csv | phase3_*.csv | ranked_scores.csv | FINAL_RECOMMENDATIONS.csv")
    print("   behavior_analysis.csv | hybrid_result.csv | RUN_INFO.txt")