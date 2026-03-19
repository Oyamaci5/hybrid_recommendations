"""
Mealpy Algoritma Karşılaştırma Sistemi
MovieLens 100K - Gray Sheep Tespiti + Kümeleme Kalitesi
Mealpy 3.x uyumlu
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

    total = matrix.size
    nonzero = np.count_nonzero(matrix)

    print(f"Matrix shape     : {matrix.shape}")
    print(f"Sparsity         : {1 - nonzero/total:.3f}")
    print(f"Rating range     : {matrix[matrix>0].min():.1f} - {matrix.max():.1f}")
    return matrix


def sample_matrix(matrix, n_users=200, n_items=200, seed=42):
    np.random.seed(seed)

    user_activity = np.count_nonzero(matrix, axis=1)
    top_users = np.argsort(user_activity)[-n_users:]

    item_popularity = np.count_nonzero(matrix, axis=0)
    top_items = np.argsort(item_popularity)[-n_items:]

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
    users    : (n_users, n_items)
    centroids: (K, n_items)
    return   : (n_users, K) mesafe matrisi
    """
    u_mean = users.mean(axis=1, keepdims=True)
    c_mean = centroids.mean(axis=1, keepdims=True)

    u_centered = users - u_mean
    c_centered = centroids - c_mean

    u_norm = np.linalg.norm(u_centered, axis=1, keepdims=True)
    c_norm = np.linalg.norm(c_centered, axis=1, keepdims=True)

    u_norm = np.where(u_norm == 0, 1, u_norm)
    c_norm = np.where(c_norm == 0, 1, c_norm)

    u_normalized = u_centered / u_norm
    c_normalized = c_centered / c_norm

    corr = u_normalized @ c_normalized.T
    corr = np.clip(corr, -1, 1)

    return 1 - corr  # 0=benzer, 2=zıt


def compute_wcss_fast(matrix, solution, K):
    n_items = matrix.shape[1]
    centroids = solution.reshape(K, n_items)

    dist_matrix = pearson_distance_batch(matrix, centroids)
    assignments = np.argmin(dist_matrix, axis=1)
    min_distances = dist_matrix[np.arange(len(matrix)), assignments]

    wcss = min_distances.sum()
    return wcss, assignments


def make_fitness_function(matrix, K):
    def fitness(solution):
        wcss, _ = compute_wcss_fast(matrix, solution, K)
        return wcss
    return fitness

# ============================================================
# BÖLÜM 3: MKMEANS++ BAŞLANGIÇ POPÜLASYONU
# ============================================================

def mkmeans_plus_plus_init(matrix, K, n_solutions=30, seed=42):
    """
    MkMeans++ ile başlangıç popülasyonu üret.
    Metaheuristik DEĞİL — sadece akıllı başlangıç noktası üretici.
    Birbirinden uzak gerçek kullanıcıları merkez olarak seçer.
    """
    np.random.seed(seed)
    n_users, n_items = matrix.shape
    solutions = []

    for _ in range(n_solutions):
        centroids_idx = []

        # İlk merkez rastgele
        first_idx = np.random.randint(0, n_users)
        centroids_idx.append(first_idx)

        # Kalan K-1 merkezi mesafeye orantılı seç
        for k in range(1, K):
            current_centroids = matrix[centroids_idx]
            dist_matrix = pearson_distance_batch(matrix, current_centroids)
            min_dists = dist_matrix.min(axis=1)
            min_dists = np.maximum(min_dists, 0)

            probs = min_dists ** 2
            total = probs.sum()

            if total == 0:
                probs = np.ones(n_users) / n_users
            else:
                probs = probs / total

            chosen = np.random.choice(n_users, p=probs)
            centroids_idx.append(chosen)

        centroids = matrix[centroids_idx]
        solution = centroids.flatten()
        solutions.append(solution)

    print(f"MkMeans++ ile {n_solutions} başlangıç çözümü oluşturuldu")
    return solutions

# ============================================================
# BÖLÜM 4: GRAY SHEEP TESPİTİ (DÜZELTİLDİ)
# ============================================================

def detect_gray_sheep(matrix, assignments, solution, K, threshold=None):
    """
    Kullanıcıları white / gray sheep olarak ayır.

    Kriter: küme merkezine Pearson korelasyon mesafesi.
    Threshold: persentil tabanlı adaptif (üst %20 gray sheep).

    NOT: Aktivite/rating sayısı ile karıştırılmamalı.
         Bir kullanıcı az film izlemiş olsa da
         izlediklerinde kümeyle tutarlıysa white user'dır.
    """
    n_items = matrix.shape[1]
    centroids = solution.reshape(K, n_items)

    # Her kullanıcının kendi küme merkezine Pearson mesafesi
    n_users = len(matrix)
    user_distances = np.zeros(n_users)

    for i in range(n_users):
        user = matrix[i]
        centroid = centroids[assignments[i]]

        std_u = np.std(user)
        std_c = np.std(centroid)

        if std_u == 0 or std_c == 0:
            user_distances[i] = 1.0
        else:
            corr = np.corrcoef(user, centroid)[0, 1]
            corr = np.clip(corr, -1, 1)
            user_distances[i] = 1 - corr  # 0=aynı, 2=zıt

    if threshold is None:
        # Üst %20 gray sheep olarak etiketle
        # Literatürle tutarlı: %15-20 gray sheep beklentisi
        threshold = np.percentile(user_distances, 80)

    gray_sheep_mask = user_distances > threshold

    return {
        'gray_sheep_mask': gray_sheep_mask,
        'user_distances': user_distances,
        'threshold': threshold,
        'gray_sheep_count': int(gray_sheep_mask.sum()),
        'gray_sheep_ratio': float(gray_sheep_mask.mean())
    }

# ============================================================
# BÖLÜM 5: ALGORİTMA LİSTESİ (BLACKLIST + ÖZEL PARAMETRELER)
# ============================================================

# Tamamen çıkarılacak algoritmalar
BLACKLIST = {
    # Çok yavaş (timeout alıyor)
    'LSHADEcnEpSin.OriginalLSHADEcnEpSin',
    'PSS.OriginalPSS',
    'ACOR.OriginalACOR',
    'FFA.OriginalFFA',
    'ALO.OriginalALO',
    'HCO.OriginalHCO',
    'SPBO.OriginalSPBO',
    # Tamamen sorunlu (düzeltilemez hata)
    'GSKA.OriginalGSKA',
    'MA.OriginalMA',
    'ESO.OriginalESO',
    'BFO.OriginalBFO',
    'DMOA.OriginalDMOA',
}

# Ekstra parametre gerektiren algoritmalar
# epoch ve pop_size dışında ek parametre lazım
def get_special_params(algo_name, epoch, pop_size):
    special = {
        'BCO.OriginalBCO'  : {'epoch': epoch, 'pop_size': pop_size,
                               'n_chemotaxis': 3},
        'IWO.OriginalIWO'  : {'epoch': epoch, 'pop_size': pop_size,
                               'seed_max': 5},
        'MA.OriginalMA'    : {'epoch': epoch, 'pop_size': pop_size,
                               'max_local_gens': 5},
        'BSO.OriginalBSO'  : {'epoch': epoch, 'pop_size': pop_size,
                               'm_clusters': 3},
        'CHIO.OriginalCHIO': {'epoch': epoch, 'pop_size': pop_size,
                               'max_age': 2},
        'SARO.OriginalSARO': {'epoch': epoch, 'pop_size': pop_size,
                               'mu': 5},
        'CEM.OriginalCEM'  : {'epoch': epoch, 'pop_size': pop_size,
                               'n_best': 5},
        'BSA.OriginalBSA'  : {'epoch': epoch, 'pop_size': pop_size,
                               'ff': 5},
        'EHO.OriginalEHO'  : {'epoch': epoch, 'pop_size': pop_size,
                               'n_clans': 3},
    }
    return special.get(algo_name, None)


def get_all_algorithms_v3():
    import mealpy
    import inspect
    import pkgutil

    algo_list = []

    for importer, modname, ispkg in pkgutil.walk_packages(
        path=mealpy.__path__,
        prefix=mealpy.__name__ + '.',
        onerror=lambda x: None
    ):
        try:
            module = __import__(modname, fromlist="dummy")
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (name.startswith('Original') and
                        hasattr(obj, 'solve') and
                        hasattr(obj, 'generate_population')):

                    short_name = modname.split('.')[-1]
                    full_name = f"{short_name}.{name}"

                    if full_name in BLACKLIST:
                        continue

                    algo_list.append({
                        'full_name': full_name,
                        'module': modname,
                        'class_name': name,
                        'class': obj
                    })
        except Exception:
            pass

    # Tekrarları kaldır
    seen = set()
    unique = []
    for a in algo_list:
        if a['full_name'] not in seen:
            seen.add(a['full_name'])
            unique.append(a)

    print(f"Toplam {len(unique)} algoritma (blacklist hariç)")

    # Swarm tabanlı olanları işaretle (bilgi amaçlı)
    SWARM_KEYWORDS = [
        'ABC', 'AGTO', 'ALO', 'AO', 'ARO', 'AVOA', 'BA', 'BES',
        'BSA', 'BeesA', 'COA', 'CSA', 'CSO', 'CoatiOA', 'DO',
        'EHO', 'ESOA', 'FA', 'FDO', 'FFO', 'FOA', 'FOX', 'GJO',
        'GOA', 'GTO', 'GWO', 'HBA', 'HGS', 'HHO', 'JA', 'MFO',
        'MGO', 'MPA', 'MRFO', 'MSA', 'NGO', 'NMRA', 'OOA', 'PFA',
        'POA', 'PSO', 'SCSO', 'SFO', 'SHO', 'SLO', 'SRSR', 'SSA',
        'SSO', 'SSpiderA', 'SSpiderO', 'STO', 'SeaHO', 'ServalOA',
        'SquirrelSA', 'TDO', 'TSO', 'WOA', 'WaOA', 'ZOA'
    ]
    swarm_count = sum(
        1 for a in unique
        if a['full_name'].split('.')[0] in SWARM_KEYWORDS
    )
    print(f"  Swarm tabanlı  : {swarm_count}")
    print(f"  Diğer          : {len(unique) - swarm_count}")

    return unique

# ============================================================
# BÖLÜM 6: TEK ALGORİTMA ÇALIŞTIR
# ============================================================

def _solve_worker_v3(algo_module: str, class_name: str, algo_name: str,
                      matrix: np.ndarray, K: int,
                      initial_solutions, epoch: int, pop_size: int,
                      out_queue) -> None:
    """
    Multiprocessing worker:
    - mealpy algoritmasını çalıştır
    - en iyi çözüm + kümeleme metriklerini hesapla
    - sonucu queue'a gönder
    """
    start = time.time()
    try:
        from mealpy import FloatVar
        import importlib

        n_items = matrix.shape[1]
        n_vars = K * n_items

        bounds = FloatVar(
            lb=[0.0] * n_vars,
            ub=[5.0] * n_vars,
            name="centroids"
        )

        fitness_fn = make_fitness_function(matrix, K)
        problem = {
            "obj_func": fitness_fn,
            "bounds": bounds,
            "minmax": "min",
            "log_to": None,
            "save_population": False
        }

        module = importlib.import_module(algo_module)
        AlgoClass = getattr(module, class_name)

        # Özel parametre var mı?
        special = get_special_params(algo_name, epoch, pop_size)
        if special:
            model = AlgoClass(**special)
        else:
            model = AlgoClass(epoch=epoch, pop_size=pop_size)

        # starting_solutions bazı sürümlerde TypeError çıkarabilir
        try:
            model.solve(
                problem,
                starting_solutions=initial_solutions[:pop_size]
            )
        except TypeError:
            model.solve(problem)

        # En iyi çözüm
        best_sol = model.g_best.solution
        best_fit = model.g_best.target.fitness

        # Kümeleme
        _, assignments = compute_wcss_fast(matrix, best_sol, K)
        gs_info = detect_gray_sheep(matrix, assignments, best_sol, K)

        white_mask = ~gs_info['gray_sheep_mask']
        white_matrix = matrix[white_mask]
        white_assign = assignments[white_mask]

        sil, db = -1.0, 999.0
        if len(np.unique(white_assign)) >= 2 and len(white_matrix) > 10:
            try:
                sil = silhouette_score(
                    white_matrix, white_assign,
                    metric='euclidean',
                    sample_size=min(500, len(white_matrix))
                )
                db = davies_bouldin_score(white_matrix, white_assign)
            except Exception:
                pass

        out_queue.put({
            'algorithm': algo_name,
            'wcss': float(best_fit),
            'silhouette': float(sil),
            'davies_bouldin': float(db),
            'gray_sheep_count': int(gs_info['gray_sheep_count']),
            'gray_sheep_ratio': float(gs_info['gray_sheep_ratio']),
            'gray_sheep_threshold': float(gs_info['threshold']),
            'time_seconds': time.time() - start,
            'success': True,
            'error': None
        })
    except Exception as e:
        out_queue.put({
            'algorithm': algo_name,
            'wcss': None, 'silhouette': None,
            'davies_bouldin': None,
            'gray_sheep_count': None,
            'gray_sheep_ratio': None,
            'gray_sheep_threshold': None,
            'time_seconds': time.time() - start,
            'success': False,
            'error': str(e)
        })

def run_algorithm_v3(algo_info, matrix, K,
                     initial_solutions, epoch, pop_size,
                     time_limit=None):
    start = time.time()

    try:
        AlgoClass = algo_info['class']
        algo_name = algo_info['full_name']

        # Özel parametre var mı kontrol et
        special = get_special_params(algo_name, epoch, pop_size)
        if special:
            model = AlgoClass(**special)
        else:
            model = AlgoClass(epoch=epoch, pop_size=pop_size)

        def solve_model():
            """
            Direkt (timeout'suz) çözüm yolu.
            time_limit != None ise bu fonksiyon çağrılmayacağı için
            bounds/problem/fitness'i burada üretmek daha verimli olur.
            """
            from mealpy import FloatVar

            n_items = matrix.shape[1]
            n_vars = K * n_items

            bounds = FloatVar(
                lb=[0.0] * n_vars,
                ub=[5.0] * n_vars,
                name="centroids"
            )

            fitness_fn = make_fitness_function(matrix, K)
            problem = {
                "obj_func": fitness_fn,
                "bounds": bounds,
                "minmax": "min",
                "log_to": None,
                "save_population": False
            }

            # starting_solutions bazı mealpy sürümlerinde TypeError çıkarabilir;
            # o durumda sadece problem ile çöz.
            try:
                model.solve(
                    problem,
                    starting_solutions=initial_solutions[:pop_size]
                )
            except TypeError:
                model.solve(problem)

        # time_limit verilmişse threading ile timeout uygula
        if time_limit is not None:
            ctx = mp.get_context("spawn")
            q = ctx.Queue()

            proc = ctx.Process(
                target=_solve_worker_v3,
                args=(
                    algo_info['module'],
                    algo_info['class_name'],
                    algo_name,
                    matrix, K,
                    initial_solutions, epoch, pop_size,
                    q
                )
            )
            proc.daemon = True
            proc.start()
            proc.join(timeout=time_limit)

            if proc.is_alive():
                proc.terminate()
                proc.join()
                return {
                    'algorithm': algo_name,
                    'wcss': None, 'silhouette': None,
                    'davies_bouldin': None,
                    'gray_sheep_count': None,
                    'gray_sheep_ratio': None,
                    'gray_sheep_threshold': None,
                    'time_seconds': time_limit,
                    'success': False,
                    'error': f'TIMEOUT ({time_limit}s aşıldı)'
                }

            try:
                return q.get(timeout=1)
            except Exception:
                return {
                    'algorithm': algo_name,
                    'wcss': None, 'silhouette': None,
                    'davies_bouldin': None,
                    'gray_sheep_count': None,
                    'gray_sheep_ratio': None,
                    'gray_sheep_threshold': None,
                    'time_seconds': time_limit,
                    'success': False,
                    'error': 'NO_RESULT_FROM_WORKER'
                }
        else:
            # timeout istemiyorsan, doğrudan çalıştır.
            solve_model()

        elapsed = time.time() - start

        # Sonuçları al
        best_sol = model.g_best.solution
        best_fit = model.g_best.target.fitness

        # Kümeleme kalitesi
        _, assignments = compute_wcss_fast(matrix, best_sol, K)
        gs_info = detect_gray_sheep(matrix, assignments, best_sol, K)

        white_mask = ~gs_info['gray_sheep_mask']
        white_matrix = matrix[white_mask]
        white_assign = assignments[white_mask]

        sil, db = -1.0, 999.0
        if len(np.unique(white_assign)) >= 2 and len(white_matrix) > 10:
            try:
                sil = silhouette_score(
                    white_matrix, white_assign,
                    metric='euclidean',
                    sample_size=min(500, len(white_matrix))
                )
                db = davies_bouldin_score(white_matrix, white_assign)
            except Exception:
                pass

        return {
            'algorithm': algo_name,
            'wcss': float(best_fit),
            'silhouette': sil,
            'davies_bouldin': db,
            'gray_sheep_count': int(gs_info['gray_sheep_count']),
            'gray_sheep_ratio': float(gs_info['gray_sheep_ratio']),
            'gray_sheep_threshold': float(gs_info['threshold']),
            'time_seconds': elapsed,
            'success': True,
            'error': None
        }

    except Exception as e:
        return {
            'algorithm': algo_info['full_name'],
            'wcss': None, 'silhouette': None,
            'davies_bouldin': None,
            'gray_sheep_count': None,
            'gray_sheep_ratio': None,
            'gray_sheep_threshold': None,
            'time_seconds': time.time() - start,
            'success': False,
            'error': str(e)
        }

# ============================================================
# BÖLÜM 7: AŞAMA ÇALIŞTIR
# ============================================================

def run_phase(phase_num, algo_list, matrix, K,
              initial_solutions, epoch, pop_size,
              time_limit=None,
              save_path=RESULTS_DIR):

    os.makedirs(save_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"AŞAMA {phase_num}: {len(algo_list)} algoritma")
    print(f"Matrix: {matrix.shape} | K={K} | "
          f"epoch={epoch} | pop={pop_size}")
    print(f"{'='*60}")

    results = []  # ← kritik

    for i, algo in enumerate(algo_list):
        print(f"\n[{i+1}/{len(algo_list)}] {algo['full_name']}...",
              end=' ', flush=True)

        result = run_algorithm_v3(
            algo, matrix, K,
            initial_solutions, epoch, pop_size,
            time_limit=time_limit
        )
        results.append(result)

        if result['success']:
            print(f"✓ WCSS={result['wcss']:.1f} | "
                  f"Sil={result['silhouette']:.3f} | "
                  f"GS={result['gray_sheep_ratio']:.1%} | "
                  f"{result['time_seconds']:.1f}s")
        else:
            print(f"✗ {result['error']}")

        # Her 10 algoritmada ara kayıt
        if (i + 1) % 10 == 0:
            pd.DataFrame(results).to_csv(
                f'{save_path}/phase{phase_num}_partial.csv',
                index=False
            )

    df = pd.DataFrame(results)
    df.to_csv(f'{save_path}/phase{phase_num}_complete.csv', index=False)

    success_df = df[df['success'] == True]
    failed_df  = df[df['success'] == False]

    success_df.to_csv(f'{save_path}/phase{phase_num}_success.csv', index=False)
    failed_df.to_csv(f'{save_path}/phase{phase_num}_failed.csv', index=False)

    print(f"\n{'='*60}")
    print(f"AŞAMA {phase_num} ÖZET:")
    print(f"  Başarılı : {len(success_df)}")
    print(f"  Başarısız: {len(failed_df)}")

    if len(failed_df) > 0:
        print(f"\nBAŞARISIZ ALGORİTMALAR:")
        for _, row in failed_df.iterrows():
            print(f"  {row['algorithm']:<35} → {row['error']}")

    return df

# ============================================================
# BÖLÜM 8: SIRALAMA VE FİLTRELEME
# ============================================================

def rank_and_filter(df, top_n=20):
    successful = df[df['success'] == True].copy()

    print(f"\nBaşarılı: {len(successful)} / {len(df)}")

    if len(successful) == 0:
        print("Hiç başarılı algoritma yok!")
        return successful

    def norm_min(col):  # düşük iyi → yüksek skor
        r = successful[col].max() - successful[col].min()
        if r == 0:
            return pd.Series(0.5, index=successful.index)
        return 1 - (successful[col] - successful[col].min()) / r

    def norm_max(col):  # yüksek iyi → yüksek skor
        r = successful[col].max() - successful[col].min()
        if r == 0:
            return pd.Series(0.5, index=successful.index)
        return (successful[col] - successful[col].min()) / r

    # Her metriği 0-1 arasına normalize et
    successful['score_wcss']       = norm_min('wcss')
    successful['score_sil']        = norm_max('silhouette')
    successful['score_db']         = norm_min('davies_bouldin')
    successful['score_time']       = norm_min('time_seconds')

    # gray_fixed_ratio sütunu varsa kullan, yoksa gray_sheep_ratio
    if 'gray_fixed_ratio' in successful.columns:
        successful['score_gray'] = norm_min('gray_fixed_ratio')
    else:
        successful['score_gray'] = norm_min('gray_sheep_ratio')

    # ── AĞIRLIKLAR ──────────────────────────────────────────
    # Toplam = 1.0
    W_WCSS  = 0.30   # küme sıkılığı, ana görev
    W_SIL   = 0.30   # küme ayrışması, öneri kalitesi
    W_DB    = 0.15   # ek kalite teyidi
    W_GRAY  = 0.20   # gray sheep tespiti, tezin katkısı
    W_TIME  = 0.05   # pratik kullanılabilirlik
    # ─────────────────────────────────────────────────────────

    successful['composite'] = (
        W_WCSS  * successful['score_wcss']  +
        W_SIL   * successful['score_sil']   +
        W_DB    * successful['score_db']    +
        W_GRAY  * successful['score_gray']  +
        W_TIME  * successful['score_time']
    )

    ranked = successful.sort_values('composite', ascending=False)

    # Detaylı tablo
    print(f"\nTOP {min(top_n, len(ranked))} ALGORİTMA:")
    print(f"{'Algoritma':<35} {'WCSS':>7} {'Sil':>6} "
          f"{'DB':>6} {'GS%':>6} {'Süre':>6} {'Score':>7}")
    print(f"{'Ağırlık':<35} {'%30':>7} {'%30':>6} "
          f"{'%15':>6} {'%20':>6} {'%5':>6} {'':>7}")
    print("-" * 80)

    for _, row in ranked.head(top_n).iterrows():
        
        # Gray sheep sütunu hangisi varsa
        gs_val = row.get('gray_fixed_ratio', row.get('gray_sheep_ratio', 0))
        
        print(f"{row['algorithm']:<35} "
              f"{row['wcss']:>7.1f} "
              f"{row['silhouette']:>6.3f} "
              f"{row['davies_bouldin']:>6.3f} "
              f"{gs_val:>5.1%} "
              f"{row['time_seconds']:>6.1f} "
              f"{row['composite']:>7.3f}")

    # Skor bileşenlerini de kaydet
    score_cols = ['algorithm', 'composite',
                  'score_wcss', 'score_sil', 'score_db',
                  'score_gray', 'score_time',
                  'wcss', 'silhouette', 'davies_bouldin',
                  'gray_sheep_ratio', 'time_seconds']
    
    available = [c for c in score_cols if c in ranked.columns]
    ranked[available].to_csv(
        os.path.join(RESULTS_DIR, 'ranked_scores.csv'),
        index=False
    )

    return ranked.head(top_n)

# ============================================================
# BÖLÜM 9: ANA ÇALIŞTIRMA
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AŞAMALI ALGORİTMA TARAMA SİSTEMİ")
    print("Mealpy 3.x | MovieLens 100K | Gray Sheep Tespiti")
    print("=" * 60)

    # Veri yükle — yolu kendi klasörüne göre ayarla
    #DATA_PATH = os.path.join(BASE_DIR, 'data', 'movielens_100k', 'u.data')
    full_matrix = load_movielens('data/movielens_100k/u.data')

    # Tüm algoritmaları al
    all_algos = get_all_algorithms_v3()

    # ─────────────────────────────────────────────
    # AŞAMA 1: Hızlı eleme — küçük matris
    # Tüm algoritmalar, düşük parametre
    # ─────────────────────────────────────────────
    print("\n>>> AŞAMA 1 BAŞLIYOR")
    matrix_small = sample_matrix(full_matrix, n_users=150, n_items=200)
    K1 = 30
    K2 = 60
    K3 = 90

    init_small = mkmeans_plus_plus_init(matrix_small, K=K1, n_solutions=20)

    df1 = run_phase(
        phase_num=1,
        algo_list=all_algos,
        matrix=matrix_small,
        K=K1,
        initial_solutions=init_small,
        epoch=30,     # 15 → 30
        pop_size=20,  # 15 → 20
        time_limit=90
    )

    top20 = rank_and_filter(df1, top_n=20)

    # Aşama 1'de mutlaka olmasını istediğimiz algoritmalar
    # (tez için önemli, elenmişse geri ekle)
    MUST_INCLUDE = {
        'SSA.OriginalSSA',   # HSC makalesi
        'WOA.OriginalWOA',   # HSC makalesi
        'GWO.OriginalGWO',   # CF literatürü
        'HHO.OriginalHHO',   # önerimiz
        'PSO.OriginalPSO',   # CF literatürü
    }

    top20_names = set(top20['algorithm'].tolist())

    # Eksik olanları ekle
    for name in MUST_INCLUDE:
        if name not in top20_names:
            print(f"  Zorla eklendi: {name}")
            top20_names.add(name)

    algos_phase2 = [a for a in all_algos if a['full_name'] in top20_names]

    # ─────────────────────────────────────────────
    # AŞAMA 2: Orta matris — top 20 + zorunlu
    # ─────────────────────────────────────────────
    print("\n>>> AŞAMA 2 BAŞLIYOR")
    matrix_mid = sample_matrix(full_matrix, n_users=300, n_items=400)

    init_mid = mkmeans_plus_plus_init(matrix_mid, K=K2, n_solutions=30)

    df2 = run_phase(
        phase_num=2,
        algo_list=algos_phase2,
        matrix=matrix_mid,
        K=K2,
        initial_solutions=init_mid,
        epoch=50,     # 25 → 50
        pop_size=30,  # 20 → 30
        time_limit=120
    )

    top5 = rank_and_filter(df2, top_n=5)

    top5_names = set(top5['algorithm'].tolist())
    PHASE3_MUST = {
        'MFO.OriginalMFO', 'SA.OriginalSA',
        'SSA.OriginalSSA', 'GWO.OriginalGWO',
        'WOA.OriginalWOA'
    }

    algos_phase3 = [
        a for a in all_algos
        if a['full_name'] in PHASE3_MUST
    ]

    # ─────────────────────────────────────────────
    # AŞAMA 3: Tam matris — top 5
    # ─────────────────────────────────────────────
    print("\n>>> AŞAMA 3 BAŞLIYOR")

    init_full = mkmeans_plus_plus_init(full_matrix, K=K3, n_solutions=50)

    df3 = run_phase(
        phase_num=3,
        algo_list=algos_phase3,
        matrix=full_matrix,
        K=90,           # 30 → 90
        initial_solutions=init_full,
        epoch=50,       # 100 → 50 (K büyüdü, epoch düştü)
        pop_size=30,    # 50 → 30
        time_limit=900  # 15 dakika
    )

    final = rank_and_filter(df3, top_n=5)

    # ─────────────────────────────────────────────
    # SONUÇ
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEZİN İÇİN ÖNERİLEN ALGORİTMALAR:")
    print("=" * 60)

    for i, (_, row) in enumerate(final.iterrows()):
        print(f"\n{i+1}. {row['algorithm']}")
        print(f"   Composite Score : {row['composite']:.3f}")
        print(f"   WCSS            : {row['wcss']:.1f}")
        print(f"   Silhouette      : {row['silhouette']:.3f}")
        print(f"   Davies-Bouldin  : {row['davies_bouldin']:.3f}")
        print(f"   Gray Sheep Oranı: {row['gray_sheep_ratio']:.1%}")
        print(f"   Süre            : {row['time_seconds']:.1f}s")

    final.to_csv(
        os.path.join(RESULTS_DIR, 'FINAL_RECOMMENDATIONS.csv'),
        index=False
    )
    print(f"\nSonuçlar {RESULTS_DIR} klasörüne kaydedildi.")