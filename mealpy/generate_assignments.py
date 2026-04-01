"""
generate_assignments.py
=======================
4 algoritma (B1_HHO, B2_HGS, H1_HHO+HGS, H4_MFO+HHO) için
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
    python generate_assignments.py --k-100k 70 --k-1m 120     # ayrı K
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
sys.path.insert(0, BASE_DIR)
_OPT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'optimizers')
if _OPT_DIR not in sys.path:
    sys.path.insert(0, _OPT_DIR)

from mealpy_comparison_v2 import (
    load_movielens,
    mkmeans_plus_plus_init,
    make_fitness_function,
    compute_wcss_fast,
    detect_gray_sheep,
    get_all_algorithms_v3,
    get_special_params,
)
from mealpy import FloatVar
from mealpy.evolutionary_based import GA
try:
    from ga_hho_optimizer import OriginalGAHHO  # pyright: ignore[reportMissingImports]
except ImportError:
    from ga_hho import OriginalGAHHO  # pyright: ignore[reportMissingImports]

# ============================================================
# AYARLAR
# ============================================================

DATA_100K = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-100k', 'u.data')
DATA_1M   = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-1m',   'ratings.dat')

K_100K_DEFAULT = 90
K_1M_DEFAULT   = 150

GLOBAL_EPOCH   = 30
LOCAL_EPOCH    = 20
BASELINE_EPOCH = 50
POP_SIZE       = 30
SEED           = 42

LOF_N_NEIGHBORS   = 20
LOF_CONTAMINATION = 'auto'

ALGO_CONFIG = [
    ('B1_HHO',     'HHO.OriginalHHO', None),
    ('B2_HGS',     'HGS.OriginalHGS', None),
    ('B3_MFO',     'MFO.OriginalMFO', None),
    ('H1_HHO+HGS', 'HHO.OriginalHHO', 'HGS.OriginalHGS'),
    ('H4_MFO+HHO', 'MFO.OriginalMFO', 'HHO.OriginalHHO'),
    ('H5_GAHHO',   'GAHHO.OriginalGAHHO', None),
    ('H5_EliteGA+HHO', 'GA.EliteMultiGA', 'HHO.OriginalHHO'),
]

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

def detect_gray_sheep_percentile(matrix, assignments, solution, K):
    """
    Sabit 80. percentile ile gray sheep tespiti.
    Her zaman ~%20 gray sheep üretir.
    Orijinal detect_gray_sheep fonksiyonunu çağırır.
    """
    return detect_gray_sheep(matrix, assignments, solution, K)


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

def _make_problem(matrix, K):
    n_items = matrix.shape[1]
    return {
        "obj_func"       : make_fitness_function(matrix, K),
        "bounds"         : FloatVar(
                               lb=[0.0] * (K * n_items),
                               ub=[5.0] * (K * n_items),
                               name="centroids"
                           ),
        "minmax"         : "min",
        "log_to"         : None,
        "save_population": False,
    }


def run_single(algo_info, matrix, K, init, epoch, pop_size):
    problem = _make_problem(matrix, K)
    sp      = get_special_params(algo_info['full_name'], epoch, pop_size)
    model   = algo_info['class'](**(sp or {'epoch': epoch, 'pop_size': pop_size}))
    try:
        model.solve(problem, starting_solutions=init[:pop_size])
    except TypeError:
        model.solve(problem)
    best_fit = float(model.g_best.target.fitness)
    print(f"    {algo_info['full_name'].split('.')[0]} WCSS: {best_fit:.4f}")
    return model.g_best.solution, best_fit


def run_hybrid(g_info, l_info, matrix, K, init, g_epoch, l_epoch, pop_size):
    problem = _make_problem(matrix, K)

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
    print(f"    Lokal iyileştirdi: {improved}  →  Final WCSS: {best_fit:.4f}")
    return best_sol, best_fit


# ============================================================
# KAYDET
# ============================================================

def save_assignment(assignments, gray_mask, best_sol, best_fit,
                    save_dir, extra_data=None):
    """
    Assignment dosyalarını kaydet.
    extra_data: LOF modunda lof_scores gibi ek veriler dict olarak geçilir.
    """
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, 'assignments.npy'),     assignments)
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

    print(f"    ✓ Kaydedildi → {save_dir}")
    print(f"      WCSS          : {best_fit:.4f}")
    print(f"      Kullanıcı     : {len(assignments)}")
    print(f"      Gray sheep    : {gray_mask.sum()} ({gray_mask.mean()*100:.1f}%)")
    print(f"      GS threshold  : {threshold:.4f}" if isinstance(threshold, float)
          else f"      GS threshold  : {threshold}")
    print(f"      Küme boyutu   : min={active.min()}, "
          f"max={active.max()}, ort={active.mean():.1f}")


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
):
    """Ortak gövde: algo_map ana süreçte veya worker’da bir kez oluşturulur."""
    mode_str = 'LOF' if use_lof else 'percentile'
    print(f"\n  [{label}] başlıyor (gray sheep: {mode_str})...")
    t0   = time.time()
    #init = mkmeans_plus_plus_init(matrix, K=K, n_solutions=50, seed=seed)
    init = _multi_start_init(matrix, K=K, pop_size=POP_SIZE, seed=seed, n_restarts=10)

    if l_name is None:
        best_sol, best_fit = run_single(
            algo_map[g_name], matrix, K, init, baseline_epoch, pop_size
        )
    else:
        best_sol, best_fit = run_hybrid(
            algo_map[g_name], algo_map[l_name],
            matrix, K, init, global_epoch, local_epoch, pop_size
        )

    _, assignments = compute_wcss_fast(matrix, best_sol, K)

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
        gs_info    = detect_gray_sheep_percentile(matrix, assignments, best_sol, K)
        gray_mask  = gs_info['gray_sheep_mask']
        extra_data = {'threshold': gs_info['threshold']}

    save_assignment(assignments, gray_mask, best_sol, best_fit,
                    save_dir, extra_data)
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
    ) = job
    algo_map = {a['full_name']: a for a in get_all_algorithms_v3()}
    algo_map['GAHHO.OriginalGAHHO'] = {
        'full_name': 'GAHHO.OriginalGAHHO',
        'class':     OriginalGAHHO,
    }
    algo_map['GA.EliteMultiGA'] = {
        'full_name': 'GA.EliteMultiGA',
        'class':     GA.EliteMultiGA,
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
    )


def run_one(label, g_name, l_name, matrix, K, seed, save_dir, algo_map,
            use_lof=False, lof_n_neighbors=LOF_N_NEIGHBORS,
            lof_contamination=LOF_CONTAMINATION):
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


# ============================================================
# DATASET ÇALIŞTIRICI
# ============================================================

def run_dataset(dataset_name, matrix, K, out_root, algo_filter=None,
                use_lof=False, lof_n_neighbors=LOF_N_NEIGHBORS,
                lof_contamination=LOF_CONTAMINATION,
                max_workers: Optional[int] = None):
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name.upper()}  |  K={K}  |  Seed={SEED}")
    mode_str = f'LOF (n={lof_n_neighbors}, cont={lof_contamination})' \
               if use_lof else 'percentile (80th)'
    print(f"Gray sheep  : {mode_str}")
    print(f"{'='*60}")

    # K=default için suffix yok, farklı K için _k{K} eklenir
    default_k = K_100K_DEFAULT if dataset_name == 'ml100k' else K_1M_DEFAULT
    k_suffix  = '' if K == default_k else f'_k{K}'

    jobs_meta = []
    for label, g_name, l_name in ALGO_CONFIG:
        if algo_filter and label not in algo_filter:
            print(f"  [{label}] atlandı (filtre)")
            continue
        save_dir = os.path.join(out_root, dataset_name, f"{label}{k_suffix}")
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
        algo_map['GA.EliteMultiGA'] = {
            'full_name': 'GA.EliteMultiGA',
            'class':     GA.EliteMultiGA,
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
            )
            for label, g_name, l_name, save_dir in jobs_meta
        ]
        with ProcessPoolExecutor(max_workers=nw) as pool:
            list(pool.map(_mp_run_assignment_job, jobs))

    print(f"\n{dataset_name.upper()} tamamlandı → {os.path.join(out_root, dataset_name)}/")


# ============================================================
# CLI
# ============================================================

def parse_args():
    labels = [c[0] for c in ALGO_CONFIG]
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
        '--k', type=int, default=None,
        help="Her iki dataset için K (default: 100K=90, 1M=150)"
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
        help='Paralel algoritma süreç sayısı (varsayılan: 1 = sıralı; '
             '2+ veya 0=CPU ile sınırlandırılmış havuz)',
    )
    return p.parse_args()

def _multi_start_init(matrix, K, pop_size, seed, n_restarts=10):
    """
    n_restarts farklı seed ile MkMeans++ çalıştır.
    Her seferinde 1 çözüm üret, WCSS hesapla.
    En iyi pop_size kadar çözümü döndür.
    """
    candidates = []
    for i in range(n_restarts * 3):   # fazla üret, en iyileri seç
        sols = mkmeans_plus_plus_init(
            matrix, K=K, n_solutions=1, seed=seed + i * 17
        )
        sol = sols[0]
        wcss, _ = compute_wcss_fast(matrix, sol, K)
        candidates.append((wcss, sol))

    # WCSS'e göre sırala, en iyi pop_size tanesini al
    candidates.sort(key=lambda x: x[0])
    best = [c[1] for c in candidates[:pop_size + 10]]
    print(f"    Init: {n_restarts*3} aday → en iyi {pop_size} seçildi  "
          f"(WCSS aralığı: {candidates[0][0]:.1f} – {candidates[pop_size-1][0]:.1f})")
    return best
# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    args = parse_args()
    selected_algos = [ALGO_CONFIG[-1][0]] if args.last_only else args.algo

    # Contamination dönüşümü
    contamination = args.contamination
    if contamination != 'auto':
        contamination = float(contamination)

    # K değerleri
    k_100k = args.k_100k or args.k or K_100K_DEFAULT
    k_1m   = args.k_1m   or args.k or K_1M_DEFAULT

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
    print(f"ML-100K K   : {k_100k}  |  ML-1M K: {k_1m}")
    if args.lof:
        print(f"LOF         : n_neighbors={args.n_neighbors}, contamination={contamination}")
    print(f"Epoch       : baseline={BASELINE_EPOCH}, global={GLOBAL_EPOCH}, local={LOCAL_EPOCH}")
    print(f"Pop size    : {POP_SIZE}  |  Seed: {SEED}")
    print(f"Çıktı kökü  : {out_root}")
    if args.jobs is None or args.jobs == 1:
        print("Paralellik    : sıralı (tek süreç)")
    elif args.jobs <= 0:
        print("Paralellik    : süreç havuzu (iş ve CPU sayısına göre)")
    else:
        print(f"Paralellik    : en fazla {args.jobs} süreç")
    print("=" * 60)

    t_total = time.time()

    def _pool_cap():
        if args.jobs is None or args.jobs == 1:
            return 1
        if args.jobs <= 0:
            return None
        return args.jobs

    if args.dataset in ('100k', 'both'):
        print(f"\nML-100K yükleniyor: {args.data_100k}")
        matrix_100k = load_movielens(args.data_100k)
        run_dataset(
            'ml100k', matrix_100k, k_100k, out_root,
            algo_filter=selected_algos,
            use_lof=args.lof,
            lof_n_neighbors=args.n_neighbors,
            lof_contamination=contamination,
            max_workers=_pool_cap(),
        )

    if args.dataset in ('1m', 'both'):
        print(f"\nML-1M yükleniyor: {args.data_1m}")
        matrix_1m = load_movielens_1m(args.data_1m)
        run_dataset(
            'ml1m', matrix_1m, k_1m, out_root,
            algo_filter=selected_algos,
            use_lof=args.lof,
            lof_n_neighbors=args.n_neighbors,
            lof_contamination=contamination,
            max_workers=_pool_cap(),
        )

    print(f"\n{'='*60}")
    print(f"TAMAMLANDI — toplam {(time.time()-t_total)/60:.1f} dakika")
    print(f"Çıktı: {out_root}/")
    print("=" * 60)