"""
phase5_ml1m.py
MovieLens 1M Ölçeklenebilirlik Testi

Aşama 3'ten seçilen 5 algoritma (HGS, MFO, WOA, OOA, AGTO)
ML-1M üzerinde K=150, epoch=30, pop=20 ile test edilir.
WCSS ve DB karşılaştırması yapılır.

Kullanım:
  python phase5_ml1m.py

Veri: data/ml-1m/ratings.dat  (pipe-separated)
Çıktı: results/ml1m/
"""

import numpy as np
import pandas as pd
import time
import os
import warnings
warnings.filterwarnings('ignore')

from mealpy.swarm_based.HGS  import OriginalHGS
from mealpy.swarm_based.MFO  import OriginalMFO
from mealpy.swarm_based.WOA  import OriginalWOA
from mealpy.swarm_based.OOA  import OriginalOOA
from mealpy.swarm_based.AGTO import OriginalAGTO

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mealpy_comparison_v2 import mkmeans_plus_plus_init

# ============================================================
# KLASÖR AYARI
# ============================================================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'ml1m')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# BÖLÜM 1: ML-1M VERİ YÜKLEME
# ============================================================

def load_movielens_1m(path: str) -> np.ndarray:
    """
    ML-1M ratings.dat dosyasını yükle.
    Format: UserID::MovieID::Rating::Timestamp
    """
    df = pd.read_csv(
        path, sep='::', engine='python',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        dtype={'user_id': int, 'item_id': int,
               'rating': float, 'timestamp': int}
    )

    matrix = df.pivot_table(
        index='user_id', columns='item_id',
        values='rating', fill_value=0
    ).values.astype(np.float32)

    total   = matrix.size
    nonzero = np.count_nonzero(matrix)
    print(f"ML-1M Matrix shape : {matrix.shape}")
    print(f"ML-1M Sparsity     : {1 - nonzero/total:.3f}")
    print(f"ML-1M Rating range : {matrix[matrix>0].min():.1f}"
          f" - {matrix.max():.1f}")
    return matrix


# ============================================================
# BÖLÜM 2: TEK ALGORİTMA ÇALIŞTIR (timeout'suz, direkt)
# ============================================================

def _ml1m_single_worker(out_q, name, module_name, class_name, matrix, K,
                        initial_solutions, epoch, pop_size):
    """
    Modül düzeyinde tanımlı olmalı: Windows spawn alt süreci nested
    fonksiyonları pickle edemez.
    """
    import importlib
    import time

    from mealpy import FloatVar
    from mealpy_comparison_v2 import make_fitness_function, _compute_metrics

    t0 = time.time()
    try:
        AlgoClass = getattr(importlib.import_module(module_name), class_name)
        n_items = matrix.shape[1]
        problem = {
            "obj_func"       : make_fitness_function(matrix, K),
            "bounds"         : FloatVar(lb=[0.0] * (K * n_items),
                                        ub=[5.0] * (K * n_items)),
            "minmax"         : "min",
            "log_to"         : None,
            "save_population": False,
        }
        model = AlgoClass(epoch=epoch, pop_size=pop_size)
        try:
            model.solve(problem,
                        starting_solutions=initial_solutions[:pop_size])
        except TypeError:
            model.solve(problem)

        metrics = _compute_metrics(matrix, model.g_best.solution, K)
        out_q.put({
            'name'        : name,
            'wcss'        : float(model.g_best.target.fitness),
            'success'     : True,
            'error'       : None,
            'time_seconds': time.time() - t0,
            **metrics,
        })
    except Exception as e:
        out_q.put({
            'name': name, 'wcss': None,
            'silhouette': None, 'davies_bouldin': None,
            'gray_sheep_count': None, 'gray_sheep_ratio': None,
            'gray_sheep_threshold': None,
            'time_seconds': time.time() - t0,
            'success': False, 'error': str(e),
        })


def run_single(name, algo_class, matrix, K,
               initial_solutions, epoch, pop_size,
               time_limit=None):
    """
    ML-1M boyutunda algoritma çalıştır.
    time_limit (saniye) verilirse timeout uygulanır.
    """
    import multiprocessing as mp
    import queue as queue_mod

    mod_name = algo_class.__module__
    cls_name = algo_class.__name__

    if time_limit is not None:
        ctx  = mp.get_context("spawn")
        q    = ctx.Queue()
        proc = ctx.Process(
            target=_ml1m_single_worker,
            args=(q, name, mod_name, cls_name, matrix, K,
                  initial_solutions, epoch, pop_size),
        )
        proc.daemon = True
        proc.start()
        proc.join(timeout=time_limit)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            return {
                'name': name, 'wcss': None,
                'silhouette': None, 'davies_bouldin': None,
                'gray_sheep_count': None, 'gray_sheep_ratio': None,
                'gray_sheep_threshold': None,
                'time_seconds': time_limit,
                'success': False,
                'error': f'TIMEOUT ({time_limit}s)',
            }
        try:
            return q.get(timeout=5)
        except Exception:
            return {
                'name': name, 'wcss': None,
                'silhouette': None, 'davies_bouldin': None,
                'gray_sheep_count': None, 'gray_sheep_ratio': None,
                'gray_sheep_threshold': None,
                'time_seconds': time_limit,
                'success': False,
                'error': 'NO_RESULT',
            }

    q = queue_mod.Queue()
    _ml1m_single_worker(q, name, mod_name, cls_name, matrix, K,
                        initial_solutions, epoch, pop_size)
    return q.get()


# ============================================================
# BÖLÜM 3: KARŞILAŞTIRMA TABLOSU
# ============================================================

def print_comparison(results_100k, results_1m):
    """ML-100K vs ML-1M yan yana karşılaştırma."""
    print("\n" + "="*75)
    print("ML-100K vs ML-1M KARŞILAŞTIRMA")
    print("="*75)
    print(f"{'Algoritma':<8} {'100K WCSS':>10} {'1M WCSS':>10} "
          f"{'100K DB':>9} {'1M DB':>9} {'1M Süre':>9}")
    print("-"*75)

    for name in results_100k:
        r1 = results_100k[name]
        r2 = results_1m.get(name, {})
        w1 = r1.get('wcss', float('nan'))
        w2 = r2.get('wcss', float('nan')) if r2.get('success') else float('nan')
        d1 = r1.get('davies_bouldin', float('nan'))
        d2 = r2.get('davies_bouldin', float('nan')) if r2.get('success') else float('nan')
        t2 = r2.get('time_seconds', float('nan'))

        w2s = f"{w2:>10.2f}" if not (isinstance(w2, float) and np.isnan(w2)) else "   TIMEOUT"
        d2s = f"{d2:>9.3f}" if not (isinstance(d2, float) and np.isnan(d2)) else "  TIMEOUT"
        t2s = f"{t2:>8.1f}s" if not (isinstance(t2, float) and np.isnan(t2)) else "       -"

        print(f"{name:<8} {w1:>10.2f} {w2s} {d1:>9.3f} {d2s} {t2s}")

    # Ölçeklenebilirlik skoru: her iki veri setinde de iyi olan
    print("\nÖlçeklenebilirlik Yorumu:")
    for name in results_100k:
        r2 = results_1m.get(name, {})
        if not r2.get('success', False):
            print(f"  {name}: TIMEOUT veya HATA")
        else:
            db_diff = r2['davies_bouldin'] - results_100k[name]['davies_bouldin']
            trend = "↑ kötüleşti" if db_diff > 0.2 else \
                    "↓ iyileşti" if db_diff < -0.2 else "→ stabil"
            print(f"  {name}: DB değişimi {db_diff:+.3f}  {trend}")


# ============================================================
# BÖLÜM 4: ANA ÇALIŞTIRMA
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ML-1M ÖLÇEKLENEBİLİRLİK TESTİ")
    print("Top-5 Algoritma | K=150 | epoch=30 | pop=20")
    print("=" * 60)

    # ── Parametreler ─────────────────────────────────────────
    K         = 150    # ML-1M için K artırıldı (6040 kullanıcı → daha fazla küme)
    EPOCH     = 30     # süre kısıtı nedeniyle azaltıldı
    POP_SIZE  = 20     # süre kısıtı nedeniyle azaltıldı
    TIME_LIMIT = 1800  # 30 dakika per algoritma

    # ── Veri yükle ───────────────────────────────────────────
    DATA_PATH = os.path.join(os.path.dirname(BASE_DIR),
                             'data', 'ml-1m', 'ratings.dat')
    matrix_1m = load_movielens_1m(DATA_PATH)

    # ── MkMeans++ başlangıç ─────────────────────────────────
    print("\nMkMeans++ başlangıç popülasyonu hazırlanıyor...")
    print("(ML-1M için bu adım birkaç dakika sürebilir)")
    init_solutions = mkmeans_plus_plus_init(
        matrix_1m, K=K, n_solutions=POP_SIZE + 5, seed=42
    )

    # ── ML-100K referans değerleri ───────────────────────────
    # Önceki Aşama 3 sonuçlarından
    results_100k = {
        "HGS" : {"wcss": 678.30, "davies_bouldin": 3.511, "time_seconds": 56.0},
        "MFO" : {"wcss": 679.47, "davies_bouldin": 3.527, "time_seconds": 59.2},
        "WOA" : {"wcss": 677.45, "davies_bouldin": 3.573, "time_seconds": 60.9},
        "OOA" : {"wcss": 683.66, "davies_bouldin": 3.513, "time_seconds": 126.1},
        "AGTO": {"wcss": 683.23, "davies_bouldin": 3.524, "time_seconds": 126.9},
    }

    # ── Test edilecek algoritmalar ───────────────────────────
    algorithms = [
        ("HGS",  OriginalHGS),
        ("MFO",  OriginalMFO),
        ("WOA",  OriginalWOA),
        ("OOA",  OriginalOOA),
        ("AGTO", OriginalAGTO),
    ]

    results_1m = {}
    all_rows   = []

    print(f"\nTest başlıyor: {len(algorithms)} algoritma")
    print(f"K={K} | epoch={EPOCH} | pop={POP_SIZE} | "
          f"time_limit={TIME_LIMIT//60}dk")
    print("-" * 60)

    for i, (name, algo_class) in enumerate(algorithms, 1):
        print(f"\n[{i}/{len(algorithms)}] {name}...", flush=True)
        result = run_single(
            name, algo_class, matrix_1m, K,
            init_solutions, EPOCH, POP_SIZE,
            time_limit=TIME_LIMIT,
        )
        results_1m[name] = result

        if result['success']:
            print(f"  ✓ WCSS={result['wcss']:.2f} | "
                  f"DB={result['davies_bouldin']:.3f} | "
                  f"GS={result['gray_sheep_ratio']:.1%} | "
                  f"{result['time_seconds']:.1f}s")
        else:
            print(f"  ✗ {result['error']}")

        all_rows.append({
            'dataset'             : 'ML-1M',
            'algorithm'           : name,
            'wcss'                : result.get('wcss'),
            'silhouette'          : result.get('silhouette'),
            'davies_bouldin'      : result.get('davies_bouldin'),
            'gray_sheep_ratio'    : result.get('gray_sheep_ratio'),
            'time_seconds'        : result.get('time_seconds'),
            'K'                   : K,
            'epoch'               : EPOCH,
            'pop_size'            : POP_SIZE,
            'success'             : result.get('success'),
            'error'               : result.get('error'),
        })

    # ── ML-100K referans satırları da ekle ───────────────────
    for name, r in results_100k.items():
        all_rows.append({
            'dataset'          : 'ML-100K',
            'algorithm'        : name,
            'wcss'             : r['wcss'],
            'silhouette'       : None,
            'davies_bouldin'   : r['davies_bouldin'],
            'gray_sheep_ratio' : None,
            'time_seconds'     : r['time_seconds'],
            'K'                : 90,
            'epoch'            : 50,
            'pop_size'         : 30,
            'success'          : True,
            'error'            : None,
        })

    # ── Kaydet ───────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'ml1m_comparison.csv'), index=False)

    # ── Karşılaştırma tablosu ────────────────────────────────
    print_comparison(results_100k, results_1m)

    # ── Normalize karşılaştırma (WCSS farklı ölçekte) ────────
    suc = {k: v for k, v in results_1m.items() if v.get('success')}
    if suc:
        print("\n" + "="*60)
        print("DB SIRALAMA TUTARLILIĞI (ölçekten bağımsız)")
        print("="*60)
        print(f"{'Algoritma':<8} {'100K DB sıra':>13} {'1M DB sıra':>11} {'Tutarlı?':>10}")
        print("-"*45)

        db_100k = sorted(results_100k.items(), key=lambda x: x[1]['davies_bouldin'])
        db_1m   = sorted(suc.items(),          key=lambda x: x[1]['davies_bouldin'])

        rank_100k = {name: i+1 for i, (name, _) in enumerate(db_100k)}
        rank_1m   = {name: i+1 for i, (name, _) in enumerate(db_1m)}

        for name in results_100k:
            r1 = rank_100k.get(name, '-')
            r2 = rank_1m.get(name, 'TIMEOUT')
            if isinstance(r1, int) and isinstance(r2, int):
                diff = abs(r1 - r2)
                flag = "✓" if diff <= 1 else "△" if diff == 2 else "✗"
            else:
                flag = "—"
            print(f"{name:<8} {str(r1):>13} {str(r2):>11} {flag:>10}")

    print(f"\nSonuçlar: {RESULTS_DIR}/ml1m_comparison.csv")