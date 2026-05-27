"""Elbow + Silhouette analizi ile optimal K seçimi (WNMF30 pipeline)."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
MEALPY_DIR = os.path.join(REPO_ROOT, 'mealpy')
if MEALPY_DIR not in sys.path:
    sys.path.insert(0, MEALPY_DIR)

from generate_assignments import prune_sparse_matrix, wnmf_feature_extract  # noqa: E402
from mealpy_comparison_v2 import load_movielens  # noqa: E402

K_LIST = [5, 10, 15, 20, 25, 30, 40, 50, 70, 90]
K_LIST_REFINE_20_40 = list(range(20, 41))
RESULTS_DIR = os.path.join(REPO_ROOT, 'results', 'elbow')


def _manual_elbow(k_list, wcss):
    """Ardışık WCSS düşüşlerinin en çok yavaşladığı noktayı bul."""
    if len(k_list) < 3:
        return k_list[0] if k_list else None
    drops = np.diff(wcss) * -1.0
    slowdowns = np.diff(drops)
    idx = int(np.argmax(slowdowns)) + 1
    return k_list[idx]


def _detect_elbow(k_list, wcss):
    try:
        from kneed import KneeLocator

        knee = KneeLocator(k_list, wcss, curve='convex', direction='decreasing', S=1.0)
        elbow_k = knee.elbow
        if elbow_k is not None:
            return elbow_k
    except ImportError:
        print('kneed kurulu değil; pip install kneed deneniyor...')
        import subprocess

        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kneed'])
        from kneed import KneeLocator

        knee = KneeLocator(k_list, wcss, curve='convex', direction='decreasing', S=1.0)
        elbow_k = knee.elbow
        if elbow_k is not None:
            return elbow_k
    except Exception as exc:
        print(f'kneed kullanılamadı ({exc}); manuel elbow hesaplanıyor.')

    return _manual_elbow(k_list, wcss)


def _prepare_features():
    data_path = os.path.join(REPO_ROOT, 'data', 'ml-100k', 'u.data')
    print(f'Veri yükleniyor: {data_path}')
    matrix = load_movielens(data_path)
    matrix = prune_sparse_matrix(matrix, min_user_ratings=5, min_item_ratings=10)
    U = wnmf_feature_extract(matrix, n_components=30, n_epochs=50, random_seed=42)
    return normalize(U, norm='l2')


def _scan_k_values(U_norm, k_list):
    wcss, sil, drops = [], [], []
    prev_wcss = None
    for K in k_list:
        km = KMeans(n_clusters=K, n_init=10, max_iter=300, random_state=42)
        km.fit(U_norm)
        inertia = float(km.inertia_)
        sil_score = float(
            silhouette_score(U_norm, km.labels_, sample_size=500, random_state=42)
        )
        drop = (prev_wcss - inertia) if prev_wcss is not None else float('nan')
        wcss.append(inertia)
        sil.append(sil_score)
        drops.append(drop)
        prev_wcss = inertia
        drop_txt = f'  dWCSS={drop:+.2f}' if not np.isnan(drop) else ''
        print(f'K={K:3d}  WCSS={inertia:.2f}  Silhouette={sil_score:.4f}{drop_txt}')
    return wcss, sil, drops


def _save_refine_plot(k_list, wcss, sil, elbow_k, best_sil_k, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(k_list, wcss, color='blue', marker='o', markersize=4)
    if elbow_k is not None:
        elbow_idx = k_list.index(elbow_k)
        ax1.scatter([elbow_k], [wcss[elbow_idx]], color='red', s=120, zorder=5)
        ax1.annotate(
            f'Elbow K={elbow_k}',
            (elbow_k, wcss[elbow_idx]),
            textcoords='offset points',
            xytext=(8, 8),
            color='red',
        )
    ax1.set_title('WCSS vs K (20-40 ince tarama)')
    ax1.set_xlabel('K (Küme Sayısı)')
    ax1.set_ylabel('WCSS')
    ax1.grid(True)

    ax2.plot(k_list, sil, color='green', marker='s', markersize=4)
    best_idx = k_list.index(best_sil_k)
    ax2.scatter([best_sil_k], [sil[best_idx]], color='red', s=120, zorder=5)
    ax2.annotate(
        f'Best K={best_sil_k}',
        (best_sil_k, sil[best_idx]),
        textcoords='offset points',
        xytext=(8, 8),
        color='red',
    )
    ax2.set_title('Silhouette vs K (20-40)')
    ax2.set_xlabel('K (Küme Sayısı)')
    ax2.set_ylabel('Silhouette Score')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def run_refined_elbow_20_40(U_norm=None):
    """K=20..40 aralığında adım=1 ince elbow taraması."""
    if U_norm is None:
        U_norm = _prepare_features()

    print('\n' + '=' * 60)
    print('İnce tarama: K = 20 .. 40 (adım 1)')
    print('=' * 60)

    k_list = K_LIST_REFINE_20_40
    wcss, sil, drops = _scan_k_values(U_norm, k_list)

    elbow_k = _detect_elbow(k_list, wcss)
    manual_k = _manual_elbow(k_list, wcss)
    best_sil_k = k_list[int(np.argmax(sil))]

    valid_drops = [(k_list[i], drops[i]) for i in range(1, len(drops)) if not np.isnan(drops[i])]
    min_drop_k = min(valid_drops, key=lambda x: x[1])[0]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, 'elbow_results_20_40.csv')
    pd.DataFrame({
        'k': k_list,
        'wcss': wcss,
        'silhouette': sil,
        'wcss_drop': drops,
    }).to_csv(csv_path, index=False)

    plot_path = os.path.join(RESULTS_DIR, 'elbow_plot_20_40.png')
    _save_refine_plot(k_list, wcss, sil, elbow_k, best_sil_k, plot_path)

    print('\n' + '─' * 60)
    print('20-40 aralığı — dirsek adayları')
    print('─' * 60)
    print(f'kneed elbow     : {elbow_k}')
    print(f'manuel slowdown : {manual_k}')
    print(f'en küçük dWCSS  : K={min_drop_k} (marjinal kazanç minimumu)')
    print(f'best silhouette : K={best_sil_k} (sil={max(sil):.4f})')
    print('─' * 60)
    print(f'Kayıt: {csv_path}')
    print(f'Grafik: {plot_path}')

    return {
        'elbow_k': elbow_k,
        'manual_k': manual_k,
        'min_drop_k': min_drop_k,
        'best_sil_k': best_sil_k,
    }


def run_elbow_analysis(refine_20_40=True):
    U_norm = _prepare_features()

    wcss = []
    sil = []
    for K in K_LIST:
        km = KMeans(n_clusters=K, n_init=10, max_iter=300, random_state=42)
        km.fit(U_norm)
        inertia = km.inertia_
        sil_score = silhouette_score(
            U_norm, km.labels_, sample_size=500, random_state=42
        )
        wcss.append(inertia)
        sil.append(sil_score)
        print(f'K={K:3d}  WCSS={inertia:.2f}  Silhouette={sil_score:.4f}')

    elbow_k = _detect_elbow(K_LIST, wcss)
    best_sil_k = K_LIST[int(np.argmax(sil))]
    suggestion = max(elbow_k, best_sil_k)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(K_LIST, wcss, color='blue', marker='o')
    if elbow_k is not None:
        elbow_idx = K_LIST.index(elbow_k)
        ax1.scatter([elbow_k], [wcss[elbow_idx]], color='red', s=120, zorder=5)
        ax1.annotate(
            f'Elbow K={elbow_k}',
            (elbow_k, wcss[elbow_idx]),
            textcoords='offset points',
            xytext=(8, 8),
            color='red',
        )
    ax1.set_title('WCSS vs K (Elbow Method)')
    ax1.set_xlabel('K (Küme Sayısı)')
    ax1.set_ylabel('WCSS')
    ax1.grid(True)

    ax2.plot(K_LIST, sil, color='green', marker='s')
    best_idx = K_LIST.index(best_sil_k)
    ax2.scatter([best_sil_k], [sil[best_idx]], color='red', s=120, zorder=5)
    ax2.annotate(
        f'Best K={best_sil_k}',
        (best_sil_k, sil[best_idx]),
        textcoords='offset points',
        xytext=(8, 8),
        color='red',
    )
    ax2.set_title('Silhouette vs K')
    ax2.set_xlabel('K (Küme Sayısı)')
    ax2.set_ylabel('Silhouette Score')
    ax2.grid(True)

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_path = os.path.join(RESULTS_DIR, 'elbow_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    csv_path = os.path.join(RESULTS_DIR, 'elbow_results.csv')
    pd.DataFrame({'k': K_LIST, 'wcss': wcss, 'silhouette': sil}).to_csv(
        csv_path, index=False
    )

    print('══════════════════════════════')
    print(f'Elbow K        : {elbow_k}')
    print(f'Best Silhouette K : {best_sil_k}')
    print(f'Öneri          : {suggestion}')
    print('══════════════════════════════')
    print(f'Sonuçlar: {RESULTS_DIR}/')

    if refine_20_40:
        refine = run_refined_elbow_20_40(U_norm)
        print('\n══════════════════════════════')
        print('20-40 ince tarama özeti')
        print(f'Gerçek elbow (kneed) : {refine["elbow_k"]}')
        print(f'Manuel slowdown      : {refine["manual_k"]}')
        print(f'Best silhouette      : {refine["best_sil_k"]}')
        print('══════════════════════════════')


if __name__ == '__main__':
    import argparse

    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    parser = argparse.ArgumentParser(description='Elbow + Silhouette K seçimi')
    parser.add_argument(
        '--refine-only',
        action='store_true',
        help='Sadece K=20..40 ince taramayı çalıştır',
    )
    parser.add_argument(
        '--no-refine',
        action='store_true',
        help='Kaba tarama sonrası 20-40 ince taramayı atla',
    )
    args = parser.parse_args()

    if args.refine_only:
        run_refined_elbow_20_40()
    else:
        run_elbow_analysis(refine_20_40=not args.no_refine)
