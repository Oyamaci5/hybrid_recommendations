"""Elbow + Silhouette analizi ile optimal K seçimi (KMeans veya FCM)."""
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
from mealpy_comparison_v2 import compute_fcm_objective, load_movielens  # noqa: E402

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


def _prepare_features(wnmf_dim=30, method='kmeans'):
    """KMeans: prune + L2 norm (WNMF30). FCM: no-prune, ham WNMF (fuzzy protokolü)."""
    data_path = os.path.join(REPO_ROOT, 'data', 'ml-100k', 'u.data')
    print(f'Veri yükleniyor: {data_path}')
    matrix = load_movielens(data_path)
    if method == 'fcm':
        print('  FCM protokolü: --no-prune, preprocess none, L2 norm yok')
    else:
        matrix = prune_sparse_matrix(matrix, min_user_ratings=5, min_item_ratings=10)
    U = wnmf_feature_extract(
        matrix, n_components=wnmf_dim, n_epochs=50, random_seed=42,
    )
    if method == 'fcm':
        return U.astype(np.float32)
    return normalize(U, norm='l2')


def _kmeans_init_centroids(U_norm, K, random_state=42):
    km = KMeans(
        n_clusters=K, init='k-means++', n_init=1, max_iter=1, random_state=random_state,
    )
    km.fit(U_norm)
    return km.cluster_centers_.astype(np.float32).flatten()


def _fcm_best_for_k(U_norm, K, fcm_m=2.0, fcm_max_iter=50, n_init=10, random_state=42):
    """n_init k-means++ başlangıcından en düşük J_m veren FCM çözümünü seç."""
    best_j, best_labels = None, None
    rng = np.random.RandomState(random_state)
    for i in range(n_init):
        seed = int(rng.randint(0, 2**31 - 1))
        init_sol = _kmeans_init_centroids(U_norm, K, random_state=seed)
        j_val, labels, _, _ = compute_fcm_objective(
            U_norm, init_sol, K, m=float(fcm_m), max_iter=fcm_max_iter, tol=1e-4,
        )
        if best_j is None or j_val < best_j:
            best_j, best_labels = float(j_val), labels
    return best_j, best_labels


def _cluster_one_k(
    U_norm, K, method='kmeans', fcm_m=2.0, fcm_max_iter=50, fcm_n_init=10,
):
    if method == 'kmeans':
        km = KMeans(n_clusters=K, n_init=10, max_iter=300, random_state=42)
        km.fit(U_norm)
        objective = float(km.inertia_)
        labels = km.labels_
        obj_label = 'WCSS'
    elif method == 'fcm':
        objective, labels = _fcm_best_for_k(
            U_norm, K, fcm_m=fcm_m, fcm_max_iter=fcm_max_iter, n_init=fcm_n_init,
        )
        obj_label = f'J_m (m={fcm_m:g})'
    else:
        raise ValueError(f'Bilinmeyen method: {method}')

    sil_score = float(
        silhouette_score(U_norm, labels, sample_size=500, random_state=42)
    )
    return objective, sil_score, obj_label


def _scan_k_values(
    U_norm, k_list, method='kmeans', fcm_m=2.0, fcm_max_iter=50, fcm_n_init=10,
):
    objectives, sil, drops = [], [], []
    obj_label = None
    prev_obj = None
    for K in k_list:
        objective, sil_score, obj_label = _cluster_one_k(
            U_norm, K, method=method, fcm_m=fcm_m, fcm_max_iter=fcm_max_iter,
            fcm_n_init=fcm_n_init,
        )
        drop = (prev_obj - objective) if prev_obj is not None else float('nan')
        objectives.append(objective)
        sil.append(sil_score)
        drops.append(drop)
        prev_obj = objective
        drop_txt = f'  dObj={drop:+.2f}' if not np.isnan(drop) else ''
        print(
            f'K={K:3d}  {obj_label}={objective:.2f}  '
            f'Silhouette={sil_score:.4f}{drop_txt}'
        )
    return objectives, sil, drops, obj_label


def _save_refine_plot(
    k_list, objectives, sil, elbow_k, best_sil_k, path,
    obj_ylabel='WCSS', title_suffix='20-40 ince tarama',
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(k_list, objectives, color='blue', marker='o', markersize=4)
    if elbow_k is not None:
        elbow_idx = k_list.index(elbow_k)
        ax1.scatter([elbow_k], [objectives[elbow_idx]], color='red', s=120, zorder=5)
        ax1.annotate(
            f'Elbow K={elbow_k}',
            (elbow_k, objectives[elbow_idx]),
            textcoords='offset points',
            xytext=(8, 8),
            color='red',
        )
    ax1.set_title(f'{obj_ylabel} vs K ({title_suffix})')
    ax1.set_xlabel('K (Küme Sayısı)')
    ax1.set_ylabel(obj_ylabel)
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


def _output_tag(method, fcm_m, wnmf_dim):
    if method == 'fcm':
        m_tag = f'_fcm_m{int(round(float(fcm_m) * 10))}'
        return f'{m_tag}_w{wnmf_dim}'
    return f'_w{wnmf_dim}'


def run_refined_elbow_20_40(
    U_norm=None, method='kmeans', fcm_m=2.0, fcm_max_iter=50, fcm_n_init=10,
    wnmf_dim=30,
):
    """K=20..40 aralığında adım=1 ince elbow taraması."""
    if U_norm is None:
        U_norm = _prepare_features(wnmf_dim=wnmf_dim, method=method)

    tag = _output_tag(method, fcm_m, wnmf_dim)
    print('\n' + '=' * 60)
    print(f'İnce tarama: K = 20 .. 40 (adım 1) [{method}{tag}]')
    print('=' * 60)

    k_list = K_LIST_REFINE_20_40
    objectives, sil, drops, obj_label = _scan_k_values(
        U_norm, k_list, method=method, fcm_m=fcm_m, fcm_max_iter=fcm_max_iter,
        fcm_n_init=fcm_n_init,
    )

    elbow_k = _detect_elbow(k_list, objectives)
    manual_k = _manual_elbow(k_list, objectives)
    best_sil_k = k_list[int(np.argmax(sil))]

    valid_drops = [(k_list[i], drops[i]) for i in range(1, len(drops)) if not np.isnan(drops[i])]
    min_drop_k = min(valid_drops, key=lambda x: x[1])[0]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f'elbow_results_20_40{tag}.csv')
    obj_col = 'j_m' if method == 'fcm' else 'wcss'
    pd.DataFrame({
        'k': k_list,
        obj_col: objectives,
        'silhouette': sil,
        'objective_drop': drops,
    }).to_csv(csv_path, index=False)

    plot_path = os.path.join(RESULTS_DIR, f'elbow_plot_20_40{tag}.png')
    _save_refine_plot(
        k_list, objectives, sil, elbow_k, best_sil_k, plot_path,
        obj_ylabel=obj_label, title_suffix=f'20-40, {method}',
    )

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


def run_elbow_analysis(
    refine_20_40=True, method='kmeans', fcm_m=2.0, fcm_max_iter=50,
    fcm_n_init=10, wnmf_dim=30,
):
    tag = _output_tag(method, fcm_m, wnmf_dim)
    print(f'Elbow analizi: method={method}, WNMF{wnmf_dim}{tag}')

    U_norm = _prepare_features(wnmf_dim=wnmf_dim, method=method)

    objectives = []
    sil = []
    obj_label = None
    for K in K_LIST:
        objective, sil_score, obj_label = _cluster_one_k(
            U_norm, K, method=method, fcm_m=fcm_m, fcm_max_iter=fcm_max_iter,
            fcm_n_init=fcm_n_init,
        )
        objectives.append(objective)
        sil.append(sil_score)
        print(
            f'K={K:3d}  {obj_label}={objective:.2f}  Silhouette={sil_score:.4f}'
        )

    elbow_k = _detect_elbow(K_LIST, objectives)
    best_sil_k = K_LIST[int(np.argmax(sil))]
    suggestion = max(elbow_k, best_sil_k)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(K_LIST, objectives, color='blue', marker='o')
    if elbow_k is not None:
        elbow_idx = K_LIST.index(elbow_k)
        ax1.scatter([elbow_k], [objectives[elbow_idx]], color='red', s=120, zorder=5)
        ax1.annotate(
            f'Elbow K={elbow_k}',
            (elbow_k, objectives[elbow_idx]),
            textcoords='offset points',
            xytext=(8, 8),
            color='red',
        )
    ax1.set_title(f'{obj_label} vs K (Elbow Method, {method})')
    ax1.set_xlabel('K (Küme Sayısı)')
    ax1.set_ylabel(obj_label)
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
    plot_path = os.path.join(RESULTS_DIR, f'elbow_plot{tag}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    obj_col = 'j_m' if method == 'fcm' else 'wcss'
    csv_path = os.path.join(RESULTS_DIR, f'elbow_results{tag}.csv')
    pd.DataFrame({'k': K_LIST, obj_col: objectives, 'silhouette': sil}).to_csv(
        csv_path, index=False
    )

    print('══════════════════════════════')
    print(f'Elbow K        : {elbow_k}')
    print(f'Best Silhouette K : {best_sil_k}')
    print(f'Öneri          : {suggestion}')
    print('══════════════════════════════')
    print(f'Sonuçlar: {RESULTS_DIR}/')

    if refine_20_40:
        refine = run_refined_elbow_20_40(
            U_norm, method=method, fcm_m=fcm_m,
            fcm_max_iter=fcm_max_iter, fcm_n_init=fcm_n_init, wnmf_dim=wnmf_dim,
        )
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
        '--method',
        choices=['kmeans', 'fcm'],
        default='kmeans',
        help='Kümeleme yöntemi (default: kmeans)',
    )
    parser.add_argument(
        '--fcm-m',
        type=float,
        default=2.0,
        metavar='M',
        help='FCM fuzzifier m (--method fcm için, default: 2.0)',
    )
    parser.add_argument(
        '--fcm-iter',
        type=int,
        default=50,
        metavar='N',
        help='FCM maksimum iterasyon (default: 50)',
    )
    parser.add_argument(
        '--fcm-n-init',
        type=int,
        default=10,
        metavar='N',
        help='FCM başlangıç denemesi (k-means++, default: 10)',
    )
    parser.add_argument(
        '--wnmf-dim',
        type=int,
        default=None,
        metavar='D',
        help='WNMF boyutu (default: 20 for fcm, 30 for kmeans)',
    )
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

    wnmf_dim = args.wnmf_dim
    if wnmf_dim is None:
        wnmf_dim = 20 if args.method == 'fcm' else 30

    common = dict(
        method=args.method,
        fcm_m=args.fcm_m,
        fcm_max_iter=args.fcm_iter,
        fcm_n_init=args.fcm_n_init,
        wnmf_dim=wnmf_dim,
    )

    if args.refine_only:
        run_refined_elbow_20_40(**common)
    else:
        run_elbow_analysis(refine_20_40=not args.no_refine, **common)
