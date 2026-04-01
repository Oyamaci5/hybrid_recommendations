"""
compare_assignments.py
======================
results/assignments_lof/ altındaki tüm assignment klasörlerini
otomatik bulur ve karşılaştırır.

Kullanım:
    python compare_assignments.py
    python compare_assignments.py --root results/assignments_lof
    python compare_assignments.py --root results/assignments
"""

import os
import sys
import argparse
import numpy as np

# ============================================================
# AYARLAR
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def find_assignment_dirs(root):
    """
    root/ altındaki tüm klasörleri tara,
    assignments.npy içerenleri döndür.
    """
    dirs = []
    for dataset in sorted(os.listdir(root)):
        dataset_path = os.path.join(root, dataset)
        if not os.path.isdir(dataset_path):
            continue
        for algo in sorted(os.listdir(dataset_path)):
            algo_path = os.path.join(dataset_path, algo)
            if not os.path.isdir(algo_path):
                continue
            if os.path.exists(os.path.join(algo_path, 'assignments.npy')):
                dirs.append((dataset, algo, algo_path))
    return dirs


def analyze_one(dataset, algo, path):
    """Tek bir assignment klasörünü analiz et."""
    assignments     = np.load(os.path.join(path, 'assignments.npy'))
    gray_sheep_mask = np.load(os.path.join(path, 'gray_sheep_mask.npy'))
    best_sol        = np.load(os.path.join(path, 'best_sol.npy'))

    lof_path = os.path.join(path, 'lof_scores.npy')
    lof_scores = np.load(lof_path) if os.path.exists(lof_path) else None

    n_users    = len(assignments)
    K          = int(assignments.max()) + 1
    n_items    = len(best_sol) // K

    # Küme boyutları
    cluster_sizes = np.bincount(assignments, minlength=K)
    active        = cluster_sizes[cluster_sizes > 0]

    # Gray sheep
    gs_count = int(gray_sheep_mask.sum())
    gs_ratio = float(gray_sheep_mask.mean())

    # Normal (white) kullanıcıların küme boyutları
    white_assign = assignments[~gray_sheep_mask]
    white_sizes  = np.bincount(white_assign, minlength=K)
    white_active = white_sizes[white_sizes > 0]

    # LOF threshold
    if lof_scores is not None and gs_count > 0:
        gs_lof    = lof_scores[gray_sheep_mask]
        norm_lof  = lof_scores[~gray_sheep_mask]
        threshold = float((gs_lof.min() + norm_lof.max()) / 2) \
                    if len(norm_lof) > 0 else float(gs_lof.min())
    else:
        threshold = None

    # Boş küme sayısı
    empty_clusters = int((cluster_sizes == 0).sum())

    return {
        'dataset'       : dataset,
        'algo'          : algo,
        'n_users'       : n_users,
        'K'             : K,
        'n_items'       : n_items,
        'gs_count'      : gs_count,
        'gs_ratio'      : gs_ratio,
        'gs_threshold'  : threshold,
        'cluster_min'   : int(active.min()),
        'cluster_max'   : int(active.max()),
        'cluster_mean'  : float(active.mean()),
        'cluster_std'   : float(active.std()),
        'empty_clusters': empty_clusters,
        'white_min'     : int(white_active.min()) if len(white_active) > 0 else 0,
        'white_max'     : int(white_active.max()) if len(white_active) > 0 else 0,
        'white_mean'    : float(white_active.mean()) if len(white_active) > 0 else 0,
        'lof_mean'      : float(lof_scores.mean()) if lof_scores is not None else None,
        'lof_max'       : float(lof_scores.max())  if lof_scores is not None else None,
    }


def print_report(results):
    """Sonuçları gruplandırarak yazdır."""

    # Dataset'e göre grupla
    datasets = sorted(set(r['dataset'] for r in results))

    for ds in datasets:
        ds_results = [r for r in results if r['dataset'] == ds]
        print(f"\n{'='*70}")
        print(f"DATASET: {ds.upper()}")
        print(f"{'='*70}")

        # Ana tablo
        print(f"\n{'Algo':<18} {'K':>4} {'Users':>6} {'GS_n':>6} {'GS_%':>6} "
              f"{'GS_thr':>8} {'Empty':>6}")
        print("-" * 60)
        for r in ds_results:
            thr = f"{r['gs_threshold']:.4f}" if r['gs_threshold'] else "  —"
            print(f"  {r['algo']:<16} {r['K']:>4} {r['n_users']:>6} "
                  f"{r['gs_count']:>6} {r['gs_ratio']*100:>5.2f}% "
                  f"{thr:>8} {r['empty_clusters']:>6}")

        # Küme boyutu tablosu
        print(f"\n{'Algo':<18} {'Clust_min':>10} {'Clust_max':>10} "
              f"{'Clust_mean':>11} {'Clust_std':>10}")
        print("-" * 65)
        for r in ds_results:
            print(f"  {r['algo']:<16} {r['cluster_min']:>10} {r['cluster_max']:>10} "
                  f"{r['cluster_mean']:>11.1f} {r['cluster_std']:>10.1f}")

        # LOF tablosu (varsa)
        if any(r['lof_mean'] is not None for r in ds_results):
            print(f"\n{'Algo':<18} {'LOF_mean':>10} {'LOF_max':>10} "
                  f"{'White_min':>10} {'White_max':>10} {'White_mean':>11}")
            print("-" * 72)
            for r in ds_results:
                lm  = f"{r['lof_mean']:.4f}"  if r['lof_mean']  is not None else "  —"
                lmx = f"{r['lof_max']:.4f}"   if r['lof_max']   is not None else "  —"
                print(f"  {r['algo']:<16} {lm:>10} {lmx:>10} "
                      f"{r['white_min']:>10} {r['white_max']:>10} "
                      f"{r['white_mean']:>11.1f}")

        # Gray sheep karşılaştırması
        print(f"\n  Gray sheep özeti:")
        for r in ds_results:
            bar_len = int(r['gs_ratio'] * 500)
            bar     = '█' * bar_len + '░' * (10 - bar_len)
            print(f"    {r['algo']:<18} {bar} {r['gs_ratio']*100:.2f}%  ({r['gs_count']} kullanıcı)")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root', default=None,
        help='Assignment kök klasörü (default: otomatik bulur)'
    )
    args = parser.parse_args()

    # Kök klasörü bul
    if args.root:
        root = args.root
    else:
        # Otomatik: önce assignments_lof, sonra assignments dene
        candidates = [
            os.path.join(BASE_DIR, 'results', 'assignments_lof'),
            os.path.join(BASE_DIR, 'results', 'assignments'),
        ]
        root = None
        for c in candidates:
            if os.path.isdir(c):
                root = c
                break
        if root is None:
            print("HATA: Assignment klasörü bulunamadı.")
            print("  Aranan: results/assignments_lof/ veya results/assignments/")
            sys.exit(1)

    print(f"Kök klasör: {root}")

    # Klasörleri bul
    dirs = find_assignment_dirs(root)
    if not dirs:
        print("HATA: Hiç assignment klasörü bulunamadı.")
        sys.exit(1)

    print(f"Bulunan klasör sayısı: {len(dirs)}")
    for dataset, algo, path in dirs:
        print(f"  {dataset}/{algo}")

    # Analiz et
    results = []
    for dataset, algo, path in dirs:
        try:
            r = analyze_one(dataset, algo, path)
            results.append(r)
        except Exception as e:
            print(f"HATA: {dataset}/{algo} → {e}")

    # Rapor
    print_report(results)

    # CSV kaydet
    import csv
    out_csv = os.path.join(root, 'assignment_comparison.csv')
    if results:
        keys = results[0].keys()
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV kaydedildi: {out_csv}")


if __name__ == '__main__':
    main()