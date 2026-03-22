"""
phase4_encr.py
ENCR Tabanlı Adaptif Hibrit Kümeleme
Meidani et al. (2022) "Online Metaheuristic Algorithm Selection"

Algoritma çifti (davranış analizinden):
  OOA  → GLOBAL (exploration_ratio=0.32, yüksek keşif)
  WOA  → LOCAL  (convergence_speed=7, erken yakınsama)

Two-way switch: her K iterasyonda landscape ölçülür,
ENCR > ε ise OOA, ENCR ≤ ε ise WOA çalışır.
Popülasyon switch'te sıfırlanmaz, önceki algoritmanın
bulduğu en iyi noktadan devam edilir.

Kullanım:
  python phase4_encr.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # GUI gerektirmez
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import silhouette_score, davies_bouldin_score
from mealpy import FloatVar
from mealpy.swarm_based.OOA import OriginalOOA
from mealpy.swarm_based.WOA import OriginalWOA

# ── Ana kod dosyasından ortak fonksiyonlar ────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mealpy_comparison_v2 import (
    load_movielens, mkmeans_plus_plus_init,
    compute_wcss_fast, detect_gray_sheep,
    make_fitness_function, pearson_distance_batch,
    _compute_metrics,
)

# ============================================================
# KLASÖR AYARI
# ============================================================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'encr')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# BÖLÜM 1: ENCR HESABI
# ============================================================

def compute_encr(agents: np.ndarray,
                 fitness_func,
                 fla_no: int = 20) -> float:
    """
    ENCR (Efficient Non-Convex Ratio) hesapla.

    Meidani et al. (2022) formülü:
      Her (i,j) çifti için orta nokta: m_ij = (x_i + x_j) / 2
      Jensen ihlali: f(m_ij) > (f(x_i) + f(x_j)) / 2  → 1
      ENCR = toplam ihlal / toplam çift sayısı

    Yüksek ENCR → çok modlu landscape → exploration gerekli
    Düşük  ENCR → tek modlu landscape  → exploitation gerekli

    Parameters
    ----------
    agents      : (N, D) mevcut popülasyon
    fitness_func: tek çözüm alıp float döndüren fonksiyon
    fla_no      : kaç ajan örnekleneceği (N'den küçük olmalı)

    Returns
    -------
    float: [0, 1] arasında ENCR değeri
    """
    n = len(agents)
    fla_no = min(fla_no, n)

    # Rastgele fla_no kadar ajan seç
    idx    = np.random.choice(n, size=fla_no, replace=False)
    sample = agents[idx]                            # (fla_no, D)

    # Fitness değerlerini hesapla (önceden biliniyorsa daha hızlı olur
    # ama genel amaçlı tutmak için burada hesaplıyoruz)
    f_vals = np.array([fitness_func(s) for s in sample], dtype=np.float64)

    violations   = 0
    total_pairs  = 0

    for i in range(fla_no - 1):
        for j in range(i + 1, fla_no):
            midpoint    = (sample[i] + sample[j]) / 2.0
            f_mid       = fitness_func(midpoint)
            f_avg       = (f_vals[i] + f_vals[j]) / 2.0
            if f_mid > f_avg:          # Jensen ihlali
                violations += 1
            total_pairs += 1

    return violations / total_pairs if total_pairs > 0 else 0.0


# ============================================================
# BÖLÜM 2: ENCR TWO-WAY SWITCH OPTİMİZASYON
# ============================================================

def run_encr_hybrid(
    matrix      : np.ndarray,
    K           : int,
    initial_solutions,
    max_epoch   : int   = 50,
    pop_size    : int   = 30,
    K_switch    : int   = 10,    # her K_switch epoch'ta ENCR hesapla
    epsilon     : float = 0.1,   # ENCR eşik değeri
    fla_no      : int   = 20,    # ENCR örnekleme boyutu
    switch_mode : str   = "two_way",
) -> dict:
    """
    ENCR two-way switch hibrit optimizasyon.

    Parameters
    ----------
    matrix           : (n_users, n_items) rating matrisi
    K                : küme sayısı
    initial_solutions: MkMeans++ başlangıç popülasyonu
    max_epoch        : toplam epoch sayısı
    pop_size         : popülasyon büyüklüğü
    K_switch         : kaç epoch'ta bir ENCR hesaplanır
    epsilon          : ENCR eşiği
    fla_no           : ENCR örnekleme boyutu
    switch_mode      : "two_way" veya "one_way"

    Returns
    -------
    dict: best_solution, best_wcss, encr_history, algo_history,
          wcss, silhouette, davies_bouldin, gray_sheep_*
    """
    start   = time.time()
    n_items = matrix.shape[1]
    n_vars  = K * n_items

    fitness_fn = make_fitness_function(matrix, K)

    problem = {
        "obj_func"       : fitness_fn,
        "bounds"         : FloatVar(lb=[0.0] * n_vars,
                                    ub=[5.0] * n_vars,
                                    name="centroids"),
        "minmax"         : "min",
        "log_to"         : None,
        "save_population": False,
    }

    # ── Başlangıç: OOA ile ilk epoch bloğu ──────────────────
    print(f"\n  ENCR Two-Way Switch Başlıyor")
    print(f"  max_epoch={max_epoch} | K_switch={K_switch} | ε={epsilon}")
    print(f"  GLOBAL: OOA  |  LOCAL: WOA")
    print(f"  {'─'*55}")

    encr_history  = []    # her switch noktasındaki ENCR
    algo_history  = []    # her epoch hangi algoritma
    wcss_history  = []    # her epoch'taki en iyi WCSS

    current_best_sol = None
    current_best_fit = float('inf')
    current_pop      = None   # mevcut popülasyon (switch'te aktarılır)

    epochs_done   = 0
    switched_once = False    # one_way için

    while epochs_done < max_epoch:
        epochs_left  = max_epoch - epochs_done
        block_epochs = min(K_switch, epochs_left)

        # ── ENCR hesapla (mevcut popülasyondan) ─────────────
        if current_pop is not None and len(current_pop) >= fla_no:
            encr = compute_encr(current_pop, fitness_fn, fla_no)
        else:
            encr = 1.0   # başlangıçta landscape bilinmiyor → global başla

        encr_history.append({'epoch': epochs_done, 'encr': encr})

        # ── Algoritma seç ────────────────────────────────────
        if switch_mode == "one_way":
            if switched_once:
                use_algo = "WOA"
            else:
                if encr > epsilon:
                    use_algo = "OOA"
                else:
                    use_algo = "WOA"
                    switched_once = True
        else:  # two_way
            use_algo = "OOA" if encr > epsilon else "WOA"

        print(f"  Epoch {epochs_done+1:>3}–{epochs_done+block_epochs:<3} | "
              f"ENCR={encr:.3f} | {'> ε' if encr > epsilon else '≤ ε'} | "
              f"→ {use_algo}")

        # ── Seçilen algoritmayı çalıştır ─────────────────────
        if use_algo == "OOA":
            model = OriginalOOA(epoch=block_epochs, pop_size=pop_size)
        else:
            model = OriginalWOA(epoch=block_epochs, pop_size=pop_size)

        # Başlangıç noktası: ilk blokta MkMeans++, sonraki bloklarda
        # önceki algoritmanın popülasyonunu aktar
        if current_pop is not None:
            start_sols = list(current_pop[:pop_size])
        else:
            start_sols = initial_solutions[:pop_size]

        try:
            model.solve(problem, starting_solutions=start_sols)
        except TypeError:
            model.solve(problem)

        # ── Sonuçları güncelle ────────────────────────────────
        block_best_sol = model.g_best.solution
        block_best_fit = model.g_best.target.fitness

        if block_best_fit < current_best_fit:
            current_best_sol = block_best_sol.copy()
            current_best_fit = block_best_fit

        # Popülasyonu bir sonraki bloğa aktar
        try:
            current_pop = np.array([ag.solution for ag in model.pop])
        except Exception:
            current_pop = None

        # Epoch bazlı geçmiş
        for ep_fit in model.history.list_global_best_fit:
            algo_history.append(use_algo)
            wcss_history.append(ep_fit)

        epochs_done += block_epochs

    # ── Final metrikler ──────────────────────────────────────
    metrics = _compute_metrics(matrix, current_best_sol, K)

    print(f"\n  ── ENCR Hibrit Tamamlandı ──────────────────────")
    print(f"  Final WCSS : {current_best_fit:.3f}")
    print(f"  DB         : {metrics['davies_bouldin']:.3f}")
    print(f"  Sil        : {metrics['silhouette']:.3f}")
    print(f"  GS         : {metrics['gray_sheep_ratio']:.1%}")
    print(f"  Süre       : {time.time()-start:.1f}s")

    return {
        'best_solution'   : current_best_sol,
        'best_wcss'       : float(current_best_fit),
        'wcss'            : float(current_best_fit),
        'encr_history'    : encr_history,
        'algo_history'    : algo_history,
        'wcss_history'    : wcss_history,
        'time_seconds'    : time.time() - start,
        **metrics,
    }


# ============================================================
# BÖLÜM 3: BASELINE KARŞILAŞTIRMASI
# ============================================================

def run_baseline(name, algo_class, matrix, K,
                 initial_solutions, epoch, pop_size):
    """Tek algoritma baseline."""
    start   = time.time()
    n_items = matrix.shape[1]
    problem = {
        "obj_func"       : make_fitness_function(matrix, K),
        "bounds"         : FloatVar(lb=[0.0] * (K * n_items),
                                    ub=[5.0] * (K * n_items)),
        "minmax"         : "min",
        "log_to"         : None,
        "save_population": False,
    }
    model = algo_class(epoch=epoch, pop_size=pop_size)
    try:
        model.solve(problem, starting_solutions=initial_solutions[:pop_size])
    except TypeError:
        model.solve(problem)

    best_sol = model.g_best.solution
    best_fit = model.g_best.target.fitness
    metrics  = _compute_metrics(matrix, best_sol, K)

    print(f"  {name:<25} WCSS={best_fit:.3f} | "
          f"DB={metrics['davies_bouldin']:.3f} | "
          f"Sil={metrics['silhouette']:.3f} | "
          f"{time.time()-start:.1f}s")

    return {
        'name'        : name,
        'wcss'        : float(best_fit),
        'wcss_history': model.history.list_global_best_fit,
        'time_seconds': time.time() - start,
        **metrics,
    }


def run_random_switch(matrix, K, initial_solutions,
                      epoch, pop_size, K_switch, seed=42):
    """Random switch baseline: her K_switch epoch'ta rastgele OOA veya WOA seç."""
    np.random.seed(seed)
    start   = time.time()
    n_items = matrix.shape[1]
    fitness_fn = make_fitness_function(matrix, K)
    problem = {
        "obj_func"       : fitness_fn,
        "bounds"         : FloatVar(lb=[0.0] * (K * n_items),
                                    ub=[5.0] * (K * n_items)),
        "minmax"         : "min",
        "log_to"         : None,
        "save_population": False,
    }

    current_best_sol = None
    current_best_fit = float('inf')
    current_pop      = None
    wcss_history     = []
    algo_history     = []
    epochs_done      = 0

    while epochs_done < epoch:
        block = min(K_switch, epoch - epochs_done)
        use_algo = np.random.choice(["OOA", "WOA"])
        model    = OriginalOOA(epoch=block, pop_size=pop_size) \
                   if use_algo == "OOA" \
                   else OriginalWOA(epoch=block, pop_size=pop_size)

        start_sols = list(current_pop[:pop_size]) \
                     if current_pop is not None \
                     else initial_solutions[:pop_size]
        try:
            model.solve(problem, starting_solutions=start_sols)
        except TypeError:
            model.solve(problem)

        if model.g_best.target.fitness < current_best_fit:
            current_best_sol = model.g_best.solution.copy()
            current_best_fit = model.g_best.target.fitness

        try:
            current_pop = np.array([ag.solution for ag in model.pop])
        except Exception:
            current_pop = None

        for f in model.history.list_global_best_fit:
            wcss_history.append(f)
            algo_history.append(use_algo)

        epochs_done += block

    metrics = _compute_metrics(matrix, current_best_sol, K)
    print(f"  {'Random Switch':<25} WCSS={current_best_fit:.3f} | "
          f"DB={metrics['davies_bouldin']:.3f} | "
          f"Sil={metrics['silhouette']:.3f} | "
          f"{time.time()-start:.1f}s")

    return {
        'name'        : 'Random Switch',
        'wcss'        : float(current_best_fit),
        'wcss_history': wcss_history,
        'algo_history': algo_history,
        'time_seconds': time.time() - start,
        **metrics,
    }


# ============================================================
# BÖLÜM 4: GÖRSELLEŞTİRME
# ============================================================

def plot_encr_analysis(encr_result, baselines, save_path):
    """
    3 grafik:
      1. WCSS yakınsama eğrileri (tüm yöntemler)
      2. ENCR geçmişi + algoritma switch noktaları
      3. Karşılaştırma çubuğu (WCSS + DB)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {
        'OOA Tek'      : '#1f77b4',
        'WOA Tek'      : '#ff7f0e',
        'Random Switch': '#2ca02c',
        'ENCR Switch'  : '#d62728',
    }

    # ── Grafik 1: Yakınsama eğrileri ────────────────────────
    ax1 = axes[0]
    for b in baselines:
        ax1.plot(b['wcss_history'], label=b['name'],
                 color=colors.get(b['name'], 'gray'), linewidth=1.5)

    ax1.plot(encr_result['wcss_history'], label='ENCR Switch',
             color=colors['ENCR Switch'], linewidth=2.0, linestyle='--')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('WCSS')
    ax1.set_title('Yakınsama Eğrileri')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Grafik 2: ENCR geçmişi ───────────────────────────────
    ax2 = axes[1]
    encr_epochs = [e['epoch'] for e in encr_result['encr_history']]
    encr_vals   = [e['encr']  for e in encr_result['encr_history']]

    ax2.plot(encr_epochs, encr_vals, 'o-', color='#9467bd',
             linewidth=1.5, markersize=6, label='ENCR')
    ax2.axhline(y=0.1, color='red', linestyle='--', linewidth=1,
                alpha=0.7, label='ε = 0.1')

    # Switch noktalarını boyalı bant olarak göster
    algo_hist = encr_result['algo_history']
    if algo_hist:
        for i in range(len(algo_hist)):
            color = '#1f77b420' if algo_hist[i] == 'OOA' else '#ff7f0e20'
            ax2.axvspan(i, i + 1, alpha=0.3, color=color[:-2],
                        linewidth=0)

    ooa_patch = mpatches.Patch(color='#1f77b4', alpha=0.3, label='OOA aktif')
    woa_patch = mpatches.Patch(color='#ff7f0e', alpha=0.3, label='WOA aktif')
    ax2.legend(handles=[ooa_patch, woa_patch,
                         plt.Line2D([0], [0], color='#9467bd', marker='o',
                                    label='ENCR'),
                         plt.Line2D([0], [0], color='red', linestyle='--',
                                    label='ε=0.1')],
               fontsize=8)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('ENCR')
    ax2.set_title('ENCR Geçmişi & Algoritma Seçimi')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    # ── Grafik 3: Karşılaştırma çubukları ────────────────────
    ax3 = axes[2]
    all_results = baselines + [{
        'name': 'ENCR Switch',
        'wcss': encr_result['wcss'],
        'davies_bouldin': encr_result['davies_bouldin'],
    }]

    names     = [r['name'] for r in all_results]
    wcss_vals = [r['wcss'] for r in all_results]
    db_vals   = [r['davies_bouldin'] for r in all_results]

    x  = np.arange(len(names))
    w  = 0.35
    b1 = ax3.bar(x - w/2, wcss_vals, w, label='WCSS',
                 color=[colors.get(n, '#7f7f7f') for n in names], alpha=0.8)
    ax3_twin = ax3.twinx()
    b2 = ax3_twin.bar(x + w/2, db_vals, w, label='DB',
                      color=[colors.get(n, '#7f7f7f') for n in names],
                      alpha=0.4, hatch='//')

    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    ax3.set_ylabel('WCSS (düşük = iyi)')
    ax3_twin.set_ylabel('Davies-Bouldin (düşük = iyi)')
    ax3.set_title('Yöntem Karşılaştırması')

    lines1, labs1 = ax3.get_legend_handles_labels()
    lines2, labs2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(save_path, 'encr_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Grafik kaydedildi: {path}")


# ============================================================
# BÖLÜM 5: SONUÇ TABLOSU
# ============================================================

def print_comparison_table(encr_result, baselines):
    """Tezde kullanılacak karşılaştırma tablosu."""
    print("\n" + "=" * 70)
    print("YÖNTEM KARŞILAŞTIRMASI")
    print("=" * 70)
    print(f"{'Yöntem':<22} {'WCSS':>8} {'DB':>7} {'Sil':>7} "
          f"{'GS%':>6} {'Süre':>7}")
    print("-" * 70)

    for b in baselines:
        sil = b.get('silhouette', float('nan'))
        gs  = b.get('gray_sheep_ratio', float('nan'))
        print(f"{b['name']:<22} {b['wcss']:>8.3f} "
              f"{b['davies_bouldin']:>7.3f} "
              f"{sil:>7.3f} "
              f"{gs:>5.1%} "
              f"{b['time_seconds']:>7.1f}s")

    e = encr_result
    print(f"{'ENCR Switch':<22} {e['wcss']:>8.3f} "
          f"{e['davies_bouldin']:>7.3f} "
          f"{e['silhouette']:>7.3f} "
          f"{e['gray_sheep_ratio']:>5.1%} "
          f"{e['time_seconds']:>7.1f}s")

    # En iyi değerleri bul
    all_wcss = [b['wcss'] for b in baselines] + [e['wcss']]
    all_db   = [b['davies_bouldin'] for b in baselines] + [e['davies_bouldin']]
    best_wcss = min(all_wcss)
    best_db   = min(all_db)

    print("-" * 70)
    print(f"En iyi WCSS: {best_wcss:.3f}  |  En iyi DB: {best_db:.3f}")

    # ENCR iyileştirmesi
    baseline_wcss = [b['wcss'] for b in baselines if b['name'] != 'Random Switch']
    avg_baseline  = np.mean(baseline_wcss) if baseline_wcss else float('nan')
    improv = (avg_baseline - e['wcss']) / avg_baseline * 100
    print(f"ENCR Switch iyileştirmesi (tek algoritmalar ortalaması): "
          f"{improv:+.1f}%")


# ============================================================
# BÖLÜM 6: ANA ÇALIŞTIRMA
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ENCR HİBRİT KÜMELEMESİ")
    print("OOA (global) + WOA (local) | Two-way Switch")
    print("=" * 60)

    # Veri yükle
    DATA_PATH   = os.path.join(os.path.dirname(BASE_DIR),
                               'data', 'ml-100k', 'u.data')
    full_matrix = load_movielens(DATA_PATH)

    K         = 90
    EPOCH     = 50
    POP_SIZE  = 30
    K_SWITCH  = 10
    EPSILON   = 0.1
    FLA_NO    = 20

    # MkMeans++ başlangıç popülasyonu
    print("\nMkMeans++ başlangıç popülasyonu hazırlanıyor...")
    init_solutions = mkmeans_plus_plus_init(full_matrix, K=K,
                                            n_solutions=POP_SIZE + 10)

    # ── Baseline 1: Sadece OOA ───────────────────────────────
    print("\n[1/4] Sadece OOA çalışıyor...")
    ooa_result = run_baseline(
        'OOA Tek', OriginalOOA, full_matrix, K,
        init_solutions, EPOCH, POP_SIZE
    )

    # ── Baseline 2: Sadece WOA ───────────────────────────────
    print("\n[2/4] Sadece WOA çalışıyor...")
    woa_result = run_baseline(
        'WOA Tek', OriginalWOA, full_matrix, K,
        init_solutions, EPOCH, POP_SIZE
    )

    # ── Baseline 3: Random Switch ────────────────────────────
    print("\n[3/4] Random Switch çalışıyor...")
    rand_result = run_random_switch(
        full_matrix, K, init_solutions,
        epoch=EPOCH, pop_size=POP_SIZE,
        K_switch=K_SWITCH
    )

    # ── ENCR Two-way Switch ──────────────────────────────────
    print("\n[4/4] ENCR Two-way Switch çalışıyor...")
    encr_result = run_encr_hybrid(
        matrix=full_matrix,
        K=K,
        initial_solutions=init_solutions,
        max_epoch=EPOCH,
        pop_size=POP_SIZE,
        K_switch=K_SWITCH,
        epsilon=EPSILON,
        fla_no=FLA_NO,
        switch_mode="two_way",
    )

    # ── Sonuçlar ─────────────────────────────────────────────
    baselines = [ooa_result, woa_result, rand_result]
    print_comparison_table(encr_result, baselines)

    # CSV kaydet
    rows = []
    for b in baselines:
        rows.append({
            'method'          : b['name'],
            'wcss'            : b['wcss'],
            'silhouette'      : b.get('silhouette', None),
            'davies_bouldin'  : b.get('davies_bouldin', None),
            'gray_sheep_ratio': b.get('gray_sheep_ratio', None),
            'time_seconds'    : b['time_seconds'],
        })
    rows.append({
        'method'          : 'ENCR Switch',
        'wcss'            : encr_result['wcss'],
        'silhouette'      : encr_result['silhouette'],
        'davies_bouldin'  : encr_result['davies_bouldin'],
        'gray_sheep_ratio': encr_result['gray_sheep_ratio'],
        'time_seconds'    : encr_result['time_seconds'],
    })
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, 'encr_comparison.csv'), index=False
    )

    # ENCR geçmişi kaydet
    pd.DataFrame(encr_result['encr_history']).to_csv(
        os.path.join(RESULTS_DIR, 'encr_history.csv'), index=False
    )

    # Grafik
    plot_encr_analysis(encr_result, baselines, RESULTS_DIR)

    print(f"\nSonuçlar: {RESULTS_DIR}/")
    print("  encr_comparison.csv  — yöntem karşılaştırması")
    print("  encr_history.csv     — ENCR değerleri + switch noktaları")
    print("  encr_analysis.png    — 3 grafikli görselleştirme")