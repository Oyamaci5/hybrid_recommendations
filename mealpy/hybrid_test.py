"""
hybrid_test.py  (v3)
====================
8 Hibrit + 3 Baseline — 15 Run, Phase 3 Tam Matris

Hibritler:
  H1: HHO(global) + HGS(local)   — zıt üstünlükler
  H2: HGS(global) + HHO(local)   — ters sıra kontrolü
  H3: MFO(global) + HGS(local)   — güvenli hibrit
  H4: MFO(global) + HHO(local)   — alternatif
  H5: OOA(global) + HHO(local)   — ENCR adayı
  H6: AGTO(global) + HGS(local)  — güçlü global
  H7: IAOA(global) + HHO(local)  — YENİ: improved AO global keşif
  H8: MFO(global)  + IAOA(local) — YENİ: IAOA lokal iyileştirme

Baseline (aynı toplam epoch ile):
  B1: HHO tek  (epoch=50)
  B2: HGS tek  (epoch=50)
  B3: MFO tek  (epoch=50)

Kullanım:
  python hybrid_test.py
  python hybrid_test.py -j 6          # her RUN içinde TEST_LIST paralel (ProcessPoolExecutor)
  HYBRID_TEST_WORKERS=6 python hybrid_test.py

Çıktılar: results/hybrid_v3/
"""

import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import wilcoxon
import time, os, sys, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# IAOA'yı optimizers/ klasöründen yükle
# mealpy/ ile aynı seviyedeki optimizers/ klasörü
_OPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'optimizers')
if _OPT_DIR not in sys.path:
    sys.path.insert(0, _OPT_DIR)

from mealpy_comparison_v2 import (
    load_movielens,
    mkmeans_plus_plus_init,
    make_fitness_function,
    _compute_metrics,
    get_all_algorithms_v3,
    get_special_params,
)
from mealpy import FloatVar

# ============================================================
# AYARLAR
# ============================================================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
HYBRID_ROOT = os.path.join(BASE_DIR, 'results', 'hybrid_v3')   # ← yeni klasör
os.makedirs(HYBRID_ROOT, exist_ok=True)

DATA_PATH = os.path.join(os.path.dirname(BASE_DIR),
                         'data', 'ml-100k', 'u.data')

K              = 90
N_RUNS         = 5
POP_SIZE       = 30
GLOBAL_EPOCH   = 30
LOCAL_EPOCH    = 20
BASELINE_EPOCH = GLOBAL_EPOCH + LOCAL_EPOCH   # 50 — adil karşılaştırma
TIME_LIMIT     = 900

# Çok süreç: worker süreçlerinde algoritma haritası (pool_init ile doldurulur)
_WORKER_ALGO_MAP = None


# ============================================================
# IAOA'YI ALGO_MAP'E EKLEYEN YARDIMCI
# ============================================================

def _register_iaoa(algo_map):
    """IAOA'yı mealpy algo_map'ine kaydet."""
    try:
        from iaoa_optimizer import OriginalIAOA
        algo_map['IAOA.OriginalIAOA'] = {
            'full_name': 'IAOA.OriginalIAOA',
            'class':     OriginalIAOA,
        }
        return True
    except ImportError as e:
        print(f"UYARI: IAOA yüklenemedi → {e}")
        return False


# ============================================================
# ARGÜMAN PARSE
# ============================================================

def parse_hybrid_parallel_workers(argv=None):
    """-j / --workers > HYBRID_TEST_WORKERS > 1 (sıralı)."""
    p = argparse.ArgumentParser(description="Hibrit + baseline testi")
    p.add_argument(
        "-j", "--workers", type=int, default=None, metavar="N",
        help="Her RUN içinde TEST_LIST paralel süreç sayısı (1=sıralı).",
    )
    ns, _ = p.parse_known_args(argv if argv is not None else sys.argv[1:])
    if ns.workers is not None:
        return max(1, int(ns.workers))
    ev = os.environ.get("HYBRID_TEST_WORKERS", "").strip()
    if ev:
        return max(1, int(ev))
    return 1


# ============================================================
# WORKER INIT (paralel mod)
# ============================================================

def _remote_pool_init(script_dir):
    import sys
    d = script_dir or BASE_DIR
    if d not in sys.path:
        sys.path.insert(0, d)
    # optimizers/ klasörünü de ekle
    opt_dir = os.path.join(d, 'optimizers')
    if opt_dir not in sys.path:
        sys.path.insert(0, opt_dir)
    global _WORKER_ALGO_MAP
    _WORKER_ALGO_MAP = {a["full_name"]: a for a in get_all_algorithms_v3()}
    _register_iaoa(_WORKER_ALGO_MAP)


def _run_one_case_remote(task):
    """
    ProcessPoolExecutor worker. task:
    (label, g_name, l_name, matrix, K, init, baseline_ep, global_ep, local_ep, pop_size)
    """
    (
        label, g_name, l_name,
        matrix, K, init,
        baseline_ep, global_ep, local_ep, pop_size,
    ) = task

    algo_map = _WORKER_ALGO_MAP
    if algo_map is None:
        algo_map = {a["full_name"]: a for a in get_all_algorithms_v3()}
        _register_iaoa(algo_map)

    g_info = algo_map.get(g_name)
    if g_info is None:
        return {"ok": False, "label": label, "g_name": g_name,
                "l_name": l_name, "err": f"algoritma yok: {g_name}"}
    try:
        if l_name is None:
            r = run_single(g_info, matrix, K, init, baseline_ep, pop_size)
        else:
            l_info = algo_map.get(l_name)
            if l_info is None:
                return {"ok": False, "label": label, "g_name": g_name,
                        "l_name": l_name, "err": f"algoritma yok: {l_name}"}
            r = run_hybrid(g_info, l_info, matrix, K, init,
                           global_ep, local_ep, pop_size)
        return {"ok": True, "label": label, "g_name": g_name,
                "l_name": l_name, "r": r}
    except Exception as e:
        return {"ok": False, "label": label, "g_name": g_name,
                "l_name": l_name, "err": str(e)}


# ============================================================
# TEST LİSTESİ
# ============================================================

TEST_LIST = [
    # (label,           global_algo_fullname,    local_algo_fullname or None)
    ('H1_HHO+HGS',  'HHO.OriginalHHO',    'HGS.OriginalHGS'),
    ('H2_HGS+HHO',  'HGS.OriginalHGS',    'HHO.OriginalHHO'),
    ('H3_MFO+HGS',  'MFO.OriginalMFO',    'HGS.OriginalHGS'),
    ('H4_MFO+HHO',  'MFO.OriginalMFO',    'HHO.OriginalHHO'),
    ('H6_AGTO+HGS', 'AGTO.OriginalAGTO',  'HGS.OriginalHGS'),
    ('H5_EliteMultiGA+HHO', 'GA.EliteMultiGA', 'HHO.OriginalHHO'),

    # ──────────────────────────────────────────────────────────────────────
    ('B1_HHO',      'HHO.OriginalHHO',    None),
    ('B2_HGS',      'HGS.OriginalHGS',    None),
    ('B3_MFO',      'MFO.OriginalMFO',    None),
]

WILCOXON_PAIRS = [
    ('H1_HHO+HGS',  'B1_HHO',      'H1 vs HHO tek'),
    ('H1_HHO+HGS',  'B2_HGS',      'H1 vs HGS tek'),
    ('H2_HGS+HHO',  'B1_HHO',      'H2 vs HHO tek'),
    ('H2_HGS+HHO',  'B2_HGS',      'H2 vs HGS tek'),
    ('H3_MFO+HGS',  'B2_HGS',      'H3 vs HGS tek'),
    ('H3_MFO+HGS',  'B3_MFO',      'H3 vs MFO tek'),
    ('H4_MFO+HHO',  'B1_HHO',      'H4 vs HHO tek'),
    ('H6_AGTO+HGS', 'B2_HGS',      'H6 vs HGS tek'),
    ('H1_HHO+HGS',  'H2_HGS+HHO',  'Sıra farkı: H1 vs H2'),
    ('H3_MFO+HGS',  'H4_MFO+HHO',  'Local farkı: HGS vs HHO'),
    ('H1_HHO+HGS',  'H3_MFO+HGS',  'Global farkı: HHO vs MFO'),
    # ── YENİ ──────────────────────────────────────────────────────────────
    ('H7_IAOA+HHO', 'B1_HHO',      'H7 vs HHO tek'),
    ('H7_IAOA+HHO', 'H4_MFO+HHO',  'H7 vs H4: global IAOA vs MFO'),
    ('H8_MFO+IAOA', 'B3_MFO',      'H8 vs MFO tek'),
    ('H8_MFO+IAOA', 'H4_MFO+HHO',  'H8 vs H4: local IAOA vs HHO'),
    ('H7_IAOA+HHO', 'H8_MFO+IAOA', 'H7 vs H8: IAOA global vs IAOA local'),
    # ──────────────────────────────────────────────────────────────────────
    ('H5_EliteMultiGA+HHO', 'H4_MFO+HHO', 'GA global vs MFO global'),
    ('H5_EliteMultiGA+HHO', 'B1_HHO',     'H5 vs HHO tek'),
    ('H5_EliteMultiGA+HHO', 'B3_MFO',     'H5 vs MFO tek'),

]


# ============================================================
# PROBLEM TANIMI
# ============================================================

def make_problem(matrix, K):
    n_items = matrix.shape[1]
    return {
        "obj_func"       : make_fitness_function(matrix, K),
        "bounds"         : FloatVar(lb=[0.0] * (K * n_items),
                                    ub=[5.0] * (K * n_items)),
        "minmax"         : "min",
        "log_to"         : None,
        "save_population": False,
    }


# ============================================================
# TEK ALGORİTMA (BASELINE)
# ============================================================

def run_single(algo_info, matrix, K, init, epoch, pop_size):
    start   = time.time()
    problem = make_problem(matrix, K)
    sp      = get_special_params(algo_info['full_name'], epoch, pop_size)
    model   = algo_info['class'](**(sp or {'epoch': epoch, 'pop_size': pop_size}))
    try:
        model.solve(problem, starting_solutions=init[:pop_size])
    except TypeError:
        model.solve(problem)
    metrics = _compute_metrics(matrix, model.g_best.solution, K)
    return {
        'wcss'          : float(model.g_best.target.fitness),
        'time_seconds'  : time.time() - start,
        'local_improved': None,
        'global_wcss'   : None,
        **metrics,
    }


# ============================================================
# SIRADALI HİBRİT
# ============================================================

def run_hybrid(g_info, l_info, matrix, K, init,
               g_epoch, l_epoch, pop_size):
    start   = time.time()
    problem = make_problem(matrix, K)

    # Global aşama
    sp_g    = get_special_params(g_info['full_name'], g_epoch, pop_size)
    g_model = g_info['class'](**(sp_g or {'epoch': g_epoch, 'pop_size': pop_size}))
    try:
        g_model.solve(problem, starting_solutions=init[:pop_size])
    except TypeError:
        g_model.solve(problem)

    best_g_sol = g_model.g_best.solution
    best_g_fit = float(g_model.g_best.target.fitness)

    # Local aşama — global'in en iyi noktasından başla
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

    if best_l_fit < best_g_fit:
        best_sol      = l_model.g_best.solution
        best_fit      = best_l_fit
        local_improved = True
    else:
        best_sol       = best_g_sol
        best_fit       = best_g_fit
        local_improved = False

    metrics = _compute_metrics(matrix, best_sol, K)
    return {
        'wcss'          : best_fit,
        'global_wcss'   : best_g_fit,
        'local_improved': local_improved,
        'time_seconds'  : time.time() - start,
        **metrics,
    }


# ============================================================
# ANA DÖNGÜ
# ============================================================

def _make_result_row(run, label, g_name, l_name, r):
    return {
        "run"            : run,
        "label"          : label,
        "type"           : "baseline" if l_name is None else "hybrid",
        "global_algo"    : g_name.split(".")[0],
        "local_algo"     : l_name.split(".")[0] if l_name else "—",
        "wcss"           : r["wcss"],
        "global_wcss"    : r["global_wcss"],
        "local_improved" : r["local_improved"],
        "davies_bouldin" : r["davies_bouldin"],
        "silhouette"     : r["silhouette"],
        "gray_sheep_ratio": r["gray_sheep_ratio"],
        "time_seconds"   : r["time_seconds"],
    }


def run_all(full_matrix, algo_map, parallel_workers=1):
    pw = max(1, int(parallel_workers or 1))

    print(f"\n{'='*65}")
    print(f"HİBRİT TESTİ v3 — {N_RUNS} RUN  (H7+H8 dahil)")
    print(f"{'='*65}")
    print(f"8 hibrit + 3 baseline = {len(TEST_LIST)} koşucu")
    print(f"Global={GLOBAL_EPOCH}ep  Local={LOCAL_EPOCH}ep  "
          f"Baseline={BASELINE_EPOCH}ep  K={K}")
    if pw > 1:
        print(f"Paralel (ProcessPoolExecutor): {pw} süreç / RUN")
    else:
        print("Paralel: kapalı (sıralı) — -j N veya HYBRID_TEST_WORKERS")
    print(f"Klasör: {HYBRID_ROOT}/run_1 ... run_{N_RUNS}")
    print(f"Tahmini süre: ~{N_RUNS * len(TEST_LIST) * 3 / 60:.0f} dakika\n")

    all_rows    = []
    run_results = {t[0]: {'wcss': [], 'db': [], 'sil': []}
                   for t in TEST_LIST}

    for run in range(1, N_RUNS + 1):
        seed    = run - 1
        run_dir = os.path.join(HYBRID_ROOT, f'run_{run}')
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n{'─'*65}  RUN {run}/{N_RUNS}  (seed={seed})")
        print(f"  Klasör: {run_dir}")

        init = mkmeans_plus_plus_init(
            full_matrix, K=K, n_solutions=POP_SIZE + 10, seed=seed
        )

        run_rows = []

        if pw > 1:
            tasks = [
                (
                    label, g_name, l_name,
                    full_matrix, K, init,
                    BASELINE_EPOCH, GLOBAL_EPOCH, LOCAL_EPOCH, POP_SIZE,
                )
                for label, g_name, l_name in TEST_LIST
            ]
            with ProcessPoolExecutor(
                max_workers=pw,
                initializer=_remote_pool_init,
                initargs=(BASE_DIR,),
            ) as ex:
                remote_out = list(ex.map(_run_one_case_remote, tasks))

            for out in remote_out:
                label  = out["label"]
                g_name = out["g_name"]
                l_name = out["l_name"]
                if not out.get("ok"):
                    print(f"  {label:<18} HATA: {out.get('err', '?')}")
                    continue
                r   = out["r"]
                imp = (" ✓local"   if r["local_improved"] is True
                       else " →global" if r["local_improved"] is False
                       else "")
                print(f"  {label:<18} WCSS={r['wcss']:.3f}  "
                      f"DB={r['davies_bouldin']:.3f}  "
                      f"Sil={r['silhouette']:.4f}  "
                      f"{r['time_seconds']:.0f}s{imp}")
                run_results[label]["wcss"].append(r["wcss"])
                run_results[label]["db"].append(r["davies_bouldin"])
                run_results[label]["sil"].append(r["silhouette"])
                row = _make_result_row(run, label, g_name, l_name, r)
                all_rows.append(row)
                run_rows.append(row)

        else:
            for label, g_name, l_name in TEST_LIST:
                print(f"  {label:<18}", end=' ', flush=True)

                g_info = algo_map.get(g_name)
                if g_info is None:
                    print(f"HATA: {g_name} yok")
                    continue

                try:
                    if l_name is None:
                        r = run_single(g_info, full_matrix, K, init,
                                       BASELINE_EPOCH, POP_SIZE)
                    else:
                        l_info = algo_map.get(l_name)
                        if l_info is None:
                            print(f"HATA: {l_name} yok")
                            continue
                        r = run_hybrid(g_info, l_info, full_matrix, K, init,
                                       GLOBAL_EPOCH, LOCAL_EPOCH, POP_SIZE)
                except Exception as e:
                    print(f"HATA: {e}")
                    continue

                imp = (" ✓local"   if r['local_improved'] is True
                       else " →global" if r['local_improved'] is False
                       else "")
                print(f"WCSS={r['wcss']:.3f}  DB={r['davies_bouldin']:.3f}  "
                      f"Sil={r['silhouette']:.4f}  {r['time_seconds']:.0f}s{imp}")

                run_results[label]['wcss'].append(r['wcss'])
                run_results[label]['db'].append(r['davies_bouldin'])
                run_results[label]['sil'].append(r['silhouette'])

                row = _make_result_row(run, label, g_name, l_name, r)
                all_rows.append(row)
                run_rows.append(row)

        # Her run'ın sonuçlarını kendi klasörüne kaydet
        if run_rows:
            df_run = pd.DataFrame(run_rows)
            df_run.to_csv(os.path.join(run_dir, 'results.csv'), index=False)

            summary_lines = [
                f"Run {run}  seed={seed}  {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"{'Label':<18} {'WCSS':>8} {'DB':>7} {'Sil':>8} {'Süre':>7}  Tip",
                "-" * 65,
            ]
            for _, row in df_run.sort_values('wcss').iterrows():
                chk = " ✓" if row['local_improved'] is True else ""
                summary_lines.append(
                    f"{row['label']:<18} {row['wcss']:>8.3f} "
                    f"{row['davies_bouldin']:>7.3f} {row['silhouette']:>8.4f} "
                    f"{row['time_seconds']:>7.0f}s  {row['type']}{chk}"
                )
            with open(os.path.join(run_dir, 'summary.txt'), 'w',
                      encoding='utf-8') as f:
                f.write('\n'.join(summary_lines))
            print(f"  → Kaydedildi: {run_dir}/results.csv")

        # Tüm runların birleşik geçici kaydı
        pd.DataFrame(all_rows).to_csv(
            os.path.join(HYBRID_ROOT, 'all_runs_partial.csv'), index=False)

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(os.path.join(HYBRID_ROOT, 'all_runs.csv'), index=False)
    return df_all, run_results


# ============================================================
# ANALİZ + RAPOR
# ============================================================

def analyze_and_report(df_all, run_results, n_runs):

    rows = []
    for label, data in run_results.items():
        w = data['wcss']; d = data['db']; s = data['sil']
        if not w:
            continue
        sub  = df_all[df_all['label'] == label]
        limp = sub['local_improved'].dropna()
        t    = sub['time_seconds'].dropna()
        rows.append({
            'label'        : label,
            'type'         : sub['type'].iloc[0],
            'n'            : len(w),
            'wcss_mean'    : np.mean(w),  'wcss_std': np.std(w),
            'wcss_min'     : np.min(w),   'wcss_max': np.max(w),
            'db_mean'      : np.mean(d),  'db_std'  : np.std(d),
            'sil_mean'     : np.mean(s),
            'time_mean'    : np.mean(t) if len(t) > 0 else np.nan,
            'local_imp_pct': limp.mean() * 100 if len(limp) > 0 else np.nan,
        })

    df_sum = pd.DataFrame(rows).sort_values('wcss_mean')
    df_sum.to_csv(os.path.join(HYBRID_ROOT, 'summary.csv'), index=False)

    # Wilcoxon
    w_rows = []
    for a, b, note in WILCOXON_PAIRS:
        va = run_results.get(a, {}).get('wcss', [])
        vb = run_results.get(b, {}).get('wcss', [])
        n  = min(len(va), len(vb))
        if n < 5:
            continue
        try:
            _, p = wilcoxon(va[:n], vb[:n])
        except Exception:
            p = None
        ma, mb = np.mean(va), np.mean(vb)
        w_rows.append({
            'pair'   : f"{a} vs {b}",
            'note'   : note,
            'n'      : n,
            'mean_a' : ma, 'mean_b': mb,
            'diff'   : ma - mb,
            'p_value': p,
            'sig'    : 'EVET' if p is not None and p < 0.05 else 'HAYIR',
            'better' : a if ma < mb else b,
        })

    df_w = pd.DataFrame(w_rows)
    df_w.to_csv(os.path.join(HYBRID_ROOT, 'wilcoxon.csv'), index=False)

    # ── Rapor ──────────────────────────────────────────────────────────
    sep = "=" * 68
    lines = [
        sep,
        f"HİBRİT TEST RAPORU v3 — {n_runs} RUN  (H7+H8 dahil)",
        f"Global={GLOBAL_EPOCH}ep  Local={LOCAL_EPOCH}ep  "
        f"Baseline={BASELINE_EPOCH}ep  K={K}",
        sep,
    ]

    lines.append(f"\n{'Sıra':<4}{'Label':<18}{'Tip':<10}"
                 f"{'WCSS_ort':>9}  {'std':>6}{'DB_ort':>8}{'LocalImp':>10}")
    lines.append("-" * 68)
    for i, (_, r) in enumerate(df_sum.iterrows(), 1):
        limp = (f"{r['local_imp_pct']:.0f}%"
                if not np.isnan(r['local_imp_pct']) else "   —")
        lines.append(
            f"  {i:<2} {r['label']:<16} {r['type']:<10}"
            f"{r['wcss_mean']:>9.3f}  {r['wcss_std']:>6.3f}"
            f"{r['db_mean']:>8.3f}{limp:>10}"
        )

    lines += [f"\n{sep}", "WİLCOXON (WCSS)", sep]
    lines.append(f"\n{'Çift':<38}{'Fark':>8}{'p-val':>9}{'Anlam':>7}{'Kazanan':>18}")
    lines.append("-" * 82)
    for _, r in df_w.iterrows():
        lines.append(
            f"  {r['pair']:<38}{r['diff']:>+8.3f}"
            f"{r['p_value']:>9.4f}{r['sig']:>7}{r['better']:>18}"
        )
        lines.append(f"    → {r['note']}")

    hibs = df_sum[df_sum['type'] == 'hybrid']
    bls  = df_sum[df_sum['type'] == 'baseline']
    if len(hibs) > 0 and len(bls) > 0:
        bh = hibs.iloc[0]
        bb = bls.iloc[0]
        lines += [f"\n{sep}", "ÖZET", sep]
        lines.append(f"""
En iyi hibrit  : {bh['label']}  WCSS={bh['wcss_mean']:.3f}±{bh['wcss_std']:.3f}  DB={bh['db_mean']:.3f}
En iyi baseline: {bb['label']}  WCSS={bb['wcss_mean']:.3f}±{bb['wcss_std']:.3f}  DB={bb['db_mean']:.3f}

WCSS kazanımı: {bb['wcss_mean'] - bh['wcss_mean']:+.3f}  \
({'hibrit üstün ✓' if bh['wcss_mean'] < bb['wcss_mean'] else 'baseline üstün'})
DB   kazanımı: {bb['db_mean'] - bh['db_mean']:+.3f}  \
({'hibrit üstün ✓' if bh['db_mean'] < bh['db_mean'] else 'baseline üstün'})

H7_IAOA+HHO  : WCSS={run_results.get('H7_IAOA+HHO', {}).get('wcss', [None])[0] or 'N/A'}
H8_MFO+IAOA  : WCSS={run_results.get('H8_MFO+IAOA', {}).get('wcss', [None])[0] or 'N/A'}
""")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HYBRID_ROOT, 'report.txt'), 'w',
              encoding='utf-8') as f:
        f.write(text)

    print(f"\nDosyalar → {HYBRID_ROOT}/")
    print("  all_runs.csv   summary.csv   wilcoxon.csv   report.txt")
    print(f"  run_1/ ... run_{n_runs}/  (her run ayrı klasörde)")


# ============================================================
# ANA
# ============================================================

if __name__ == "__main__":
    parallel_workers = parse_hybrid_parallel_workers()
    print("Yükleniyor...")
    full_matrix = load_movielens(DATA_PATH)
    all_algos   = get_all_algorithms_v3()
    algo_map    = {a['full_name']: a for a in all_algos}

    # IAOA'yı kaydet
    iaoa_ok = _register_iaoa(algo_map)

    needed  = [
        'HHO.OriginalHHO', 'HGS.OriginalHGS', 'MFO.OriginalMFO',
        'OOA.OriginalOOA', 'AGTO.OriginalAGTO',
        'IAOA.OriginalIAOA',
    ]
    missing = [n for n in needed if n not in algo_map]
    if missing:
        print(f"UYARI — eksik algoritmalar: {missing}")
        if 'IAOA.OriginalIAOA' in missing:
            print("  → optimizers/iaoa_optimizer.py dosyasını kontrol et")
    else:
        print(f"✓ Tüm algoritmalar mevcut (IAOA dahil, {len(algo_map)} toplam)")

    print(
        f"Paralel işçi: {parallel_workers}  "
        f"(HYBRID_TEST_WORKERS={os.environ.get('HYBRID_TEST_WORKERS', '')!r})"
    )
    print(f"Sonuçlar → {HYBRID_ROOT}/")

    df_all, run_results = run_all(full_matrix, algo_map, parallel_workers)

    print("\nAnaliz...")
    analyze_and_report(df_all, run_results, N_RUNS)