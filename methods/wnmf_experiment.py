"""
wnmf_experiment.py
==================
Küme ataması kaynakları: baseline + hibrit + literatür meta-sezgiselleri (LIT_*)
için WNMF karşılaştırma deneyi.

3 senaryo:
    global       — kümeleme yok, tüm kullanıcılar tek model (ablation baseline)
    cluster_full — her küme hem U hem V öğrenir (eski versiyon, karşılaştırma için)
    cluster_sharedV — global V + küme bazlı U (önerilen versiyon)

Kullanım:
    python methods/wnmf_experiment.py --dataset 100k
    python methods/wnmf_experiment.py --dataset 100k --algo H4_MFO+HHO
    python methods/wnmf_experiment.py --dataset 100k --algo H9_QSA+CDO H12_MFO+CDO
    python methods/wnmf_experiment.py --dataset 100k --algo LIT_GOA LIT_GWO LIT_SSA
    python methods/wnmf_experiment.py --dataset 100k --no-global
    python methods/wnmf_experiment.py --dataset 100k --mode sharedV   # sadece sharedV
    python methods/wnmf_experiment.py --dataset 100k --mode all       # her iki mod
    python methods/wnmf_experiment.py --dataset 100k --jobs 4        # küme başına 4 süreç (-j 4)
    python methods/wnmf_experiment.py --dataset 100k --algo-jobs 4   # algo başına 4 süreç (yeni)
    python methods/wnmf_experiment.py --dataset 100k --algo-jobs 0   # algo-düzeyinde otomatik paralel (yeni)
    python methods/wnmf_experiment.py --dataset 100k --k 70        # assignment ..._k70 ile aynı K
    python methods/wnmf_experiment.py --dataset 100k --k 20 30 50 90  # birden fazla K sırayla (tek komut)
    python methods/wnmf_experiment.py --dataset 100k --k 30 --no-global --latent-dim 10 20 50 100 --algo H4_MFO+HHO

Birden fazla K ve latent + atlanmış koşular (assignment suffix'teki wnmf boyutunu latent ile eşle):
    python methods/wnmf_experiment.py --dataset 100k --mode sharedV --k 20 30 70 \\
        --latent-dim 10 40 \\
        --assign-suffix _pruneu5_i10_zscore_wnmf20_inmed_trim5_95_fuzzy \\
        --sync-assign-suffix-latent --skip-existing \\
        --assign-root mealpy/results/assignments_lof
    python methods/wnmf_experiment.py --dataset 100k --k 30 --no-global --reg 0.001 0.01 0.1 --lr 0.001 0.01
    python methods/wnmf_experiment.py --dataset 100k --k 30 --epochs-global 50 100 150 200 --epochs-cluster 25 50 75 100
    python methods/wnmf_experiment.py --dataset 1m --k 30 --algo H4_MFO+HHO --epochs-grid \\
        --epochs-global 50 100 150 --epochs-cluster 50 75 100 150 200
    python methods/wnmf_experiment.py --dataset both --k 30 70 --algo H4_MFO+HHO --compare-mf-svdpp
    python methods/wnmf_experiment.py --dataset 1m --k 70            # .../B1_HHO_k70/ vb.
    python methods/wnmf_experiment.py --k-100k 90 --k-1m 70        # dataset başına ayrı K (--k çoklu ile birlikte kullanılmaz)

Çıktı dizinleri (üstüne yazmaz; her koşu yeni run klasörü):
    results/wnmf/ml100k/k70/run1/wnmf_results_ml100k_k70_sharedV.csv
    --mode all → aynı klasörde ek olarak all_split_full.csv ve all_split_sharedV.csv (GLOBAL her ikisinde)
    results/wnmf/ml1m/k150/run2/...
    --dataset both → ek olarak: results/wnmf/combined/ml100k_k70__ml1m_k150__mode-sharedV/run1/...
"""

import argparse
import os
import re
import shlex
import sys
import sys as _sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(BASE_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, BASE_DIR)

try:
    from assignment_db import (
        start_run,
        finish_run,
        save_wnmf_result,
        init_db,
        get_assignment_id,
    )
    init_db()
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False
_CURRENT_RUN_ID = None

from models.wnmf import ClusterWNMF, WNMFModel, WNMFSharedV
from core.loaders import load_ratings_100k, load_ratings_1m
from core.utils import (
    load_assignment,
    load_memberships,
    split_by_cluster,
    remap_user_ids,
    save_dataframe_csv,
    save_results,
)
from wnmf_experiment_baselines import run_cluster_average, run_cluster_knn
from core.metrics import (
    compute_metrics as _compute_metrics,
    compute_topn_metrics as _compute_topn_metrics,
)

# ============================================================
# AYARLAR
# ============================================================

DATA_100K_TRAIN = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-100k', 'u1.base')
DATA_100K_TEST  = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-100k', 'u1.test')
DATA_1M         = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-1m', 'ratings.dat')
ASSIGN_ROOT     = os.path.join(os.path.dirname(BASE_DIR), 'mealpy','results', 'assignments_lof')
OUT_ROOT        = os.path.join(os.path.dirname(BASE_DIR), 'results', 'wnmf')

# generate_assignments.py ile aynı: K bu değerlerden biriyse klasörde _k{K} eki yok
ASSIGN_K_DEFAULT_100K = 30
ASSIGN_K_DEFAULT_1M   = 70

LATENT_DIM       = 20
LEARNING_RATE    = 0.01
REGULARIZATION   = 0.01
N_EPOCHS_GLOBAL  = 100   # global V ve global baseline için
N_EPOCHS_CLUSTER = 100   # küme U eğitimi için
RANDOM_SEED      = 42

ALGO_LABELS = [
    'B0_KMEANS',
    'B1_HHO', 'B2_HGS', 'H1_HHO+HGS', 'H4_MFO+HHO',
    'H9_QSA+CDO', 'H12_MFO+CDO', 'H13_HHO+GAop',
    'DE_HHO',
    # generate_assignments.py ALGO_CONFIG — önce --lof ile atama üretin
    'LIT_GOA', 'LIT_GWO', 'LIT_SSA',
]

# Yaygın yazım hatası: --algo H1_HGS+HHO → H1_HHO+HGS
ALGO_ALIASES = {
    'H1_HGS+HHO': 'H1_HHO+HGS',
}


def _expand_epoch_pairs(
    eg: List[int],
    ec: List[int],
) -> List[Tuple[int, int]]:
    """
    --epochs-global ve --epochs-cluster listelerinden (eg, ec) çiftleri.
    Uzunluklar eşitse zip; biri 1 elemanlıysa diğerine yayılır.
    """
    if len(eg) == len(ec):
        return list(zip(eg, ec))
    if len(eg) == 1:
        return [(eg[0], c) for c in ec]
    if len(ec) == 1:
        return [(g, ec[0]) for g in eg]
    raise ValueError(
        f"--epochs-global ({len(eg)} değer) ile --epochs-cluster ({len(ec)} değer) "
        "uyumsuz; birini tek bırakın veya uzunlukları eşitleyin. "
        "Tüm (global×cluster) çiftleri için --epochs-grid kullanın."
    )


def _epoch_cartesian_pairs(eg: List[int], ec: List[int]) -> List[Tuple[int, int]]:
    """Tüm (epochs_global, epochs_cluster) kombinasyonları."""
    return list(product(eg, ec))


def _format_hyperparam_tag(k_used: int) -> str:
    """CSV / dosya adlarında hangi hiperparametre koşusunun olduğunu gösterir."""
    return (
        f"k{k_used}_ld{LATENT_DIM}_eg{N_EPOCHS_GLOBAL}_ec{N_EPOCHS_CLUSTER}_"
        f"lr{LEARNING_RATE:g}_r{REGULARIZATION:g}"
    )


def _result_row_meta(k_used: int) -> dict:
    return {
        'assignment_k'    : k_used,
        'latent_dim'      : LATENT_DIM,
        'learning_rate'   : LEARNING_RATE,
        'regularization'  : REGULARIZATION,
        'epochs_global'   : N_EPOCHS_GLOBAL,
        'epochs_cluster'  : N_EPOCHS_CLUSTER,
        'hyperparam_tag'  : _format_hyperparam_tag(k_used),
    }


def _sync_assign_suffix_with_latent(suffix: str, latent_dim: int) -> str:
    """
    --assign-suffix içindeki ilk _wnmf<RAKAM> parçasını mevcut latent_dim ile değiştirir.
    Örn. ..._wnmf20_... + ld=40 -> ..._wnmf40_...
    Eşleşme yoksa suffix aynen döner.
    """
    if not suffix:
        return suffix
    new_s, n = re.subn(r'(_wnmf)\d+', rf'\g<1>{int(latent_dim)}', suffix, count=1)
    return new_s if n else suffix


def _effective_assign_suffix(args) -> str:
    """CLI'den gelen taban suffix + isteğe bağlı latent ile senkron."""
    base = getattr(args, 'assign_suffix_cli', '') or ''
    if getattr(args, 'sync_assign_suffix_latent', False):
        return _sync_assign_suffix_with_latent(base, LATENT_DIM)
    return base


def _normalize_db_dataset_name(dataset_name: str) -> str:
    """DB'deki dataset anahtarlarıyla uyumlu ad döndür."""
    ds = (dataset_name or '').strip().lower()
    if ds == '100k':
        return 'ml100k'
    if ds == '1m':
        return 'ml1m'
    return ds


def _normalize_db_preprocessing(prep: str) -> str:
    """
    DB eşleşmesi için preprocessing'i standardize et.
    Örn: pruneu5... -> prune_u5..., çoklu '_' temizliği.
    """
    p = (prep or '').strip().lower().lstrip('_')
    if not p:
        return 'none'
    p = re.sub(r'^pruneu(?=\d)', 'prune_u', p)
    p = re.sub(r'__+', '_', p).strip('_')
    return p or 'none'


def _db_preprocessing_candidates(prep: str) -> List[str]:
    """
    get_assignment_id için olası preprocessing adayları.
    Bazı eski kayıtlarda _fuzzy veya prune_u yazımı farklı olabiliyor.
    """
    base_raw = (prep or '').strip().lstrip('_')
    base = _normalize_db_preprocessing(base_raw)
    cands: List[str] = []
    for p in (base, base_raw):
        pn = _normalize_db_preprocessing(p)
        if pn not in cands:
            cands.append(pn)
        if pn.endswith('_fuzzy'):
            pf = pn[:-6]
            if pf and pf not in cands:
                cands.append(pf)
        else:
            pf = f'{pn}_fuzzy'
            if pf not in cands:
                cands.append(pf)
    return cands


def _iter_cv_mean_result_paths(dataset_name: str, k_used: int, mode: str):
    """cv5_mean/run*/ altındaki ortalama CSV yolları."""
    root = os.path.join(OUT_ROOT, dataset_name, f'k{k_used}', 'cv5_mean')
    if not os.path.isdir(root):
        return
    fname = f'wnmf_results_{dataset_name}_k{k_used}_{mode}_cv5_mean.csv'
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, fname)
        if os.path.isfile(p):
            yield p


def _iter_holdout_result_paths(dataset_name: str, k_used: int, mode: str, fold: Optional[int]):
    """Tek holdout / fold koşusu: k.../run*/ veya k.../fold{f}/run*/ altındaki ana sonuç CSV."""
    if fold is None:
        kd = os.path.join(OUT_ROOT, dataset_name, f'k{k_used}')
    else:
        kd = os.path.join(OUT_ROOT, dataset_name, f'k{k_used}', f'fold{fold}')
    if not os.path.isdir(kd):
        return
    fname = f'wnmf_results_{dataset_name}_k{k_used}_{mode}.csv'
    for name in sorted(os.listdir(kd)):
        if not name.lower().startswith('run'):
            continue
        p = os.path.join(kd, name, fname)
        if os.path.isfile(p):
            yield p


def _csv_has_hyperparam_combo(path: str, hyper_tag: str, use_svdpp: bool) -> bool:
    """CSV'de aynı hyperparam_tag (+ varsa use_svdpp) satırı var mı."""
    try:
        df = pd.read_csv(path, encoding='utf-8', comment='#')
    except Exception:
        return False
    if 'hyperparam_tag' not in df.columns:
        return False
    m = df['hyperparam_tag'].astype(str) == str(hyper_tag)
    if 'use_svdpp' in df.columns:
        m = m & (df['use_svdpp'].astype(bool) == bool(use_svdpp))
    return bool(m.any())


def _should_skip_existing_run(
    args,
    dataset_name: str,
    k_used: int,
    mode: str,
    fold_arg: Optional[int],
    use_sp: bool,
    use_cv_mean_branch: bool,
) -> bool:
    """
    --skip-existing: Önceki koşuda aynı hiperparametre etiketi yazılmışsa bu (dataset, K)
    denemesini atla.
    """
    if not getattr(args, 'skip_existing', False):
        return False
    tag = _format_hyperparam_tag(k_used)
    if use_cv_mean_branch:
        for p in _iter_cv_mean_result_paths(dataset_name, k_used, mode):
            if _csv_has_hyperparam_combo(p, tag, use_sp):
                return True
        return False
    for p in _iter_holdout_result_paths(dataset_name, k_used, mode, fold_arg):
        if _csv_has_hyperparam_combo(p, tag, use_sp):
            return True
    return False


def _resolved_assignment_k(
    k_explicit_dataset: Optional[int],
    k_global: Optional[int],
    default_k: int,
) -> int:
    """Hangi K ile üretilmiş assignment klasörü okunacak (generate_assignments ile uyumlu)."""
    if k_explicit_dataset is not None:
        return k_explicit_dataset
    if k_global is not None:
        return k_global
    return default_k


def _algo_assignment_dir(
    assign_root: str,
    dataset_name: str,
    label: str,
    k_used: int,
    assign_suffix: str = '',
) -> str:
    """
    Örnek: k_used=70, default 90 ise → .../ml100k/B1_HHO_k70
    k_used=90, default 90 ise → .../ml100k/B1_HHO
    assign_suffix: örn. _wnmf20, _zscore (generate_assignments ile aynı klasör adları)
    """
    default_k = ASSIGN_K_DEFAULT_100K if dataset_name == 'ml100k' else ASSIGN_K_DEFAULT_1M
    suffix    = '' if k_used == default_k else f'_k{k_used}'
    return os.path.join(assign_root, dataset_name, f'{label}{suffix}{assign_suffix}')


def _next_run_index(k_dir: str) -> int:
    """
    k_dir altında run1, run2, ... klasörlerine bakıp bir sonraki run numarasını döndürür.
    İlk koşu: 1 (klasör yoksa veya run* yoksa).
    """
    if not os.path.isdir(k_dir):
        return 1
    best = 0
    for name in os.listdir(k_dir):
        m = re.fullmatch(r'run(\d+)', name, flags=re.IGNORECASE)
        if m and os.path.isdir(os.path.join(k_dir, name)):
            best = max(best, int(m.group(1)))
    return best + 1


def _resolve_pool_workers(requested: Optional[int], n_tasks: int) -> int:
    """İş sayısını ve CPU’yu aşmayacak worker sayısı."""
    if n_tasks <= 0:
        return 1
    cpu = os.cpu_count() or 1
    if requested is None or requested <= 0:
        cap = cpu
    else:
        cap = requested
    return max(1, min(cap, n_tasks))


def _parallel_cluster_map(func, jobs: list, max_workers: Optional[int]):
    """Sıra korunur (ex.map). Tek iş / tek worker ise süreç havuzu kurulmaz."""
    if not jobs:
        return []
    nw = _resolve_pool_workers(max_workers, len(jobs))
    if nw == 1:
        return [func(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=nw) as pool:
        return list(pool.map(func, jobs))


def _mp_fit_predict_cluster_full(job):
    """
    Pickle edilebilir küme işi — full WNMF (U+V).
    job: (c_train, c_test, n_items, latent_dim, lr, reg, n_epochs, random_seed, use_svdpp)
    """
    (
        c_train,
        c_test,
        n_items,
        latent_dim,
        lr,
        reg,
        n_epochs,
        random_seed,
        use_svdpp,
    ) = job
    c_train_r, c_test_r, _, n_loc = remap_user_ids(c_train, c_test, n_items)
    model = WNMFModel(
        n_users        = n_loc,
        n_items        = n_items,
        latent_dim     = latent_dim,
        learning_rate  = lr,
        regularization = reg,
        n_epochs       = n_epochs,
        random_seed    = random_seed,
        use_svdpp      = use_svdpp,
    )
    model.fit(c_train_r)
    if len(c_test_r) == 0:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
        )
    pred = model.predict(
        c_test_r[:, 0].astype(np.int32),
        c_test_r[:, 1].astype(np.int32),
    )
    # remap_user_ids train'de görülmeyen kullanıcıları testten filtreler.
    test_mask = np.isin(c_test[:, 0].astype(np.int32), np.unique(c_train[:, 0].astype(np.int32)))
    c_test_filtered = c_test[test_mask]
    eval_rows = np.column_stack((
        c_test_filtered[:, 0].astype(np.float32),
        c_test_filtered[:, 1].astype(np.float32),
        c_test_filtered[:, 2].astype(np.float32),
        pred.astype(np.float32),
    ))
    return c_test_r[:, 2].astype(np.float32), pred.astype(np.float32), eval_rows


def _mp_fit_predict_cluster_sharedV(job):
    """
    Pickle edilebilir küme işi — sabit V, sadece U.
    job (14): (cid, V_global, mu, b_i_global, c_train, c_test, n_items,
              latent_dim, lr, reg, n_epochs, random_seed_base, use_bias, use_cluster_bias)
    job (10, eski): (cid, V_global, c_train, c_test, n_items, latent_dim,
                     lr, reg, n_epochs, random_seed_base) — mu/b_i/use_bias varsayılan
    """
    if len(job) == 14:
        (
            cid,
            V_global,
            mu,
            b_i_global,
            c_train,
            c_test,
            n_items,
            latent_dim,
            lr,
            reg,
            n_epochs,
            random_seed_base,
            use_bias,
            use_cluster_bias,
        ) = job
    elif len(job) == 10:
        (
            cid,
            V_global,
            c_train,
            c_test,
            n_items,
            latent_dim,
            lr,
            reg,
            n_epochs,
            random_seed_base,
        ) = job
        mu           = 0.0
        b_i_global   = np.zeros(int(n_items), dtype=np.float32)
        use_bias     = True
        use_cluster_bias = False
    else:
        raise ValueError(
            f"_mp_fit_predict_cluster_sharedV: job uzunluğu {len(job)}; 10 veya 14 beklenir."
        )
    c_train_r, c_test_r, _, n_loc = remap_user_ids(c_train, c_test, n_items)
    cluster_model = ClusterWNMF(
        n_users        = n_loc,
        n_items        = n_items,
        latent_dim     = latent_dim,
        V_shared       = V_global,
        learning_rate  = lr,
        regularization = reg,
        n_epochs       = n_epochs,
        random_seed    = int(random_seed_base) + int(cid),
        use_bias       = use_bias,
        mu             = mu,
        b_i_global     = b_i_global,
        cluster_ratings= c_train_r if use_cluster_bias else None,
    )
    cluster_model.fit_cluster_U(c_train_r)
    if len(c_test_r) == 0:
        return (
            cid,
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
        )
    pred = cluster_model.predict(
        c_test_r[:, 0].astype(np.int32),
        c_test_r[:, 1].astype(np.int32),
    )
    test_mask = np.isin(c_test[:, 0].astype(np.int32), np.unique(c_train[:, 0].astype(np.int32)))
    c_test_filtered = c_test[test_mask]
    eval_rows = np.column_stack((
        c_test_filtered[:, 0].astype(np.float32),
        c_test_filtered[:, 1].astype(np.float32),
        c_test_filtered[:, 2].astype(np.float32),
        pred.astype(np.float32),
    ))
    return (
        cid,
        c_test_r[:, 2].astype(np.float32),
        pred.astype(np.float32),
        eval_rows,
    )


def _predict_full_profile(profile: dict, user_id: int, item_id: int) -> float:
    model = profile['model']
    uid_map = profile['uid_map']
    loc_u = uid_map.get(int(user_id))
    if loc_u is not None:
        return float(model.predict(
            np.array([loc_u], dtype=np.int32),
            np.array([item_id], dtype=np.int32),
        )[0])
    mean_u = profile['mean_u']
    if model.use_bias:
        pred = (
            float(model.mu)
            + float(model.b_i[item_id])
            + float(np.dot(mean_u, model.V[item_id]))
        )
    else:
        pred = float(np.dot(mean_u, model.V[item_id]))
    return float(np.clip(pred, 1.0, 5.0))


def _predict_sharedv_profile(profile: dict, user_id: int, item_id: int) -> float:
    model = profile['model']
    uid_map = profile['uid_map']
    loc_u = uid_map.get(int(user_id))
    if loc_u is not None:
        return float(model.predict(
            np.array([loc_u], dtype=np.int32),
            np.array([item_id], dtype=np.int32),
        )[0])
    mean_u = profile['mean_u']
    if model.use_bias:
        pred = (
            float(model.mu_k)
            + float(model.b_i[item_id])
            + float(np.dot(mean_u, model.V[item_id]))
        )
    else:
        pred = float(np.dot(mean_u, model.V[item_id]))
    return float(np.clip(pred, 1.0, 5.0))


def _membership_weighted_prediction(
    memberships_row: np.ndarray,
    profiles: dict,
    predictor_fn,
    user_id: int,
    item_id: int,
) -> float:
    num = 0.0
    den = 0.0
    for cid, w in enumerate(np.asarray(memberships_row, dtype=np.float64)):
        if w <= 0:
            continue
        profile = profiles.get(int(cid))
        if profile is None:
            continue
        pred_k = predictor_fn(profile, int(user_id), int(item_id))
        num += float(w) * float(pred_k)
        den += float(w)
    if den <= 0:
        return float('nan')
    return float(np.clip(num / den, 1.0, 5.0))


# ============================================================
# SENARYO 1: GLOBAL WNMF BASELINE
# ============================================================

def run_global_wnmf(train, test, n_items, verbose=False, use_bias=True,
                    use_svdpp: bool = False, top_n: int = 10,
                    relevance_threshold: float = 4.0):
    """
    Tüm kullanıcılara tek model — ablation baseline.
    'Kümeleme eklemek ne kadar iyileştiriyor?' sorusunu cevaplar.
    """
    print("\n  [Global WNMF] başlıyor...")
    t0      = time.time()
    n_users = int(train[:, 0].max()) + 1

    model = WNMFModel(
        n_users        = n_users,
        n_items        = n_items,
        latent_dim     = LATENT_DIM,
        learning_rate  = LEARNING_RATE,
        regularization = REGULARIZATION,
        n_epochs       = N_EPOCHS_GLOBAL,
        random_seed    = RANDOM_SEED,
        use_bias       = use_bias,
        use_svdpp      = use_svdpp,
    )
    model.fit(train, verbose=verbose)
    mae, rmse = model.evaluate(test)
    pred = model.predict(
        test[:, 0].astype(np.int32),
        test[:, 1].astype(np.int32),
    )
    eval_rows = np.column_stack((test[:, 0], test[:, 1], test[:, 2], pred))
    precision, recall, f1, ndcg = _compute_topn_metrics(
        eval_rows, top_n=top_n, threshold=relevance_threshold,
    )

    print(f"  [Global WNMF] MAE={mae:.4f}  RMSE={rmse:.4f}  ({time.time()-t0:.1f}s)")
    return {
        'scenario'    : 'global',
        'algo_label'  : 'GLOBAL_WNMF',
        'mae'         : mae,
        'rmse'        : rmse,
        'gray_mae'    : float('nan'),
        'gray_rmse'   : float('nan'),
        'white_mae'   : mae,
        'white_rmse'  : rmse,
        'n_clusters'  : 1,
        'n_train'     : len(train),
        'n_test'      : len(test),
        'time_seconds': time.time() - t0,
        'accuracy'        : float('nan'),
        'precision_at_10' : precision,
        'recall_at_10'    : recall,
        'f1_at_10'        : f1,
        'ndcg_at_10'      : ndcg,
    }


# ============================================================
# SENARYO 2: CLUSTER WNMF — HER KÜME U+V ÖĞRENİR
# ============================================================

def run_cluster_full(train, test, assignments, gray_mask, memberships,
                     n_items, algo_label, verbose=False,
                     max_workers: Optional[int] = None,
                     use_svdpp: bool = False,
                     top_n: int = 10,
                     relevance_threshold: float = 4.0):
    """
    Her küme bağımsız U ve V öğrenir.
    Karşılaştırma için tutulur — SharedV ile farkı görmek için.
    """
    print(f"\n  [{algo_label} | Full Cluster] başlıyor...")
    t0 = time.time()

    cluster_train, gray_train = split_by_cluster(train, assignments, gray_mask)
    cluster_test,  gray_test  = split_by_cluster(test,  assignments, gray_mask)

    use_soft = (
        memberships is not None
        and memberships.shape[0] >= len(assignments)
        and memberships.shape[1] >= (int(assignments.max()) + 1)
    )

    if not use_soft:
        jobs = []
        for cid, c_train in cluster_train.items():
            c_test = cluster_test.get(cid, np.empty((0, 3), dtype=np.float32))
            if len(c_train) < 5 or len(c_test) == 0:
                continue
            jobs.append(
                (
                    c_train,
                    c_test,
                    n_items,
                    LATENT_DIM,
                    LEARNING_RATE,
                    REGULARIZATION,
                    N_EPOCHS_CLUSTER,
                    RANDOM_SEED,
                    use_svdpp,
                )
            )

        parts = _parallel_cluster_map(_mp_fit_predict_cluster_full, jobs, max_workers)
        true_chunks = [p[0] for p in parts if len(p[0]) > 0]
        pred_chunks = [p[1] for p in parts if len(p[1]) > 0]
        eval_chunks = [p[2] for p in parts if len(p[2]) > 0]
        if true_chunks:
            all_true = np.concatenate(true_chunks).tolist()
            all_pred = np.concatenate(pred_chunks).tolist()
        else:
            all_true, all_pred = [], []
        if eval_chunks:
            eval_rows = np.concatenate(eval_chunks, axis=0)
        else:
            eval_rows = np.zeros((0, 4), dtype=np.float32)
    else:
        profiles = {}
        for cid, c_train in cluster_train.items():
            if len(c_train) < 5:
                continue
            c_train_r, _, uid_map, n_loc = remap_user_ids(
                c_train,
                np.empty((0, 3), dtype=np.float32),
                n_items,
            )
            model = WNMFModel(
                n_users=n_loc,
                n_items=n_items,
                latent_dim=LATENT_DIM,
                learning_rate=LEARNING_RATE,
                regularization=REGULARIZATION,
                n_epochs=N_EPOCHS_CLUSTER,
                random_seed=RANDOM_SEED + int(cid),
                use_svdpp=use_svdpp,
            )
            model.fit(c_train_r)
            profiles[int(cid)] = {
                'model': model,
                'uid_map': uid_map,
                'mean_u': model.U.mean(axis=0).astype(np.float32),
            }

        eval_rows_list = []
        for row in test:
            u, i, r = int(row[0]), int(row[1]), float(row[2])
            if gray_mask[u]:
                continue
            pred = _membership_weighted_prediction(
                memberships[u], profiles, _predict_full_profile, u, i,
            )
            if np.isnan(pred):
                hard_profile = profiles.get(int(assignments[u]))
                if hard_profile is None:
                    continue
                pred = _predict_full_profile(hard_profile, u, i)
            eval_rows_list.append((u, i, r, pred))

        eval_rows = (
            np.array(eval_rows_list, dtype=np.float32)
            if eval_rows_list else np.zeros((0, 4), dtype=np.float32)
        )
        all_true = eval_rows[:, 2].tolist() if len(eval_rows) > 0 else []
        all_pred = eval_rows[:, 3].tolist() if len(eval_rows) > 0 else []

    mae, rmse = _compute_metrics(all_true, all_pred)
    precision, recall, f1, ndcg = _compute_topn_metrics(
        eval_rows, top_n=top_n, threshold=relevance_threshold,
    )
    gray_mae, gray_rmse = _run_gray_sheep(
        gray_train, gray_test, n_items, 'full', use_svdpp=use_svdpp,
    )

    elapsed = time.time() - t0
    print(f"  [{algo_label} | Full] MAE={mae:.4f} RMSE={rmse:.4f} | "
          f"Gray MAE={gray_mae:.4f}  ({elapsed:.1f}s)")

    return {
        'scenario'    : 'cluster_full',
        'algo_label'  : algo_label,
        'mae'         : mae,
        'rmse'        : rmse,
        'gray_mae'    : gray_mae,
        'gray_rmse'   : gray_rmse,
        'n_clusters'  : len(cluster_train),
        'n_train'     : len(train),
        'n_test'      : len(test),
        'time_seconds': elapsed,
        'accuracy'        : float('nan'),
        'precision_at_10' : precision,
        'recall_at_10'    : recall,
        'f1_at_10'        : f1,
        'ndcg_at_10'      : ndcg,
    }


# ============================================================
# SENARYO 3: CLUSTER WNMF — SHARED V (ÖNERİLEN)
# ============================================================

def run_cluster_sharedV(train, test, assignments, gray_mask, memberships,
                        n_items, algo_label, verbose=False,
                        max_workers: Optional[int] = None,
                        out_dir: Optional[str] = None,
                        weighted_v: bool = False,
                        use_bias: bool = True,
                        use_cluster_bias: bool = False,
                        use_svdpp: bool = False,
                        run_command: Optional[str] = None,
                        top_n: int = 10,
                        relevance_threshold: float = 4.0):
    """
    Global V + küme bazlı U.

    Aşama 1: Tüm train verisiyle global V öğren (~global WNMF kadar sürer)
    Aşama 2: Her küme için V sabit, sadece U eğit (çok hızlı)

    Bu yaklaşımın avantajı:
    - Film embedding'leri tüm veriden öğrenilir → kaliteli
    - Kullanıcı embedding'leri küme özelinde → uzmanlaşmış
    - Az verili kümelerde bile iyi sonuç verir
    """
    print(f"\n  [{algo_label} | SharedV] başlıyor...")
    t0      = time.time()
    n_users = int(train[:, 0].max()) + 1

    # Aşama 1: Global V öğren
    shared = WNMFSharedV(
        n_users_global  = n_users,
        n_items         = n_items,
        latent_dim      = LATENT_DIM,
        learning_rate   = LEARNING_RATE,
        regularization  = REGULARIZATION,
        n_epochs_global = N_EPOCHS_GLOBAL,
        random_seed     = RANDOM_SEED,
        use_bias        = use_bias,
        use_svdpp       = use_svdpp,
    )
    if weighted_v:
        full_gray_mask = np.zeros(int(train[:, 0].max()) + 1, dtype=bool)
        gray_user_ids  = np.unique(
            train[np.isin(train[:, 0], np.where(gray_mask)[0]), 0]
        ).astype(int)
        full_gray_mask[gray_user_ids] = True
        shared.fit_global_V(train, gray_mask=full_gray_mask, verbose=verbose)
    else:
        shared.fit_global_V(train, verbose=verbose)
    V_global = shared.V.copy()

    # Kümeleri böl
    cluster_train, gray_train = split_by_cluster(train, assignments, gray_mask)
    cluster_test,  gray_test  = split_by_cluster(test,  assignments, gray_mask)

    use_soft = (
        memberships is not None
        and memberships.shape[0] >= len(assignments)
        and memberships.shape[1] >= (int(assignments.max()) + 1)
    )

    if not use_soft:
        jobs = []
        for cid, c_train in cluster_train.items():
            c_test = cluster_test.get(cid, np.empty((0, 3), dtype=np.float32))
            if len(c_train) < 5 or len(c_test) == 0:
                continue
            jobs.append(
                (
                    cid,
                    V_global,
                    float(shared.mu),
                    shared.b_i.copy(),
                    c_train,
                    c_test,
                    n_items,
                    LATENT_DIM,
                    LEARNING_RATE,
                    REGULARIZATION,
                    N_EPOCHS_CLUSTER,
                    RANDOM_SEED,
                    use_bias,
                    use_cluster_bias,
                )
            )

        parts = _parallel_cluster_map(_mp_fit_predict_cluster_sharedV, jobs, max_workers)
        cluster_metrics = []
        for cid, true_vals, pred_vals, _eval_rows in parts:
            if len(true_vals) == 0:
                continue
            errors = true_vals - pred_vals
            c_mae  = float(np.mean(np.abs(errors)))
            c_rmse = float(np.sqrt(np.mean(errors ** 2)))
            cluster_metrics.append({
                'cluster_id': cid,
                'mae': c_mae,
                'rmse': c_rmse,
                'n_test': len(true_vals),
            })

        true_chunks = [p[1] for p in parts if len(p[1]) > 0]
        pred_chunks = [p[2] for p in parts if len(p[2]) > 0]
        eval_chunks = [p[3] for p in parts if len(p[3]) > 0]
        n_clusters_fit = len(true_chunks)
        if true_chunks:
            all_true = np.concatenate(true_chunks).tolist()
            all_pred = np.concatenate(pred_chunks).tolist()
        else:
            all_true, all_pred = [], []
        if eval_chunks:
            eval_rows = np.concatenate(eval_chunks, axis=0)
        else:
            eval_rows = np.zeros((0, 4), dtype=np.float32)
    else:
        profiles = {}
        for cid, c_train in cluster_train.items():
            if len(c_train) < 5:
                continue
            c_train_r, _, uid_map, n_loc = remap_user_ids(
                c_train,
                np.empty((0, 3), dtype=np.float32),
                n_items,
            )
            cluster_model = shared.make_cluster_model(
                n_users_cluster=n_loc,
                n_epochs_cluster=N_EPOCHS_CLUSTER,
                random_seed=RANDOM_SEED + int(cid),
                cluster_ratings=c_train_r if use_cluster_bias else None,
            )
            cluster_model.fit_cluster_U(c_train_r)
            profiles[int(cid)] = {
                'model': cluster_model,
                'uid_map': uid_map,
                'mean_u': cluster_model.U.mean(axis=0).astype(np.float32),
            }

        n_clusters_fit = len(profiles)
        eval_rows_list = []
        for row in test:
            u, i, r = int(row[0]), int(row[1]), float(row[2])
            if gray_mask[u]:
                continue
            pred = _membership_weighted_prediction(
                memberships[u], profiles, _predict_sharedv_profile, u, i,
            )
            if np.isnan(pred):
                hard_profile = profiles.get(int(assignments[u]))
                if hard_profile is None:
                    continue
                pred = _predict_sharedv_profile(hard_profile, u, i)
            eval_rows_list.append((u, i, r, pred))

        eval_rows = (
            np.array(eval_rows_list, dtype=np.float32)
            if eval_rows_list else np.zeros((0, 4), dtype=np.float32)
        )
        all_true = eval_rows[:, 2].tolist() if len(eval_rows) > 0 else []
        all_pred = eval_rows[:, 3].tolist() if len(eval_rows) > 0 else []

        cluster_metrics = []
        if len(eval_rows) > 0:
            for cid in sorted(profiles.keys()):
                user_ids_eval = eval_rows[:, 0].astype(np.int32)
                mask = assignments[user_ids_eval] == int(cid)
                if np.sum(mask) == 0:
                    continue
                errs = eval_rows[mask, 2] - eval_rows[mask, 3]
                cluster_metrics.append({
                    'cluster_id': int(cid),
                    'mae': float(np.mean(np.abs(errs))),
                    'rmse': float(np.sqrt(np.mean(errs ** 2))),
                    'n_test': int(np.sum(mask)),
                })

    mae, rmse = _compute_metrics(all_true, all_pred)
    precision, recall, f1, ndcg = _compute_topn_metrics(
        eval_rows, top_n=top_n, threshold=relevance_threshold,
    )

    if cluster_metrics:
        maes = [c['mae'] for c in cluster_metrics]
        cluster_mae_std  = float(np.std(maes))
        cluster_mae_mean = float(np.mean(maes))
        cluster_mae_min  = float(np.min(maes))
        cluster_mae_max  = float(np.max(maes))
    else:
        cluster_mae_std = cluster_mae_mean = cluster_mae_min = cluster_mae_max = float('nan')

    if out_dir is not None and cluster_metrics:
        cm_path = os.path.join(out_dir, f'{algo_label}_cluster_mae.csv')
        save_dataframe_csv(
            pd.DataFrame(cluster_metrics),
            cm_path,
            run_command=run_command,
        )

    # Gray sheep — shared V kullanarak
    gray_mae, gray_rmse = _run_gray_sheep_sharedV(
        shared, gray_train, gray_test, n_items
    )

    white_mae, white_rmse = _compute_metrics(all_true, all_pred)

    elapsed = time.time() - t0
    print(f"  [{algo_label} | SharedV] MAE={mae:.4f} RMSE={rmse:.4f} | "
          f"Gray MAE={gray_mae:.4f}  ({elapsed:.1f}s)")

    return {
        'scenario'    : 'cluster_sharedV',
        'algo_label'  : algo_label,
        'mae'         : mae,
        'rmse'        : rmse,
        'gray_mae'    : gray_mae,
        'gray_rmse'   : gray_rmse,
        'white_mae'   : white_mae,
        'white_rmse'  : white_rmse,
        'n_clusters'  : n_clusters_fit,
        'n_train'     : len(train),
        'n_test'      : len(test),
        'time_seconds': elapsed,
        'cluster_mae_std' : cluster_mae_std,
        'cluster_mae_mean': cluster_mae_mean,
        'cluster_mae_min' : cluster_mae_min,
        'cluster_mae_max' : cluster_mae_max,
        'accuracy'        : float('nan'),
        'precision_at_10' : precision,
        'recall_at_10'    : recall,
        'f1_at_10'        : f1,
        'ndcg_at_10'      : ndcg,
    }


# ============================================================
# GRAY SHEEP YARDIMCILAR
# ============================================================

def _run_gray_sheep(gray_train, gray_test, n_items, mode='full',
                    use_svdpp: bool = False):
    """Gray sheep için bağımsız WNMFModel."""
    if len(gray_train) < 5 or len(gray_test) == 0:
        return float('nan'), float('nan')

    g_train_r, g_test_r, _, n_gs = remap_user_ids(gray_train, gray_test, n_items)

    model = WNMFModel(
        n_users        = n_gs,
        n_items        = n_items,
        latent_dim     = LATENT_DIM,
        learning_rate  = LEARNING_RATE,
        regularization = REGULARIZATION,
        n_epochs       = N_EPOCHS_CLUSTER,
        random_seed    = RANDOM_SEED,
        use_svdpp      = use_svdpp,
    )
    model.fit(g_train_r)

    if len(g_test_r) == 0:
        return float('nan'), float('nan')
    return model.evaluate(g_test_r)


def _run_gray_sheep_sharedV(shared, gray_train, gray_test, n_items):
    """Gray sheep için SharedV'den türetilmiş model."""
    if len(gray_train) < 5 or len(gray_test) == 0:
        return float('nan'), float('nan')

    g_train_r, g_test_r, _, n_gs = remap_user_ids(gray_train, gray_test, n_items)

    gs_model = shared.make_cluster_model(
        n_users_cluster  = n_gs,
        n_epochs_cluster = N_EPOCHS_CLUSTER,
        random_seed      = RANDOM_SEED,
    )
    gs_model.fit_cluster_U(g_train_r)

    if len(g_test_r) == 0:
        return float('nan'), float('nan')
    return gs_model.evaluate(g_test_r)


# ============================================================
# ALGO-DÜZEYİNDE PARALEL WORKER
# ============================================================

def _mp_run_algo_job(job):
    """
    Pickle edilebilir modül-düzeyinde worker — bir algoritma etiketinin
    tüm senaryolarını çalıştırır (--algo-jobs ile açılan ProcessPoolExecutor için).

    Çocuk süreçler modülü sıfırdan yükler ve modül globallerini varsayılan
    değerlerle başlatır; hiperparametre değerleri job tuple'ından aktarılır.
    """
    (
        label,
        assign_dir,
        train,
        test,
        n_items,
        dataset_name,
        mode,
        eff_cluster_workers,
        weighted_v,
        use_bias,
        use_cluster_bias,
        use_svdpp,
        run_cluster_avg_flag,
        do_cluster_knn_flag,
        top_n,
        relevance_threshold,
        similarity,
        knn,
        min_common,
        out_dir,
        run_command,
        ld, lr, reg, eg, ec,
    ) = job

    # Çocuk süreçte hiperparametre globallerini geçerli değerlere ayarla
    global LATENT_DIM, LEARNING_RATE, REGULARIZATION, N_EPOCHS_GLOBAL, N_EPOCHS_CLUSTER
    LATENT_DIM       = ld
    LEARNING_RATE    = lr
    REGULARIZATION   = reg
    N_EPOCHS_GLOBAL  = eg
    N_EPOCHS_CLUSTER = ec

    try:
        assignments, gray_mask = load_assignment(assign_dir)
        memberships = load_memberships(assign_dir)
        rows: List[dict] = []

        if run_cluster_avg_flag:
            row = run_cluster_average(
                train, test, assignments, gray_mask, memberships, n_items, label,
                top_n=top_n,
                relevance_threshold=relevance_threshold,
            )
            row['dataset'] = dataset_name
            rows.append(row)

        if do_cluster_knn_flag:
            row = run_cluster_knn(
                train, test, assignments, gray_mask, memberships, n_items, label,
                similarity=similarity,
                min_common=min_common,
                k_neighbors=knn,
                top_n=top_n,
                relevance_threshold=relevance_threshold,
            )
            row['dataset'] = dataset_name
            rows.append(row)

        if mode in ('full', 'all'):
            row = run_cluster_full(
                train, test, assignments, gray_mask, memberships, n_items, label,
                max_workers=eff_cluster_workers,
                use_svdpp=use_svdpp,
                top_n=top_n,
                relevance_threshold=relevance_threshold,
            )
            row['dataset'] = dataset_name
            rows.append(row)

        if mode in ('sharedV', 'all'):
            row = run_cluster_sharedV(
                train, test, assignments, gray_mask, memberships, n_items, label,
                max_workers=eff_cluster_workers,
                out_dir=out_dir,
                weighted_v=weighted_v,
                use_bias=use_bias,
                use_cluster_bias=use_cluster_bias,
                use_svdpp=use_svdpp,
                run_command=run_command,
                top_n=top_n,
                relevance_threshold=relevance_threshold,
            )
            row['dataset'] = dataset_name
            rows.append(row)

        return label, rows

    except Exception as exc:
        import traceback
        print(f"\n  [{label}] HATA — paralel worker başarısız:\n"
              f"    {type(exc).__name__}: {exc}\n"
              + "".join(f"    {l}" for l in traceback.format_exc().splitlines(keepends=True)),
              flush=True)
        return label, []



def _save_row_to_db(dataset_name: str, row: dict, k_used: int, args,
                    fold_override: Optional[int] = None):
    if not _DB_AVAILABLE:
        return
    db_dataset = _normalize_db_dataset_name(dataset_name)
    prep_raw = getattr(args, 'assign_suffix', '')
    prep = _normalize_db_preprocessing(prep_raw)
    scenario = row['scenario']
    assignment_id_override = None
    if scenario != 'global':
        for prep_try in _db_preprocessing_candidates(prep_raw):
            try:
                assignment_id_override = get_assignment_id(
                    db_dataset, row['algo_label'], k_used, prep_try
                )
            except Exception:
                assignment_id_override = None
            if assignment_id_override is not None:
                prep = prep_try
                break

    save_wnmf_result(
        dataset=db_dataset,
        algo=row['algo_label'],
        k=k_used,
        preprocessing=prep,
        scenario=scenario,
        mae=row['mae'],
        rmse=row['rmse'],
        gray_mae=row.get('gray_mae'),
        gray_rmse=row.get('gray_rmse'),
        white_mae=row.get('white_mae'),
        white_rmse=row.get('white_rmse'),
        precision_at_10=row.get('precision_at_10'),
        recall_at_10=row.get('recall_at_10'),
        f1_at_10=row.get('f1_at_10'),
        ndcg_at_10=row.get('ndcg_at_10'),
        n_train=row['n_train'],
        n_test=row['n_test'],
        latent_dim=args.latent_dim,
        epochs_global=args.epochs_global,
        epochs_cluster=args.epochs_cluster,
        reg=args.reg,
        lr=args.lr,
        time_seconds=row['time_seconds'],
        cv_fold=row.get('cv_fold', fold_override),
        cv_n_splits=row.get('cv_n_splits'),
        is_cv_mean=bool(row.get('is_cv_mean', False)),
        mean_mae=row.get('mean_mae'),
        mean_rmse=row.get('mean_rmse'),
        fold_mae_values=row.get('fold_mae_values'),
        fold_rmse_values=row.get('fold_rmse_values'),
        run_id=_CURRENT_RUN_ID,
        assignment_id_override=assignment_id_override,
    )


def _append_result_row(results: List[dict], row: dict, dataset_name: str, k_used: int, args,
                       fold: Optional[int]) -> None:
    row['dataset'] = dataset_name
    results.append(row)
    _save_row_to_db(dataset_name, row, k_used, args, fold_override=fold)


def _collect_active_algos(
    dataset_name: str,
    k_used: int,
    algo_filter: Optional[List[str]],
    assign_root: str,
    assign_suffix: str,
) -> List[Tuple[str, str]]:
    active_algos: List[Tuple[str, str]] = []
    for label in ALGO_LABELS:
        if algo_filter and label not in algo_filter:
            print(f"\n  [{label}] atlandı (filtre)")
            continue
        assign_dir = _algo_assignment_dir(
            assign_root, dataset_name, label, k_used, assign_suffix=assign_suffix,
        )
        if not os.path.exists(assign_dir):
            print(f"\n  [{label}] ATLANDI — assignment bulunamadı: {assign_dir}")
            continue
        active_algos.append((label, assign_dir))
    return active_algos


def _run_algo_rows_single_process(
    label: str,
    assign_dir: str,
    train: np.ndarray,
    test: np.ndarray,
    n_items: int,
    mode: str,
    cluster_workers: Optional[int],
    weighted_v: bool,
    use_bias: bool,
    use_cluster_bias: bool,
    use_svdpp: bool,
    run_cluster_avg: bool,
    do_cluster_knn: bool,
    top_n: int,
    relevance_threshold: float,
    similarity: str,
    knn: int,
    min_common: int,
    out_dir: str,
    run_command: Optional[str],
) -> List[dict]:
    assignments, gray_mask = load_assignment(assign_dir)
    memberships = load_memberships(assign_dir)
    rows: List[dict] = []

    if run_cluster_avg:
        rows.append(run_cluster_average(
            train, test, assignments, gray_mask, memberships, n_items, label,
            top_n=top_n, relevance_threshold=relevance_threshold,
        ))
    if do_cluster_knn:
        rows.append(run_cluster_knn(
            train, test, assignments, gray_mask, memberships, n_items, label,
            similarity=similarity, min_common=min_common, k_neighbors=knn,
            top_n=top_n, relevance_threshold=relevance_threshold,
        ))
    if mode in ('full', 'all'):
        rows.append(run_cluster_full(
            train, test, assignments, gray_mask, memberships, n_items, label,
            max_workers=cluster_workers, use_svdpp=use_svdpp,
            top_n=top_n, relevance_threshold=relevance_threshold,
        ))
    if mode in ('sharedV', 'all'):
        rows.append(run_cluster_sharedV(
            train, test, assignments, gray_mask, memberships, n_items, label,
            max_workers=cluster_workers, out_dir=out_dir,
            weighted_v=weighted_v, use_bias=use_bias, use_cluster_bias=use_cluster_bias,
            use_svdpp=use_svdpp, run_command=run_command,
            top_n=top_n, relevance_threshold=relevance_threshold,
        ))
    return rows


def _run_algorithms_for_dataset(
    results: List[dict],
    active_algos: List[Tuple[str, str]],
    train: np.ndarray,
    test: np.ndarray,
    n_items: int,
    dataset_name: str,
    mode: str,
    cluster_workers: Optional[int],
    algo_workers: Optional[int],
    weighted_v: bool,
    use_bias: bool,
    use_cluster_bias: bool,
    use_svdpp: bool,
    run_cluster_avg: bool,
    do_cluster_knn: bool,
    top_n: int,
    relevance_threshold: float,
    similarity: str,
    knn: int,
    min_common: int,
    out_dir: str,
    run_command: Optional[str],
    k_used: int,
    args,
    fold: Optional[int],
) -> None:
    nw_algo = _resolve_pool_workers(algo_workers, len(active_algos))
    eff_cluster_workers = 1 if nw_algo > 1 else cluster_workers

    if nw_algo > 1 and len(active_algos) > 1:
        print(f"\n  Algoritmalar paralel çalıştırılıyor: "
              f"{len(active_algos)} iş, en fazla {nw_algo} süreç "
              f"(küme içi paralellik: kapalı)")
        jobs = [
            (
                label, assign_dir,
                train, test, n_items, dataset_name, mode,
                eff_cluster_workers, weighted_v, use_bias, use_cluster_bias, use_svdpp,
                run_cluster_avg, do_cluster_knn, top_n, relevance_threshold, similarity, knn,
                min_common,
                out_dir, run_command,
                LATENT_DIM, LEARNING_RATE, REGULARIZATION, N_EPOCHS_GLOBAL, N_EPOCHS_CLUSTER,
            )
            for label, assign_dir in active_algos
        ]
        with ProcessPoolExecutor(max_workers=nw_algo) as pool:
            algo_results = list(pool.map(_mp_run_algo_job, jobs))
        label_to_rows = {lbl: rows for lbl, rows in algo_results}
        for label, _ in active_algos:
            for row in label_to_rows.get(label, []):
                _append_result_row(results, row, dataset_name, k_used, args, fold)
        return

    for label, assign_dir in active_algos:
        rows = _run_algo_rows_single_process(
            label, assign_dir, train, test, n_items, mode, cluster_workers,
            weighted_v, use_bias, use_cluster_bias, use_svdpp,
            run_cluster_avg, do_cluster_knn, top_n, relevance_threshold,
            similarity, knn, min_common, out_dir, run_command,
        )
        for row in rows:
            _append_result_row(results, row, dataset_name, k_used, args, fold)


def run_dataset(dataset_name, train, test, algo_filter=None,
                run_global=True, mode='sharedV',
                cluster_workers: Optional[int] = None,
                algo_workers: Optional[int] = None,
                assign_root: Optional[str] = None,
                assign_suffix: str = '',
                assignment_k: Optional[int] = None,
                assignment_k_100k: Optional[int] = None,
                assignment_k_1m: Optional[int] = None,
                weighted_v: bool = False,
                use_bias: bool = True,
                use_cluster_bias: bool = False,
                run_cluster_avg: bool = True,
                do_cluster_knn: bool = True,
                top_n: int = 10,
                relevance_threshold: float = 4.0,
                similarity: str = 'pearson',
                knn: int = 30,
                min_common: int = 3,
                use_svdpp: bool = False,
                run_command: Optional[str] = None,
                args=None,
                fold: Optional[int] = None):
    """
    Bir dataset üzerinde tüm senaryoları çalıştır.

    mode parametresi:
        'baselines' — küme ortalaması + küme içi kNN CF (+ isteğe bağlı global WNMF);
                      WNMF küme modeli (full/sharedV) çalışmaz. Tüm --assign-suffix ile uyumlu.
                      --no-cluster-avg / --no-cluster-knn ile parça parça kapatılabilir.
        'sharedV'   — baselines (varsayılan açık) + SharedV WNMF (önerilen)
        'full'      — baselines + her küme ayrı U+V WNMF
        'all'       — baselines + full + sharedV

    Assignment klasörü: mealpy/results/assignments_lof/{dataset}/{label} veya
    K ≠ varsayılan ise {label}_k{K} (generate_assignments ile aynı kural).
    """
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name.upper()}  |  mode={mode}")
    print(f"{'='*60}")

    root = assign_root if assign_root is not None else ASSIGN_ROOT
    dk   = ASSIGN_K_DEFAULT_100K if dataset_name == 'ml100k' else ASSIGN_K_DEFAULT_1M
    k_ds = assignment_k_100k if dataset_name == 'ml100k' else assignment_k_1m
    k_used = _resolved_assignment_k(k_ds, assignment_k, dk)
    print(f"Assignment K  : {k_used} (klasör eki: {'yok' if k_used == dk else f'_k{k_used}'})")
    print(f"Assignment kök: {root}")

    # CV/holdout'ta testte train'de görülmeyen item olabilir; item boyutunu
    # train+test birleşik maksimum ID'den al ki predict sırasında taşma olmasın.
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    results = []
    if fold is None:
        k_dir = os.path.join(OUT_ROOT, dataset_name, f'k{k_used}')
    else:
        k_dir = os.path.join(OUT_ROOT, dataset_name, f'k{k_used}', f'fold{fold}')
    run_n  = _next_run_index(k_dir)
    out_dir = os.path.join(k_dir, f'run{run_n}')
    os.makedirs(out_dir, exist_ok=True)
    print(f"Sonuç klasörü  : {out_dir}")

    # Global WNMF baseline
    if run_global:
        row = run_global_wnmf(
            train, test, n_items, use_bias=use_bias, use_svdpp=use_svdpp,
            top_n=top_n, relevance_threshold=relevance_threshold,
        )
        _append_result_row(results, row, dataset_name, k_used, args, fold)

    active_algos = _collect_active_algos(
        dataset_name=dataset_name,
        k_used=k_used,
        algo_filter=algo_filter,
        assign_root=root,
        assign_suffix=assign_suffix,
    )
    _run_algorithms_for_dataset(
        results=results,
        active_algos=active_algos,
        train=train,
        test=test,
        n_items=n_items,
        dataset_name=dataset_name,
        mode=mode,
        cluster_workers=cluster_workers,
        algo_workers=algo_workers,
        weighted_v=weighted_v,
        use_bias=use_bias,
        use_cluster_bias=use_cluster_bias,
        use_svdpp=use_svdpp,
        run_cluster_avg=run_cluster_avg,
        do_cluster_knn=do_cluster_knn,
        top_n=top_n,
        relevance_threshold=relevance_threshold,
        similarity=similarity,
        knn=knn,
        min_common=min_common,
        out_dir=out_dir,
        run_command=run_command,
        k_used=k_used,
        args=args,
        fold=fold,
    )

    meta = _result_row_meta(k_used)
    sub = os.path.basename(out_dir)
    results = [
        {**meta, **r, 'result_subdir': sub, 'use_svdpp': bool(use_svdpp)}
        for r in results
    ]

    # Kaydet ve özet yazdır (her koşu: results/wnmf/{dataset}/k{K}/run{N}/...)
    save_results(
        results,
        out_dir,
        f'wnmf_results_{dataset_name}_k{k_used}_{mode}.csv',
        run_command=run_command,
    )
    # mode=all: tek CSV'de her iki senaryo vardır; ayrıca full / sharedV alt kümeleri (GLOBAL her ikisinde)
    if mode == 'all':
        rows_full = [r for r in results if r['scenario'] in ('global', 'cluster_full')]
        rows_sv   = [r for r in results if r['scenario'] in ('global', 'cluster_sharedV')]
        save_results(
            rows_full,
            out_dir,
            f'wnmf_results_{dataset_name}_k{k_used}_all_split_full.csv',
            run_command=run_command,
        )
        save_results(
            rows_sv,
            out_dir,
            f'wnmf_results_{dataset_name}_k{k_used}_all_split_sharedV.csv',
            run_command=run_command,
        )

    _print_summary(results, dataset_name)

    return results


# ============================================================
# ÖZET TABLOSU
# ============================================================

def _print_summary(results, dataset_name):
    print(f"\n{'='*60}")
    print(f"ÖZET — {dataset_name.upper()}")
    print(f"{'='*60}")
    tag_hdr = 'hyperparam_tag' if results and 'hyperparam_tag' in results[0] else None
    if tag_hdr:
        print(
            f"{'tag':<36} {'Algoritma':<16} {'Senaryo':<14} {'MAE':>8} {'RMSE':>8} "
            f"{'ClStd':>8} {'GS MAE':>8} {'Wh MAE':>8} "
            f"{'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'NDCG':>8}"
        )
        w = 36 + 16 + 14 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 6
    else:
        print(
            f"{'Algoritma':<20} {'Senaryo':<16} {'MAE':>8} {'RMSE':>8} "
            f"{'ClStd':>8} {'GS MAE':>8} {'Wh MAE':>8} "
            f"{'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'NDCG':>8}"
        )
        w = 72 + 8 + 8 + 8 + 8 + 8
    print("-" * max(w, 72))

    for r in results:
        gs = r.get('gray_mae', float('nan'))
        gs_str = f"{gs:.4f}" if not (isinstance(gs, float) and np.isnan(gs)) else "  —  "
        wh = r.get('white_mae', float('nan'))
        wh_str = f"{wh:.4f}" if not (isinstance(wh, float) and np.isnan(wh)) else "  —  "
        acc = r.get('accuracy', float('nan'))
        acc_str = f"{acc:.4f}" if not (isinstance(acc, float) and np.isnan(acc)) else "  —  "
        pr = r.get('precision_at_10', float('nan'))
        pr_str = f"{pr:.4f}" if not (isinstance(pr, float) and np.isnan(pr)) else "  —  "
        rc = r.get('recall_at_10', float('nan'))
        rc_str = f"{rc:.4f}" if not (isinstance(rc, float) and np.isnan(rc)) else "  —  "
        f1v = r.get('f1_at_10', float('nan'))
        f1_str = f"{f1v:.4f}" if not (isinstance(f1v, float) and np.isnan(f1v)) else "  —  "
        nd = r.get('ndcg_at_10', float('nan'))
        nd_str = f"{nd:.4f}" if not (isinstance(nd, float) and np.isnan(nd)) else "  —  "
        cms = r.get('cluster_mae_std', float('nan'))
        cms_str = (
            f"{cms:.4f}"
            if not (isinstance(cms, float) and np.isnan(cms))
            else "  —  "
        )
        if tag_hdr:
            tg = str(r.get('hyperparam_tag', ''))[:34]
            print(
                f"{tg:<36} "
                f"{r['algo_label']:<16} "
                f"{r['scenario']:<14} "
                f"{r['mae']:>8.4f} "
                f"{r['rmse']:>8.4f} "
                f"{cms_str:>8} "
                f"{gs_str:>8} "
                f"{wh_str:>8} "
                f"{acc_str:>8} "
                f"{pr_str:>8} "
                f"{rc_str:>8} "
                f"{f1_str:>8} "
                f"{nd_str:>8}"
            )
        else:
            print(
                f"{r['algo_label']:<20} "
                f"{r['scenario']:<16} "
                f"{r['mae']:>8.4f} "
                f"{r['rmse']:>8.4f} "
                f"{cms_str:>8} "
                f"{gs_str:>8} "
                f"{wh_str:>8} "
                f"{acc_str:>8} "
                f"{pr_str:>8} "
                f"{rc_str:>8} "
                f"{f1_str:>8} "
                f"{nd_str:>8}"
            )
    print("=" * 72)


def _build_kfold_splits(data: np.ndarray, n_splits: int = 5,
                        shuffle: bool = True, random_state: int = 42):
    """Veriyi KFold ile train/test parçalarına böler."""
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for train_index, test_index in kf.split(data):
        yield data[train_index], data[test_index]


def _aggregate_fold_results(rows: List[dict], n_splits: int) -> List[dict]:
    """
    Fold bazlı satırları (aynı algo/scenario) gruplayıp MAE/RMSE ortalamasını alır.
    CSV/log tarafında yalnızca mean metrikleri kullanmak için tasarlanmıştır.
    """
    grouped = {}
    for row in rows:
        key = (
            row.get('dataset'),
            row.get('algo_label'),
            row.get('scenario'),
            row.get('assignment_k'),
            row.get('hyperparam_tag'),
            row.get('use_svdpp'),
        )
        grouped.setdefault(key, []).append(row)

    aggregated: List[dict] = []
    for _, group_rows in grouped.items():
        base = dict(group_rows[0])
        maes = [float(r.get('mae', np.nan)) for r in group_rows]
        rmses = [float(r.get('rmse', np.nan)) for r in group_rows]
        base['fold_count'] = len(group_rows)
        base['cv_n_splits'] = int(n_splits)
        base['is_cv_mean'] = True
        base['fold_mae_values'] = ';'.join(f"{v:.6f}" for v in maes)
        base['fold_rmse_values'] = ';'.join(f"{v:.6f}" for v in rmses)
        base['mean_mae'] = float(np.mean(maes))
        base['mean_rmse'] = float(np.mean(rmses))
        for metric_key in ('precision_at_10', 'recall_at_10', 'f1_at_10', 'ndcg_at_10'):
            vals = [float(r.get(metric_key, np.nan)) for r in group_rows]
            if np.all(np.isnan(vals)):
                base[metric_key] = float('nan')
            else:
                base[metric_key] = float(np.nanmean(vals))
        # Geriye dönük uyum: mevcut tablo/çıktı mae-rmse kolonlarını da ortalama ile doldur.
        base['mae'] = base['mean_mae']
        base['rmse'] = base['mean_rmse']
        aggregated.append(base)

    aggregated.sort(key=lambda r: (str(r.get('dataset')), str(r.get('algo_label')), str(r.get('scenario'))))
    return aggregated


def _save_cv_mean_results(dataset_name: str, k_used: int, mode: str,
                          rows: List[dict], run_command: Optional[str] = None):
    """5-fold ortalama sonuçları tek CSV olarak kaydeder."""
    cv_root = os.path.join(OUT_ROOT, dataset_name, f'k{k_used}', 'cv5_mean')
    run_n = _next_run_index(cv_root)
    out_dir = os.path.join(cv_root, f'run{run_n}')
    fname = f'wnmf_results_{dataset_name}_k{k_used}_{mode}_cv5_mean.csv'
    save_results(rows, out_dir, fname, run_command=run_command)
    _print_summary(rows, f"{dataset_name} (CV5 mean)")


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="WNMF karşılaştırma deneyi")
    p.add_argument('--dataset', choices=['100k', '1m', 'both'], default='both')
    p.add_argument(
        '--fold', type=int, default=None, metavar='N',
        help='İsteğe bağlı CV fold (1-5). Verilmezse: ML-100K u1.base/u1.test, ML-1M %%20 holdout, '
             'sonuçlar .../k{K}/runN. Verilirse: ML-100K u{N}.base/test, ML-1M fold 1=holdout 2-5=KFold, '
             'sonuçlar .../k{K}/fold{N}/runN.',
    )
    p.add_argument(
        '--algo',
        nargs='+',
        metavar='LABEL',
        default=None,
        help='İzin verilen etiketler: ' + ', '.join(ALGO_LABELS),
    )
    p.add_argument('--no-global', action='store_true')
    p.add_argument(
        '--mode',
        choices=['baselines', 'sharedV', 'full', 'all'],
        default='sharedV',
        help="baselines=küme ort.+kNN CF (+global), WNMF küme yok; sharedV=önerilen; full; all=hepsi",
    )
    p.add_argument(
        '--epochs-global', nargs='+', type=int, default=None,
        metavar='N',
        help='Global epoch; birden fazla: --epochs-global 50 100 (cluster ile eşleşir veya tek cluster yayılır)',
    )
    p.add_argument(
        '--epochs-cluster', nargs='+', type=int, default=None,
        metavar='N',
        help='Küme epoch; birden fazla: --epochs-cluster 25 50',
    )
    p.add_argument(
        '--epochs-grid',
        action='store_true',
        help='epochs_global × epochs_cluster tüm çiftleri (Cartesian). '
             'Kapalıyken eski kural: eşit uzunlukta zip, tek eleman yayılımı.',
    )
    p.add_argument(
        '--latent-dim', nargs='+', type=int, default=None,
        metavar='D',
        help='Gizli boyut; birden fazla: --latent-dim 10 20 50 (tüm kombinasyonlar)',
    )
    p.add_argument(
        '--wnmf-features', type=int, default=None, metavar='D',
        help='Geriye dönük uyumluluk: --latent-dim için tek değerli kısayol.',
    )
    p.add_argument(
        '--reg', nargs='+', type=float, default=None,
        dest='regularization',
        metavar='R',
        help='L2 düzenleme; birden fazla: --reg 0.001 0.01 0.1 (verilmezse 0.01)',
    )
    p.add_argument(
        '--lr', nargs='+', type=float, default=None,
        dest='learning_rate',
        metavar='LR',
        help='Öğrenme oranı; birden fazla: --lr 0.001 0.01 (varsayılan: 0.01)',
    )
    p.add_argument(
        '-j', '--jobs', type=int, default=None,
        dest='jobs',
        help='Küme eğitiminde paralel süreç sayısı; verilmezse CPU çekirdek sayısı',
    )
    p.add_argument(
        '--algo-jobs', type=int, default=None,
        dest='algo_jobs',
        metavar='N',
        help='Algoritma düzeyinde paralel süreç sayısı; verilmezse sıralı (1). '
             '0=CPU sayısına göre otomatik. Her algoritmayı ayrı süreçte çalıştırır; '
             '--algo-jobs etkinken --jobs (küme içi) otomatik 1 olur.',
    )
    p.add_argument(
        '--k', nargs='+', type=int, default=None,
        dest='assignment_k',
        metavar='K',
        help='Assignment K; birden fazla: --k 20 30 50 (her K için ayrı results/.../k{K}/run). '
             'Tek değer: --k 70. Verilmezse --k-100k / --k-1m veya varsayılanlar. '
             '--k-100k / --k-1m ile birlikte kullanılamaz.',
    )
    p.add_argument(
        '--k-100k', type=int, default=None,
        dest='assignment_k_100k',
        help='Sadece ML-100K için assignment K (--k üzerine yazar)',
    )
    p.add_argument(
        '--k-1m', type=int, default=None,
        dest='assignment_k_1m',
        help='Sadece ML-1M için assignment K (--k üzerine yazar)',
    )
    p.add_argument(
        '--assign-root', type=str, default=None,
        help='Assignment kök dizini (varsayılan: mealpy/results/assignments_lof)',
    )
    p.add_argument(
        '--assign-suffix', type=str, default='',
        help='Assignment klasör adına ek suffix (örn: _wnmf20, _zscore)',
    )
    p.add_argument(
        '--sync-assign-suffix-latent',
        action='store_true',
        help='--assign-suffix içindeki ilk _wnmf<RAKAM> parçasını her --latent-dim değeriyle '
             'otomatik değiştirir (assignment klasörü ile latent boyutu eşleşsin).',
    )
    p.add_argument(
        '--skip-existing',
        action='store_true',
        help='Daha önce yazılmış sonuç CSV varsa (aynı hyperparam_tag ve use_svdpp) bu K için '
             'yeniden çalıştırma. 5-fold ortalama: cv5_mean/run*/..._cv5_mean.csv; '
             'holdout/fold: k.../run*/wnmf_results_*.csv taranır.',
    )
    p.add_argument(
        '--weighted-v', action='store_true',
        help='SharedV global V eğitiminde gray sheep rating ağırlığı 0.1',
    )
    p.add_argument(
        '--no-bias', action='store_true',
        help='Global WNMF ve SharedV yolunda bias terimleri kapalı',
    )
    p.add_argument(
        '--cluster-bias', action='store_true',
        help='Küme bazlı mu_k kullan (global mu yerine)',
    )
    p.add_argument(
        '--no-cluster-avg', action='store_true',
        help='ClusterAvg senaryosunu çalıştırma',
    )
    p.add_argument(
        '--no-cluster-knn', action='store_true',
        help='ClusterKNN (küme içi CF) senaryosunu çalıştırma',
    )
    p.add_argument(
        '--top-n', type=int, default=10,
        help='Precision/recall Top-N (cluster_avg)',
    )
    p.add_argument(
        '--relevance-threshold', type=float, default=4.0,
        help='Test rating >= bu değer relevant sayılır (cluster_avg P/R)',
    )
    p.add_argument(
        '--similarity',
        choices=['pearson', 'cosine'],
        default='pearson',
        help='kNN benzerlik metriği: pearson (default) veya cosine'
    )
    p.add_argument(
        '--knn', type=int, default=30, metavar='K',
        help='ClusterKNN: küme içi en fazla K komşu (varsayılan: 30)',
    )
    p.add_argument(
        '--min-common', type=int, default=3, metavar='N',
        help='ClusterKNN: benzerlik için minimum ortak film sayısı (varsayılan: 3)',
    )
    p.add_argument(
        '--svdpp', action='store_true',
        help='WNMFModel / global WNMFSharedV için SVD++ (implicit Y) kullan',
    )
    p.add_argument(
        '--compare-mf-svdpp',
        action='store_true',
        help='Aynı hiperparametrelerle ardışık iki koşu: önce MF (SVD++ kapalı), sonra SVD++. '
             'CSV’de use_svdpp sütunu ile ayrışır; --svdpp ile birlikte verilirse bu bayrak önceliklidir.',
    )
    p.add_argument(
        '--note', type=str, default=None,
        help='Bu run için açıklama notu (örn: "zscore karşılaştırma")',
    )
    args = p.parse_args()
    if args.knn < 1:
        p.error('--knn en az 1 olmalı')
    if args.min_common < 1:
        p.error('--min-common en az 1 olmalı')
    if args.assignment_k is not None and (
        args.assignment_k_100k is not None or args.assignment_k_1m is not None
    ):
        p.error('--k (tek veya çoklu) ile --k-100k / --k-1m birlikte kullanılamaz')
    if args.fold is not None and args.fold not in (1, 2, 3, 4, 5):
        p.error('--fold 1..5 olmalı veya tamamen verilmemeli')
    if args.algo is not None:
        resolved = []
        for raw in args.algo:
            canon = ALGO_ALIASES.get(raw, raw)
            if canon not in ALGO_LABELS:
                p.error(
                    f"bilinmeyen --algo '{raw}'; izin verilenler: {', '.join(ALGO_LABELS)}"
                )
            if raw != canon:
                print(f"not: --algo '{raw}' -> '{canon}' olarak yorumlandi", file=sys.stderr)
            resolved.append(canon)
        args.algo = resolved
    if args.wnmf_features is not None:
        if args.wnmf_features <= 0:
            p.error('--wnmf-features pozitif olmalı')
        if args.latent_dim is None:
            args.latent_dim = [int(args.wnmf_features)]
        elif len(args.latent_dim) == 1 and int(args.latent_dim[0]) == int(args.wnmf_features):
            pass
        else:
            p.error('--wnmf-features ile --latent-dim çakışıyor; tek birini kullanın')
    return args


# ============================================================
# MAIN
# ============================================================

def _iter_assignment_k(args, dataset_name: str) -> List[int]:
    if args.assignment_k is not None:
        return list(args.assignment_k)
    default_k = ASSIGN_K_DEFAULT_100K if dataset_name == 'ml100k' else ASSIGN_K_DEFAULT_1M
    k_specific = args.assignment_k_100k if dataset_name == 'ml100k' else args.assignment_k_1m
    return [_resolved_assignment_k(k_specific, None, default_k)]


def _collect_dataset_tags(args) -> List[str]:
    tags: List[str] = []
    if args.dataset in ('100k', 'both'):
        ks = _iter_assignment_k(args, 'ml100k')
        tags.append(f"ml100k_k{'-'.join(map(str, ks)) if len(ks) > 1 else ks[0]}")
    if args.dataset in ('1m', 'both'):
        ks = _iter_assignment_k(args, 'ml1m')
        tags.append(f"ml1m_k{'-'.join(map(str, ks)) if len(ks) > 1 else ks[0]}")
    return tags


def _load_datasets(args):
    data: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    if args.dataset in ('100k', 'both'):
        if args.fold is None:
            train, test = load_ratings_100k(DATA_100K_TRAIN, DATA_100K_TEST)
        else:
            train_path = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-100k', f'u{args.fold}.base')
            test_path = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-100k', f'u{args.fold}.test')
            train, test = load_ratings_100k(train_path, test_path)
        data['ml100k'] = {'train': train, 'test': test}
    if args.dataset in ('1m', 'both'):
        train, test = load_ratings_1m(DATA_1M, random_seed=RANDOM_SEED, fold=args.fold)
        data['ml1m'] = {'train': train, 'test': test}
    if args.fold is None:
        for ds in data:
            train = data[ds]['train']
            test = data[ds]['test']
            data[ds]['full'] = np.concatenate([train, test], axis=0) if train is not None and test is not None else None
    else:
        for ds in data:
            data[ds]['full'] = None
    return data


def _run_for_dataset_k(
    dataset_name: str,
    K: int,
    args,
    use_sp: bool,
    train: np.ndarray,
    test: np.ndarray,
    full_data: Optional[np.ndarray],
    run_command: str,
) -> List[dict]:
    if _should_skip_existing_run(
        args,
        dataset_name,
        K,
        args.mode,
        args.fold,
        use_sp,
        args.fold is None and full_data is not None,
    ):
        print(
            f"[skip-existing] {dataset_name} K={K}  "
            f"tag={_format_hyperparam_tag(K)}  use_svdpp={use_sp}"
        )
        return []

    if args.fold is None and full_data is not None:
        fold_rows: List[dict] = []
        print(f"\n[{dataset_name}] 5-Fold CV başlatılıyor (n_splits=5, shuffle=True, random_state=42)")
        for fold_idx, (cv_train, cv_test) in enumerate(
            _build_kfold_splits(full_data, n_splits=5, shuffle=True, random_state=42),
            start=1,
        ):
            print(f"\n[{dataset_name}] Fold {fold_idx}/5")
            rows = run_dataset(
                dataset_name, cv_train, cv_test,
                algo_filter=args.algo,
                run_global=not args.no_global,
                mode=args.mode,
                cluster_workers=args.jobs,
                algo_workers=args.algo_jobs,
                assign_root=args.assign_root,
                assign_suffix=args.assign_suffix,
                assignment_k=K,
                assignment_k_100k=None,
                assignment_k_1m=None,
                weighted_v=args.weighted_v,
                use_bias=not args.no_bias,
                use_cluster_bias=args.cluster_bias,
                run_cluster_avg=not args.no_cluster_avg,
                do_cluster_knn=not args.no_cluster_knn,
                top_n=args.top_n,
                relevance_threshold=args.relevance_threshold,
                similarity=args.similarity,
                knn=args.knn,
                min_common=args.min_common,
                use_svdpp=use_sp,
                run_command=run_command,
                args=args,
                fold=fold_idx,
            )
            for row in rows:
                row['cv_fold'] = fold_idx
            fold_rows.extend(rows)
        mean_rows = _aggregate_fold_results(fold_rows, n_splits=5)
        for row in mean_rows:
            _save_row_to_db(dataset_name, row, K, args)
        _save_cv_mean_results(dataset_name, K, args.mode, mean_rows, run_command)
        return mean_rows

    return run_dataset(
        dataset_name, train, test,
        algo_filter=args.algo,
        run_global=not args.no_global,
        mode=args.mode,
        cluster_workers=args.jobs,
        algo_workers=args.algo_jobs,
        assign_root=args.assign_root,
        assign_suffix=args.assign_suffix,
        assignment_k=K,
        assignment_k_100k=None,
        assignment_k_1m=None,
        weighted_v=args.weighted_v,
        use_bias=not args.no_bias,
        use_cluster_bias=args.cluster_bias,
        run_cluster_avg=not args.no_cluster_avg,
        do_cluster_knn=not args.no_cluster_knn,
        top_n=args.top_n,
        relevance_threshold=args.relevance_threshold,
        similarity=args.similarity,
        knn=args.knn,
        min_common=args.min_common,
        use_svdpp=use_sp,
        run_command=run_command,
        args=args,
        fold=args.fold,
    )


def _print_run_header(args, epoch_pairs, combos, ld_list, lr_list, reg_list):
    print("=" * 60)
    print("WNMF DENEY")
    print("=" * 60)
    print(f"Dataset       : {args.dataset}")
    print(f"Mod           : {args.mode}")
    print(f"Algoritmalar  : {args.algo or ALGO_LABELS}")
    print(f"Epoch çiftleri: {len(epoch_pairs)} adet {'(grid/Cartesian)' if args.epochs_grid else '(zip/yayılım)'}")
    if len(epoch_pairs) <= 12:
        print(f"  -> {epoch_pairs}")
    else:
        print(f"  -> {epoch_pairs[:6]} ... {epoch_pairs[-3:]}")
    print(f"Hiper tarama  : {len(combos)} kombinasyon (epoch çift × latent × lr × reg)")
    print(f"  latent_dim    : {ld_list}")
    print(f"  lr            : {lr_list}")
    print(f"  reg           : {reg_list}")
    print(f"Küme işçileri : {args.jobs or 'otomatik (CPU)'}")
    print(f"Algo işçileri : {args.algo_jobs if args.algo_jobs else 'sıralı (1)'}")
    if args.assignment_k is not None:
        print(f"Assignment K  : {list(args.assignment_k)}")
    else:
        print(f"Assignment K  : --k-100k={args.assignment_k_100k}  --k-1m={args.assignment_k_1m} "
              f"(veya varsayılan 90/150)")
    if args.assign_root:
        print(f"Assignment kök: {args.assign_root}")
    if args.sync_assign_suffix_latent:
        print("assign_suffix   : --sync-assign-suffix-latent (ilk _wnmf<N> → mevcut latent_dim)")
    if args.skip_existing:
        print("Mevcut sonuçlar : --skip-existing (aynı hyperparam_tag + use_svdpp varsa K atlanır)")
    if args.compare_mf_svdpp:
        print("MF vs SVD++   : --compare-mf-svdpp (her hiperparametre seti iki kez: MF sonra SVD++)")
    elif args.svdpp:
        print("Model         : SVD++ açık (--svdpp)")
    print("=" * 60)


def main():
    args        = parse_args()
    # --sync-assign-suffix-latent için taban suffix (her hiperparametre turunda yeniden uygulanır)
    args.assign_suffix_cli = getattr(args, 'assign_suffix', '') or ''
    RUN_COMMAND = shlex.join(sys.argv)

    if _DB_AVAILABLE:
        import sys as _sys
        args.k = args.assignment_k[0] if args.assignment_k else None
        args.reg = args.regularization[0] if args.regularization is not None else REGULARIZATION
        args.lr = args.learning_rate[0] if args.learning_rate is not None else LEARNING_RATE
        _CURRENT_RUN_ID = start_run(
            command=' '.join(_sys.argv),
            dataset=_normalize_db_dataset_name(args.dataset),
            k=args.k,
            preprocessing=_normalize_db_preprocessing(getattr(args, 'assign_suffix', '')),
            latent_dim=(args.latent_dim[0] if args.latent_dim is not None else LATENT_DIM),
            epochs_global=(args.epochs_global[0] if args.epochs_global is not None else N_EPOCHS_GLOBAL),
            epochs_cluster=(args.epochs_cluster[0] if args.epochs_cluster is not None else N_EPOCHS_CLUSTER),
            reg=args.reg,
            lr=args.lr,
            note=getattr(args, 'note', None),
        )

    ld_list  = args.latent_dim if args.latent_dim is not None else [LATENT_DIM]
    reg_list = args.regularization if args.regularization is not None else [REGULARIZATION]
    lr_list  = args.learning_rate if args.learning_rate is not None else [LEARNING_RATE]
    eg_list  = args.epochs_global if args.epochs_global is not None else [N_EPOCHS_GLOBAL]
    ec_list  = args.epochs_cluster if args.epochs_cluster is not None else [N_EPOCHS_CLUSTER]

    try:
        if args.epochs_grid:
            epoch_pairs = _epoch_cartesian_pairs(eg_list, ec_list)
        else:
            epoch_pairs = _expand_epoch_pairs(eg_list, ec_list)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        sys.exit(2)

    combos      = list(product(epoch_pairs, ld_list, lr_list, reg_list))
    multi_hyper = len(combos) > 1

    os.makedirs(OUT_ROOT, exist_ok=True)

    _print_run_header(args, epoch_pairs, combos, ld_list, lr_list, reg_list)

    t_total  = time.time()
    all_rows = []
    all_tags = _collect_dataset_tags(args)

    multi_k = args.assignment_k is not None and len(args.assignment_k) > 1

    datasets = _load_datasets(args)

    svdpp_sequence = [False, True] if args.compare_mf_svdpp else [bool(args.svdpp)]

    for use_sp in svdpp_sequence:
        tag_sp = 'SVD++' if use_sp else 'MF'
        for (eg, ec), ld, lr, reg in combos:
            N_EPOCHS_GLOBAL  = eg
            N_EPOCHS_CLUSTER = ec
            LATENT_DIM       = ld
            LEARNING_RATE    = lr
            REGULARIZATION   = reg
            args.reg = reg
            args.lr = lr
            args.latent_dim = ld
            args.epochs_global = eg
            args.epochs_cluster = ec
            print(f"\n>>> [{tag_sp}] Hiperparametre seti: ld={ld} lr={lr} reg={reg} eg={eg} ec={ec}")

            args.assign_suffix = _effective_assign_suffix(args)
            if args.sync_assign_suffix_latent and args.assign_suffix_cli:
                print(f"  assign_suffix (senkron): {args.assign_suffix!r}")

            combo_rows: List[dict] = []

            for dataset_name in ('ml100k', 'ml1m'):
                if dataset_name not in datasets:
                    continue
                for K in _iter_assignment_k(args, dataset_name):
                    combo_rows.extend(_run_for_dataset_k(
                        dataset_name=dataset_name,
                        K=K,
                        args=args,
                        use_sp=use_sp,
                        train=datasets[dataset_name]['train'],
                        test=datasets[dataset_name]['test'],
                        full_data=datasets[dataset_name]['full'],
                        run_command=RUN_COMMAND,
                    ))

            all_rows.extend(combo_rows)

    if all_rows and len(all_tags) > 1 and not multi_k and not multi_hyper:
        mode_tag     = f"mode-{args.mode}"
        combined_key = '__'.join(all_tags + [mode_tag])
        combo_base   = os.path.join(OUT_ROOT, 'combined', combined_key)
        combo_run    = _next_run_index(combo_base)
        combo_dir    = os.path.join(combo_base, f'run{combo_run}')
        fname        = f"wnmf_all_results_{combined_key}.csv"
        save_results(all_rows, combo_dir, fname, run_command=RUN_COMMAND)

    if _DB_AVAILABLE and _CURRENT_RUN_ID:
        finish_run(_CURRENT_RUN_ID, status='done')

    print(f"\n{'='*60}")
    print(f"TAMAMLANDI — {(time.time()-t_total)/60:.1f} dakika")
    print(f"Çıktı kökü: {OUT_ROOT}/  (dataset başına: .../{{dataset}}/k{{K}}/run{{N}}/)")
    print("=" * 60)


if __name__ == '__main__':
    main()