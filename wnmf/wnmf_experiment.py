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
    python wnmf_experiment.py --dataset 100k
    python wnmf_experiment.py --dataset 100k --algo H4_MFO+HHO
    python wnmf_experiment.py --dataset 100k --algo H9_QSA+CDO H12_MFO+CDO
    python wnmf_experiment.py --dataset 100k --algo LIT_GOA LIT_GWO LIT_SSA
    python wnmf_experiment.py --dataset 100k --no-global
    python wnmf_experiment.py --dataset 100k --mode sharedV   # sadece sharedV
    python wnmf_experiment.py --dataset 100k --mode all       # her iki mod
    python wnmf_experiment.py --dataset 100k --jobs 4        # küme başına 4 süreç (-j 4)
    python wnmf_experiment.py --dataset 100k --algo-jobs 4   # algo başına 4 süreç (yeni)
    python wnmf_experiment.py --dataset 100k --algo-jobs 0   # algo-düzeyinde otomatik paralel (yeni)
    python wnmf_experiment.py --dataset 100k --k 70        # assignment ..._k70 ile aynı K
    python wnmf_experiment.py --dataset 100k --k 20 30 50 90  # birden fazla K sırayla (tek komut)
    python wnmf_experiment.py --dataset 100k --k 30 --no-global --latent-dim 10 20 50 100 --algo H4_MFO+HHO

Birden fazla K ve latent + atlanmış koşular (assignment suffix'teki wnmf boyutunu latent ile eşle):
    python wnmf_experiment.py --dataset 100k --mode sharedV --k 20 30 70 \\
        --latent-dim 10 40 \\
        --assign-suffix _pruneu5_i10_zscore_wnmf20_inmed_trim5_95_fuzzy \\
        --sync-assign-suffix-latent --skip-existing \\
        --assign-root mealpy/results/assignments_lof
    python wnmf_experiment.py --dataset 100k --mode baselines --k 30 --algo H4_MFO+HHO \\
        --cluster-knn-backend native --knn 30 --min-common 3 \\
        --assign-root mealpy/results/assignments_lof
    python wnmf_experiment.py --dataset 100k --k 30 --no-global --reg 0.001 0.01 0.1 --lr 0.001 0.01
    python wnmf_experiment.py --dataset 100k --k 30 --epochs-global 50 100 150 200 --epochs-cluster 25 50 75 100
    python wnmf_experiment.py --dataset 1m --k 30 --algo H4_MFO+HHO --epochs-grid \\
        --epochs-global 50 100 150 --epochs-cluster 50 75 100 150 200
    python wnmf_experiment.py --dataset both --k 30 70 --algo H4_MFO+HHO --compare-mf-svdpp
    python wnmf_experiment.py --dataset 1m --k 70            # .../B1_HHO_k70/ vb.
    python wnmf_experiment.py --k-100k 90 --k-1m 70        # dataset başına ayrı K (--k çoklu ile birlikte kullanılmaz)

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
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

RelevanceThreshold = Union[float, str]
RELEVANCE_THRESHOLD_CLUSTER_MEAN = 'cluster_mean'

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
        get_assignment_id_by_suffix,
        ensure_assignment_on_disk,
    )
    init_db()
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False
_CURRENT_RUN_ID = None

from wnmf_model import ALSModel, ClusterWNMF, WNMFModel, WNMFSharedV
from wnmf_utils import (
    load_ratings_100k,
    load_ratings_100k_all,
    load_ratings_1m,
    load_assignment,
    load_memberships,
    load_centroids,
    load_user_features,
    resolve_test_cluster_ids,
    split_by_cluster,
    remap_user_ids,
    compute_cluster_item_means_imputed,
    save_dataframe_csv,
    save_results,
)

# ============================================================
# AYARLAR
# ============================================================

DATA_100K_TRAIN = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-100k', 'u1.base')
DATA_100K_TEST  = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-100k', 'u1.test')
DATA_100K_ALL   = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-100k', 'u.data')
DATA_1M         = os.path.join(os.path.dirname(BASE_DIR), 'data', 'ml-1m', 'ratings.dat')
ASSIGN_ROOT     = os.path.join(os.path.dirname(BASE_DIR), 'mealpy','results', 'assignments_lof')
OUT_ROOT        = os.path.join(os.path.dirname(BASE_DIR), 'results', 'wnmf')

# generate_assignments.py K_100K_DEFAULT / K_1M_DEFAULT ile aynı olmalı (klasör adı _k{K} eki)
ASSIGN_K_DEFAULT_100K = 7
ASSIGN_K_DEFAULT_1M   = 7

LATENT_DIM       = 20
LEARNING_RATE    = 0.01
REGULARIZATION   = 0.01
N_EPOCHS_GLOBAL  = 100   # global V ve global baseline için
N_EPOCHS_CLUSTER = 100   # küme U eğitimi için
RANDOM_SEED      = 42

# Soft-membership (FCM): bu eşiğin altındaki küme ağırlıkları ihmal edilir.
# Kalan ağırlıklarla yeniden normalize edilir; hiçbiri geçmezse argmax küme kullanılır.
SOFT_MEMBERSHIP_THRESHOLD = 0.10

ALGO_LABELS = [
    'B0_KMEANS',
    'B1_HHO', 'B2_HGS', 'B3_MFO', 'LF_HHO', 'IWO_HHO', 'SFOA', 'SFOA_06', 'H1_HHO+HGS', 'H4_MFO+HHO', 'DOA',
    'H9_QSA+CDO', 'H12_MFO+CDO', 'H13_HHO+GAop', 'HA_AVOAHGS',
    # generate_assignments.py ALGO_CONFIG — önce --lof ile atama üretin
    'LIT_CIRCLESA', 'LIT_GOA', 'LIT_GWO', 'LIT_SSA', 'LIT_PSO', 'LIT_CSO',
    'B_AVOA',
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


def _format_hyperparam_tag(
    k_used: int,
    cknn_suffix: Optional[int] = None,
) -> str:
    """CSV / dosya adlarında hangi hiperparametre koşusunun olduğunu gösterir."""
    base = (
        f"k{k_used}_ld{LATENT_DIM}_eg{N_EPOCHS_GLOBAL}_ec{N_EPOCHS_CLUSTER}_"
        f"lr{LEARNING_RATE:g}_r{REGULARIZATION:g}"
    )
    if cknn_suffix is not None:
        base += f"_cknn{int(cknn_suffix)}"
    return base


def _result_row_meta(
    k_used: int,
    cknn_suffix: Optional[int] = None,
) -> dict:
    return {
        'assignment_k'    : k_used,
        'latent_dim'      : LATENT_DIM,
        'learning_rate'   : LEARNING_RATE,
        'regularization'  : REGULARIZATION,
        'epochs_global'   : N_EPOCHS_GLOBAL,
        'epochs_cluster'  : N_EPOCHS_CLUSTER,
        'hyperparam_tag'  : _format_hyperparam_tag(k_used, cknn_suffix),
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
    knn_list = getattr(args, 'knn', [30])
    if isinstance(knn_list, int):
        knn_list = [knn_list]
    if len(list(knn_list)) > 1:
        # Farklı kNN değerleri farklı satırlarda; tek etiketle güvenilir eşleşme olmaz
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


# KMeans refinement klasör etiketi (generate_assignments: B0_KMEANS hariç _kmref eklenir)
_ASSIGN_KMREF_SUFFIX = '_kmref'


def _assign_suffix_strip_trailing_kmref(assign_suffix: str) -> Optional[str]:
    """Sonda _kmref varsa kaldırılmış sonek; yoksa None."""
    if assign_suffix.endswith(_ASSIGN_KMREF_SUFFIX):
        return assign_suffix[: -len(_ASSIGN_KMREF_SUFFIX)]
    return None


def _assign_suffix_trailing_cluster_k(assign_suffix: str) -> Optional[int]:
    """
    generate_assignments yeni klasör adının sonundaki küme K'sı:
    ..._k{K}, ..._k{K}_pwcss, ..._k{K}_kmref veya ..._k{K}_pwcss_kmref.
    _wnmf30 gibi latent etiketleri ile karışmaz (_wnmf{digits} ortada kalır).
    """
    if not assign_suffix:
        return None
    base = assign_suffix
    if base.endswith(_ASSIGN_KMREF_SUFFIX):
        base = base[: -len(_ASSIGN_KMREF_SUFFIX)]
    m = re.search(r'_k(\d+)(?:_pwcss)?$', base)
    if m:
        return int(m.group(1))
    m = re.search(r'_k(\d+)$', base)
    return int(m.group(1)) if m else None


def _assign_suffix_contains_cluster_k(assign_suffix: str, k_used: int) -> bool:
    """--assign-suffix sondaki küme K'sı k_used ile aynı mı (geriye dönük)."""
    trailing = _assign_suffix_trailing_cluster_k(assign_suffix)
    if trailing is not None:
        return trailing == int(k_used)
    if not assign_suffix:
        return False
    return (
        re.search(rf'_k{int(k_used)}(?=_|$)', assign_suffix) is not None
    )


def _algo_assignment_dir(
    assign_root: str,
    dataset_name: str,
    label: str,
    k_used: int,
    assign_suffix: str = '',
) -> str:
    """
    Assignment klasörü: {label}{assign_suffix} veya eski düzen {label}_k{K}{assign_suffix}.

    Yeni düzen (suffix sonunda _k{K}): label önüne _k{K} eklenmez; --k ile suffix K'sı
    farklı olsa bile klasör eşlemesi suffix sondaki K'ya göre yapılır.
    """
    default_k = ASSIGN_K_DEFAULT_100K if dataset_name == 'ml100k' else ASSIGN_K_DEFAULT_1M
    trailing_k = _assign_suffix_trailing_cluster_k(assign_suffix)
    if trailing_k is not None:
        label_k = ''
    elif k_used == default_k:
        label_k = ''
    else:
        label_k = f'_k{k_used}'
    return os.path.join(assign_root, dataset_name, f'{label}{label_k}{assign_suffix}')


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
            float(model.mu)
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
    soft_threshold: float = SOFT_MEMBERSHIP_THRESHOLD,
) -> float:
    num = 0.0
    den = 0.0
    for cid, w in enumerate(np.asarray(memberships_row, dtype=np.float64)):
        if w < soft_threshold:
            continue
        profile = profiles.get(int(cid))
        if profile is None:
            continue
        pred_k = predictor_fn(profile, int(user_id), int(item_id))
        num += float(w) * float(pred_k)
        den += float(w)
    if den < 1e-8:
        w_row = np.asarray(memberships_row, dtype=np.float64)
        cid = int(np.argmax(w_row))
        profile = profiles.get(cid)
        if profile is None:
            return float('nan')
        return predictor_fn(profile, int(user_id), int(item_id))
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
        train=train,
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


def run_global_als(train, test, n_items, verbose=False, use_bias=True,
                   use_svdpp: bool = False, top_n: int = 10,
                   relevance_threshold: float = 4.0):
    """
    Tüm kullanıcılara tek ALS model — ablation baseline.
    'Kümeleme eklemek ne kadar iyileştiriyor?' sorusunu cevaplar.
    """
    print("\n  [Global ALS] başlıyor...")
    t0      = time.time()
    n_users = int(train[:, 0].max()) + 1

    model = ALSModel(
        n_users        = n_users,
        n_items        = n_items,
        latent_dim     = LATENT_DIM,
        regularization = REGULARIZATION,
        n_epochs       = N_EPOCHS_GLOBAL,
        random_seed    = RANDOM_SEED,
    )
    model.fit(train)
    mae, rmse = model.evaluate(test)
    pred = model.predict(
        test[:, 0].astype(np.int32),
        test[:, 1].astype(np.int32),
    )
    eval_rows = np.column_stack((test[:, 0], test[:, 1], test[:, 2], pred))
    precision, recall, f1, ndcg = _compute_topn_metrics(
        eval_rows, top_n=top_n, threshold=relevance_threshold,
        train=train,
    )

    print(f"  [Global ALS] MAE={mae:.4f}  RMSE={rmse:.4f}  ({time.time()-t0:.1f}s)")
    return {
        'scenario'    : 'global',
        'algo_label'  : 'GLOBAL_ALS',
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


def run_global_svd(train, test, n_items, top_n: int = 10,
                   relevance_threshold: float = 4.0,
                   n_factors: int = 100, n_epochs: int = 20,
                   lr_all: float = 0.005, reg_all: float = 0.02):
    """
    Surprise SVD (Simon Funk biased MF) — global baseline.
    Non-negativity kısıtı yok; WNMF ile karşılaştırma için kullanılır.

    Tahmin: r̂_ui = μ + b_u + b_i + q_i · p_u
    WNMF'den farkı: U/V negatif olabilir → daha düşük MAE.
    """
    try:
        from surprise import SVD as SurpriseSVD, Reader
        from surprise import Dataset as SurpriseDataset
    except ImportError:
        print("  [Global SVD] HATA: 'surprise' kurulu değil → pip install scikit-surprise",
              file=sys.stderr)
        return None

    print(f"\n  [Global SVD] başlıyor... (n_factors={n_factors}, n_epochs={n_epochs})")
    t0 = time.time()

    # numpy array → Surprise trainset
    reader = Reader(rating_scale=(1, 5))
    df_train = pd.DataFrame({
        'uid': train[:, 0].astype(int).astype(str),
        'iid': train[:, 1].astype(int).astype(str),
        'rating': train[:, 2].astype(float),
    })
    surprise_data = SurpriseDataset.load_from_df(df_train[['uid', 'iid', 'rating']], reader)
    trainset = surprise_data.build_full_trainset()

    algo = SurpriseSVD(n_factors=n_factors, n_epochs=n_epochs,
                       lr_all=lr_all, reg_all=reg_all, random_state=RANDOM_SEED)
    algo.fit(trainset)

    # Test tahmini
    preds, trues = [], []
    for row in test:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        p = algo.predict(str(u), str(i)).est
        p = float(np.clip(p, 1.0, 5.0))
        preds.append(p)
        trues.append(r)

    preds = np.array(preds, dtype=np.float32)
    trues = np.array(trues, dtype=np.float32)
    errors = trues - preds
    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    eval_rows = np.column_stack((test[:, 0], test[:, 1], trues, preds))
    precision, recall, f1, ndcg = _compute_topn_metrics(
        eval_rows, top_n=top_n, threshold=relevance_threshold,
        train=train,
    )

    print(f"  [Global SVD] MAE={mae:.4f}  RMSE={rmse:.4f}  ({time.time()-t0:.1f}s)")
    return {
        'scenario'        : 'global',
        'algo_label'      : f'GLOBAL_SVD_f{n_factors}',
        'mae'             : mae,
        'rmse'            : rmse,
        'gray_mae'        : float('nan'),
        'gray_rmse'       : float('nan'),
        'white_mae'       : mae,
        'white_rmse'      : rmse,
        'n_clusters'      : 1,
        'n_train'         : len(train),
        'n_test'          : len(test),
        'time_seconds'    : time.time() - t0,
        'accuracy'        : float('nan'),
        'precision_at_10' : precision,
        'recall_at_10'    : recall,
        'f1_at_10'        : f1,
        'ndcg_at_10'      : ndcg,
    }


def _knn_predict_from_sims(
    u: int,
    i: int,
    sims: List[Tuple[float, int]],
    user_ratings: dict,
    user_means: dict,
    global_mean: float,
    k_neighbors: int,
    base_mean: Optional[float] = None,
) -> float:
    """Ağırlıklı Pearson kNN tahmini (ortak formül). base_mean verilirse mean_u yerine o kullanılır."""
    if base_mean is not None:
        base = float(base_mean)
    else:
        base = float(user_means.get(u, global_mean))
    if not sims:
        return float(np.clip(base, 1.0, 5.0))
    sims.sort(key=lambda x: -abs(x[0]))
    top_k = sims[:k_neighbors]
    num = sum(
        s * (user_ratings[v][i] - user_means.get(v, global_mean))
        for s, v in top_k
    )
    den = sum(abs(s) for s, _ in top_k)
    if den < 1e-8:
        return float(np.clip(base, 1.0, 5.0))
    return float(np.clip(base + num / den, 1.0, 5.0))


def _cluster_avg_mu_base(
    u: int,
    i: int,
    cid: int,
    *,
    cluster_avg_base: str,
    cluster_means: dict,
    user_means: dict,
    item_mean_offsets: dict,
    global_mean: float,
) -> float:
    """ClusterAvg taban ortalaması: user | cluster | cluster_item_pop."""
    mode = (cluster_avg_base or 'user').strip().lower()
    if mode == 'cluster':
        return float(np.clip(cluster_means.get(cid, global_mean), 1.0, 5.0))
    if mode in ('cluster_item_pop', 'cluster+item', 'cluster_item'):
        pop_off = float(item_mean_offsets.get(i, 0.0))
        return float(np.clip(cluster_means.get(cid, global_mean) + pop_off, 1.0, 5.0))
    return float(np.clip(user_means.get(u, global_mean), 1.0, 5.0))


def _predict_weighted_cluster_avg(
    u: int,
    i: int,
    cid: int,
    cluster_users: dict,
    user_ratings: dict,
    user_means: dict,
    cluster_means: dict,
    item_mean_offsets: dict,
    global_mean: float,
    sim_index: dict,
    *,
    cluster_avg_base: str = 'user',
    similarity: str = 'pearson',
    item_popularity: dict,
    min_common: int = 3,
    sig_weight: int = 0,
    sim_amp: float = 1.0,
    k_neighbors: int = 0,
) -> float:
    """
    Küme-içi ağırlıklı ortalama (GOA / Sparrow / Firefly tarzı):
      pred(u,i) = mu_base + Σ_v sim(u,v)·(r(v,i)-mean_v) / Σ_v |sim(u,v)|
    mu_base: --cluster-avg-base (user | cluster | cluster_item_pop).
    yalnızca cid kümesindeki ve item i'yi puanlamış v için.
    """
    base = _cluster_avg_mu_base(
        u,
        i,
        cid,
        cluster_avg_base=cluster_avg_base,
        cluster_means=cluster_means,
        user_means=user_means,
        item_mean_offsets=item_mean_offsets,
        global_mean=global_mean,
    )
    members = cluster_users.get(cid, [])
    sim_map = {v: s for s, v in sim_index.get(u, [])}
    sims: List[Tuple[float, int]] = []

    for v in members:
        if v == u:
            continue
        if i not in user_ratings.get(v, {}):
            continue
        s = sim_map.get(v)
        if s is None:
            s = knn_user_similarity(
                u,
                v,
                similarity=similarity,
                user_ratings=user_ratings,
                user_means=user_means,
                item_popularity=item_popularity,
                min_common=min_common,
                sig_threshold=sig_weight,
                sim_amp=sim_amp,
            )
        if abs(s) < 1e-8:
            continue
        sims.append((float(s), v))

    if not sims:
        return float(np.clip(base, 1.0, 5.0))

    k_use = int(k_neighbors) if int(k_neighbors) > 0 else len(sims)
    return _knn_predict_from_sims(
        u,
        i,
        sims,
        user_ratings,
        user_means,
        global_mean,
        k_use,
        base_mean=base,
    )


def _build_knn_sim_index(
    user_ratings: dict,
    user_means: dict,
    item_popularity: dict,
    similarity: str = 'pearson',
    min_common: int = 3,
    sig_weight: int = 0,
    sim_amp: float = 1.0,
    allowed_neighbors: Optional[Set[int]] = None,
) -> dict:
    """
    Kullanıcı–kullanıcı benzerlik indeksi: sim_index[u] = [(sim, v), ...].
    allowed_neighbors verilirse yalnızca o kümeden komşu eklenir (gray→white global KNN).
    """
    item_users: dict = {}
    for u, items in user_ratings.items():
        for it in items:
            item_users.setdefault(it, set()).add(u)

    all_users = list(user_ratings.keys())
    sim_index: dict = {u: [] for u in all_users}
    for ua in all_users:
        candidates: dict = {}
        for it in user_ratings[ua]:
            for v in item_users.get(it, []):
                if v == ua:
                    continue
                if allowed_neighbors is not None and v not in allowed_neighbors:
                    continue
                candidates[v] = candidates.get(v, 0) + 1

        for v, cnt in candidates.items():
            if cnt < min_common:
                continue
            if v <= ua:
                continue

            s = knn_user_similarity(
                ua, v,
                similarity=similarity,
                user_ratings=user_ratings,
                user_means=user_means,
                item_popularity=item_popularity,
                min_common=min_common,
                sig_threshold=sig_weight,
                sim_amp=sim_amp,
            )
            if abs(s) < 1e-8:
                continue

            sim_index[ua].append((s, v))
            sim_index[v].append((s, ua))

    for u in all_users:
        sim_index[u].sort(key=lambda x: -abs(x[0]))
    return sim_index


def run_global_knn(train, test, n_items,
                   similarity: str = 'pearson',
                   k_neighbors: int = 20,
                   min_common: int = 3,
                   top_n: int = 10,
                   relevance_threshold: float = 4.0):
    """
    Kümeleme olmadan tüm kullanıcılar üzerinde User-Based KNN.

    'Kümeleme KNN'i gerçekten iyileştiriyor mu?' sorusunu cevaplar:
        GLOBAL_KNN   → kümeleme yok, tüm veri
        ClusterKNN   → küme bazlı, aynı benzerlik metriği
    Fark pozitifse kümeleme işe yarıyor.

    Hız optimizasyonu: kullanıcı-kullanıcı similarity matrisi bir kez
    önceden hesaplanır (O(n²) build, O(1) lookup), test tahminleri hızlı.
    """
    print(f"\n  [Global KNN] başlıyor... (sim={similarity}, k={k_neighbors})")
    t0 = time.time()

    # ── Veri yapıları ────────────────────────────────────────────
    global_mean = float(train[:, 2].mean())
    user_ratings: dict = {}
    for row in train:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        user_ratings.setdefault(u, {})[i] = r

    user_means = {
        u: float(np.mean(list(d.values())))
        for u, d in user_ratings.items()
    }
    item_popularity = build_item_popularity(user_ratings=user_ratings)

    sim_index = _build_knn_sim_index(
        user_ratings,
        user_means,
        item_popularity,
        similarity=similarity,
        min_common=min_common,
    )
    n_pairs = sum(len(v) for v in sim_index.values()) // 2
    print(f"  [Global KNN] {n_pairs:,} anlamlı çift hesaplandı "
          f"({time.time()-t0:.1f}s)", flush=True)

    def _predict(u: int, i: int) -> float:
        neighbors = [
            (s, v) for s, v in sim_index.get(u, [])
            if i in user_ratings.get(v, {})
        ]
        return _knn_predict_from_sims(
            u, i, neighbors, user_ratings, user_means, global_mean, k_neighbors,
        )

    # ── Test değerlendirmesi ──────────────────────────────────────
    preds, trues = [], []
    eval_rows = []
    for row in test:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        p = _predict(u, i)
        preds.append(p)
        trues.append(r)
        eval_rows.append((u, i, r, p))

    preds_arr = np.array(preds, dtype=np.float32)
    trues_arr = np.array(trues, dtype=np.float32)
    errors = trues_arr - preds_arr
    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    precision, recall, f1, ndcg = _compute_topn_metrics(
        np.array(eval_rows, dtype=np.float32),
        top_n=top_n,
        threshold=relevance_threshold,
        train=train,
    )

    elapsed = time.time() - t0
    print(f"  [Global KNN] MAE={mae:.4f}  RMSE={rmse:.4f}  ({elapsed:.1f}s)")
    return {
        'scenario'        : 'global',
        'algo_label'      : f'GLOBAL_KNN_{similarity}_k{k_neighbors}',
        'mae'             : mae,
        'rmse'            : rmse,
        'gray_mae'        : float('nan'),
        'gray_rmse'       : float('nan'),
        'white_mae'       : mae,
        'white_rmse'      : rmse,
        'n_clusters'      : 1,
        'n_train'         : len(train),
        'n_test'          : len(test),
        'time_seconds'    : elapsed,
        'accuracy'        : float('nan'),
        'precision_at_10' : precision,
        'recall_at_10'    : recall,
        'f1_at_10'        : f1,
        'ndcg_at_10'      : ndcg,
        'k_neighbors'     : k_neighbors,
    }


def _surprise_similarity_name(similarity: str, *, baseline: bool = False) -> str:
    """Surprise sim_options['name']: msd, pearson, pearson_baseline, cosine."""
    s = (similarity or ('pearson_baseline' if baseline else 'msd')).strip().lower()
    if s == 'pearson_baseline':
        return 'pearson_baseline'
    if s == 'msd':
        return 'msd'
    if s == 'cosine':
        return 'cosine'
    if s in ('pearson', 'pearson_iuf'):
        return 'pearson_baseline' if baseline else 'pearson'
    raise ValueError(f"Surprise KNN desteklemiyor: {similarity}")


def _surprise_sim_options(
    similarity: str,
    min_common: int,
    user_based: bool = True,
    *,
    baseline: bool = False,
) -> dict:
    return {
        'name': _surprise_similarity_name(similarity, baseline=baseline),
        'min_support': max(1, int(min_common)),
        'user_based': bool(user_based),
    }


def _run_global_surprise_knn(
    train,
    test,
    n_items,
    *,
    knn_variant: str,
    k_neighbors: int = 40,
    similarity: str = 'msd',
    min_common: int = 1,
    user_based: bool = True,
    top_n: int = 10,
    relevance_threshold: float = 4.0,
):
    """Surprise global KNN — KNNWithMeans veya KNNBaseline."""
    variant = (knn_variant or 'withmeans').strip().lower()
    use_baseline = variant in ('baseline', 'knnbaseline', 'knn_baseline')
    if use_baseline:
        display = 'KNNBaseline'
        tag = 'KNB'
    else:
        display = 'KNNWithMeans'
        tag = 'KWM'

    sim_label = _surprise_similarity_name(similarity, baseline=use_baseline)
    ub_tag = 'user' if user_based else 'item'
    print(
        f"\n  [Global {display}] başlıyor... "
        f"(k={k_neighbors}, sim={sim_label}, min_support={min_common}, {ub_tag}-based)",
        flush=True,
    )
    t0 = time.time()
    global_mean = float(train[:, 2].mean()) if len(train) else 3.0

    try:
        from surprise import Reader
        reader = Reader(rating_scale=(1, 5))
        use_surprise = True
    except ImportError:
        reader = None
        use_surprise = False
        print(
            f"  [Global {display}] HATA: 'surprise' kurulu değil "
            "(pip install scikit-surprise)",
            file=sys.stderr,
        )
        if not use_baseline:
            print(
                f"  [Global {display}] yerel fallback deneniyor...",
                flush=True,
            )

    if use_surprise and (similarity or '').strip().lower() == 'pearson_iuf':
        print(
            f"  [Global {display}] uyarı: pearson_iuf -> "
            f"{'pearson_baseline' if use_baseline else 'pearson'}",
            flush=True,
        )

    algo = None
    if use_surprise:
        try:
            if use_baseline:
                from surprise import KNNBaseline as AlgoCls
            else:
                from surprise import KNNWithMeans as AlgoCls
            trainset = _numpy_ratings_to_surprise_trainset(train, reader)
            algo = AlgoCls(
                k=max(1, int(k_neighbors)),
                sim_options=_surprise_sim_options(
                    similarity, min_common, user_based, baseline=use_baseline,
                ),
                verbose=False,
            )
            algo.fit(trainset)
        except ImportError:
            use_surprise = False

    if algo is None and not use_baseline:
        algo = _fit_local_knn_with_means(
            train,
            k_neighbors=k_neighbors,
            similarity=similarity,
            min_common=min_common,
        )

    if algo is None:
        print(f"  [Global {display}] HATA: model fit edilemedi", file=sys.stderr)
        return None

    preds, trues = [], []
    eval_rows = []
    for row in test:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        p = _predict_surprise_kwm(algo, u, i, global_mean)
        preds.append(p)
        trues.append(r)
        eval_rows.append((u, i, r, p))

    preds_arr = np.array(preds, dtype=np.float32)
    trues_arr = np.array(trues, dtype=np.float32)
    errors = trues_arr - preds_arr
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    precision, recall, f1, ndcg = _compute_topn_metrics(
        np.array(eval_rows, dtype=np.float32),
        top_n=top_n,
        threshold=relevance_threshold,
        train=train,
    )
    elapsed = time.time() - t0
    engine = 'surprise' if use_surprise else 'local'
    print(
        f"  [Global {display}|{engine}] MAE={mae:.4f}  RMSE={rmse:.4f}  ({elapsed:.1f}s)",
        flush=True,
    )
    return {
        'scenario'        : 'global',
        'algo_label'      : f'GLOBAL_{tag}_{sim_label}_{ub_tag}_k{k_neighbors}',
        'mae'             : mae,
        'rmse'            : rmse,
        'gray_mae'        : float('nan'),
        'gray_rmse'       : float('nan'),
        'white_mae'       : mae,
        'white_rmse'      : rmse,
        'n_clusters'      : 1,
        'n_train'         : len(train),
        'n_test'          : len(test),
        'time_seconds'    : elapsed,
        'accuracy'        : float('nan'),
        'precision_at_10' : precision,
        'recall_at_10'    : recall,
        'f1_at_10'        : f1,
        'ndcg_at_10'      : ndcg,
        'similarity'      : sim_label,
        'k_neighbors'     : k_neighbors,
        'knn_mode'        : f'global_{tag.lower()}_{engine}',
        'surprise_knn'    : display,
    }


def run_global_surprise_knn_with_means(
    train,
    test,
    n_items,
    k_neighbors: int = 40,
    similarity: str = 'msd',
    min_common: int = 1,
    user_based: bool = True,
    top_n: int = 10,
    relevance_threshold: float = 4.0,
):
    """Surprise KNNWithMeans — kümeleme olmadan global baseline."""
    return _run_global_surprise_knn(
        train, test, n_items,
        knn_variant='withmeans',
        k_neighbors=k_neighbors,
        similarity=similarity,
        min_common=min_common,
        user_based=user_based,
        top_n=top_n,
        relevance_threshold=relevance_threshold,
    )


def run_global_surprise_knn_baseline(
    train,
    test,
    n_items,
    k_neighbors: int = 40,
    similarity: str = 'pearson_baseline',
    min_common: int = 1,
    user_based: bool = True,
    top_n: int = 10,
    relevance_threshold: float = 4.0,
):
    """
    Surprise KNNBaseline — kümeleme olmadan global baseline.

    r_hat = b_ui + weighted sum of (r_vi - b_vi) over neighbors.
    Önerilen sim: pearson_baseline.
    """
    return _run_global_surprise_knn(
        train, test, n_items,
        knn_variant='baseline',
        k_neighbors=k_neighbors,
        similarity=similarity,
        min_common=min_common,
        user_based=user_based,
        top_n=top_n,
        relevance_threshold=relevance_threshold,
    )


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
        train=train, assignments=assignments,
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
                        relevance_threshold: float = 4.0,
                        hybrid_alpha: Optional[float] = None):
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
            cluster_model.mu = mu_eff.get(int(cid), mu_global_val)

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

    if hybrid_alpha is not None:
        global_mean = float(train[:, 2].mean())
        user_ratings = {}
        for row in train:
            u, i, r = int(row[0]), int(row[1]), float(row[2])
            if u not in user_ratings:
                user_ratings[u] = {}
            user_ratings[u][i] = r
        user_means = {
            u: float(np.mean(list(d.values())))
            for u, d in user_ratings.items()
        }
        n_users = len(assignments)
        cluster_users = {}
        for u in range(n_users):
            cid = int(assignments[u])
            cluster_users.setdefault(cid, []).append(u)

        kwm_models, _, _ = _build_cluster_surprise_knn_models(
            train,
            cluster_users,
            knn_variant='baseline',
            k_neighbors=30,
            similarity='pearson_baseline',
            min_common=3,
        )

        knn_preds = {}
        for row in test:
            u, i, r = int(row[0]), int(row[1]), float(row[2])
            cid = int(assignments[u])
            knn_pred = _predict_knn(
                u, i, cid, user_ratings, cluster_users, user_means,
                similarity='pearson',
                k_neighbors=30,
                min_common=3,
                global_mean=global_mean,
                kwm_models=kwm_models,
            )
            knn_preds[(u, i)] = knn_pred

        for idx, row in enumerate(eval_rows):
            u, i = int(row[0]), int(row[1])
            wnmf_pred = float(row[3])
            knn_pred = knn_preds.get((u, i), wnmf_pred)
            hybrid = (1 - hybrid_alpha) * wnmf_pred + hybrid_alpha * knn_pred
            eval_rows[idx, 3] = float(np.clip(hybrid, 1.0, 5.0))

        all_pred = eval_rows[:, 3].tolist() if len(eval_rows) > 0 else []

    mae, rmse = _compute_metrics(all_true, all_pred)
    precision, recall, f1, ndcg = _compute_topn_metrics(
        eval_rows, top_n=top_n, threshold=relevance_threshold,
        train=train, assignments=assignments,
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
        'scenario'    : (
            'cluster_sharedV'
            if hybrid_alpha is None
            else f'cluster_hybrid_{hybrid_alpha:.2f}'
        ),
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


def compute_precision_recall(train, test, assignments,
                             cluster_item_means, N=10,
                             threshold=3.5):

    # Her kullanıcının train'de izlediği filmleri bul
    user_rated = {}
    for row in train:
        u = int(row[0])
        i = int(row[1])
        if u not in user_rated:
            user_rated[u] = set()
        user_rated[u].add(i)

    # Her kullanıcının test'te beğendiği filmleri bul (threshold üstü)
    user_relevant = {}
    for row in test:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        if r >= threshold:
            if u not in user_relevant:
                user_relevant[u] = set()
            user_relevant[u].add(i)

    precisions, recalls = [], []

    for u in user_relevant:
        cid = int(assignments[u])
        scores = cluster_item_means[cid].copy()

        # Zaten izlenenleri çıkar
        rated = user_rated.get(u, set())
        for i in rated:
            scores[i] = -1.0

        # Top-N öner
        top_n = set(np.argsort(scores)[-N:][::-1].tolist())
        relevant = user_relevant[u]

        hits = len(top_n & relevant)
        precisions.append(hits / N)
        recalls.append(hits / len(relevant) if relevant else 0.0)

    if not precisions:
        return float('nan'), float('nan'), float('nan')

    p = float(np.mean(precisions))
    r = float(np.mean(recalls))
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def _parse_relevance_threshold_arg(value: str) -> RelevanceThreshold:
    """CLI: sabit sayı veya cluster-mean (küme train ortalaması)."""
    s = str(value).strip().lower().replace('-', '_')
    if s in ('cluster_mean', 'mean', 'cluster'):
        return RELEVANCE_THRESHOLD_CLUSTER_MEAN
    return float(value)


def _is_cluster_mean_threshold(threshold: RelevanceThreshold) -> bool:
    return threshold == RELEVANCE_THRESHOLD_CLUSTER_MEAN


def _compute_item_mean_offsets(
    train: np.ndarray,
    global_mean: float,
) -> dict:
    """Film ortalaması − global_mean (popülarite ofseti)."""
    item_sums: dict = {}
    item_counts: dict = {}
    for row in train:
        i, r = int(row[1]), float(row[2])
        item_sums[i] = item_sums.get(i, 0.0) + r
        item_counts[i] = item_counts.get(i, 0) + 1
    return {
        i: float(item_sums[i] / item_counts[i] - global_mean)
        for i in item_sums
        if item_counts[i] > 0
    }


def _compute_cluster_rating_means(
    train: np.ndarray,
    assignments: np.ndarray,
) -> np.ndarray:
    """Her kümenin train rating ortalaması (kümedeki tüm kullanıcı puanları)."""
    n_clusters = int(assignments.max()) + 1
    sums = np.zeros(n_clusters, dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.int64)
    global_mean = float(train[:, 2].mean()) if len(train) else 4.0
    n_users = len(assignments)
    for row in train:
        u, r = int(row[0]), float(row[2])
        if u >= n_users:
            continue
        cid = int(assignments[u])
        sums[cid] += r
        counts[cid] += 1
    means = np.full(n_clusters, global_mean, dtype=np.float32)
    mask = counts > 0
    means[mask] = (sums[mask] / counts[mask]).astype(np.float32)
    return means


def _resolve_relevance_threshold(
    threshold: RelevanceThreshold,
    train: Optional[np.ndarray],
    assignments: Optional[np.ndarray],
    users: Set[int],
) -> Union[float, Dict[int, float]]:
    if not _is_cluster_mean_threshold(threshold):
        return float(threshold)
    global_mean = float(train[:, 2].mean()) if train is not None and len(train) else 4.0
    if assignments is None:
        return global_mean
    cluster_means = _compute_cluster_rating_means(train, assignments)
    resolved: Dict[int, float] = {}
    for u in users:
        if 0 <= u < len(assignments):
            resolved[u] = float(cluster_means[int(assignments[u])])
        else:
            resolved[u] = global_mean
    return resolved


def _user_relevance_threshold(
    threshold: Union[float, Dict[int, float]],
    user_id: int,
) -> float:
    if isinstance(threshold, dict):
        return threshold.get(user_id, 4.0)
    return float(threshold)


def _compute_binary_accuracy(
    true_vals,
    pred_vals,
    threshold: RelevanceThreshold = 3.5,
    *,
    eval_rows: Optional[np.ndarray] = None,
    train: Optional[np.ndarray] = None,
    assignments: Optional[np.ndarray] = None,
):
    if eval_rows is not None and _is_cluster_mean_threshold(threshold):
        if len(eval_rows) == 0:
            return float('nan')
        users = {int(row[0]) for row in eval_rows}
        th = _resolve_relevance_threshold(threshold, train, assignments, users)
        correct = [
            (float(row[2]) >= _user_relevance_threshold(th, int(row[0])))
            == (float(row[3]) >= _user_relevance_threshold(th, int(row[0])))
            for row in eval_rows
        ]
        return float(np.mean(correct))

    if not true_vals:
        return float('nan')
    th_val = float(threshold) if not isinstance(threshold, dict) else 4.0
    true_bin = np.array(true_vals, dtype=np.float32) >= th_val
    pred_bin = np.array(pred_vals, dtype=np.float32) >= th_val
    return float(np.mean(true_bin == pred_bin))


def _compute_topn_metrics(
    eval_rows: np.ndarray,
    top_n: int = 10,
    threshold: RelevanceThreshold = 4.0,
    train: Optional[np.ndarray] = None,
    assignments: Optional[np.ndarray] = None,
):
    """
    eval_rows: [user_id, item_id, true_rating, pred_rating]
    Kullanıcı bazında Top-N kesişiminden Precision/Recall/F1/NDCG hesaplar.
    threshold=cluster_mean: kullanıcının kümesinin train ortalaması (assignments gerekir).
    """
    if eval_rows is None or len(eval_rows) == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')

    by_user = {}
    for row in eval_rows:
        u = int(row[0])
        i = int(row[1])
        r_true = float(row[2])
        r_pred = float(row[3])
        by_user.setdefault(u, []).append((i, r_true, r_pred))

    users = set(by_user.keys())
    th = _resolve_relevance_threshold(threshold, train, assignments, users)

    precisions, recalls, f1s, ndcgs = [], [], [], []
    k = max(1, int(top_n))
    for u, items in by_user.items():
        t_u = _user_relevance_threshold(th, u)
        relevant = {i for i, r_true, _ in items if r_true >= t_u}
        if not relevant:
            continue
        ranked = sorted(items, key=lambda x: x[2], reverse=True)[:k]
        top_items = [i for i, _, _ in ranked]
        hits = len(set(top_items) & relevant)

        p = hits / k
        r = hits / len(relevant) if relevant else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        dcg = 0.0
        for rank_idx, item_id in enumerate(top_items):
            rel = 1.0 if item_id in relevant else 0.0
            dcg += (2.0 ** rel - 1.0) / np.log2(rank_idx + 2.0)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2.0) for i in range(ideal_hits))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        ndcgs.append(ndcg)

    if not precisions:
        return float('nan'), float('nan'), float('nan'), float('nan')
    return (
        float(np.mean(precisions)),
        float(np.mean(recalls)),
        float(np.mean(f1s)),
        float(np.mean(ndcgs)),
    )


def _compute_topn_jaccard(
    eval_rows: np.ndarray,
    top_n: int = 10,
    threshold: RelevanceThreshold = 4.0,
    train: Optional[np.ndarray] = None,
    assignments: Optional[np.ndarray] = None,
) -> float:
    """Kullanıcı bazında Top-N öneri ve ilgili öğe kümeleri Jaccard ortalaması."""
    if eval_rows is None or len(eval_rows) == 0:
        return float('nan')

    by_user = {}
    for row in eval_rows:
        u = int(row[0])
        i = int(row[1])
        r_true = float(row[2])
        r_pred = float(row[3])
        by_user.setdefault(u, []).append((i, r_true, r_pred))

    users = set(by_user.keys())
    th = _resolve_relevance_threshold(threshold, train, assignments, users)
    k = max(1, int(top_n))
    jaccards = []
    for u, items in by_user.items():
        t_u = _user_relevance_threshold(th, u)
        relevant = {int(i) for i, r_true, _ in items if r_true >= t_u}
        if not relevant:
            continue
        ranked = sorted(items, key=lambda x: x[2], reverse=True)[:k]
        top_items = {int(i) for i, _, _ in ranked}
        union = len(top_items | relevant)
        jaccards.append((len(top_items & relevant) / union) if union > 0 else 0.0)

    if not jaccards:
        return float('nan')
    return float(np.mean(jaccards))


def _cluster_avg_predict_kwargs(args) -> dict:
    """run_cluster_average için benzerlik / komşu hiperparametreleri."""
    if args is None:
        return {
            'similarity': 'pearson',
            'min_common': 3,
            'sig_weight': 0,
            'sim_amp': 1.0,
            'cluster_avg_k_neighbors': 0,
            'cluster_avg_base': 'user',
        }
    # 0 = kümedeki tüm uygun rater'lar (literatür formülü); >0 ise üst sınır.
    knn_cap = int(getattr(args, 'cluster_avg_k_neighbors', 0) or 0)
    return {
        'similarity': str(getattr(args, 'similarity', 'pearson') or 'pearson'),
        'min_common': int(getattr(args, 'min_common', 3) or 3),
        'sig_weight': int(getattr(args, 'sig_weight', 0) or 0),
        'sim_amp': float(getattr(args, 'sim_amp', 1.0) or 1.0),
        'cluster_avg_k_neighbors': knn_cap,
        'cluster_avg_base': str(getattr(args, 'cluster_avg_base', 'user') or 'user'),
    }


def _nearest_centroid_bundle(args, assign_dir: str, assignments: np.ndarray) -> dict:
    """--nearest-centroid için centroid yükleme ve kwargs paketi."""
    if args is None or not getattr(args, 'nearest_centroid', False):
        return {
            'centroids': None,
            'nearest_centroid': False,
            'centroid_metric': 'euclidean',
        }
    n_clusters = int(assignments.max()) + 1
    return {
        'centroids': load_centroids(assign_dir, n_clusters),
        'nearest_centroid': True,
        'centroid_metric': getattr(args, 'centroid_metric', 'euclidean'),
    }


def _parse_cluster_weight_alpha(value):
    """--cluster-weight-alpha: sabit float veya 'auto'."""
    s = str(value).strip().lower()
    if s == 'auto':
        return 'auto'
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"--cluster-weight-alpha: sayı veya 'auto' bekleniyor, alınan: {value!r}"
        ) from exc


def compute_cluster_weight_alpha(
    cid: int,
    cluster_users: dict,
    n_users: int,
    base: float = 1.0,
) -> float:
    """Küme-bazlı dinamik alpha: küçük küme → düşük alpha → dış komşulara daha açık."""
    cluster_size = len(cluster_users.get(cid, []))
    k_cluster = len(cluster_users)
    if k_cluster <= 0 or n_users <= 0:
        return float(base)
    return float(base * (cluster_size / n_users) * k_cluster)


def _knn_centroid_bundle(
    args,
    assign_dir: str,
    assignments: np.ndarray,
    knn_mode: Optional[str] = None,
) -> dict:
    """ClusterKNN için centroid kwargs; weighted_cluster modunda best_sol.npy yükler."""
    mode = (knn_mode or getattr(args, 'knn_mode', 'cluster') or 'cluster').strip().lower()
    nc = _nearest_centroid_bundle(args, assign_dir, assignments)
    if mode == 'weighted_cluster' and nc.get('centroids') is None and assign_dir:
        n_clusters = int(assignments.max()) + 1
        loaded = load_centroids(assign_dir, n_clusters)
        if loaded is not None:
            nc = {**nc, 'centroids': loaded}
    alpha: Union[float, str] = 1.0
    base = 1.0
    if args is not None:
        a = getattr(args, 'cluster_weight_alpha', None)
        if a is not None:
            if isinstance(a, str) and a.strip().lower() == 'auto':
                alpha = 'auto'
            else:
                alpha = float(a)
        base = float(getattr(args, 'cluster_weight_base', 1.0) or 1.0)
    nc['cluster_weight_alpha'] = alpha
    nc['cluster_weight_base'] = base
    return nc


def run_cluster_average(train, test, assignments, gray_mask,
                        memberships,
                        n_items, algo_label, top_n: int = 10,
                        relevance_threshold: float = 4.0,
                        centroids=None,
                        nearest_centroid: bool = False,
                        centroid_metric: str = 'euclidean',
                        cluster_avg_hard: bool = False,
                        cluster_avg_leaky: bool = False,
                        cluster_mean_impute_train: bool = False,
                        debug_cluster_avg_hard: bool = False,
                        debug_cluster_avg_hard_limit: int = 5,
                        similarity: str = 'cosine',
                        min_common: int = 3,
                        sig_weight: int = 0,
                        sim_amp: float = 1.0,
                        cluster_avg_k_neighbors: int = 0,
                        cluster_avg_base: str = 'user',
                        assign_dir: Optional[str] = None):
    t0 = time.time()
    cluster_avg_base = (cluster_avg_base or 'user').strip().lower()

    test_cluster_ids = resolve_test_cluster_ids(
        train, assignments, centroids, nearest_centroid, n_items,
        centroid_metric=centroid_metric, algo_label=algo_label,
        assign_dir=assign_dir,
    )

    # Her küme için her item'ın ortalama rating'ini hesapla
    n_clusters = int(assignments.max()) + 1
    # Thakrar et al. (2025) Algorithm 6: kümedeki tüm kullanıcıların ortalaması;
    # soft membership yok (--cluster-avg-hard).
    use_soft = (
        not cluster_avg_hard
        and not cluster_avg_leaky
        and memberships is not None
        and memberships.shape[0] >= len(assignments)
        and memberships.shape[1] == n_clusters
    )

    cluster_item_means = np.zeros((n_clusters, n_items), dtype=np.float32)
    cluster_item_counts = np.zeros((n_clusters, n_items), dtype=np.int32)

    # Ortalama kaynağı: train (doğru holdout) veya train+test (makale-tipi sızıntı)
    mean_rows = np.vstack([train, test]) if cluster_avg_leaky else train
    n_users = len(assignments)
    if cluster_mean_impute_train and not cluster_avg_leaky:
        item_counts = np.zeros(n_items, dtype=np.int32)
        user_counts = np.zeros(n_users, dtype=np.int32)
        for row in train:
            u, i = int(row[0]), int(row[1])
            item_counts[i] += 1
            if u < n_users:
                user_counts[u] += 1
        (
            cluster_item_means,
            cluster_item_counts,
            global_mean,
            item_means_arr,
            user_means_arr,
        ) = compute_cluster_item_means_imputed(
            train,
            assignments,
            gray_mask,
            n_items,
            cluster_avg_hard=cluster_avg_hard,
        )
        print(
            f"  [{algo_label}] küme-ortalama imputation: train eksik hücreler "
            f"küme ortalamasıyla dolduruldu (2 geçiş)",
            flush=True,
        )
    else:
        item_sums = np.zeros(n_items, dtype=np.float64)
        item_counts = np.zeros(n_items, dtype=np.int32)
        user_sums = np.zeros(n_users, dtype=np.float64)
        user_counts = np.zeros(n_users, dtype=np.int32)
        for row in mean_rows:
            u, i, r = int(row[0]), int(row[1]), float(row[2])
            cid = int(assignments[u])
            cluster_item_means[cid, i] += r
            cluster_item_counts[cid, i] += 1
            item_sums[i] += r
            item_counts[i] += 1
            if u < n_users:
                user_sums[u] += r
                user_counts[u] += 1

        global_mean = float(mean_rows[:, 2].mean())
        item_means_arr = np.where(
            item_counts > 0, item_sums / np.maximum(item_counts, 1), global_mean,
        ).astype(np.float32)
        user_means_arr = np.where(
            user_counts > 0, user_sums / np.maximum(user_counts, 1), global_mean,
        ).astype(np.float32)
        for cid in range(n_clusters):
            mask = cluster_item_counts[cid] > 0
            cluster_item_means[cid, mask] /= cluster_item_counts[cid, mask]
            if not cluster_avg_hard:
                cluster_item_means[cid, ~mask] = global_mean
            else:
                cluster_item_means[cid, ~mask] = np.nan

    # ClusterAvg (varsayılan): küme-içi sim ağırlıklı sapma; CalcAvgRating (--cluster-avg-hard) eski düz ortalama.
    use_weighted_cluster_avg = not cluster_avg_hard

    user_ratings: dict = {}
    user_means_dict: dict = {}
    cluster_means_dict: dict = {}
    item_mean_offsets_dict: dict = {}
    cluster_users: dict = {}
    sim_index: dict = {}
    item_popularity: dict = {}

    if use_weighted_cluster_avg:
        for row in mean_rows:
            u_row, i_row, r_row = int(row[0]), int(row[1]), float(row[2])
            user_ratings.setdefault(u_row, {})[i_row] = r_row
        for u_row in range(n_users):
            if user_counts[u_row] > 0:
                user_means_dict[u_row] = float(user_means_arr[u_row])
        for u_row in range(n_users):
            cid_u = int(assignments[u_row])
            if cid_u < 0:
                continue
            cluster_users.setdefault(cid_u, []).append(u_row)
        cluster_mean_arr = _compute_cluster_rating_means(mean_rows, assignments)
        cluster_means_dict = {
            int(cid): float(cluster_mean_arr[cid]) for cid in range(len(cluster_mean_arr))
        }
        item_mean_offsets_dict = _compute_item_mean_offsets(mean_rows, global_mean)
        item_popularity = build_item_popularity(user_ratings=user_ratings)
        sim_index = _build_knn_sim_index(
            user_ratings,
            user_means_dict,
            item_popularity,
            similarity=similarity,
            min_common=min_common,
            sig_weight=sig_weight,
            sim_amp=sim_amp,
        )
        base_labels = {
            'user': 'mean_u',
            'cluster': 'mean_cluster',
            'cluster_item_pop': 'mean_cluster+item_pop',
        }
        base_tag = base_labels.get(cluster_avg_base, cluster_avg_base)
        print(
            f"  [{algo_label}] ClusterAvg: küme-içi ağırlıklı tahmin "
            f"(taban={base_tag}, sim={similarity}, min_common={min_common})",
            flush=True,
        )

    # Test verisinde tahmin yap
    true_vals, pred_vals = [], []
    gray_true, gray_pred = [], []
    eval_rows = []
    src_counts = {
        'weighted_cluster': 0,
        'cluster_mean': 0,
        'item_mean': 0,
        'user_mean': 0,
        'global_mean': 0,
    }
    debug_left = max(int(debug_cluster_avg_hard_limit), 0)

    for row in test:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        if use_soft:
            w = np.asarray(memberships[u], dtype=np.float64)
            active = w >= SOFT_MEMBERSHIP_THRESHOLD
            if not np.any(active) or float(w[active].sum()) < 1e-8:
                if use_weighted_cluster_avg:
                    cid_fb = int(np.argmax(w))
                    pred = _predict_weighted_cluster_avg(
                        u,
                        i,
                        cid_fb,
                        cluster_users,
                        user_ratings,
                        user_means_dict,
                        cluster_means_dict,
                        item_mean_offsets_dict,
                        global_mean,
                        sim_index,
                        cluster_avg_base=cluster_avg_base,
                        similarity=similarity,
                        item_popularity=item_popularity,
                        min_common=min_common,
                        sig_weight=sig_weight,
                        sim_amp=sim_amp,
                        k_neighbors=cluster_avg_k_neighbors,
                    )
                else:
                    pred = float(np.clip(
                        cluster_item_means[int(np.argmax(w)), i], 1.0, 5.0,
                    ))
            else:
                w_use = np.where(active, w, 0.0)
                w_use /= w_use.sum()
                if use_weighted_cluster_avg:
                    pred = 0.0
                    for cid_soft, w_c in enumerate(w_use):
                        if w_c <= 0.0:
                            continue
                        pred += w_c * _predict_weighted_cluster_avg(
                            u,
                            i,
                            int(cid_soft),
                            cluster_users,
                            user_ratings,
                            user_means_dict,
                            cluster_means_dict,
                            item_mean_offsets_dict,
                            global_mean,
                            sim_index,
                            cluster_avg_base=cluster_avg_base,
                            similarity=similarity,
                            item_popularity=item_popularity,
                            min_common=min_common,
                            sig_weight=sig_weight,
                            sim_amp=sim_amp,
                            k_neighbors=cluster_avg_k_neighbors,
                        )
                else:
                    pred = float(np.clip(np.dot(w_use, cluster_item_means[:, i]), 1.0, 5.0))
        elif use_weighted_cluster_avg:
            cid = int(test_cluster_ids[u])
            pred = _predict_weighted_cluster_avg(
                u,
                i,
                cid,
                cluster_users,
                user_ratings,
                user_means_dict,
                cluster_means_dict,
                item_mean_offsets_dict,
                global_mean,
                sim_index,
                cluster_avg_base=cluster_avg_base,
                similarity=similarity,
                item_popularity=item_popularity,
                min_common=min_common,
                sig_weight=sig_weight,
                sim_amp=sim_amp,
                k_neighbors=cluster_avg_k_neighbors,
            )
            src_counts['weighted_cluster'] += 1
        else:
            cid = int(test_cluster_ids[u])
            val = cluster_item_means[cid, i]
            source = 'cluster_mean'
            if cluster_avg_hard and (np.isnan(val) or cluster_item_counts[cid, i] == 0):
                if user_counts[u] > 0:
                    pred = float(user_means_arr[u])
                    source = 'user_mean'
                else:
                    pred = global_mean
                    source = 'global_mean'
            else:
                pred = float(val if not np.isnan(val) else global_mean)
                if np.isnan(val):
                    source = 'global_mean'
            pred = float(np.clip(pred, 1.0, 5.0))
            if cluster_avg_hard:
                src_counts[source] += 1
                if debug_cluster_avg_hard and debug_left > 0:
                    cluster_mean_dbg = float(val) if not np.isnan(val) else float('nan')
                    print(
                        f"  [{algo_label}] Tahmin kaynağı: source={source}, "
                        f"mu={pred:.4f}, cluster_mean={cluster_mean_dbg:.4f} "
                        f"(u={u}, i={i}, cid={cid})",
                        flush=True,
                    )
                    debug_left -= 1

        if use_weighted_cluster_avg or use_soft:
            pred = float(np.clip(pred, 1.0, 5.0))

        if gray_mask[u]:
            gray_true.append(r)
            gray_pred.append(pred)
        else:
            true_vals.append(r)
            pred_vals.append(pred)
        eval_rows.append((u, i, r, pred))

    all_true = true_vals + gray_true
    all_pred = pred_vals + gray_pred
    mae, rmse = _compute_metrics(all_true, all_pred)
    eval_rows_arr = np.array(eval_rows, dtype=np.float32)
    accuracy = _compute_binary_accuracy(
        all_true, all_pred, threshold=relevance_threshold,
        eval_rows=eval_rows_arr, train=train, assignments=assignments,
    )
    precision, recall, f1, ndcg = _compute_topn_metrics(
        eval_rows_arr,
        top_n=top_n,
        threshold=relevance_threshold,
        train=train, assignments=assignments,
    )
    jaccard = _compute_topn_jaccard(
        eval_rows_arr,
        top_n=top_n,
        threshold=relevance_threshold,
        train=train, assignments=assignments,
    )
    jaccard = _compute_topn_jaccard(
        eval_rows_arr,
        top_n=top_n,
        threshold=relevance_threshold,
        train=train, assignments=assignments,
    )
    jaccard = _compute_topn_jaccard(
        eval_rows_arr,
        top_n=top_n,
        threshold=relevance_threshold,
        train=train, assignments=assignments,
    )

    gray_mae, gray_rmse = float('nan'), float('nan')
    if gray_true:
        gray_errors = np.array(gray_true) - np.array(gray_pred)
        gray_mae  = float(np.mean(np.abs(gray_errors)))
        gray_rmse = float(np.sqrt(np.mean(gray_errors**2)))

    white_mae, white_rmse = _compute_metrics(true_vals, pred_vals)

    elapsed = time.time() - t0
    use_nc = (
        nearest_centroid and centroids is not None
        and centroids.shape[1] == n_items
    )
    if cluster_avg_leaky:
        scenario = 'calc_avg_rating_leaky_nc' if use_nc else 'calc_avg_rating_leaky'
        tag = 'CalcAvgRating+LEAK'
    elif cluster_avg_hard:
        scenario = 'calc_avg_rating_nc' if use_nc else 'calc_avg_rating'
        tag = 'CalcAvgRating'
    else:
        base_scenario = {
            'user': 'cluster_avg',
            'cluster': 'cluster_avg_cmean',
            'cluster_item_pop': 'cluster_avg_cmean_itempop',
        }.get(cluster_avg_base, 'cluster_avg')
        scenario = f'{base_scenario}_nc' if use_nc else base_scenario
        tag = 'ClusterAvg'
    leak_note = ' [train+test ort.]' if cluster_avg_leaky else ''
    print(
        f"  [{algo_label} | {tag}{'+NC' if use_nc else ''}{leak_note}] "
        f"MAE={mae:.4f} RMSE={rmse:.4f} | "
        f"P@10={precision:.4f} R@10={recall:.4f} NDCG@10={ndcg:.4f} J@10={jaccard:.4f} | "
        f"Gray MAE={gray_mae:.4f} ({elapsed:.1f}s)",
    )
    if cluster_avg_hard:
        print(
            f"  [{algo_label} | {tag}] kaynak dağılımı: "
            f"cluster_mean={src_counts['cluster_mean']} "
            f"item_mean={src_counts['item_mean']} "
            f"user_mean={src_counts['user_mean']} "
            f"global_mean={src_counts['global_mean']}",
            flush=True,
        )

    return {
        'scenario'    : scenario,
        'algo_label'  : algo_label,
        'mae'         : mae,
        'rmse'        : rmse,
        'gray_mae'    : gray_mae,
        'gray_rmse'   : gray_rmse,
        'white_mae'   : white_mae,
        'white_rmse'  : white_rmse,
        'n_clusters'  : n_clusters,
        'n_train'     : len(train),
        'n_test'      : len(test),
        'time_seconds': elapsed,
        'cluster_mae_std' : float('nan'),
        'cluster_mae_mean': float('nan'),
        'cluster_mae_min' : float('nan'),
        'cluster_mae_max' : float('nan'),
        'accuracy'        : accuracy,
        'precision_at_10' : precision,
        'recall_at_10'    : recall,
        'f1_at_10'        : f1,
        'ndcg_at_10'      : ndcg,
        'jaccard_at_10'   : jaccard,
    }


def compute_cluster_mu_shrinkage(
    train: np.ndarray,
    cluster_users: dict,
    mu_global: float,
    lambda_shrink: float = 25.0,
) -> dict:
    """
    Her küme için shrinkage ile ayarlanmış etkin ortalama.

    mu_eff[cid] = alpha * mu_cluster + (1 - alpha) * mu_global
    alpha       = n_cluster_ratings / (n_cluster_ratings + lambda_shrink)

    Küme büyüdükçe kendi ortalamasına (alpha→1),
    küme küçüldükçe global ortalamaya (alpha→0) yaklaşır.
    Bu sayede az veri olan kümeler global gürültüye çekilmez.
    """
    mu_eff: dict = {}
    for cid, users in cluster_users.items():
        user_set = set(int(u) for u in users)
        mask = np.isin(train[:, 0].astype(np.int64), list(user_set))
        cluster_ratings = train[mask, 2]
        n_c = len(cluster_ratings)
        mu_c = float(cluster_ratings.mean()) if n_c > 0 else mu_global
        alpha = n_c / (n_c + lambda_shrink)
        mu_eff[int(cid)] = alpha * mu_c + (1.0 - alpha) * mu_global
    return mu_eff


def build_item_popularity(train=None, user_ratings=None):
    """Train satırlarından veya user_ratings sözlüğünden film popülerliği."""
    item_popularity = {}
    if train is not None:
        for row in train:
            i = int(row[1])
            item_popularity[i] = item_popularity.get(i, 0) + 1
    elif user_ratings is not None:
        for items in user_ratings.values():
            for i in items:
                item_popularity[i] = item_popularity.get(i, 0) + 1
    return item_popularity


def iuf_weight(item_id: int, item_popularity: dict) -> float:
    n = item_popularity.get(item_id, 1)
    return 1.0 / np.log(1 + n)


def knn_user_similarity(
    ua: int,
    va: int,
    *,
    similarity: str,
    user_ratings: dict,
    user_means: dict,
    item_popularity: dict,
    min_common: int = 3,
    sig_threshold: int = 0,
    sim_amp: float = 1.0,
) -> float:
    """kNN kullanıcı–kullanıcı benzerliği: pearson | pearson_iuf | cosine | msd."""
    u_items = set(user_ratings.get(ua, {}).keys())
    v_items = set(user_ratings.get(va, {}).keys())
    common = list(u_items & v_items)
    if len(common) < min_common:
        return 0.0

    if similarity == 'cosine':
        u_r = np.array([user_ratings[ua][ix] for ix in common], dtype=np.float32)
        v_r = np.array([user_ratings[va][ix] for ix in common], dtype=np.float32)
        denom = np.sqrt((u_r ** 2).sum()) * np.sqrt((v_r ** 2).sum())
        if denom < 1e-8:
            return 0.0
        return float(np.clip(np.dot(u_r, v_r) / denom, 0.0, 1.0))

    if similarity == 'msd':
        return _msd_user_similarity(
            ua, va, user_ratings, min_common=min_common,
        )

    if similarity not in ('pearson', 'pearson_iuf'):
        raise ValueError(f"desteklenmeyen similarity: {similarity}")

    u_c = np.array(
        [user_ratings[ua][ix] - user_means[ua] for ix in common],
        dtype=np.float64,
    )
    v_c = np.array(
        [user_ratings[va][ix] - user_means[va] for ix in common],
        dtype=np.float64,
    )

    if similarity == 'pearson_iuf':
        weights = np.array(
            [iuf_weight(ix, item_popularity) for ix in common],
            dtype=np.float64,
        )
        num = float(np.sum(weights * u_c * v_c))
        norm_u = np.sqrt(float(np.sum(weights * u_c ** 2)))
        norm_v = np.sqrt(float(np.sum(weights * v_c ** 2)))
    else:
        num = float(np.dot(u_c, v_c))
        norm_u = np.sqrt(float(np.sum(u_c ** 2)))
        norm_v = np.sqrt(float(np.sum(v_c ** 2)))

    if norm_u < 1e-8 or norm_v < 1e-8:
        return 0.0

    sim = float(np.clip(num / (norm_u * norm_v), -1.0, 1.0))
    if sig_threshold > 0:
        sim = sim * min(len(common), sig_threshold) / float(sig_threshold)
    if sim_amp != 1.0:
        sim = float(np.sign(sim) * (abs(sim) ** sim_amp))
    return sim


def _msd_user_similarity(
    ua: int,
    va: int,
    user_ratings: dict,
    min_common: int = 1,
) -> float:
    """Surprise MSD benzerliği: 1 / (1 + ortalama kare fark)."""
    u_items = set(user_ratings.get(ua, {}).keys())
    v_items = set(user_ratings.get(va, {}).keys())
    common = list(u_items & v_items)
    if len(common) < min_common:
        return 0.0
    diffs = [
        (float(user_ratings[ua][ix]) - float(user_ratings[va][ix])) ** 2
        for ix in common
    ]
    return float(1.0 / (1.0 + np.mean(diffs)))


def _numpy_ratings_to_surprise_trainset(rows: np.ndarray, reader):
    from surprise import Dataset as SurpriseDataset

    df = pd.DataFrame({
        'uid': rows[:, 0].astype(int).astype(str),
        'iid': rows[:, 1].astype(int).astype(str),
        'rating': rows[:, 2].astype(float),
    })
    data = SurpriseDataset.load_from_df(df[['uid', 'iid', 'rating']], reader)
    return data.build_full_trainset()


def _fit_surprise_knn_algo(
    rows: np.ndarray,
    *,
    knn_variant: str = 'baseline',
    k_neighbors: int,
    similarity: str,
    min_common: int,
    reader=None,
):
    """Tek rating alt-kümesi: Surprise KNNBaseline / KNNWithMeans / CoClustering."""
    variant = (knn_variant or 'baseline').strip().lower()
    use_baseline = variant in ('baseline', 'knnbaseline', 'knn_baseline')
    use_coclustering = variant in ('coclustering', 'co_clustering', 'co-clustering')

    try:
        from surprise import Reader
        if use_coclustering:
            from surprise import CoClustering as AlgoCls
        elif use_baseline:
            from surprise import KNNBaseline as AlgoCls
        else:
            from surprise import KNNWithMeans as AlgoCls
    except ImportError:
        if use_baseline or use_coclustering:
            return None
        return _fit_local_knn_with_means(
            rows,
            k_neighbors=k_neighbors,
            similarity=similarity,
            min_common=min_common,
        )

    if reader is None:
        reader = Reader(rating_scale=(1, 5))
    if rows is None or len(rows) < 2:
        return None

    trainset = _numpy_ratings_to_surprise_trainset(rows, reader)
    if use_coclustering:
        n_u = len(np.unique(rows[:, 0].astype(np.int64)))
        n_i = len(np.unique(rows[:, 1].astype(np.int64)))
        algo = AlgoCls(
            n_cltr_u=max(2, min(20, int(np.sqrt(max(n_u, 1))))),
            n_cltr_i=max(2, min(20, int(np.sqrt(max(n_i, 1))))),
            n_epochs=20,
            random_state=42,
            verbose=False,
        )
    else:
        algo = AlgoCls(
            k=max(1, int(k_neighbors)),
            sim_options=_surprise_sim_options(
                similarity, min_common, baseline=use_baseline,
            ),
            verbose=False,
        )
    algo.fit(trainset)
    return algo


def _fit_surprise_knn_with_means(
    rows: np.ndarray,
    *,
    k_neighbors: int,
    similarity: str,
    min_common: int,
    reader=None,
):
    """Geriye dönük: KNNWithMeans."""
    return _fit_surprise_knn_algo(
        rows,
        knn_variant='withmeans',
        k_neighbors=k_neighbors,
        similarity=similarity,
        min_common=min_common,
        reader=reader,
    )


class _LocalKNNWithMeansModel:
    """Surprise yokken küme-içi user-based KNNWithMeans (mean-centered CF)."""

    def __init__(
        self,
        rows: np.ndarray,
        *,
        k_neighbors: int,
        similarity: str,
        min_common: int,
    ):
        self.k_neighbors = max(1, int(k_neighbors))
        self.similarity = similarity
        self.min_common = int(min_common)
        self.user_ratings: dict = {}
        for row in rows:
            u, i, r = int(row[0]), int(row[1]), float(row[2])
            self.user_ratings.setdefault(u, {})[i] = r
        self.global_mean = float(rows[:, 2].mean()) if len(rows) else 3.0
        self.user_means = {
            u: float(np.mean(list(d.values())))
            for u, d in self.user_ratings.items()
        }
        self.item_popularity = build_item_popularity(user_ratings=self.user_ratings)
        self._users = set(self.user_ratings.keys())

    def predict(self, uid: str, iid: str):
        u, i = int(uid), int(iid)
        fallback = float(self.user_means.get(u, self.global_mean))
        sims: List[Tuple[float, int]] = []
        for v in self._users:
            if v == u or i not in self.user_ratings.get(v, {}):
                continue
            if self.similarity == 'msd':
                s = _msd_user_similarity(
                    u, v, self.user_ratings, min_common=self.min_common,
                )
            else:
                s = knn_user_similarity(
                    u, v,
                    similarity=self.similarity,
                    user_ratings=self.user_ratings,
                    user_means=self.user_means,
                    item_popularity=self.item_popularity,
                    min_common=self.min_common,
                )
            if abs(s) > 0.0:
                sims.append((s, v))
        est = _knn_predict_from_sims(
            u, i, sims, self.user_ratings, self.user_means,
            self.global_mean, self.k_neighbors,
        )
        return type('Prediction', (), {'est': est})()


def _fit_local_knn_with_means(
    rows: np.ndarray,
    *,
    k_neighbors: int,
    similarity: str,
    min_common: int,
):
    if rows is None or len(rows) < 2:
        return None
    return _LocalKNNWithMeansModel(
        rows,
        k_neighbors=k_neighbors,
        similarity=similarity,
        min_common=min_common,
    )


def _resolve_cluster_surprise_sim(similarity: str, knn_variant: str) -> str:
    """Cluster Surprise yolu: baseline modunda pearson -> pearson_baseline."""
    s = (similarity or 'pearson').strip().lower()
    variant = (knn_variant or 'baseline').strip().lower()
    if variant in ('coclustering', 'co_clustering', 'co-clustering'):
        return s
    if variant in ('baseline', 'knnbaseline', 'knn_baseline'):
        if s in ('pearson', 'pearson_iuf', 'pearson_baseline'):
            return 'pearson_baseline'
    return s


def _build_cluster_surprise_knn_models(
    train: np.ndarray,
    cluster_users: dict,
    *,
    knn_variant: str = 'baseline',
    k_neighbors: int,
    similarity: str,
    min_common: int,
    allowed_users: Optional[Set[int]] = None,
):
    """
    Küme başına Surprise KNN (Baseline veya WithMeans) + isteğe bağlı global model.
    Returns (cluster_models, global_model, reader) veya (None, None, None).
    """
    variant = (knn_variant or 'baseline').strip().lower()
    use_baseline = variant in ('baseline', 'knnbaseline', 'knn_baseline')
    label = 'KNNBaseline' if use_baseline else 'KNNWithMeans'
    surprise_sim = _resolve_cluster_surprise_sim(similarity, variant)

    reader = None
    try:
        from surprise import Reader
        reader = Reader(rating_scale=(1, 5))
    except ImportError:
        if use_baseline:
            print(
                f"  [Cluster{label}] HATA: 'surprise' yok (pip install scikit-surprise)",
                file=sys.stderr,
            )
            return None, None, None
        print(
            f"  [Cluster{label}] uyarı: 'surprise' yok; yerel KNNWithMeans fallback.",
            flush=True,
        )

    if (similarity or '').strip().lower() == 'pearson_iuf':
        print(
            f"  [Cluster{label}] uyarı: pearson_iuf -> "
            f"{'pearson_baseline' if use_baseline else 'pearson'}",
            flush=True,
        )

    cluster_models: dict = {}
    for cid, users in cluster_users.items():
        user_set = set(users)
        if not user_set:
            continue
        mask = np.isin(train[:, 0].astype(np.int64), list(user_set))
        algo = _fit_surprise_knn_algo(
            train[mask],
            knn_variant=variant,
            k_neighbors=k_neighbors,
            similarity=surprise_sim,
            min_common=min_common,
            reader=reader,
        )
        if algo is not None:
            cluster_models[int(cid)] = algo

    if allowed_users is not None:
        mask = np.isin(train[:, 0].astype(np.int64), list(allowed_users))
        global_rows = train[mask]
    else:
        global_rows = train
    global_model = _fit_surprise_knn_algo(
        global_rows,
        knn_variant=variant,
        k_neighbors=k_neighbors,
        similarity=surprise_sim,
        min_common=min_common,
        reader=reader,
    )

    if not cluster_models and global_model is None:
        return None, None, None
    return cluster_models, global_model, reader


def _build_cluster_knn_with_means_models(
    train: np.ndarray,
    cluster_users: dict,
    *,
    k_neighbors: int,
    similarity: str,
    min_common: int,
    allowed_users: Optional[Set[int]] = None,
):
    """Geriye dönük: küme başına KNNWithMeans."""
    return _build_cluster_surprise_knn_models(
        train,
        cluster_users,
        knn_variant='withmeans',
        k_neighbors=k_neighbors,
        similarity=similarity,
        min_common=min_common,
        allowed_users=allowed_users,
    )


def _predict_surprise_kwm(
    algo,
    u: int,
    i: int,
    fallback: float,
) -> float:
    """Surprise KNNWithMeans tek (u,i) tahmini."""
    if algo is None:
        return float(np.clip(fallback, 1.0, 5.0))
    try:
        est = float(algo.predict(str(int(u)), str(int(i))).est)
    except Exception:
        est = fallback
    return float(np.clip(est, 1.0, 5.0))


def _predict_knn(
    u: int,
    i: int,
    cid: int,
    user_ratings: dict,
    cluster_users: dict,
    user_means: dict,
    similarity: str = 'pearson',
    k_neighbors: int = 30,
    min_common: int = 3,
    global_mean: float = 3.0,
    item_popularity: Optional[dict] = None,
    kwm_models: Optional[dict] = None,
) -> float:
    """Küme içi CF tahmini — kwm_models verilirse Surprise KNNWithMeans; yoksa manuel kNN."""
    fallback = float(user_means.get(u, global_mean))
    if kwm_models is not None:
        return _predict_surprise_kwm(
            kwm_models.get(int(cid)), u, i, fallback,
        )

    k_neighbors = max(1, int(k_neighbors))
    if item_popularity is None:
        item_popularity = build_item_popularity(user_ratings=user_ratings)

    neighbors = cluster_users.get(cid, [])
    sims = []
    for v in neighbors:
        if v == u:
            continue
        if i not in user_ratings.get(v, {}):
            continue

        s = knn_user_similarity(
            u, v,
            similarity=similarity,
            user_ratings=user_ratings,
            user_means=user_means,
            item_popularity=item_popularity,
            min_common=min_common,
        )

        if abs(s) > 0.0:
            sims.append((s, v))

    if not sims:
        return float(user_means.get(u, global_mean))

    sims.sort(key=lambda x: -abs(x[0]))
    top_k = sims[:k_neighbors]

    num = sum(s * (user_ratings[v][i] - user_means.get(v, global_mean))
              for s, v in top_k)
    den = sum(abs(s) for s, v in top_k)

    if den < 1e-8:
        return float(user_means.get(u, global_mean))

    pred = user_means.get(u, global_mean) + num / den
    return float(np.clip(pred, 1.0, 5.0))


def _predict_full_soft_knn(
    u: int,
    i: int,
    memberships: np.ndarray,
    candidate_users: Sequence[int],
    user_ratings: dict,
    user_means: dict,
    global_mean: float,
    similarity: str = 'pearson',
    min_common: int = 3,
    k_neighbors: int = 30,
    sig_weight: int = 0,
    sim_amp: float = 1.0,
) -> float:
    """
    Tam soft komşuluk: aday kullanıcılar arasında rating benzerliği × üyelik uyumu.
    combined_sim = p_sim * (1 + dot(memberships[u], memberships[v]))
    """
    k_neighbors = max(1, int(k_neighbors))
    mu = np.asarray(memberships[u], dtype=np.float64)
    item_popularity = build_item_popularity(user_ratings=user_ratings)

    sims: List[Tuple[float, int]] = []
    for v in candidate_users:
        if v == u:
            continue
        if i not in user_ratings.get(v, {}):
            continue
        p_sim = knn_user_similarity(
            u, v,
            similarity=similarity,
            user_ratings=user_ratings,
            user_means=user_means,
            item_popularity=item_popularity,
            min_common=min_common,
            sig_threshold=sig_weight,
            sim_amp=sim_amp,
        )
        if abs(p_sim) < 1e-8:
            continue
        m_sim = float(np.dot(mu, np.asarray(memberships[v], dtype=np.float64)))
        combined = p_sim * (1.0 + m_sim)
        sims.append((combined, v))

    base_u = float(user_means.get(u, global_mean))
    if not sims:
        return float(np.clip(base_u, 1.0, 5.0))

    sims.sort(key=lambda x: -abs(x[0]))
    top_k = sims[:k_neighbors]
    num = sum(
        s * (user_ratings[v][i] - user_means.get(v, global_mean))
        for s, v in top_k
    )
    den = sum(abs(s) for s, _ in top_k)
    if den < 1e-8:
        return float(np.clip(base_u, 1.0, 5.0))
    return float(np.clip(base_u + num / den, 1.0, 5.0))


def run_cluster_knn(train, test, assignments, gray_mask, memberships,
                    n_items, algo_label,
                    user_features=None,
                    use_latent_similarity: bool = False,
                    similarity: str = 'pearson',
                    min_common: int = 3,
                    k_neighbors: int = 30,
                    expand_knn: bool = False,
                    sig_weight: int = 0,
                    sim_amp: float = 1.0,
                    knn_mode: str = 'cluster',
                    surprise_knn_variant: str = 'baseline',
                    cluster_knn_backend: str = 'surprise',
                    top_n: int = 10,
                    relevance_threshold: float = 4.0,
                    centroids=None,
                    nearest_centroid: bool = False,
                    centroid_metric: str = 'euclidean',
                    assign_dir: Optional[str] = None,
                    cluster_weight_alpha: Union[float, str] = 1.0,
                    cluster_weight_base: float = 1.0,
                    return_eval_rows: bool = False):
    t0 = time.time()
    k_neighbors = max(1, int(k_neighbors))

    test_cluster_ids = resolve_test_cluster_ids(
        train, assignments, centroids, nearest_centroid, n_items,
        centroid_metric=centroid_metric, algo_label=algo_label,
        assign_dir=assign_dir,
    )

    global_mean = float(train[:, 2].mean())
    user_ratings = {}
    for row in train:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        if u not in user_ratings:
            user_ratings[u] = {}
        user_ratings[u][i] = r
    user_means = {
        u: float(np.mean(list(d.values())))
        for u, d in user_ratings.items()
    }
    item_popularity = build_item_popularity(user_ratings=user_ratings)

    gray_mask = np.asarray(gray_mask, dtype=bool)
    n_users = len(assignments)
    white_set = {u for u in range(n_users) if not gray_mask[u]}

    # White-only küme üyelikleri (komşu havuzu)
    cluster_users: dict = {}
    for u in white_set:
        cid = int(assignments[u])
        cluster_users.setdefault(cid, []).append(u)

    mu_eff = compute_cluster_mu_shrinkage(
        train, cluster_users, global_mean, lambda_shrink=25.0,
    )
    print(
        f"  Cluster mu_eff: min={min(mu_eff.values()):.3f} "
        f"max={max(mu_eff.values()):.3f} "
        f"global={global_mean:.3f}"
    )

    # Gray sheep → global KNN (yalnızca white komşular)
    global_sim_index = _build_knn_sim_index(
        user_ratings,
        user_means,
        item_popularity,
        similarity=similarity,
        min_common=min_common,
        sig_weight=sig_weight,
        sim_amp=sim_amp,
        allowed_neighbors=white_set,
    )
    n_gray = int(gray_mask.sum())
    if n_gray > 0:
        print(
            f"  [{algo_label}] gray/white split: {n_gray} gray -> global KNN, "
            f"{len(white_set)} white -> cluster KNN (white neighbors only)",
            flush=True,
        )

    def _pair_sim(u, v, mc=None):
        return knn_user_similarity(
            u, v,
            similarity=similarity,
            user_ratings=user_ratings,
            user_means=user_means,
            item_popularity=item_popularity,
            min_common=mc if mc is not None else min_common,
            sig_threshold=sig_weight,
            sim_amp=sim_amp,
        )

    knn_mode_norm = (knn_mode or 'cluster').strip().lower()
    use_weighted_cluster = knn_mode_norm == 'weighted_cluster'
    alpha_auto = (
        isinstance(cluster_weight_alpha, str)
        and str(cluster_weight_alpha).strip().lower() == 'auto'
    )
    cluster_weight_base = float(
        cluster_weight_base if cluster_weight_base is not None else 1.0
    )
    cluster_weight_alpha_fixed = 1.0
    if not alpha_auto:
        cluster_weight_alpha_fixed = float(
            cluster_weight_alpha if cluster_weight_alpha is not None else 1.0
        )
    cluster_alphas: dict = {}
    use_full_soft = knn_mode_norm == 'full_soft'
    if use_full_soft:
        if memberships is None:
            print(
                f"  [{algo_label}] uyarı: --knn-mode full_soft için memberships.npy yok; "
                f"cluster moduna düşülüyor.",
                flush=True,
            )
            use_full_soft = False
        elif memberships.shape[0] < len(assignments):
            print(
                f"  [{algo_label}] uyarı: memberships satır sayısı yetersiz; "
                f"cluster moduna düşülüyor.",
                flush=True,
            )
            use_full_soft = False

    if use_weighted_cluster:
        if centroids is None and assign_dir:
            n_clusters = int(assignments.max()) + 1
            centroids = load_centroids(assign_dir, n_clusters)
        if centroids is None:
            print(
                f"  [{algo_label}] uyarı: --knn-mode weighted_cluster için "
                f"best_sol.npy yok; cluster moduna düşülüyor.",
                flush=True,
            )
            use_weighted_cluster = False
        else:
            n_clusters = int(assignments.max()) + 1
            if alpha_auto:
                for cid in range(n_clusters):
                    cluster_alphas[cid] = compute_cluster_weight_alpha(
                        cid, cluster_users, n_users, base=cluster_weight_base,
                    )
                alpha_summary = ", ".join(
                    f"c{cid}={cluster_alphas[cid]:.4g}"
                    for cid in sorted(cluster_alphas)
                )
                print(
                    f"  [{algo_label}] weighted_cluster: alpha=auto "
                    f"(base={cluster_weight_base:g}), "
                    f"centroids shape={np.asarray(centroids).shape}",
                    flush=True,
                )
                print(
                    f"  [{algo_label}] per-cluster alpha: {alpha_summary}",
                    flush=True,
                )
            else:
                print(
                    f"  [{algo_label}] weighted_cluster: alpha="
                    f"{cluster_weight_alpha_fixed:g}, "
                    f"centroids shape={np.asarray(centroids).shape}",
                    flush=True,
                )

    cent_dist = None
    if use_weighted_cluster:
        centroids_arr = np.asarray(centroids, dtype=np.float64)
        K_cent = centroids_arr.shape[0]
        cent_dist = np.zeros((K_cent, K_cent), dtype=np.float64)
        for a in range(K_cent):
            cent_dist[a] = np.linalg.norm(centroids_arr - centroids_arr[a], axis=1)

    use_soft = (
        not use_full_soft
        and not use_weighted_cluster
        and memberships is not None
        and memberships.shape[0] >= len(assignments)
        and memberships.shape[1] >= (int(assignments.max()) + 1)
    )

    backend_norm = (cluster_knn_backend or 'surprise').strip().lower()
    use_native_backend = backend_norm in (
        'native', 'cluster_predictor', 'baseline_native',
    )
    use_manual_backend = backend_norm in ('manual', 'legacy', 'pearson')
    if use_native_backend and (
        use_weighted_cluster or use_full_soft or expand_knn
    ):
        print(
            f"  [{algo_label}] uyarı: --cluster-knn-backend native, "
            f"weighted_cluster/full_soft/expand-knn ile uyumsuz; manual kullanılıyor.",
            flush=True,
        )
        use_native_backend = False
        use_manual_backend = True
    use_legacy_knn = (
        use_manual_backend
        or bool(expand_knn)
        or use_weighted_cluster
        or use_full_soft
    )
    surprise_variant = (surprise_knn_variant or 'baseline').strip().lower()
    use_baseline_surprise = surprise_variant in (
        'baseline', 'knnbaseline', 'knn_baseline',
    )
    use_coclustering_surprise = surprise_variant in (
        'coclustering', 'co_clustering', 'co-clustering',
    )
    if use_coclustering_surprise:
        surprise_label = 'CoClustering'
    else:
        surprise_label = 'KNNBaseline' if use_baseline_surprise else 'KNNWithMeans'
    cluster_kwm_models = None
    global_kwm_model = None
    if use_latent_similarity and user_features is not None:
        from sklearn.preprocessing import normalize

        U = np.asarray(user_features, dtype=np.float64)
        if U.shape[0] < n_users:
            print(
                f"  [{algo_label}] uyarı: user_features satır sayısı ({U.shape[0]}) "
                f"kullanıcı sayısından ({n_users}) az; latent kNN atlanıyor.",
                flush=True,
            )
        else:
            U_norm = normalize(U, norm='l2')

            latent_sim_index = {}
            for u in range(n_users):
                cid = int(assignments[u])
                neighbors = cluster_users.get(cid, [])
                sims = []
                for v in neighbors:
                    if v == u:
                        continue
                    sim = float(np.dot(U_norm[u], U_norm[v]))
                    if sim > 0:
                        sims.append((sim, v))
                sims.sort(key=lambda x: -x[0])
                latent_sim_index[u] = sims

            print(
                f"  [{algo_label}] tahmin: latent uzay kNN "
                f"(k={k_neighbors}, küme-içi cosine)",
                flush=True,
            )

            true_vals, pred_vals = [], []
            gray_true, gray_pred = [], []
            eval_rows = []
            for row in test:
                u, i, r = int(row[0]), int(row[1]), float(row[2])
                neighbors = [
                    (s, v) for s, v in latent_sim_index.get(u, [])
                    if i in user_ratings.get(v, {})
                ][:k_neighbors]

                base = float(user_means.get(u, global_mean))
                if not neighbors:
                    pred = base
                else:
                    num = sum(
                        s * (user_ratings[v][i] - user_means.get(v, global_mean))
                        for s, v in neighbors
                    )
                    den = sum(abs(s) for s, _ in neighbors)
                    pred = float(np.clip(
                        base + (num / den if den > 1e-8 else 0),
                        1.0, 5.0,
                    ))

                if gray_mask[u]:
                    gray_true.append(r)
                    gray_pred.append(pred)
                else:
                    true_vals.append(r)
                    pred_vals.append(pred)
                eval_rows.append((u, i, r, pred))

            all_true = true_vals + gray_true
            all_pred = pred_vals + gray_pred
            mae, rmse = _compute_metrics(all_true, all_pred)
            eval_rows_arr = np.array(eval_rows, dtype=np.float32)
            accuracy = _compute_binary_accuracy(
                all_true, all_pred, threshold=relevance_threshold,
                eval_rows=eval_rows_arr, train=train, assignments=assignments,
            )

            gray_mae, gray_rmse = float('nan'), float('nan')
            if gray_true:
                gray_errors = np.array(gray_true) - np.array(gray_pred)
                gray_mae = float(np.mean(np.abs(gray_errors)))
                gray_rmse = float(np.sqrt(np.mean(gray_errors ** 2)))

            white_mae, white_rmse = _compute_metrics(true_vals, pred_vals)
            precision, recall, f1, ndcg = _compute_topn_metrics(
                eval_rows_arr,
                top_n=top_n,
                threshold=relevance_threshold,
                train=train, assignments=assignments,
            )
            jaccard = _compute_topn_jaccard(
                eval_rows_arr,
                top_n=top_n,
                threshold=relevance_threshold,
                train=train, assignments=assignments,
            )

            elapsed = time.time() - t0
            _knn_tag = 'ClusterKNN-latent'
            _scenario = 'cluster_knn_latent'
            _knn_mode_out = 'cluster_latent'
            if n_gray > 0:
                _knn_tag += '+GSsplit'
                _scenario = 'cluster_knn_latent_gs_split'
            print(
                f"  [{algo_label} | {_knn_tag}|latent|k={k_neighbors}] "
                f"MAE={mae:.4f} RMSE={rmse:.4f} | "
                f"P@10={precision:.4f} R@10={recall:.4f} NDCG@10={ndcg:.4f} J@10={jaccard:.4f} | "
                f"Gray MAE={gray_mae:.4f} ({elapsed:.1f}s)",
            )

            out = {
                'scenario': _scenario,
                'algo_label': algo_label,
                'mae': mae,
                'rmse': rmse,
                'gray_mae': gray_mae,
                'gray_rmse': gray_rmse,
                'white_mae': white_mae,
                'white_rmse': white_rmse,
                'n_clusters': int(assignments.max()) + 1,
                'n_train': len(train),
                'n_test': len(test),
                'time_seconds': elapsed,
                'cluster_mae_std': float('nan'),
                'cluster_mae_mean': float('nan'),
                'cluster_mae_min': float('nan'),
                'cluster_mae_max': float('nan'),
                'accuracy': accuracy,
                'precision_at_10': precision,
                'recall_at_10': recall,
                'f1_at_10': f1,
                'ndcg_at_10': ndcg,
                'jaccard_at_10': jaccard,
                'similarity': 'latent',
                'k_neighbors': k_neighbors,
                'knn_mode': _knn_mode_out,
                'cluster_weight_alpha': float('nan'),
                'cluster_weight_alpha_mode': '',
                'cluster_weight_base': float('nan'),
            }
            if return_eval_rows:
                out['eval_rows'] = eval_rows_arr
            return out
    elif use_latent_similarity and user_features is None:
        print(
            f"  [{algo_label}] uyarı: --use-latent-similarity açık ama "
            f"user_features.npy yok; standart kNN kullanılıyor.",
            flush=True,
        )

    if use_native_backend:
        from cluster_predictor import (
            ClusterPredictor,
            build_rating_matrix,
            map_wnmf_similarity,
        )

        native_sim = map_wnmf_similarity(similarity)
        R_train = build_rating_matrix(train, n_users, n_items)
        native_pred = ClusterPredictor(
            k_neighbors=k_neighbors,
            sim_metric=native_sim,
            min_support=min_common,
        )
        native_pred.fit(R_train, assignments)
        print(
            f"  [{algo_label}] tahmin: native küme kNN "
            f"(sim={native_sim}, k={k_neighbors}, min_support={min_common})",
            flush=True,
        )

        def _predict_global_knn_native(u, i):
            neighbors = [
                (s, v) for s, v in global_sim_index.get(u, [])
                if i in user_ratings.get(v, {})
            ]
            return _knn_predict_from_sims(
                u, i, neighbors, user_ratings, user_means, global_mean, k_neighbors,
            )

        true_vals, pred_vals = [], []
        gray_true, gray_pred = [], []
        eval_rows = []
        for row in test:
            u, i, r = int(row[0]), int(row[1]), float(row[2])
            if gray_mask[u]:
                pr = _predict_global_knn_native(u, i)
            else:
                pr = native_pred.predict(u, i)

            if gray_mask[u]:
                gray_true.append(r)
                gray_pred.append(pr)
            else:
                true_vals.append(r)
                pred_vals.append(pr)
            eval_rows.append((u, i, r, pr))

        all_true = true_vals + gray_true
        all_pred = pred_vals + gray_pred
        mae, rmse = _compute_metrics(all_true, all_pred)
        eval_rows_arr = np.array(eval_rows, dtype=np.float32)
        accuracy = _compute_binary_accuracy(
            all_true, all_pred, threshold=relevance_threshold,
            eval_rows=eval_rows_arr, train=train, assignments=assignments,
        )

        gray_mae, gray_rmse = float('nan'), float('nan')
        if gray_true:
            gray_errors = np.array(gray_true) - np.array(gray_pred)
            gray_mae = float(np.mean(np.abs(gray_errors)))
            gray_rmse = float(np.sqrt(np.mean(gray_errors ** 2)))

        white_mae, white_rmse = _compute_metrics(true_vals, pred_vals)
        precision, recall, f1, ndcg = _compute_topn_metrics(
            eval_rows_arr,
            top_n=top_n,
            threshold=relevance_threshold,
            train=train, assignments=assignments,
        )
        jaccard = _compute_topn_jaccard(
            eval_rows_arr,
            top_n=top_n,
            threshold=relevance_threshold,
            train=train, assignments=assignments,
        )

        elapsed = time.time() - t0
        _knn_tag = 'ClusterKNN-native'
        _scenario = 'cluster_knn_native_baseline'
        _knn_mode_out = 'cluster_native'
        if n_gray > 0:
            _knn_tag += '+GSsplit'
            _scenario = 'cluster_knn_native_baseline_gs_split'
        print(
            f"  [{algo_label} | {_knn_tag}|{native_sim}|k={k_neighbors}] "
            f"MAE={mae:.4f} RMSE={rmse:.4f} | "
            f"P@10={precision:.4f} R@10={recall:.4f} NDCG@10={ndcg:.4f} J@10={jaccard:.4f} | "
            f"Gray MAE={gray_mae:.4f} ({elapsed:.1f}s)",
        )

        out = {
            'scenario': _scenario,
            'algo_label': algo_label,
            'mae': mae,
            'rmse': rmse,
            'gray_mae': gray_mae,
            'gray_rmse': gray_rmse,
            'white_mae': white_mae,
            'white_rmse': white_rmse,
            'n_clusters': int(assignments.max()) + 1,
            'n_train': len(train),
            'n_test': len(test),
            'time_seconds': elapsed,
            'cluster_mae_std': float('nan'),
            'cluster_mae_mean': float('nan'),
            'cluster_mae_min': float('nan'),
            'cluster_mae_max': float('nan'),
            'accuracy': accuracy,
            'precision_at_10': precision,
            'recall_at_10': recall,
            'f1_at_10': f1,
            'ndcg_at_10': ndcg,
            'jaccard_at_10': jaccard,
            'similarity': native_sim,
            'k_neighbors': k_neighbors,
            'knn_mode': _knn_mode_out,
            'cluster_weight_alpha': float('nan'),
            'cluster_weight_alpha_mode': '',
            'cluster_weight_base': float('nan'),
        }
        if return_eval_rows:
            out['eval_rows'] = eval_rows_arr
        return out

    if not use_legacy_knn:
        cluster_kwm_models, global_kwm_model, _ = _build_cluster_surprise_knn_models(
            train,
            cluster_users,
            knn_variant=surprise_variant,
            k_neighbors=k_neighbors,
            similarity=similarity,
            min_common=min_common,
            allowed_users=white_set if n_gray > 0 else None,
        )
        if cluster_kwm_models is None and global_kwm_model is None:
            print(
                f"  [{algo_label}] uyarı: {surprise_label} kullanılamadı; "
                f"manuel kNN'e düşülüyor.",
                flush=True,
            )
            use_legacy_knn = True
        else:
            sim_out = _resolve_cluster_surprise_sim(similarity, surprise_variant)
            print(
                f"  [{algo_label}] tahmin: Surprise {surprise_label} "
                f"(sim={sim_out}, {len(cluster_kwm_models or {})} küme"
                f"{', gray->global' if global_kwm_model is not None and n_gray > 0 else ''})",
                flush=True,
            )
    elif use_manual_backend:
        expand_note = ', expand-knn açık' if expand_knn else ', küme-içi komşular'
        print(
            f"  [{algo_label}] tahmin: manuel {similarity} kNN "
            f"(k={k_neighbors}{expand_note})",
            flush=True,
        )

    knn_candidates = [
        v for v in range(n_users)
        if not gray_mask[v]
    ]

    def _predict_for_cluster(u, i, cid, similarity='pearson', min_common=3):
        neighbors = cluster_users.get(cid, [])
        sims = []
        for v in neighbors:
            if v == u or gray_mask[v]:
                continue
            if i not in user_ratings.get(v, {}):
                continue
            s = _pair_sim(u, v)
            if abs(s) > 0:
                sims.append((s, v))

        if len(sims) < k_neighbors and expand_knn:
            for other_cid, other_users in cluster_users.items():
                if other_cid == cid:
                    continue
                for v in other_users:
                    if v == u or gray_mask[v]:
                        continue
                    if i not in user_ratings.get(v, {}):
                        continue
                    s = _pair_sim(u, v)
                    if abs(s) > 0:
                        sims.append((s, v))

        return _knn_predict_from_sims(
            u, i, sims, user_ratings, user_means, global_mean, k_neighbors,
        )

    def _predict_global_knn(u, i):
        neighbors = [
            (s, v) for s, v in global_sim_index.get(u, [])
            if i in user_ratings.get(v, {})
        ]
        return _knn_predict_from_sims(
            u, i, neighbors, user_ratings, user_means, global_mean, k_neighbors,
        )

    def _alpha_for_cluster(cid: int) -> float:
        if alpha_auto:
            return float(cluster_alphas.get(cid, cluster_weight_base))
        return cluster_weight_alpha_fixed

    def _membership_weight(cid: int, v: int) -> float:
        v_cid = int(assignments[v])
        if v_cid == cid:
            return 1.0
        alpha = _alpha_for_cluster(cid)
        if alpha <= 0.0:
            return 1.0
        return float(1.0 / (1.0 + alpha * cent_dist[cid, v_cid]))

    def _predict_weighted_cluster(u, i, cid):
        sims = []
        for v in knn_candidates:
            if v == u:
                continue
            if i not in user_ratings.get(v, {}):
                continue
            p_sim = _pair_sim(u, v)
            if p_sim <= 0.0:
                continue
            eff = p_sim * _membership_weight(cid, v)
            if eff > 0.0:
                sims.append((eff, v))
        return _knn_predict_from_sims(
            u, i, sims, user_ratings, user_means, global_mean, k_neighbors,
        )

    def predict(u, i, similarity='pearson', min_common=3):
        if not use_legacy_knn:
            cid_u = int(assignments[u]) if u < len(assignments) else -1
            cluster_mu = mu_eff.get(cid_u, global_mean)
            fallback = float(user_means.get(u, cluster_mu))
            if gray_mask[u]:
                return _predict_surprise_kwm(global_kwm_model, u, i, fallback)

            if use_soft:
                weights = np.asarray(memberships[u], dtype=np.float64)
                active = [
                    (cid, float(w))
                    for cid, w in enumerate(weights)
                    if w >= SOFT_MEMBERSHIP_THRESHOLD
                ]
                if not active:
                    cid = int(test_cluster_ids[u])
                    return _predict_surprise_kwm(
                        (cluster_kwm_models or {}).get(cid), u, i, fallback,
                    )
                pred = 0.0
                total_w = 0.0
                for cid, w in active:
                    p = _predict_surprise_kwm(
                        (cluster_kwm_models or {}).get(cid), u, i, fallback,
                    )
                    pred += w * p
                    total_w += w
                return float(np.clip(pred / max(total_w, 1e-8), 1.0, 5.0))

            cid = int(test_cluster_ids[u])
            return _predict_surprise_kwm(
                (cluster_kwm_models or {}).get(cid), u, i, fallback,
            )

        if gray_mask[u]:
            return _predict_global_knn(u, i)

        if use_weighted_cluster:
            cid = int(test_cluster_ids[u])
            return _predict_weighted_cluster(u, i, cid)

        if use_full_soft:
            return _predict_full_soft_knn(
                u,
                i,
                memberships,
                knn_candidates,
                user_ratings,
                user_means,
                global_mean,
                similarity=similarity,
                min_common=min_common,
                k_neighbors=k_neighbors,
                sig_weight=sig_weight,
                sim_amp=sim_amp,
            )

        if not use_soft:
            cid = int(test_cluster_ids[u])
            return _predict_for_cluster(
                u, i, cid, similarity=similarity, min_common=min_common,
            )

        weights = np.asarray(memberships[u], dtype=np.float64)
        active = [
            (cid, float(w))
            for cid, w in enumerate(weights)
            if w >= SOFT_MEMBERSHIP_THRESHOLD
        ]

        if not active:
            cid = int(np.argmax(weights))
            return _predict_for_cluster(
                u, i, cid, similarity=similarity, min_common=min_common,
            )

        pred = 0.0
        total_w = 0.0
        for cid, w in active:
            p = _predict_for_cluster(
                u, i, cid, similarity=similarity, min_common=min_common,
            )
            pred += w * p
            total_w += w

        return float(np.clip(pred / total_w, 1.0, 5.0))

    true_vals, pred_vals = [], []
    gray_true, gray_pred = [], []
    eval_rows = []

    for row in test:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        pr = predict(u, i, similarity=similarity, min_common=min_common)

        if gray_mask[u]:
            gray_true.append(r)
            gray_pred.append(pr)
        else:
            true_vals.append(r)
            pred_vals.append(pr)
        eval_rows.append((u, i, r, pr))

    all_true = true_vals + gray_true
    all_pred = pred_vals + gray_pred
    mae, rmse = _compute_metrics(all_true, all_pred)
    eval_rows_arr = np.array(eval_rows, dtype=np.float32)
    accuracy = _compute_binary_accuracy(
        all_true, all_pred, threshold=relevance_threshold,
        eval_rows=eval_rows_arr, train=train, assignments=assignments,
    )

    gray_mae, gray_rmse = float('nan'), float('nan')
    if gray_true:
        gray_errors = np.array(gray_true) - np.array(gray_pred)
        gray_mae  = float(np.mean(np.abs(gray_errors)))
        gray_rmse = float(np.sqrt(np.mean(gray_errors**2)))

    white_mae, white_rmse = _compute_metrics(true_vals, pred_vals)
    precision, recall, f1, ndcg = _compute_topn_metrics(
        eval_rows_arr,
        top_n=top_n,
        threshold=relevance_threshold,
        train=train, assignments=assignments,
    )
    jaccard = _compute_topn_jaccard(
        eval_rows_arr,
        top_n=top_n,
        threshold=relevance_threshold,
        train=train, assignments=assignments,
    )

    elapsed = time.time() - t0
    if not use_legacy_knn:
        if use_coclustering_surprise:
            _knn_tag = 'ClusterCoClustering'
            _scenario = 'cluster_coclustering'
            _knn_mode_out = 'cluster_coclustering'
        elif use_baseline_surprise:
            _knn_tag = 'ClusterKNNBaseline'
            _scenario = 'cluster_knn_baseline'
            _knn_mode_out = 'cluster_knb'
        else:
            _knn_tag = 'ClusterKNNWithMeans'
            _scenario = 'cluster_knn_with_means'
            _knn_mode_out = 'cluster_kwm'
    elif use_weighted_cluster:
        if alpha_auto:
            _knn_tag = f'ClusterKNN-wcluster-auto-b{cluster_weight_base:g}'
        else:
            _knn_tag = f'ClusterKNN-wcluster-a{cluster_weight_alpha_fixed:g}'
        _scenario = 'cluster_knn_weighted'
        _knn_mode_out = 'weighted_cluster'
    elif use_full_soft:
        _knn_tag = 'ClusterKNN-fullsoft'
        _scenario = 'cluster_knn_full_soft'
        _knn_mode_out = 'full_soft'
    else:
        if use_manual_backend:
            _knn_tag = 'ClusterKNN-manual'
            _knn_mode_out = 'cluster_manual'
        else:
            _knn_tag = 'ClusterKNN'
            _knn_mode_out = 'cluster'
        _scenario = 'cluster_knn'
    if n_gray > 0 and not use_full_soft:
        _knn_tag += '+GSsplit'
        if _scenario == 'cluster_knn_baseline':
            _scenario = 'cluster_knn_baseline_gs_split'
        elif _scenario == 'cluster_knn_with_means':
            _scenario = 'cluster_knn_with_means_gs_split'
        elif _scenario == 'cluster_knn':
            _scenario = 'cluster_knn_gs_split'
        elif _scenario == 'cluster_knn_weighted':
            _scenario = 'cluster_knn_weighted_gs_split'
    if (
        nearest_centroid and centroids is not None and not use_full_soft
        and centroids.shape[1] == n_items
    ):
        _knn_tag += '+NC'
        _scenario = 'cluster_knn_nc'
    print(
        f"  [{algo_label} | {_knn_tag}|{similarity}|k={k_neighbors}] "
        f"MAE={mae:.4f} RMSE={rmse:.4f} | "
        f"P@10={precision:.4f} R@10={recall:.4f} NDCG@10={ndcg:.4f} J@10={jaccard:.4f} | "
        f"Gray MAE={gray_mae:.4f} ({elapsed:.1f}s)",
    )

    out = {
        'scenario'    : _scenario,
        'algo_label'  : algo_label,
        'mae'         : mae,
        'rmse'        : rmse,
        'gray_mae'    : gray_mae,
        'gray_rmse'   : gray_rmse,
        'white_mae'   : white_mae,
        'white_rmse'  : white_rmse,
        'n_clusters'  : int(assignments.max()) + 1,
        'n_train'     : len(train),
        'n_test'      : len(test),
        'time_seconds': elapsed,
        'cluster_mae_std' : float('nan'),
        'cluster_mae_mean': float('nan'),
        'cluster_mae_min' : float('nan'),
        'cluster_mae_max' : float('nan'),
        'accuracy'        : accuracy,
        'precision_at_10' : precision,
        'recall_at_10'    : recall,
        'f1_at_10'        : f1,
        'ndcg_at_10'      : ndcg,
        'jaccard_at_10'   : jaccard,
        'similarity'      : similarity,
        'k_neighbors'     : k_neighbors,
        'knn_mode'        : _knn_mode_out,
        'cluster_weight_alpha': (
            float('nan') if (use_weighted_cluster and alpha_auto)
            else (cluster_weight_alpha_fixed if use_weighted_cluster else float('nan'))
        ),
        'cluster_weight_alpha_mode': (
            'auto' if (use_weighted_cluster and alpha_auto)
            else ('fixed' if use_weighted_cluster else '')
        ),
        'cluster_weight_base': (
            cluster_weight_base if (use_weighted_cluster and alpha_auto) else float('nan')
        ),
    }
    if return_eval_rows:
        out['eval_rows'] = eval_rows_arr
    return out


# ============================================================
# ITEM-CF ALTYAPISI (Görev 3) + WEIGHTED FUSION (Görev 4-5)
# ============================================================

def build_item_ratings(train):
    """Train satırlarından (item_ratings, item_means) sözlüklerini kur.

    item_ratings : {item_id: {user_id: rating}}
    item_means   : {item_id: ortalama_puan}
    """
    item_ratings: dict = {}
    for row in train:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        if i not in item_ratings:
            item_ratings[i] = {}
        item_ratings[i][u] = r
    item_means = {
        i: float(np.mean(list(d.values())))
        for i, d in item_ratings.items()
    }
    return item_ratings, item_means


def build_user_activity(train=None, user_ratings=None):
    """{user_id: kaç film oyladı} — IUF-user için aktivite sayacı."""
    ua: dict = {}
    if train is not None:
        for row in train:
            u = int(row[0])
            ua[u] = ua.get(u, 0) + 1
    elif user_ratings is not None:
        for u, items in user_ratings.items():
            ua[int(u)] = len(items)
    return ua


def iuf_weight_user(user_id: int, user_activity: dict) -> float:
    """Aşırı aktif kullanıcıyı cezalandır: w(u) = 1/log(1+n_u)."""
    n = user_activity.get(user_id, 1)
    return 1.0 / np.log(1.0 + n)


def pearson_sim_items(
    i: int,
    j: int,
    item_ratings: dict,
    item_means: dict,
    user_activity: dict,
    min_common: int = 3,
    cache: Optional[dict] = None,
) -> float:
    """Film i–j arası IUF-ağırlıklı Pearson benzerliği.

    cache: {(min(i,j), max(i,j)): sim} — verilirse simetrik tekrar hesabı yok.
    """
    if cache is not None:
        key = (i, j) if i <= j else (j, i)
        if key in cache:
            return cache[key]

    ri = item_ratings.get(i)
    rj = item_ratings.get(j)
    if not ri or not rj:
        result = 0.0
    else:
        common_users = list(set(ri.keys()) & set(rj.keys()))
        if len(common_users) < min_common:
            result = 0.0
        else:
            weights = np.array(
                [iuf_weight_user(u, user_activity) for u in common_users],
                dtype=np.float64,
            )
            mi = item_means.get(i, 0.0)
            mj = item_means.get(j, 0.0)
            i_c = np.array([ri[u] - mi for u in common_users], dtype=np.float64)
            j_c = np.array([rj[u] - mj for u in common_users], dtype=np.float64)

            num    = float(np.sum(weights * i_c * j_c))
            norm_i = float(np.sqrt(np.sum(weights * i_c ** 2)))
            norm_j = float(np.sqrt(np.sum(weights * j_c ** 2)))

            if norm_i < 1e-8 or norm_j < 1e-8:
                result = 0.0
            else:
                result = float(np.clip(num / (norm_i * norm_j), -1.0, 1.0))

    if cache is not None:
        cache[(i, j) if i <= j else (j, i)] = result
    return result


def _predict_item_knn(
    u: int,
    i: int,
    item_cid: int,
    item_ratings: dict,
    cluster_items: dict,
    item_means: dict,
    user_activity: dict,
    k_neighbors: int = 20,
    min_common: int = 3,
    global_mean: float = 3.0,
    sim_cache: Optional[dict] = None,
) -> float:
    """Film i'nin kümesindeki benzer filmlerden kullanıcı u için tahmin.

    r̂_item(u,i) = μ_i + Σⱼ sim(i,j) × (r_uj - μ_j) / Σⱼ|sim(i,j)|
    """
    k_neighbors = max(1, int(k_neighbors))
    base_i = float(item_means.get(i, global_mean))

    candidate_items = [
        j for j in cluster_items.get(item_cid, [])
        if j != i and u in item_ratings.get(j, {})
    ]
    if not candidate_items:
        return float(np.clip(base_i, 1.0, 5.0))

    sims = []
    for j in candidate_items:
        s = pearson_sim_items(
            i, j, item_ratings, item_means, user_activity,
            min_common=min_common, cache=sim_cache,
        )
        if abs(s) > 1e-8:
            sims.append((s, j))

    if not sims:
        return float(np.clip(base_i, 1.0, 5.0))

    sims.sort(key=lambda x: -abs(x[0]))
    top_k = sims[:k_neighbors]

    num = sum(
        s * (item_ratings[j].get(u, item_means.get(j, global_mean))
             - item_means.get(j, global_mean))
        for s, j in top_k
    )
    den = sum(abs(s) for s, _ in top_k)
    if den < 1e-8:
        return float(np.clip(base_i, 1.0, 5.0))

    return float(np.clip(base_i + num / den, 1.0, 5.0))


def _cosine_dense(a: np.ndarray, b: np.ndarray) -> float:
    """İki dense vektör için cosine — sıfır vektör güvenli."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    sim = float(np.dot(a, b) / (norm_a * norm_b))
    return float(np.clip(sim, -1.0, 1.0))


def compute_cluster_centroids_rawspace(
    R_matrix: np.ndarray,
    assignments: np.ndarray,
    K: int,
    axis: str = 'user',
) -> np.ndarray:
    """Ham rating uzayında küme ortalamaları (centroid) hesabı.

    axis='user': R_matrix (n_users × n_items), her kullanıcı kümesi için
                 rated hücrelerin ortalamasını al → (K × n_items)
    axis='item': R_matrix.T (n_items × n_users) ile aynı mantık → (K × n_users)

    Sadece rated (>0) hücreler ortalamada sayılır. Hiç rating olmayan sütunlar 0.
    Feature uzayında değil rating uzayında centroid üretir; fusion için kullanılır.
    """
    if axis == 'item':
        R = R_matrix.T
    elif axis == 'user':
        R = R_matrix
    else:
        raise ValueError(f"axis bilinmiyor: {axis!r}")

    n_rows, n_cols = R.shape
    assignments = np.asarray(assignments, dtype=np.int64)
    if len(assignments) != n_rows:
        raise ValueError(
            f"assignments uzunluğu ({len(assignments)}) "
            f"matris satır sayısıyla ({n_rows}) eşleşmiyor (axis={axis})."
        )

    centroids = np.zeros((K, n_cols), dtype=np.float32)
    counts    = np.zeros((K, n_cols), dtype=np.int32)
    for row in range(n_rows):
        cid = int(assignments[row])
        if cid < 0 or cid >= K:
            continue
        vec = R[row]
        mask = vec > 0
        centroids[cid, mask] += vec[mask]
        counts[cid, mask]    += 1

    nonzero = counts > 0
    centroids[nonzero] /= counts[nonzero]
    return centroids


def compute_fusion_weights(
    u: int,
    i: int,
    u_vec: np.ndarray,
    i_vec: np.ndarray,
    user_centroids: np.ndarray,
    item_centroids: np.ndarray,
    user_assignments: np.ndarray,
    item_assignments: np.ndarray,
    mode: str = 'dynamic',
    alpha: float = 0.5,
) -> Tuple[float, float]:
    """User-CF / Item-CF dinamik ağırlıkları.

    mode='fixed' : (1-alpha, alpha)
    mode='dynamic':
        m = cos(u_vec, user_centroids[cu])
        n = cos(i_vec, item_centroids[ci])
        Negatif/sıfır benzerlikleri 0'a kırparak (m, n) → m/(m+n+eps), n/(m+n+eps)
    """
    if mode == 'fixed':
        w_user = float(np.clip(1.0 - alpha, 0.0, 1.0))
        w_item = float(np.clip(alpha,        0.0, 1.0))
        s = w_user + w_item
        if s < 1e-10:
            return 0.5, 0.5
        return w_user / s, w_item / s

    if mode != 'dynamic':
        raise ValueError(f"fusion_mode bilinmiyor: {mode!r} (fixed | dynamic)")

    cu = int(user_assignments[u])
    ci = int(item_assignments[i])
    if cu < 0 or cu >= user_centroids.shape[0] \
       or ci < 0 or ci >= item_centroids.shape[0]:
        return 0.5, 0.5

    m = max(0.0, _cosine_dense(u_vec, user_centroids[cu]))
    n_w = max(0.0, _cosine_dense(i_vec, item_centroids[ci]))
    total = m + n_w
    if total < 1e-10:
        return 0.5, 0.5
    return m / total, n_w / total


def _build_rating_matrix_from_train(
    train: np.ndarray, n_users: int, n_items: int,
) -> np.ndarray:
    """Train rating'lerinden dense (n_users × n_items) matris üret."""
    R = np.zeros((n_users, n_items), dtype=np.float32)
    for row in train:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        if 0 <= u < n_users and 0 <= i < n_items:
            R[u, i] = r
    return R


def compute_coverage(user_recommendations: dict, n_items: int) -> float:
    """Tüm kullanıcılara önerilen benzersiz film oranı."""
    if not user_recommendations or n_items <= 0:
        return float('nan')
    all_recs = set()
    for recs in user_recommendations.values():
        all_recs.update(int(x) for x in recs)
    return len(all_recs) / float(n_items)


def compute_diversity(
    user_recommendations: dict,
    sim_fn,
    max_users: Optional[int] = None,
) -> float:
    """Önerilen filmler arası 1 - sim ortalaması.

    sim_fn(i, j) -> float in [-1, 1]
    max_users: hızlandırma için kullanıcı alt-örneği (varsayılan: hepsi).
    """
    if not user_recommendations:
        return float('nan')

    users = list(user_recommendations.keys())
    if max_users is not None and max_users > 0 and len(users) > max_users:
        rng = np.random.default_rng(seed=42)
        users = list(rng.choice(users, size=max_users, replace=False))

    divs = []
    for u in users:
        recs = list(user_recommendations[u])
        if len(recs) < 2:
            continue
        pair_divs = []
        for a in range(len(recs)):
            for b in range(a + 1, len(recs)):
                s = float(sim_fn(int(recs[a]), int(recs[b])))
                pair_divs.append(1.0 - s)
        if pair_divs:
            divs.append(float(np.mean(pair_divs)))
    return float(np.mean(divs)) if divs else float('nan')


def _compute_topn_metrics_with_recs(
    eval_rows: np.ndarray,
    top_n: int = 10,
    threshold: RelevanceThreshold = 4.0,
    train: Optional[np.ndarray] = None,
    assignments: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, dict]:
    """_compute_topn_metrics'in genişletilmiş versiyonu: per-user top-N rec listesi de döner.

    Geri uyumluluk: _compute_topn_metrics değişmedi; bu yeni helper yalnız
    coverage/diversity hesabı için çağrılır.
    """
    if eval_rows is None or len(eval_rows) == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), {}

    by_user: dict = {}
    for row in eval_rows:
        u = int(row[0]); i = int(row[1])
        r_true = float(row[2]); r_pred = float(row[3])
        by_user.setdefault(u, []).append((i, r_true, r_pred))

    users = set(by_user.keys())
    th = _resolve_relevance_threshold(threshold, train, assignments, users)

    precisions, recalls, f1s, ndcgs = [], [], [], []
    user_recommendations: dict = {}
    k = max(1, int(top_n))
    for u, items in by_user.items():
        ranked = sorted(items, key=lambda x: x[2], reverse=True)[:k]
        top_items = [int(i) for i, _, _ in ranked]
        user_recommendations[u] = top_items

        t_u = _user_relevance_threshold(th, u)
        relevant = {int(i) for i, r_true, _ in items if r_true >= t_u}
        if not relevant:
            continue
        hits = len(set(top_items) & relevant)
        p = hits / k
        r = hits / len(relevant) if relevant else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        dcg = 0.0
        for rank_idx, item_id in enumerate(top_items):
            rel = 1.0 if item_id in relevant else 0.0
            dcg += (2.0 ** rel - 1.0) / np.log2(rank_idx + 2.0)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2.0) for i in range(ideal_hits))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        ndcgs.append(ndcg)

    if not precisions:
        return float('nan'), float('nan'), float('nan'), float('nan'), user_recommendations
    return (
        float(np.mean(precisions)),
        float(np.mean(recalls)),
        float(np.mean(f1s)),
        float(np.mean(ndcgs)),
        user_recommendations,
    )


def run_cluster_knn_fusion(
    train,
    test,
    user_assignments: np.ndarray,
    item_assignments: np.ndarray,
    R_matrix_train: np.ndarray,
    gray_mask: np.ndarray,
    n_items: int,
    n_users: int,
    algo_label_user: str,
    algo_label_item: str,
    similarity: str = 'pearson_iuf',
    min_common: int = 3,
    k_user: int = 20,
    k_item: int = 20,
    fusion_mode: str = 'dynamic',
    fusion_alpha: float = 0.5,
    top_n: int = 10,
    relevance_threshold: float = 4.0,
    compute_cov: bool = False,
    compute_div: bool = False,
):
    """Weighted Fusion: User-CF + Item-CF tahminlerini centroid-bazlı veya sabit
    ağırlıkla birleştirir.

    r̂(u,i) = m × pred_user + n × pred_item
    """
    t0 = time.time()
    k_user = max(1, int(k_user))
    k_item = max(1, int(k_item))

    global_mean = float(train[:, 2].mean())
    user_ratings: dict = {}
    for row in train:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        if u not in user_ratings:
            user_ratings[u] = {}
        user_ratings[u][i] = r
    user_means = {
        u: float(np.mean(list(d.values()))) for u, d in user_ratings.items()
    }
    item_popularity = build_item_popularity(user_ratings=user_ratings)
    item_ratings, item_means = build_item_ratings(train)
    user_activity = build_user_activity(train=train)

    cluster_users: dict = {}
    for u in range(len(user_assignments)):
        cid = int(user_assignments[u])
        cluster_users.setdefault(cid, []).append(u)

    cluster_items: dict = {}
    for i in range(len(item_assignments)):
        cid = int(item_assignments[i])
        if cid < 0:
            continue  # pruned item — kümeye dahil etme
        cluster_items.setdefault(cid, []).append(i)

    K_user = int(user_assignments.max()) + 1
    K_item = int(item_assignments.max()) + 1
    user_centroids = compute_cluster_centroids_rawspace(
        R_matrix_train, user_assignments, K_user, axis='user',
    )
    item_centroids = compute_cluster_centroids_rawspace(
        R_matrix_train, item_assignments, K_item, axis='item',
    )

    test_users = sorted({int(row[0]) for row in test})
    test_items = sorted({int(row[1]) for row in test})
    u_vec_cache: dict = {}
    for u in test_users:
        v = np.zeros(n_items, dtype=np.float32)
        for j, r in user_ratings.get(u, {}).items():
            if 0 <= j < n_items:
                v[j] = r
        u_vec_cache[u] = v
    i_vec_cache: dict = {}
    for i in test_items:
        v = np.zeros(n_users, dtype=np.float32)
        for w, r in item_ratings.get(i, {}).items():
            if 0 <= w < n_users:
                v[w] = r
        i_vec_cache[i] = v

    item_sim_cache: dict = {}

    true_vals, pred_vals = [], []
    gray_true, gray_pred = [], []
    eval_rows = []
    fusion_weights_log: List[Tuple[float, float]] = []

    for row in test:
        u, i, r = int(row[0]), int(row[1]), float(row[2])

        if u < len(user_assignments):
            cid_user = int(user_assignments[u])
            pred_u = _predict_knn(
                u, i, cid_user,
                user_ratings, cluster_users, user_means,
                similarity=similarity,
                k_neighbors=k_user,
                min_common=min_common,
                global_mean=global_mean,
                item_popularity=item_popularity,
            )
        else:
            pred_u = float(np.clip(global_mean, 1.0, 5.0))

        if i < len(item_assignments) and int(item_assignments[i]) >= 0:
            cid_item = int(item_assignments[i])
            pred_i = _predict_item_knn(
                u, i, cid_item,
                item_ratings, cluster_items, item_means,
                user_activity,
                k_neighbors=k_item,
                min_common=min_common,
                global_mean=global_mean,
                sim_cache=item_sim_cache,
            )
        else:
            # Item, pruning sırasında çıkarıldı veya atanmadı — item-ortalama ile fallback
            pred_i = float(np.clip(item_means.get(i, global_mean), 1.0, 5.0))

        _item_assigned = i < len(item_assignments) and int(item_assignments[i]) >= 0
        if u in u_vec_cache and i in i_vec_cache \
           and u < len(user_assignments) and _item_assigned:
            m, n_w = compute_fusion_weights(
                u, i,
                u_vec_cache[u], i_vec_cache[i],
                user_centroids, item_centroids,
                user_assignments, item_assignments,
                mode=fusion_mode, alpha=fusion_alpha,
            )
        else:
            m, n_w = 0.5, 0.5

        fusion_weights_log.append((m, n_w))
        pred = float(np.clip(m * pred_u + n_w * pred_i, 1.0, 5.0))

        if u < len(gray_mask) and gray_mask[u]:
            gray_true.append(r)
            gray_pred.append(pred)
        else:
            true_vals.append(r)
            pred_vals.append(pred)
        eval_rows.append((u, i, r, pred))

    all_true = true_vals + gray_true
    all_pred = pred_vals + gray_pred
    mae, rmse = _compute_metrics(all_true, all_pred)
    eval_rows_arr = np.array(eval_rows, dtype=np.float32)
    accuracy = _compute_binary_accuracy(
        all_true, all_pred, threshold=relevance_threshold,
        eval_rows=eval_rows_arr, train=train, assignments=user_assignments,
    )

    gray_mae, gray_rmse = float('nan'), float('nan')
    if gray_true:
        gray_errors = np.array(gray_true) - np.array(gray_pred)
        gray_mae = float(np.mean(np.abs(gray_errors)))
        gray_rmse = float(np.sqrt(np.mean(gray_errors ** 2)))

    white_mae, white_rmse = _compute_metrics(true_vals, pred_vals)

    if compute_cov or compute_div:
        precision, recall, f1, ndcg, user_recs = _compute_topn_metrics_with_recs(
            eval_rows_arr, top_n=top_n, threshold=relevance_threshold,
            train=train, assignments=user_assignments,
        )
    else:
        precision, recall, f1, ndcg = _compute_topn_metrics(
            eval_rows_arr, top_n=top_n, threshold=relevance_threshold,
            train=train, assignments=user_assignments,
        )
        user_recs = {}
    jaccard = _compute_topn_jaccard(
        eval_rows_arr, top_n=top_n, threshold=relevance_threshold,
        train=train, assignments=user_assignments,
    )

    coverage_val = float('nan')
    diversity_val = float('nan')
    if compute_cov and user_recs:
        coverage_val = compute_coverage(user_recs, n_items)
    if compute_div and user_recs:
        def _div_sim(a: int, b: int) -> float:
            return pearson_sim_items(
                a, b, item_ratings, item_means, user_activity,
                min_common=min_common, cache=item_sim_cache,
            )
        diversity_val = compute_diversity(user_recs, _div_sim, max_users=500)

    mean_m = float(np.mean([w[0] for w in fusion_weights_log])) if fusion_weights_log else float('nan')
    mean_n = float(np.mean([w[1] for w in fusion_weights_log])) if fusion_weights_log else float('nan')

    elapsed = time.time() - t0
    scenario_name = f'cluster_knn_fusion_{fusion_mode}'
    print(
        f"  [{algo_label_user}+{algo_label_item} | Fusion-{fusion_mode}|"
        f"{similarity}|ku={k_user},ki={k_item}] "
        f"MAE={mae:.4f} RMSE={rmse:.4f} | mean(m,n)=({mean_m:.3f},{mean_n:.3f}) "
        f"({elapsed:.1f}s)"
    )

    return {
        'scenario'        : scenario_name,
        'algo_label'      : algo_label_user,
        'algo_label_item' : algo_label_item,
        'mae'             : mae,
        'rmse'            : rmse,
        'gray_mae'        : gray_mae,
        'gray_rmse'       : gray_rmse,
        'white_mae'       : white_mae,
        'white_rmse'      : white_rmse,
        'n_clusters'      : K_user,
        'n_clusters_item' : K_item,
        'n_train'         : len(train),
        'n_test'          : len(test),
        'time_seconds'    : elapsed,
        'cluster_mae_std' : float('nan'),
        'cluster_mae_mean': float('nan'),
        'cluster_mae_min' : float('nan'),
        'cluster_mae_max' : float('nan'),
        'accuracy'        : accuracy,
        'precision_at_10' : precision,
        'recall_at_10'    : recall,
        'f1_at_10'        : f1,
        'ndcg_at_10'      : ndcg,
        'jaccard_at_10'   : jaccard,
        'similarity'      : similarity,
        'k_neighbors'     : k_user,
        'k_neighbors_item': k_item,
        'knn_mode'        : f'fusion_{fusion_mode}',
        'fusion_mode'     : fusion_mode,
        'fusion_alpha'    : fusion_alpha if fusion_mode == 'fixed' else float('nan'),
        'fusion_mean_m'   : mean_m,
        'fusion_mean_n'   : mean_n,
        'coverage'        : coverage_val,
        'diversity'       : diversity_val,
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


def _compute_metrics(true_list, pred_list):
    """Toplanmış tahminlerden MAE ve RMSE hesapla."""
    if not true_list:
        return float('nan'), float('nan')
    true_arr = np.array(true_list, dtype=np.float32)
    pred_arr = np.array(pred_list, dtype=np.float32)
    errors   = true_arr - pred_arr
    return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors ** 2)))


def compute_wcss(user_matrix: np.ndarray, assignments: np.ndarray) -> float:
    """
    Kullanıcı vektörleri üzerinde klasik WCSS (within-cluster sum of squares).
    Geçersiz küme id'leri (<0) hesaplamaya katılmaz.
    """
    if user_matrix is None or assignments is None:
        return float('nan')
    if len(assignments) == 0 or user_matrix.shape[0] == 0:
        return float('nan')

    n = min(int(user_matrix.shape[0]), int(len(assignments)))
    X = np.asarray(user_matrix[:n], dtype=np.float64)
    a = np.asarray(assignments[:n], dtype=np.int64)
    valid = a >= 0
    if not np.any(valid):
        return float('nan')

    Xv = X[valid]
    av = a[valid]
    wcss = 0.0
    for cid in np.unique(av):
        m = av == cid
        Xi = Xv[m]
        if Xi.shape[0] <= 1:
            continue
        centroid = Xi.mean(axis=0, dtype=np.float64)
        diff = Xi - centroid
        wcss += float(np.einsum('ij,ij->', diff, diff))
    return float(wcss)


def _print_assignment_diagnostics(
    algo_label: str,
    assignments: np.ndarray,
    user_features: Optional[np.ndarray],
) -> None:
    """Assignment kalite/dağılım tanı logları."""
    a = np.asarray(assignments, dtype=np.int64)
    valid = a[a >= 0]
    if valid.size == 0:
        print(f"  [{algo_label}] Assignment diag: geçerli küme etiketi yok.", flush=True)
        return

    sizes = np.bincount(valid)
    print(
        f"  [{algo_label}] Cluster size: min={int(sizes.min())} "
        f"max={int(sizes.max())} std={float(sizes.std()):.1f}",
        flush=True,
    )

    if user_features is None:
        print(
            f"  [{algo_label}] WCSS: user_features yok (hesap atlandı).",
            flush=True,
        )
        return

    wcss = compute_wcss(np.asarray(user_features), a)
    if np.isnan(wcss):
        print(f"  [{algo_label}] WCSS: hesaplanamadi.", flush=True)
    else:
        print(f"  [{algo_label}] WCSS={wcss:.2f}", flush=True)


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
        knn_spec,
        min_common,
        hybrid_alpha,
        out_dir,
        run_command,
        ld, lr, reg, eg, ec,
        mp_args,
    ) = job

    # Çocuk süreçte hiperparametre globallerini geçerli değerlere ayarla
    global LATENT_DIM, LEARNING_RATE, REGULARIZATION, N_EPOCHS_GLOBAL, N_EPOCHS_CLUSTER
    LATENT_DIM       = ld
    LEARNING_RATE    = lr
    REGULARIZATION   = reg
    N_EPOCHS_GLOBAL  = eg
    N_EPOCHS_CLUSTER = ec

    expand_knn = bool(getattr(mp_args, 'expand_knn', False))
    knn_mode = str(getattr(mp_args, 'knn_mode', 'cluster') or 'cluster')
    use_latent_similarity = bool(getattr(mp_args, 'use_latent_similarity', False))
    surprise_knn_variant = str(
        getattr(mp_args, 'cluster_knn_variant', 'baseline') or 'baseline'
    )
    cluster_knn_backend = str(
        getattr(mp_args, 'cluster_knn_backend', 'surprise') or 'surprise'
    )

    if isinstance(knn_spec, int):
        _knv = [knn_spec]
    else:
        _knv = list(knn_spec)

    try:
        n_users_expected = int(max(train[:, 0].max(), test[:, 0].max())) + 1
        assignments, gray_mask = load_assignment(assign_dir)
        memberships = load_memberships(assign_dir)
        user_features = load_user_features(assign_dir, len(assignments))
        assignments, gray_mask, memberships, user_features = _align_assignment_bundle(
            assignments,
            gray_mask,
            memberships,
            user_features,
            n_users_expected=n_users_expected,
            algo_label=label,
            assign_dir=assign_dir,
        )
        _print_assignment_diagnostics(label, assignments, user_features)
        rows: List[dict] = []

        if run_cluster_avg_flag:
            nc = _nearest_centroid_bundle(mp_args, assign_dir, assignments)
            row = run_cluster_average(
                train, test, assignments, gray_mask, memberships, n_items, label,
                top_n=top_n,
                relevance_threshold=relevance_threshold,
                cluster_avg_hard=bool(getattr(mp_args, 'cluster_avg_hard', False)),
                cluster_avg_leaky=bool(getattr(mp_args, 'cluster_avg_leaky', False)),
                cluster_mean_impute_train=bool(
                    getattr(mp_args, 'cluster_mean_impute_train', False)
                ),
                debug_cluster_avg_hard=bool(
                    getattr(mp_args, 'debug_cluster_avg_hard', False)
                ),
                debug_cluster_avg_hard_limit=int(
                    getattr(mp_args, 'debug_cluster_avg_hard_limit', 5) or 5
                ),
                assign_dir=assign_dir,
                **_cluster_avg_predict_kwargs(mp_args),
                **nc,
            )
            row['dataset'] = dataset_name
            rows.append(row)

        if do_cluster_knn_flag:
            nc = _knn_centroid_bundle(mp_args, assign_dir, assignments, knn_mode=knn_mode)
            for kv in _knv:
                row = run_cluster_knn(
                    train, test, assignments, gray_mask, memberships, n_items, label,
                    user_features=user_features,
                    use_latent_similarity=use_latent_similarity,
                    similarity=similarity,
                    min_common=min_common,
                    k_neighbors=int(kv),
                    expand_knn=expand_knn,
                    knn_mode=knn_mode,
                    surprise_knn_variant=surprise_knn_variant,
                    cluster_knn_backend=cluster_knn_backend,
                    sig_weight=int(getattr(mp_args, 'sig_weight', 0) or 0),
                    sim_amp=float(getattr(mp_args, 'sim_amp', 1.0) or 1.0),
                    top_n=top_n,
                    relevance_threshold=relevance_threshold,
                    assign_dir=assign_dir,
                    **nc,
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
                hybrid_alpha=hybrid_alpha,
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


def _align_assignment_bundle(
    assignments: np.ndarray,
    gray_mask: np.ndarray,
    memberships: Optional[np.ndarray],
    user_features: Optional[np.ndarray],
    *,
    n_users_expected: int,
    algo_label: str,
    assign_dir: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Prune ile üretilmiş assignment'lar kullanıcı sayısını düşürebilir.
    Eval tarafında user-id uzayı tam (0..N-1) beklendiği için dizileri güvenli şekilde
    hizala: eksik kullanıcıları gray/fallback yap, fazla satırları kırp.
    """
    cur_n = int(len(assignments))
    exp_n = int(n_users_expected)
    if cur_n == exp_n:
        return assignments, gray_mask, memberships, user_features

    print(
        f"  [{algo_label}] Not: assignment user sayısı ({cur_n}) "
        f"eval kullanıcı sayısından ({exp_n}) farklı; "
        f"eksik kullanıcılar gray/fallback ile hizalanıyor. ({assign_dir})",
        flush=True,
    )

    if cur_n < exp_n:
        pad = exp_n - cur_n
        assignments = np.pad(assignments, (0, pad), mode='constant', constant_values=-1)
        gray_mask = np.pad(gray_mask, (0, pad), mode='constant', constant_values=True)
        if memberships is not None:
            if memberships.ndim == 1:
                memberships = memberships[:, None]
            memberships = np.pad(
                memberships, ((0, pad), (0, 0)),
                mode='constant', constant_values=0.0
            )
        if user_features is not None:
            if user_features.ndim == 1:
                user_features = user_features[:, None]
            user_features = np.pad(
                user_features, ((0, pad), (0, 0)),
                mode='constant', constant_values=0.0
            )
        return assignments, gray_mask, memberships, user_features

    # cur_n > exp_n
    assignments = assignments[:exp_n]
    gray_mask = gray_mask[:exp_n]
    if memberships is not None:
        memberships = memberships[:exp_n]
    if user_features is not None:
        user_features = user_features[:exp_n]
    return assignments, gray_mask, memberships, user_features



def _save_row_to_db(dataset_name: str, row: dict, k_used: int, args,
                    fold_override: Optional[int] = None):
    if not _DB_AVAILABLE:
        return
    db_dataset = _normalize_db_dataset_name(dataset_name)
    prep_raw = getattr(args, 'assign_suffix', '')
    prep = _normalize_db_preprocessing(prep_raw)
    scenario = row['scenario']
    assignment_id_override = None
    if scenario != 'global' and prep_raw:
        db_dataset = _normalize_db_dataset_name(dataset_name)
        try:
            assignment_id_override = get_assignment_id_by_suffix(
                db_dataset, row['algo_label'], k_used, prep_raw,
            )
        except Exception:
            assignment_id_override = None
    if assignment_id_override is None and scenario != 'global':
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
        assign_suffix=prep_raw,
        knn=row.get('k_neighbors'),
        similarity=row.get('similarity'),
        fold=row.get('cv_fold', fold_override),
    )


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
                knn: Union[int, Sequence[int]] = 30,
                min_common: int = 3,
                expand_knn: bool = False,
                knn_mode: str = 'cluster',
                hybrid_alpha: Optional[float] = None,
                use_svdpp: bool = False,
                run_command: Optional[str] = None,
                args=None,
                fold: Optional[int] = None,
                do_fusion: bool = False,
                fusion_mode: str = 'dynamic',
                fusion_alpha: float = 0.5,
                item_assign_root: Optional[str] = None,
                item_assign_suffix: str = '',
                assignment_k_item: Optional[int] = None,
                assignment_k_item_100k: Optional[int] = None,
                assignment_k_item_1m: Optional[int] = None,
                knn_item: Union[int, Sequence[int]] = 20,
                compute_cov: bool = False,
                compute_div: bool = False):
    """
    Bir dataset üzerinde tüm senaryoları çalıştır.

    mode parametresi:
        'baselines' — küme ortalaması + küme içi kNN CF (+ isteğe bağlı global WNMF);
                      WNMF küme modeli (full/sharedV) çalışmaz. Tüm --assign-suffix ile uyumlu.
                      --no-cluster-avg / --no-cluster-knn ile parça parça kapatılabilir.
        'sharedV'   — baselines (varsayılan açık) + SharedV WNMF (önerilen)
        'full'      — baselines + her küme ayrı U+V WNMF
        'all'       — baselines + full + sharedV

    Assignment klasörü: mealpy/results/assignments_lof/{dataset}/{label}{assign_suffix}
    veya (yalnızca eski / soneksiz düzen) K ≠ varsayılan ve suffix'te _k{K} yoksa
    {label}_k{K}{assign_suffix}.
    """
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name.upper()}  |  mode={mode}")
    print(f"{'='*60}")

    # Fusion ayarlarını args'tan miras al (çağıranlar değişmesin diye)
    if args is not None:
        if not do_fusion:
            do_fusion = bool(getattr(args, 'fusion', False))
            fusion_mode = str(getattr(args, 'fusion_mode', fusion_mode) or fusion_mode)
            fusion_alpha = float(getattr(args, 'fusion_alpha', fusion_alpha))
            if item_assign_root is None:
                item_assign_root = getattr(args, 'item_assign_root', None)
            if not item_assign_suffix:
                item_assign_suffix = getattr(args, 'item_assign_suffix', '') or ''
            if assignment_k_item is None:
                assignment_k_item = getattr(args, 'assignment_k_item', None)
            if assignment_k_item_100k is None:
                assignment_k_item_100k = getattr(args, 'assignment_k_item_100k', None)
            if assignment_k_item_1m is None:
                assignment_k_item_1m = getattr(args, 'assignment_k_item_1m', None)
            knn_item_arg = getattr(args, 'knn_item', None)
            if knn_item_arg:
                knn_item = knn_item_arg
            if not compute_cov:
                compute_cov = bool(getattr(args, 'coverage', False))
            if not compute_div:
                compute_div = bool(getattr(args, 'diversity', False))

    root = assign_root if assign_root is not None else ASSIGN_ROOT
    dk   = ASSIGN_K_DEFAULT_100K if dataset_name == 'ml100k' else ASSIGN_K_DEFAULT_1M
    k_ds = assignment_k_100k if dataset_name == 'ml100k' else assignment_k_1m
    k_used = _resolved_assignment_k(k_ds, assignment_k, dk)
    suffix_k = _assign_suffix_trailing_cluster_k(assign_suffix)
    if suffix_k is not None and suffix_k != k_used:
        print(
            f"  Uyarı: --k {k_used} ile --assign-suffix sondaki _k{suffix_k} farklı; "
            f"assignment klasörü _k{suffix_k} ile eşleştiriliyor.",
            file=sys.stderr,
        )
        k_used = suffix_k
    if suffix_k is not None:
        _dir_note = f'yok (suffix sonu _k{suffix_k})'
    elif k_used == dk:
        _dir_note = 'yok'
    else:
        _dir_note = f'_k{k_used} (eski düzen)'
    print(f"Assignment K  : {k_used} (label ön eki: {_dir_note})")
    print(f"Assignment kök: {root}")

    # CV/holdout'ta testte train'de görülmeyen item olabilir; item boyutunu
    # train+test birleşik maksimum ID'den al ki predict sırasında taşma olmasın.
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1

    knn_vals: List[int]
    if isinstance(knn, int):
        knn_vals = [knn]
    else:
        knn_vals = list(knn)
    if not knn_vals or any(kv < 1 for kv in knn_vals):
        raise ValueError('knn değerleri boş olamaz ve her biri en az 1 olmalı')
    multi_knn_sweep = len(knn_vals) > 1
    if multi_knn_sweep and do_cluster_knn:
        print(f"ClusterKNN k sırası: {knn_vals}")
    if do_cluster_knn and (knn_mode or 'cluster').strip().lower() == 'full_soft':
        print("ClusterKNN modu: full_soft (tüm kullanıcılar, membership×rating benzerliği)")
    if do_cluster_knn and (knn_mode or 'cluster').strip().lower() == 'weighted_cluster':
        _a = getattr(args, 'cluster_weight_alpha', 1.0) if args is not None else 1.0
        if isinstance(_a, str) and _a.strip().lower() == 'auto':
            _b = float(getattr(args, 'cluster_weight_base', 1.0) or 1.0)
            print(
                f"ClusterKNN modu: weighted_cluster "
                f"(global havuz, alpha=auto, base={_b:g})"
            )
        else:
            print(
                f"ClusterKNN modu: weighted_cluster "
                f"(global havuz, alpha={float(_a):g})"
            )

    # --- Fusion (Görev 5) hazırlığı ---
    fusion_setup = None
    if do_fusion:
        knn_item_vals: List[int]
        if isinstance(knn_item, int):
            knn_item_vals = [int(knn_item)]
        else:
            knn_item_vals = [int(k) for k in knn_item]
        if not knn_item_vals:
            knn_item_vals = [20]

        item_root = item_assign_root if item_assign_root is not None \
            else os.path.join(os.path.dirname(BASE_DIR), 'mealpy', 'results', 'item_assignments')

        ITEM_K_DEFAULT = {'ml100k': 30, 'ml1m': 60}
        k_item_ds = assignment_k_item_100k if dataset_name == 'ml100k' else assignment_k_item_1m
        if assignment_k_item is not None:
            if isinstance(assignment_k_item, (list, tuple)):
                k_item_used = int(assignment_k_item[0])
            else:
                k_item_used = int(assignment_k_item)
        elif k_item_ds is not None:
            k_item_used = int(k_item_ds)
        else:
            k_item_used = ITEM_K_DEFAULT.get(dataset_name, 30)

        n_users_full = int(max(train[:, 0].max(), test[:, 0].max())) + 1
        R_matrix_train = _build_rating_matrix_from_train(train, n_users_full, n_items)

        fusion_setup = {
            'item_root'        : item_root,
            'item_assign_suffix': item_assign_suffix,
            'k_item_used'      : k_item_used,
            'knn_item_vals'    : knn_item_vals,
            'R_matrix_train'   : R_matrix_train,
            'n_users_full'     : n_users_full,
        }
        print(f"Fusion        : mode={fusion_mode}"
              + (f" alpha={fusion_alpha}" if fusion_mode == 'fixed' else ''))
        print(f"Item assign K : {k_item_used}")
        print(f"Item kNN      : {knn_item_vals}")
        print(f"Item root     : {item_root}")

    results = []
    if fold is None:
        k_dir = os.path.join(OUT_ROOT, dataset_name, f'k{k_used}')
    else:
        k_dir = os.path.join(OUT_ROOT, dataset_name, f'k{k_used}', f'fold{fold}')
    run_n  = _next_run_index(k_dir)
    out_dir = os.path.join(k_dir, f'run{run_n}')
    os.makedirs(out_dir, exist_ok=True)
    print(f"Sonuç klasörü  : {out_dir}")

    # Mevcut hiperparametre değerlerini yakala (worker süreçlerde kullanılır)
    _ld  = LATENT_DIM
    _lr  = LEARNING_RATE
    _reg = REGULARIZATION
    _eg  = N_EPOCHS_GLOBAL
    _ec  = N_EPOCHS_CLUSTER

    # Algo-düzeyinde paralel iş sayısını belirle
    _n_candidate = len(algo_filter) if algo_filter else len(ALGO_LABELS)
    nw_algo = _resolve_pool_workers(algo_workers, _n_candidate)
    # --fusion için R_matrix_train pickle maliyeti yüksek (ML-1M ~80MB);
    # algo paralelliğini sıralıya zorla.
    if do_fusion and nw_algo > 1:
        print(
            "  Uyarı: --fusion ile algoritma paralelliği desteklenmiyor; "
            "--algo-jobs yok sayılıyor (sıralı mod).",
            file=sys.stderr,
        )
        nw_algo = 1
    # İç içe ProcessPoolExecutor CPU'yu aşırı yükler; algo paralelse küme içi
    # paralelliği kapat.
    eff_cluster_workers = 1 if nw_algo > 1 else cluster_workers

    # Global baseline(lar)
    if run_global:
        _global_rows = []

        if bool(getattr(args, 'als', False)):
            _global_rows.append(run_global_als(
                train, test, n_items, use_bias=use_bias, use_svdpp=use_svdpp,
                top_n=top_n, relevance_threshold=relevance_threshold,
            ))
        elif bool(getattr(args, 'svd', False)):
            _svd_n_factors = int(getattr(args, 'svd_factors', 100))
            _svd_n_epochs  = int(getattr(args, 'svd_epochs',  20))
            row_svd = run_global_svd(
                train, test, n_items,
                top_n=top_n, relevance_threshold=relevance_threshold,
                n_factors=_svd_n_factors, n_epochs=_svd_n_epochs,
            )
            if row_svd is not None:
                _global_rows.append(row_svd)
        elif bool(getattr(args, 'global_knn', False)):
            _gknn_k   = int(getattr(args, 'global_knn_k',   20))
            _gknn_sim = str(getattr(args, 'global_knn_sim', similarity))
            _global_rows.append(run_global_knn(
                train, test, n_items,
                similarity=_gknn_sim,
                k_neighbors=_gknn_k,
                min_common=min_common,
                top_n=top_n,
                relevance_threshold=relevance_threshold,
            ))
        elif bool(getattr(args, 'global_knn_baseline', False)):
            _gkb_k   = int(getattr(args, 'global_knn_baseline_k', 40))
            _gkb_sim = str(getattr(args, 'global_knn_baseline_sim', 'pearson_baseline'))
            _gkb_mc  = int(getattr(args, 'global_knn_baseline_min_common', 1))
            _gkb_ub  = not bool(getattr(args, 'global_knn_baseline_item_based', False))
            row_knb = run_global_surprise_knn_baseline(
                train, test, n_items,
                k_neighbors=_gkb_k,
                similarity=_gkb_sim,
                min_common=_gkb_mc,
                user_based=_gkb_ub,
                top_n=top_n,
                relevance_threshold=relevance_threshold,
            )
            if row_knb is not None:
                _global_rows.append(row_knb)
        elif bool(getattr(args, 'global_kwm', False)):
            _gkwm_k   = int(getattr(args, 'global_kwm_k', 40))
            _gkwm_sim = str(getattr(args, 'global_kwm_sim', 'msd'))
            _gkwm_mc  = int(getattr(args, 'global_kwm_min_common', 1))
            _gkwm_ub  = not bool(getattr(args, 'global_kwm_item_based', False))
            row_kwm = run_global_surprise_knn_with_means(
                train, test, n_items,
                k_neighbors=_gkwm_k,
                similarity=_gkwm_sim,
                min_common=_gkwm_mc,
                user_based=_gkwm_ub,
                top_n=top_n,
                relevance_threshold=relevance_threshold,
            )
            if row_kwm is not None:
                _global_rows.append(row_kwm)
        else:
            _global_rows.append(run_global_wnmf(
                train, test, n_items, use_bias=use_bias, use_svdpp=use_svdpp,
                top_n=top_n, relevance_threshold=relevance_threshold,
            ))

        for row in _global_rows:
            row['dataset'] = dataset_name
            results.append(row)
            _save_row_to_db(dataset_name, row, k_used, args, fold_override=fold)

    # Her algoritma — önce geçerli etiket/dizin çiftlerini topla
    active_algos: List[Tuple[str, str]] = []
    for label in ALGO_LABELS:
        if algo_filter and label not in algo_filter:
            print(f"\n  [{label}] atlandı (filtre)")
            continue

        assign_dir = _algo_assignment_dir(
            root, dataset_name, label, k_used, assign_suffix=assign_suffix,
        )
        db_dataset = _normalize_db_dataset_name(dataset_name)
        use_db = bool(getattr(args, 'assign_from_db', False)) if args else False
        if use_db or not os.path.isdir(assign_dir):
            if _DB_AVAILABLE and assign_suffix:
                ok = ensure_assignment_on_disk(
                    db_dataset,
                    label,
                    k_used,
                    assign_suffix,
                    assign_dir,
                    strategy=getattr(args, 'assign_db_strategy', 'best_wcss'),
                    overwrite=bool(getattr(args, 'assign_db_overwrite', False)),
                )
                if ok:
                    print(
                        f"\n  [{label}] assignment DB'den export edildi -> {assign_dir}",
                        flush=True,
                    )
                elif use_db:
                    print(
                        f"\n  [{label}] ATLANDI — DB'de suffix bulunamadi: "
                        f"{assign_suffix}",
                        flush=True,
                    )
                    continue
        if (
            not os.path.isdir(assign_dir)
            and label == 'B0_KMEANS'
        ):
            alt_suffix = _assign_suffix_strip_trailing_kmref(assign_suffix)
            if alt_suffix is not None:
                cand = _algo_assignment_dir(
                    root,
                    dataset_name,
                    label,
                    k_used,
                    assign_suffix=alt_suffix,
                )
                if os.path.isdir(cand):
                    print(
                        f"\n  [{label}] Not: --assign-suffix {_ASSIGN_KMREF_SUFFIX!r} "
                        f"B0_KMEANS klasöründe yok (refinement uygulanmaz); "
                        f"soneksiz kullanılıyor: …{alt_suffix}",
                        flush=True,
                    )
                    assign_dir = cand
        if not os.path.isdir(assign_dir):
            kmref_try = assign_dir + _ASSIGN_KMREF_SUFFIX
            if os.path.isdir(kmref_try):
                print(
                    f"\n  [{label}] Not: kmref klasörü kullanılıyor "
                    f"(…{_ASSIGN_KMREF_SUFFIX})",
                    flush=True,
                )
                assign_dir = kmref_try
        if not os.path.isdir(assign_dir):
            print(f"\n  [{label}] ATLANDI — assignment bulunamadı: {assign_dir}")
            continue

        active_algos.append((label, assign_dir))

    if nw_algo > 1 and len(active_algos) > 1:
        print(f"\n  Algoritmalar paralel çalıştırılıyor: "
              f"{len(active_algos)} iş, en fazla {nw_algo} süreç "
              f"(küme içi paralellik: kapalı)")
        jobs = [
            (
                label, assign_dir,
                train, test, n_items, dataset_name, mode,
                eff_cluster_workers, weighted_v, use_bias, use_cluster_bias, use_svdpp,
                run_cluster_avg, do_cluster_knn, top_n, relevance_threshold, similarity,
                tuple(knn_vals),
                min_common,
                hybrid_alpha,
                out_dir, run_command,
                _ld, _lr, _reg, _eg, _ec,
                args,
            )
            for label, assign_dir in active_algos
        ]
        with ProcessPoolExecutor(max_workers=nw_algo) as pool:
            algo_results = list(pool.map(_mp_run_algo_job, jobs))
        # ALGO_LABELS sırasını koru
        label_to_rows = {lbl: rows for lbl, rows in algo_results}
        for label, _ in active_algos:
            chunk = label_to_rows.get(label, [])
            results.extend(chunk)
            for row in chunk:
                _save_row_to_db(dataset_name, row, k_used, args, fold_override=fold)
    else:
        for label, assign_dir in active_algos:
            n_users_expected = int(max(train[:, 0].max(), test[:, 0].max())) + 1
            assignments, gray_mask = load_assignment(assign_dir)
            memberships = load_memberships(assign_dir)
            user_features = load_user_features(assign_dir, len(assignments))
            assignments, gray_mask, memberships, user_features = _align_assignment_bundle(
                assignments,
                gray_mask,
                memberships,
                user_features,
                n_users_expected=n_users_expected,
                algo_label=label,
                assign_dir=assign_dir,
            )
            _print_assignment_diagnostics(label, assignments, user_features)
            nc = _nearest_centroid_bundle(args, assign_dir, assignments)

            if run_cluster_avg:
                row = run_cluster_average(
                    train, test, assignments, gray_mask, memberships, n_items, label,
                    top_n=top_n,
                    relevance_threshold=relevance_threshold,
                    cluster_avg_hard=bool(getattr(args, 'cluster_avg_hard', False)),
                    cluster_avg_leaky=bool(getattr(args, 'cluster_avg_leaky', False)),
                    cluster_mean_impute_train=bool(
                        getattr(args, 'cluster_mean_impute_train', False)
                    ),
                    debug_cluster_avg_hard=bool(
                        getattr(args, 'debug_cluster_avg_hard', False)
                    ),
                    debug_cluster_avg_hard_limit=int(
                        getattr(args, 'debug_cluster_avg_hard_limit', 5) or 5
                    ),
                    assign_dir=assign_dir,
                    **_cluster_avg_predict_kwargs(args),
                    **nc,
                )
                row['dataset'] = dataset_name
                results.append(row)

                _save_row_to_db(dataset_name, row, k_used, args, fold_override=fold)

            if do_cluster_knn:
                nc = _knn_centroid_bundle(args, assign_dir, assignments, knn_mode=knn_mode)
                surprise_knn_variant = str(
                    getattr(args, 'cluster_knn_variant', 'baseline') or 'baseline'
                ) if args else 'baseline'
                cluster_knn_backend = str(
                    getattr(args, 'cluster_knn_backend', 'surprise') or 'surprise'
                ) if args else 'surprise'
                for kv in knn_vals:
                    row = run_cluster_knn(
                        train, test, assignments, gray_mask, memberships, n_items, label,
                        user_features=user_features,
                        use_latent_similarity=bool(
                            getattr(args, 'use_latent_similarity', False)
                        ) if args else False,
                        similarity=similarity,
                        min_common=min_common,
                        k_neighbors=int(kv),
                        expand_knn=expand_knn,
                        knn_mode=knn_mode,
                        surprise_knn_variant=surprise_knn_variant,
                        cluster_knn_backend=cluster_knn_backend,
                        sig_weight=int(getattr(args, 'sig_weight', 0) or 0) if args else 0,
                        sim_amp=float(getattr(args, 'sim_amp', 1.0) or 1.0) if args else 1.0,
                        top_n=top_n,
                        relevance_threshold=relevance_threshold,
                        assign_dir=assign_dir,
                        **nc,
                    )
                    row['dataset'] = dataset_name
                    results.append(row)

                    _save_row_to_db(dataset_name, row, k_used, args, fold_override=fold)

            if do_fusion and fusion_setup is not None:
                item_assign_dir = _algo_assignment_dir(
                    fusion_setup['item_root'],
                    dataset_name,
                    label,
                    fusion_setup['k_item_used'],
                    assign_suffix=fusion_setup['item_assign_suffix'],
                )
                item_path = os.path.join(item_assign_dir, 'assignments.npy')
                if not os.path.isfile(item_path):
                    print(
                        f"  [{label} | Fusion] ATLANDI — item assignment yok: "
                        f"{item_path}",
                        flush=True,
                    )
                else:
                    item_assignments_arr = np.load(item_path)
                    # item_indices.npy varsa: pruned item uzayındaki assignment'ları
                    # orijinal item uzayına yay (-1 = atanmamış/pruned).
                    _item_idx_path = os.path.join(item_assign_dir, 'item_indices.npy')
                    if os.path.isfile(_item_idx_path):
                        _item_indices = np.load(_item_idx_path)
                        _full = np.full(n_items, -1, dtype=np.int32)
                        _full[_item_indices] = item_assignments_arr.astype(np.int32)
                        item_assignments_arr = _full
                        print(
                            f"  [{label} | Fusion] item_indices yüklendi: "
                            f"{len(_item_indices)}/{n_items} item eşlendi",
                            flush=True,
                        )
                    for k_user_v in knn_vals:
                        for k_item_v in fusion_setup['knn_item_vals']:
                            row = run_cluster_knn_fusion(
                                train, test,
                                assignments, item_assignments_arr,
                                fusion_setup['R_matrix_train'],
                                gray_mask,
                                n_items, fusion_setup['n_users_full'],
                                algo_label_user=label,
                                algo_label_item=label,
                                similarity=similarity,
                                min_common=min_common,
                                k_user=int(k_user_v),
                                k_item=int(k_item_v),
                                fusion_mode=fusion_mode,
                                fusion_alpha=fusion_alpha,
                                top_n=top_n,
                                relevance_threshold=relevance_threshold,
                                compute_cov=compute_cov,
                                compute_div=compute_div,
                            )
                            row['dataset'] = dataset_name
                            results.append(row)
                            _save_row_to_db(
                                dataset_name, row, k_used, args, fold_override=fold,
                            )

            if mode in ('full', 'all'):
                row = run_cluster_full(
                    train, test, assignments, gray_mask, memberships, n_items, label,
                    max_workers=cluster_workers,
                    use_svdpp=use_svdpp,
                    top_n=top_n,
                    relevance_threshold=relevance_threshold,
                )
                row['dataset'] = dataset_name
                results.append(row)

                _save_row_to_db(dataset_name, row, k_used, args, fold_override=fold)

            if mode in ('sharedV', 'all'):
                row = run_cluster_sharedV(
                    train, test, assignments, gray_mask, memberships, n_items, label,
                    max_workers=cluster_workers,
                    out_dir=out_dir,
                    weighted_v=weighted_v,
                    use_bias=use_bias,
                    use_cluster_bias=use_cluster_bias,
                    use_svdpp=use_svdpp,
                    run_command=run_command,
                    top_n=top_n,
                    relevance_threshold=relevance_threshold,
                    hybrid_alpha=hybrid_alpha,
                )
                row['dataset'] = dataset_name
                results.append(row)

                _save_row_to_db(dataset_name, row, k_used, args, fold_override=fold)

    sub = os.path.basename(out_dir)
    tagged: List[dict] = []
    for r in results:
        ck_tag: Optional[int] = None
        if multi_knn_sweep and r.get('scenario') in _CLUSTER_KNN_SCENARIOS:
            knv = r.get('k_neighbors')
            if knv is not None:
                ck_tag = int(knv)
        meta = _result_row_meta(k_used, cknn_suffix=ck_tag)
        tagged.append({**meta, **r, 'result_subdir': sub, 'use_svdpp': bool(use_svdpp)})
    results = tagged

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
    _print_scenario_compare_table(
        results,
        ('calc_avg_rating', 'calc_avg_rating_leaky', 'cluster_avg', 'cluster_avg_nc'),
        title=f'{dataset_name.upper()} · Küme ortalaması karşılaştırma',
    )
    _print_knn_cluster_mae_matrix(
        results,
        title=f'{dataset_name.upper()} · ClusterKNN karşılaştırma',
    )

    return results


_CLUSTER_KNN_SCENARIOS = (
    'cluster_knn',
    'cluster_knn_full_soft',
    'cluster_knn_with_means',
    'cluster_knn_with_means_gs_split',
    'cluster_knn_baseline',
    'cluster_knn_baseline_gs_split',
    'cluster_knn_native_baseline',
    'cluster_knn_native_baseline_gs_split',
)


_COMPARE_METRIC_ROWS = (
    ('mae', 'MAE'),
    ('rmse', 'RMSE'),
    ('precision_at_10', 'Prec@10'),
    ('recall_at_10', 'Rec@10'),
    ('ndcg_at_10', 'NDCG@10'),
    ('jaccard_at_10', 'Jacc@10'),
)


def _fmt_compare_cell(v: float, width: int) -> str:
    if isinstance(v, float) and np.isnan(v):
        return f'{"—":>{width}}'
    return f'{v:>{width}.4f}'


def _print_scenario_compare_table(
    results: List[dict],
    scenarios: Tuple[str, ...],
    *,
    title: str = 'Senaryo karşılaştırma',
) -> None:
    """Aynı senaryo grubundaki satırlar için algoritma × metrik tablosu."""
    rows = [r for r in results if r.get('scenario') in scenarios]
    if not rows:
        return
    algos: List[str] = []
    for r in rows:
        al = str(r.get('algo_label', ''))
        if al not in algos:
            algos.append(al)
    if len(algos) < 2:
        return

    al_w = max(16, max(len(a) for a in algos))
    col_w = 9
    print(f"\n{'=' * (al_w + col_w * len(_COMPARE_METRIC_ROWS) + 2)}")
    print(title)
    header = f"{'Algoritma':<{al_w}}"
    for _, label in _COMPARE_METRIC_ROWS:
        header += f'{label:>{col_w}}'
    print(header)
    print('—' * len(header))
    for al in algos:
        match = next((r for r in rows if str(r.get('algo_label', '')) == al), None)
        if match is None:
            continue
        line = f"{al:<{al_w}}"
        for key, _ in _COMPARE_METRIC_ROWS:
            line += _fmt_compare_cell(float(match.get(key, float('nan'))), col_w)
        print(line)
    print('=' * len(header))


def _print_knn_cluster_mae_matrix(results: List[dict], *,
                                  title: str = 'ClusterKNN · k seçimi') -> None:
    """cluster_knn satırlarından algoritma × k_neighbors pivot (MAE/RMSE/P/R/NDCG)."""
    ck = [
        r for r in results
        if r.get('scenario') in _CLUSTER_KNN_SCENARIOS
        and r.get('k_neighbors') is not None
    ]
    if not ck:
        return
    knn_seen: List[int] = []
    algo_seen: List[str] = []
    for r in ck:
        k = int(r['k_neighbors'])
        if k not in knn_seen:
            knn_seen.append(k)
        al = str(r.get('algo_label', ''))
        if al not in algo_seen:
            algo_seen.append(al)
    knn_seen.sort()
    if len(algo_seen) < 2 and len(knn_seen) <= 1:
        return
    cell_map: Dict[Tuple[str, int], dict] = {}
    for r in ck:
        key = (str(r.get('algo_label', '')), int(r['k_neighbors']))
        cell_map[key] = r

    cw = max(8, max(len(str(kv)) for kv in knn_seen) + 2)
    al_w = max(14, max((len(a) for a in algo_seen), default=14))

    def _fmt_row(metric_key: str) -> None:
        label = next(lbl for k, lbl in _COMPARE_METRIC_ROWS if k == metric_key)
        print(label)
        for al in algo_seen:
            parts = ''.join(
                _fmt_compare_cell(
                    float(cell_map.get((al, kv), {}).get(metric_key, float('nan'))),
                    cw,
                )
                for kv in knn_seen
            )
            print(f"{al:<{al_w}} {parts}")

    print(f"\n{'='*72}")
    print(title)
    print(f"{'Algoritma':<{al_w}} " + ''.join(f'{kv:>{cw}}' for kv in knn_seen))
    print('—' * (al_w + 1 + cw * len(knn_seen)))
    for metric_key, _ in _COMPARE_METRIC_ROWS:
        _fmt_row(metric_key)
    print('=' * 72)


# ============================================================
# ÖZET TABLOSU
# ============================================================

def _print_summary(results, dataset_name):
    print(f"\n{'='*60}")
    print(f"ÖZET — {dataset_name.upper()}")
    print(f"{'='*60}")
    tag_hdr = 'hyperparam_tag' if results and 'hyperparam_tag' in results[0] else None
    show_knn = bool(
        results
        and any(
            r.get('scenario') in _CLUSTER_KNN_SCENARIOS
            and r.get('k_neighbors') is not None
            for r in results
        )
    )
    _kpref = (f"{'kNN':>5} ") if show_knn else ''

    def _row_knn_part(r):
        if not show_knn:
            return ''
        if r.get('k_neighbors') is not None:
            return f"{int(r['k_neighbors']):>5} "
        return f"{'—':>5} "

    if tag_hdr:
        print(
            f"{'tag':<36} {'Algoritma':<16} {'Senaryo':<14} {_kpref}"
            f"{'MAE':>8} {'RMSE':>8} "
            f"{'ClStd':>8} {'GS MAE':>8} {'Wh MAE':>8} "
            f"{'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'NDCG':>8} {'Jacc':>8}"
        )
        w = 36 + 16 + 14 + len(_kpref) + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 6
    else:
        print(
            f"{'Algoritma':<20} {'Senaryo':<16} {_kpref}"
            f"{'MAE':>8} {'RMSE':>8} "
            f"{'ClStd':>8} {'GS MAE':>8} {'Wh MAE':>8} "
            f"{'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'NDCG':>8} {'Jacc':>8}"
        )
        w = 72 + len(_kpref) + 8 + 8 + 8 + 8 + 8 + 8
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
        jc = r.get('jaccard_at_10', float('nan'))
        jc_str = f"{jc:.4f}" if not (isinstance(jc, float) and np.isnan(jc)) else "  —  "
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
                f"{_row_knn_part(r)}"
                f"{r['mae']:>8.4f} "
                f"{r['rmse']:>8.4f} "
                f"{cms_str:>8} "
                f"{gs_str:>8} "
                f"{wh_str:>8} "
                f"{acc_str:>8} "
                f"{pr_str:>8} "
                f"{rc_str:>8} "
                f"{f1_str:>8} "
                f"{nd_str:>8} "
                f"{jc_str:>8}"
            )
        else:
            print(
                f"{r['algo_label']:<20} "
                f"{r['scenario']:<16} "
                f"{_row_knn_part(r)}"
                f"{r['mae']:>8.4f} "
                f"{r['rmse']:>8.4f} "
                f"{cms_str:>8} "
                f"{gs_str:>8} "
                f"{wh_str:>8} "
                f"{acc_str:>8} "
                f"{pr_str:>8} "
                f"{rc_str:>8} "
                f"{f1_str:>8} "
                f"{nd_str:>8} "
                f"{jc_str:>8}"
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
            row.get('k_neighbors'),
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
        for metric_key in ('precision_at_10', 'recall_at_10', 'f1_at_10', 'ndcg_at_10', 'jaccard_at_10'):
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
        '--eval-split',
        choices=['official', 'random'],
        default='official',
        help='ML-100K değerlendirme bölmesi: official=u1.base/u1.test (varsayılan); '
             'random=u.data üzerinde rastgele %%20 (HSC makalesi protokolü). ML-1M her zaman rastgele.',
    )
    p.add_argument(
        '--nearest-centroid',
        action='store_true',
        help='Test tahmininde train profiline göre en yakın centroid kümesini seç (HSC). '
             'best_sol.npy gerekir; boyut n_items ile eşleşmeli (fe=none).',
    )
    p.add_argument(
        '--centroid-metric',
        choices=['euclidean', 'pearson'],
        default='euclidean',
        help='--nearest-centroid ile centroid uzaklık metriği (default: euclidean).',
    )
    p.add_argument(
        '--fold', type=int, default=None, metavar='N',
        help='İsteğe bağlı holdout fold (1–5). Verilmezse: 5-fold CV (u1 veya u.data). '
             'Verilirse: official→u{N}.base/u{N}.test; random→u.data KFold fold N (N=1: %%20 holdout). '
             'Sonuçlar .../k{K}/fold{N}/run*/ veya fold yoksa .../k{K}/run*/.',
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
        '--assign-from-db', action='store_true',
        help='Assignment dosyalarini diskte yoksa (veya --assign-db-overwrite ile) '
             'SQLite assignments tablosundan assign_suffix ile export et.',
    )
    p.add_argument(
        '--assign-db-overwrite', action='store_true',
        help='--assign-from-db: mevcut klasoru DB kaydi ile ustune yaz.',
    )
    p.add_argument(
        '--assign-db-strategy', choices=['best_wcss', 'latest', 'worst_wcss'],
        default='best_wcss',
        help='Ayni suffix icin birden fazla assignment varsa secim (varsayilan: best_wcss).',
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
    _bias_grp = p.add_mutually_exclusive_group()
    _bias_grp.add_argument(
        '--bias',
        dest='use_bias',
        action='store_true',
        help='Bias açık (mu, b_u, b_i; varsayılan)',
    )
    _bias_grp.add_argument(
        '--no-bias',
        dest='use_bias',
        action='store_false',
        help='Bias kapalı',
    )
    p.set_defaults(use_bias=True)
    p.add_argument(
        '--cluster-bias', action='store_true',
        help='Küme bazlı mu_k kullan (global mu yerine)',
    )
    p.add_argument(
        '--no-cluster-avg', action='store_true',
        help='ClusterAvg senaryosunu çalıştırma',
    )
    p.add_argument(
        '--cluster-avg-hard',
        action='store_true',
        help='Thakrar et al. (2025) CalculateAverageRating: küme-içi train ortalaması, '
             'soft membership yok (senaryo: calc_avg_rating).',
    )
    p.add_argument(
        '--cluster-avg-base',
        choices=['user', 'cluster', 'cluster_item_pop'],
        default='user',
        help='ClusterAvg taban: user=mean_u (varsayılan); cluster=mean_cluster(cid); '
             'cluster_item_pop=mean_cluster+(mean_item-global_mean).',
    )
    p.add_argument(
        '--cluster-avg-leaky',
        action='store_true',
        help='Küme-item ortalamasını train+test ile hesapla (makale-tipi sızıntı; '
             'senaryo: calc_avg_rating_leaky). Tahmin yine test çiftlerinde.',
    )
    p.add_argument(
        '--cluster-mean-impute-train',
        action='store_true',
        help='ClusterAvg: train eksik (u,i) hücrelerini küme ortalamasıyla doldurup '
             'küme×item ortalamasını yeniden hesapla (2 geçiş).',
    )
    p.add_argument(
        '--debug-cluster-avg-hard',
        action='store_true',
        help='calc_avg_rating sırasında tahmin kaynağını (cluster/item/user/global) '
             'satır bazında kısa örneklerle logla.',
    )
    p.add_argument(
        '--debug-cluster-avg-hard-limit',
        type=int,
        default=5,
        help='--debug-cluster-avg-hard için algoritma başına maksimum örnek log satırı '
             '(varsayılan: 5).',
    )
    p.add_argument(
        '--no-cluster-knn', action='store_true',
        help='ClusterKNNWithMeans (Surprise küme-içi CF) senaryosunu çalıştırma',
    )
    p.add_argument(
        '--paper-mode', action='store_true',
        help='GOA makalesi eval preset: yalnızca küme ortalama tahmini '
             '(--cluster-avg-hard, --no-cluster-knn; senaryo: calc_avg_rating).',
    )
    p.add_argument(
        '--meta-eval', action='store_true',
        help='Meta-algoritma kümeleme değerlendirmesi: küme ortalaması kapalı, '
             'Surprise KNNBaseline/küme açık (--no-cluster-avg).',
    )
    p.add_argument(
        '--top-n', type=int, default=10,
        help='Precision/recall Top-N (cluster_avg)',
    )
    p.add_argument(
        '--relevance-threshold', type=_parse_relevance_threshold_arg, default=4.0,
        help='Top-N/accuracy için relevant eşiği: sabit sayı (varsayılan 4.0) veya '
             'cluster-mean (kullanıcının kümesinin train rating ortalaması)',
    )
    p.add_argument(
        '--similarity',
        choices=['pearson', 'pearson_iuf', 'cosine', 'msd'],
        default='pearson',
        help='kNN benzerlik: pearson (default), pearson_iuf (IUF ağırlıklı), cosine, '
             'msd (Surprise MSD; küme Surprise kNN ile önerilir)',
    )
    p.add_argument(
        '--knn', nargs='+', type=int, default=[30], metavar='K',
        help='Cluster Surprise kNN komşu sayısı (KNNBaseline vars.; --cluster-knn-variant); '
             'birden fazla: --knn 5 10 15 20 '
             '(her K için ayrı satır; çoklu K iken --skip-existing bu koşu için devre dışı). Varsayılan: 30.',
    )
    p.add_argument(
        '--cluster-knn-variant',
        choices=['baseline', 'withmeans', 'coclustering'],
        default='baseline',
        dest='cluster_knn_variant',
        help='Cluster kNN Surprise modeli: baseline=KNNBaseline (vars.), '
             'withmeans=KNNWithMeans, coclustering=CoClustering. '
             'Yalnız --cluster-knn-backend surprise iken geçerli.',
    )
    p.add_argument(
        '--cluster-knn-backend',
        choices=['surprise', 'manual', 'native'],
        default='surprise',
        dest='cluster_knn_backend',
        help='Küme kNN motoru: surprise=Surprise KNNBaseline/WithMeans (vars.); '
             'native=Surprise\'sız sapma tabanlı küme kNN (cluster_predictor.py); '
             'manual=manuel Pearson/cosine kNN. '
             'expand-knn / weighted_cluster / full_soft → manual (native uyumsuz).',
    )
    p.add_argument(
        '--use-latent-similarity',
        action='store_true',
        dest='use_latent_similarity',
        help='Cluster kNN: Surprise yerine WNMF user_features (U) üzerinde latent cosine '
             'benzerliği kullan (user_features.npy gerekir).',
    )
    p.add_argument(
        '--min-common', type=int, default=3, metavar='N',
        help='ClusterKNN: benzerlik için minimum ortak film sayısı (varsayılan: 3)',
    )
    p.add_argument(
        '--expand-knn',
        action='store_true',
        help='Küme sınırını aşarak komşu ara',
    )
    p.add_argument(
        '--knn-mode',
        choices=['cluster', 'full_soft', 'weighted_cluster'],
        default='cluster',
        help='cluster (varsayılan): küme içi kNN veya soft-blend; '
             'weighted_cluster: global havuz, sim×centroid-üyelik ağırlığı (best_sol.npy); '
             'full_soft: tüm kullanıcılar, sim = pearson×(1+⟨m_u,m_v⟩) '
             '(memberships.npy gerekir)',
    )
    p.add_argument(
        '--cluster-weight-alpha', type=_parse_cluster_weight_alpha, default=1.0,
        metavar='A',
        help='weighted_cluster: sabit A (membership_w=1/(1+A×d_centroid)) veya '
             'auto (küme-bazlı dinamik alpha); A=0 global KNN eşdeğeri (varsayılan: 1.0)',
    )
    p.add_argument(
        '--cluster-weight-base', type=float, default=1.0, metavar='B',
        help='weighted_cluster + alpha=auto: compute_alpha(..., base=B) '
             '(varsayılan: 1.0; grid: 0.5, 1.0, 2.0)',
    )
    p.add_argument(
        '--sig-weight', type=int, default=0,
        help='Significance weighting eşiği (0=kapalı, önerilen=50)',
    )
    p.add_argument(
        '--sim-amp', type=float, default=1.0,
        help='Similarity amplification üssü (1.0=kapalı, önerilen=1.5)',
    )
    p.add_argument(
        '--hybrid-alpha', type=float, default=None,
        metavar='A',
        help='SharedV+kNN hibrit tahmin ağırlığı (0.0-1.0). '
             '0=sadece WNMF, 1=sadece kNN, 0.3=önerilen. '
             'Verilmezse hibrit kapalı.',
    )
    p.add_argument(
        '--svdpp', action='store_true',
        help='WNMFModel / global WNMFSharedV için SVD++ (implicit Y) kullan',
    )
    p.add_argument(
        '--als',
        action='store_true',
        help='Global baseline için WNMF yerine ALS çalıştır',
    )
    p.add_argument(
        '--svd',
        action='store_true',
        help='Global baseline için Surprise SVD (Simon Funk biased MF) çalıştır. '
             'Non-negativity kısıtı yok; WNMF ile üst sınır karşılaştırması için.',
    )
    p.add_argument(
        '--svd-factors', type=int, default=100, metavar='K',
        dest='svd_factors',
        help='Surprise SVD gizli faktör sayısı (varsayılan: 100)',
    )
    p.add_argument(
        '--svd-epochs', type=int, default=20, metavar='E',
        dest='svd_epochs',
        help='Surprise SVD epoch sayısı (varsayılan: 20)',
    )
    p.add_argument(
        '--global-knn',
        action='store_true',
        dest='global_knn',
        help='Global baseline olarak kümelemesiz User-KNN çalıştır. '
             'ClusterKNN ile karşılaştırma: kümeleme KNN\'i iyileştiriyor mu?',
    )
    p.add_argument(
        '--global-knn-k', type=int, default=20, metavar='K',
        dest='global_knn_k',
        help='Global KNN komşu sayısı (varsayılan: 20)',
    )
    p.add_argument(
        '--global-knn-sim',
        choices=['pearson', 'pearson_iuf', 'cosine'],
        default='pearson',
        dest='global_knn_sim',
        help='Global KNN similarity metriği (varsayılan: pearson)',
    )
    p.add_argument(
        '--global-knn-baseline',
        action='store_true',
        dest='global_knn_baseline',
        help='Global baseline: Surprise KNNBaseline (bias + kNN; önerilen sim=pearson_baseline).',
    )
    p.add_argument(
        '--global-knn-baseline-k', type=int, default=40, metavar='K',
        dest='global_knn_baseline_k',
        help='Global KNNBaseline komşu sayısı (varsayılan: 40)',
    )
    p.add_argument(
        '--global-knn-baseline-sim',
        choices=['pearson_baseline', 'msd', 'pearson', 'cosine'],
        default='pearson_baseline',
        dest='global_knn_baseline_sim',
        help='Global KNNBaseline benzerlik (Surprise önerisi: pearson_baseline)',
    )
    p.add_argument(
        '--global-knn-baseline-min-common', type=int, default=1, metavar='N',
        dest='global_knn_baseline_min_common',
        help='Global KNNBaseline min_support (varsayılan: 1)',
    )
    p.add_argument(
        '--global-knn-baseline-item-based',
        action='store_true',
        dest='global_knn_baseline_item_based',
        help='Item-based KNNBaseline (varsayılan: user-based)',
    )
    p.add_argument(
        '--global-kwm',
        action='store_true',
        dest='global_kwm',
        help='Global baseline: Surprise KNNWithMeans (varsayılan k=40, sim=msd). '
             'Kümeleme yok; Surprise dokümantasyonundaki standart user-based CF.',
    )
    p.add_argument(
        '--global-kwm-k', type=int, default=40, metavar='K',
        dest='global_kwm_k',
        help='Global KNNWithMeans komşu sayısı (Surprise varsayılan: 40)',
    )
    p.add_argument(
        '--global-kwm-sim',
        choices=['msd', 'pearson', 'cosine'],
        default='msd',
        dest='global_kwm_sim',
        help='Global KNNWithMeans benzerlik (Surprise varsayılan: msd)',
    )
    p.add_argument(
        '--global-kwm-min-common', type=int, default=1, metavar='N',
        dest='global_kwm_min_common',
        help='Global KNNWithMeans min_support (Surprise varsayılan: 1)',
    )
    p.add_argument(
        '--global-kwm-item-based',
        action='store_true',
        dest='global_kwm_item_based',
        help='Item-based KNNWithMeans (varsayılan: user-based)',
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
    # --- Fusion / Item-CF flag'leri (Görev 7) ---
    p.add_argument(
        '--item-assign-root', type=str, default=None,
        help='Item assignment kök dizini (varsayılan: mealpy/results/item_assignments). '
             '--fusion ile kullanılır.',
    )
    p.add_argument(
        '--item-assign-suffix', type=str, default='',
        help='Item assignment klasör adına ek suffix (örn: _euc_imkpp_nogs_none_svd50_k30). '
             '--fusion ile kullanılır.',
    )
    p.add_argument(
        '--k-item', nargs='+', type=int, default=None,
        dest='assignment_k_item',
        metavar='K',
        help='Item assignment K (film kümesi sayısı); varsayılan ML-100K=30, ML-1M=60. '
             'Çoklu değer kabul edilir; her K user-K ile eşleştirilir (zip).',
    )
    p.add_argument(
        '--k-item-100k', type=int, default=None,
        dest='assignment_k_item_100k',
        help='Sadece ML-100K item K',
    )
    p.add_argument(
        '--k-item-1m', type=int, default=None,
        dest='assignment_k_item_1m',
        help='Sadece ML-1M item K',
    )
    p.add_argument(
        '--knn-item', nargs='+', type=int, default=[20], metavar='K',
        help='Item-CF: film kümesi içi komşu sayısı (default: 20)',
    )
    p.add_argument(
        '--fusion', action='store_true',
        help='Weighted Fusion senaryosu: User-CF + Item-CF birleşimini '
             'çalıştır (mevcut ClusterKNN yanında ek satır olarak).',
    )
    p.add_argument(
        '--fusion-mode', choices=['fixed', 'dynamic'], default='dynamic',
        help='fixed=alpha sabit, dynamic=centroid cosine benzerliğine göre m,n (default).',
    )
    p.add_argument(
        '--fusion-alpha', type=float, default=0.5,
        help='--fusion-mode fixed iken item-CF ağırlığı (1-alpha user-CF). default 0.5.',
    )
    p.add_argument(
        '--coverage', action='store_true',
        help='Fusion sonucu için coverage (önerilen benzersiz film oranı) hesapla.',
    )
    p.add_argument(
        '--diversity', action='store_true',
        help='Fusion sonucu için diversity (önerilen filmler arası 1-sim ortalaması) hesapla. '
             'Maliyetli; max 500 kullanıcı altörneklenir.',
    )
    args = p.parse_args()
    if any(k < 1 for k in args.knn):
        p.error('--knn içindeki her değer en az 1 olmalı')
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
    # --- Fusion validasyonu (Görev 7) ---
    if args.assignment_k_item is not None and (
        args.assignment_k_item_100k is not None or args.assignment_k_item_1m is not None
    ):
        p.error('--k-item ile --k-item-100k / --k-item-1m birlikte kullanılamaz')
    if any(k < 1 for k in args.knn_item):
        p.error('--knn-item içindeki her değer en az 1 olmalı')
    if not (0.0 <= args.fusion_alpha <= 1.0):
        p.error('--fusion-alpha [0, 1] aralığında olmalı')
    if args.fusion and args.fusion_mode == 'dynamic' and not args.assign_root \
       and not args.item_assign_root:
        # Bilgi notu — kullanıcı her iki rootu da default'a bırakıyor (sorun yok)
        pass
    if args.diversity and not args.fusion:
        print("Uyarı: --diversity yalnız --fusion ile birlikte hesaplanır; "
              "yok sayılıyor.", file=sys.stderr)
    if args.coverage and not args.fusion:
        print("Uyarı: --coverage yalnız --fusion ile birlikte hesaplanır; "
              "yok sayılıyor.", file=sys.stderr)
    if getattr(args, 'paper_mode', False):
        args.no_cluster_knn = True
        args.cluster_avg_hard = True
        if args.no_cluster_avg:
            p.error('--paper-mode ile --no-cluster-avg birlikte kullanılamaz')
        print(
            'Not: --paper-mode aktif → yalnızca CalcAvgRating (cluster_avg_hard), '
            'ClusterKNN kapalı.',
            file=sys.stderr,
        )
    if getattr(args, 'meta_eval', False):
        if getattr(args, 'paper_mode', False):
            p.error('--meta-eval ile --paper-mode birlikte kullanılamaz')
        args.no_cluster_avg = True
        args.no_cluster_knn = False
        print(
            'Not: --meta-eval aktif → küme ortalaması kapalı, '
            'Surprise KNNBaseline/küme (--cluster-knn-backend surprise vars.).',
            file=sys.stderr,
        )
    return args


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
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

    print("=" * 60)
    print("WNMF DENEY")
    print("=" * 60)
    print(f"Dataset       : {args.dataset}")
    if args.dataset in ('100k', 'both'):
        print(f"Eval split    : {args.eval_split}"
              + (' (u.data, rastgele %20)' if args.eval_split == 'random' else ' (u1.base/u1.test)'))
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

    t_total  = time.time()
    all_rows = []
    all_tags = []

    multi_k = args.assignment_k is not None and len(args.assignment_k) > 1

    def _k_iter_100k():
        if args.assignment_k is not None:
            return args.assignment_k
        return [
            _resolved_assignment_k(
                args.assignment_k_100k, None, ASSIGN_K_DEFAULT_100K,
            ),
        ]

    def _k_iter_1m():
        if args.assignment_k is not None:
            return args.assignment_k
        return [
            _resolved_assignment_k(
                args.assignment_k_1m, None, ASSIGN_K_DEFAULT_1M,
            ),
        ]

    if args.dataset in ('100k', 'both'):
        ks = _k_iter_100k()
        tag_k = '-'.join(map(str, ks)) if len(ks) > 1 else str(ks[0])
        all_tags.append(f"ml100k_k{tag_k}")
    if args.dataset in ('1m', 'both'):
        ks = _k_iter_1m()
        tag_k = '-'.join(map(str, ks)) if len(ks) > 1 else str(ks[0])
        all_tags.append(f"ml1m_k{tag_k}")

    train_100k = test_100k = None
    train_1m   = test_1m = None
    full_100k = full_1m = None
    if args.dataset in ('100k', 'both'):
        if args.eval_split == 'random':
            train_100k, test_100k = load_ratings_100k_all(
                DATA_100K_ALL,
                random_seed=RANDOM_SEED,
                fold=args.fold,
            )
        elif args.fold is None:
            train_100k, test_100k = load_ratings_100k(DATA_100K_TRAIN, DATA_100K_TEST)
        else:
            data_100k_train = os.path.join(
                os.path.dirname(BASE_DIR), 'data', 'ml-100k', f'u{args.fold}.base'
            )
            data_100k_test = os.path.join(
                os.path.dirname(BASE_DIR), 'data', 'ml-100k', f'u{args.fold}.test'
            )
            train_100k, test_100k = load_ratings_100k(data_100k_train, data_100k_test)
    if args.dataset in ('1m', 'both'):
        train_1m, test_1m = load_ratings_1m(
            DATA_1M, random_seed=RANDOM_SEED, fold=args.fold,
        )
    if args.fold is None:
        if train_100k is not None and test_100k is not None:
            full_100k = np.concatenate([train_100k, test_100k], axis=0)
        if train_1m is not None and test_1m is not None:
            full_1m = np.concatenate([train_1m, test_1m], axis=0)

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

            if args.dataset in ('100k', 'both'):
                for K in _k_iter_100k():
                    if _should_skip_existing_run(
                        args,
                        'ml100k',
                        K,
                        args.mode,
                        args.fold,
                        use_sp,
                        args.fold is None and full_100k is not None,
                    ):
                        print(
                            f"[skip-existing] ml100k K={K}  "
                            f"tag={_format_hyperparam_tag(K)}  use_svdpp={use_sp}"
                        )
                        continue
                    if args.fold is None and full_100k is not None:
                        fold_rows: List[dict] = []
                        print("\n[ml100k] 5-Fold CV başlatılıyor (n_splits=5, shuffle=True, random_state=42)")
                        for fold_idx, (cv_train, cv_test) in enumerate(
                            _build_kfold_splits(full_100k, n_splits=5, shuffle=True, random_state=42),
                            start=1,
                        ):
                            print(f"\n[ml100k] Fold {fold_idx}/5")
                            rows = run_dataset(
                                'ml100k', cv_train, cv_test,
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
                                use_bias=args.use_bias,
                                use_cluster_bias=args.cluster_bias,
                                run_cluster_avg=not args.no_cluster_avg,
                                do_cluster_knn=not args.no_cluster_knn,
                                top_n=args.top_n,
                                relevance_threshold=args.relevance_threshold,
                                similarity=args.similarity,
                                knn=args.knn,
                                min_common=args.min_common,
                                expand_knn=args.expand_knn,
                                knn_mode=args.knn_mode,
                                hybrid_alpha=args.hybrid_alpha,
                                use_svdpp=use_sp,
                                run_command=RUN_COMMAND,
                                args=args,
                                fold=fold_idx,
                            )
                            for row in rows:
                                row['cv_fold'] = fold_idx
                            fold_rows.extend(rows)
                        mean_rows = _aggregate_fold_results(fold_rows, n_splits=5)
                        for row in mean_rows:
                            _save_row_to_db('ml100k', row, K, args)
                        _save_cv_mean_results('ml100k', K, args.mode, mean_rows, RUN_COMMAND)
                        combo_rows.extend(mean_rows)
                    else:
                        rows = run_dataset(
                            'ml100k', train_100k, test_100k,
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
                            use_bias=args.use_bias,
                            use_cluster_bias=args.cluster_bias,
                            run_cluster_avg=not args.no_cluster_avg,
                            do_cluster_knn=not args.no_cluster_knn,
                            top_n=args.top_n,
                            relevance_threshold=args.relevance_threshold,
                            similarity=args.similarity,
                            knn=args.knn,
                            min_common=args.min_common,
                            expand_knn=args.expand_knn,
                            knn_mode=args.knn_mode,
                            hybrid_alpha=args.hybrid_alpha,
                            use_svdpp=use_sp,
                            run_command=RUN_COMMAND,
                            args=args,
                            fold=args.fold,
                        )
                        combo_rows.extend(rows)

            if args.dataset in ('1m', 'both'):
                for K in _k_iter_1m():
                    if _should_skip_existing_run(
                        args,
                        'ml1m',
                        K,
                        args.mode,
                        args.fold,
                        use_sp,
                        args.fold is None and full_1m is not None,
                    ):
                        print(
                            f"[skip-existing] ml1m K={K}  "
                            f"tag={_format_hyperparam_tag(K)}  use_svdpp={use_sp}"
                        )
                        continue
                    if args.fold is None and full_1m is not None:
                        fold_rows: List[dict] = []
                        print("\n[ml1m] 5-Fold CV başlatılıyor (n_splits=5, shuffle=True, random_state=42)")
                        for fold_idx, (cv_train, cv_test) in enumerate(
                            _build_kfold_splits(full_1m, n_splits=5, shuffle=True, random_state=42),
                            start=1,
                        ):
                            print(f"\n[ml1m] Fold {fold_idx}/5")
                            rows = run_dataset(
                                'ml1m', cv_train, cv_test,
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
                                use_bias=args.use_bias,
                                use_cluster_bias=args.cluster_bias,
                                run_cluster_avg=not args.no_cluster_avg,
                                do_cluster_knn=not args.no_cluster_knn,
                                top_n=args.top_n,
                                relevance_threshold=args.relevance_threshold,
                                similarity=args.similarity,
                                knn=args.knn,
                                min_common=args.min_common,
                                expand_knn=args.expand_knn,
                                knn_mode=args.knn_mode,
                                hybrid_alpha=args.hybrid_alpha,
                                use_svdpp=use_sp,
                                run_command=RUN_COMMAND,
                                args=args,
                                fold=fold_idx,
                            )
                            for row in rows:
                                row['cv_fold'] = fold_idx
                            fold_rows.extend(rows)
                        mean_rows = _aggregate_fold_results(fold_rows, n_splits=5)
                        for row in mean_rows:
                            _save_row_to_db('ml1m', row, K, args)
                        _save_cv_mean_results('ml1m', K, args.mode, mean_rows, RUN_COMMAND)
                        combo_rows.extend(mean_rows)
                    else:
                        rows = run_dataset(
                            'ml1m', train_1m, test_1m,
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
                            use_bias=args.use_bias,
                            use_cluster_bias=args.cluster_bias,
                            run_cluster_avg=not args.no_cluster_avg,
                            do_cluster_knn=not args.no_cluster_knn,
                            top_n=args.top_n,
                            relevance_threshold=args.relevance_threshold,
                            similarity=args.similarity,
                            knn=args.knn,
                            min_common=args.min_common,
                            expand_knn=args.expand_knn,
                            knn_mode=args.knn_mode,
                            hybrid_alpha=args.hybrid_alpha,
                            use_svdpp=use_sp,
                            run_command=RUN_COMMAND,
                            args=args,
                            fold=args.fold,
                        )
                        combo_rows.extend(rows)

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