"""

Surprise global KNN baseline (kümeleme yok).



Varsayılan: KNNBaseline (bias + kNN), sim=pearson_baseline.

Alternatif: --variant withmeans → KNNWithMeans.



Örnek:

  python scripts/run_surprise_kwm_baseline.py

  python scripts/run_surprise_kwm_baseline.py --variant baseline --sim pearson_baseline

  python scripts/run_surprise_kwm_baseline.py --compare-cluster --algo HA_AVOAHGS

"""

from __future__ import annotations



import argparse

import os

import sys



REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WNMF_DIR = os.path.join(REPO, 'wnmf')

sys.path.insert(0, REPO)

sys.path.insert(0, WNMF_DIR)



from wnmf_experiment import (  # noqa: E402

    DATA_100K_ALL,

    DATA_100K_TEST,

    DATA_100K_TRAIN,

    RANDOM_SEED,

    _ASSIGN_KMREF_SUFFIX,

    run_cluster_knn,

    run_global_surprise_knn_baseline,

    run_global_surprise_knn_with_means,

)

from wnmf_utils import (  # noqa: E402

    load_assignment,

    load_memberships,

    load_ratings_100k,

    load_ratings_100k_all,

)



_COMPARE_COLS = (

    ('mae', 'MAE'),

    ('rmse', 'RMSE'),

    ('precision_at_10', 'P@10'),

    ('recall_at_10', 'R@10'),

    ('ndcg_at_10', 'NDCG'),

)





def _load_split(dataset: str, eval_split: str, fold):

    if dataset == '100k':

        if eval_split == 'random':

            return load_ratings_100k_all(DATA_100K_ALL, random_seed=RANDOM_SEED, fold=fold)

        if fold is None:

            return load_ratings_100k(DATA_100K_TRAIN, DATA_100K_TEST)

        data_dir = os.path.join(REPO, 'data', 'ml-100k')

        return load_ratings_100k(

            os.path.join(data_dir, f'u{fold}.base'),

            os.path.join(data_dir, f'u{fold}.test'),

            fold=1,

        )

    raise ValueError('Yalnız --dataset 100k destekleniyor')





def _resolve_assign_dir(algo: str, suffix: str, assign_root: str) -> str:

    ds = 'ml100k'

    assign_dir = os.path.join(REPO, assign_root, ds, f'{algo}{suffix}')

    kmref = assign_dir + _ASSIGN_KMREF_SUFFIX

    if os.path.isdir(kmref):

        return kmref

    return assign_dir





def _fmt(v) -> str:

    if v is None or (isinstance(v, float) and v != v):

        return '   —   '

    return f'{float(v):8.4f}'





def _print_metrics_table(rows: list[tuple[str, dict]]) -> None:

    name_w = max(22, max(len(name) for name, _ in rows))

    header = f"{'Kaynak':<{name_w}}"

    for _, label in _COMPARE_COLS:

        header += f'{label:>9}'

    print(header)

    print('-' * len(header))

    for name, row in rows:

        line = f"{name:<{name_w}}"

        for key, _ in _COMPARE_COLS:

            line += _fmt(row.get(key))

        print(line)





def main() -> None:

    p = argparse.ArgumentParser(description='Surprise global KNN baseline')

    p.add_argument('--dataset', choices=['100k'], default='100k')

    p.add_argument('--eval-split', choices=['official', 'random'], default='official')

    p.add_argument('--fold', type=int, default=None)

    p.add_argument(

        '--variant',

        choices=['baseline', 'withmeans'],

        default='baseline',

        help='baseline=KNNBaseline (varsayılan), withmeans=KNNWithMeans',

    )

    p.add_argument('--k', type=int, default=40)

    p.add_argument(

        '--sim',

        default=None,

        help='Sim: baseline için pearson_baseline (vars.), withmeans için msd',

    )

    p.add_argument('--min-common', type=int, default=1)

    p.add_argument('--compare-cluster', action='store_true')

    p.add_argument('--algo', default='HA_AVOAHGS')

    p.add_argument('--assign-root', default=os.path.join('mealpy', 'results', 'assignments_lof'))

    p.add_argument('--assign-suffix', default='_pruneu5_i10_zscore_euc_imkpp_none_wnmf30_k7')

    args = p.parse_args()



    if args.sim is None:

        args.sim = 'pearson_baseline' if args.variant == 'baseline' else 'msd'



    train, test = _load_split(args.dataset, args.eval_split, args.fold)

    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1

    print(f'train={len(train):,}  test={len(test):,}  split={args.eval_split}  fold={args.fold}')



    if args.variant == 'baseline':

        print('\n=== GLOBAL Surprise KNNBaseline (kümeleme yok) ===')

        global_row = run_global_surprise_knn_baseline(

            train, test, n_items,

            k_neighbors=args.k,

            similarity=args.sim,

            min_common=args.min_common,

        )

        global_label = 'Global KNNBaseline'

    else:

        print('\n=== GLOBAL Surprise KNNWithMeans (kümeleme yok) ===')

        global_row = run_global_surprise_knn_with_means(

            train, test, n_items,

            k_neighbors=args.k,

            similarity=args.sim,

            min_common=args.min_common,

        )

        global_label = 'Global KNNWithMeans'



    if global_row is None:

        raise SystemExit(1)



    if not args.compare_cluster:

        _print_metrics_table([(global_label, global_row)])

        return



    print('\n=== CLUSTER KNNBaseline (AVOA+HHO atama, expand-knn yok) ===')

    assign_dir = _resolve_assign_dir(args.algo, args.assign_suffix, args.assign_root)

    if not os.path.isdir(assign_dir):

        raise FileNotFoundError(f'Assignment yok: {assign_dir}')



    assignments, gray_mask = load_assignment(assign_dir)

    memberships = load_memberships(assign_dir)

    cluster_sim = 'pearson' if args.sim in ('msd', 'pearson_baseline') else args.sim

    cluster_variant = args.variant if args.compare_cluster else 'baseline'

    cluster_row = run_cluster_knn(

        train, test, assignments, gray_mask, memberships,

        n_items, args.algo,

        similarity=cluster_sim,

        min_common=max(3, args.min_common),

        k_neighbors=args.k,

        surprise_knn_variant=cluster_variant,

    )

    cluster_label = (

        'Cluster KNNBaseline' if cluster_variant == 'baseline' else 'Cluster KNNWithMeans'

    )

    print()

    _print_metrics_table([

        (global_label, global_row),

        (cluster_label, cluster_row),

    ])

    delta = cluster_row['mae'] - global_row['mae']

    verdict = 'kümeleme iyilestirdi' if delta < 0 else 'global daha iyi'

    print(f"\nCluster - Global MAE farki: {delta:+.4f}  ({verdict})")





if __name__ == '__main__':

    main()


