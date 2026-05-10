"""
WNMF Parametre Optimizasyonu
Meta-heuristic algoritma ile WNMF hiperparametrelerini optimize eder.
Fitness: validation MAE (minimize)

Optimize edilen parametreler:
    lr          : U,V learning rate       [0.001, 0.05]
    reg         : U,V regularization      [0.001, 0.05]
    latent_dim  : latent boyut            [10, 40]
    n_epochs    : epoch sayısı            [50, 200]
    bias_lr     : b_u,b_i learning rate   [0.01, 0.3]
    bias_reg    : b_u,b_i regularization  [0.0001, 0.02]
"""

import numpy as np
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WNMF = os.path.join(_ROOT, 'wnmf')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _WNMF not in sys.path:
    sys.path.insert(0, _WNMF)

from mealpy import FloatVar, Problem

from wnmf_model import WNMFModel


class WNMFParamProblem(Problem):
    def __init__(self, optimizer_instance, bounds, minmax):
        self.opt = optimizer_instance
        super().__init__(bounds=bounds, minmax=minmax)

    def obj_func(self, x):
        return self.opt.fitness(x)


class WNMFParamOptimizer:
    """
    WNMF parametrelerini DOA benzeri meta-heuristic ile optimize eder.
    Basit PSO kullanır (hız için) — mealpy bağımsız.
    """

    BOUNDS = [
        (0.001, 0.05),   # lr
        (0.001, 0.05),   # reg
        (10,    40),     # latent_dim (int)
        (50,    200),    # n_epochs   (int)
        (0.01,  0.3),    # bias_lr
        (0.0001, 0.02),  # bias_reg
    ]
    PARAM_NAMES = ['lr', 'reg', 'latent_dim', 'n_epochs', 'bias_lr', 'bias_reg']

    def __init__(self, train, val, n_agents=20, n_iter=30, seed=42):
        self.train    = train
        self.val      = val
        self.n_agents = n_agents
        self.n_iter   = n_iter
        self.rng      = np.random.default_rng(seed)
        self.n_users  = int(train[:, 0].max()) + 1
        self.n_items  = int(max(train[:, 1].max(), val[:, 1].max())) + 1

    def _decode(self, ind):
        """Vektörü parametre dict'e çevir."""
        return {
            'lr'        : float(np.clip(ind[0], *self.BOUNDS[0])),
            'reg'       : float(np.clip(ind[1], *self.BOUNDS[1])),
            'latent_dim': int(np.clip(round(ind[2]), *self.BOUNDS[2])),
            'n_epochs'  : int(np.clip(round(ind[3]), *self.BOUNDS[3])),
            'bias_lr'   : float(np.clip(ind[4], *self.BOUNDS[4])),
            'bias_reg'  : float(np.clip(ind[5], *self.BOUNDS[5])),
        }

    def fitness(self, ind):
        """MAE hesapla — düşük = iyi."""
        p = self._decode(ind)
        try:
            model = WNMFModel(
                n_users        = self.n_users,
                n_items        = self.n_items,
                latent_dim     = p['latent_dim'],
                learning_rate  = p['lr'],
                regularization = p['reg'],
                n_epochs       = p['n_epochs'],
                random_seed    = 42,
                use_bias       = True,
                bias_lr        = p['bias_lr'],
                bias_reg       = p['bias_reg'],
            )
            model.fit(self.train)
            mae, _ = model.evaluate(self.val)
            return float(mae)
        except Exception:
            return 9999.0

    def _init_population(self):
        pop = []
        for i, (lo, hi) in enumerate(self.BOUNDS):
            col = self.rng.uniform(lo, hi, self.n_agents)
            pop.append(col)
        return np.array(pop).T  # (n_agents, 6)

    def optimize(self, algo='GOA'):
        bounds = [FloatVar(lb=lo, ub=hi) for lo, hi in self.BOUNDS]

        problem = WNMFParamProblem(self, bounds, minmax='min')

        if algo == 'GOA':
            from mealpy.swarm_based.GOA import OriginalGOA
            model = OriginalGOA(epoch=self.n_iter, pop_size=self.n_agents)
        elif algo == 'HHO':
            from mealpy.swarm_based.HHO import OriginalHHO
            model = OriginalHHO(epoch=self.n_iter, pop_size=self.n_agents)
        elif algo == 'MFO':
            from mealpy.swarm_based.MFO import OriginalMFO
            model = OriginalMFO(epoch=self.n_iter, pop_size=self.n_agents)
        elif algo == 'HGS':
            from mealpy.swarm_based.HGS import OriginalHGS
            model = OriginalHGS(epoch=self.n_iter, pop_size=self.n_agents)
        else:
            raise ValueError(
                f"Bilinmeyen algo: {algo!r}; GOA, HHO, MFO veya HGS kullanın."
            )

        model.solve(problem, seed=42)
        gbest = float(model.g_best.target.fitness)
        best_params = self._decode(np.asarray(model.g_best.solution, dtype=np.float64))
        return best_params, gbest


def run_wnmf_param_opt(dataset='100k', n_agents=20, n_iter=30, algo='GOA'):
    """
    Kolay çalıştırma fonksiyonu.

    Kullanım:
        python mealpy/wnmf_param_optimizer.py --dataset 100k --agents 20 --iter 30 --algo HHO
    """
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _wnmf = os.path.join(_root, 'wnmf')
    if _root not in sys.path:
        sys.path.insert(0, _root)
    if _wnmf not in sys.path:
        sys.path.insert(0, _wnmf)
    from wnmf.wnmf_utils import load_ratings_100k, load_ratings_1m

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if dataset == '100k':
        train, test = load_ratings_100k(
            os.path.join(BASE, 'data/ml-100k/u1.base'),
            os.path.join(BASE, 'data/ml-100k/u1.test'),
        )
    else:
        train, test = load_ratings_1m(
            os.path.join(BASE, 'data/ml-1m/ratings.dat'),
            random_seed=42,
        )

    # %80 train, %20 validation
    idx   = np.random.default_rng(42).permutation(len(train))
    split = int(len(train) * 0.8)
    tr    = train[idx[:split]]
    val   = train[idx[split:]]

    print(f"Train: {len(tr)}  Val: {len(val)}  Test: {len(test)}")

    opt = WNMFParamOptimizer(tr, val, n_agents=n_agents, n_iter=n_iter)
    best_params, best_mae = opt.optimize(algo=algo)

    # Test seti üzerinde doğrula
    print("\n[WNMF-OPT] Test seti doğrulaması...")
    n_users = int(train[:, 0].max()) + 1
    n_items = int(max(train[:, 1].max(), test[:, 1].max())) + 1
    model = WNMFModel(
        n_users        = n_users,
        n_items        = n_items,
        latent_dim     = best_params['latent_dim'],
        learning_rate  = best_params['lr'],
        regularization = best_params['reg'],
        n_epochs       = best_params['n_epochs'],
        random_seed    = 42,
        use_bias       = True,
        bias_lr        = best_params['bias_lr'],
        bias_reg       = best_params['bias_reg'],
    )
    model.fit(train)
    mae_test, rmse_test = model.evaluate(test)
    print(f"Test MAE : {mae_test:.4f}")
    print(f"Test RMSE: {rmse_test:.4f}")

    return best_params, mae_test


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='100k', choices=['100k', '1m'])
    p.add_argument('--agents',  type=int, default=20)
    p.add_argument('--iter',    type=int, default=30)
    p.add_argument(
        '--algo', default='GOA', choices=['GOA', 'HHO', 'MFO', 'HGS'],
        help='mealpy meta-sezgisel (optimize içinde kullanılır)',
    )
    args = p.parse_args()
    run_wnmf_param_opt(args.dataset, args.agents, args.iter, algo=args.algo)
