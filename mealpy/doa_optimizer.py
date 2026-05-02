"""
Dhole Optimization Algorithm (DOA) implementations for this project.

- `OriginalDOA`: mealpy-compatible optimizer class for metaheuristic runs.
- `DOA_Clustering`: standalone numpy helper for direct clustering usage.
"""

from copy import deepcopy

import numpy as np

try:
    from mealpy import Optimizer
except ImportError:  # pragma: no cover
    Optimizer = None


if Optimizer is not None:
    class OriginalDOA(Optimizer):
        """
        Dhole Optimization Algorithm (mealpy 3.x compatible).
        """

        def __init__(self, epoch=100, pop_size=30, c3=2.0, **kwargs):
            super().__init__(**kwargs)
            self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
            self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
            self.c3 = self.validator.check_float("c3", c3, [1.0, 4.0])
            self.set_parameters(["epoch", "pop_size", "c3"])
            self.nfe_per_epoch = self.pop_size
            self.sort_flag = False
            self._global_best_pos = None
            self._prev_best_target = None

        def initialize_variables(self):
            self._global_best_pos = None
            self._prev_best_target = None

        def evolve(self, epoch):
            t = epoch + 1
            T = self.epoch
            n = len(self.pop)

            if self._global_best_pos is None:
                self._global_best_pos = np.asarray(self.g_best.solution, dtype=np.float64).copy()
            elif self._prev_best_target is not None and self.compare_target(
                self.g_best.target, self._prev_best_target, self.problem.minmax
            ):
                self._global_best_pos = np.asarray(self.g_best.solution, dtype=np.float64).copy()

            self._prev_best_target = deepcopy(self.g_best.target)

            prey_local = np.asarray(self.g_best.solution, dtype=np.float64)
            prey_global = np.asarray(self._global_best_pos, dtype=np.float64)
            prey = (prey_local + prey_global) / 2.0

            pmn = n
            c2 = 1.0 - (t / T)
            ps = 1.0 / (1.0 + np.exp(-((t / T) - 0.5) * 10.0))
            s_energy = self.c3 * self.generator.random()

            pop_new = []
            for i in range(n):
                pos_old = np.asarray(self.pop[i].solution, dtype=np.float64)
                vocal = self.generator.random()

                if vocal < 0.5:  # Exploration
                    if pmn < 10:
                        pos_new = pos_old + c2 * self.generator.random() * (prey - pos_old)
                    else:
                        z = self.generator.integers(0, n)
                        pos_z = np.asarray(self.pop[z].solution, dtype=np.float64)
                        pos_new = pos_old - pos_z + prey
                else:  # Exploitation
                    if s_energy > 2.0:
                        pos_new = (
                            prey_global
                            + c2 * self.generator.random() * (prey - pos_old)
                            + (1.0 - c2) * self.generator.random() * (pos_old - prey)
                        )
                    else:
                        pos_new = ((pos_old - prey_global) * ps) + (ps * self.generator.random() * pos_old)

                pos_new = self.correct_solution(pos_new)
                pop_new.append(self.generate_agent(pos_new))
            self.pop = self.greedy_selection_population(self.pop, pop_new)


else:
    class OriginalDOA:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError("mealpy is required to use OriginalDOA")


class DOA_Clustering:
    """
    Standalone DOA-based centroid optimizer.
    """

    def __init__(self, n_agents=30, n_iter=100, k=10, dim=None, c3=2.0, seed=None):
        self.n_agents = n_agents
        self.n_iter = n_iter
        self.k = k
        self.dim = dim
        self.c3 = c3
        self.rng = np.random.default_rng(seed)

    def _decode(self, individual):
        return individual.reshape(self.k, self.dim)

    def _clip(self, individual, lb, ub):
        return np.clip(individual, lb, ub)

    def _wcss(self, X, centers):
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        wcss = 0.0
        for c in range(self.k):
            mask = labels == c
            if np.any(mask):
                wcss += float(np.sum((X[mask] - centers[c]) ** 2))
        return wcss

    def _init_population(self, X):
        pop = []
        for _ in range(self.n_agents):
            idx = self.rng.choice(len(X), size=self.k, replace=False)
            pop.append(X[idx].flatten().copy())
        return np.asarray(pop)

    def optimize(self, X):
        n_users, dim = X.shape
        self.dim = dim
        lb = X.min(axis=0).repeat(self.k)
        ub = X.max(axis=0).repeat(self.k)

        pop = self._init_population(X)
        fitness = np.asarray([self._wcss(X, self._decode(p)) for p in pop])
        best_idx = int(np.argmin(fitness))
        global_best = pop[best_idx].copy()
        global_best_fit = float(fitness[best_idx])

        for t in range(1, self.n_iter + 1):
            local_best = pop[int(np.argmin(fitness))].copy()
            prey = (local_best + global_best) / 2.0
            pmn = self.n_agents
            c2 = 1.0 - (t / self.n_iter)
            ps = 1.0 / (1.0 + np.exp(-((t / self.n_iter) - 0.5) * 10.0))
            s_energy = self.c3 * self.rng.random()

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(self.n_agents):
                pos_old = pop[i].copy()
                vocal = self.rng.random()

                if vocal < 0.5:
                    if pmn < 10:
                        pos_new = pos_old + c2 * self.rng.random() * (prey - pos_old)
                    else:
                        z = self.rng.integers(0, self.n_agents)
                        pos_new = pos_old - pop[z] + prey
                else:
                    if s_energy > 2.0:
                        pos_new = (
                            global_best
                            + c2 * self.rng.random() * (prey - pos_old)
                            + (1.0 - c2) * self.rng.random() * (pos_old - prey)
                        )
                    else:
                        pos_new = ((pos_old - global_best) * ps) + (ps * self.rng.random() * pos_old)

                pos_new = self._clip(pos_new, lb, ub)
                f_new = self._wcss(X, self._decode(pos_new))
                if f_new < fitness[i]:
                    new_pop[i] = pos_new
                    new_fitness[i] = f_new

            pop = new_pop
            fitness = new_fitness
            best_idx = int(np.argmin(fitness))
            if float(fitness[best_idx]) < global_best_fit:
                global_best_fit = float(fitness[best_idx])
                global_best = pop[best_idx].copy()

        return self._decode(global_best)

    def assign(self, X, centers):
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        return np.argmin(dists, axis=1)
