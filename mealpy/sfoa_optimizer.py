"""
Starfish Optimization Algorithm (SFOA) for clustering.
"""

import numpy as np
from math import pi


def _wcss(X, centers):
    dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    wcss = 0.0
    for c in range(centers.shape[0]):
        mask = labels == c
        if mask.sum() > 0:
            wcss += np.sum((X[mask] - centers[c]) ** 2)
    return wcss


class SFOA_Clustering:
    """SFOA-based cluster center optimization."""

    def __init__(self, n_agents=30, n_iter=100, k=10, Gp=0.5, seed=42):
        self.n_agents = n_agents
        self.n_iter = n_iter
        self.k = k
        self.Gp = Gp
        self.seed = seed
        self.dim = None
        self.pop = None

    def _decode(self, ind):
        return ind.reshape(self.k, self.dim)

    def _clip(self, ind, lb, ub):
        return np.clip(ind, lb, ub)

    def _init_population(self, X):
        pop = []
        for _ in range(self.n_agents):
            idx = np.random.choice(len(X), size=self.k, replace=False)
            centers = X[idx].flatten()
            pop.append(centers.copy())
        return np.array(pop)

    def _exploration(self, xi, x_best, t, sol_dim):
        D = sol_dim
        theta = pi * (1 - t / self.n_iter)
        if D > 5:
            r = np.random.random()
            x_right = xi + np.sin(theta) * (x_best - xi) * r
            x_left = xi + np.cos(theta) * (x_best - xi) * (1 - r)
            x_new = (x_right + x_left) / 2.0
        else:
            Et = np.exp(-t / self.n_iter)
            j1, j2 = np.random.choice(self.n_agents, size=2, replace=False)
            xj1 = self.pop[j1]
            xj2 = self.pop[j2]
            r1, r2 = np.random.random(), np.random.random()
            x_new = xi + Et * (r1 * (xj1 - xi) + r2 * (xj2 - xi))
        return x_new

    def _preying(self, xi, x_best):
        distances = []
        for _ in range(5):
            sign = np.random.choice([-1, 1])
            distances.append(sign * np.random.random() * (x_best - xi))
        idx1, idx2 = np.random.choice(5, size=2, replace=False)
        r1, r2 = np.random.random(), np.random.random()
        return xi + r1 * distances[idx1] + r2 * distances[idx2]

    def _regeneration(self, xi, t):
        decay = np.exp(-t * self.n_agents / self.n_iter)
        return decay * xi

    def optimize(self, X):
        np.random.seed(self.seed)
        _, dim = X.shape
        self.dim = dim
        sol_dim = self.k * dim
        lb = X.min(axis=0).repeat(self.k)
        ub = X.max(axis=0).repeat(self.k)

        self.pop = self._init_population(X)
        fitness = np.array([_wcss(X, self._decode(p)) for p in self.pop])
        best_idx = np.argmin(fitness)
        best_pos = self.pop[best_idx].copy()
        best_fit = fitness[best_idx]
        print(f"[SFOA] Initial WCSS: {best_fit:.4f}")

        for t in range(1, self.n_iter + 1):
            worst_idx = np.argmax(fitness)
            new_pop = self.pop.copy()
            new_fitness = fitness.copy()

            for i in range(self.n_agents):
                xi = self.pop[i].copy()
                if np.random.random() < self.Gp:
                    x_new = self._exploration(xi, best_pos, t, sol_dim)
                else:
                    if i == worst_idx:
                        x_new = self._regeneration(xi, t)
                    else:
                        x_new = self._preying(xi, best_pos)

                x_new = self._clip(x_new, lb, ub)
                f_new = _wcss(X, self._decode(x_new))
                if f_new < fitness[i]:
                    new_pop[i] = x_new
                    new_fitness[i] = f_new

            self.pop = new_pop
            fitness = new_fitness
            best_idx_new = np.argmin(fitness)
            if fitness[best_idx_new] < best_fit:
                best_fit = fitness[best_idx_new]
                best_pos = self.pop[best_idx_new].copy()

            if t % 10 == 0 or t == 1:
                print(f"[SFOA] iter {t:4d}/{self.n_iter} WCSS: {best_fit:.4f}")

        print(f"[SFOA] Final WCSS: {best_fit:.4f}")
        return self._decode(best_pos)

    def assign(self, X, centers):
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        return np.argmin(dists, axis=1)


try:
    from mealpy import Optimizer

    class OriginalSFOA(Optimizer):
        """Mealpy 3.x compatible SFOA."""

        def __init__(self, epoch=100, pop_size=30, Gp=0.5, **kwargs):
            super().__init__(**kwargs)
            self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
            self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
            self.Gp = self.validator.check_float("Gp", Gp, [0.1, 0.9])
            self.set_parameters(["epoch", "pop_size", "Gp"])
            self.nfe_per_epoch = self.pop_size
            self.sort_flag = False

        def evolve(self, epoch):
            t = epoch + 1
            T = self.epoch
            D = self.problem.n_dims
            rabbit = np.array(self.g_best[0])
            theta = pi * (1 - t / T)

            if self.problem.minmax == "min":
                worst_idx = max(range(len(self.pop)), key=lambda i: self.pop[i][1].fitness)
            else:
                worst_idx = min(range(len(self.pop)), key=lambda i: self.pop[i][1].fitness)

            pop_new = []
            for i, agent in enumerate(self.pop):
                xi = np.array(agent[0])
                if np.random.random() < self.Gp:
                    if D > 5:
                        r = np.random.random()
                        x_right = xi + np.sin(theta) * (rabbit - xi) * r
                        x_left = xi + np.cos(theta) * (rabbit - xi) * (1 - r)
                        x_new = (x_right + x_left) / 2.0
                    else:
                        Et = np.exp(-t / T)
                        j1, j2 = np.random.choice(len(self.pop), 2, replace=False)
                        xj1 = np.array(self.pop[j1][0])
                        xj2 = np.array(self.pop[j2][0])
                        r1, r2 = np.random.random(), np.random.random()
                        x_new = xi + Et * (r1 * (xj1 - xi) + r2 * (xj2 - xi))
                else:
                    if i == worst_idx:
                        decay = np.exp(-t * len(self.pop) / T)
                        x_new = decay * xi
                    else:
                        distances = []
                        for _ in range(5):
                            sign = np.random.choice([-1, 1])
                            distances.append(sign * np.random.random() * (rabbit - xi))
                        idx1, idx2 = np.random.choice(5, 2, replace=False)
                        r1, r2 = np.random.random(), np.random.random()
                        x_new = xi + r1 * distances[idx1] + r2 * distances[idx2]

                x_new = self.correct_solution(x_new)
                pop_new.append(self.generate_empty_agent(x_new))

            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)

except ImportError:
    OriginalSFOA = None

