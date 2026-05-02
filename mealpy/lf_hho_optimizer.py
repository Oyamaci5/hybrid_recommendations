"""
Levy Flight Harris Hawks Optimization (LF-HHO)
==============================================
HHO uzerine Levy flight perturbasyonu eklenmis kukume merkezi optimizasyonu.
"""

import numpy as np
from math import gamma, pi, sin


def levy_flight(dim, beta=1.5):
    """Mantegna algoritmasi ile Levy adimi uret."""
    num = gamma(1 + beta) * sin(pi * beta / 2)
    denom = gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma = (num / denom) ** (1 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    return 0.01 * u / (np.abs(v) ** (1 / beta))


class LevyHHO_Clustering:
    """Levy Flight eklenmis HHO ile kume merkezi optimizasyonu."""

    def __init__(
        self,
        n_agents=30,
        n_iter=100,
        k=10,
        beta=1.5,
        levy_scale=1.0,
        stagnation_tol=10,
        seed=None,
    ):
        self.n_agents = n_agents
        self.n_iter = n_iter
        self.k = k
        self.beta = beta
        self.levy_scale = levy_scale
        self.stagnation_tol = stagnation_tol
        self.dim = None
        np.random.seed(seed)

    def _wcss(self, X, centers):
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        wcss = 0.0
        for c in range(self.k):
            mask = labels == c
            if mask.sum() > 0:
                wcss += np.sum((X[mask] - centers[c]) ** 2)
        return wcss

    def _decode(self, ind):
        return ind.reshape(self.k, self.dim)

    def _clip(self, ind, lb, ub):
        return np.clip(ind, lb, ub)

    def _init_population(self, X):
        pop = []
        for _ in range(self.n_agents):
            idx = np.random.choice(len(X), size=self.k, replace=False)
            pop.append(X[idx].flatten().copy())
        return np.array(pop)

    def optimize(self, X):
        n, dim = X.shape
        self.dim = dim
        sol_dim = self.k * dim
        lb = X.min(axis=0).repeat(self.k)
        ub = X.max(axis=0).repeat(self.k)

        pop = self._init_population(X)
        fitness = np.array([self._wcss(X, self._decode(p)) for p in pop])
        best_idx = np.argmin(fitness)
        rabbit_pos = pop[best_idx].copy()
        rabbit_fit = fitness[best_idx]

        stag_count = 0
        prev_best_fit = rabbit_fit
        print(f"[LF-HHO] Baslangic WCSS: {rabbit_fit:.4f}")

        for t in range(1, self.n_iter + 1):
            E0 = 2 * np.random.random() - 1
            E = 2 * E0 * (1 - t / self.n_iter)
            new_pop = pop.copy()

            for i in range(self.n_agents):
                x = pop[i].copy()
                if abs(E) >= 1:
                    q = np.random.random()
                    if q >= 0.5:
                        j = np.random.randint(0, self.n_agents)
                        x_rand = pop[j].copy()
                        lf = levy_flight(sol_dim, self.beta)
                        x_new = (
                            x_rand
                            - np.random.random() * np.abs(x_rand - 2 * np.random.random() * x)
                            + self.levy_scale * lf
                        )
                    else:
                        x_m = pop.mean(axis=0)
                        x_new = (
                            rabbit_pos
                            - x_m
                            - np.random.random() * (lb + np.random.random() * (ub - lb))
                        )
                else:
                    delta = rabbit_pos - x
                    J = 2 * (1 - np.random.random())
                    r = np.random.random()
                    if r >= 0.5 and abs(E) >= 0.5:
                        lf = levy_flight(sol_dim, self.beta)
                        x_new = delta - E * np.abs(J * rabbit_pos - x + self.levy_scale * lf)
                    elif r >= 0.5 and abs(E) < 0.5:
                        x_new = rabbit_pos - E * np.abs(delta)
                    elif r < 0.5 and abs(E) >= 0.5:
                        lf = levy_flight(sol_dim, self.beta)
                        y = rabbit_pos - E * np.abs(J * rabbit_pos - x)
                        z = y + self.levy_scale * lf
                        f_y = self._wcss(X, self._decode(self._clip(y, lb, ub)))
                        f_z = self._wcss(X, self._decode(self._clip(z, lb, ub)))
                        x_new = y if f_y < f_z else z
                    else:
                        lf = levy_flight(sol_dim, self.beta)
                        y = rabbit_pos - E * np.abs(J * rabbit_pos - pop.mean(axis=0))
                        z = y + self.levy_scale * lf
                        f_y = self._wcss(X, self._decode(self._clip(y, lb, ub)))
                        f_z = self._wcss(X, self._decode(self._clip(z, lb, ub)))
                        x_new = y if f_y < f_z else z

                new_pop[i] = self._clip(x_new, lb, ub)

            new_fitness = np.array([self._wcss(X, self._decode(p)) for p in new_pop])
            improved = new_fitness < fitness
            pop[improved] = new_pop[improved]
            fitness[improved] = new_fitness[improved]

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < rabbit_fit:
                rabbit_fit = fitness[best_idx]
                rabbit_pos = pop[best_idx].copy()

            if rabbit_fit >= prev_best_fit - 1e-8:
                stag_count += 1
            else:
                stag_count = 0
            prev_best_fit = rabbit_fit

            if stag_count >= self.stagnation_tol:
                lf = levy_flight(sol_dim, self.beta)
                perturbed = self._clip(rabbit_pos + self.levy_scale * 2 * lf, lb, ub)
                f_perturbed = self._wcss(X, self._decode(perturbed))
                if f_perturbed < rabbit_fit:
                    rabbit_fit = f_perturbed
                    rabbit_pos = perturbed.copy()
                    print(f"[LF-HHO] iter {t:4d} stagnation jump iyilestirdi: {rabbit_fit:.4f}")
                stag_count = 0

            if t % 10 == 0 or t == 1:
                print(f"[LF-HHO] iter {t:4d}/{self.n_iter} WCSS: {rabbit_fit:.4f} |E|={abs(E):.3f}")

        print(f"[LF-HHO] Final WCSS: {rabbit_fit:.4f}")
        return self._decode(rabbit_pos)

    def assign(self, X, centers):
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        return np.argmin(dists, axis=1)

