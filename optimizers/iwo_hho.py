"""
IWO + HHO hybrid clustering optimizer.

Exploration (|E| >= 1): invasive weed colonization with fitness-proportional seeding.
Exploitation (|E| < 1): Harris Hawks besiege operators with optional rapid dive.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


class IWO_HHO_Clustering:
    """IWO seeding + HHO besiege hybrid for cluster-centroid search."""

    def __init__(
        self,
        epoch: int = 50,
        pop_size: int = 30,
        sigma_min: float = 0.001,
        sigma_max: float = 3.0,
        n: int = 3,
        min_seeds: int = 1,
        max_seeds: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        self.epoch = epoch
        self.pop_size = pop_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.n = n
        self.min_seeds = min_seeds
        self.max_seeds = max_seeds
        self.seed = seed
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: Optional[float] = None

    def _bounds(self, problem: dict) -> tuple[np.ndarray, np.ndarray]:
        bounds = problem["bounds"]
        lb = np.asarray(bounds.lb, dtype=np.float64)
        ub = np.asarray(bounds.ub, dtype=np.float64)
        return lb, ub

    def _init_population(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
        rng: np.random.Generator,
        starting_solutions: Optional[Sequence] = None,
    ) -> list[np.ndarray]:
        pop: list[np.ndarray] = []
        if starting_solutions:
            for sol in starting_solutions[: self.pop_size]:
                pop.append(np.clip(np.asarray(sol, dtype=np.float64), lb, ub))
        while len(pop) < self.pop_size:
            pop.append(rng.uniform(lb, ub))
        return pop[: self.pop_size]

    def _seed_count(self, fit_i: float, fits: list[float]) -> int:
        f_min = min(fits)
        f_max = max(fits)
        if abs(f_max - f_min) < 1e-12:
            return self.min_seeds
        n_seeds = int(
            self.min_seeds
            + (self.max_seeds - self.min_seeds)
            * (fit_i - f_max)
            / (f_min - f_max)
        )
        return max(self.min_seeds, min(self.max_seeds, n_seeds))

    def solve(
        self,
        problem: dict,
        starting_solutions: Optional[Sequence] = None,
    ) -> tuple[np.ndarray, float]:
        obj_fn = problem["obj_func"]
        lb, ub = self._bounds(problem)
        dim = len(lb)
        rng = np.random.default_rng(self.seed)

        pop = self._init_population(lb, ub, rng, starting_solutions)
        fits = [float(obj_fn(p)) for p in pop]
        best_idx = int(np.argmin(fits))
        best_sol = pop[best_idx].copy()
        best_fit = fits[best_idx]

        print(f"[IWO-HHO] Baslangic fitness: {best_fit:.4f}")

        for t in range(self.epoch):
            e0 = 2 * rng.random() - 1
            e = 2 * e0 * (1 - t / self.epoch)
            sigma = ((self.epoch - t) / self.epoch) ** self.n * (
                self.sigma_max - self.sigma_min
            ) + self.sigma_min

            new_pop: list[np.ndarray] = []
            new_fits: list[float] = []

            for i, agent in enumerate(pop):
                if abs(e) >= 1:
                    n_seeds = self._seed_count(fits[i], fits)
                    for _ in range(n_seeds):
                        seed = agent + rng.normal(0, sigma, dim)
                        seed = np.clip(seed, lb, ub)
                        seed_fit = float(obj_fn(seed))
                        new_pop.append(seed)
                        new_fits.append(seed_fit)
                else:
                    j = 2 * (1 - rng.random())
                    r = rng.random()

                    if r >= 0.5 and abs(e) >= 0.5:
                        delta = best_sol - e * np.abs(j * best_sol - agent)
                        new_agent = delta - e * np.abs(delta - agent)
                    elif r >= 0.5 and abs(e) < 0.5:
                        new_agent = best_sol - e * np.abs(best_sol - agent)
                    elif r < 0.5 and abs(e) >= 0.5:
                        y = best_sol - e * np.abs(j * best_sol - agent)
                        z = y + rng.normal(0, 1, dim) * sigma
                        new_agent = y if obj_fn(y) < obj_fn(z) else z
                    else:
                        y = best_sol - e * np.abs(j * best_sol - agent)
                        z = y + rng.normal(0, 1, dim) * sigma
                        new_agent = y if obj_fn(y) < obj_fn(z) else z

                    new_agent = np.clip(new_agent, lb, ub)
                    new_fit = float(obj_fn(new_agent))
                    new_pop.append(new_agent)
                    new_fits.append(new_fit)

            all_pop = pop + new_pop
            all_fits = fits + new_fits
            sorted_idx = np.argsort(all_fits)[: self.pop_size]
            pop = [all_pop[i] for i in sorted_idx]
            fits = [all_fits[i] for i in sorted_idx]

            if fits[0] < best_fit:
                best_fit = fits[0]
                best_sol = pop[0].copy()

            if (t + 1) % 10 == 0 or t == 0:
                print(
                    f"[IWO-HHO] iter {t + 1:4d}/{self.epoch} "
                    f"fitness: {best_fit:.4f} |E|={abs(e):.3f}"
                )

        print(f"[IWO-HHO] Final fitness: {best_fit:.4f}")
        self.best_solution = best_sol
        self.best_fitness = best_fit
        self.last_population = [p.copy() for p in pop]
        return best_sol, best_fit

    def optimize(self, X: np.ndarray, k: int) -> np.ndarray:
        """LF-HHO / SFOA ile uyumlu: euclidean WCSS ile merkez optimizasyonu."""
        n, dim = X.shape
        sol_dim = k * dim
        lb = X.min(axis=0).repeat(k)
        ub = X.max(axis=0).repeat(k)

        def _wcss(ind: np.ndarray) -> float:
            centers = ind.reshape(k, dim)
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            wcss = 0.0
            for c in range(k):
                mask = labels == c
                if mask.sum() > 0:
                    wcss += float(np.sum((X[mask] - centers[c]) ** 2))
            return wcss

        bounds = type("Bounds", (), {"lb": lb, "ub": ub})()

        problem: dict[str, Any] = {
            "obj_func": _wcss,
            "bounds": bounds,
        }
        best_sol, _ = self.solve(problem)
        return best_sol.reshape(k, dim)
