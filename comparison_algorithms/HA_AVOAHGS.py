#!/usr/bin/env python
"""HA-AVOAHGS implemented from Optimizer base class."""

from __future__ import annotations

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class HA_AVOAHGS(Optimizer):
    def __init__(
        self,
        epoch: int = 500,
        pop_size: int = 30,
        p1: float = 0.6,
        p2: float = 0.4,
        p3: float = 0.6,
        alpha: float = 0.8,
        gama: float = 2.5,
        PUP: float = 0.08,
        LH: float = 10000,
        hgs_rate: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.p1 = self.validator.check_float("p1", p1, (0, 1))
        self.p2 = self.validator.check_float("p2", p2, (0, 1))
        self.p3 = self.validator.check_float("p3", p3, (0, 1))
        self.alpha = self.validator.check_float("alpha", alpha, (0, 1))
        self.gama = self.validator.check_float("gama", gama, (0, 5.0))
        self.PUP = self.validator.check_float("PUP", PUP, (0, 1.0))
        self.LH = self.validator.check_float("LH", LH, [1, 20000])
        self.hgs_rate = self.validator.check_float("hgs_rate", hgs_rate, (0, 1.0))
        self.set_parameters(
            ["epoch", "pop_size", "p1", "p2", "p3", "alpha", "gama", "PUP", "LH", "hgs_rate"]
        )
        self.sort_flag = False
        self.is_parallelizable = False
        self._hgs_apply_count = 0
        self._hgs_accept_count = 0

    def _avoa_full_evolve(self, epoch: int) -> None:
        a = self.generator.uniform(-2, 2) * (
            (np.sin((np.pi / 2) * (epoch / self.epoch)) ** self.gama)
            + np.cos((np.pi / 2) * (epoch / self.epoch))
            - 1
        )
        ppp = (2 * self.generator.random() + 1) * (1 - epoch / self.epoch) + a
        _, best_list, _ = self.get_special_agents(self.pop, n_best=2, minmax=self.problem.minmax)
        pop_new = []
        for idx in range(self.pop_size):
            F = ppp * (2 * self.generator.random() - 1)
            rand_idx = self.generator.choice([0, 1], p=[self.alpha, 1 - self.alpha])
            rand_pos = best_list[rand_idx].solution
            if np.abs(F) >= 1:
                if self.generator.random() < self.p1:
                    pos_new = rand_pos - (
                        np.abs((2 * self.generator.random()) * rand_pos - self.pop[idx].solution)
                    ) * F
                else:
                    pos_new = rand_pos - F + self.generator.random() * (
                        (self.problem.ub - self.problem.lb) * self.generator.random() + self.problem.lb
                    )
            else:
                if np.abs(F) < 0.5:
                    best_x1 = best_list[0].solution
                    best_x2 = best_list[1].solution
                    if self.generator.random() < self.p2:
                        A = best_x1 - (
                            (best_x1 * self.pop[idx].solution)
                            / (best_x1 - self.pop[idx].solution**2 + self.EPSILON)
                        ) * F
                        B = best_x2 - (
                            (best_x2 * self.pop[idx].solution)
                            / (best_x2 - self.pop[idx].solution**2 + self.EPSILON)
                        ) * F
                        pos_new = (A + B) / 2
                    else:
                        pos_new = rand_pos - np.abs(rand_pos - self.pop[idx].solution) * F * self.get_levy_flight_step(
                            beta=1.5, multiplier=1.0, size=self.problem.n_dims, case=-1
                        )
                else:
                    if self.generator.random() < self.p3:
                        pos_new = np.abs(
                            (2 * self.generator.random()) * rand_pos - self.pop[idx].solution
                        ) * (F + self.generator.random()) - (rand_pos - self.pop[idx].solution)
                    else:
                        s1 = rand_pos * (
                            self.generator.random() * self.pop[idx].solution / (2 * np.pi)
                        ) * np.cos(self.pop[idx].solution)
                        s2 = rand_pos * (
                            self.generator.random() * self.pop[idx].solution / (2 * np.pi)
                        ) * np.sin(self.pop[idx].solution)
                        pos_new = rand_pos - (s1 + s2)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        self.pop = self.update_target_for_population(pop_new)

    def _sech(self, x: float) -> float:
        if np.abs(x) > 50:
            return 0.5
        return 2.0 / (np.exp(x) + np.exp(-x))

    def _update_hunger(self, g_best: Agent, g_worst: Agent) -> None:
        space = float(np.mean(self.problem.ub - self.problem.lb))
        fit_range = g_worst.target.fitness - g_best.target.fitness + self.EPSILON
        for agent in self.pop:
            if (not hasattr(agent, "hunger")) or (agent.hunger is None):
                agent.hunger = 1.0
            r = self.generator.random()
            H = (agent.target.fitness - g_best.target.fitness) / fit_range * r * 2 * space
            if H < self.LH:
                H = self.LH * (1 + r)
            agent.hunger += H
            if g_best.target.fitness == agent.target.fitness:
                agent.hunger = 0.0

    def _hgs_step(self, idx: int, g_best: Agent, total_hunger: float, shrink: float) -> np.ndarray:
        agent = self.pop[idx]
        if (not hasattr(agent, "hunger")) or (agent.hunger is None):
            agent.hunger = 1.0
        E = self._sech(agent.target.fitness - g_best.target.fitness)
        R = 2 * shrink * self.generator.random() - shrink
        r1 = self.generator.random()
        r2 = self.generator.random()
        if r1 < self.PUP:
            W1 = agent.hunger * self.pop_size / (total_hunger + self.EPSILON) * self.generator.random()
        else:
            W1 = 1.0
        W2 = (1 - np.exp(-np.abs(agent.hunger - total_hunger))) * self.generator.random() * 2
        if r1 < self.PUP:
            pos_new = agent.solution * (1 + self.generator.normal(0, 1))
        else:
            diff = np.abs(g_best.solution - agent.solution)
            if r2 > E:
                pos_new = W1 * g_best.solution + R * W2 * diff
            else:
                pos_new = W1 * g_best.solution - R * W2 * diff
        return pos_new

    def evolve(self, epoch: int) -> None:
        self._avoa_full_evolve(epoch)
        _, (g_best,), (g_worst,) = self.get_special_agents(
            self.pop, n_best=1, n_worst=1, minmax=self.problem.minmax
        )
        self._update_hunger(g_best, g_worst)
        total_hunger = sum(
            (1.0 if getattr(a, "hunger", None) is None else getattr(a, "hunger", 1.0))
            for a in self.pop
        )
        shrink = 2.0 * (1.0 - epoch / self.epoch)
        n_apply = max(1, int(self.hgs_rate * self.pop_size))
        sorted_indices = sorted(
            range(self.pop_size),
            key=lambda i: self.pop[i].target.fitness,
            reverse=(self.problem.minmax == "max"),
        )
        best_indices = sorted_indices[:n_apply]
        for idx in best_indices:
            self._hgs_apply_count += 1
            pos_new = self._hgs_step(idx, g_best, total_hunger, shrink)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
                self._hgs_accept_count += 1
