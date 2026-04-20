"""
ga_hho_optimizer.py
===================
Paralel GA+HHO Hibrit Optimizer — mealpy 3.x uyumlu

Mimari:
    Her epoch'ta popülasyon iki eşit gruba ayrılır:
      - Alt yarı  → GA operatörleri (tournament selection + uniform crossover + Gaussian mutation)
      - Üst yarı  → HHO operatörleri (rabbit chase + Lévy flight)

    İki grup bağımsız güncellenir, sonra birleştirilip en iyi pop_size birey seçilir.
    g_best her grup tarafından paylaşılır: HHO'nun rabbit pozisyonu = global en iyi.

Kullanım (generate_assignments.py ALGO_CONFIG'e eklemek için):
    from ga_hho_optimizer import OriginalGAHHO

    ALGO_CONFIG = [
        ...
        ('H5_GA+HHO', 'GAHHO.OriginalGAHHO', None),   # None → run_single (hibrit kendi içinde paralel)
        ...
    ]

    # algo_map'e manuel kayıt:
    algo_map['GAHHO.OriginalGAHHO'] = {
        'full_name': 'GAHHO.OriginalGAHHO',
        'class':     OriginalGAHHO,
    }

Parametreler:
    epoch       : toplam iterasyon sayısı (önerilen: 50 — mevcut baseline ile eşit)
    pop_size    : toplam popülasyon boyutu (önerilen: 30)
    pc          : crossover olasılığı (önerilen: 0.8)
    pm          : mutasyon olasılığı (önerilen: 0.1)
    tournament_k: GA tournament büyüklüğü (önerilen: 3)
"""

import numpy as np
from mealpy import Optimizer


class OriginalGAHHO(Optimizer):
    """
    Paralel GA + HHO Hibrit Optimizer.

    pop_size bireyin yarısı GA, yarısı HHO ile güncellenir.
    Her iki grup da g_best'i paylaşır.
    """

    def __init__(self, epoch=50, pop_size=30,
                 pc=0.8, pm=0.1, tournament_k=3, **kwargs):
        super().__init__(**kwargs)
        self.epoch        = self.validator.check_int("epoch",        epoch,        [1, 100_000])
        self.pop_size     = self.validator.check_int("pop_size",     pop_size,     [10, 10_000])
        self.pc           = self.validator.check_float("pc",         pc,           [0.1, 1.0])
        self.pm           = self.validator.check_float("pm",         pm,           [0.001, 1.0])
        self.tournament_k = self.validator.check_int("tournament_k", tournament_k, [2, 10])

        self.is_parallelizable = False   # mealpy iç paralel hesabını kapat
        self.set_parameters(["epoch", "pop_size", "pc", "pm", "tournament_k"])
        self.nfe_per_epoch = self.pop_size
        self.sort_flag     = False

    # ------------------------------------------------------------------
    # GA YARDIMCI METODlar
    # ------------------------------------------------------------------

    def _tournament_select(self, pop):
        """
        Tournament selection: k birey seç, en iyisini döndür.
        """
        k       = min(self.tournament_k, len(pop))
        indices = self.generator.choice(len(pop), k, replace=False)
        best_i  = min(indices, key=lambda i: pop[i].target.fitness)
        return pop[best_i]

    def _uniform_crossover(self, p1_sol, p2_sol, lb, ub):
        """
        Uniform crossover: her boyut için 50/50 olasılıkla ebeveyn seç.
        """
        mask  = self.generator.random(len(p1_sol)) < 0.5
        child = np.where(mask, p1_sol, p2_sol)
        return np.clip(child, lb, ub)

    def _gaussian_mutation(self, sol, lb, ub):
        """
        Gaussian mutation: pm olasılığıyla her boyutu perturbe et.
        Gürültü ölçeği = arama aralığının %5'i.
        """
        lb_arr     = np.array(lb)
        ub_arr     = np.array(ub)
        mask       = self.generator.random(len(sol)) < self.pm
        noise      = self.generator.normal(0, 0.05 * (ub_arr - lb_arr))
        mutated    = np.where(mask, sol + noise, sol)
        return np.clip(mutated, lb_arr, ub_arr)

    def _ga_update(self, pop, lb, ub):
        """
        Bir GA adımı: tournament → crossover → mutation → yeni birey.
        """
        p1 = self._tournament_select(pop)
        p2 = self._tournament_select(pop)

        if self.generator.random() < self.pc:
            child_sol = self._uniform_crossover(
                np.array(p1.solution), np.array(p2.solution), lb, ub
            )
        else:
            child_sol = np.array(p1.solution).copy()

        child_sol = self._gaussian_mutation(child_sol, lb, ub)
        return self.generate_agent(child_sol)

    # ------------------------------------------------------------------
    # HHO YARDIMCI METODlar
    # ------------------------------------------------------------------

    def _levy_flight(self, dim):
        """
        Lévy uçuşu: Mantegna algoritması ile.
        """
        beta  = 1.5
        sigma = (
            np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
            / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = self.generator.normal(0, sigma, dim)
        v = self.generator.normal(0, 1,     dim)
        return u / (np.abs(v) ** (1 / beta))

    def _hho_update(self, agent, rabbit_sol, epoch, lb, ub):
        """
        HHO güncelleme adımı.
        Kaçış enerjisi E ile 4 faz: soft/hard besiege + Lévy.
        """
        lb_arr     = np.array(lb)
        ub_arr     = np.array(ub)
        dim        = len(agent.solution)
        pos        = np.array(agent.solution)
        rabbit_pos = np.array(rabbit_sol)

        # Kaçış enerjisi
        E0 = 2 * self.generator.random() - 1
        E  = 2 * E0 * (1 - epoch / self.epoch)

        r1, r2 = self.generator.random(), self.generator.random()
        J      = 2 * (1 - r2) * rabbit_pos

        if abs(E) >= 1:
            # Keşif: rastgele tavşan yaklaşımı
            rand_idx = self.generator.integers(0, len(self.pop))
            rand_pos = np.array(self.pop[rand_idx].solution)
            if r1 >= 0.5:
                new_pos = rand_pos - self.generator.random() * np.abs(
                    rand_pos - 2 * self.generator.random() * pos
                )
            else:
                new_pos = (rabbit_pos - pos.mean()) - self.generator.random() * (
                    lb_arr + self.generator.random() * (ub_arr - lb_arr)
                )
        elif abs(E) < 1:
            r3 = self.generator.random()
            if r3 >= 0.5 and abs(E) >= 0.5:
                # Soft besiege
                new_pos = rabbit_pos - E * np.abs(J - pos)
            elif r3 >= 0.5 and abs(E) < 0.5:
                # Hard besiege
                new_pos = rabbit_pos - E * np.abs(rabbit_pos - pos)
            elif r3 < 0.5 and abs(E) >= 0.5:
                # Soft besiege + Lévy
                LF      = self._levy_flight(dim)
                Y       = rabbit_pos - E * np.abs(J - pos)
                Z       = Y + self.generator.random(dim) * LF
                fit_Y   = self.problem.obj_func(np.clip(Y, lb_arr, ub_arr))
                fit_Z   = self.problem.obj_func(np.clip(Z, lb_arr, ub_arr))
                fit_cur = agent.target.fitness
                if fit_Y < fit_cur:
                    new_pos = Y
                elif fit_Z < fit_cur:
                    new_pos = Z
                else:
                    new_pos = pos
            else:
                # Hard besiege + Lévy
                LF      = self._levy_flight(dim)
                Y       = rabbit_pos - E * np.abs(J - rabbit_pos)
                Z       = Y + self.generator.random(dim) * LF
                fit_Y   = self.problem.obj_func(np.clip(Y, lb_arr, ub_arr))
                fit_Z   = self.problem.obj_func(np.clip(Z, lb_arr, ub_arr))
                fit_cur = agent.target.fitness
                if fit_Y < fit_cur:
                    new_pos = Y
                elif fit_Z < fit_cur:
                    new_pos = Z
                else:
                    new_pos = pos
        else:
            new_pos = pos

        new_pos = np.clip(new_pos, lb_arr, ub_arr)
        return self.generate_agent(new_pos)

    # ------------------------------------------------------------------
    # ANA EVRIM ADIMI
    # ------------------------------------------------------------------

    def evolve(self, epoch):
        """
        Her epoch'ta:
          - pop[0 : half]  → GA operatörleri ile güncellenir
          - pop[half : N]  → HHO operatörleri ile güncellenir
          Sonra yeni + eski birleştirilip en iyi pop_size birey seçilir.
        """
        # mealpy sürümüne/Problem iç temsilinə göre bounds bazen list dönebiliyor.
        # En uyumlu yol: önce Problem.lb/ub kullan, yoksa bounds üstünden fallback.
        if hasattr(self.problem, "lb") and hasattr(self.problem, "ub"):
            lb = np.array(self.problem.lb, dtype=np.float64)
            ub = np.array(self.problem.ub, dtype=np.float64)
        else:
            b = self.problem.bounds
            if isinstance(b, list):
                b = b[0]
            lb = np.array(b.lb, dtype=np.float64)
            ub = np.array(b.ub, dtype=np.float64)

        rabbit_sol = self.g_best.solution  # HHO için tavşan pozisyonu

        half    = self.pop_size // 2
        new_pop = []

        # ── GA yarısı ─────────────────────────────────────────
        for _ in range(half):
            child = self._ga_update(self.pop, lb, ub)
            new_pop.append(child)

        # ── HHO yarısı ────────────────────────────────────────
        for i in range(half, self.pop_size):
            child = self._hho_update(self.pop[i], rabbit_sol, epoch, lb, ub)
            new_pop.append(child)

        # ── Seçim: yeni + eski → en iyi pop_size birey ────────
        combined    = self.pop + new_pop
        sorted_pop  = sorted(combined, key=lambda a: a.target.fitness)
        self.pop    = sorted_pop[:self.pop_size]