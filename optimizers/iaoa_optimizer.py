"""
Improved Aquila Optimizer (IAOA) — mealpy 3.x uyumlu implementasyon
====================================================================
Kaynak:
  Ye F. et al. (2022/2023). "An Improved Aquila Optimizer Algorithm
  with Application." — Lévy uçuşu, kaotik başlatma ve lokal arama
  ile geliştirilmiş AO versiyonu.

Orijinal AO (Abualigah et al., 2021) 4 strateji kullanır:
  X1 — Genişletilmiş keşif (yüksek irtifa süzülmesi)
  X2 — Daralma keşfi (kısa dalış planlaması)
  X3 — Yoğun yararlanma (yavaş alçalma saldırısı)
  X4 — Delta yararlanma (Lévy tabanlı yürüyüş saldırısı)

IAOA iyileştirmeleri:
  1) Kaotik (Tent map) başlatma — çeşitlilik artırır
  2) Lévy uçuşu X1 stratejisine entegre — lokal optimumdan kaçış
  3) Gaussian random walk — X3 stratejisini güçlendirir
  4) Adaptive sinusoidal perturbation — son iterasyonlarda ince ayar

Kullanım:
  from iaoa_optimizer import OriginalIAOA
  model = OriginalIAOA(epoch=500, pop_size=30)
  # mealpy Problem nesnesiyle normal kullanım
"""

import numpy as np
from mealpy import Optimizer


class OriginalIAOA(Optimizer):
    """
    Improved Aquila Optimizer (IAOA)

    Parametreler
    ----------
    epoch : int
        Maksimum iterasyon sayısı (default: 500)
    pop_size : int
        Popülasyon büyüklüğü (default: 30)
    alpha : float
        Lévy uçuşu ölçek faktörü (default: 0.1)
    delta : float
        Sinüsoidal pertürbasyon oranı (default: 0.1)
        Son %delta oranındaki iterasyonlarda lokal ince ayar uygulanır
    """

    def __init__(self, epoch=500, pop_size=30, alpha=0.1, delta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.alpha = self.validator.check_float("alpha", alpha, [0.01, 1.0])
        self.delta = self.validator.check_float("delta", delta, [0.01, 0.5])
        self.set_parameters(["epoch", "pop_size", "alpha", "delta"])
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    # ------------------------------------------------------------------
    # Yardımcı fonksiyonlar
    # ------------------------------------------------------------------

    def _levy_flight(self, beta=1.5):
        """Mantegna algoritması ile Lévy uçuşu adımı üretir."""
        num = np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        den = np.math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        sigma_u = (num / den) ** (1.0 / beta)
        u = np.random.normal(0, sigma_u, size=self.problem.n_dims)
        v = np.random.normal(0, 1, size=self.problem.n_dims)
        step = u / (np.abs(v) ** (1.0 / beta))
        return step

    def _tent_map(self, x, mu=0.7):
        """Tent kaotik harita: tek adım."""
        return mu * x if x < 0.5 else mu * (1 - x)

    def _gaussian_walk(self, pos, best_pos):
        """Gaussian random walk: mevcut pozisyon ile best arasında."""
        g = np.random.normal(0, 1, size=self.problem.n_dims)
        return pos + g * np.abs(pos - best_pos)

    # ------------------------------------------------------------------
    # Kaotik başlatma (Tent map)
    # ------------------------------------------------------------------

    def initialization(self):
        """Tent map ile kaotik başlatma — standart random'ın yerini alır."""
        self.pop = []
        lb = np.array(self.problem.lb)
        ub = np.array(self.problem.ub)

        for i in range(self.pop_size):
            # Her birey için boyut sayısı kadar kaotik değer üret
            pos = np.zeros(self.problem.n_dims)
            chaos = np.random.rand(self.problem.n_dims)  # başlangıç seed
            for j in range(self.problem.n_dims):
                # 10 iterasyon ısınma
                for _ in range(10):
                    chaos[j] = self._tent_map(chaos[j])
                pos[j] = lb[j] + chaos[j] * (ub[j] - lb[j])

            pos = self.correct_solution(pos)
            agent = self.generate_empty_agent(pos)
            self.pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos)

        # Paralel/vektörize modlarda fitness hesapla
        self.pop = self.update_target_for_population(self.pop)

    # ------------------------------------------------------------------
    # Ana evrim döngüsü
    # ------------------------------------------------------------------

    def evolve(self, epoch):
        """IAOA ana güncelleme adımı."""
        pop_size = self.pop_size
        lb = np.array(self.problem.lb)
        ub = np.array(self.problem.ub)
        n_dims = self.problem.n_dims

        # En iyi çözümü al
        best_pos = self.g_best.solution.copy()

        # Ortalama popülasyon pozisyonu (X_M)
        all_pos = np.array([agent.solution for agent in self.pop])
        x_mean = np.mean(all_pos, axis=0)

        # İterasyon oranı (0 → 1)
        t_ratio = (epoch + 1) / self.epoch

        # Adaptif kontrol parametresi (orijinal AO'daki g1, g2)
        g1 = 2 * np.random.rand() - 1          # [-1, 1]
        g2 = 2 * (1 - t_ratio)                 # [2 → 0]

        pop_new = []
        for i in range(pop_size):
            pos = self.pop[i].solution.copy()
            r_i = np.random.rand(n_dims)

            # Strateji seçimi: t_ratio < 2/3 → keşif, >= 2/3 → sömürü
            if t_ratio < (2.0 / 3.0):

                if np.random.rand() < 0.5:
                    # ---- X1: Genişletilmiş keşif + Lévy uçuşu (IAOA iyileştirmesi)
                    levy = self._levy_flight()
                    # Orijinal AO X1 formülü + Lévy pertürbasyon
                    x1 = best_pos * (1 - t_ratio) + \
                         (x_mean - best_pos) * np.random.rand() + \
                         self.alpha * levy * (best_pos - pos)
                    pos_new = x1
                else:
                    # ---- X2: Daralma keşfi (yıldız izi spirali)
                    # Rastgele bir birey seç
                    idx_r = np.random.randint(0, pop_size)
                    x_rand = self.pop[idx_r].solution.copy()
                    # Spiral parametresi
                    D = np.abs(x_rand - pos)
                    theta = -np.random.rand() * np.pi  # [-π, 0]
                    r1 = np.random.rand()
                    pos_new = D * np.exp(theta) * np.cos(2 * np.pi * r1) + x_rand

            else:
                if np.random.rand() < 0.5:
                    # ---- X3: Yoğun yararlanma + Gaussian walk (IAOA iyileştirmesi)
                    # Orijinal AO X3 + Gaussian bileşen
                    x3_ao = best_pos - \
                            (best_pos * r_i - x_mean * r_i) * g2
                    x3_gaussian = self._gaussian_walk(pos, best_pos)
                    # İki bileşeni lineer kombinasyonla birleştir
                    blend = t_ratio  # geç iterasyonlarda Gaussian ağırlığı artar
                    pos_new = (1 - blend) * x3_ao + blend * x3_gaussian
                else:
                    # ---- X4: Delta yararlanma (AO orijinali)
                    levy = self._levy_flight()
                    pos_new = best_pos - \
                              g1 * (best_pos - pos) - \
                              g2 * (x_mean - pos) + \
                              self.alpha * levy

            # ---- Sinüsoidal lokal ince ayar (son delta% iterasyonlarda)
            if t_ratio > (1 - self.delta):
                freq = 2 * np.pi * np.random.rand()
                amplitude = 0.01 * (ub - lb)
                perturbation = amplitude * np.sin(freq * (epoch + 1))
                pos_new = pos_new + perturbation

            # Sınır düzeltme ve ajan oluşturma
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)

        # Fitness güncellemesi (paralel mod desteği)
        self.pop = self.update_target_for_population(pop_new)

        # Greedy seçim: eskiden daha iyi değilse mevcut pozisyona geri dön
        for i in range(pop_size):
            if self.compare_target(
                self.pop[i].target,
                self.pop_before_evolution[i].target if hasattr(self, 'pop_before_evolution') else self.pop[i].target,
                self.problem.minmax
            ):
                pass  # yeni daha iyi, koru
            else:
                # eski daha iyi, geri al
                self.pop[i] = self.pop_before_evolution[i] \
                    if hasattr(self, 'pop_before_evolution') else self.pop[i]

    # ------------------------------------------------------------------
    # Greedy seçim için eski popülasyonu sakla
    # ------------------------------------------------------------------

    def before_evolve(self, epoch):
        """Her iterasyon başında mevcut popülasyonun kopyasını sakla."""
        import copy
        self.pop_before_evolution = copy.deepcopy(self.pop)


# ======================================================================
# Kümeleme problemine entegrasyon helper'ı
# ======================================================================

class IAAOClusteringProblem:
    """
    IAOA'yı generate_assignments.py ile uyumlu hale getiren sarmalayıcı.

    Kullanım
    --------
    from iaoa_optimizer import OriginalIAOA, IAAOClusteringProblem
    from mealpy import Problem

    # generate_assignments.py içindeki run_optimizer() çağrısını şöyle değiştir:
    optimizer = OriginalIAOA(epoch=500, pop_size=30, alpha=0.1, delta=0.1)
    # Diğer her şey aynı — Problem nesnesi ve solve() çağrısı değişmez.
    """
    pass


# ======================================================================
# Test: standalone çalıştır (mealpy kurulu ortamda)
# ======================================================================

if __name__ == "__main__":
    """
    Basit test: Sphere fonksiyonu üzerinde IAOA çalıştır.
    Bunu kendi ortamında çalıştırarak doğruluğu test edebilirsin.
    """
    try:
        from mealpy import FloatVar, Problem

        class SphereProblem(Problem):
            def __init__(self):
                bounds = FloatVar(lb=[-100]*30, ub=[100]*30)
                super().__init__(bounds, minmax="min")

            def obj_func(self, solution):
                return np.sum(solution ** 2)

        problem = SphereProblem()
        model = OriginalIAOA(epoch=200, pop_size=30, alpha=0.1, delta=0.1)
        model.solve(problem, seed=42)
        print(f"Best fitness: {model.g_best.target.fitness:.6e}")
        print(f"Expected: < 1e-5 for Sphere (D=30)")

    except ImportError:
        print("mealpy yüklü değil — test atlandı.")
        print("Kendi ortamında çalıştırabilirsin: python iaoa_optimizer.py")