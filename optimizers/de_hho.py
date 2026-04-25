import numpy as np
from mealpy.swarm_based.HHO import OriginalHHO

class DE_HHO(OriginalHHO):
    """
    Differential Evolutionary Harris Hawks Optimization (DE-HHO)
    Makale için özel tasarlanmış 'Gömülü' (Embedded) Keşif-Sömürü Hibriti.
    """
    def __init__(self, epoch=10000, pop_size=100, F=0.5, CR=0.9, **kwargs):
        super().__init__(epoch=epoch, pop_size=pop_size, **kwargs)
        self.F = F    # DE Mutasyon Faktörü (Keşif büyüklüğü)
        self.CR = CR  # DE Çaprazlama Oranı (Sömürü dengesi)

    def evolve(self, epoch):
        # 1. STANDART HHO ADIMI (Şahinlerin Avlanması - Sömürü)
        # Orijinal HHO'nun hareketlerini yapmasına izin veriyoruz
        super().evolve(epoch)
        
        # 2. DE MUTASYON ADIMI (Buzkıran ve Yerel Minimumdan Kaçış - Keşif)
        # HHO'nun güncellediği pozisyonların üzerinden DE ile geçeriz
        for i in range(self.pop_size):
            # Rastgele 3 farklı birey (şahin) seçiyoruz
            choices = list(range(self.pop_size))
            choices.remove(i)
            r1, r2, r3 = np.random.choice(choices, 3, replace=False)
            
            x1 = self.pop[r1].solution
            x2 = self.pop[r2].solution
            x3 = self.pop[r3].solution
            
            # DE Mutasyon Denklemi: V = x1 + F * (x2 - x3)
            mutant = x1 + self.F * (x2 - x3)
            
            # Çözümün uzay sınırları dışına çıkmasını engelle
            # Mealpy'ın iç hata fırlatmaması için numpy clip kullanıyoruz.
            mutant = np.clip(mutant, self.problem.lb, self.problem.ub)
            
            # DE Çaprazlama (Crossover) Denklemi
            # Şahinin kendi aklı ile mutasyona uğramış aklını harmanlıyoruz
            crossover_mask = np.random.rand(self.problem.n_dims) < self.CR
            candidate_pos = np.where(crossover_mask, mutant, self.pop[i].solution)
            
            # Garanti olması için tekrar sınırları kontrol et
            candidate_pos = np.clip(candidate_pos, self.problem.lb, self.problem.ub)
            
            # Yeni adayın FCM (Fitness) Skoru hesaplanır
            candidate_target = self.get_target(candidate_pos)
            
            # SEÇİLİM (Selection - Greedy)
            # Eğer DE'nin ürettiği aday, HHO'nun bulduğundan daha iyiyse onu kullan!
            if self.compare_target(candidate_target, self.pop[i].target, self.problem.minmax):
                self.pop[i].update(solution=candidate_pos, target=candidate_target)
                
                # Global en iyiyi bulduysak onu da güncelle
                if self.compare_target(candidate_target, self.g_best.target, self.problem.minmax):
                    self.g_best.update(solution=candidate_pos, target=candidate_target)