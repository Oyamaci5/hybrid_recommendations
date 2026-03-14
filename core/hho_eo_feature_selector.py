"""
Hybrid HHO + EO tabanlı binary feature selection aracı.

Bu modül, IBEO (Improved Binary Equilibrium Optimizer) ve
Harris Hawks Optimization (HHO) fikirlerini birleştirir:

- EO / IBEO'dan:
  * Sürekli uzayda çözüm güncellemesi
  * S-şekilli transfer fonksiyonu ile binary'e geçiş
  * Opposition-Based Learning (OBL) ile başlangıç çeşitliliği
  * En iyi çözüm etrafında küçük yerel arama (bit flip)

- HHO'dan:
  * Kaçış enerjisine (E) bağlı keşif / sömürü fazları
  * Tavşan (en iyi çözüm) etrafında farklı kuşatma stratejileri

Kullanım senaryosu:
  - Genel binary feature selection (sağlık verisi, içerik özellikleri vb.)
  - Soğuk başlangıç (cold-start) tavsiye sistemlerinde kullanıcı / item
    özelliklerinden en açıklayıcı olanları seçmek için.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class HybridHHOEOConfig:
    """HHO+EO feature selector için temel hiperparametreler."""

    n_agents: int = 20
    max_iter: int = 50
    alpha: float = 0.99  # IBEO'daki gibi: accuracy ağırlığı
    cv_splits: int = 5
    random_state: Optional[int] = None


class HybridHHOEOFeatureSelector:
    """
    HHO + EO hibritine dayalı binary feature selection.

    Notlar:
      - Amaç fonksiyonu IBEO'daki gibi:
            Fitness = α * Error + β * (|M| / |N|)
        burada Error = 1 - accuracy, M = seçilen feature sayısı, N = toplam feature sayısı.
      - Daha küçük fitness daha iyidir.
    """

    def __init__(
        self,
        n_agents: int = 20,
        max_iter: int = 50,
        alpha: float = 0.99,
        cv_splits: int = 5,
        classifier=None,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.cv_splits = cv_splits
        self.classifier = classifier
        self.random_state = random_state

        # Çıktılar
        self.best_mask_: Optional[np.ndarray] = None
        self.best_score_: Optional[float] = None
        self.selected_features_indices_: Optional[np.ndarray] = None

        if random_state is not None:
            np.random.seed(random_state)

    # ------------------------------------------------------------------
    # Yardımcı fonksiyonlar
    # ------------------------------------------------------------------

    def _init_classifier(self):
        if self.classifier is not None:
            return self.classifier
        return KNeighborsClassifier(n_neighbors=5)

    def _evaluate_mask(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
        """IBEO fitness: α*Error + β*(|M|/|N|). Küçük olan daha iyi."""
        if mask.sum() == 0:
            # Hiç feature seçilmezse, çok kötü fitness ver
            return 1e6

        X_sub = X[:, mask.astype(bool)]
        clf = self._init_classifier()

        cv = StratifiedKFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        scores = cross_val_score(clf, X_sub, y, cv=cv, scoring="accuracy")
        acc = scores.mean()
        error = 1.0 - acc

        feature_ratio = mask.sum() / float(len(mask))
        fitness = self.alpha * error + self.beta * feature_ratio
        return float(fitness)

    @staticmethod
    def _sigmoid_transfer(C: np.ndarray) -> np.ndarray:
        """IBEO'daki S-şekilli transfer fonksiyonu."""
        return 1.0 / (1.0 + np.exp(-C))

    def _continuous_to_binary(self, C: np.ndarray) -> np.ndarray:
        """Sürekli konumu [0,1] aralığında binary maskeye çevir."""
        S = self._sigmoid_transfer(C)
        rand = np.random.rand(*S.shape)
        return (rand < S).astype(int)

    def _OBL_initialization(
        self, lb: float, ub: float, dim: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Opposition-Based Learning ile başlangıç popülasyonu."""
        # Normal popülasyon
        C = lb + (ub - lb) * np.random.rand(self.n_agents, dim)
        # Zıt popülasyon
        C_opp = lb + ub - C
        return C.astype(np.float32), C_opp.astype(np.float32)

    # ------------------------------------------------------------------
    # EO bileşenleri
    # ------------------------------------------------------------------

    @staticmethod
    def _build_equilibrium_pool(
        C: np.ndarray, fitness: np.ndarray
    ) -> np.ndarray:
        """En iyi 4 ajan + onların ortalaması (EO / IBEO fikri)."""
        idx_sorted = np.argsort(fitness)
        best_indices = idx_sorted[:4]
        top4 = C[best_indices]
        avg = top4.mean(axis=0)
        pool = np.vstack([top4, avg[None, :]])  # shape (5, dim)
        return pool

    def _eo_update(
        self, C: np.ndarray, eq_pool: np.ndarray, iter_idx: int
    ) -> np.ndarray:
        """
        Basitleştirilmiş EO güncellemesi.
        C_new = Ceq + (C - Ceq)*F + G
        """
        n_agents, dim = C.shape

        # Parametreler (IBEO / EO makalesindeki tipik değerler)
        a1 = 2.0
        a2 = 1.0
        GP = 0.5

        iter_ratio = iter_idx / float(self.max_iter)
        t = (1.0 - iter_ratio) * (iter_ratio**a2)

        # Eλ, Er ~ U(0,1)
        E_lambda = np.random.rand(n_agents, dim)
        Er = np.random.rand(n_agents, dim)

        # F terimi (yaklaşık formül)
        F = a1 * np.sign(Er - 0.5) * (np.exp(-E_lambda * t) - 1.0)

        # Rastgele Ceq seç
        rand_idx = np.random.randint(0, eq_pool.shape[0], size=n_agents)
        Ceq = eq_pool[rand_idx]

        # G terimi
        r1 = np.random.rand(n_agents, dim)
        r2 = np.random.rand(n_agents, dim)
        GCP = np.where(r2 >= GP, 0.5 * r1, 0.0)  # 0 veya 0.5*r1
        G0 = GCP * (Ceq - E_lambda * C)
        G = G0 * F

        C_new = Ceq + (C - Ceq) * F + G
        return C_new.astype(np.float32)

    # ------------------------------------------------------------------
    # HHO bileşenleri
    # ------------------------------------------------------------------

    def _hho_update(
        self, C: np.ndarray, best_C: np.ndarray, iter_idx: int
    ) -> np.ndarray:
        """
        Basitleştirilmiş HHO güncellemesi: keşif / sömürü.
        Orijinal HHO'dan ana fikirler alınmıştır.
        """
        n_agents, dim = C.shape
        M = self.max_iter

        # Kaçış enerjisi
        E0 = 2 * np.random.rand() - 1.0  # [-1,1]
        E = 2 * E0 * (1 - iter_idx / float(M))

        q = np.random.rand(n_agents, 1)
        r = np.random.rand(n_agents, dim)

        C_new = np.copy(C)

        if abs(E) >= 1.0:
            # Keşif fazı
            rand_indices = np.random.randint(0, n_agents, size=n_agents)
            X_rand = C[rand_indices]
            C_new = X_rand - r * np.abs(X_rand - 2 * r * C)
        else:
            # Sömürü fazı (soft/hard besiege karışık basit versiyon)
            J = 2 * (1 - np.random.rand())  # sıçrama kuvveti, burada sadece ölçek olarak
            D = np.abs(best_C - C)
            C_new = D * np.exp(-E) * np.cos(2 * np.pi * r) * J + best_C

        return C_new.astype(np.float32)

    # ------------------------------------------------------------------
    # Local search (IBEO LSA'ya benzer)
    # ------------------------------------------------------------------

    def _local_search(
        self,
        best_mask: np.ndarray,
        max_flips: int = 3,
    ) -> np.ndarray:
        """
        En iyi çözüm üzerinde küçük bit flip'lerle arama.
        Burada sadece maskeyi değiştiriyoruz; fitness değerlendirmesi dışarıda yapılır.
        """
        dim = len(best_mask)
        candidate = best_mask.copy()
        max_flips = min(max_flips, dim)
        idx = np.random.choice(dim, size=max_flips, replace=False)
        candidate[idx] = 1 - candidate[idx]
        return candidate

    # ------------------------------------------------------------------
    # Ana optimize fonksiyonu
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HybridHHOEOFeatureSelector":
        """
        HHO+EO hibritini kullanarak binary feature selection gerçekleştir.

        Args:
            X: (n_samples, n_features) özellik matrisi
            y: (n_samples,) sınıf etiketleri
        """
        X = np.asarray(X)
        y = np.asarray(y)
        _, dim = X.shape

        lb, ub = 0.0, 1.0

        # OBL initialization
        C, C_opp = self._OBL_initialization(lb, ub, dim)

        # Her iki popülasyonu da değerlendir ve en iyileri seç
        all_C = np.vstack([C, C_opp])  # (2*n_agents, dim)
        all_masks = self._continuous_to_binary(all_C)
        fitness_all = np.array(
            [self._evaluate_mask(X, y, m) for m in all_masks],
            dtype=np.float64,
        )

        idx_sorted = np.argsort(fitness_all)
        C = all_C[idx_sorted[: self.n_agents]]

        # Başlangıç maskeleri ve fitness
        masks = self._continuous_to_binary(C)
        fitness = np.array(
            [self._evaluate_mask(X, y, m) for m in masks],
            dtype=np.float64,
        )

        best_idx = int(np.argmin(fitness))
        best_mask = masks[best_idx].copy()
        best_C = C[best_idx].copy()
        best_fit = float(fitness[best_idx])

        for iter_idx in range(1, self.max_iter + 1):
            # EO equilibrium pool
            eq_pool = self._build_equilibrium_pool(C, fitness)

            # Popülasyonu ikiye böl: ilk yarı EO, ikinci yarı HHO
            half = max(1, self.n_agents // 2)

            C_eo = self._eo_update(C[:half], eq_pool, iter_idx)
            C_hho = self._hho_update(C[half:], best_C, iter_idx)

            # Yeni popülasyon
            C_new = np.vstack([C_eo, C_hho])

            # Sınırla [lb,ub]
            C_new = np.clip(C_new, lb, ub)

            # Binary maskeye dönüştür
            masks_new = self._continuous_to_binary(C_new)
            fitness_new = np.array(
                [self._evaluate_mask(X, y, m) for m in masks_new],
                dtype=np.float64,
            )

            # Local search sadece global en iyi üzerinde
            ls_mask = self._local_search(best_mask, max_flips=3)
            ls_fit = self._evaluate_mask(X, y, ls_mask)
            if ls_fit < best_fit:
                best_fit = float(ls_fit)
                best_mask = ls_mask.copy()

            # Yeni popülasyonu kabul et
            C = C_new
            masks = masks_new
            fitness = fitness_new

            # Global en iyiyi güncelle
            iter_best_idx = int(np.argmin(fitness))
            if fitness[iter_best_idx] < best_fit:
                best_fit = float(fitness[iter_best_idx])
                best_mask = masks[iter_best_idx].copy()
                best_C = C[iter_best_idx].copy()

        # Sonuçları sakla
        self.best_mask_ = best_mask
        # Daha intuitif bir skor için: 1 - fitness (sadece raporlama amaçlı)
        self.best_score_ = 1.0 - best_fit
        self.selected_features_indices_ = np.where(best_mask == 1)[0]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.best_mask_ is None:
            raise RuntimeError("Önce fit(X, y) çağırmalısınız.")
        return np.asarray(X)[:, self.best_mask_.astype(bool)]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

