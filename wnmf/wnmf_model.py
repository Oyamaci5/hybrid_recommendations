"""
wnmf_model.py
=============
İki WNMF modeli:

1. WNMFModel
   Temel WNMF — hem U hem V öğrenir.
   Global baseline olarak kullanılır.

2. WNMFSharedV
   Paylaşımlı V (film embedding) + küme bazlı U (kullanıcı embedding).

   Problem: Per-cluster WNMF'de her küme V'yi sıfırdan öğrenir.
   Az veriyle (8-10 kullanıcı) V iyi öğrenilemiyor → MAE kötü çıkıyor.

   Çözüm: V'yi tüm veriden global öğren, U'yu küme bazında ince ayar yap.

   Aşama 1 — Global V öğren:
       Tüm kullanıcıların rating'leriyle tek bir WNMFModel eğit.
       Bu modelin V matrisi (film embedding'leri) kaliteli olur.

   Aşama 2 — Küme bazlı U öğren:
       Her küme için V sabit tutulur, sadece U güncellenir.
       Küme, kendi kullanıcı profilini V'nin üzerinde öğrenir.

   Matematiksel fark:
       Standart WNMF : U[u] ve V[i] her ikisi de güncellenir
       SharedV WNMF  : Sadece U[u] güncellenir, V[i] sabit kalır

   Kayıp fonksiyonu aynı:
       L = Σ_(u,i)∈Ω (r_ui - U[u]·V[i])² + λ||U||²
       (V regularization yok — V sabit olduğu için)
"""

import numpy as np
from typing import Optional


# ============================================================
# TEMEL WNMF MODELİ
# ============================================================

class WNMFModel:
    """
    Temel WNMF modeli — hem U hem V öğrenir.
    Global baseline ve global V eğitimi için kullanılır.

    Parametreler
    ------------
    n_users        : gruptaki kullanıcı sayısı
    n_items        : toplam film sayısı
    latent_dim     : gizli faktör boyutu (k)
    learning_rate  : SGD öğrenme hızı (α)
    regularization : L2 katsayısı (λ)
    n_epochs       : eğitim döngüsü sayısı
    random_seed    : tekrarlanabilirlik
    """

    def __init__(
        self,
        n_users        : int,
        n_items        : int,
        latent_dim     : int   = 20,
        learning_rate  : float = 0.01,
        regularization : float = 0.01,
        n_epochs       : int   = 100,
        random_seed    : int   = 42,
        use_bias       : bool  = True,
        use_svdpp      : bool  = False,
        init_method    : str   = "random",
        inmed_trim     : tuple = (5.0, 95.0),
        inmed_jitter   : float = 0.01,
    ):
        self.n_users        = n_users
        self.n_items        = n_items
        self.latent_dim     = latent_dim
        self.learning_rate  = learning_rate
        self.regularization = regularization
        self.n_epochs       = n_epochs
        self.random_seed    = random_seed
        self.use_bias       = use_bias
        self.use_svdpp      = use_svdpp
        self.init_method    = str(init_method).lower()
        self.inmed_trim     = inmed_trim
        self.inmed_jitter   = float(inmed_jitter)
        self.is_fitted      = False
        self.user_implicit  = None
        self._svdpp_train_items: Optional[dict] = None

        self.mu  = 0.0
        self.b_u = np.zeros(n_users, dtype=np.float32)
        self.b_i = np.zeros(n_items, dtype=np.float32)

        rng   = np.random.RandomState(random_seed)
        scale = np.sqrt(5.0 / latent_dim)
        self.U = rng.uniform(0, scale, (n_users, latent_dim)).astype(np.float32)
        self.V = rng.uniform(0, scale, (n_items, latent_dim)).astype(np.float32)
        if use_svdpp:
            rng2 = np.random.RandomState(random_seed + 1)
            self.Y = rng2.uniform(0, scale, (n_items, latent_dim)).astype(np.float32)
        else:
            self.Y = None

    def _initialize_inmed_factors(self, ratings: np.ndarray) -> None:
        """
        INMED: gözlenen rating dağılımının trimmed mean'i ile U/V başlat.
        """
        if ratings.size == 0:
            return

        low, high = self.inmed_trim
        low = float(max(0.0, min(100.0, low)))
        high = float(max(0.0, min(100.0, high)))
        if high <= low:
            low, high = 5.0, 95.0

        trimmed_mean = float(np.mean(ratings))
        if ratings.size >= 5:
            q_low, q_high = np.percentile(ratings, [low, high])
            keep = (ratings >= q_low) & (ratings <= q_high)
            if np.any(keep):
                trimmed_mean = float(np.mean(ratings[keep]))

        base_val = float(np.sqrt(max(trimmed_mean, 1e-6) / max(self.latent_dim, 1)))
        rng = np.random.RandomState(self.random_seed)
        jitter_u = rng.uniform(
            -self.inmed_jitter, self.inmed_jitter, size=self.U.shape
        ).astype(np.float32)
        jitter_v = rng.uniform(
            -self.inmed_jitter, self.inmed_jitter, size=self.V.shape
        ).astype(np.float32)
        self.U[:] = np.maximum(base_val + jitter_u, 0.0).astype(np.float32)
        self.V[:] = np.maximum(base_val + jitter_v, 0.0).astype(np.float32)

    def fit(
        self,
        train_ratings : np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        verbose       : bool = False,
    ) -> "WNMFModel":
        """
        Modeli eğit — hem U hem V güncellenir.

        Parametreler
        ------------
        train_ratings : shape (n, 3) — [user_id, item_id, rating], 0-indexed
        sample_weights: shape (n,) — gözlem ağırlıkları; None ise 1.0
        verbose       : her 10 epoch'ta kayıp yazdır
        """
        np.random.seed(self.random_seed)

        user_ids = train_ratings[:, 0].astype(np.int32)
        item_ids = train_ratings[:, 1].astype(np.int32)
        ratings  = train_ratings[:, 2].astype(np.float32)
        n        = len(ratings)
        lr       = self.learning_rate
        reg      = self.regularization

        if self.init_method == "inmed":
            self._initialize_inmed_factors(ratings)

        user_items: dict = {}
        user_implicit: dict = {}
        if self.use_svdpp:
            for k in range(n):
                u = int(user_ids[k])
                i = int(item_ids[k])
                if u not in user_items:
                    user_items[u] = []
                user_items[u].append(i)
            for u, items in user_items.items():
                ni = len(items)
                if ni > 0:
                    implicit = self.Y[items].sum(axis=0) / np.sqrt(ni)
                else:
                    implicit = np.zeros(self.latent_dim, dtype=np.float32)
                user_implicit[u] = implicit
            self._svdpp_train_items = user_items

        if self.use_bias:
            self.mu = float(ratings.mean())

        for epoch in range(self.n_epochs):
            idx    = np.random.permutation(n)
            u_shuf = user_ids[idx]
            i_shuf = item_ids[idx]
            r_shuf = ratings[idx]

            for k in range(n):
                u = u_shuf[k]
                i = i_shuf[k]
                r = r_shuf[k]

                w = (
                    sample_weights[idx[k]]
                    if sample_weights is not None
                    else 1.0
                )

                if self.use_svdpp:
                    u_vec = self.U[u] + user_implicit[u]
                    if self.use_bias:
                        pred = (
                            self.mu
                            + self.b_u[u]
                            + self.b_i[i]
                            + float(np.dot(u_vec, self.V[i]))
                        )
                        error = r - pred
                        grad_u = -error * w * self.V[i] + reg * self.U[u]
                        self.U[u] -= lr * grad_u
                        np.maximum(self.U[u], 0, out=self.U[u])
                        n_u = len(user_items[u])
                        sqrt_n = np.sqrt(n_u) if n_u > 0 else 1.0
                        for j in user_items[u]:
                            grad_y = (
                                -error * w * self.V[i] / sqrt_n
                                + reg * self.Y[j]
                            )
                            self.Y[j] -= lr * grad_y
                            np.maximum(self.Y[j], 0, out=self.Y[j])
                        grad_v = -error * w * u_vec + reg * self.V[i]
                        self.V[i] -= lr * grad_v
                        np.maximum(self.V[i], 0, out=self.V[i])
                        self.b_u[u] += lr * (error * w - reg * self.b_u[u])
                        self.b_i[i] += lr * (error * w - reg * self.b_i[i])
                        if n_u > 0:
                            user_implicit[u] = (
                                self.Y[user_items[u]].sum(axis=0) / sqrt_n
                            )
                    else:
                        pred = float(np.dot(u_vec, self.V[i]))
                        error = r - pred
                        grad_u = -error * w * self.V[i] + reg * self.U[u]
                        self.U[u] -= lr * grad_u
                        np.maximum(self.U[u], 0, out=self.U[u])
                        n_u = len(user_items[u])
                        sqrt_n = np.sqrt(n_u) if n_u > 0 else 1.0
                        for j in user_items[u]:
                            grad_y = (
                                -error * w * self.V[i] / sqrt_n
                                + reg * self.Y[j]
                            )
                            self.Y[j] -= lr * grad_y
                            np.maximum(self.Y[j], 0, out=self.Y[j])
                        grad_v = -error * w * u_vec + reg * self.V[i]
                        self.V[i] -= lr * grad_v
                        np.maximum(self.V[i], 0, out=self.V[i])
                        if n_u > 0:
                            user_implicit[u] = (
                                self.Y[user_items[u]].sum(axis=0) / sqrt_n
                            )
                elif self.use_bias:
                    pred = (
                        self.mu
                        + self.b_u[u]
                        + self.b_i[i]
                        + float(np.dot(self.U[u], self.V[i]))
                    )
                    error = r - pred
                    grad_u = -error * w * self.V[i] + reg * self.U[u]
                    grad_v = -error * w * self.U[u] + reg * self.V[i]
                    self.U[u] -= lr * grad_u
                    self.V[i] -= lr * grad_v
                    self.b_u[u] += lr * (error * w - reg * self.b_u[u])
                    self.b_i[i] += lr * (error * w - reg * self.b_i[i])
                    np.maximum(self.U[u], 0, out=self.U[u])
                    np.maximum(self.V[i], 0, out=self.V[i])
                else:
                    error = r - float(np.dot(self.U[u], self.V[i]))
                    if sample_weights is not None:
                        grad_u = -error * w * self.V[i] + reg * self.U[u]
                        grad_v = -error * w * self.U[u] + reg * self.V[i]
                    else:
                        grad_u = -error * self.V[i] + reg * self.U[u]
                        grad_v = -error * self.U[u] + reg * self.V[i]

                    self.U[u] -= lr * grad_u
                    self.V[i] -= lr * grad_v

                    np.maximum(self.U[u], 0, out=self.U[u])
                    np.maximum(self.V[i], 0, out=self.V[i])

            if verbose and (epoch % 10 == 0 or epoch == self.n_epochs - 1):
                loss = self._compute_loss(user_ids, item_ids, ratings)
                print(f"    Epoch {epoch:3d}/{self.n_epochs}  loss={loss:.4f}")

        if self.use_svdpp:
            self.user_implicit = np.zeros(
                (self.n_users, self.latent_dim), dtype=np.float32,
            )
            for u, items in user_items.items():
                ni = len(items)
                if ni > 0:
                    self.user_implicit[u] = (
                        self.Y[items].sum(axis=0) / np.sqrt(ni)
                    )

        self.is_fitted = True
        return self

    def _svdpp_implicit_u(self, u: int) -> np.ndarray:
        if (
            not self.use_svdpp
            or self.Y is None
            or self._svdpp_train_items is None
        ):
            return np.zeros(self.latent_dim, dtype=np.float32)
        items = self._svdpp_train_items.get(u, [])
        ni = len(items)
        if ni <= 0:
            return np.zeros(self.latent_dim, dtype=np.float32)
        return self.Y[items].sum(axis=0) / np.sqrt(ni)

    def predict(
        self,
        user_ids : np.ndarray,
        item_ids : np.ndarray,
        clip     : bool = True,
    ) -> np.ndarray:
        """Rating tahmini — U[u] · V[i], [1,5] aralığına clip edilir."""
        user_ids = np.asarray(user_ids, dtype=np.int32)
        item_ids = np.asarray(item_ids, dtype=np.int32)
        if self.use_svdpp:
            preds = []
            for u, i in zip(user_ids, item_ids):
                uu, ii = int(u), int(i)
                if self.is_fitted and self.user_implicit is not None:
                    implicit = self.user_implicit[uu]
                else:
                    implicit = self._svdpp_implicit_u(uu)
                u_vec = self.U[uu] + implicit
                if self.use_bias:
                    pred = (
                        self.mu
                        + self.b_u[uu]
                        + self.b_i[ii]
                        + float(np.dot(u_vec, self.V[ii]))
                    )
                else:
                    pred = float(np.dot(u_vec, self.V[ii]))
                if clip:
                    pred = float(np.clip(pred, 1.0, 5.0))
                preds.append(pred)
            return np.array(preds, dtype=np.float32)
        if self.use_bias:
            preds = (
                self.mu
                + self.b_u[user_ids]
                + self.b_i[item_ids]
                + np.sum(self.U[user_ids] * self.V[item_ids], axis=1)
            )
        else:
            preds = np.sum(self.U[user_ids] * self.V[item_ids], axis=1)
        if clip:
            np.clip(preds, 1.0, 5.0, out=preds)
        return preds.astype(np.float32)

    def evaluate(self, test_ratings: np.ndarray) -> tuple:
        """MAE ve RMSE hesapla."""
        user_ids = test_ratings[:, 0].astype(np.int32)
        item_ids = test_ratings[:, 1].astype(np.int32)
        true_r   = test_ratings[:, 2].astype(np.float32)
        pred_r   = self.predict(user_ids, item_ids)
        errors   = true_r - pred_r
        return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors ** 2)))

    def _compute_loss(self, user_ids, item_ids, ratings) -> float:
        preds  = self.predict(user_ids, item_ids, clip=False)
        errors = ratings - preds
        reg    = self.regularization * (
            float(np.mean(self.U ** 2)) + float(np.mean(self.V ** 2))
        )
        return float(np.mean(errors ** 2)) + reg


# ============================================================
# PAYLAŞIMLI V WNMF MODELİ
# ============================================================

class WNMFSharedV:
    """
    Paylaşımlı film embedding (V) + küme bazlı kullanıcı embedding (U).

    Nasıl çalışır:
        Aşama 1: fit_global_V(tüm_train)
            → Tüm kullanıcıların rating'leriyle V öğren
            → Bu V artık sabit

        Aşama 2: fit_cluster_U(küme_train)
            → V sabit, sadece U güncellenir
            → Her küme için ayrı çağrılır

    Neden daha iyi:
        Küçük bir kümede (8-10 kullanıcı) V'yi sıfırdan öğrenmek imkansız.
        Global V 80.000 rating'den öğrenildiği için kaliteli film
        temsilleri içerir. Küme sadece "bu filmlere nasıl tepki verir?"
        sorusunu cevaplamak için U'yu öğrenir.

    Kullanım:
        shared = WNMFSharedV(n_users_global, n_items, latent_dim=20)
        shared.fit_global_V(all_train)          # bir kez

        for cid, c_train in clusters.items():
            model = shared.make_cluster_model(n_users_cluster, seed=cid)
            model.fit_cluster_U(c_train)
            mae, rmse = model.evaluate(c_test)
    """

    def __init__(
        self,
        n_users_global : int,
        n_items        : int,
        latent_dim     : int   = 20,
        learning_rate  : float = 0.01,
        regularization : float = 0.01,
        n_epochs_global: int   = 100,
        random_seed    : int   = 42,
        use_bias       : bool  = True,
        use_svdpp      : bool  = False,
    ):
        self.n_items         = n_items
        self.latent_dim      = latent_dim
        self.learning_rate   = learning_rate
        self.regularization  = regularization
        self.n_epochs_global = n_epochs_global
        self.random_seed     = random_seed
        self._use_bias       = use_bias
        self._use_svdpp      = use_svdpp
        self.V               = None   # global V, fit_global_V sonrası dolar
        self.V_fitted        = False
        self.mu              = 0.0
        self.b_i             = np.zeros(n_items, dtype=np.float32)

        # Global modeli içeride tut — V'yi buradan alacağız
        self._global_model = WNMFModel(
            n_users        = n_users_global,
            n_items        = n_items,
            latent_dim     = latent_dim,
            learning_rate  = learning_rate,
            regularization = regularization,
            n_epochs       = n_epochs_global,
            random_seed    = random_seed,
            use_bias       = use_bias,
            use_svdpp      = use_svdpp,
        )

    def fit_global_V(
        self,
        all_train : np.ndarray,
        gray_mask : Optional[np.ndarray] = None,
        verbose   : bool = False,
    ) -> "WNMFSharedV":
        """
        Tüm kullanıcıların rating'leriyle global V'yi öğren.

        Bu adım bir kez çalışır. Sonraki tüm küme modelleri
        bu V'yi paylaşır.

        Parametreler
        ------------
        all_train : shape (n, 3) — tüm train rating'leri
        gray_mask : shape (n_users,) bool; True ise o kullanıcının rating'leri düşük ağırlıkla
        verbose   : eğitim ilerlemesini yazdır
        """
        print("  [Global V] eğitiliyor...")
        if gray_mask is not None:
            user_ids       = all_train[:, 0].astype(np.int32)
            sample_weights = np.where(gray_mask[user_ids], 0.1, 1.0).astype(np.float32)
        else:
            sample_weights = None
        self._global_model.fit(all_train, sample_weights=sample_weights, verbose=verbose)
        self.V        = self._global_model.V.copy()   # (n_items, latent_dim)
        self.mu       = self._global_model.mu
        self.b_i      = self._global_model.b_i.copy()
        self.V_fitted = True
        print(f"  [Global V] tamamlandı — V shape: {self.V.shape}")
        return self

    def make_cluster_model(
        self,
        n_users_cluster : int,
        n_epochs_cluster: int = 50,
        random_seed     : int = 42,
        cluster_ratings : Optional[np.ndarray] = None,
    ) -> "ClusterWNMF":
        """
        Bir küme için ClusterWNMF nesnesi oluştur.
        Global V bu nesneye kopyalanır, U sıfırdan başlar.

        Parametreler
        ------------
        n_users_cluster  : bu kümedeki kullanıcı sayısı
        n_epochs_cluster : küme U eğitim epoch sayısı (global'den az olabilir)
        random_seed      : tekrarlanabilirlik

        Döndürür
        --------
        ClusterWNMF — fit_cluster_U ve evaluate metodları olan nesne
        """
        if not self.V_fitted:
            raise RuntimeError("Önce fit_global_V() çağrılmalı")

        return ClusterWNMF(
            n_users        = n_users_cluster,
            n_items        = self.n_items,
            latent_dim     = self.latent_dim,
            V_shared       = self.V,             # global V paylaşılıyor
            learning_rate  = self.learning_rate,
            regularization = self.regularization,
            n_epochs       = n_epochs_cluster,
            random_seed    = random_seed,
            use_bias       = self._use_bias,
            mu             = self.mu,
            b_i_global     = self.b_i,
            cluster_ratings= cluster_ratings,
        )


class ClusterWNMF:
    """
    Tek bir küme için WNMF — V sabit, sadece U öğrenilir.

    Doğrudan oluşturulmaz — WNMFSharedV.make_cluster_model() kullanılır.
    """

    def __init__(
        self,
        n_users        : int,
        n_items        : int,
        latent_dim     : int,
        V_shared       : np.ndarray,
        learning_rate  : float = 0.01,
        regularization : float = 0.01,
        n_epochs       : int   = 50,
        random_seed    : int   = 42,
        use_bias       : bool  = True,
        mu             : float = 0.0,
        b_i_global     : Optional[np.ndarray] = None,
        cluster_ratings: Optional[np.ndarray] = None,
    ):
        self.n_users        = n_users
        self.n_items        = n_items
        self.latent_dim     = latent_dim
        self.learning_rate  = learning_rate
        self.regularization = regularization
        self.n_epochs       = n_epochs
        self.random_seed    = random_seed
        self.use_bias       = use_bias
        self.mu             = mu
        if cluster_ratings is not None and len(cluster_ratings) > 0:
            self.mu_k = float(cluster_ratings[:, 2].mean())
        else:
            self.mu_k = mu

        # V sabit — global modelden kopyalanmış
        self.V = V_shared.copy()

        if b_i_global is not None:
            self.b_i = b_i_global.copy()
        else:
            self.b_i = np.zeros(n_items, dtype=np.float32)
        self.b_u = np.zeros(n_users, dtype=np.float32)

        # U sıfırdan başlar — küme özelinde öğrenilecek
        rng    = np.random.RandomState(random_seed)
        scale  = np.sqrt(5.0 / latent_dim)
        self.U = rng.uniform(0, scale, (n_users, latent_dim)).astype(np.float32)

    def fit_cluster_U(
        self,
        train_ratings : np.ndarray,
        verbose       : bool = False,
    ) -> "ClusterWNMF":
        """
        V sabit tutarak sadece U'yu eğit.

        Parametreler
        ------------
        train_ratings : shape (n, 3) — kümenin train rating'leri
                        user_id'ler 0-indexed (remap_user_ids ile)
        """
        np.random.seed(self.random_seed)

        user_ids = train_ratings[:, 0].astype(np.int32)
        item_ids = train_ratings[:, 1].astype(np.int32)
        ratings  = train_ratings[:, 2].astype(np.float32)
        n        = len(ratings)
        lr       = self.learning_rate
        reg      = self.regularization

        for epoch in range(self.n_epochs):
            idx    = np.random.permutation(n)
            u_shuf = user_ids[idx]
            i_shuf = item_ids[idx]
            r_shuf = ratings[idx]

            for k in range(n):
                u = u_shuf[k]
                i = i_shuf[k]
                r = r_shuf[k]

                if self.use_bias:
                    pred = (
                        self.mu_k
                        + self.b_u[u]
                        + self.b_i[i]
                        + float(np.dot(self.U[u], self.V[i]))
                    )
                    error = r - pred
                    grad_u = -error * self.V[i] + reg * self.U[u]
                    self.U[u] -= lr * grad_u
                    self.b_u[u] += lr * (error - reg * self.b_u[u])
                    np.maximum(self.U[u], 0, out=self.U[u])
                else:
                    error = r - float(np.dot(self.U[u], self.V[i]))
                    grad_u     = -error * self.V[i] + reg * self.U[u]
                    self.U[u] -= lr * grad_u
                    np.maximum(self.U[u], 0, out=self.U[u])

            if verbose and (epoch % 10 == 0 or epoch == self.n_epochs - 1):
                loss = self._compute_loss(user_ids, item_ids, ratings)
                print(f"    Epoch {epoch:3d}/{self.n_epochs}  loss={loss:.4f}")

        return self

    def predict(
        self,
        user_ids : np.ndarray,
        item_ids : np.ndarray,
        clip     : bool = True,
    ) -> np.ndarray:
        """Rating tahmini — U[u] · V[i]"""
        user_ids = np.asarray(user_ids, dtype=np.int32)
        item_ids = np.asarray(item_ids, dtype=np.int32)
        if self.use_bias:
            preds = (
                self.mu_k
                + self.b_u[user_ids]
                + self.b_i[item_ids]
                + np.sum(self.U[user_ids] * self.V[item_ids], axis=1)
            )
        else:
            preds = np.sum(self.U[user_ids] * self.V[item_ids], axis=1)
        if clip:
            np.clip(preds, 1.0, 5.0, out=preds)
        return preds.astype(np.float32)

    def evaluate(self, test_ratings: np.ndarray) -> tuple:
        """MAE ve RMSE hesapla."""
        user_ids = test_ratings[:, 0].astype(np.int32)
        item_ids = test_ratings[:, 1].astype(np.int32)
        true_r   = test_ratings[:, 2].astype(np.float32)
        pred_r   = self.predict(user_ids, item_ids)
        errors   = true_r - pred_r
        return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors ** 2)))

    def _compute_loss(self, user_ids, item_ids, ratings) -> float:
        preds  = self.predict(user_ids, item_ids, clip=False)
        errors = ratings - preds
        reg    = self.regularization * float(np.mean(self.U ** 2))
        return float(np.mean(errors ** 2)) + reg