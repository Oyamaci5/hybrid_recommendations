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
    ):
        self.n_users        = n_users
        self.n_items        = n_items
        self.latent_dim     = latent_dim
        self.learning_rate  = learning_rate
        self.regularization = regularization
        self.n_epochs       = n_epochs
        self.random_seed    = random_seed
        self.is_fitted      = False

        rng   = np.random.RandomState(random_seed)
        scale = np.sqrt(5.0 / latent_dim)
        self.U = rng.uniform(0, scale, (n_users, latent_dim)).astype(np.float32)
        self.V = rng.uniform(0, scale, (n_items, latent_dim)).astype(np.float32)

    def fit(self, train_ratings: np.ndarray, verbose: bool = False) -> "WNMFModel":
        """
        Modeli eğit — hem U hem V güncellenir.

        Parametreler
        ------------
        train_ratings : shape (n, 3) — [user_id, item_id, rating], 0-indexed
        verbose       : her 10 epoch'ta kayıp yazdır
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
                u     = u_shuf[k]
                i     = i_shuf[k]
                r     = r_shuf[k]
                error = r - float(np.dot(self.U[u], self.V[i]))

                grad_u = -error * self.V[i] + reg * self.U[u]
                grad_v = -error * self.U[u] + reg * self.V[i]

                self.U[u] -= lr * grad_u
                self.V[i] -= lr * grad_v

                np.maximum(self.U[u], 0, out=self.U[u])
                np.maximum(self.V[i], 0, out=self.V[i])

            if verbose and (epoch % 10 == 0 or epoch == self.n_epochs - 1):
                loss = self._compute_loss(user_ids, item_ids, ratings)
                print(f"    Epoch {epoch:3d}/{self.n_epochs}  loss={loss:.4f}")

        self.is_fitted = True
        return self

    def predict(
        self,
        user_ids : np.ndarray,
        item_ids : np.ndarray,
        clip     : bool = True,
    ) -> np.ndarray:
        """Rating tahmini — U[u] · V[i], [1,5] aralığına clip edilir."""
        user_ids = np.asarray(user_ids, dtype=np.int32)
        item_ids = np.asarray(item_ids, dtype=np.int32)
        preds    = np.sum(self.U[user_ids] * self.V[item_ids], axis=1)
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
    ):
        self.n_items         = n_items
        self.latent_dim      = latent_dim
        self.learning_rate   = learning_rate
        self.regularization  = regularization
        self.n_epochs_global = n_epochs_global
        self.random_seed     = random_seed
        self.V               = None   # global V, fit_global_V sonrası dolar
        self.V_fitted        = False

        # Global modeli içeride tut — V'yi buradan alacağız
        self._global_model = WNMFModel(
            n_users        = n_users_global,
            n_items        = n_items,
            latent_dim     = latent_dim,
            learning_rate  = learning_rate,
            regularization = regularization,
            n_epochs       = n_epochs_global,
            random_seed    = random_seed,
        )

    def fit_global_V(
        self,
        all_train : np.ndarray,
        verbose   : bool = False,
    ) -> "WNMFSharedV":
        """
        Tüm kullanıcıların rating'leriyle global V'yi öğren.

        Bu adım bir kez çalışır. Sonraki tüm küme modelleri
        bu V'yi paylaşır.

        Parametreler
        ------------
        all_train : shape (n, 3) — tüm train rating'leri
        verbose   : eğitim ilerlemesini yazdır
        """
        print("  [Global V] eğitiliyor...")
        self._global_model.fit(all_train, verbose=verbose)
        self.V       = self._global_model.V.copy()   # (n_items, latent_dim)
        self.V_fitted = True
        print(f"  [Global V] tamamlandı — V shape: {self.V.shape}")
        return self

    def make_cluster_model(
        self,
        n_users_cluster : int,
        n_epochs_cluster: int = 50,
        random_seed     : int = 42,
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
    ):
        self.n_users        = n_users
        self.n_items        = n_items
        self.latent_dim     = latent_dim
        self.learning_rate  = learning_rate
        self.regularization = regularization
        self.n_epochs       = n_epochs
        self.random_seed    = random_seed

        # V sabit — global modelden kopyalanmış
        self.V = V_shared.copy()

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
                u     = u_shuf[k]
                i     = i_shuf[k]
                r     = r_shuf[k]

                # Hata hesapla
                error = r - float(np.dot(self.U[u], self.V[i]))

                # Sadece U güncellenir — V sabit
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
        preds    = np.sum(self.U[user_ids] * self.V[item_ids], axis=1)
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