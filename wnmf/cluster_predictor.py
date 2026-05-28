"""
Surprise'dan bağımsız küme-içi KNNBaseline benzeri tahmin.

Formül (küme c için ayrı SGD ile öğrenilen biaslar):
    pred(u, i) = mu + b_u[c][u] + b_i[c][i]
                 + Σ sim(u,v) * (r(v,i) - (mu + b_u[c][v] + b_i[c][i]))
                   ─────────────────────────────────────────────────────
                                    Σ |sim(u,v)|

Benzerlik matrisi yalnızca küme içi; sapmalar bias-merkezli.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

RatingTriple = Tuple[int, int, float]


def map_wnmf_similarity(similarity: str) -> str:
    """wnmf_experiment similarity adını ClusterPredictor sim_metric'e çevir."""
    s = (similarity or 'pearson').strip().lower()
    if s in ('cosine', 'cos'):
        return 'cosine'
    if s in ('pearson', 'pearson_iuf', 'pearson_baseline'):
        return 'pearson'
    if s == 'msd':
        return 'cosine'
    return 'pearson'


def build_rating_matrix(
    train: np.ndarray,
    n_users: int,
    n_items: int,
) -> np.ndarray:
    """Train (u, i, r) satırlarından dense matris; 0 = puanlanmamış."""
    R = np.zeros((int(n_users), int(n_items)), dtype=np.float32)
    for row in np.asarray(train):
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        if 0 <= u < n_users and 0 <= i < n_items:
            R[u, i] = r
    return R


def test_rows_to_pairs(test: np.ndarray) -> List[RatingTriple]:
    """(N, 3) test dizisini (u, i, r) listesine çevir."""
    out: List[RatingTriple] = []
    for row in np.asarray(test):
        out.append((int(row[0]), int(row[1]), float(row[2])))
    return out


def _pearson_on_common(ru: np.ndarray, rv: np.ndarray) -> float:
    common = (ru > 0) & (rv > 0)
    n = int(common.sum())
    if n < 2:
        return 0.0
    a = ru[common].astype(np.float64)
    b = rv[common].astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom < 1e-12:
        return 0.0
    return float((a * b).sum() / denom)


class ClusterPredictor:
    """
    Küme-içi KNNBaseline: her küme için ayrı b_u, b_i (SGD); küme-içi kNN sapması.
    """

    def __init__(
        self,
        k_neighbors: int = 30,
        sim_metric: str = 'cosine',
        min_support: int = 2,
        bias_epochs: int = 20,
        bias_lr: float = 0.005,
        bias_reg: float = 0.02,
        bias_seed: int = 42,
    ):
        self.k = max(1, int(k_neighbors))
        self.sim_metric = (sim_metric or 'cosine').strip().lower()
        self.min_support = max(1, int(min_support))
        self.bias_epochs = max(1, int(bias_epochs))
        self.bias_lr = float(bias_lr)
        self.bias_reg = float(bias_reg)
        self.bias_seed = int(bias_seed)

        self.R: Optional[np.ndarray] = None
        self.assignments: Optional[np.ndarray] = None
        self.global_mean: float = 3.0
        self.b_u_cluster: Dict[int, np.ndarray] = {}
        self.b_i_cluster: Dict[int, np.ndarray] = {}
        self.sim: Optional[np.ndarray] = None

    def fit(self, rating_matrix: np.ndarray, assignments: np.ndarray):
        """
        rating_matrix : (n_users, n_items) numpy array, 0 = puanlanmamış
        assignments   : (n_users,) integer array, küme id'leri
        """
        self.R = np.asarray(rating_matrix, dtype=np.float32)
        self.assignments = np.asarray(assignments, dtype=np.int64).ravel()
        if self.assignments.shape[0] != self.R.shape[0]:
            raise ValueError(
                f'assignments uzunluğu ({self.assignments.shape[0]}) '
                f'R satır sayısı ({self.R.shape[0]}) ile uyuşmuyor',
            )

        rated = self.R > 0
        self.global_mean = float(self.R[rated].mean()) if rated.any() else 3.0

        n_users, n_items = self.R.shape
        self.b_u_cluster = {}
        self.b_i_cluster = {}

        for cid in np.unique(self.assignments):
            members = np.where(self.assignments == cid)[0]
            b_u, b_i = self._fit_biases(members, n_users, n_items)
            self.b_u_cluster[int(cid)] = b_u
            self.b_i_cluster[int(cid)] = b_i

        self._build_sim_matrix()
        return self

    def _fit_biases(
        self,
        members: np.ndarray,
        n_users: int,
        n_items: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Küme train verisiyle bias-only SGD: r ≈ mu + b_u[u] + b_i[i].
        Yalnızca members satırlarındaki ratingler günceller.
        """
        mu = self.global_mean
        b_u = np.zeros(n_users, dtype=np.float64)
        b_i = np.zeros(n_items, dtype=np.float64)

        triples: List[Tuple[int, int, float]] = []
        for u in members:
            row = self.R[u]
            mask = row > 0
            for i in np.where(mask)[0]:
                triples.append((int(u), int(i), float(row[i])))

        if not triples:
            return b_u, b_i

        rng = np.random.default_rng(self.bias_seed + int(members[0]) if len(members) else 0)
        lr, reg = self.bias_lr, self.bias_reg

        for _ in range(self.bias_epochs):
            rng.shuffle(triples)
            for u, i, r in triples:
                pred = mu + b_u[u] + b_i[i]
                err = r - pred
                b_u[u] += lr * (err - reg * b_u[u])
                b_i[i] += lr * (err - reg * b_i[i])

        return b_u, b_i

    def _baseline(self, u: int, i: int, cid: int) -> float:
        """mu + b_u + b_i (küme c biasları)."""
        bu = self.b_u_cluster.get(cid)
        bi = self.b_i_cluster.get(cid)
        if bu is None or bi is None:
            return self.global_mean
        return float(self.global_mean + bu[u] + bi[i])

    def _cluster_dev_row(self, u: int, cid: int) -> np.ndarray:
        """Kullanıcı u için bias-merkezli sapma vektörü (küme c)."""
        assert self.R is not None
        bu = self.b_u_cluster[cid]
        bi = self.b_i_cluster[cid]
        mu = self.global_mean
        row = self.R[u]
        mask = row > 0
        out = np.zeros_like(row, dtype=np.float64)
        out[mask] = row[mask] - (mu + bu[u] + bi[mask])
        return out

    def _build_sim_matrix(self) -> None:
        assert self.R is not None and self.assignments is not None
        n = self.R.shape[0]
        self.sim = np.zeros((n, n), dtype=np.float32)

        if self.sim_metric not in ('cosine', 'pearson'):
            raise ValueError(
                f"sim_metric '{self.sim_metric}' desteklenmiyor; "
                "'cosine' veya 'pearson' kullanın.",
            )

        for cid in np.unique(self.assignments):
            cid = int(cid)
            members = np.where(self.assignments == cid)[0]
            if len(members) < 2:
                continue

            if self.sim_metric == 'cosine':
                dev_rows = np.stack(
                    [self._cluster_dev_row(int(u), cid) for u in members],
                    axis=0,
                ).astype(np.float64)
                norms = np.linalg.norm(dev_rows, axis=1, keepdims=True)
                norms = np.where(norms < 1e-9, 1e-9, norms)
                R_norm = dev_rows / norms
                cluster_sim = (R_norm @ R_norm.T).astype(np.float32)
                for ii, u in enumerate(members):
                    for jj, v in enumerate(members):
                        if u != v:
                            self.sim[u, v] = cluster_sim[ii, jj]
            else:
                for ii, u in enumerate(members):
                    for jj, v in enumerate(members):
                        if u >= v:
                            continue
                        r = _pearson_on_common(self.R[u], self.R[v])
                        if abs(r) > 0.0:
                            self.sim[u, v] = r
                            self.sim[v, u] = r

        np.fill_diagonal(self.sim, 0.0)

    def predict(self, u: int, i: int) -> float:
        """Tek bir (kullanıcı, film) çifti için tahmin."""
        if self.R is None or self.assignments is None or self.sim is None:
            raise RuntimeError('fit() çağrılmadan predict() kullanılamaz')

        u = int(u)
        i = int(i)
        cid = int(self.assignments[u])
        mu = self.global_mean
        bu = self.b_u_cluster[cid]
        bi = self.b_i_cluster[cid]

        cluster_mask = self.assignments == cid
        rated_mask = self.R[:, i] > 0
        candidates = np.where(cluster_mask & rated_mask)[0]
        candidates = candidates[candidates != u]

        base = self._baseline(u, i, cid)

        if len(candidates) == 0:
            cluster_users = np.where(self.assignments == cid)[0]
            cluster_item_raters = cluster_users[self.R[cluster_users, i] > 0]
            if len(cluster_item_raters) > 0:
                return float(np.clip(self.R[cluster_item_raters, i].mean(), 1.0, 5.0))
            return float(np.clip(base, 1.0, 5.0))

        sims = self.sim[u, candidates].astype(np.float64)
        if len(candidates) > self.k:
            top_k_idx = np.argsort(-np.abs(sims))[: self.k]
            candidates = candidates[top_k_idx]
            sims = sims[top_k_idx]

        devs = np.array(
            [
                float(self.R[v, i] - (mu + bu[v] + bi[i]))
                for v in candidates
            ],
            dtype=np.float64,
        )
        denom = float(np.sum(np.abs(sims))) + 1e-9
        pred = base + float(np.dot(sims, devs)) / denom
        return float(np.clip(pred, 1.0, 5.0))

    def evaluate(
        self,
        test_pairs: Union[Sequence[RatingTriple], np.ndarray],
        *,
        return_sources: bool = False,
    ):
        """test_pairs: (u, i, r_true) listesi veya (N, 3) numpy dizisi."""
        if isinstance(test_pairs, np.ndarray):
            pairs = test_rows_to_pairs(test_pairs)
        else:
            pairs = list(test_pairs)

        errors: List[float] = []
        for u, i, r_true in pairs:
            r_pred = self.predict(u, i)
            errors.append(abs(r_pred - r_true))

        err_arr = np.asarray(errors, dtype=np.float64)
        mae = float(err_arr.mean()) if len(err_arr) else float('nan')
        rmse = float(np.sqrt((err_arr ** 2).mean())) if len(err_arr) else float('nan')
        if return_sources:
            return mae, rmse, []
        return mae, rmse
