"""Pipeline bileşenlerini oluşturan fabrika."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from models.base import MatrixFactorizationBase
from utils.config import Config


log = logging.getLogger(__name__)

MODEL_MAP = {
    "wnmf": "models.wnmf.WNMF",
    "svd": "models.svd.SVDModel",
    "pmf": "models.pmf.PMFModel",
}


def _instantiate_matrix_model(cfg: Config) -> MatrixFactorizationBase:
    name = cfg.model.name.strip().lower()
    dotted = MODEL_MAP.get(name)
    if dotted is None:
        raise ValueError(f"MODEL_MAP içinde '{name}' tanımlı değil")
    module_path, class_name = dotted.rsplit(".", 1)
    mod = __import__(module_path, fromlist=[class_name])
    cls = getattr(mod, class_name)
    return cls(cfg.model)


def build_pipeline(cfg: Config) -> dict[str, Any]:
    from clustering.module import ClusteringModule
    from clustering.problem import ClusteringProblem
    from recommender.cf_recommender import CFRecommender
    from recommender.mf_only_recommender import MFOnlyRecommender

    model_obj = _instantiate_matrix_model(cfg)
    clus = ClusteringModule(
        n_clusters=int(cfg.clustering.n_clusters),
        algorithm=cfg.clustering.algorithm,
        fcm_m=float(cfg.clustering.fcm_m),
        kmeans_inner_iter=int(getattr(cfg.clustering, "kmeans_inner_iter", 5)),
        space=cfg.clustering.space,
        init_method=getattr(cfg.clustering, "init_method", "kmeans++"),
        mkmeans_init_max_iter=int(getattr(cfg.clustering, "mkmeans_init_max_iter", 50)),
    )
    k = int(cfg.clustering.n_clusters)

    return {
        "config": cfg,
        "mf_model_raw": model_obj,
        "clustering_module": clus,
        "problem_factory": lambda X, metric="pearson": ClusteringProblem(X, k, metric=metric),
        "build_cf": lambda R, lbl, cen, top_k=None: CFRecommender(
            R, lbl, cen, top_k=top_k or cfg.recommender.n_neighbors
        ),
        "build_mf_reco": lambda trained: MFOnlyRecommender(trained),
    }


def fit_pipeline(cfg: Config, R_train: np.ndarray, mask_train: np.ndarray) -> dict[str, Any]:
    from clustering.module import ClusteringModule
    from optimizers.factory import build_optimizer
    from preprocess.gray_sheep import GraySheepDetector
    from recommender.cf_recommender import CFRecommender

    if cfg.clustering.space == "latent":
        mf_model = _instantiate_matrix_model(cfg)
        mf_model.fit(R_train, mask=mask_train)
        user_factors = mf_model.get_user_factors()
        pca_reducer = None
    elif cfg.clustering.space == "pca":
        from data.pca_reducer import PCAReducer

        mf_model = None
        pca_reducer = PCAReducer(cfg.pca)
        user_factors = pca_reducer.fit_transform(R_train)
        summary = pca_reducer.explained_variance_summary()
        log.info(
            "PCA: %d components, total explained variance=%.3f",
            summary["n_components"],
            summary["total_explained"],
        )
    elif cfg.clustering.space == "raw":
        mf_model = None
        pca_reducer = None
        user_factors = R_train
    else:
        raise ValueError("clustering.space must be one of: raw, latent, pca")

    if cfg.clustering.algorithm in ("meta_kmeans", "meta_fcm"):
        optimizer = build_optimizer(cfg.optimizer)
    else:
        optimizer = None

    clus = ClusteringModule(
        optimizer=optimizer,
        n_clusters=int(cfg.clustering.n_clusters),
        algorithm=cfg.clustering.algorithm,
        fcm_m=float(cfg.clustering.fcm_m),
        kmeans_inner_iter=int(getattr(cfg.clustering, "kmeans_inner_iter", 5)),
        space=cfg.clustering.space,
        init_method=getattr(cfg.clustering, "init_method", "kmeans++"),
        mkmeans_init_max_iter=int(getattr(cfg.clustering, "mkmeans_init_max_iter", 50)),
    )
    labels, meta = clus.fit_predict(user_factors, space=cfg.clustering.space)
    centroids = meta.get("centers")

    gs_detector = None
    gray_mask = None
    if cfg.gray_sheep.enabled:
        gs_detector = GraySheepDetector(cfg.gray_sheep)
        gs_detector.fit(R_train)
        gray_mask = gs_detector.get_mask()

    gray_strategy = cfg.recommender.gray_sheep_strategy
    if gray_strategy == "multi_cluster":
        gray_strategy = "same_cluster"

    recommender = CFRecommender(
        R_train,
        labels,
        centroids,
        top_k=cfg.recommender.n_neighbors,
        mf_model=mf_model,
        pca_reducer=pca_reducer,
        gray_mask=gray_mask,
        gray_strategy=gray_strategy,
        membership=meta.get("membership"),
        recommend_top_n=cfg.recommender.top_n,
    )

    return {
        "mf_model": mf_model,
        "pca_reducer": pca_reducer,
        "clustering": clus,
        "recommender": recommender,
        "labels": labels,
        "centroids": centroids,
        "meta": meta,
        "gs_detector": gs_detector,
    }
