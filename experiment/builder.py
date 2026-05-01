"""Pipeline bileşenlerini oluşturan fabrika."""

from __future__ import annotations

from typing import Any

from models.base import MatrixFactorizationBase
from utils.config import Config


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
