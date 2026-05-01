from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json

import yaml


@dataclass
class DataConfig:
    dataset: str = "ml100k"
    path: str = "data"
    min_user_ratings: int = 5
    min_item_ratings: int = 3
    test_size: float = 0.2
    n_folds: int = 5
    random_state: int = 42


@dataclass
class PreprocessConfig:
    normalization: str = "none"
    missing_strategy: str = "ignore"
    apply_bias_removal: bool = False
    min_user_ratings: int = 5
    min_item_ratings: int = 3


@dataclass
class GraySheepConfig:
    enabled: bool = False
    method: str = "pcc_std"
    threshold: float = 0.5


@dataclass
class ModelConfig:
    name: str = "wnmf"
    n_components: int = 20
    alpha: float = 0.01
    beta: float = 0.01
    max_iter: int = 200
    tol: float = 1e-4
    init: str = "nndsvda"
    random_state: int = 42
    lr: float = 0.005
    reg: float = 0.02
    n_epochs: int = 50
    use_bias: bool = True
    sigma_sq: float = 1.0
    sigma_U_sq: float = 1.0
    sigma_V_sq: float = 1.0
    n_iter: int = 200


@dataclass
class OptimizerConfig:
    name: str = "pso"
    source: str = "mealpy"
    n_agents: int = 30
    n_iter: int = 100
    seed: int = 42
    algo_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusteringConfig:
    algorithm: str = "kmeans"
    n_clusters: int = 50
    space: str = "latent"
    fcm_m: float = 2.0
    kmeans_inner_iter: int = 5
    init_method: str = "kmeans++"
    mkmeans_init_max_iter: int = 50


@dataclass
class PCAConfig:
    n_components: int = 20
    whiten: bool = False
    random_state: int = 42


@dataclass
class RecommenderConfig:
    top_n: int = 10
    gray_sheep_strategy: str = "multi_cluster"
    n_neighbors: int = 30


@dataclass
class EvaluationConfig:
    metrics: list[str] = field(default_factory=lambda: ["mae", "rmse", "precision", "recall", "f1", "ndcg"])
    at_n: list[int] = field(default_factory=lambda: [5, 10])


@dataclass
class OutputConfig:
    results_dir: str = "results"
    save_model: bool = False
    save_plots: bool = True
    log_level: str = "INFO"


@dataclass
class MFTuningConfig:
    param_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "n_components": [5, 100],
            "alpha": [1e-4, 1.0],
            "beta": [1e-4, 1.0],
            "lr": [1e-4, 0.1],
            "reg": [1e-4, 0.5],
            "sigma_sq": [1e-4, 2.0],
            "sigma_U_sq": [1e-4, 2.0],
            "sigma_V_sq": [1e-4, 2.0],
        }
    )


@dataclass
class Config:
    pipeline_mode: str = "cluster"
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    gray_sheep: GraySheepConfig = field(default_factory=GraySheepConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    recommender: RecommenderConfig = field(default_factory=RecommenderConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    mf_tuning: MFTuningConfig = field(default_factory=MFTuningConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        p = Path(path)
        if not p.exists():
            return cls()
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        cfg = cls(
            pipeline_mode=data.get("pipeline_mode", "cluster"),
            data=DataConfig(**data.get("data", {})),
            preprocess=PreprocessConfig(**data.get("preprocess", {})),
            gray_sheep=GraySheepConfig(**data.get("gray_sheep", {})),
            model=ModelConfig(**data.get("model", {})),
            optimizer=OptimizerConfig(**{k: v for k, v in data.get("optimizer", {}).items() if k != data.get("optimizer", {}).get("name", "pso")}),
            clustering=ClusteringConfig(**data.get("clustering", {})),
            pca=PCAConfig(**data.get("pca", {})),
            recommender=RecommenderConfig(**data.get("recommender", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
            output=OutputConfig(**data.get("output", {})),
            mf_tuning=MFTuningConfig(**data.get("mf_tuning", {})),
        )
        opt_section = data.get("optimizer", {})
        algo_name = cfg.optimizer.name
        cfg.optimizer.algo_params = opt_section.get(algo_name, cfg.optimizer.algo_params or {})
        return cfg

    def override(self, key: str, value: Any) -> "Config":
        section, field_name = key.split(".", 1)
        target = getattr(self, section)
        current = getattr(target, field_name)
        if isinstance(current, bool):
            casted = str(value).lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int):
            casted = int(value)
        elif isinstance(current, float):
            casted = float(value)
        elif isinstance(current, list):
            casted = json.loads(value) if isinstance(value, str) and value.startswith("[") else [value]
        elif isinstance(current, dict):
            casted = json.loads(value) if isinstance(value, str) else value
        else:
            casted = value
        setattr(target, field_name, casted)
        return self

    def experiment_id(self) -> str:
        return f"{self.data.dataset}__{self.model.name}__k{self.model.n_components}__{self.optimizer.name}__c{self.clustering.n_clusters}__{self.preprocess.normalization}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(yaml_path: str = "configs/default.yaml", overrides: list[str] | None = None) -> Config:
    cfg = Config.from_yaml(yaml_path)
    for override in overrides or []:
        key, value = override.split("=", 1)
        if key == "pipeline_mode":
            cfg.pipeline_mode = value
        elif "." in key:
            cfg.override(key, value)
    return cfg
