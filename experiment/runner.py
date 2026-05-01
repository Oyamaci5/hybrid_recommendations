"""Deney orkestrasyonu iskeleti."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from data.loader import DatasetLoader
from experiment.builder import build_pipeline, fit_pipeline
from experiment.logger import setup_logger
from preprocess.preprocessor import Preprocessor
from recommender.evaluator import Evaluator
from utils.io_utils import save_results
from utils.config import Config, load_config


@dataclass
class ExperimentRunner:
    """Tek bir YAML konfigürasyonunu yükler ve yapı taşlarını döndürür."""

    config_path: str = "configs/default.yaml"
    overrides: list[str] | None = None

    def load(self) -> Config:
        return load_config(self.config_path, self.overrides)

    def components(self) -> dict[str, Any]:
        cfg = self.load()
        setup_logger(level=cfg.output.log_level)
        return build_pipeline(cfg)

    def run_dummy_smoke(self) -> dict[str, float]:
        """Küçük rastgele matris ile MF + kümeleme zincirinin import/çalışma duman testi."""
        cfg = self.load()
        log = setup_logger(level=cfg.output.log_level)
        # WNMF nndsvda SVD gereği n_components <= min(n_users, n_items)
        R = np.clip(np.random.rand(25, 32) * 5, 0, 5)
        R[R < 2.0] = 0
        G = (R > 0).astype(np.float64)
        parts = fit_pipeline(cfg, R, G)
        labels = parts["labels"]
        meta = parts["meta"]
        cc = meta.get("centers")
        log.info(
            "smoke labels %s centers %s",
            labels.shape,
            None if cc is None else getattr(cc, "shape", None),
        )
        return {"fitness_smoke": float(labels.mean())}

    def run(self) -> dict:
        cfg = self.load()
        log = setup_logger(level=cfg.output.log_level)

        log.info("Veri yükleniyor: %s", cfg.data.dataset)
        loader = DatasetLoader(cfg.data)
        R_sparse, _user_map, _item_map = loader.load()

        log.info(
            "Preprocess: norm=%s missing=%s bias=%s",
            cfg.preprocess.normalization,
            cfg.preprocess.missing_strategy,
            cfg.preprocess.apply_bias_removal,
        )
        preprocessor = Preprocessor(cfg.preprocess)
        R_proc, mask, bias = preprocessor.fit_transform(R_sparse)

        kf = KFold(n_splits=cfg.data.n_folds, shuffle=True, random_state=cfg.data.random_state)
        evaluator = Evaluator(cfg.evaluation)
        fold_results = []
        observed_idx = np.argwhere(mask > 0)

        for fold, (train_idx, test_idx) in enumerate(kf.split(observed_idx), 1):
            log.info("Fold %d/%d", fold, cfg.data.n_folds)

            train_pairs = observed_idx[train_idx]
            test_pairs = observed_idx[test_idx]

            mask_train = np.zeros_like(mask, dtype=np.float64)
            mask_train[train_pairs[:, 0], train_pairs[:, 1]] = 1.0
            R_train = R_proc * mask_train

            mask_test = np.zeros_like(mask, dtype=np.float64)
            mask_test[test_pairs[:, 0], test_pairs[:, 1]] = 1.0

            parts = fit_pipeline(cfg, R_train, mask_train)
            fold_metrics = evaluator.evaluate_fold(
                recommender=parts["recommender"],
                R_test=R_proc,
                mask_test=mask_test,
                preprocessor=preprocessor,
                bias=bias,
            )
            fold_results.append(fold_metrics)
            log.info(
                "Fold %d -> MAE=%.4f RMSE=%.4f",
                fold,
                fold_metrics.get("mae", -1.0),
                fold_metrics.get("rmse", -1.0),
            )

        summary = evaluator.aggregate(fold_results)
        summary["experiment_id"] = cfg.experiment_id()
        summary["dataset"] = cfg.data.dataset
        summary["model"] = cfg.model.name
        summary["optimizer"] = cfg.optimizer.name
        summary["clustering"] = cfg.clustering.algorithm
        summary["n_clusters"] = cfg.clustering.n_clusters
        summary["n_components"] = cfg.model.n_components
        summary["normalization"] = cfg.preprocess.normalization

        out_dir = Path(cfg.output.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_results(summary, out_dir / f"{cfg.experiment_id()}.json")
        log.info("Sonuç kaydedildi: %s", cfg.experiment_id())
        log.info(
            "MAE=%.4f±%.4f RMSE=%.4f±%.4f",
            summary.get("mae_mean", 0.0),
            summary.get("mae_std", 0.0),
            summary.get("rmse_mean", 0.0),
            summary.get("rmse_std", 0.0),
        )
        return summary

    def run_offline_assignments(self) -> dict[str, Any]:
        raise NotImplementedError(
            "offline_assignments mode is not wired in this revision. "
            "Use cluster/meta_mf_tune or add artifact paths and adapter."
        )
