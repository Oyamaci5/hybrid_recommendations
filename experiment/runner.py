"""Deney orkestrasyonu iskeleti."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from experiment.builder import build_pipeline
from experiment.logger import setup_logger
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

    def run_offline_assignments(self) -> dict:
        """``pipeline_mode='offline_assignments'`` için kanonik akış.

        Önceden üretilmiş assignment .npy dosyasını tüketir, gray-aware
        CFRecommender + Evaluator zinciriyle tahmin + metrik raporu üretir.
        """
        from experiment.offline_assignments import run as run_offline

        cfg = self.load()
        log = setup_logger(level=cfg.output.log_level)
        log.info("offline_assignments başlıyor: assignment=%s", cfg.offline_assignments.assignment_path)
        out = run_offline(cfg)
        log.info("offline_assignments bitti: %s", out["evaluation_dir"])
        return out

    def run_dummy_smoke(self) -> dict[str, float]:
        """Küçük rastgele matris ile MF + kümeleme zincirinin import/çalışma duman testi."""
        cfg = self.load()
        log = setup_logger(level=cfg.output.log_level)
        parts = build_pipeline(cfg)
        # WNMF nndsvda SVD gereği n_components <= min(n_users, n_items)
        R = np.clip(np.random.rand(25, 32) * 5, 0, 5)
        R[R < 2.0] = 0
        G = R > 0
        parts["mf_model_raw"].fit(R, mask=G.astype(np.float64))
        X = parts["mf_model_raw"].get_user_factors()
        labels, meta = parts["clustering_module"].fit_predict(X, space="latent")
        cc = meta.get("centers")
        log.info(
            "smoke labels %s centers %s",
            labels.shape,
            None if cc is None else getattr(cc, "shape", None),
        )
        return {"fitness_smoke": float(labels.mean())}
