"""
Tek başına HHO-EO-MF (EO kılavuzlu HHO) çalıştırma script'i.

MovieLens 1M için:
  - Veri yolları ve diğer temel hiperparametreler Config üzerinden geliyor.
  - Asıl HHO-EO-MF akışı, var olan run_hho_eo_mf_ml1m fonksiyonunu kullanır.
"""

import sys
from pathlib import Path

# Proje kökünü path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import Config
from experiments.run_ml1m_experiments import run_hho_eo_mf_ml1m


def main() -> None:
    """Sadece HHO-EO-MF deneyini çalıştır."""
    config = Config(
        data_dir="data/ml-1m",
        train_split="train.dat",
        test_split="test.dat",
        latent_dim=10,
        learning_rate=0.01,
        regularization=0.01,
        n_iterations=100,
        random_seed=42,
    )

    # Varsayılan parametrelerle HHO-EO-MF çalıştır
    run_hho_eo_mf_ml1m(
        config=config,
        n_agents=60,
        escape_energy_initial=1.5,
        boundary=1.0,
        n_iterations=150,
        verbose=True,
    )


if __name__ == "__main__":
    main()

