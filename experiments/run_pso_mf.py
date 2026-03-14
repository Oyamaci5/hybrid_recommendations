"""
PSO-MF experiment runner.
"""

import sys
import os
import json
import csv
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from core.utils import load_train_test_split, get_data_info
from core.config import Config, set_random_seed
from core.metrics import evaluate_predictions
from methods.pso_mf import PSOMF


def run_pso_mf_experiment(config: Config = None, swarm_size: int = 30,
                          inertia_weight_max: float = 0.9, inertia_weight_min: float = 0.2,
                          cognitive_coeff: float = 2.0, social_coeff: float = 2.0,
                          boundary: float = 1.0, n_iterations: int = 100,
                          verbose: bool = True) -> dict:
    """
    Run PSO-MF experiment.
    
    Args:
        config: Configuration object (uses default if None)
        swarm_size: Number of particles in swarm
        inertia_weight_max: Maximum PSO inertia weight (wMax)
        inertia_weight_min: Minimum PSO inertia weight (wMin)
        cognitive_coeff: PSO cognitive coefficient
        social_coeff: PSO social coefficient
        boundary: Boundary constraint for embeddings
        n_iterations: Number of PSO iterations
        verbose: Whether to print results
        
    Returns:
        Dictionary with experiment results
    """
    if config is None:
        config = Config()
    
    # Set random seed for reproducibility
    set_random_seed(config.random_seed)
    
    # Load data
    train_path = config.get_train_path()
    test_path = config.get_test_path()
    
    train_ratings, test_ratings = load_train_test_split(train_path, test_path)
    
    # Get data info
    n_users, n_items, n_train = get_data_info(train_ratings)
    _, _, n_test = get_data_info(test_ratings)
    
    if verbose:
        print("=" * 60)
        print("PSO-MF Experiment")
        print("=" * 60)
        print(f"Dataset: {config.data_dir}")
        print(f"Train split: {config.train_split} ({n_train} ratings)")
        print(f"Test split: {config.test_split} ({n_test} ratings)")
        print(f"Users: {n_users}, Items: {n_items}")
        print(f"Latent dimension: {config.latent_dim}")
        print(f"Swarm size: {swarm_size}")
        print(f"Inertia weight max: {inertia_weight_max}")
        print(f"Inertia weight min: {inertia_weight_min}")
        print(f"Cognitive coefficient: {cognitive_coeff}")
        print(f"Social coefficient: {social_coeff}")
        print(f"Regularization: {config.regularization}")
        print(f"Boundary: {boundary}")
        print(f"Iterations: {n_iterations}")
        print(f"Random seed: {config.random_seed}")
        print("=" * 60)
    
    # Initialize and train model
    model = PSOMF(
        n_users=n_users,
        n_items=n_items,
        latent_dim=config.latent_dim,
        swarm_size=swarm_size,
        inertia_weight_max=inertia_weight_max,
        inertia_weight_min=inertia_weight_min,
        cognitive_coeff=cognitive_coeff,
        social_coeff=social_coeff,
        regularization=config.regularization,
        boundary=boundary,
        random_seed=config.random_seed
    )
    
    # Train
    if verbose:
        print("\nTraining...")
    
    history = model.fit(train_ratings, n_iterations=n_iterations, verbose=verbose)
    
    # Evaluate
    if verbose:
        print("\nEvaluating...")
    
    rmse_value, mae_value = model.evaluate(test_ratings)
    
    # Results
    results = {
        'rmse': rmse_value,
        'mae': mae_value,
        'n_users': n_users,
        'n_items': n_items,
        'n_train': n_train,
        'n_test': n_test,
        'config': config,
        'history': history,
        'swarm_size': swarm_size,
        'inertia_weight_max': inertia_weight_max,
        'inertia_weight_min': inertia_weight_min,
        'cognitive_coeff': cognitive_coeff,
        'social_coeff': social_coeff,
        'boundary': boundary
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        print(f"RMSE: {rmse_value:.6f}")
        print(f"MAE:  {mae_value:.6f}")
        print("=" * 60)
    
    # Save results
    save_results(results, config, swarm_size, inertia_weight_max, inertia_weight_min,
                 cognitive_coeff, social_coeff, boundary, n_iterations)
    
    return results


def save_results(results: dict, config: Config, swarm_size: int,
                 inertia_weight_max: float, inertia_weight_min: float,
                 cognitive_coeff: float, social_coeff: float, boundary: float,
                 n_iterations: int):
    """
    Save experiment results to files.
    
    Args:
        results: Dictionary with experiment results
        config: Configuration object
        swarm_size: Swarm size
        inertia_weight_max: Maximum inertia weight
        inertia_weight_min: Minimum inertia weight
        cognitive_coeff: Cognitive coefficient
        social_coeff: Social coefficient
        boundary: Boundary constraint
        n_iterations: Number of iterations
    """
    # Create results directory
    results_dir = Path("results") / "pso_mf"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save loss curve to CSV
    loss_curve_path = results_dir / "loss_curve.csv"
    with open(loss_curve_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'loss'])
        for iteration, loss in zip(results['history']['iterations'], results['history']['losses']):
            writer.writerow([iteration, loss])
    
    # Save summary to JSON (convert numpy types to native Python types)
    summary = {
        'method': 'PSO-MF',
        'dataset': config.data_dir,
        'train_split': config.train_split,
        'test_split': config.test_split,
        'n_users': int(results['n_users']),
        'n_items': int(results['n_items']),
        'latent_dim': int(config.latent_dim),
        'swarm_size': int(swarm_size),
        'inertia_weight_max': float(inertia_weight_max),
        'inertia_weight_min': float(inertia_weight_min),
        'cognitive_coeff': float(cognitive_coeff),
        'social_coeff': float(social_coeff),
        'regularization': float(config.regularization),
        'boundary': float(boundary),
        'n_iterations': int(n_iterations),
        'random_seed': int(config.random_seed),
        'rmse': float(results['rmse']),
        'mae': float(results['mae'])
    }
    
    summary_path = results_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


# Set to True to run PSO-MF when this script is executed directly; False to skip
RUN_PSO_MF_EXPERIMENT = False

if __name__ == "__main__":
    if not RUN_PSO_MF_EXPERIMENT:
        print("PSO-MF experiment skipped (RUN_PSO_MF_EXPERIMENT=False). Set to True to run.")
        exit(0)
    # Run PSO-MF experiment with default parameters
    results = run_pso_mf_experiment(verbose=True)
