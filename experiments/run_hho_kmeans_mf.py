"""
Run HHO-KMeans-MF experiment on MovieLens 100K.
Set RUN_HHO_KMEANS_EXPERIMENT = True to run when executing this script directly.
"""

# Set to True to run when this script is executed directly; False to skip
RUN_HHO_KMEANS_EXPERIMENT = False

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
from methods.hho_kmeans_mf import HHOKMeansMF


def save_results(results: dict, config: Config, method_name: str, **kwargs):
    """Save experiment results."""
    results_dir = Path("results") / method_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save combined loss curve: HHO final + K-Means
    loss_curve_path = results_dir / "loss_curve.csv"
    with open(loss_curve_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'loss'])
        # HHO final loss at iteration 0
        if len(results['history']['hho_losses']) > 0:
            writer.writerow([0, results['history']['hho_losses'][-1]])
        # K-Means losses
        for i, loss in enumerate(results['history']['kmeans_losses']):
            writer.writerow([i + 1, loss])
    
    # Save HHO-specific loss curve
    hho_loss_curve_path = results_dir / "hho_loss_curve.csv"
    with open(hho_loss_curve_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'loss'])
        for iteration, loss in zip(results['history']['hho_iterations'],
                                  results['history']['hho_losses']):
            writer.writerow([iteration, loss])
    
    # Save K-Means-specific loss curve
    kmeans_loss_curve_path = results_dir / "kmeans_loss_curve.csv"
    with open(kmeans_loss_curve_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'loss'])
        for iteration, loss in zip(results['history']['kmeans_iterations'],
                                  results['history']['kmeans_losses']):
            writer.writerow([iteration, loss])
    
    # Save summary
    summary = {
        'method': results['method'],
        'dataset': results['dataset'],
        'train_split': config.train_split,
        'test_split': config.test_split,
        'n_users': int(results['n_users']),
        'n_items': int(results['n_items']),
        'latent_dim': int(config.latent_dim),
        'n_hawks': int(kwargs.get('n_hawks', 30)),
        'escape_energy_initial': float(kwargs.get('escape_energy_initial', 1.0)),
        'n_clusters_users': int(kwargs.get('n_clusters_users', 0)) if kwargs.get('n_clusters_users') else None,
        'n_clusters_items': int(kwargs.get('n_clusters_items', 0)) if kwargs.get('n_clusters_items') else None,
        'learning_rate': float(kwargs.get('learning_rate', 0.1)),
        'regularization': float(config.regularization),
        'boundary': float(kwargs.get('boundary', 1.0)),
        'hho_iterations': int(kwargs.get('hho_iterations', 50)),
        'kmeans_iterations': int(kwargs.get('kmeans_iterations', 100)),
        'n_iterations': int(kwargs.get('hho_iterations', 50) + kwargs.get('kmeans_iterations', 100)),
        'random_seed': int(config.random_seed),
        'rmse': float(results['rmse']),
        'mae': float(results['mae']),
        'final_loss': float(results['final_loss'])
    }
    
    summary_path = results_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


def run_hho_kmeans_mf_experiment(config: Config = None, n_hawks: int = 30,
                                escape_energy_initial: float = 1.0,
                                n_clusters_users: int = None,
                                n_clusters_items: int = None,
                                learning_rate: float = 0.1, boundary: float = 1.0,
                                hho_iterations: int = 50, kmeans_iterations: int = 100,
                                verbose: bool = True) -> dict:
    """
    Run HHO-KMeans-MF experiment.
    
    Args:
        config: Configuration object (uses default if None)
        n_hawks: Number of hawks in HHO population
        escape_energy_initial: Initial escape energy E0 for HHO
        n_clusters_users: Number of clusters for user embeddings
        n_clusters_items: Number of clusters for item embeddings
        learning_rate: K-Means learning rate
        boundary: Boundary constraint for embeddings
        hho_iterations: Number of HHO iterations
        kmeans_iterations: Number of K-Means iterations
        verbose: Whether to print progress
        
    Returns:
        Dictionary with experiment results
    """
    if config is None:
        config = Config()
    
    set_random_seed(config.random_seed)
    
    train_path = config.get_train_path()
    test_path = config.get_test_path()
    train_ratings, test_ratings = load_train_test_split(train_path, test_path)
    
    n_users, n_items, n_train = get_data_info(train_ratings)
    _, _, n_test = get_data_info(test_ratings)
    
    if verbose:
        print("=" * 60)
        print("HHO-KMeans-MF Experiment")
        print("=" * 60)
        print(f"Dataset: {config.data_dir}")
        print(f"Train: {n_train} ratings, Test: {n_test} ratings")
        print(f"Users: {n_users}, Items: {n_items}")
        print(f"Latent dim: {config.latent_dim}")
        print(f"HHO: {n_hawks} hawks, {hho_iterations} iterations")
        print(f"K-Means: {kmeans_iterations} iterations")
        print("=" * 60)
    
    model = HHOKMeansMF(n_users, n_items, config.latent_dim,
                       n_hawks, escape_energy_initial,
                       n_clusters_users, n_clusters_items,
                       learning_rate, config.regularization,
                       boundary, config.random_seed)
    
    history = model.fit(train_ratings, hho_iterations, kmeans_iterations, verbose=verbose)
    rmse_value, mae_value = model.evaluate(test_ratings)
    
    # Final loss is from K-Means phase
    final_loss = history['kmeans_losses'][-1]
    
    results = {
        'method': 'HHO-KMeans-MF',
        'dataset': 'MovieLens 100K',
        'rmse': rmse_value,
        'mae': mae_value,
        'final_loss': final_loss,
        'n_users': n_users,
        'n_items': n_items,
        'n_train': n_train,
        'n_test': n_test,
        'history': history
    }
    
    save_results(results, config, 'hho_kmeans_mf', n_hawks=n_hawks,
                escape_energy_initial=escape_energy_initial,
                n_clusters_users=n_clusters_users,
                n_clusters_items=n_clusters_items,
                learning_rate=learning_rate, boundary=boundary,
                hho_iterations=hho_iterations, kmeans_iterations=kmeans_iterations)
    
    if verbose:
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        print(f"RMSE: {rmse_value:.6f}")
        print(f"MAE: {mae_value:.6f}")
        print(f"Final Loss: {final_loss:.6f}")
        print("=" * 60)
    
    return results


if __name__ == "__main__":
    if not RUN_HHO_KMEANS_EXPERIMENT:
        print("HHO-KMeans-MF experiment skipped (RUN_HHO_KMEANS_EXPERIMENT=False). Set to True to run.")
        exit(0)
    config = Config()
    run_hho_kmeans_mf_experiment(config, verbose=True)

