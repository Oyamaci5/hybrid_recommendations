"""
Baseline experiment runner for MF-SGD.
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
from methods.mf_sgd import MFSGD


def run_baseline_experiment(config: Config = None, verbose: bool = True) -> dict:
    """
    Run baseline MF-SGD experiment.
    
    Args:
        config: Configuration object (uses default if None)
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
        print("MF-SGD Baseline Experiment")
        print("=" * 60)
        print(f"Dataset: {config.data_dir}")
        print(f"Train split: {config.train_split} ({n_train} ratings)")
        print(f"Test split: {config.test_split} ({n_test} ratings)")
        print(f"Users: {n_users}, Items: {n_items}")
        print(f"Latent dimension: {config.latent_dim}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Regularization: {config.regularization}")
        print(f"Iterations: {config.n_iterations}")
        print(f"Random seed: {config.random_seed}")
        print("=" * 60)
    
    # Initialize and train model
    model = MFSGD(
        n_users=n_users,
        n_items=n_items,
        latent_dim=config.latent_dim,
        learning_rate=config.learning_rate,
        regularization=config.regularization,
        random_seed=config.random_seed
    )
    
    # Train
    if verbose:
        print("\nTraining...")
    
    history = model.fit(train_ratings, n_iterations=config.n_iterations, verbose=verbose)
    
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
        'history': history
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        print(f"RMSE: {rmse_value:.6f}")
        print(f"MAE:  {mae_value:.6f}")
        print("=" * 60)
    
    # Save results
    save_results(results, config)
    
    return results


def save_results(results: dict, config: Config):
    """
    Save experiment results to files.
    
    Args:
        results: Dictionary with experiment results
        config: Configuration object
    """
    # Create results directory
    results_dir = Path("results") / "mf_sgd"
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
        'method': 'MF-SGD',
        'dataset': config.data_dir,
        'train_split': config.train_split,
        'test_split': config.test_split,
        'n_users': int(results['n_users']),
        'n_items': int(results['n_items']),
        'latent_dim': int(config.latent_dim),
        'learning_rate': float(config.learning_rate),
        'regularization': float(config.regularization),
        'n_iterations': int(config.n_iterations),
        'random_seed': int(config.random_seed),
        'rmse': float(results['rmse']),
        'mae': float(results['mae'])
    }
    
    summary_path = results_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    # Run baseline experiment with default config
    results = run_baseline_experiment(verbose=True)

