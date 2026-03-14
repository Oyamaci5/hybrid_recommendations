"""
HHO-SGD-MF experiment runner.
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
from methods.hho_sgd_mf import HHOSGDMF


def run_hho_sgd_mf_experiment(config: Config = None, n_hawks: int = 30,
                              escape_energy_initial: float = 1.0,
                              learning_rate: float = 0.01, boundary: float = 1.0,
                              hho_iterations: int = 50, sgd_iterations: int = 100,
                              verbose: bool = True) -> dict:
    """
    Run HHO-SGD-MF experiment.
    
    Args:
        config: Configuration object (uses default if None)
        n_hawks: Number of hawks in HHO population
        escape_energy_initial: Initial escape energy E0 for HHO
        learning_rate: SGD learning rate
        boundary: Boundary constraint for embeddings
        hho_iterations: Number of HHO iterations
        sgd_iterations: Number of SGD iterations
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
        print("HHO-SGD-MF Experiment")
        print("=" * 60)
        print(f"Dataset: {config.data_dir}")
        print(f"Train split: {config.train_split} ({n_train} ratings)")
        print(f"Test split: {config.test_split} ({n_test} ratings)")
        print(f"Users: {n_users}, Items: {n_items}")
        print(f"Latent dimension: {config.latent_dim}")
        print(f"Number of hawks: {n_hawks}")
        print(f"Escape energy initial: {escape_energy_initial}")
        print(f"Learning rate: {learning_rate}")
        print(f"Regularization: {config.regularization}")
        print(f"Boundary: {boundary}")
        print(f"HHO iterations: {hho_iterations}")
        print(f"SGD iterations: {sgd_iterations}")
        print(f"Random seed: {config.random_seed}")
        print("=" * 60)
    
    # Initialize and train model
    model = HHOSGDMF(
        n_users=n_users,
        n_items=n_items,
        latent_dim=config.latent_dim,
        n_hawks=n_hawks,
        escape_energy_initial=escape_energy_initial,
        learning_rate=learning_rate,
        regularization=config.regularization,
        boundary=boundary,
        random_seed=config.random_seed
    )
    
    # Train (HHO initialization + SGD refinement)
    if verbose:
        print("\nTraining...")
    
    history = model.fit(train_ratings, hho_iterations=hho_iterations,
                       sgd_iterations=sgd_iterations, verbose=verbose)
    
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
        'n_hawks': n_hawks,
        'escape_energy_initial': escape_energy_initial,
        'learning_rate': learning_rate,
        'boundary': boundary,
        'hho_iterations': hho_iterations,
        'sgd_iterations': sgd_iterations
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        print(f"RMSE: {rmse_value:.6f}")
        print(f"MAE:  {mae_value:.6f}")
        print("=" * 60)
    
    # Save results
    save_results(results, config, n_hawks, escape_energy_initial, learning_rate,
                 boundary, hho_iterations, sgd_iterations)
    
    return results


def save_results(results: dict, config: Config, n_hawks: int,
                 escape_energy_initial: float, learning_rate: float,
                 boundary: float, hho_iterations: int, sgd_iterations: int):
    """
    Save experiment results to files.
    
    Args:
        results: Dictionary with experiment results
        config: Configuration object
        n_hawks: Number of hawks
        escape_energy_initial: Initial escape energy
        learning_rate: Learning rate
        boundary: Boundary constraint
        hho_iterations: Number of HHO iterations
        sgd_iterations: Number of SGD iterations
    """
    # Create results directory
    results_dir = Path("results") / "hho_sgd_mf"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save HHO phase loss curve
    hho_loss_curve_path = results_dir / "hho_loss_curve.csv"
    with open(hho_loss_curve_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'loss'])
        for iteration, loss in zip(results['history']['hho_iterations'],
                                  results['history']['hho_losses']):
            writer.writerow([iteration, loss])
    
    # Save SGD phase loss curve
    sgd_loss_curve_path = results_dir / "sgd_loss_curve.csv"
    with open(sgd_loss_curve_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'loss'])
        for iteration, loss in zip(results['history']['sgd_iterations'],
                                  results['history']['sgd_losses']):
            writer.writerow([iteration, loss])
    
    # Save combined loss curve (for comparison with other methods)
    # Use HHO final loss as iteration 0, then SGD losses
    combined_loss_curve_path = results_dir / "loss_curve.csv"
    with open(combined_loss_curve_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'loss'])
        # HHO final loss
        if len(results['history']['hho_losses']) > 0:
            writer.writerow([0, results['history']['hho_losses'][-1]])
        # SGD losses (starting from iteration 1)
        for i, (iteration, loss) in enumerate(zip(results['history']['sgd_iterations'],
                                                 results['history']['sgd_losses'])):
            writer.writerow([iteration + 1, loss])
    
    # Save summary to JSON
    summary = {
        'method': 'HHO-SGD-MF',
        'dataset': config.data_dir,
        'train_split': config.train_split,
        'test_split': config.test_split,
        'n_users': int(results['n_users']),
        'n_items': int(results['n_items']),
        'latent_dim': int(config.latent_dim),
        'n_hawks': int(n_hawks),
        'escape_energy_initial': float(escape_energy_initial),
        'learning_rate': float(learning_rate),
        'regularization': float(config.regularization),
        'boundary': float(boundary),
        'hho_iterations': int(hho_iterations),
        'sgd_iterations': int(sgd_iterations),
        'n_iterations': int(hho_iterations + sgd_iterations),
        'random_seed': int(config.random_seed),
        'rmse': float(results['rmse']),
        'mae': float(results['mae'])
    }
    
    summary_path = results_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    # Run HHO-SGD-MF experiment with default parameters
    results = run_hho_sgd_mf_experiment(verbose=True)

