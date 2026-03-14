"""
Run methods (MF-SGD, HHO-SGD-MF, COA-SGD-MF, optionally PSO-MF and K-Means) on MovieLens 1M.
PSO-MF and K-Means methods are skipped by default (set to True to run them).
"""

# Set to True to run these methods; False to skip (use existing results for comparison table)
RUN_PSO_MF = False
RUN_HHO_KMEANS_MF = False
RUN_COA_KMEANS_MF = False

import sys
import os
import json
import csv
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from core.utils import load_train_test_split, get_data_info, create_train_test_split, load_ratings
from core.config import Config, set_random_seed
from core.metrics import evaluate_predictions
from methods.mf_sgd import MFSGD
from methods.pso_mf import PSOMF
from methods.hho_sgd_mf import HHOSGDMF
from methods.hho_eo_mf import HHOEOMF
from methods.hho_dbo_mf import HHODBOMF
from methods.hho_ssa_mf import HHOSSAMF
from methods.coa_sgd_mf import COASGDMF
from methods.hho_kmeans_mf import HHOKMeansMF
from methods.coa_kmeans_mf import COAKMeansMF
from methods.hybrid_coa_sma_mf import HybridCOASMAMF
from methods.goa_kmeans_mf import GOAKMeansMF
from optimizers.sgd import SGDOptimizer


def run_mf_sgd_ml1m(config: Config, verbose: bool = True) -> dict:
    """Run MF-SGD on MovieLens 1M."""
    set_random_seed(config.random_seed)
    
    train_path = config.get_train_path()
    test_path = config.get_test_path()
    train_ratings, test_ratings = load_train_test_split(train_path, test_path)
    
    n_users, n_items, n_train = get_data_info(train_ratings)
    _, _, n_test = get_data_info(test_ratings)
    
    if verbose:
        print("=" * 60)
        print("MF-SGD on MovieLens 1M")
        print("=" * 60)
        print(f"Train: {n_train} ratings, Test: {n_test} ratings")
        print(f"Users: {n_users}, Items: {n_items}")
        print(f"Latent dim: {config.latent_dim}, Iterations: {config.n_iterations}")
        print("=" * 60)
    
    model = MFSGD(n_users, n_items, config.latent_dim,
                 config.learning_rate, config.regularization, config.random_seed)
    
    history = model.fit(train_ratings, config.n_iterations, verbose=verbose)
    rmse_value, mae_value = model.evaluate(test_ratings)
    
    results = {
        'method': 'MF-SGD',
        'dataset': 'MovieLens 1M',
        'rmse': rmse_value,
        'mae': mae_value,
        'final_loss': history['losses'][-1],
        'n_users': n_users,
        'n_items': n_items,
        'n_train': n_train,
        'n_test': n_test,
        'history': history
    }
    
    save_results_ml1m(results, config, 'mf_sgd')
    return results


def cross_validate_latent_dim_ml1m(
    config: Config,
    latent_dims: list[int] | tuple[int, ...] = (5, 10, 20, 50),
    n_folds: int = 5,
    verbose: bool = True,
) -> tuple[int, dict]:
    """
    Simple k-fold cross-validation on latent dimension using MF-SGD.

    Uses only the training split (e.g. train.dat). For each candidate latent
    dimension, performs k-fold CV on the training ratings and computes the
    average validation RMSE. Returns the best latent dimension and the RMSE
    per candidate.
    """
    # Load full training ratings
    train_file = os.path.join(config.data_dir, config.train_split)
    ratings = load_ratings(train_file)

    if ratings.shape[0] == 0:
        raise ValueError("No ratings loaded for cross-validation.")

    n_users, n_items, _ = get_data_info(ratings)

    n_samples = ratings.shape[0]
    indices = np.random.permutation(n_samples)

    fold_sizes = [n_samples // n_folds] * n_folds
    for i in range(n_samples % n_folds):
        fold_sizes[i] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        start = current
        stop = current + fold_size
        folds.append(indices[start:stop])
        current = stop

    cv_results: dict[int, float] = {}

    if verbose:
        print("\n" + "=" * 60)
        print("LATENT DIMENSION CROSS-VALIDATION (MF-SGD, MovieLens 1M)")
        print("=" * 60)
        print(f"Candidates (k): {list(latent_dims)}")
        print(f"Number of folds: {n_folds}")
        print(f"Total ratings (from {config.train_split}): {n_samples}")
        print("=" * 60)

    for k in latent_dims:
        fold_rmses = []
        if verbose:
            print(f"\n[CV] Latent dim k={k}")

        for fold_idx in range(n_folds):
            val_idx = folds[fold_idx]
            train_idx = np.concatenate(
                [folds[i] for i in range(n_folds) if i != fold_idx]
            )

            train_ratings = ratings[train_idx]
            val_ratings = ratings[val_idx]

            model = MFSGD(
                n_users,
                n_items,
                k,
                config.learning_rate,
                config.regularization,
                config.random_seed + fold_idx,
            )

            model.fit(train_ratings, config.n_iterations, verbose=False)
            rmse_value, _ = model.evaluate(val_ratings)
            fold_rmses.append(rmse_value)

            if verbose:
                print(f"  Fold {fold_idx + 1}/{n_folds}: RMSE = {rmse_value:.6f}")

        mean_rmse = float(np.mean(fold_rmses))
        cv_results[int(k)] = mean_rmse

        if verbose:
            print(f"  -> Mean CV RMSE for k={k}: {mean_rmse:.6f}")

    # Select best k (lowest mean RMSE)
    best_k = min(cv_results, key=cv_results.get)

    if verbose:
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION SUMMARY (latent dimension)")
        print("=" * 60)
        for k in sorted(cv_results.keys()):
            print(f"k={k:3d} -> mean RMSE = {cv_results[k]:.6f}")
        print("-" * 60)
        print(f"Best latent dimension (k): {best_k} "
              f"(RMSE = {cv_results[best_k]:.6f})")
        print("=" * 60)

    return best_k, cv_results


def run_pso_mf_ml1m(config: Config, swarm_size: int = 30,
                    inertia_weight_max: float = 0.9, inertia_weight_min: float = 0.2,
                    cognitive_coeff: float = 2.0, social_coeff: float = 2.0,
                    boundary: float = 1.0, n_iterations: int = 100,
                    verbose: bool = True) -> dict:
    """Run PSO-MF on MovieLens 1M."""
    set_random_seed(config.random_seed)
    
    train_path = config.get_train_path()
    test_path = config.get_test_path()
    train_ratings, test_ratings = load_train_test_split(train_path, test_path)
    
    n_users, n_items, n_train = get_data_info(train_ratings)
    _, _, n_test = get_data_info(test_ratings)
    
    if verbose:
        print("=" * 60)
        print("PSO-MF on MovieLens 1M")
        print("=" * 60)
        print(f"Train: {n_train} ratings, Test: {n_test} ratings")
        print(f"Users: {n_users}, Items: {n_items}")
        print(f"Latent dim: {config.latent_dim}, Swarm size: {swarm_size}")
        print(f"Iterations: {n_iterations}")
        print("=" * 60)
    
    model = PSOMF(n_users, n_items, config.latent_dim,
                 swarm_size, inertia_weight_max, inertia_weight_min,
                 cognitive_coeff, social_coeff, config.regularization,
                 boundary, config.random_seed)
    
    history = model.fit(train_ratings, n_iterations, verbose=verbose)
    rmse_value, mae_value = model.evaluate(test_ratings)
    
    results = {
        'method': 'PSO-MF',
        'dataset': 'MovieLens 1M',
        'rmse': rmse_value,
        'mae': mae_value,
        'final_loss': history['losses'][-1],
        'n_users': n_users,
        'n_items': n_items,
        'n_train': n_train,
        'n_test': n_test,
        'history': history
    }
    
    save_results_ml1m(results, config, 'pso_mf', swarm_size=swarm_size,
                     inertia_weight_max=inertia_weight_max,
                     inertia_weight_min=inertia_weight_min,
                     cognitive_coeff=cognitive_coeff, social_coeff=social_coeff,
                     boundary=boundary, n_iterations=n_iterations)
    return results


def run_coa_sgd_mf_ml1m(config: Config, n_coatis: int = 30,
                        learning_rate: float = 0.01, boundary: float = 1.0,
                        coa_iterations: int = 50, sgd_iterations: int = 100,
                        verbose: bool = True) -> dict:
    """Run COA-SGD-MF on MovieLens 1M."""
    set_random_seed(config.random_seed)
    
    train_path = config.get_train_path()
    test_path = config.get_test_path()
    train_ratings, test_ratings = load_train_test_split(train_path, test_path)
    
    n_users, n_items, n_train = get_data_info(train_ratings)
    _, _, n_test = get_data_info(test_ratings)
    
    if verbose:
        print("=" * 60)
        print("COA-SGD-MF on MovieLens 1M")
        print("=" * 60)
        print(f"Train: {n_train} ratings, Test: {n_test} ratings")
        print(f"Users: {n_users}, Items: {n_items}")
        print(f"Latent dim: {config.latent_dim}, Coatis: {n_coatis}")
        print(f"COA iterations: {coa_iterations}, SGD iterations: {sgd_iterations}")
        print("=" * 60)
    
    model = COASGDMF(n_users, n_items, config.latent_dim,
                    n_coatis, learning_rate, config.regularization,
                    boundary, config.random_seed)
    
    history = model.fit(train_ratings, coa_iterations, sgd_iterations, verbose=verbose)
    rmse_value, mae_value = model.evaluate(test_ratings)
    
    # Final loss is from SGD phase
    final_loss = history['sgd_losses'][-1] if 'sgd_losses' in history else history['losses'][-1]
    
    results = {
        'method': 'COA-SGD-MF',
        'dataset': 'MovieLens 1M',
        'rmse': rmse_value,
        'mae': mae_value,
        'final_loss': final_loss,
        'n_users': n_users,
        'n_items': n_items,
        'n_train': n_train,
        'n_test': n_test,
        'history': history
    }
    
    save_results_ml1m(results, config, 'coa_sgd_mf', n_coatis=n_coatis,
                     learning_rate=learning_rate, boundary=boundary,
                     coa_iterations=coa_iterations, sgd_iterations=sgd_iterations)
    return results


def run_hho_sgd_mf_ml1m(config: Config, n_hawks: int = 30,
                        escape_energy_initial: float = 1.0,
                        learning_rate: float = 0.01, boundary: float = 1.0,
                        hho_iterations: int = 50, sgd_iterations: int = 100,
                        verbose: bool = True) -> dict:
    """Run HHO-SGD-MF on MovieLens 1M."""
    set_random_seed(config.random_seed)
    
    train_path = config.get_train_path()
    test_path = config.get_test_path()
    train_ratings, test_ratings = load_train_test_split(train_path, test_path)
    
    n_users, n_items, n_train = get_data_info(train_ratings)
    _, _, n_test = get_data_info(test_ratings)
    
    if verbose:
        print("=" * 60)
        print("HHO-SGD-MF on MovieLens 1M")
        print("=" * 60)
        print(f"Train: {n_train} ratings, Test: {n_test} ratings")
        print(f"Users: {n_users}, Items: {n_items}")
        print(f"Latent dim: {config.latent_dim}, Hawks: {n_hawks}")
        print(f"HHO iterations: {hho_iterations}, SGD iterations: {sgd_iterations}")
        print("=" * 60)
    
    model = HHOSGDMF(n_users, n_items, config.latent_dim,
                    n_hawks, escape_energy_initial, learning_rate,
                    config.regularization, boundary, config.random_seed)
    
    history = model.fit(train_ratings, hho_iterations, sgd_iterations, verbose=verbose)
    rmse_value, mae_value = model.evaluate(test_ratings)
    
    # Final loss is from SGD phase
    final_loss = history['sgd_losses'][-1] if 'sgd_losses' in history else history['losses'][-1]
    
    results = {
        'method': 'HHO-SGD-MF',
        'dataset': 'MovieLens 1M',
        'rmse': rmse_value,
        'mae': mae_value,
        'final_loss': final_loss,
        'n_users': n_users,
        'n_items': n_items,
        'n_train': n_train,
        'n_test': n_test,
        'history': history
    }
    
    save_results_ml1m(results, config, 'hho_sgd_mf', n_hawks=n_hawks,
                     escape_energy_initial=escape_energy_initial,
                     learning_rate=learning_rate, boundary=boundary,
                     hho_iterations=hho_iterations, sgd_iterations=sgd_iterations)
    return results


def run_hho_eo_mf_ml1m(
    config: Config,
    n_agents: int = 80,
    escape_energy_initial: float = 1.5,
    boundary: float = 1.0,
    n_iterations: int = 100,
    verbose: bool = True,
) -> dict:
    """Run HHO-EO-MF on MovieLens 1M."""
    set_random_seed(config.random_seed)

    train_path = config.get_train_path()
    test_path = config.get_test_path()
    train_ratings, test_ratings = load_train_test_split(train_path, test_path)

    n_users, n_items, n_train = get_data_info(train_ratings)
    _, _, n_test = get_data_info(test_ratings)

    if verbose:
        print("=" * 60)
        print("HHO-EO-MF on MovieLens 1M")
        print("=" * 60)
        print(f"Train: {n_train} ratings, Test: {n_test} ratings")
        print(f"Users: {n_users}, Items: {n_items}")
        print(f"Latent dim: {config.latent_dim}, Agents: {n_agents}")
        print(f"HHO+EO iterations: {n_iterations}")
        print("=" * 60)

    model = HHOEOMF(
        n_users,
        n_items,
        config.latent_dim,
        n_agents=n_agents,
        escape_energy_initial=escape_energy_initial,
        regularization=config.regularization,
        boundary=boundary,
        random_seed=config.random_seed,
    )

    # Phase 1: HHO+EO meta-heuristic optimization
    history = model.fit(train_ratings, n_iterations=n_iterations, verbose=verbose)

    # Optional Phase 2: SGD refinement starting from HHO+EO solution
    sgd_iterations = 50
    sgd_learning_rate = config.learning_rate

    if verbose:
        print("\n" + "=" * 60)
        print("Phase 2: SGD Refinement after HHO-EO")
        print("=" * 60)

    sgd_optimizer = SGDOptimizer(
        learning_rate=sgd_learning_rate, regularization=config.regularization
    )
    base_model = model.get_model()
    sgd_history = sgd_optimizer.optimize(
        base_model, train_ratings, n_iterations=sgd_iterations, verbose=verbose
    )

    # Append SGD losses to main history for unified loss_curve.csv
    if history["iterations"]:
        offset = history["iterations"][-1] + 1
    else:
        offset = 0
    for it, loss in zip(sgd_history["iterations"], sgd_history["losses"]):
        history["iterations"].append(offset + it)
        history["losses"].append(loss)

    rmse_value, mae_value = model.evaluate(test_ratings)

    results = {
        "method": "HHO-EO-MF",
        "dataset": "MovieLens 1M",
        "rmse": rmse_value,
        "mae": mae_value,
        "final_loss": history["losses"][-1],
        "n_users": n_users,
        "n_items": n_items,
        "n_train": n_train,
        "n_test": n_test,
        "history": history,
    }

    save_results_ml1m(
        results,
        config,
        "hho_eo_mf",
        n_agents=n_agents,
        escape_energy_initial=escape_energy_initial,
        boundary=boundary,
        n_iterations=n_iterations,
    )
    return results


def run_hho_dbo_mf_ml1m(
    config: Config,
    n_agents: int = 60,
    escape_energy_initial: float = 1.5,
    boundary: float = 1.0,
    n_iterations: int = 200,
    verbose: bool = True,
) -> dict:
    """Run HHO-DBO-MF on MovieLens 1M."""
    set_random_seed(config.random_seed)

    train_path = config.get_train_path()
    test_path = config.get_test_path()
    train_ratings, test_ratings = load_train_test_split(train_path, test_path)

    n_users, n_items, n_train = get_data_info(train_ratings)
    _, _, n_test = get_data_info(test_ratings)

    if verbose:
        print("=" * 60)
        print("HHO-DBO-MF on MovieLens 1M")
        print("=" * 60)
        print(f"Train: {n_train} ratings, Test: {n_test} ratings")
        print(f"Users: {n_users}, Items: {n_items}")
        print(f"Latent dim: {config.latent_dim}, Agents: {n_agents}")
        print(f"HHO+DBO iterations: {n_iterations}")
        print("=" * 60)

    model = HHODBOMF(
        n_users,
        n_items,
        config.latent_dim,
        n_agents=n_agents,
        escape_energy_initial=escape_energy_initial,
        regularization=config.regularization,
        boundary=boundary,
        random_seed=config.random_seed,
    )

    history = model.fit(train_ratings, n_iterations=n_iterations, verbose=verbose)
    rmse_value, mae_value = model.evaluate(test_ratings)

    results = {
        "method": "HHO-DBO-MF",
        "dataset": "MovieLens 1M",
        "rmse": rmse_value,
        "mae": mae_value,
        "final_loss": history["losses"][-1],
        "n_users": n_users,
        "n_items": n_items,
        "n_train": n_train,
        "n_test": n_test,
        "history": history,
    }

    save_results_ml1m(
        results,
        config,
        "hho_dbo_mf",
        n_agents=n_agents,
        escape_energy_initial=escape_energy_initial,
        boundary=boundary,
        n_iterations=n_iterations,
    )
    return results


def run_hho_ssa_mf_ml1m(
    config: Config,
    n_agents: int = 40,
    escape_energy_initial: float = 1.5,
    boundary: float = 2.0,
    n_iterations: int = 150,
    safety_threshold: float = 0.7,
    producer_ratio: float = 0.2,
    awareness_ratio: float = 0.1,
    verbose: bool = True,
) -> dict:
    """Run HHO-SSA-MF on MovieLens 1M."""
    set_random_seed(config.random_seed)

    train_path = config.get_train_path()
    test_path = config.get_test_path()
    train_ratings, test_ratings = load_train_test_split(train_path, test_path)

    n_users, n_items, n_train = get_data_info(train_ratings)
    _, _, n_test = get_data_info(test_ratings)

    if verbose:
        print("=" * 60)
        print("HHO-SSA-MF on MovieLens 1M")
        print("=" * 60)
        print(f"Train: {n_train} ratings, Test: {n_test} ratings")
        print(f"Users: {n_users}, Items: {n_items}")
        print(f"Latent dim: {config.latent_dim}, Agents: {n_agents}")
        print(f"HHO+SSA iterations: {n_iterations}")
        print(f"Boundary: [-{boundary}, +{boundary}]")
        print(
            f"SSA params -> ST={safety_threshold}, PR={producer_ratio}, "
            f"AR={awareness_ratio}"
        )
        print("=" * 60)

    model = HHOSSAMF(
        n_users,
        n_items,
        config.latent_dim,
        n_agents=n_agents,
        escape_energy_initial=1.0,
        regularization=config.regularization,
        boundary=boundary,
        safety_threshold=safety_threshold,
        producer_ratio=producer_ratio,
        awareness_ratio=awareness_ratio,
        random_seed=config.random_seed,
    )

    # Phase 1: HHO+SSA meta-sezgisel optimizasyon
    history = model.fit(
        train_ratings, n_iterations=n_iterations, verbose=verbose
    )

    # HHO+SSA tek başına ne kadar iyi? (SGD'den önce)
    if verbose:
        rmse_hhosa, mae_hhosa = model.evaluate(test_ratings)
        print("\n" + "=" * 60)
        print("Performance after HHO-SSA (before SGD)")
        print("=" * 60)
        print(f"HHO-SSA only -> RMSE = {rmse_hhosa:.6f}, MAE = {mae_hhosa:.6f}")

    # Phase 2: SGD refinement (HHO-SSA çözümünden sonra)
    sgd_iterations = 50
    sgd_learning_rate = config.learning_rate

    if verbose:
        print("\n" + "=" * 60)
        print("Phase 2: SGD Refinement after HHO-SSA")
        print("=" * 60)

    sgd_optimizer = SGDOptimizer(
        learning_rate=sgd_learning_rate,
        regularization=config.regularization,
    )
    base_model = model.get_model()
    sgd_history = sgd_optimizer.optimize(
        base_model,
        train_ratings,
        n_iterations=sgd_iterations,
        verbose=verbose,
    )

    # HHO-SSA loss eğrisine SGD loss'larını ekle (tek birleşik loss_curve için)
    if history["iterations"]:
        offset = history["iterations"][-1] + 1
    else:
        offset = 0
    for it, loss in zip(sgd_history["iterations"], sgd_history["losses"]):
        history["iterations"].append(offset + it)
        history["losses"].append(loss)

    rmse_value, mae_value = model.evaluate(test_ratings)

    results = {
        "method": "HHO-SSA-MF",
        "dataset": "MovieLens 1M",
        "rmse": rmse_value,
        "mae": mae_value,
        "final_loss": history["losses"][-1],
        "n_users": n_users,
        "n_items": n_items,
        "n_train": n_train,
        "n_test": n_test,
        "history": history,
    }

    save_results_ml1m(
        results,
        config,
        "hho_ssa_mf",
        n_agents=n_agents,
        escape_energy_initial=escape_energy_initial,
        boundary=boundary,
        safety_threshold=safety_threshold,
        producer_ratio=producer_ratio,
        awareness_ratio=awareness_ratio,
        n_iterations=n_iterations,
    )
    return results


def run_hybrid_coa_sma_mf_ml1m(
    config: Config,
    n_agents: int = 60,
    sma_ratio_min: float = 0.2,
    sma_ratio_max: float = 0.9,
    boundary: float = 1.0,
    n_iterations: int = 150,
    verbose: bool = True,
) -> dict:
    """Run Hybrid-COA-SMA-MF on MovieLens 1M."""
    set_random_seed(config.random_seed)

    train_path = config.get_train_path()
    test_path = config.get_test_path()
    train_ratings, test_ratings = load_train_test_split(train_path, test_path)

    n_users, n_items, n_train = get_data_info(train_ratings)
    _, _, n_test = get_data_info(test_ratings)

    if verbose:
        print("=" * 60)
        print("Hybrid-COA-SMA-MF on MovieLens 1M")
        print("=" * 60)
        print(f"Train: {n_train} ratings, Test: {n_test} ratings")
        print(f"Users: {n_users}, Items: {n_items}")
        print(f"Latent dim: {config.latent_dim}, Agents: {n_agents}")
        print(f"Hybrid COA+SMA iterations: {n_iterations}")
        print("=" * 60)

    model = HybridCOASMAMF(
        n_users,
        n_items,
        config.latent_dim,
        n_agents=n_agents,
        sma_ratio_min=sma_ratio_min,
        sma_ratio_max=sma_ratio_max,
        regularization=config.regularization,
        boundary=boundary,
        random_seed=config.random_seed,
    )

    history = model.fit(train_ratings, n_iterations=n_iterations, verbose=verbose)
    rmse_value, mae_value = model.evaluate(test_ratings)

    results = {
        "method": "Hybrid-COA-SMA-MF",
        "dataset": "MovieLens 1M",
        "rmse": rmse_value,
        "mae": mae_value,
        "final_loss": history["losses"][-1],
        "n_users": n_users,
        "n_items": n_items,
        "n_train": n_train,
        "n_test": n_test,
        "history": history,
    }

    save_results_ml1m(
        results,
        config,
        "hybrid_coa_sma_mf",
        n_agents=n_agents,
        sma_ratio_min=sma_ratio_min,
        sma_ratio_max=sma_ratio_max,
        boundary=boundary,
        n_iterations=n_iterations,
    )
    return results


def run_hho_kmeans_mf_ml1m(config: Config, n_hawks: int = 30,
                           escape_energy_initial: float = 1.0,
                           n_clusters_users: int = None,
                           n_clusters_items: int = None,
                           learning_rate: float = 0.1, boundary: float = 1.0,
                           hho_iterations: int = 50, kmeans_iterations: int = 100,
                           verbose: bool = True) -> dict:
    """Run HHO-KMeans-MF on MovieLens 1M."""
    set_random_seed(config.random_seed)
    
    train_path = config.get_train_path()
    test_path = config.get_test_path()
    train_ratings, test_ratings = load_train_test_split(train_path, test_path)
    
    n_users, n_items, n_train = get_data_info(train_ratings)
    _, _, n_test = get_data_info(test_ratings)
    
    if verbose:
        print("=" * 60)
        print("HHO-KMeans-MF on MovieLens 1M")
        print("=" * 60)
        print(f"Train: {n_train} ratings, Test: {n_test} ratings")
        print(f"Users: {n_users}, Items: {n_items}")
        print(f"Latent dim: {config.latent_dim}, Hawks: {n_hawks}")
        print(f"HHO iterations: {hho_iterations}, K-Means iterations: {kmeans_iterations}")
        print("=" * 60)
    
    model = HHOKMeansMF(n_users, n_items, config.latent_dim,
                       n_hawks, escape_energy_initial,
                       n_clusters_users, n_clusters_items,
                       learning_rate, config.regularization,
                       boundary, config.random_seed)
    
    history = model.fit(train_ratings, hho_iterations, kmeans_iterations, verbose=verbose)
    rmse_value, mae_value = model.evaluate(test_ratings)
    
    # Final loss is from K-Means phase
    final_loss = history['kmeans_losses'][-1] if 'kmeans_losses' in history else history['losses'][-1]
    
    results = {
        'method': 'HHO-KMeans-MF',
        'dataset': 'MovieLens 1M',
        'rmse': rmse_value,
        'mae': mae_value,
        'final_loss': final_loss,
        'n_users': n_users,
        'n_items': n_items,
        'n_train': n_train,
        'n_test': n_test,
        'history': history
    }
    
    save_results_ml1m(results, config, 'hho_kmeans_mf', n_hawks=n_hawks,
                     escape_energy_initial=escape_energy_initial,
                     n_clusters_users=n_clusters_users,
                     n_clusters_items=n_clusters_items,
                     learning_rate=learning_rate, boundary=boundary,
                     hho_iterations=hho_iterations, kmeans_iterations=kmeans_iterations)
    return results


def run_coa_kmeans_mf_ml1m(config: Config, n_coatis: int = 30,
                           n_clusters_users: int = None,
                           n_clusters_items: int = None,
                           learning_rate: float = 0.1, boundary: float = 1.0,
                           coa_iterations: int = 50, kmeans_iterations: int = 100,
                           verbose: bool = True) -> dict:
    """Run COA-KMeans-MF on MovieLens 1M."""
    set_random_seed(config.random_seed)
    
    train_path = config.get_train_path()
    test_path = config.get_test_path()
    train_ratings, test_ratings = load_train_test_split(train_path, test_path)
    
    n_users, n_items, n_train = get_data_info(train_ratings)
    _, _, n_test = get_data_info(test_ratings)
    
    if verbose:
        print("=" * 60)
        print("COA-KMeans-MF on MovieLens 1M")
        print("=" * 60)
        print(f"Train: {n_train} ratings, Test: {n_test} ratings")
        print(f"Users: {n_users}, Items: {n_items}")
        print(f"Latent dim: {config.latent_dim}, Coatis: {n_coatis}")
        print(f"COA iterations: {coa_iterations}, K-Means iterations: {kmeans_iterations}")
        print("=" * 60)
    
    model = COAKMeansMF(n_users, n_items, config.latent_dim,
                       n_coatis, n_clusters_users, n_clusters_items,
                       learning_rate, config.regularization,
                       boundary, config.random_seed)
    
    history = model.fit(train_ratings, coa_iterations, kmeans_iterations, verbose=verbose)
    rmse_value, mae_value = model.evaluate(test_ratings)
    
    # Final loss is from K-Means phase
    final_loss = history['kmeans_losses'][-1] if 'kmeans_losses' in history else history['losses'][-1]
    
    results = {
        'method': 'COA-KMeans-MF',
        'dataset': 'MovieLens 1M',
        'rmse': rmse_value,
        'mae': mae_value,
        'final_loss': final_loss,
        'n_users': n_users,
        'n_items': n_items,
        'n_train': n_train,
        'n_test': n_test,
        'history': history
    }
    
    save_results_ml1m(results, config, 'coa_kmeans_mf', n_coatis=n_coatis,
                     n_clusters_users=n_clusters_users,
                     n_clusters_items=n_clusters_items,
                     learning_rate=learning_rate, boundary=boundary,
                     coa_iterations=coa_iterations, kmeans_iterations=kmeans_iterations)
    return results


def run_goa_kmeans_mf_ml1m(
    config: Config,
    n_grasshoppers: int = 40,
    n_clusters_users: int | None = None,
    n_clusters_items: int | None = None,
    learning_rate: float = 0.1,
    boundary: float = 1.0,
    goa_iterations: int = 50,
    kmeans_iterations: int = 100,
    verbose: bool = True,
) -> dict:
    """Run GOA-KMeans-MF on MovieLens 1M."""
    set_random_seed(config.random_seed)

    train_path = config.get_train_path()
    test_path = config.get_test_path()
    train_ratings, test_ratings = load_train_test_split(train_path, test_path)

    n_users, n_items, n_train = get_data_info(train_ratings)
    _, _, n_test = get_data_info(test_ratings)

    if verbose:
        print("=" * 60)
        print("GOA-KMeans-MF on MovieLens 1M")
        print("=" * 60)
        print(f"Train: {n_train} ratings, Test: {n_test} ratings")
        print(f"Users: {n_users}, Items: {n_items}")
        print(f"Latent dim: {config.latent_dim}, Grasshoppers: {n_grasshoppers}")
        print(f"GOA iterations: {goa_iterations}, K-Means iterations: {kmeans_iterations}")
        print("=" * 60)

    model = GOAKMeansMF(
        n_users,
        n_items,
        config.latent_dim,
        n_grasshoppers=n_grasshoppers,
        n_clusters_users=n_clusters_users,
        n_clusters_items=n_clusters_items,
        learning_rate=learning_rate,
        regularization=config.regularization,
        boundary=boundary,
        random_seed=config.random_seed,
    )

    history = model.fit(
        train_ratings,
        goa_iterations=goa_iterations,
        kmeans_iterations=kmeans_iterations,
        verbose=verbose,
    )
    rmse_value, mae_value = model.evaluate(test_ratings)

    final_loss = history["kmeans_losses"][-1] if "kmeans_losses" in history else history["goa_losses"][-1]

    results = {
        "method": "GOA-KMeans-MF",
        "dataset": "MovieLens 1M",
        "rmse": rmse_value,
        "mae": mae_value,
        "final_loss": final_loss,
        "n_users": n_users,
        "n_items": n_items,
        "n_train": n_train,
        "n_test": n_test,
        "history": history,
    }

    save_results_ml1m(
        results,
        config,
        "goa_kmeans_mf",
        n_grasshoppers=n_grasshoppers,
        n_clusters_users=n_clusters_users,
        n_clusters_items=n_clusters_items,
        learning_rate=learning_rate,
        boundary=boundary,
        goa_iterations=goa_iterations,
        kmeans_iterations=kmeans_iterations,
    )
    return results


def save_results_ml1m(results: dict, config: Config, method_name: str, **kwargs):
    """Save results for MovieLens 1M experiments."""
    results_dir = Path("results") / "movielens-1m" / method_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save loss curve
    loss_curve_path = results_dir / "loss_curve.csv"
    with open(loss_curve_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'loss'])
        
        if method_name in ['hho_sgd_mf', 'coa_sgd_mf', 'hho_kmeans_mf', 'coa_kmeans_mf']:
            # Combined loss curve: Metaheuristic final + Refinement
            if method_name in ['hho_sgd_mf', 'hho_kmeans_mf']:
                if len(results['history']['hho_losses']) > 0:
                    writer.writerow([0, results['history']['hho_losses'][-1]])
                if method_name == 'hho_sgd_mf':
                    for i, loss in enumerate(results['history']['sgd_losses']):
                        writer.writerow([i + 1, loss])
                elif method_name == 'hho_kmeans_mf':
                    for i, loss in enumerate(results['history']['kmeans_losses']):
                        writer.writerow([i + 1, loss])
            elif method_name in ['coa_sgd_mf', 'coa_kmeans_mf']:
                if len(results['history']['coa_losses']) > 0:
                    writer.writerow([0, results['history']['coa_losses'][-1]])
                if method_name == 'coa_sgd_mf':
                    for i, loss in enumerate(results['history']['sgd_losses']):
                        writer.writerow([i + 1, loss])
                elif method_name == 'coa_kmeans_mf':
                    for i, loss in enumerate(results['history']['kmeans_losses']):
                        writer.writerow([i + 1, loss])
        else:
            # Standard loss curve
            for iteration, loss in zip(results['history']['iterations'],
                                      results['history']['losses']):
                writer.writerow([iteration, loss])
    
    # Save metaheuristic-specific loss curves if applicable
    if method_name == 'hho_sgd_mf':
        hho_loss_curve_path = results_dir / "hho_loss_curve.csv"
        with open(hho_loss_curve_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'loss'])
            for iteration, loss in zip(results['history']['hho_iterations'],
                                      results['history']['hho_losses']):
                writer.writerow([iteration, loss])
        
        sgd_loss_curve_path = results_dir / "sgd_loss_curve.csv"
        with open(sgd_loss_curve_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'loss'])
            for iteration, loss in zip(results['history']['sgd_iterations'],
                                      results['history']['sgd_losses']):
                writer.writerow([iteration, loss])
    
    elif method_name == 'coa_sgd_mf':
        coa_loss_curve_path = results_dir / "coa_loss_curve.csv"
        with open(coa_loss_curve_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'loss'])
            for iteration, loss in zip(results['history']['coa_iterations'],
                                      results['history']['coa_losses']):
                writer.writerow([iteration, loss])
        
        sgd_loss_curve_path = results_dir / "sgd_loss_curve.csv"
        with open(sgd_loss_curve_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'loss'])
            for iteration, loss in zip(results['history']['sgd_iterations'],
                                      results['history']['sgd_losses']):
                writer.writerow([iteration, loss])
    
    elif method_name == 'hho_kmeans_mf':
        hho_loss_curve_path = results_dir / "hho_loss_curve.csv"
        with open(hho_loss_curve_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'loss'])
            for iteration, loss in zip(results['history']['hho_iterations'],
                                      results['history']['hho_losses']):
                writer.writerow([iteration, loss])
        
        kmeans_loss_curve_path = results_dir / "kmeans_loss_curve.csv"
        with open(kmeans_loss_curve_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'loss'])
            for iteration, loss in zip(results['history']['kmeans_iterations'],
                                      results['history']['kmeans_losses']):
                writer.writerow([iteration, loss])
    
    elif method_name == 'coa_kmeans_mf':
        coa_loss_curve_path = results_dir / "coa_loss_curve.csv"
        with open(coa_loss_curve_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'loss'])
            for iteration, loss in zip(results['history']['coa_iterations'],
                                      results['history']['coa_losses']):
                writer.writerow([iteration, loss])
        
        kmeans_loss_curve_path = results_dir / "kmeans_loss_curve.csv"
        with open(kmeans_loss_curve_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'loss'])
            for iteration, loss in zip(results['history']['kmeans_iterations'],
                                      results['history']['kmeans_losses']):
                writer.writerow([iteration, loss])
    elif method_name == 'goa_kmeans_mf':
        goa_loss_curve_path = results_dir / "goa_loss_curve.csv"
        with open(goa_loss_curve_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'loss'])
            for iteration, loss in zip(results['history']['goa_iterations'],
                                      results['history']['goa_losses']):
                writer.writerow([iteration, loss])

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
        'regularization': float(config.regularization),
        'random_seed': int(config.random_seed),
        'rmse': float(results['rmse']),
        'mae': float(results['mae']),
        'final_loss': float(results['final_loss'])
    }
    
    # Add method-specific parameters
    if method_name == 'mf_sgd':
        summary['learning_rate'] = float(config.learning_rate)
        summary['n_iterations'] = int(config.n_iterations)
    elif method_name == 'pso_mf':
        summary['swarm_size'] = int(kwargs.get('swarm_size', 30))
        summary['inertia_weight_max'] = float(kwargs.get('inertia_weight_max', 0.9))
        summary['inertia_weight_min'] = float(kwargs.get('inertia_weight_min', 0.2))
        summary['cognitive_coeff'] = float(kwargs.get('cognitive_coeff', 2.0))
        summary['social_coeff'] = float(kwargs.get('social_coeff', 2.0))
        summary['boundary'] = float(kwargs.get('boundary', 1.0))
        summary['n_iterations'] = int(kwargs.get('n_iterations', 100))
    elif method_name == 'hho_sgd_mf':
        summary['n_hawks'] = int(kwargs.get('n_hawks', 30))
        summary['escape_energy_initial'] = float(kwargs.get('escape_energy_initial', 1.0))
        summary['learning_rate'] = float(kwargs.get('learning_rate', 0.01))
        summary['boundary'] = float(kwargs.get('boundary', 1.0))
        summary['hho_iterations'] = int(kwargs.get('hho_iterations', 50))
        summary['sgd_iterations'] = int(kwargs.get('sgd_iterations', 100))
        summary['n_iterations'] = int(kwargs.get('hho_iterations', 50) + 
                                     kwargs.get('sgd_iterations', 100))
    elif method_name == 'coa_sgd_mf':
        summary['n_coatis'] = int(kwargs.get('n_coatis', 30))
        summary['learning_rate'] = float(kwargs.get('learning_rate', 0.01))
        summary['boundary'] = float(kwargs.get('boundary', 1.0))
        summary['coa_iterations'] = int(kwargs.get('coa_iterations', 50))
        summary['sgd_iterations'] = int(kwargs.get('sgd_iterations', 100))
        summary['n_iterations'] = int(kwargs.get('coa_iterations', 50) + 
                                     kwargs.get('sgd_iterations', 100))
    elif method_name == 'hho_kmeans_mf':
        summary['n_hawks'] = int(kwargs.get('n_hawks', 30))
        summary['escape_energy_initial'] = float(kwargs.get('escape_energy_initial', 1.0))
        summary['n_clusters_users'] = int(kwargs.get('n_clusters_users', 0)) if kwargs.get('n_clusters_users') else None
        summary['n_clusters_items'] = int(kwargs.get('n_clusters_items', 0)) if kwargs.get('n_clusters_items') else None
        summary['learning_rate'] = float(kwargs.get('learning_rate', 0.1))
        summary['boundary'] = float(kwargs.get('boundary', 1.0))
        summary['hho_iterations'] = int(kwargs.get('hho_iterations', 50))
        summary['kmeans_iterations'] = int(kwargs.get('kmeans_iterations', 100))
        summary['n_iterations'] = int(kwargs.get('hho_iterations', 50) + 
                                     kwargs.get('kmeans_iterations', 100))
    elif method_name == 'coa_kmeans_mf':
        summary['n_coatis'] = int(kwargs.get('n_coatis', 30))
        summary['n_clusters_users'] = int(kwargs.get('n_clusters_users', 0)) if kwargs.get('n_clusters_users') else None
        summary['n_clusters_items'] = int(kwargs.get('n_clusters_items', 0)) if kwargs.get('n_clusters_items') else None
        summary['learning_rate'] = float(kwargs.get('learning_rate', 0.1))
        summary['boundary'] = float(kwargs.get('boundary', 1.0))
        summary['coa_iterations'] = int(kwargs.get('coa_iterations', 50))
        summary['kmeans_iterations'] = int(kwargs.get('kmeans_iterations', 100))
        summary['n_iterations'] = int(kwargs.get('coa_iterations', 50) + 
                                     kwargs.get('kmeans_iterations', 100))
    elif method_name == 'hho_eo_mf':
        summary['n_agents'] = int(kwargs.get('n_agents', 60))
        summary['escape_energy_initial'] = float(kwargs.get('escape_energy_initial', 1.5))
        summary['boundary'] = float(kwargs.get('boundary', 1.0))
        summary['n_iterations'] = int(kwargs.get('n_iterations', 150))
    elif method_name == 'hho_dbo_mf':
        summary['n_agents'] = int(kwargs.get('n_agents', 60))
        summary['escape_energy_initial'] = float(kwargs.get('escape_energy_initial', 1.5))
        summary['boundary'] = float(kwargs.get('boundary', 1.0))
        summary['n_iterations'] = int(kwargs.get('n_iterations', 200))
    elif method_name == 'hho_ssa_mf':
        summary['n_agents'] = int(kwargs.get('n_agents', 40))
        summary['escape_energy_initial'] = float(kwargs.get('escape_energy_initial', 1.5))
        summary['boundary'] = float(kwargs.get('boundary', 2.0))
        summary['safety_threshold'] = float(kwargs.get('safety_threshold', 0.7))
        summary['producer_ratio'] = float(kwargs.get('producer_ratio', 0.2))
        summary['awareness_ratio'] = float(kwargs.get('awareness_ratio', 0.1))
        summary['n_iterations'] = int(kwargs.get('n_iterations', 150))
    elif method_name == 'hybrid_coa_sma_mf':
        summary['n_agents'] = int(kwargs.get('n_agents', 60))
        summary['sma_ratio_min'] = float(kwargs.get('sma_ratio_min', 0.2))
        summary['sma_ratio_max'] = float(kwargs.get('sma_ratio_max', 0.9))
        summary['boundary'] = float(kwargs.get('boundary', 1.0))
        summary['n_iterations'] = int(kwargs.get('n_iterations', 150))
    elif method_name == 'goa_kmeans_mf':
        summary['n_grasshoppers'] = int(kwargs.get('n_grasshoppers', 40))
        summary['n_clusters_users'] = int(kwargs.get('n_clusters_users', 0)) if kwargs.get('n_clusters_users') else None
        summary['n_clusters_items'] = int(kwargs.get('n_clusters_items', 0)) if kwargs.get('n_clusters_items') else None
        summary['learning_rate'] = float(kwargs.get('learning_rate', 0.1))
        summary['boundary'] = float(kwargs.get('boundary', 1.0))
        summary['goa_iterations'] = int(kwargs.get('goa_iterations', 50))
        summary['kmeans_iterations'] = int(kwargs.get('kmeans_iterations', 100))
        summary['n_iterations'] = int(kwargs.get('goa_iterations', 50) + kwargs.get('kmeans_iterations', 100))
    
    summary_path = results_dir / "summary.json"

    # Eğer daha önce summary.json varsa, eski özetleri kaybetmeden listeye ekle
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = None

        if isinstance(existing, list):
            summaries = existing
        elif isinstance(existing, dict):
            # Eski tek-kayıt formatını koru ve listeye çevir
            summaries = [existing]
        else:
            summaries = []

        summaries.append(summary)
    else:
        # İlk kez yazılıyorsa tek elemanlı liste olarak başlat
        summaries = [summary]

    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)


def generate_comparison_table(results_dir: Path = Path("results/movielens-1m")) -> None:
    """Generate comparison table for MovieLens 1M results."""
    rows = []
    
    for method_dir in results_dir.iterdir():
        if method_dir.is_dir():
            summary_path = method_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    summary_obj = json.load(f)

                    # summary.json artık liste olabilir; son koşuyu kullan
                    if isinstance(summary_obj, list) and len(summary_obj) > 0:
                        summary = summary_obj[-1]
                    elif isinstance(summary_obj, dict):
                        summary = summary_obj
                    else:
                        continue

                    rows.append({
                        'Dataset': summary.get('dataset', 'MovieLens 1M'),
                        'Method': summary.get('method', ''),
                        'RMSE': summary.get('rmse', None),
                        'MAE': summary.get('mae', None),
                        'Final Loss': summary.get('final_loss', None)
                    })
    
    # Create DataFrame and save
    import pandas as pd
    df = pd.DataFrame(rows)
    df = df.sort_values('Method')
    
    table_path = results_dir / "comparison_table.csv"
    df.to_csv(table_path, index=False)
    
    print("\n" + "=" * 60)
    print("COMPARISON TABLE: MovieLens 1M")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    print(f"\nTable saved to: {table_path}")


def main(
    method: str | None = None,
    cv_latent: bool = False,
    latent_candidates: list[int] | None = None,
    cv_folds: int = 5,
) -> None:
    """
    Run experiments on MovieLens 1M.

    Parameters
    ----------
    method : str or None
        If None, run all enabled methods.
        If one of {'mf_sgd', 'pso_mf', 'hho_sgd_mf', 'hho_eo_mf',
                   'coa_sgd_mf', 'hho_kmeans_mf', 'coa_kmeans_mf'},
        run only that method.
    """
    # Create ML-1M config
    config = Config(
        data_dir="data/ml-1m",
        train_split="train.dat",
        test_split="test.dat",
        latent_dim=10,
        learning_rate=0.01,
        regularization=0.01,
        n_iterations=100,
        random_seed=140
    )

    # Optionally perform cross-validation over latent dimensions (Method 1)
    if cv_latent:
        if latent_candidates is None:
            latent_candidates = [5, 10, 20, 50]

        best_k, _ = cross_validate_latent_dim_ml1m(
            config,
            latent_dims=latent_candidates,
            n_folds=cv_folds,
            verbose=True,
        )
        config.latent_dim = int(best_k)
        print("\n" + "=" * 60)
        print("USING BEST LATENT DIMENSION FROM CROSS-VALIDATION")
        print("=" * 60)
        print(f"Selected latent dimension: {config.latent_dim}")
        print("=" * 60)
    
    print("=" * 60)
    print("MOVIELENS 1M EXPERIMENTS")
    print("=" * 60)
    print(f"Dataset: {config.data_dir}")
    print(f"Train: {config.train_split}, Test: {config.test_split}")
    print(f"Latent dimension: {config.latent_dim}")
    print(f"Random seed: {config.random_seed}")
    print("=" * 60)
    
    all_results = {}
    run_all = method is None or method == "all"

    # Run MF-SGD
    if run_all or method == "mf_sgd":
        print("\n[MF-SGD] Running MF-SGD...")
        all_results["mf_sgd"] = run_mf_sgd_ml1m(config, verbose=True)

    # Run PSO-MF
    if run_all:
        if RUN_PSO_MF:
            print("\n[PSO-MF] Running PSO-MF...")
            all_results["pso_mf"] = run_pso_mf_ml1m(config, verbose=True)
        else:
            print("\n[PSO-MF] Skip PSO-MF (RUN_PSO_MF=False)")
    elif method == "pso_mf":
        print("\n[PSO-MF] Running PSO-MF (forced by --method)...")
        all_results["pso_mf"] = run_pso_mf_ml1m(config, verbose=True)

    # Run HHO-SGD-MF
    if run_all or method == "hho_sgd_mf":
        print("\n[HHO-SGD-MF] Running HHO-SGD-MF...")
        all_results["hho_sgd_mf"] = run_hho_sgd_mf_ml1m(config, verbose=True)

    # Run HHO-EO-MF
    if run_all or method == "hho_eo_mf":
        print("\n[HHO-EO-MF] Running HHO-EO-MF...")
        all_results["hho_eo_mf"] = run_hho_eo_mf_ml1m(config, verbose=True)

    # Run HHO-DBO-MF
    if run_all or method == "hho_dbo_mf":
        print("\n[HHO-DBO-MF] Running HHO-DBO-MF...")
        all_results["hho_dbo_mf"] = run_hho_dbo_mf_ml1m(config, verbose=True)

    # Run HHO-SSA-MF
    if run_all or method == "hho_ssa_mf":
        print("\n[HHO-SSA-MF] Running HHO-SSA-MF...")
        all_results["hho_ssa_mf"] = run_hho_ssa_mf_ml1m(config, verbose=True)

    # Run Hybrid-COA-SMA-MF
    if run_all or method == "hybrid_coa_sma_mf":
        print("\n[Hybrid-COA-SMA-MF] Running Hybrid-COA-SMA-MF...")
        all_results["hybrid_coa_sma_mf"] = run_hybrid_coa_sma_mf_ml1m(config, verbose=True)

    # Run COA-SGD-MF
    if run_all or method == "coa_sgd_mf":
        print("\n[COA-SGD-MF] Running COA-SGD-MF...")
        all_results["coa_sgd_mf"] = run_coa_sgd_mf_ml1m(config, verbose=True)

    # Run HHO-KMeans-MF
    if run_all:
        if RUN_HHO_KMEANS_MF:
            print("\n[HHO-KMeans-MF] Running HHO-KMeans-MF...")
            all_results["hho_kmeans_mf"] = run_hho_kmeans_mf_ml1m(config, verbose=True)
        else:
            print("\n[HHO-KMeans-MF] Skip HHO-KMeans-MF (RUN_HHO_KMEANS_MF=False)")
    elif method == "hho_kmeans_mf":
        print("\n[HHO-KMeans-MF] Running HHO-KMeans-MF (forced by --method)...")
        all_results["hho_kmeans_mf"] = run_hho_kmeans_mf_ml1m(config, verbose=True)

    # Run COA-KMeans-MF
    if run_all:
        if RUN_COA_KMEANS_MF:
            print("\n[COA-KMeans-MF] Running COA-KMeans-MF...")
            all_results["coa_kmeans_mf"] = run_coa_kmeans_mf_ml1m(config, verbose=True)
        else:
            print("\n[COA-KMeans-MF] Skip COA-KMeans-MF (RUN_COA_KMEANS_MF=False)")
    elif method == "coa_kmeans_mf":
        print("\n[COA-KMeans-MF] Running COA-KMeans-MF (forced by --method)...")
        all_results["coa_kmeans_mf"] = run_coa_kmeans_mf_ml1m(config, verbose=True)

    # Run GOA-KMeans-MF (always optional, no flag)
    if run_all or method == "goa_kmeans_mf":
        print("\n[GOA-KMeans-MF] Running GOA-KMeans-MF...")
        all_results["goa_kmeans_mf"] = run_goa_kmeans_mf_ml1m(config, verbose=True)

    # Generate comparison table only when running all methods
    if run_all:
        print("\n" + "=" * 60)
        print("GENERATING COMPARISON TABLE")
        print("=" * 60)
        generate_comparison_table()

        print("\n" + "=" * 60)
        print("ALL EXPERIMENTS COMPLETE!")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MovieLens 1M recommendation experiments.")
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=[
            "all",
            "mf_sgd",
            "pso_mf",
            "hho_sgd_mf",
            "hho_eo_mf",
            "coa_sgd_mf",
            "hho_kmeans_mf",
            "coa_kmeans_mf",
            "hybrid_coa_sma_mf",
            "goa_kmeans_mf",
            "hho_dbo_mf",
            "hho_ssa_mf",
        ],
        help=(
            "Which method to run. "
            "'all' runs the full pipeline (respecting RUN_* flags); "
            "a specific method name runs only that algorithm."
        ),
    )
    parser.add_argument(
        "--cv-latent",
        action="store_true",
        help=(
            "If set, perform k-fold cross-validation on the latent dimension "
            "using MF-SGD before running experiments. "
            "The best latent dimension will be used for all methods."
        ),
    )
    parser.add_argument(
        "--latent-candidates",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50],
        help=(
            "Candidate latent dimensions (k values) to try during cross-validation. "
            "Default: 5 10 20 50"
        ),
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation on latent dimension. Default: 5",
    )
    args = parser.parse_args()
    main(
        method=args.method,
        cv_latent=args.cv_latent,
        latent_candidates=args.latent_candidates,
        cv_folds=args.cv_folds,
    )

