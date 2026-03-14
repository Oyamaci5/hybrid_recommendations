"""
Run Cold Start MF experiment on MovieLens 1M.
Implements the 6-step flow:
1. Load data, train/test split
2. Compute global mean μ, bias terms
3. V matrix initialized with content features (genre)
4. SGD loop: update U, V
5. New user: folding-in
6. New item: metadata embedding
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from core.utils import (
    load_train_test_split,
    get_data_info,
    load_movies_ml1m,
    load_users_ml1m,
    compute_biases,
)
from core.config import Config, set_random_seed
from core.metrics import evaluate_predictions
from methods.mf_sgd import MFSGD
from methods.coa_sgd_mf import COASGDMF
from methods.hho_sgd_mf import HHOSGDMF
from methods.hho_eo_mf import HHOEOMF
from core.hho_eo_feature_selector import HybridHHOEOFeatureSelector


def run_cold_start_mf_ml1m(
    config: Config,
    method: str = "mf_sgd",
    use_hho_eo_fs: bool = False,
    verbose: bool = True,
) -> dict:
    """Run Cold Start MF on MovieLens 1M. method: mf_sgd, coa_sgd_mf, hho_sgd_mf, or hho_eo_mf."""
    set_random_seed(config.random_seed)
    
    data_dir = Path(config.data_dir)
    train_path = config.get_train_path()
    test_path = config.get_test_path()
    
    # Step 1: Load data, train/test split
    train_ratings, test_ratings = load_train_test_split(train_path, test_path)
    n_users, n_items, n_train = get_data_info(train_ratings)
    _, _, n_test = get_data_info(test_ratings)
    
    # Step 2: Compute biases
    mu, b_u, b_i = compute_biases(train_ratings, n_users, n_items)
    
    # Step 3: Load content, initialize V from genre
    movies_path = data_dir / "movies.dat"
    X_items = None
    X_users = None
    if movies_path.exists():
        X_items = load_movies_ml1m(str(movies_path), n_items)
    users_path = data_dir / "users.dat"
    if users_path.exists():
        X_users = load_users_ml1m(str(users_path), n_users)
    
    # Optional: HHO+EO hybrid feature selection on content features
    if use_hho_eo_fs:
        if verbose:
            print("Applying HHO+EO hybrid feature selection on content features...")

        # Item-side feature selection (X_items)
        if X_items is not None:
            item_ids = train_ratings[:, 1].astype(int)
            ratings_vals = train_ratings[:, 2]
            # Her item için ortalama rating
            item_sum = np.bincount(item_ids, weights=ratings_vals, minlength=n_items)
            item_count = np.bincount(item_ids, minlength=n_items)
            item_mean = np.zeros(n_items, dtype=np.float32)
            nonzero_items = item_count > 0
            item_mean[nonzero_items] = item_sum[nonzero_items] / item_count[nonzero_items]
            # Hedef: global ortalamadan yüksek / düşük binary etiket
            y_items = (item_mean >= mu).astype(int)

            item_selector = HybridHHOEOFeatureSelector(
                n_agents=20,
                max_iter=30,
                alpha=0.99,
                cv_splits=5,
                random_state=config.random_seed,
            )
            X_items = item_selector.fit_transform(X_items, y_items)

            if verbose:
                print(f"  Item features reduced to {X_items.shape[1]} dims via HHO+EO.")

        # User-side feature selection (X_users)
        if X_users is not None:
            user_ids = train_ratings[:, 0].astype(int)
            ratings_vals = train_ratings[:, 2]
            user_sum = np.bincount(user_ids, weights=ratings_vals, minlength=n_users)
            user_count = np.bincount(user_ids, minlength=n_users)
            user_mean = np.zeros(n_users, dtype=np.float32)
            nonzero_users = user_count > 0
            user_mean[nonzero_users] = user_sum[nonzero_users] / user_count[nonzero_users]
            y_users = (user_mean >= mu).astype(int)

            user_selector = HybridHHOEOFeatureSelector(
                n_agents=20,
                max_iter=30,
                alpha=0.99,
                cv_splits=5,
                random_state=config.random_seed,
            )
            X_users = user_selector.fit_transform(X_users, y_users)

            if verbose:
                print(f"  User features reduced to {X_users.shape[1]} dims via HHO+EO.")
    
    if verbose:
        print("=" * 60)
        print(f"COLD START {method.upper()} on MovieLens 1M")
        print("=" * 60)
        print(f"Train: {n_train} ratings, Test: {n_test} ratings")
        print(f"Users: {n_users}, Items: {n_items}")
        print(f"μ = {mu:.4f}")
        print(f"V content-init: {X_items is not None}")
        print(f"U content-init: {X_users is not None}")
        print("=" * 60)
    
    # Step 4: Create model with bias + content init, run optimizer
    if method == "mf_sgd":
        model = MFSGD(
            n_users, n_items, config.latent_dim,
            learning_rate=config.learning_rate,
            regularization=config.regularization,
            random_seed=config.random_seed,
            use_bias=True,
            X_items=X_items,
            X_users=X_users
        )
        history = model.fit(train_ratings, config.n_iterations, verbose=verbose)
    elif method == "coa_sgd_mf":
        model = COASGDMF(
            n_users, n_items, config.latent_dim,
            n_coatis=30,
            learning_rate=config.learning_rate,
            regularization=config.regularization,
            boundary=1.0,
            random_seed=config.random_seed,
            use_bias=True,
            X_items=X_items,
            X_users=X_users
        )
        history = model.fit(
            train_ratings,
            coa_iterations=50,
            sgd_iterations=config.n_iterations,
            verbose=verbose
        )
    elif method == "hho_sgd_mf":
        # HHO hyperparameters
        hho_n_hawks = 60
        hho_escape_energy_initial = 1.5
        hho_boundary = 1.5
        hho_iterations = 50

        model = HHOSGDMF(
            n_users, n_items, config.latent_dim,
            n_hawks=hho_n_hawks,
            escape_energy_initial=hho_escape_energy_initial,
            learning_rate=config.learning_rate,
            regularization=config.regularization,
            boundary=hho_boundary,
            random_seed=config.random_seed,
            use_bias=True,
            X_items=X_items,
            X_users=X_users
        )
        if verbose:
            print(
                f"HHO params: n_hawks={hho_n_hawks}, "
                f"E0={hho_escape_energy_initial}, "
                f"boundary={hho_boundary}, "
                f"hho_iterations={hho_iterations}, "
                f"sgd_iterations={config.n_iterations}"
            )
        history = model.fit(
            train_ratings,
            hho_iterations=hho_iterations,
            sgd_iterations=config.n_iterations,
            verbose=verbose,
        )
    elif method == "hho_eo_mf":
        # HHO+EO pure metaheuristic MF
        hho_eo_n_agents = 40
        hho_eo_iterations = 100
        hho_eo_boundary = 1.5

        model = HHOEOMF(
            n_users,
            n_items,
            config.latent_dim,
            n_agents=hho_eo_n_agents,
            escape_energy_initial=1.5,
            regularization=config.regularization,
            boundary=hho_eo_boundary,
            random_seed=config.random_seed,
            use_bias=True,
            X_items=X_items,
            X_users=X_users,
        )
        if verbose:
            print(
                f"HHO-EO params: n_agents={hho_eo_n_agents}, "
                f"E0=1.5, boundary={hho_eo_boundary}, "
                f"iterations={hho_eo_iterations}"
            )
        history = model.fit(
            train_ratings,
            n_iterations=hho_eo_iterations,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use mf_sgd, coa_sgd_mf, or hho_sgd_mf")
    
    rmse_value, mae_value = model.evaluate(test_ratings)
    
    # Get final loss from history (format varies by method)
    if "losses" in history:
        final_loss = history["losses"][-1]
    elif "sgd_losses" in history:
        final_loss = history["sgd_losses"][-1]
    else:
        final_loss = float("nan")
    
    # Step 5: Demonstrate folding-in (new user with ratings)
    if verbose and n_items > 0:
        # Simulate new user: take first 5 ratings from test, treat as "new user"
        test_user_ids = test_ratings[:, 0].astype(int)
        first_user = test_user_ids[0]
        mask = test_user_ids == first_user
        user_test = test_ratings[mask]
        if len(user_test) >= 3:
            user_ratings = user_test[:3, [1, 2]]  # item_id, rating
            U_new = model.fold_in_new_user(user_ratings)
            print(f"\nFolding-in: new user with 3 ratings -> U shape {U_new.shape}")
    
    # Step 6: Demonstrate add_new_item (new film)
    if verbose and X_items is not None and model.get_model().W_item is not None:
        # New film: Comedy|Drama (example)
        genre_vec = np.zeros(X_items.shape[1], dtype=np.float32)
        # ML1M genres: Comedy=4, Drama=7
        if X_items.shape[1] >= 8:
            genre_vec[4] = 1.0  # Comedy
            genre_vec[7] = 1.0  # Drama
        try:
            new_idx = model.add_new_item(genre_vec)
            print(f"add_new_item: new film (Comedy|Drama) -> item index {new_idx}")
        except ValueError:
            pass
    
    results = {
        'method': f'Cold-Start-{method.upper()}',
        'dataset': 'MovieLens 1M',
        'rmse': rmse_value,
        'mae': mae_value,
        'final_loss': final_loss,
        'n_users': n_users,
        'n_items': n_items,
        'n_train': n_train,
        'n_test': n_test,
        'use_bias': True,
        'content_init': X_items is not None,
        'use_hho_eo_fs': use_hho_eo_fs,
        'history': history
    }

    if method == "hho_sgd_mf":
        results['hho_n_hawks'] = hho_n_hawks
        results['hho_escape_energy_initial'] = hho_escape_energy_initial
        results['hho_boundary'] = hho_boundary
        results['hho_iterations'] = hho_iterations
        results['sgd_iterations'] = config.n_iterations
        results['latent_dim'] = config.latent_dim
    
    # Save results
    results_dir = Path("results") / "movielens-1m" / f"cold_start_{method}"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'method': results['method'],
        'dataset': results['dataset'],
        'rmse': float(results['rmse']),
        'mae': float(results['mae']),
        'final_loss': float(results['final_loss']),
        'use_bias': results['use_bias'],
        'content_init': results['content_init'],
        'use_hho_eo_fs': results['use_hho_eo_fs'],
    }

    if method == "hho_sgd_mf":
        summary['hho_n_hawks'] = results['hho_n_hawks']
        summary['hho_escape_energy_initial'] = results['hho_escape_energy_initial']
        summary['hho_boundary'] = results['hho_boundary']
        summary['hho_iterations'] = results['hho_iterations']
        summary['sgd_iterations'] = results['sgd_iterations']
        summary['latent_dim'] = results['latent_dim']
    if method == "hho_eo_mf":
        summary["hho_eo_n_agents"] = hho_eo_n_agents
        summary["hho_eo_iterations"] = hho_eo_iterations
        summary["hho_eo_boundary"] = hho_eo_boundary
        summary["latent_dim"] = config.latent_dim

    with open(results_dir / "summary.json", "w") as f:
        import json
        json.dump(summary, f, indent=2)
    
    return results


def main():
    """Run Cold Start MF experiment."""
    parser = argparse.ArgumentParser(
        description="Run Cold Start MF on MovieLens 1M"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mf_sgd",
        choices=["mf_sgd", "coa_sgd_mf", "hho_sgd_mf", "hho_eo_mf"],
        help="Algorithm: mf_sgd (default), coa_sgd_mf, hho_sgd_mf, hho_eo_mf",
    )
    parser.add_argument(
        "--use_hho_eo_fs",
        action="store_true",
        help="Apply HHO+EO hybrid feature selection on content features before MF init.",
    )
    args = parser.parse_args()
    
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
    run_cold_start_mf_ml1m(
        config,
        method=args.method,
        use_hho_eo_fs=args.use_hho_eo_fs,
        verbose=True,
    )


if __name__ == "__main__":
    main()
