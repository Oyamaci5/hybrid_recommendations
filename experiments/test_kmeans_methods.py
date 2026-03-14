"""
Quick test script for K-Means methods before creating full experiment runners.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from core.config import Config, set_random_seed
from core.utils import load_train_test_split, get_data_info
from methods.hho_kmeans_mf import HHOKMeansMF
from methods.coa_kmeans_mf import COAKMeansMF


def test_hho_kmeans_mf():
    """Test HHO-KMeans-MF with small dataset."""
    print("=" * 60)
    print("Testing HHO-KMeans-MF")
    print("=" * 60)
    
    config = Config()
    set_random_seed(config.random_seed)
    
    # Load small subset of data for quick test
    train_path = config.get_train_path()
    test_path = config.get_test_path()
    train_ratings, test_ratings = load_train_test_split(train_path, test_path)
    
    # Use more ratings to ensure enough users/items for clustering
    train_ratings = train_ratings[:500]
    
    n_users, n_items, n_train = get_data_info(train_ratings)
    
    print(f"Test data: {n_train} ratings, {n_users} users, {n_items} items")
    
    # Initialize model
    model = HHOKMeansMF(
        n_users=n_users,
        n_items=n_items,
        latent_dim=5,  # Small dimension for quick test
        n_hawks=10  # Small population
    )
    
    # Test fit
    print("\nTesting fit()...")
    history = model.fit(train_ratings, hho_iterations=5, kmeans_iterations=5, verbose=True)
    
    print(f"\nHHO losses: {len(history['hho_losses'])} iterations")
    print(f"K-Means losses: {len(history['kmeans_losses'])} iterations")
    
    # Test predict
    print("\nTesting predict()...")
    user_ids = test_ratings[:10, 0].astype(int)
    item_ids = test_ratings[:10, 1].astype(int)
    predictions = model.predict(user_ids, item_ids)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    
    # Test evaluate
    print("\nTesting evaluate()...")
    rmse, mae = model.evaluate(test_ratings[:50])  # Small test set
    print(f"RMSE: {rmse:.6f}, MAE: {mae:.6f}")
    
    print("\n[+] HHO-KMeans-MF test passed!")
    return True


def test_coa_kmeans_mf():
    """Test COA-KMeans-MF with small dataset."""
    print("\n" + "=" * 60)
    print("Testing COA-KMeans-MF")
    print("=" * 60)
    
    config = Config()
    set_random_seed(config.random_seed)
    
    # Load small subset of data for quick test
    train_path = config.get_train_path()
    test_path = config.get_test_path()
    train_ratings, test_ratings = load_train_test_split(train_path, test_path)
    
    # Use more ratings to ensure enough users/items for clustering
    train_ratings = train_ratings[:500]
    
    n_users, n_items, n_train = get_data_info(train_ratings)
    
    print(f"Test data: {n_train} ratings, {n_users} users, {n_items} items")
    
    # Initialize model
    model = COAKMeansMF(
        n_users=n_users,
        n_items=n_items,
        latent_dim=5,  # Small dimension for quick test
        n_coatis=10  # Small population
    )
    
    # Test fit
    print("\nTesting fit()...")
    history = model.fit(train_ratings, coa_iterations=5, kmeans_iterations=5, verbose=True)
    
    print(f"\nCOA losses: {len(history['coa_losses'])} iterations")
    print(f"K-Means losses: {len(history['kmeans_losses'])} iterations")
    
    # Test predict
    print("\nTesting predict()...")
    user_ids = test_ratings[:10, 0].astype(int)
    item_ids = test_ratings[:10, 1].astype(int)
    predictions = model.predict(user_ids, item_ids)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    
    # Test evaluate
    print("\nTesting evaluate()...")
    rmse, mae = model.evaluate(test_ratings[:50])  # Small test set
    print(f"RMSE: {rmse:.6f}, MAE: {mae:.6f}")
    
    print("\n[+] COA-KMeans-MF test passed!")
    return True


def main():
    """Run all tests."""
    try:
        test_hho_kmeans_mf()
        test_coa_kmeans_mf()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[-] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True


if __name__ == "__main__":
    main()

