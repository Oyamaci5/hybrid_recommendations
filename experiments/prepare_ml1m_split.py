"""
Prepare train/test split for MovieLens 1M dataset.
Creates standard 80/20 split with stratified sampling per user.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from core.utils import load_ratings, create_train_test_split


def prepare_ml1m_split(data_dir: str = "data/ml-1m", output_dir: str = "data/ml-1m",
                       test_ratio: float = 0.2, random_seed: int = 42):
    """
    Create train/test split for MovieLens 1M.
    
    Args:
        data_dir: Directory containing ratings.dat
        output_dir: Directory to save train.dat and test.dat
        test_ratio: Ratio of test ratings (default: 0.2)
        random_seed: Random seed for reproducibility
    """
    ratings_path = Path(data_dir) / "ratings.dat"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading ratings from: {ratings_path}")
    ratings = load_ratings(str(ratings_path))
    print(f"Loaded {len(ratings)} ratings")
    
    print(f"Creating train/test split (test_ratio={test_ratio})...")
    train_ratings, test_ratings = create_train_test_split(
        ratings, test_ratio=test_ratio, random_seed=random_seed
    )
    
    print(f"Train: {len(train_ratings)} ratings")
    print(f"Test: {len(test_ratings)} ratings")
    
    # Save train split (using MovieLens 1M format: :: separator)
    train_path = output_path / "train.dat"
    with open(train_path, 'w', encoding='latin-1') as f:
        for rating in train_ratings:
            user_id = int(rating[0]) + 1  # Convert back to 1-indexed
            item_id = int(rating[1]) + 1  # Convert back to 1-indexed
            rating_val = float(rating[2])
            timestamp = 978300000  # Placeholder timestamp
            f.write(f"{user_id}::{item_id}::{rating_val}::{timestamp}\n")
    
    # Save test split
    test_path = output_path / "test.dat"
    with open(test_path, 'w', encoding='latin-1') as f:
        for rating in test_ratings:
            user_id = int(rating[0]) + 1  # Convert back to 1-indexed
            item_id = int(rating[1]) + 1  # Convert back to 1-indexed
            rating_val = float(rating[2])
            timestamp = 978300000  # Placeholder timestamp
            f.write(f"{user_id}::{item_id}::{rating_val}::{timestamp}\n")
    
    print(f"\nTrain split saved to: {train_path}")
    print(f"Test split saved to: {test_path}")
    print("Split preparation complete!")


if __name__ == "__main__":
    prepare_ml1m_split(random_seed=42)

