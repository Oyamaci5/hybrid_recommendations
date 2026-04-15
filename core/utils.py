"""
Data loading and preprocessing utilities for MovieLens datasets.
Supports both MovieLens 100K and MovieLens 1M formats.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List

# Genre lists for MovieLens (excluding 'unknown' for ML-1M compatibility)
ML1M_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

ML100K_GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]


def load_ratings(filepath: str) -> np.ndarray:
    """
    Load ratings from MovieLens format (supports both 100K and 1M).
    
    MovieLens 100K format: user_id | item_id | rating | timestamp (tab-separated)
    MovieLens 1M format: UserID::MovieID::Rating::Timestamp (double colon-separated)
    
    Args:
        filepath: Path to the ratings file
        
    Returns:
        Array of shape (n_ratings, 3) with columns [user_id, item_id, rating]
        Note: user_id and item_id are 0-indexed (original data is 1-indexed)
    """
    data = []
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Try double colon separator first (MovieLens 1M)
            if '::' in line:
                parts = line.split('::')
                if len(parts) >= 3:
                    user_id = int(parts[0]) - 1  # Convert to 0-indexed
                    item_id = int(parts[1]) - 1  # Convert to 0-indexed
                    rating = float(parts[2])
                    data.append([user_id, item_id, rating])
            # Try tab separator (MovieLens 100K)
            elif '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 3:
                    user_id = int(parts[0]) - 1  # Convert to 0-indexed
                    item_id = int(parts[1]) - 1  # Convert to 0-indexed
                    rating = float(parts[2])
                    data.append([user_id, item_id, rating])
    
    return np.array(data, dtype=np.float32)


def load_train_test_split(base_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load train and test splits.
    
    Args:
        base_path: Path to training file (e.g., u1.base, train.dat)
        test_path: Path to test file (e.g., u1.test, test.dat)
        
    Returns:
        Tuple of (train_ratings, test_ratings) arrays
    """
    train_ratings = load_ratings(base_path)
    test_ratings = load_ratings(test_path)
    return train_ratings, test_ratings


def create_train_test_split(ratings: np.ndarray, test_ratio: float = 0.2,
                           random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create train/test split from ratings array.
    
    Uses stratified split: for each user, randomly select test_ratio of their ratings.
    
    Args:
        ratings: Ratings array of shape (n_ratings, 3) with [user_id, item_id, rating]
        test_ratio: Ratio of test ratings (default: 0.2 for 80/20 split)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_ratings, test_ratings) arrays
    """
    np.random.seed(random_seed)
    
    train_data = []
    test_data = []
    
    # Group ratings by user
    user_ratings = {}
    for rating in ratings:
        user_id = int(rating[0])
        if user_id not in user_ratings:
            user_ratings[user_id] = []
        user_ratings[user_id].append(rating)
    
    # Split each user's ratings
    for user_id, user_ratings_list in user_ratings.items():
        n_ratings = len(user_ratings_list)
        n_test = max(1, int(n_ratings * test_ratio))
        
        # perm is a shuffled list of positions (0..n_ratings-1); the first
        # n_test positions are designated as test.
        perm = np.random.permutation(n_ratings)
        test_positions = set(perm[:n_test].tolist())

        for i, rating in enumerate(user_ratings_list):
            if i in test_positions:
                test_data.append(rating)
            else:
                train_data.append(rating)
    
    train_ratings = np.array(train_data, dtype=np.float32)
    test_ratings = np.array(test_data, dtype=np.float32)
    
    return train_ratings, test_ratings


def get_data_info(ratings: np.ndarray) -> Tuple[int, int, int]:
    """
    Get dataset information from ratings array.
    
    Args:
        ratings: Array of shape (n_ratings, 3) with [user_id, item_id, rating]
        
    Returns:
        Tuple of (n_users, n_items, n_ratings)
    """
    if len(ratings) == 0:
        return 0, 0, 0
    
    n_users = int(ratings[:, 0].max() + 1)
    n_items = int(ratings[:, 1].max() + 1)
    n_ratings = len(ratings)
    
    return n_users, n_items, n_ratings


def create_rating_matrix(ratings: np.ndarray, n_users: int, n_items: int) -> np.ndarray:
    """
    Create a dense rating matrix from sparse ratings array.
    
    Args:
        ratings: Array of shape (n_ratings, 3) with [user_id, item_id, rating]
        n_users: Number of users
        n_items: Number of items
        
    Returns:
        Rating matrix of shape (n_users, n_items), unrated items are 0
    """
    R = np.zeros((n_users, n_items), dtype=np.float32)
    user_ids = ratings[:, 0].astype(int)
    item_ids = ratings[:, 1].astype(int)
    rating_values = ratings[:, 2]
    
    R[user_ids, item_ids] = rating_values
    return R


def load_movies_ml1m(filepath: str, n_items: int) -> np.ndarray:
    """
    Load movie genre matrix from MovieLens 1M movies.dat.
    
    Format: MovieID::Title::Genres (pipe-separated genres)
    Returns multi-hot genre matrix of shape (n_items, n_genres).
    
    Args:
        filepath: Path to movies.dat
        n_items: Number of items (from ratings; max item_id + 1)
        
    Returns:
        Genre matrix of shape (n_items, n_genres), dtype float32.
        Genre order: ML1M_GENRES. Unknown/missing movies get zero vector.
    """
    n_genres = len(ML1M_GENRES)
    genre_to_idx = {g: i for i, g in enumerate(ML1M_GENRES)}
    X = np.zeros((n_items, n_genres), dtype=np.float32)
    
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('::')
            if len(parts) < 3:
                continue
            movie_id = int(parts[0])
            item_id = movie_id - 1  # 0-indexed
            if item_id < 0 or item_id >= n_items:
                continue
            genres_str = parts[2].strip()
            if genres_str and genres_str != '(no genres listed)':
                for g in genres_str.split('|'):
                    g = g.strip()
                    if g in genre_to_idx:
                        X[item_id, genre_to_idx[g]] = 1.0
    return X


def load_movies_100k(filepath: str, n_items: int) -> np.ndarray:
    """
    Load movie genre matrix from MovieLens 100K u.item.
    
    Format: movie_id | title | ... | 19 genre columns (0/1)
    Returns multi-hot genre matrix of shape (n_items, n_genres).
    
    Args:
        filepath: Path to u.item
        n_items: Number of items (from ratings; max item_id + 1)
        
    Returns:
        Genre matrix of shape (n_items, n_genres), dtype float32.
        Genre order: ML100K_GENRES. Unknown/missing movies get zero vector.
    """
    n_genres = len(ML100K_GENRES)
    X = np.zeros((n_items, n_genres), dtype=np.float32)
    
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) < 6 + n_genres:
                continue
            movie_id = int(parts[0])
            item_id = movie_id - 1  # 0-indexed
            if item_id < 0 or item_id >= n_items:
                continue
            for i in range(n_genres):
                idx = 5 + i  # columns 5-23 are genres
                if idx < len(parts):
                    X[item_id, i] = float(parts[idx]) if parts[idx] else 0.0
    return X


def load_users_ml1m(filepath: str, n_users: int) -> np.ndarray:
    """
    Load user demographics (gender, age, occupation) for cold start.
    
    Format: UserID::Gender::Age::Occupation::Zip-code
    Returns encoded matrix of shape (n_users, n_features).
    
    Args:
        filepath: Path to users.dat
        n_users: Number of users
        
    Returns:
        User feature matrix of shape (n_users, n_features).
        Features: gender (1), age (1), occupation (21 one-hot) -> 23 dims.
    """
    # Occupation: 0-20 (21 values)
    n_features = 1 + 1 + 21  # gender + age + occupation
    X = np.zeros((n_users, n_features), dtype=np.float32)
    
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('::')
            if len(parts) < 4:
                continue
            user_id = int(parts[0])
            u_idx = user_id - 1  # 0-indexed
            if u_idx < 0 or u_idx >= n_users:
                continue
            gender = 1.0 if parts[1].strip().upper() == 'M' else 0.0
            age = float(parts[2]) if parts[2] else 25.0
            occ = int(parts[3]) if parts[3].isdigit() else 0
            occ = min(max(occ, 0), 20)
            X[u_idx, 0] = gender
            X[u_idx, 1] = age / 56.0  # normalize
            X[u_idx, 2 + occ] = 1.0
    return X


def load_users_100k(filepath: str, n_users: int) -> np.ndarray:
    """
    Load user demographics from MovieLens 100K u.user.
    
    Format: user_id | age | gender | occupation | zip
    Returns encoded matrix of shape (n_users, n_features).
    
    Args:
        filepath: Path to u.user
        n_users: Number of users
        
    Returns:
        User feature matrix of shape (n_users, n_features).
    """
    occupations = [
        "other", "academic/educator", "artist", "clerical/admin", "college/grad student",
        "customer service", "doctor/health care", "executive/managerial", "farmer",
        "homemaker", "K-12 student", "lawyer", "programmer", "retired", "sales/marketing",
        "scientist", "self-employed", "technician/engineer", "tradesman/craftsman",
        "unemployed", "writer"
    ]
    occ_to_idx = {o: i for i, o in enumerate(occupations)}
    n_features = 1 + 1 + len(occupations)  # gender + age + occupation
    X = np.zeros((n_users, n_features), dtype=np.float32)
    
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) < 4:
                continue
            user_id = int(parts[0])
            u_idx = user_id - 1
            if u_idx < 0 or u_idx >= n_users:
                continue
            age = float(parts[1]) if parts[1].isdigit() else 25.0
            gender = 1.0 if parts[2].strip().upper() == 'M' else 0.0
            occ = parts[3].strip().lower() if len(parts) > 3 else "other"
            occ_idx = occ_to_idx.get(occ, 0)
            X[u_idx, 0] = gender
            X[u_idx, 1] = age / 100.0  # normalize
            X[u_idx, 2 + occ_idx] = 1.0
    return X


def genre_matrix_to_embedding(X_items: np.ndarray, latent_dim: int,
                              random_seed: Optional[int] = None
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project genre multi-hot matrix to latent embeddings for V initialization.
    
    V_init[j] = X_items[j] @ W, where W is (n_genres, latent_dim).
    
    Args:
        X_items: Genre matrix of shape (n_items, n_genres)
        latent_dim: Latent dimension
        random_seed: Random seed for W
        
    Returns:
        Tuple of (V_init, W) where V_init is (n_items, latent_dim), W is (n_genres, latent_dim).
        W is stored for add_new_item cold start.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    n_genres = X_items.shape[1]
    scale = np.sqrt(1.0 / (n_genres * latent_dim))
    W = np.random.normal(0, scale, (n_genres, latent_dim)).astype(np.float32)
    V_init = X_items @ W
    return V_init, W


def user_matrix_to_embedding(X_users: np.ndarray, latent_dim: int,
                             random_seed: Optional[int] = None) -> np.ndarray:
    """
    Project user demographics to latent embeddings for U initialization.
    
    Args:
        X_users: User feature matrix of shape (n_users, n_features)
        latent_dim: Latent dimension
        random_seed: Random seed for W
        
    Returns:
        U_init of shape (n_users, latent_dim)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    n_features = X_users.shape[1]
    scale = np.sqrt(1.0 / (n_features * latent_dim))
    W = np.random.normal(0, scale, (n_features, latent_dim)).astype(np.float32)
    return (X_users @ W).astype(np.float32)


def compute_biases(ratings: np.ndarray, n_users: int, n_items: int
                  ) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute global mean and user/item biases from ratings.
    
    Uses iterative updates: b_u = mean(r - μ - b_i), b_i = mean(r - μ - b_u).
    
    Args:
        ratings: Array of shape (n_ratings, 3) with [user_id, item_id, rating]
        n_users: Number of users
        n_items: Number of items
        
    Returns:
        Tuple of (mu, b_u, b_i)
    """
    mu = float(np.mean(ratings[:, 2]))
    b_u = np.zeros(n_users, dtype=np.float32)
    b_i = np.zeros(n_items, dtype=np.float32)
    
    user_ids = ratings[:, 0].astype(int)
    item_ids = ratings[:, 1].astype(int)
    r = ratings[:, 2]
    
    # User/item counts for averaging
    user_counts = np.bincount(user_ids, minlength=n_users)
    item_counts = np.bincount(item_ids, minlength=n_items)
    
    # Iterate a few times to converge
    for _ in range(10):
        resid = r - mu - b_i[item_ids]
        for u in range(n_users):
            mask = user_ids == u
            if user_counts[u] > 0:
                b_u[u] = np.mean(resid[mask])
        
        resid = r - mu - b_u[user_ids]
        for i in range(n_items):
            mask = item_ids == i
            if item_counts[i] > 0:
                b_i[i] = np.mean(resid[mask])
    
    return mu, b_u, b_i

