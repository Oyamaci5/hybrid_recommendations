"""
Evaluation metrics for recommender systems.
"""

import numpy as np
from typing import Tuple


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Args:
        y_true: True ratings
        y_pred: Predicted ratings
        
    Returns:
        RMSE value
    """
    if len(y_true) == 0:
        return 0.0
    
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        y_true: True ratings
        y_pred: Predicted ratings
        
    Returns:
        MAE value
    """
    if len(y_true) == 0:
        return 0.0
    
    return np.mean(np.abs(y_true - y_pred))


def evaluate_predictions(
    true_ratings: np.ndarray,
    pred_ratings: np.ndarray
) -> Tuple[float, float]:
    """
    Evaluate predictions using both RMSE and MAE.
    
    Args:
        true_ratings: True ratings array
        pred_ratings: Predicted ratings array
        
    Returns:
        Tuple of (RMSE, MAE)
    """
    rmse_value = rmse(true_ratings, pred_ratings)
    mae_value = mae(true_ratings, pred_ratings)
    return rmse_value, mae_value


def pearson_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Compute Pearson-based distance between two user vectors.

    Returns:
        0.0 -> highly similar trend
        2.0 -> highly dissimilar/opposite trend
    """
    common_indices = np.where((vector_a > 0) & (vector_b > 0))[0]
    if len(common_indices) < 2:
        return 2.0

    a_vals = vector_a[common_indices]
    b_vals = vector_b[common_indices]

    a_diff = a_vals - np.mean(a_vals)
    b_diff = b_vals - np.mean(b_vals)

    numerator = np.sum(a_diff * b_diff)
    denominator = np.sqrt(np.sum(a_diff ** 2) * np.sum(b_diff ** 2))
    if denominator == 0:
        return 2.0

    correlation = numerator / denominator
    return float(1.0 - correlation)

