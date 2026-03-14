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

