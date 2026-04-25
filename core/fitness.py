"""
Clustering fitness utilities used by custom optimizers.
"""

from __future__ import annotations

import numpy as np

from core.metrics import pearson_distance


def calculate_clustering_fitness(
    centroids: np.ndarray,
    user_data: np.ndarray,
    k: int,
) -> float:
    """
    Evaluate centroid quality using Pearson distance assignments.
    """
    total_error = 0.0
    cluster_assignments = [[] for _ in range(int(k))]

    for user in user_data:
        distances = [pearson_distance(user, c) for c in centroids]
        best_cluster_idx = int(np.argmin(distances))
        cluster_assignments[best_cluster_idx].append(float(distances[best_cluster_idx]))

    for i in range(int(k)):
        if len(cluster_assignments[i]) == 0:
            total_error += 1000.0
        else:
            total_error += float(np.sum(cluster_assignments[i]))

    return float(total_error)

