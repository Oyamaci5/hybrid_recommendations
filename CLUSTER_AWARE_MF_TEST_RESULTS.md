# Cluster-Aware Matrix Factorization Test Results

## Test Setup
- MovieLens 100K dataset (943 users, 1682 items, 100K ratings)
- Train/Val split: 80/20
- K = 4 clusters
- 4 algorithms: B0_KMEANS, HA_AVOAHGS, IWO_HHO, LIT_GWO
- 15-20 epochs training

## Results

### Baseline (Previous Analysis)
- **cluster_avg**: 0.95% MAE variation (0.753-0.760)
- **cluster_knn_native_baseline**: 1.98% MAE variation (0.753-0.768)

### Cluster-Aware Approaches Tested

#### 1. SimplePureClusterMF
All users in same cluster share identical learned factors.

| Algorithm | MAE |
|-----------|-----|
| IWO_HHO | 0.828226 |
| LIT_GWO | 0.828621 |
| B0_KMEANS | 0.831295 |
| HA_AVOAHGS | 0.833427 |

**Variation: 0.63%**

#### 2. ClusterAwareMF (lambda_cluster=0.1)
Users regularized toward cluster centroid with medium strength.

| Algorithm | MAE |
|-----------|-----|
| B0_KMEANS | 0.848076 |
| HA_AVOAHGS | 0.848155 |
| LIT_GWO | 0.848421 |
| IWO_HHO | 0.848624 |

**Variation: 0.06%**

## Key Finding

**Cluster-aware MF approaches show LESS variation, not more!**

| Approach | K=4 Variation |
|----------|---------------|
| cluster_avg baseline | 0.95% |
| cluster_knn baseline | 1.98% |
| SimplePureClusterMF | 0.63% |
| ClusterAwareMF (lambda=0.1) | 0.06% |

## Interpretation

### What This Means

1. **Cluster-aware regularization REDUCES differentiation** between algorithms
2. **The constraints pull solutions toward convergence** despite different starting cluster structures
3. **Matrix factorization learning smooths out cluster differences** during training

### Why This Happens

- The cluster-aware regularization term acts like an additional loss term that forces convergence
- As users learn factors close to cluster centroids, the model becomes less sensitive to exact centroid positions
- The learning process for item factors adapts to average user patterns, masking cluster differences

## Conclusion

**Cluster-aware MF was the wrong approach for this problem.**

The hypothesis was: "If we tie user factors to clusters, we'll see differentiation."

The reality is: "Tying factors to clusters actually INCREASES convergence, reducing differentiation."

### Better Approaches (Not Yet Tested)

1. **Cluster-specific models**
   - Train separate item factors for each cluster
   - Share across clusters only during initialization

2. **Non-negative constraints**
   - Enforce user factors stay close to cluster centroid bounds
   - Prevent convergence to identical solutions

3. **Per-user bias with cluster priors**
   - User bias = cluster_bias + personal_offset
   - Learn cluster biases separately

4. **Cluster purity metrics**
   - Measure how well predictions correlate with cluster assignments
   - A metric that directly tests cluster quality rather than prediction accuracy

5. **Assignment-based evaluation**
   - Use the cluster assignments themselves as targets
   - Test: can we predict which cluster a user belongs to?
   - Different algorithms -> different assignments -> different prediction accuracy

## Next Step

The fundamental issue is that we're trying to use **prediction quality** as a proxy for **cluster quality**.

For different clusters to matter, they must structure the data in meaningfully different ways that affect some target.

**Recommendation**: Evaluate clusters using **clustering-specific metrics** (silhouette, Davies-Bouldin, calinski-harabasz) or use clusters for **interpretability** (e.g., "what do cluster 1 users prefer?") rather than trying to force differentiation in prediction accuracy.
