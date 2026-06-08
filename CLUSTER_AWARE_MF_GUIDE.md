# Cluster-Aware Matrix Factorization Guide

## Problem Statement

Your current findings show:
- Different clustering algorithms (B0_KMEANS, HA_AVOAHGS, IWO_HHO, LIT_GWO) produce **meaningfully different clusters**
  - K=4: 20-97% assignment agreement
  - K=10: 4-65% assignment agreement
  - K=27: 0-30% assignment agreement

- But prediction metrics are almost identical (0.95-1.98% variation)
  - This means the clustering differences are being **masked by the prediction method**
  - Cluster-weighted KNN is too robust to cluster structure

**Solution**: Use a prediction method where **cluster quality directly impacts prediction quality**.

## How Cluster-Aware MF Works

Traditional MF: 
```
U_i ~ N(0, 1)    (random initialization)
V_j ~ N(0, 1)
rating_ij = U_i . V_j
```

Cluster-Aware MF:
```
U_i = C_{cluster(i)} + delta_i
    (initialized at cluster centroid + small offset)

Loss = ||R - UV^T||^2 + lambda * ||U_i - C_{cluster(i)}||^2
```

**Key idea**: Pull user factors toward their cluster centroid during training.

Different clusterings -> Different cluster centroids -> Different user factors -> Different predictions

## Three Implementation Options

### Option 1: ClusterAwareMF (Recommended for your system)

```python
from cluster_aware_mf import ClusterAwareMF

model = ClusterAwareMF(
    n_latent=20,           # dimension of your WNMF latent space
    n_epochs=50,
    learning_rate=0.01,
    lambda_cluster=0.1     # strength of cluster regularization (0.01-1.0)
)

# cluster_centroids: shape (K, 20) from your centroid optimization
# user_assignments: shape (943,) from clustering
model.fit(ratings_train, cluster_centroids, user_assignments, 
          val_ratings=ratings_val, verbose=True)

predictions = model.predict_batch(user_ids, item_ids)
```

**Pros**:
- Users can deviate from cluster centroid based on personal preferences
- Flexible: lambda_cluster tunes how much clustering matters
- Works like standard MF but grounded in clusters

**Cons**:
- Need to tune lambda_cluster per dataset
- More parameters to optimize

### Option 2: SimplePureClusterMF (Strictest cluster binding)

```python
from cluster_aware_mf import SimplePureClusterMF

model = SimplePureClusterMF(
    n_latent=20,
    n_epochs=50,
    learning_rate=0.01,
    lambda_cluster=0.01
)

# All users in same cluster share identical factors
model.fit(ratings_train, user_assignments, val_ratings=ratings_val)
```

**Pros**:
- Simplest approach: users in same cluster have identical factors
- Pure test: does cluster structure matter?
- Fewer parameters

**Cons**:
- Very rigid: doesn't account for user-specific preferences within cluster
- May perform worse than traditional CF

### Option 3: Hard cluster averaging (Baseline)

This is simpler and requires no learning:
```python
# For each user, predict as average of their cluster's ratings
cluster_avg = {}
for cluster_id in range(n_clusters):
    members = np.where(user_assignments == cluster_id)[0]
    cluster_avg[cluster_id] = ratings[members].mean(axis=0)

prediction[user_i] = cluster_avg[cluster(i)]
```

## Integration with Your System

### Step 1: Modify WNMF Experiment to include cluster-aware scenarios

In `wnmf_experiment.py`, add new scenarios:

```python
SCENARIOS = {
    # ... existing scenarios ...
    'cluster_aware_mf_lambda01': {
        'predictor': 'cluster_aware_mf',
        'lambda_cluster': 0.1
    },
    'cluster_aware_mf_lambda1': {
        'predictor': 'cluster_aware_mf',
        'lambda_cluster': 1.0
    },
    'pure_cluster_mf': {
        'predictor': 'pure_cluster_mf'
    }
}
```

### Step 2: Load cluster data in WNMF

```python
def load_cluster_centroids(algo, k, suffix):
    """Load saved centroids from clustering optimization."""
    dir_name = f"mealpy/results/assignments_lof/ml100k/{algo}_{suffix}_k{k}"
    
    best_sol = np.load(f"{dir_name}/best_sol.npy")
    assignments = np.load(f"{dir_name}/assignments.npy")
    user_vectors = np.load(f"{dir_name}/wnmf_user_vectors.npy")
    
    dim = user_vectors.shape[1]
    centroids = best_sol.reshape(k, dim)
    
    return centroids, assignments
```

### Step 3: Train and evaluate

```python
for algo in ['B0_KMEANS', 'HA_AVOAHGS', 'IWO_HHO', 'LIT_GWO']:
    for k in [4, 10, 27]:
        centroids, assignments = load_cluster_centroids(algo, k, suffix)
        
        # Train cluster-aware MF
        model = ClusterAwareMF(lambda_cluster=0.1)
        model.fit(ratings_train, centroids, assignments, 
                  val_ratings=ratings_val)
        
        # Evaluate on test set
        predictions = model.predict_batch(test_user_ids, test_item_ids)
        mae = mean_absolute_error(test_ratings, predictions)
        
        # Store result
        store_result(algo, scenario='cluster_aware_mf_lambda01', 
                    mae=mae, k=k)
```

## Expected Results

If your hypothesis is correct:

**Traditional cluster_avg**:
- K=4: MAE 0.753-0.760 (0.95% variation)
- K=10: MAE 0.755-0.822 (8.9% variation - higher due to larger K)
- K=27: MAE 0.790-0.820 (3.8% variation)

**Cluster-Aware MF with lambda_cluster=0.1**:
- K=4: MAE should differ by 3-8% between algorithms
- K=10: MAE should differ by 5-15% between algorithms
- K=27: MAE should differ by 5-12% between algorithms

**SimplePureClusterMF**:
- K=4: MAE should differ by 2-5% between algorithms
- K=10: MAE should differ by 3-8% between algorithms
- K=27: MAE should differ by 1-3% between algorithms

## Tuning lambda_cluster

This parameter controls how strongly user factors are pulled toward cluster centroids:

- **lambda_cluster = 0.01**: Weak regularization, users can deviate far from cluster
  - Similar to regular MF, clustering has weak effect
  - Higher MAE overall, small variation between algorithms

- **lambda_cluster = 0.1**: Medium regularization (recommended)
  - Good balance between cluster influence and user preferences
  - Should show meaningful variation between algorithms

- **lambda_cluster = 1.0**: Strong regularization, users stay close to cluster
  - User factors dominated by cluster centroid
  - Lower MAE if clusters are good, higher if clusters are bad
  - Maximum variation between algorithms

## Running the test

```bash
# Test cluster-aware MF on your data
python test_cluster_aware_cf.py
```

This will:
1. Load your cluster data for B0_KMEANS, HA_AVOAHGS, IWO_HHO, LIT_GWO
2. Train SimplePureClusterMF and ClusterAwareMF on each
3. Show MAE variation between algorithms
4. Compare to your current cluster_avg baseline

## Conclusion

Cluster-aware MF directly ties recommendation quality to cluster structure.

If different clustering algorithms still produce similar MAE with cluster-aware MF,
it means **the clustering differences don't actually matter for the user preference structure in this dataset**.

If they show significant variation (>5%), then **you've found a true differentiator for your clustering algorithms**.
