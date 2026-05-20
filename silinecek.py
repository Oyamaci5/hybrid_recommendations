import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load ML-100k
df = pd.read_csv(
    r"d:\hybrid_recommendations\data\ml-100k\u.data",
    sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"],
)
matrix = df.pivot_table(
    index="user_id", columns="item_id", values="rating", fill_value=0
).values.astype(np.float32)

print(f"Matrix shape: {matrix.shape}")

nmf = NMF(n_components=50, random_state=42, max_iter=1000)
X_latent = normalize(nmf.fit_transform(matrix))

scores = {}
for k in range(3, 25):
    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    labels = km.fit_predict(X_latent)
    s = silhouette_score(X_latent, labels, metric='euclidean')
    scores[k] = s
    print(f"K={k}: {s:.4f}")

best_k = max(scores, key=scores.get)
print(f"\nOptimal K={best_k}, score={scores[best_k]:.4f}")
