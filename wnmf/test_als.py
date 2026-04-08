# test_als.py
from wnmf_model import WNMFModel
import numpy as np

# Küçük test
train = np.array([
    [0, 0, 4.0], [0, 1, 3.0], [1, 0, 5.0], [1, 2, 2.0],
    [2, 1, 4.0], [2, 2, 3.0], [3, 0, 2.0], [3, 1, 5.0],
], dtype=np.float32)

test = np.array([[0, 2, 3.0], [1, 1, 4.0]], dtype=np.float32)

# ALS testi
model = WNMFModel(
    n_users=4, n_items=3, latent_dim=5,
    learning_rate=0.01, regularization=0.01,
    n_epochs=10, random_seed=42,
    use_bias=True, solver='als'
)
print("solver:", model.solver)
model.fit(train, verbose=True)
mae, rmse = model.evaluate(test)
print(f"ALS MAE={mae:.4f} RMSE={rmse:.4f}")