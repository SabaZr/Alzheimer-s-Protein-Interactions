import numpy as np  # Needed for numerical operations
# -------------------------------
# Hyperdimensional Computing Function
# -------------------------------
def hdc_encode_features(X, hd_dim=4096):
    n_samples, n_features = X.shape
    feature_hypervectors = np.random.choice([-1, 1], size=(n_features, hd_dim)).astype(np.float32)
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    normalized_X = (X - min_vals) / (max_vals - min_vals + 1e-8)
    encoded_samples = np.sign(normalized_X @ feature_hypervectors).astype(np.float32)
    return encoded_samples, feature_hypervectors