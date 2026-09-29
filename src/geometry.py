"""Validate distance settings and describe evaluation geometry (not student distances)."""
import numpy as np


def diagonal_weights(cfg, n_features):
    distance = cfg['kmeans'].get('distance', 'euclidean')
    if distance == 'euclidean':
        return np.ones(n_features)
    if distance != 'mahalanobis':
        raise ValueError(f'Unknown distance: {distance!r}')
    values = cfg['kmeans'].get('mahalanobis_diag')
    weights = np.ones(n_features) if values is None else np.asarray(values, dtype=float)
    if weights.shape != (n_features,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError(f'mahalanobis_diag must contain {n_features} finite positive weights')
    return weights
