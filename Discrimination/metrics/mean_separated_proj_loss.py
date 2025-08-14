from __future__ import annotations

"""
Mean-separated projection CRPS loss.
"""

from typing import Optional

import numpy as np

from .crps_score import crps_score_vectorized


def mean_separated_projection_vector(
    forecast_samples: np.ndarray,
    observed_samples: np.ndarray,
    normalise: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    X = np.asarray(forecast_samples)
    Y = np.asarray(observed_samples)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Inputs must be 2D arrays of shape (n_samples, d)")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Dimension mismatch: forecast has {X.shape[1]} dims, observed has {Y.shape[1]} dims"
        )

    mu_X = np.mean(X, axis=0)
    mu_Y = np.mean(Y, axis=0)
    theta = mu_X - mu_Y

    if not normalise:
        return theta

    norm_theta = float(np.linalg.norm(theta))
    if norm_theta <= eps:
        d = X.shape[1]
        return np.ones((d,), dtype=float) / np.sqrt(float(d))

    return theta / norm_theta


def mean_separated_proj_loss(
    forecast_samples: np.ndarray,
    observed_samples: np.ndarray,
    normalise: bool = True,
) -> float:
    X = np.asarray(forecast_samples)
    Y = np.asarray(observed_samples)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Inputs must be 2D arrays of shape (n_samples, d)")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Dimension mismatch: forecast has {X.shape[1]} dims, observed has {Y.shape[1]} dims"
        )

    theta = mean_separated_projection_vector(X, Y, normalise=normalise)

    proj_forecast = X @ theta
    proj_observed = Y @ theta

    return float(crps_score_vectorized(proj_forecast, proj_observed))
