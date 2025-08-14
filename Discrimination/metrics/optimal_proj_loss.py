"""
Optimal projection CRPS loss.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .crps_score import crps_score_vectorized


def _compute_sample_covariance(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError("samples must be a 2D array of shape (n_samples, d)")
    n = samples.shape[0]
    if n <= 0:
        raise ValueError("samples must contain at least one row")
    return (samples.T @ samples) / float(n)


def _riemannian_gradient_ascent_direction(
    forecast_samples: np.ndarray,
    observed_samples: np.ndarray,
    eta: float,
    n_iterations: Optional[int] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    X = np.asarray(forecast_samples)
    Y = np.asarray(observed_samples)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Inputs must be 2D arrays of shape (n_samples, d)")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Dimension mismatch: forecast has {X.shape[1]} dims, observed has {Y.shape[1]} dims"
        )

    n_forecast = X.shape[0]
    d = X.shape[1]

    if n_iterations is None:
        iters = int(np.ceil(np.log(max(n_forecast, 2))))
        n_iterations = max(1, iters)

    Sigma_X = _compute_sample_covariance(X)
    Sigma_Y = _compute_sample_covariance(Y)

    theta = np.random.normal(size=(d,)).astype(float)
    norm_theta = np.linalg.norm(theta)
    if norm_theta == 0.0:
        theta = np.ones((d,), dtype=float) / np.sqrt(float(d))
    else:
        theta = theta / norm_theta

    for _ in range(int(n_iterations)):
        E_X = Sigma_X @ theta
        E_Y = Sigma_Y @ theta

        S_X = float(theta @ E_X)
        S_Y = float(theta @ E_Y)

        Sigma_theta = 0.5 * (E_X + E_Y)
        Delta_theta = 0.5 * (E_Y - E_X)

        b = 0.5 * (S_X + S_Y)
        a = 0.5 * (S_Y - S_X)

        b_safe = max(b, eps)

        f = (a * a) / (b_safe ** 1.5)
        grad_euclid = (4.0 * a * b * Delta_theta - 3.0 * (a * a) * Sigma_theta) / (b_safe ** 2.5)

        g = grad_euclid - f * theta

        theta = theta + float(eta) * g
        norm_theta = np.linalg.norm(theta)
        if norm_theta == 0.0:
            theta = np.ones((d,), dtype=float) / np.sqrt(float(d))
        else:
            theta = theta / norm_theta

    return theta


def optimal_proj_loss(
    forecast_samples: np.ndarray,
    observed_samples: np.ndarray,
    eta: float = 0.1,
    n_iterations: Optional[int] = None,
) -> float:
    X = np.asarray(forecast_samples)
    Y = np.asarray(observed_samples)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Inputs must be 2D arrays of shape (n_samples, d)")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Dimension mismatch: forecast has {X.shape[1]} dimensions, observed has {Y.shape[1]} dimensions"
        )

    theta = _riemannian_gradient_ascent_direction(X, Y, eta=eta, n_iterations=n_iterations)

    proj_forecast = X @ theta
    proj_observed = Y @ theta

    return float(crps_score_vectorized(proj_forecast, proj_observed))


def optimal_projection_vector(
    forecast_samples: np.ndarray,
    observed_samples: np.ndarray,
    eta: float = 0.1,
    n_iterations: Optional[int] = None,
) -> np.ndarray:
    X = np.asarray(forecast_samples)
    Y = np.asarray(observed_samples)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Inputs must be 2D arrays of shape (n_samples, d)")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Dimension mismatch: forecast has {X.shape[1]} dimensions, observed has {Y.shape[1]} dimensions"
        )
    theta = _riemannian_gradient_ascent_direction(X, Y, eta=eta, n_iterations=n_iterations)
    return theta


__all__ = ["optimal_proj_loss"]


