"""
Fixed projected Energy Score (ES).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


_cached_bases: Dict[int, np.ndarray] = {}


def _get_fixed_orthonormal_basis(d: int, seed: Optional[int] = None) -> np.ndarray:
    global _cached_bases
    if d in _cached_bases:
        return _cached_bases[d]

    rng = np.random.default_rng(seed)
    random_matrix = rng.normal(size=(d, d))
    q, r = np.linalg.qr(random_matrix)

    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q = q * signs

    _cached_bases[d] = q
    return q


def _univariate_energy_score(forecast_1d: np.ndarray, observed_1d: np.ndarray) -> float:
    forecast_1d = forecast_1d.reshape(-1)
    observed_1d = observed_1d.reshape(-1)

    n_f = forecast_1d.shape[0]

    diff_fo = np.abs(forecast_1d[:, None] - observed_1d[None, :])
    term1 = float(np.mean(diff_fo))

    diff_ff = np.abs(forecast_1d[:, None] - forecast_1d[None, :])
    if n_f > 1:
        term2 = 0.5 * (np.sum(diff_ff) - np.sum(np.diag(diff_ff))) / (n_f * (n_f - 1))
    else:
        term2 = 0.0

    return float(term1 - term2)


def fixed_projected_es(
    forecast_samples: np.ndarray,
    observed_samples: np.ndarray,
    seed: Optional[int] = None,
) -> float:
    forecast_samples = np.asarray(forecast_samples)
    observed_samples = np.asarray(observed_samples)

    if forecast_samples.ndim == 1:
        forecast_samples = forecast_samples.reshape(-1, 1)
    if observed_samples.ndim == 1:
        observed_samples = observed_samples.reshape(-1, 1)

    if forecast_samples.shape[1] != observed_samples.shape[1]:
        raise ValueError(
            f"Dimension mismatch: forecast has {forecast_samples.shape[1]} dimensions, "
            f"observed has {observed_samples.shape[1]} dimensions",
        )

    d = forecast_samples.shape[1]
    if d <= 0:
        raise ValueError("Input dimension must be positive")

    basis = _get_fixed_orthonormal_basis(d, seed=seed)

    proj_forecast = forecast_samples @ basis
    proj_observed = observed_samples @ basis

    scores = [
        _univariate_energy_score(proj_forecast[:, i], proj_observed[:, i])
        for i in range(d)
    ]
    return float(np.max(scores))


def fixed_projected_es_component_wise(
    forecast_samples: np.ndarray,
    observed_samples: np.ndarray,
    seed: Optional[int] = None,
) -> np.ndarray:
    forecast_samples = np.asarray(forecast_samples)
    observed_samples = np.asarray(observed_samples)

    if forecast_samples.ndim == 1:
        forecast_samples = forecast_samples.reshape(-1, 1)
    if observed_samples.ndim == 1:
        observed_samples = observed_samples.reshape(-1, 1)

    if forecast_samples.shape[1] != observed_samples.shape[1]:
        raise ValueError(
            f"Dimension mismatch: forecast has {forecast_samples.shape[1]} dimensions, "
            f"observed has {observed_samples.shape[1]} dimensions",
        )

    d = forecast_samples.shape[1]
    if d <= 0:
        raise ValueError("Input dimension must be positive")

    basis = _get_fixed_orthonormal_basis(d, seed=seed)
    proj_forecast = forecast_samples @ basis
    proj_observed = observed_samples @ basis

    scores = np.array([
        _univariate_energy_score(proj_forecast[:, i], proj_observed[:, i])
        for i in range(d)
    ], dtype=float)
    return scores


