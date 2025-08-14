"""
Almost-fair CRPS metric.
"""

from typing import Optional

import numpy as np


def _crps_terms_1d(
    forecast_dim: np.ndarray,
    observed_dim: np.ndarray,
) -> tuple[float, float, float]:
    n_forecast = forecast_dim.shape[0]

    diff_fo = np.abs(forecast_dim[:, None] - observed_dim[None, :])
    term1 = float(np.mean(diff_fo))

    diff_ff = np.abs(forecast_dim[:, None] - forecast_dim[None, :])

    if n_forecast > 1:
        sum_offdiag = float(np.sum(diff_ff) - np.sum(np.diag(diff_ff)))
        term2_unbiased = 0.5 * (sum_offdiag / (n_forecast * (n_forecast - 1)))
    else:
        term2_unbiased = 0.0

    term2_biased = 0.5 * float(np.mean(diff_ff))

    return term1, term2_unbiased, term2_biased


def almost_fair_crps(
    forecast_samples: np.ndarray,
    observed_samples: np.ndarray,
    unbiased_weight: float = 0.95,
    normalise_by_dim: bool = False,
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
    unbiased_weight = float(unbiased_weight)
    if not (0.0 <= unbiased_weight <= 1.0):
        raise ValueError("unbiased_weight must be in [0, 1]")

    total = 0.0
    for dim in range(d):
        forecast_dim = forecast_samples[:, dim]
        observed_dim = observed_samples[:, dim]
        term1, term2_unbiased, term2_biased = _crps_terms_1d(forecast_dim, observed_dim)

        crps_unbiased = term1 - term2_unbiased
        crps_biased = term1 - term2_biased
        total += unbiased_weight * crps_unbiased + (1.0 - unbiased_weight) * crps_biased

    if normalise_by_dim and d > 0:
        return float(total) / float(d)

    return float(total)


