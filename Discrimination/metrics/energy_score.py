"""
Energy Score metric implementation.
"""

import numpy as np
from typing import Union


def energy_score(
    forecast_samples: np.ndarray,
    observed_samples: np.ndarray,
    normalise_by_sqrt_dim: bool = False,
) -> float:
    forecast_samples = np.asarray(forecast_samples)
    observed_samples = np.asarray(observed_samples)
    
    if forecast_samples.ndim == 1:
        forecast_samples = forecast_samples.reshape(-1, 1)
    if observed_samples.ndim == 1:
        observed_samples = observed_samples.reshape(-1, 1)
        
    if forecast_samples.shape[1] != observed_samples.shape[1]:
        raise ValueError(f"Dimension mismatch: forecast has {forecast_samples.shape[1]} dimensions, "
                        f"observed has {observed_samples.shape[1]} dimensions")
    
    n_forecast = forecast_samples.shape[0]
    n_observed = observed_samples.shape[0]
    d = forecast_samples.shape[1]
    
    term1 = 0.0
    for i in range(n_forecast):
        for j in range(n_observed):
            term1 += np.linalg.norm(forecast_samples[i] - observed_samples[j])
    term1 /= (n_forecast * n_observed)
    
    term2 = 0.0
    for i in range(n_forecast):
        for j in range(n_forecast):
            if i != j:
                term2 += np.linalg.norm(forecast_samples[i] - forecast_samples[j])
    if n_forecast > 1:
        term2 /= (n_forecast * (n_forecast - 1))
    term2 *= 0.5
    
    value = term1 - term2

    if normalise_by_sqrt_dim and d > 0:
        return float(value) / float(np.sqrt(d))

    return float(value)


def energy_score_vectorized(
    forecast_samples: np.ndarray,
    observed_samples: np.ndarray,
    normalise_by_sqrt_dim: bool = False,
) -> float:
    forecast_samples = np.asarray(forecast_samples)
    observed_samples = np.asarray(observed_samples)
    
    if forecast_samples.ndim == 1:
        forecast_samples = forecast_samples.reshape(-1, 1)
    if observed_samples.ndim == 1:
        observed_samples = observed_samples.reshape(-1, 1)
        
    if forecast_samples.shape[1] != observed_samples.shape[1]:
        raise ValueError(f"Dimension mismatch: forecast has {forecast_samples.shape[1]} dimensions, "
                        f"observed has {observed_samples.shape[1]} dimensions")
    
    n_forecast = forecast_samples.shape[0]
    d = forecast_samples.shape[1]
    
    diff_fo = forecast_samples[:, None, :] - observed_samples[None, :, :]
    distances_fo = np.linalg.norm(diff_fo, axis=2)
    term1 = np.mean(distances_fo)
    
    diff_ff = forecast_samples[:, None, :] - forecast_samples[None, :, :]
    distances_ff = np.linalg.norm(diff_ff, axis=2)
    
    if n_forecast > 1:
        term2 = 0.5 * (np.sum(distances_ff) - np.trace(distances_ff)) / (n_forecast * (n_forecast - 1))
    else:
        term2 = 0.0
    
    value = term1 - term2

    if normalise_by_sqrt_dim and d > 0:
        return float(value) / float(np.sqrt(d))

    return float(value)
