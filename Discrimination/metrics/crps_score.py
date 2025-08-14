"""
CRPS (Continuous Ranked Probability Score) metric implementation.
"""

import numpy as np
from typing import Union


def crps_score(
    forecast_samples: np.ndarray,
    observed_samples: np.ndarray,
    normalise_by_dim: bool = False,
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
    
    total_crps = 0.0
    
    for dim in range(d):
        forecast_dim = forecast_samples[:, dim]
        observed_dim = observed_samples[:, dim]
        
        term1 = 0.0
        for i in range(n_forecast):
            for j in range(n_observed):
                term1 += np.abs(forecast_dim[i] - observed_dim[j])
        term1 /= (n_forecast * n_observed)
        
        term2 = 0.0
        for i in range(n_forecast):
            for j in range(n_forecast):
                if i != j:
                    term2 += np.abs(forecast_dim[i] - forecast_dim[j])
        if n_forecast > 1:
            term2 /= (n_forecast * (n_forecast - 1))
        term2 *= 0.5
        
        total_crps += term1 - term2
    
    if normalise_by_dim and d > 0:
        return float(total_crps) / float(d)

    return float(total_crps)


def crps_score_vectorized(
    forecast_samples: np.ndarray,
    observed_samples: np.ndarray,
    normalise_by_dim: bool = False,
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
    
    total_crps = 0.0
    
    for dim in range(d):
        forecast_dim = forecast_samples[:, dim]
        observed_dim = observed_samples[:, dim]
        
        diff_fo = np.abs(forecast_dim[:, None] - observed_dim[None, :])
        term1 = np.mean(diff_fo)
        
        diff_ff = np.abs(forecast_dim[:, None] - forecast_dim[None, :])
        
        if n_forecast > 1:
            term2 = 0.5 * (np.sum(diff_ff) - np.sum(np.diag(diff_ff))) / (n_forecast * (n_forecast - 1))
        else:
            term2 = 0.0
        
        total_crps += term1 - term2
    
    if normalise_by_dim and d > 0:
        return float(total_crps) / float(d)

    return float(total_crps)


def crps_component_wise_vectorized(
    forecast_samples: np.ndarray,
    observed_samples: np.ndarray,
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
            f"observed has {observed_samples.shape[1]} dimensions"
        )

    n_forecast = forecast_samples.shape[0]
    d = forecast_samples.shape[1]

    per_dim_crps = np.zeros(d, dtype=float)

    for dim in range(d):
        forecast_dim = forecast_samples[:, dim]
        observed_dim = observed_samples[:, dim]

        diff_fo = np.abs(forecast_dim[:, None] - observed_dim[None, :])
        term1 = float(np.mean(diff_fo))

        diff_ff = np.abs(forecast_dim[:, None] - forecast_dim[None, :])
        if n_forecast > 1:
            term2 = 0.5 * (float(np.sum(diff_ff)) - float(np.sum(np.diag(diff_ff)))) / (n_forecast * (n_forecast - 1))
        else:
            term2 = 0.0

        per_dim_crps[dim] = term1 - term2

    return per_dim_crps
