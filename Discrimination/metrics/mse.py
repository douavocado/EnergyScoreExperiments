"""
MSE (Mean Squared Error) metric implementation.
"""

import numpy as np
from typing import Union


def mse(forecast_samples: np.ndarray, observed_samples: np.ndarray) -> float:
    forecast_samples = np.asarray(forecast_samples)
    observed_samples = np.asarray(observed_samples)
    
    if forecast_samples.ndim == 1:
        forecast_samples = forecast_samples.reshape(-1, 1)
    if observed_samples.ndim == 1:
        observed_samples = observed_samples.reshape(-1, 1)
        
    if forecast_samples.shape[1] != observed_samples.shape[1]:
        raise ValueError(f"Dimension mismatch: forecast has {forecast_samples.shape[1]} dimensions, "
                        f"observed has {observed_samples.shape[1]} dimensions")
    
    forecast_mean = np.mean(forecast_samples, axis=0)
    observed_mean = np.mean(observed_samples, axis=0)
    
    squared_error = np.sum((forecast_mean - observed_mean) ** 2)
    
    return squared_error


def mse_per_observation(forecast_samples: np.ndarray, observed_samples: np.ndarray) -> float:
    forecast_samples = np.asarray(forecast_samples)
    observed_samples = np.asarray(observed_samples)
    
    if forecast_samples.ndim == 1:
        forecast_samples = forecast_samples.reshape(-1, 1)
    if observed_samples.ndim == 1:
        observed_samples = observed_samples.reshape(-1, 1)
        
    if forecast_samples.shape[1] != observed_samples.shape[1]:
        raise ValueError(f"Dimension mismatch: forecast has {forecast_samples.shape[1]} dimensions, "
                        f"observed has {observed_samples.shape[1]} dimensions")
    
    forecast_mean = np.mean(forecast_samples, axis=0)
    
    n_observed = observed_samples.shape[0]
    total_se = 0.0
    
    for i in range(n_observed):
        squared_error = np.sum((forecast_mean - observed_samples[i]) ** 2)
        total_se += squared_error
    
    return total_se / n_observed


def mse_vectorized(forecast_samples: np.ndarray, observed_samples: np.ndarray) -> float:
    forecast_samples = np.asarray(forecast_samples)
    observed_samples = np.asarray(observed_samples)
    
    if forecast_samples.ndim == 1:
        forecast_samples = forecast_samples.reshape(-1, 1)
    if observed_samples.ndim == 1:
        observed_samples = observed_samples.reshape(-1, 1)
        
    if forecast_samples.shape[1] != observed_samples.shape[1]:
        raise ValueError(f"Dimension mismatch: forecast has {forecast_samples.shape[1]} dimensions, "
                        f"observed has {observed_samples.shape[1]} dimensions")
    
    forecast_mean = np.mean(forecast_samples, axis=0)
    
    squared_errors = (observed_samples - forecast_mean[None, :]) ** 2
    
    return np.mean(np.sum(squared_errors, axis=1))
