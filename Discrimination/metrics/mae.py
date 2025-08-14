"""
MAE (Mean Absolute Error) metric implementation.
"""

import numpy as np
from typing import Union


def mae(forecast_samples: np.ndarray, observed_samples: np.ndarray) -> float:
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
    
    absolute_error = np.sum(np.abs(forecast_mean - observed_mean))
    
    return absolute_error


def mae_per_observation(forecast_samples: np.ndarray, observed_samples: np.ndarray) -> float:
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
    total_ae = 0.0
    
    for i in range(n_observed):
        absolute_error = np.sum(np.abs(forecast_mean - observed_samples[i]))
        total_ae += absolute_error
    
    return total_ae / n_observed


def mae_vectorized(forecast_samples: np.ndarray, observed_samples: np.ndarray) -> float:
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
    
    absolute_errors = np.abs(observed_samples - forecast_mean[None, :])
    
    return np.mean(np.sum(absolute_errors, axis=1))


def mae_component_wise(forecast_samples: np.ndarray, observed_samples: np.ndarray) -> np.ndarray:
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
    
    absolute_errors = np.abs(observed_samples - forecast_mean[None, :])
    
    return np.mean(absolute_errors, axis=0)
