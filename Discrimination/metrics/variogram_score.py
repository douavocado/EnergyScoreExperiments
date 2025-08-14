"""
Variogram Score metric implementation.
"""

import numpy as np
from typing import Union, Optional


def variogram_score(
    forecast_samples: np.ndarray,
    observed_samples: np.ndarray,
    p: float = 1.0,
    weights: Optional[np.ndarray] = None
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
    
    if weights is None:
        weights = np.ones((d, d)) / (d * (d - 1))
        np.fill_diagonal(weights, 0)
    else:
        weights = np.asarray(weights)
        if weights.shape != (d, d):
            raise ValueError(f"Weights must have shape ({d}, {d}), got {weights.shape}")
    
    expected_diff = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i != j:
                diffs = np.abs(forecast_samples[:, i] - forecast_samples[:, j]) ** p
                expected_diff[i, j] = np.mean(diffs)
    
    total_score = 0.0
    for obs_idx in range(n_observed):
        obs = observed_samples[obs_idx]
        
        observed_diff = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if i != j:
                    observed_diff[i, j] = np.abs(obs[i] - obs[j]) ** p
        
        score = 0.0
        for i in range(d):
            for j in range(d):
                if i != j:
                    diff = observed_diff[i, j] - expected_diff[i, j]
                    score += weights[i, j] * (diff ** 2)
        
        total_score += score
    
    return total_score / n_observed


def variogram_score_vectorized(
    forecast_samples: np.ndarray,
    observed_samples: np.ndarray,
    p: float = 1.0,
    weights: Optional[np.ndarray] = None
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
    
    if weights is None:
        weights = np.ones((d, d)) / (d * (d - 1))
        np.fill_diagonal(weights, 0)
    else:
        weights = np.asarray(weights)
        if weights.shape != (d, d):
            raise ValueError(f"Weights must have shape ({d}, {d}), got {weights.shape}")
    
    forecast_expanded_i = forecast_samples[:, :, None]
    forecast_expanded_j = forecast_samples[:, None, :]
    forecast_diffs = np.abs(forecast_expanded_i - forecast_expanded_j) ** p
    expected_diff = np.mean(forecast_diffs, axis=0)
    
    total_score = 0.0
    for obs_idx in range(n_observed):
        obs = observed_samples[obs_idx]
        
        obs_i = obs[:, None]
        obs_j = obs[None, :]
        observed_diff = np.abs(obs_i - obs_j) ** p
        
        diff_squared = (observed_diff - expected_diff) ** 2
        score = np.sum(weights * diff_squared)
        
        total_score += score
    
    return total_score / n_observed


def variogram_score_with_spatial_weights(
    forecast_samples: np.ndarray,
    observed_samples: np.ndarray,
    coordinates: np.ndarray,
    p: float = 1.0,
    weight_power: float = 1.0
) -> float:
    coordinates = np.asarray(coordinates)
    d = forecast_samples.shape[1]
    
    if coordinates.shape[0] != d:
        raise ValueError(f"Number of coordinates ({coordinates.shape[0]}) must match "
                        f"number of dimensions ({d})")
    
    weights = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i != j:
                distance = np.linalg.norm(coordinates[i] - coordinates[j])
                if distance > 0:
                    weights[i, j] = 1.0 / (distance ** weight_power)
    
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights /= weight_sum
    
    return variogram_score_vectorized(forecast_samples, observed_samples, p, weights)
