"""
Evaluation metrics for multivariate probabilistic forecasts.
"""

from .energy_score import energy_score, energy_score_vectorized
from .crps_score import crps_score, crps_score_vectorized, crps_component_wise_vectorized
from .variogram_score import (
    variogram_score,
    variogram_score_vectorized,
    variogram_score_with_spatial_weights
)
from .mse import mse, mse_per_observation, mse_vectorized
from .mae import mae, mae_per_observation, mae_vectorized, mae_component_wise
from .almost_fair_crps import almost_fair_crps
from .fixed_projected_es import fixed_projected_es, fixed_projected_es_component_wise
from .dependence_proj_energy import dependence_proj_energy
from .optimal_proj_loss import optimal_proj_loss
from .optimal_proj_loss import optimal_projection_vector
from .mean_separated_proj_loss import mean_separated_proj_loss, mean_separated_projection_vector
from .multivariate_proj_loss import multivariate_proj_loss, generate_projection_matrices

__all__ = [
    'energy_score',
    'energy_score_vectorized',
    'crps_score',
    'crps_score_vectorized',
    'crps_component_wise_vectorized',
    'variogram_score',
    'variogram_score_vectorized',
    'variogram_score_with_spatial_weights',
    'mse',
    'mse_per_observation',
    'mse_vectorized',
    'mae',
    'mae_per_observation',
    'mae_vectorized',
    'mae_component_wise',
    'almost_fair_crps',
    'fixed_projected_es',
    'fixed_projected_es_component_wise',
    'dependence_proj_energy',
    'optimal_proj_loss',
    'optimal_projection_vector',
    'mean_separated_proj_loss',
    'mean_separated_projection_vector',
    'multivariate_proj_loss',
    'generate_projection_matrices',
]
