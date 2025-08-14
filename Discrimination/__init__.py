"""
Discrimination experiments for multivariate probabilistic forecasting.

This module contains:
- metrics/: Evaluation metrics (Energy Score, CRPS, Variogram Score, MSE, MAE)
- data/: Data generation with configurable dependence structures
"""

from . import metrics
from . import data

__all__ = ['metrics', 'data']
