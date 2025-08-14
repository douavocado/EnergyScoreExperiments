"""
Data generation module for multivariate experiments.

Provides functionality to generate:
1. Independent multivariate samples (i.i.d. across dimensions)
2. Dependent multivariate samples using:
   - Common-shock method (equi-correlation)
   - Gaussian copula method (flexible dependence)
"""

from .data_generator import (
    MultivariateDataGenerator,
    generate_ar1_correlation_matrix,
    generate_block_correlation_matrix,
)

__all__ = [
    'MultivariateDataGenerator',
    'generate_ar1_correlation_matrix',
    'generate_block_correlation_matrix',
]
