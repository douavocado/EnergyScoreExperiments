"""
Multivariate projected Energy Score loss.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .energy_score import energy_score_vectorized


def _generate_row_orthonormal_matrix(num_rows: int, d: int, rng: np.random.Generator) -> np.ndarray:
    if num_rows <= 0 or d <= 0:
        raise ValueError("num_rows and d must be positive integers")
    if num_rows > d:
        raise ValueError("num_rows cannot exceed d when generating orthonormal rows")

    random_matrix = rng.normal(size=(d, num_rows))
    q, r = np.linalg.qr(random_matrix, mode='reduced')

    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q = q * signs

    return q.T.astype(float)


def generate_projection_matrices(d: int, seed: Optional[int] = None) -> List[np.ndarray]:
    if d <= 0:
        raise ValueError("d must be a positive integer")

    rng = np.random.default_rng(seed)

    projections: List[np.ndarray] = []
    for i in range(1, d + 1):
        if i == d:
            A_i = np.eye(d, dtype=float)
        else:
            A_i = _generate_row_orthonormal_matrix(i, d, rng)
        projections.append(A_i)
    return projections


def multivariate_proj_loss(
    forecast_samples: np.ndarray,
    observed_samples: np.ndarray,
    projections: Optional[Sequence[np.ndarray]] = None,
    seed: Optional[int] = None,
    normalise_by_dim: bool = False,
) -> float:
    X = np.asarray(forecast_samples)
    Y = np.asarray(observed_samples)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Dimension mismatch: forecast has {X.shape[1]} dimensions, observed has {Y.shape[1]} dimensions"
        )

    d = X.shape[1]
    A_list: Sequence[np.ndarray]
    if projections is None:
        A_list = generate_projection_matrices(d, seed=seed)
    else:
        A_list = projections
        if len(A_list) != d:
            raise ValueError(f"projections must contain exactly d={d} matrices, got {len(A_list)}")

    normalized_scores = []
    for i, A_i in enumerate(A_list, start=1):
        if A_i.shape != (i, d):
            raise ValueError(f"A_{i} must have shape ({i}, {d}), got {A_i.shape}")
        proj_X = X @ A_i.T
        proj_Y = Y @ A_i.T
        es_i = float(energy_score_vectorized(proj_X, proj_Y))
        normalized_es_i = es_i / float(np.sqrt(i))
        normalized_scores.append(normalized_es_i)

    return float(np.max(normalized_scores))


__all__ = [
    "generate_projection_matrices",
    "multivariate_proj_loss",
]


