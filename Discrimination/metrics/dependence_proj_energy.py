"""
Dependence-projected Energy Score (projection-enhanced).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .energy_score import energy_score_vectorized
from .crps_score import crps_score_vectorized
from .optimal_proj_loss import optimal_projection_vector


def _validate_partitions(partitions: Sequence[Sequence[int]], d: int) -> List[List[int]]:
    if partitions is None or len(partitions) == 0:
        raise ValueError("Partitions must be a non-empty sequence of index lists")
    validated: List[List[int]] = []
    for part in partitions:
        idxs = list(map(int, part))
        if len(idxs) == 0:
            continue
        for idx in idxs:
            if idx < 0 or idx >= d:
                raise ValueError(f"Partition index {idx} is out of bounds for dimension {d}")
        validated.append(sorted(idxs))
    return validated


def dependence_proj_energy(
    forecast_samples: np.ndarray,
    observed_samples: np.ndarray,
    partitions: Sequence[Sequence[int]],
    lambda_weight: float = 1.0,
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
    parts = _validate_partitions(partitions, d)

    within_total = 0.0
    partition_thetas: List[np.ndarray] = []
    for idxs in parts:
        sub_forecast = forecast_samples[:, idxs]
        sub_observed = observed_samples[:, idxs]

        theta_p = optimal_projection_vector(sub_forecast, sub_observed)
        partition_thetas.append(theta_p)

        proj_forecast = sub_forecast @ theta_p
        proj_observed = sub_observed @ theta_p
        within_total += float(crps_score_vectorized(proj_forecast, proj_observed))

    agg_forecast = np.stack(
        [forecast_samples[:, idxs] @ theta for idxs, theta in zip(parts, partition_thetas)],
        axis=1,
    )
    agg_observed = np.stack(
        [observed_samples[:, idxs] @ theta for idxs, theta in zip(parts, partition_thetas)],
        axis=1,
    )
    cross_term = float(energy_score_vectorized(agg_forecast, agg_observed))

    return float(within_total + float(lambda_weight) * cross_term)



