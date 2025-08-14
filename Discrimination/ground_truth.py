"""
Utility to compute ground-truth metrics: metrics evaluated between two
independently generated samples from the same (independent) distribution.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import numpy as np

from metrics import (
    energy_score_vectorized,
    crps_score_vectorized,
    variogram_score_vectorized,
    mse_vectorized,
    mae_vectorized,
    almost_fair_crps,
    fixed_projected_es,
    dependence_proj_energy,
    optimal_proj_loss,
    mean_separated_proj_loss,
    multivariate_proj_loss,
)


def compute_ground_truth_metrics(
    generator,
    n_observed_samples: int,
    d: int,
    metric_cfg: Dict[str, Any],
    perturb_details: Dict[str, Any] | None = None,
    projection_matrices: Optional[List[np.ndarray]] = None,
) -> Dict[str, float]:
    A = generator.generate_independent(n_observed_samples, d)
    B = generator.generate_independent(n_observed_samples, d)
    
    if generator.base_mean is not None:
        A = A - generator.base_mean
        B = B - generator.base_mean

    names = (metric_cfg.get('metrics', ['energy', 'crps', 'variogram', 'mse', 'mae']))
    names = [n.lower() for n in names]
    params = metric_cfg.get('metric_params', {}) or {}

    results: Dict[str, float] = {}
    for name in names:
        if name == 'energy':
            e_params = (params.get('energy', {}) or {})
            normalise = bool(e_params.get('normalise_by_sqrt_dim', False))
            results['energy'] = float(
                energy_score_vectorized(A, B, normalise_by_sqrt_dim=normalise)
            )
        elif name == 'crps':
            c_params = (params.get('crps', {}) or {})
            normalise = bool(c_params.get('normalise_by_dim', False))
            results['crps'] = float(
                crps_score_vectorized(A, B, normalise_by_dim=normalise)
            )
        elif name in ('almost_fair_crps', 'almost_fair', 'af_crps'):
            af_params = (params.get('almost_fair_crps', {}) or {})
            w = float(af_params.get('unbiased_weight', 0.95))
            normalise = bool(af_params.get('normalise_by_dim', False))
            results['almost_fair_crps'] = float(
                almost_fair_crps(A, B, unbiased_weight=w, normalise_by_dim=normalise)
            )
        elif name == 'variogram':
            p = float((params.get('variogram', {}) or {}).get('p', 1.0))
            results['variogram'] = float(variogram_score_vectorized(A, B, p=p))
        elif name == 'mse':
            results['mse'] = float(mse_vectorized(A, B))
        elif name == 'mae':
            results['mae'] = float(mae_vectorized(A, B))
        elif name in ('fixed_projected_es', 'projected_es', 'fixed_projection'):
            fp_params = (params.get('fixed_projected_es', {}) or {})
            seed = fp_params.get('seed', None)
            results['fixed_projected_es'] = float(fixed_projected_es(A, B, seed=seed))
        elif name in ('optimal_proj_loss', 'optimal_projection', 'opt_proj_crps'):
            op_params = (params.get('optimal_proj_loss', {}) or {})
            eta = float(op_params.get('eta', 0.1))
            results['optimal_proj_loss'] = float(optimal_proj_loss(A, B, eta=eta))
        elif name in ('dependence_proj_energy', 'dpe', 'dep_proj_es'):
            if perturb_details is not None and 'partitions' in perturb_details:
                dpe_params = (params.get('dependence_proj_energy', {}) or {})
                lambda_weight = float(dpe_params.get('lambda', 1.0))
                partitions = perturb_details['partitions']
                results['dependence_proj_energy'] = float(
                    dependence_proj_energy(A, B, partitions=partitions, lambda_weight=lambda_weight)
                )
            else:
                continue
        elif name in ('mean_separated_proj_loss', 'mean_sep_proj', 'mean_proj_crps'):
            ms_params = (params.get('mean_separated_proj_loss', {}) or {})
            normalise = bool(ms_params.get('normalise', True))
            results['mean_separated_proj_loss'] = float(
                mean_separated_proj_loss(A, B, normalise=normalise)
            )
        elif name in ('multivariate_proj_loss', 'mv_proj_loss'):
            mv_params = (params.get('multivariate_proj_loss', {}) or {})
            normalise = bool(mv_params.get('normalise_by_dim', False))
            results['multivariate_proj_loss'] = float(
                multivariate_proj_loss(A, B, projections=projection_matrices, normalise_by_dim=normalise)
            )
        else:
            raise ValueError(f"Unknown metric: {name}")
    return results


