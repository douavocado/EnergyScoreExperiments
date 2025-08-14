"""
Main entry point for running Discrimination experiments from a YAML config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Tuple, List

import numpy as np
import yaml

from data import MultivariateDataGenerator, generate_ar1_correlation_matrix
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
    generate_projection_matrices,
)
from data.data_generator import generate_block_correlation_matrix
from ground_truth import compute_ground_truth_metrics
from visualise import visualise_single_run
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _build_base_sampler_from_config(cfg: Dict[str, Any]):
    name = cfg.get('base_distribution', 'gaussian').lower()
    params = cfg.get('base_params', {}) or {}

    if name == 'beta':
        beta_cfg = cfg.get('beta', None)
        if not beta_cfg:
            beta_cfg = params
        if beta_cfg:
            alpha = float(beta_cfg.get('alpha', 2.0))
            beta_param = float(beta_cfg.get('beta', 2.0))
            theoretical_mean = alpha / (alpha + beta_param)
            return lambda size: np.random.beta(alpha, beta_param, size), theoretical_mean
        return name, None

    if not params:
        return name, None

    if name in ('gaussian', 'normal'):
        mean = float(params.get('mean', 0.0))
        std = float(params.get('std', 1.0))
        return lambda size: np.random.normal(mean, std, size), mean
    if name == 'student_t':
        df = float(params.get('df', 5.0))
        theoretical_mean = 0.0 if df > 1 else None
        return lambda size: np.random.standard_t(df=df, size=size), theoretical_mean
    if name == 'exponential':
        scale = float(params.get('scale', 1.0))
        theoretical_mean = scale
        return lambda size: np.random.exponential(scale=scale, size=size), theoretical_mean
    if name == 'gamma':
        shape = float(params.get('shape', 2.0))
        scale = float(params.get('scale', 1.0))
        theoretical_mean = shape * scale
        return lambda size: np.random.gamma(shape=shape, scale=scale, size=size), theoretical_mean

    return name, None


def _build_gaussian_corr_from_cfg(d: int, gcfg: Dict[str, Any]) -> np.ndarray:
    structure = (gcfg.get('structure', 'equi') or 'equi').lower()
    if structure in ('equi', 'equicorrelation', 'equi_corr'):
        rho = float(gcfg['rho'])
        min_rho = -1 / (d - 1)
        if not (min_rho < rho < 1):
            raise ValueError(f"Equi-correlation rho must be in ({min_rho:.3f}, 1)")
        return (1 - rho) * np.eye(d) + rho * np.ones((d, d))
    if structure in ('ar1', 'ar'): 
        rho = float(gcfg['rho'])
        return generate_ar1_correlation_matrix(d, rho)
    if structure in ('full', 'matrix'):
        mat = np.asarray(gcfg['matrix'], dtype=float)
        if mat.shape != (d, d):
            raise ValueError(f"Provided full correlation matrix must be ({d},{d})")
        return mat
    if structure in ('block', 'blocks'):
        block_sizes = list(map(int, gcfg['block_sizes']))
        within = float(gcfg.get('within_block_corr', 0.5))
        between = float(gcfg.get('between_block_corr', 0.0))
        return generate_block_correlation_matrix(d, block_sizes, within, between)
    raise ValueError(f"Unknown Gaussian corr structure: {structure}")


def _build_perturbed_samples(
    gen: MultivariateDataGenerator,
    cfg: Dict[str, Any],
    n_forecast: int,
    d: int
) -> Tuple[str, np.ndarray, Dict[str, Any]]:
    pert = cfg.get('perturbation', {}) or {}
    method = (pert.get('method', 'copula') or 'copula').lower()
    details: Dict[str, Any] = {'method': method}

    if method == 'common_shock':
        rho = float(pert['rho'])
        preserve = bool(pert.get('preserve_variance', True))
        Y = gen.generate_common_shock(n_forecast, d, rho=rho, preserve_variance=preserve)
        details.update({'rho': rho, 'preserve_variance': preserve})
        desc = f"common_shock(rho={rho}, preserve={preserve})"
        return desc, Y, details

    if method == 'copula':
        ctype = (pert.get('type', 'gaussian') or 'gaussian').lower()
        details['type'] = ctype
        if ctype == 'gaussian':
            gcfg = pert.get('gaussian', {}) or {}
            corr = _build_gaussian_corr_from_cfg(d, gcfg)
            Y = gen.generate_copula_general(
                n_samples=n_forecast,
                d=d,
                copula='gaussian',
                correlation_matrix=corr,
            )
            desc = f"gaussian_copula(structure={gcfg.get('structure','equi')})"
            details['gaussian'] = gcfg
            details['output_dimension'] = d
            return desc, Y, details
        if ctype == 'clayton':
            ccfg = pert.get('clayton', {}) or {}
            theta = ccfg.get('theta')
            tau = ccfg.get('tau')
            Y = gen.generate_copula_general(
                n_samples=n_forecast,
                d=d,
                copula='clayton',
                clayton_theta=theta,
                clayton_tau=tau,
            )
            desc = f"clayton_copula(theta={theta}, tau={tau})"
            details['clayton'] = ccfg
            details['output_dimension'] = d
            return desc, Y, details
        raise ValueError(f"Unsupported copula type: {ctype}")

    if method == 'linear_mixing':
        lcfg = pert.get('linear_mixing', {}) or {}
        D = int(lcfg.get('D', d))
        sparsity = float(lcfg.get('sparsity', 0.0))
        if not (0.0 <= sparsity < 1.0):
            raise ValueError("sparsity must be in [0, 1)")

        A = np.zeros((D, d), dtype=float)
        num_nonzero_per_row = max(1, int(round((1.0 - sparsity) * d)))
        for i in range(D):
            cols = np.random.choice(d, size=num_nonzero_per_row, replace=False)
            A[i, cols] = np.random.normal(size=num_nonzero_per_row)

        svals = np.linalg.svd(A, compute_uv=False)
        max_sv = float(svals[0]) if svals.size > 0 else 1.0
        if max_sv > 0:
            A = A / max_sv

        X = gen.generate_independent(n_forecast, d)
        
        if gen.base_mean is not None:
            X = X - gen.base_mean
        
        Y = X @ A.T

        row_supports = [set(np.nonzero(A[i])[0].tolist()) for i in range(D)]
        visited = [False] * D
        partitions = []
        for i in range(D):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            component = []
            while stack:
                u = stack.pop()
                component.append(u)
                supp_u = row_supports[u]
                if not supp_u:
                    continue
                for v in range(D):
                    if visited[v]:
                        continue
                    if row_supports[v] and not row_supports[v].isdisjoint(supp_u):
                        visited[v] = True
                        stack.append(v)
            partitions.append(sorted(component))

        desc = f"linear_mixing(D={D}, sparsity={sparsity:.2f})"
        details.update({
            'linear_mixing': {'D': D, 'sparsity': sparsity},
            'output_dimension': D,
            'partitions': partitions,
        })
        return desc, Y, details

    raise ValueError(f"Unknown perturbation method: {method}")


def _compute_metrics(
    forecast: np.ndarray,
    observed: np.ndarray,
    metric_cfg: Dict[str, Any],
    perturb_details: Dict[str, Any],
    projection_matrices: List[np.ndarray] | None = None,
) -> Dict[str, float]:
    names = metric_cfg.get('metrics', ['energy', 'crps', 'variogram', 'mse', 'mae'])
    names = [n.lower() for n in names]
    params = metric_cfg.get('metric_params', {}) or {}

    results: Dict[str, float] = {}
    for name in tqdm(names, desc='Computing metrics', unit='metric'):
        if name == 'energy':
            e_params = (params.get('energy', {}) or {})
            normalise = bool(e_params.get('normalise_by_sqrt_dim', False))
            results['energy'] = float(
                energy_score_vectorized(forecast, observed, normalise_by_sqrt_dim=normalise)
            )
        elif name == 'crps':
            c_params = (params.get('crps', {}) or {})
            normalise = bool(c_params.get('normalise_by_dim', False))
            results['crps'] = float(
                crps_score_vectorized(forecast, observed, normalise_by_dim=normalise)
            )
        elif name in ('almost_fair_crps', 'almost_fair', 'af_crps'):
            af_params = (params.get('almost_fair_crps', {}) or {})
            w = float(af_params.get('unbiased_weight', 0.95))
            normalise = bool(af_params.get('normalise_by_dim', False))
            results['almost_fair_crps'] = float(
                almost_fair_crps(forecast, observed, unbiased_weight=w, normalise_by_dim=normalise)
            )
        elif name == 'variogram':
            p = float((params.get('variogram', {}) or {}).get('p', 1.0))
            results['variogram'] = float(variogram_score_vectorized(forecast, observed, p=p))
        elif name == 'mse':
            results['mse'] = float(mse_vectorized(forecast, observed))
        elif name == 'mae':
            results['mae'] = float(mae_vectorized(forecast, observed))
        elif name in ('fixed_projected_es', 'projected_es', 'fixed_projection'):
            fp_params = (params.get('fixed_projected_es', {}) or {})
            seed = fp_params.get('seed', None)
            results['fixed_projected_es'] = float(fixed_projected_es(forecast, observed, seed=seed))
        elif name in ('optimal_proj_loss', 'optimal_projection', 'opt_proj_crps'):
            op_params = (params.get('optimal_proj_loss', {}) or {})
            eta = float(op_params.get('eta', 0.1))
            results['optimal_proj_loss'] = float(optimal_proj_loss(forecast, observed, eta=eta))
        elif name in ('dependence_proj_energy', 'dpe', 'dep_proj_es'):
            if perturb_details.get('method') == 'linear_mixing' and 'partitions' in perturb_details:
                dpe_params = (params.get('dependence_proj_energy', {}) or {})
                lambda_weight = float(dpe_params.get('lambda', 1.0))
                partitions = perturb_details['partitions']
                results['dependence_proj_energy'] = float(
                    dependence_proj_energy(forecast, observed, partitions=partitions, lambda_weight=lambda_weight)
                )
            else:
                continue
        elif name in ('mean_separated_proj_loss', 'mean_sep_proj', 'mean_proj_crps'):
            ms_params = (params.get('mean_separated_proj_loss', {}) or {})
            normalise = bool(ms_params.get('normalise', True))
            results['mean_separated_proj_loss'] = float(
                mean_separated_proj_loss(forecast, observed, normalise=normalise)
            )
        elif name in ('multivariate_proj_loss', 'mv_proj_loss'):
            mv_params = (params.get('multivariate_proj_loss', {}) or {})
            normalise = bool(mv_params.get('normalise_by_dim', False))
            results['multivariate_proj_loss'] = float(
                multivariate_proj_loss(
                    forecast,
                    observed,
                    projections=projection_matrices,
                    normalise_by_dim=normalise,
                )
            )
        else:
            raise ValueError(f"Unknown metric: {name}")
    return results


def run_experiment(cfg_path: str) -> str:
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get('seed', 42))
    np.random.seed(seed)

    base, theoretical_mean = _build_base_sampler_from_config(cfg)
    gen = MultivariateDataGenerator(base, theoretical_mean)

    n_forecast = int(cfg.get('n_forecast_samples', 1000))
    n_observed = int(cfg.get('n_observed_samples', 200))
    d = int(cfg.get('dimension', 10))

    desc, forecast, perturb_details = _build_perturbed_samples(gen, cfg, n_forecast, d)
    eval_dim = int(perturb_details.get('output_dimension', forecast.shape[1]))

    mean_cfg = (cfg.get('mean_perturbation', {}) or {})
    if bool(mean_cfg.get('enabled', False)):
        magnitude = float(mean_cfg.get('magnitude', 0.0))
        if magnitude != 0.0:
            direction = np.random.normal(size=eval_dim)
            norm = float(np.linalg.norm(direction))
            if norm > 0.0:
                direction = direction / norm
            shift = magnitude * direction
            forecast = forecast + shift.reshape(1, -1)
            perturb_details = dict(perturb_details)
            perturb_details['mean_perturbation'] = {
                'enabled': True,
                'magnitude': magnitude,
            }
            desc = f"{desc} + mean_shift(|mu|={magnitude})"

    observed = gen.generate_independent(n_observed, eval_dim)
    
    if gen.base_mean is not None:
        observed = observed - gen.base_mean

    metric_cfg = cfg.get('evaluation', {}) or {}
    proj_mats = generate_projection_matrices(eval_dim, seed=seed)
    main_metrics = _compute_metrics(forecast, observed, metric_cfg, perturb_details, projection_matrices=proj_mats)

    gt_sizes = cfg.get('ground_truth', {}) or {}
    gt_n = int(gt_sizes.get('n_observed_samples', n_observed))
    ground_truth = compute_ground_truth_metrics(
        generator=gen,
        n_observed_samples=gt_n,
        d=eval_dim,
        metric_cfg=metric_cfg,
        perturb_details=perturb_details,
        projection_matrices=proj_mats,
    )

    exp_name = cfg.get('experiment_name', 'discrimination_exp')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(os.path.dirname(cfg_path), '..', 'results', f"{exp_name}_{timestamp}")
    out_dir = os.path.abspath(out_dir)
    _ensure_dir(out_dir)

    with open(os.path.join(out_dir, 'config_used.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    np.save(os.path.join(out_dir, 'forecast.npy'), forecast)
    np.save(os.path.join(out_dir, 'observed.npy'), observed)

    percentage_increases: Dict[str, float] = {}
    for metric in main_metrics.keys():
        if metric in ground_truth and ground_truth[metric] > 0:
            percentage_increases[metric] = ((main_metrics[metric] - ground_truth[metric]) / ground_truth[metric]) * 100.0
        else:
            percentage_increases[metric] = float('nan')

    results: Dict[str, Any] = {
        'experiment_name': exp_name,
        'seed': seed,
        'timestamp': timestamp,
        'dimension': eval_dim,
        'n_forecast_samples': n_forecast,
        'n_observed_samples': n_observed,
        'base_distribution': cfg.get('base_distribution', 'gaussian'),
        'perturbation': perturb_details,
        'metrics': {
            'perturbed_vs_observed': main_metrics,
            'ground_truth_independent_vs_independent': ground_truth,
            'percentage_increase_from_ground_truth': percentage_increases,
        },
        'notes': 'Lower is better for all metrics. Observed are independent samples from the base distribution.'
    }

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    lines = [
        f"Experiment: {exp_name}",
        f"Timestamp: {timestamp}",
        f"Seed: {seed}",
        f"Dimension: {d}",
        f"Observed (independent): {n_observed} samples",
        f"Forecast (perturbed): {n_forecast} samples via {desc}",
        "",
        "Metrics (perturbed vs observed):",
    ]
    for k, v in main_metrics.items():
        lines.append(f"  - {k}: {v:.6f}")
    lines += [
        "",
        "Ground truth (independent vs independent):",
    ]
    for k, v in ground_truth.items():
        lines.append(f"  - {k}: {v:.6f}")
    lines += [
        "",
        "Percentage increase from ground truth:",
    ]
    for k in sorted(percentage_increases.keys()):
        pct = percentage_increases[k]
        if np.isnan(pct):
            lines.append(f"  - {k}: N/A (ground truth is 0 or missing)")
        else:
            lines.append(f"  - {k}: {pct:+.2f}%")

    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print('\n'.join(lines))
    print(f"\nSaved outputs to: {out_dir}")

    try:
        visualise_single_run(out_dir, seed=seed)
    except Exception as e:
        print(f"[visualise] Failed to generate figures for {out_dir}: {e}")
    return out_dir


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Run Discrimination experiment from config')
    p.add_argument('--config', '-c', type=str, required=True, help='Path to YAML config file')
    p.add_argument('--n_runs', type=int, default=1, help='Number of independent runs with different seeds')
    return p.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()

    if args.n_runs <= 1:
        run_experiment(args.config)
        sys.exit(0)

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    base_seed = int(cfg.get('seed', 42))
    exp_name = cfg.get('experiment_name', 'discrimination_exp')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    parent_dir = os.path.join(
        os.path.dirname(args.config), '..', 'results', f"{exp_name}_multi_{timestamp}"
    )
    parent_dir = os.path.abspath(parent_dir)
    _ensure_dir(parent_dir)

    with open(os.path.join(parent_dir, 'config_used.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    n_forecast = int(cfg.get('n_forecast_samples', 1000))
    n_observed = int(cfg.get('n_observed_samples', 200))
    d = int(cfg.get('dimension', 10))

    metric_cfg = cfg.get('evaluation', {}) or {}

    per_run_main_metrics: List[Dict[str, float]] = []
    per_run_ground_truth: List[Dict[str, float]] = []
    run_seeds: List[int] = []
    eval_dim_final: int = d

    for run_idx in range(int(args.n_runs)):
        run_seed = base_seed + run_idx
        run_seeds.append(run_seed)
        np.random.seed(run_seed)

        base, theoretical_mean = _build_base_sampler_from_config(cfg)
        gen = MultivariateDataGenerator(base, theoretical_mean)

        desc, forecast, perturb_details = _build_perturbed_samples(gen, cfg, n_forecast, d)
        eval_dim = int(perturb_details.get('output_dimension', forecast.shape[1]))

        mean_cfg = (cfg.get('mean_perturbation', {}) or {})
        if bool(mean_cfg.get('enabled', False)):
            magnitude = float(mean_cfg.get('magnitude', 0.0))
            if magnitude != 0.0:
                direction = np.random.normal(size=eval_dim)
                norm = float(np.linalg.norm(direction))
                if norm > 0.0:
                    direction = direction / norm
                shift = magnitude * direction
                forecast = forecast + shift.reshape(1, -1)
                perturb_details = dict(perturb_details)
                perturb_details['mean_perturbation'] = {
                    'enabled': True,
                    'magnitude': magnitude,
                }
                desc = f"{desc} + mean_shift(|mu|={magnitude})"
        eval_dim_final = eval_dim

        observed = gen.generate_independent(n_observed, eval_dim)
        
        if gen.base_mean is not None:
            observed = observed - gen.base_mean
        else:
            print(f"Warning: Cannot center observed data - base distribution mean unknown")
            print(f"This may introduce bias when comparing against centered forecast data")

        proj_mats = generate_projection_matrices(eval_dim, seed=run_seed)
        main_metrics = _compute_metrics(forecast, observed, metric_cfg, perturb_details, projection_matrices=proj_mats)

        gt_sizes = cfg.get('ground_truth', {}) or {}
        gt_n = int(gt_sizes.get('n_observed_samples', n_observed))
        ground_truth = compute_ground_truth_metrics(
            generator=gen,
            n_observed_samples=gt_n,
            d=eval_dim,
            metric_cfg=metric_cfg,
            perturb_details=perturb_details,
            projection_matrices=proj_mats,
        )

        per_run_main_metrics.append(main_metrics)
        per_run_ground_truth.append(ground_truth)

        run_percentage_increases: Dict[str, float] = {}
        for metric in main_metrics.keys():
            if metric in ground_truth and ground_truth[metric] > 0:
                run_percentage_increases[metric] = ((main_metrics[metric] - ground_truth[metric]) / ground_truth[metric]) * 100.0
            else:
                run_percentage_increases[metric] = float('nan')

        out_dir_run = os.path.join(parent_dir, f"run_{run_idx:03d}")
        _ensure_dir(out_dir_run)

        cfg_with_run = dict(cfg)
        cfg_with_run['seed'] = run_seed
        cfg_with_run['run_index'] = run_idx
        with open(os.path.join(out_dir_run, 'config_used.yaml'), 'w') as f:
            yaml.safe_dump(cfg_with_run, f, sort_keys=False)

        results_run: Dict[str, Any] = {
            'experiment_name': exp_name,
            'seed': run_seed,
            'timestamp_batch': timestamp,
            'run_index': run_idx,
            'dimension': eval_dim,
            'n_forecast_samples': n_forecast,
            'n_observed_samples': n_observed,
            'base_distribution': cfg.get('base_distribution', 'gaussian'),
            'perturbation': perturb_details,
            'metrics': {
                'perturbed_vs_observed': main_metrics,
                'ground_truth_independent_vs_independent': ground_truth,
                'percentage_increase_from_ground_truth': run_percentage_increases,
            },
            'notes': 'Lower is better for all metrics. Both observed and forecast data are centered (zero-mean) for fair comparison.'
        }

        with open(os.path.join(out_dir_run, 'results.json'), 'w') as f:
            json.dump(results_run, f, indent=2)

        lines = [
            f"Experiment: {exp_name}",
            f"Batch Timestamp: {timestamp}",
            f"Run: {run_idx}",
            f"Seed: {run_seed}",
            f"Dimension: {eval_dim}",
            f"Observed (independent): {n_observed} samples",
            f"Forecast (perturbed): {n_forecast} samples via {desc}",
            "",
            "Metrics (perturbed vs observed):",
        ]
        for k, v in main_metrics.items():
            lines.append(f"  - {k}: {v:.6f}")
        lines += [
            "",
            "Ground truth (independent vs independent):",
        ]
        for k, v in ground_truth.items():
            lines.append(f"  - {k}: {v:.6f}")
        lines += [
            "",
            "Percentage increase from ground truth:",
        ]
        for k in sorted(run_percentage_increases.keys()):
            pct = run_percentage_increases[k]
            if np.isnan(pct):
                lines.append(f"  - {k}: N/A (ground truth is 0 or missing)")
            else:
                lines.append(f"  - {k}: {pct:+.2f}%")
        with open(os.path.join(out_dir_run, 'summary.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')

        print('\n'.join(lines))

        try:
            np.save(os.path.join(out_dir_run, 'forecast.npy'), forecast)
            np.save(os.path.join(out_dir_run, 'observed.npy'), observed)
            visualise_single_run(out_dir_run, seed=run_seed)
        except Exception as e:
            print(f"[visualise] Run {run_idx}: failed to generate figures: {e}")

    def _aggregate(stats_list: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        if not stats_list:
            return {}, {}
        keys = sorted({k for dct in stats_list for k in dct.keys()})
        means: Dict[str, float] = {}
        stds: Dict[str, float] = {}
        for k in keys:
            vals = np.array([dct[k] for dct in stats_list if k in dct], dtype=float)
            if vals.size == 0:
                continue
            means[k] = float(np.mean(vals))
            stds[k] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        return means, stds

    mean_main, std_main = _aggregate(per_run_main_metrics)
    mean_gt, std_gt = _aggregate(per_run_ground_truth)

    percentage_increases: Dict[str, float] = {}
    for metric in mean_main.keys():
        if metric in mean_gt and mean_gt[metric] > 0:
            percentage_increases[metric] = ((mean_main[metric] - mean_gt[metric]) / mean_gt[metric]) * 100.0
        else:
            percentage_increases[metric] = float('nan')

    aggregate: Dict[str, Any] = {
        'experiment_name': exp_name,
        'timestamp': timestamp,
        'n_runs': int(args.n_runs),
        'base_seed': base_seed,
        'run_seeds': run_seeds,
        'dimension': eval_dim_final,
        'n_forecast_samples': n_forecast,
        'n_observed_samples': n_observed,
        'metrics': {
            'perturbed_vs_observed': {
                'mean': mean_main,
                'std': std_main,
            },
            'ground_truth_independent_vs_independent': {
                'mean': mean_gt,
                'std': std_gt,
            },
            'percentage_increase_from_ground_truth': percentage_increases,
        },
    }

    with open(os.path.join(parent_dir, 'aggregate_results.json'), 'w') as f:
        json.dump(aggregate, f, indent=2)

    lines = [
        f"Experiment (multi-run): {exp_name}",
        f"Timestamp: {timestamp}",
        f"Runs: {int(args.n_runs)}",
        f"Base seed: {base_seed}",
        f"Dimension: {eval_dim_final}",
        f"Observed (independent): {n_observed} samples",
        f"Forecast (perturbed): {n_forecast} samples",
        "",
        "Metrics (perturbed vs observed): mean ± std",
    ]
    for k in sorted(mean_main.keys()):
        lines.append(f"  - {k}: {mean_main[k]:.6f} ± {std_main.get(k, 0.0):.6f}")
    lines += [
        "",
        "Ground truth (independent vs independent): mean ± std",
    ]
    for k in sorted(mean_gt.keys()):
        lines.append(f"  - {k}: {mean_gt[k]:.6f} ± {std_gt.get(k, 0.0):.6f}")
    lines += [
        "",
        "Percentage increase from ground truth:",
    ]
    for k in sorted(percentage_increases.keys()):
        pct = percentage_increases[k]
        if np.isnan(pct):
            lines.append(f"  - {k}: N/A (ground truth is 0 or missing)")
        else:
            lines.append(f"  - {k}: {pct:+.2f}%")

    with open(os.path.join(parent_dir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print('\n'.join(lines))
    print(f"\nSaved multi-run outputs to: {parent_dir}")


