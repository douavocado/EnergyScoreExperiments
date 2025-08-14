"""
Visualisation utilities for Discrimination experiments.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import yaml

from metrics import (
    crps_component_wise_vectorized,
    crps_score_vectorized,
    optimal_projection_vector,
    mean_separated_projection_vector,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_run_arrays(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    f_path = os.path.join(run_dir, 'forecast.npy')
    o_path = os.path.join(run_dir, 'observed.npy')
    if not (os.path.exists(f_path) and os.path.exists(o_path)):
        raise FileNotFoundError(
            f"Missing required arrays in {run_dir}. Expected 'forecast.npy' and 'observed.npy'"
        )
    forecast = np.load(f_path)
    observed = np.load(o_path)
    return forecast, observed


def _set_pub_style() -> None:
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'STIX', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
    })


def _compute_bins(values: np.ndarray, min_bins: int = 15, max_bins: int = 40) -> np.ndarray:
    try:
        edges = np.histogram_bin_edges(values, bins='fd')
    except Exception:
        edges = np.histogram_bin_edges(values, bins=30)
    if edges.size - 1 > max_bins:
        edges = np.linspace(values.min(), values.max(), max_bins + 1)
    if edges.size - 1 < min_bins:
        edges = np.linspace(values.min(), values.max(), min_bins + 1)
    return edges


def _hist_two(
    ax,
    a: np.ndarray,
    b: np.ndarray,
    label_a: str = 'Forecast',
    label_b: str = 'Observed',
) -> None:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    both = np.concatenate([a, b])
    if np.allclose(both.min(), both.max()):
        m = both.min()
        both = np.array([m - 1e-6, m + 1e-6])
    bins = _compute_bins(both)

    color_a = '#1f77b4'
    color_b = '#d62728'
    lw = 1.2

    ax.hist(a, bins=bins, alpha=0.45, density=True, label=label_a,
            color=color_a, edgecolor=color_a, linewidth=lw, histtype='stepfilled')
    ax.hist(b, bins=bins, alpha=0.45, density=True, label=label_b,
            color=color_b, edgecolor=color_b, linewidth=lw, histtype='stepfilled')

    try:
        from scipy.stats import gaussian_kde

        xs = np.linspace(both.min(), both.max(), 400)
        kde_a = gaussian_kde(a)
        kde_b = gaussian_kde(b)
        ax.plot(xs, kde_a(xs), color=color_a, linewidth=1.6)
        ax.plot(xs, kde_b(xs), color=color_b, linewidth=1.6)
    except Exception:
        pass

    ax.legend(frameon=True, framealpha=0.9, fancybox=False, loc='best')
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.6)


def _save_tikz(fig, base_path_no_ext: str) -> None:
    try:
        import tikzplotlib
        tikz_path = base_path_no_ext + '.tex'
        tikzplotlib.save(tikz_path, figure=fig)
        print(f"  Saved TikZ: {os.path.basename(tikz_path)}")
    except ImportError:
        try:
            pgf_path = base_path_no_ext + '.pgf'
            fig.savefig(pgf_path, backend='pgf')
            print(f"  Saved PGF: {os.path.basename(pgf_path)} (use \\input{{path}} in LaTeX)")
        except Exception:
            print(f"  Warning: tikzplotlib not available and PGF backend failed, skipping TikZ export for {os.path.basename(base_path_no_ext)}")
    except Exception as e:
        try:
            pgf_path = base_path_no_ext + '.pgf'
            fig.savefig(pgf_path, backend='pgf')
            print(f"  TikZ failed ({e}), saved PGF instead: {os.path.basename(pgf_path)}")
        except Exception:
            print(f"  Warning: Both TikZ and PGF export failed for {os.path.basename(base_path_no_ext)}: {e}")


def visualise_single_run(run_dir: str, seed: Optional[int] = None) -> None:
    figures_dir = os.path.join(run_dir, 'figures')
    _ensure_dir(figures_dir)

    forecast, observed = _load_run_arrays(run_dir)

    _set_pub_style()

    eta_opt = 0.1
    cfg_path = os.path.join(run_dir, 'config_used.yaml')
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
            eval_cfg = cfg.get('evaluation', {}) or {}
            metric_params = eval_cfg.get('metric_params', {}) or {}
            opt_cfg = metric_params.get('optimal_proj_loss', {}) or {}
            eta_opt = float(opt_cfg.get('eta', eta_opt))
        except Exception:
            pass

    crps_per_dim = crps_component_wise_vectorized(forecast, observed)
    worst_dim = int(np.argmax(crps_per_dim))

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    _hist_two(ax, forecast[:, worst_dim], observed[:, worst_dim],
              label_a='Forecast', label_b='Observed')
    ax.set_title('CRPS Maximal Projection')
    ax.set_xlabel('Projected Subspace')
    ax.set_ylabel('Probability density')
    fig.tight_layout()
    out_base = os.path.join(figures_dir, f'crps_worst_component_{worst_dim}')
    fig.savefig(out_base + '.png')
    fig.savefig(out_base + '.pdf')
    _save_tikz(fig, out_base)
    plt.close(fig)

    from metrics.fixed_projected_es import _get_fixed_orthonormal_basis
    d = forecast.shape[1]
    basis = _get_fixed_orthonormal_basis(d, seed=seed)
    crps_proj = []
    for j in range(d):
        proj_f_j = forecast @ basis[:, j]
        proj_o_j = observed @ basis[:, j]
        crps_j = crps_score_vectorized(proj_f_j, proj_o_j)
        crps_proj.append(float(crps_j))
    crps_proj = np.asarray(crps_proj, dtype=float)
    worst_proj = int(np.argmax(crps_proj))
    proj_f = forecast @ basis[:, worst_proj]
    proj_o = observed @ basis[:, worst_proj]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    _hist_two(ax, proj_f, proj_o, label_a='Forecast (proj)', label_b='Observed (proj)')
    ax.set_title('Fixed Basis Maximal Projection')
    ax.set_xlabel('Projected Subspace')
    ax.set_ylabel('Probability density')
    fig.tight_layout()
    out_base = os.path.join(figures_dir, f'fixed_basis_worst_crps_{worst_proj}')
    fig.savefig(out_base + '.png')
    fig.savefig(out_base + '.pdf')
    _save_tikz(fig, out_base)
    plt.close(fig)

    theta = optimal_projection_vector(forecast, observed, eta=eta_opt)
    proj_f_opt = forecast @ theta
    proj_o_opt = observed @ theta

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    _hist_two(ax, proj_f_opt, proj_o_opt, label_a='Forecast (opt proj)', label_b='Observed (opt proj)')
    ax.set_title('Optimal Projection')
    ax.set_xlabel('Projected Subspace')
    ax.set_ylabel('Probability density')
    fig.tight_layout()
    out_base = os.path.join(figures_dir, 'optimal_projection_hist')
    fig.savefig(out_base + '.png')
    fig.savefig(out_base + '.pdf')
    _save_tikz(fig, out_base)
    plt.close(fig)

    theta_mean = mean_separated_projection_vector(forecast, observed, normalise=True)
    proj_f_mean = forecast @ theta_mean
    proj_o_mean = observed @ theta_mean

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    _hist_two(ax, proj_f_mean, proj_o_mean, label_a='Forecast (mean proj)', label_b='Observed (mean proj)')
    ax.set_title('Mean Separated Projection')
    ax.set_xlabel('Projected Subspace')
    ax.set_ylabel('Probability density')
    fig.tight_layout()
    out_base = os.path.join(figures_dir, 'mean_separated_projection_hist')
    fig.savefig(out_base + '.png')
    fig.savefig(out_base + '.pdf')
    _save_tikz(fig, out_base)
    plt.close(fig)


def visualise_parent_batch(parent_dir: str, n_runs: Optional[int] = None, seed: Optional[int] = None) -> None:
    run_dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if d.startswith('run_')]
    run_dirs = sorted(run_dirs)
    if n_runs is not None:
        run_dirs = run_dirs[: int(n_runs)]
    for rd in run_dirs:
        try:
            visualise_single_run(rd, seed=seed)
        except Exception as e:
            print(f"[visualise_parent_batch] Skipping {rd}: {e}")


__all__ = [
    'visualise_single_run',
    'visualise_parent_batch',
]


