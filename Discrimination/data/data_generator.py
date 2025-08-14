"""
Data generation module for multivariate random vectors.
"""

import numpy as np
from scipy.stats import norm
from typing import Callable, Union, Optional, Tuple
import warnings
from .copulas import (
    inverse_rosenblatt_batch,
    clayton_cond_inv,
    theta_from_tau_clayton,
)


class MultivariateDataGenerator:
    
    def __init__(self, base_distribution: Union[str, Callable] = 'normal', theoretical_mean: Optional[float] = None):
        """
        Initialise the data generator.
        """
        if isinstance(base_distribution, str):
            self.base_dist_name = base_distribution
            self.base_sampler = self._get_base_sampler(base_distribution)
            self.base_mean = self._get_base_mean(base_distribution)
        else:
            self.base_dist_name = 'custom'
            self.base_sampler = base_distribution
            self.base_mean = theoretical_mean  # Use provided mean or None
    
    def _get_base_sampler(self, dist_name: str) -> Callable:
        samplers = {
            # Requested 1D base distributions and a few extras
            'gaussian': lambda size: np.random.normal(0.0, 1.0, size),
            'normal': lambda size: np.random.normal(0.0, 1.0, size),
            'student_t': lambda size: np.random.standard_t(df=5, size=size),
            'exponential': lambda size: np.random.exponential(scale=1.0, size=size),
            'gamma': lambda size: np.random.gamma(shape=2.0, scale=1.0, size=size),
            'beta': lambda size: np.random.beta(a=2.0, b=2.0, size=size),
            # Legacy options kept for completeness
            'uniform': lambda size: np.random.uniform(-1, 1, size),
            'laplace': lambda size: np.random.laplace(0.0, 1.0, size),
        }
        
        if dist_name not in samplers:
            raise ValueError(
                f"Unknown distribution: {dist_name}. Available: {list(samplers.keys())}"
            )
        
        return samplers[dist_name]
    
    def _get_base_mean(self, dist_name: str) -> float:
        means = {
            'gaussian': 0.0,
            'normal': 0.0, 
            'student_t': 0.0,  # for df > 1
            'exponential': 1.0,  # scale=1.0
            'gamma': 2.0,  # shape=2.0, scale=1.0 -> mean = shape * scale
            'beta': 2.0/7.0,  # a=2.0, b=5.0 -> mean = a/(a+b)
            'uniform': 0.0,  # uniform(-1, 1) -> mean = (a+b)/2
            'laplace': 0.0,
        }
        
        if dist_name not in means:
            raise ValueError(f"Unknown distribution: {dist_name}")
        
        return means[dist_name]
    
    def generate_independent(self, n_samples: int, d: int, center: bool = False) -> np.ndarray:
        """
        Generate independent multivariate vectors.
        
        Each component is an i.i.d. sample from the base distribution.
            
        Returns:
            Array of shape (n_samples, d)
        """
        X = self.base_sampler((n_samples, d))
        
        if center and self.base_mean is not None:
            X = X - self.base_mean
        elif center and self.base_mean is None:
            warnings.warn("Cannot center custom distribution - mean unknown. Returning uncentered samples.")
        
        return X
    
    def generate_common_shock(
        self,
        n_samples: int,
        d: int,
        rho: float,
        preserve_variance: bool = True
    ) -> np.ndarray:
        """
        Generate equi-correlated multivariate vectors using common-shock method.
        
        The base distribution is always centered before applying perturbations.
        Y_i = a * X_i + b * X̄        
        where X_i are centered i.i.d. samples from base distribution and X̄ is their mean.
        The resulting samples will have zero mean.
        Returns:
            Array of shape (n_samples, d) with zero mean
        """
        # Validate correlation range
        min_rho = -1 / (d - 1)
        if not (min_rho < rho < 1):
            raise ValueError(f"Correlation rho must be in ({min_rho:.3f}, 1), got {rho}")
        
        # Generate base samples and center them
        X = self.base_sampler((n_samples, d))
        
        # Center the base distribution
        if self.base_mean is not None:
            X = X - self.base_mean
        else:
            warnings.warn("Cannot center custom distribution - mean unknown. Using uncentered samples.")
        
        if preserve_variance:
            # Coefficients to preserve variance and achieve target correlation
            a = np.sqrt(1 - rho)
            b = -a + np.sqrt(1 + (d - 1) * rho)
        else:
            # Simpler version: Y_i = X_i + λ * X̄
            # rho = λ(λ+2)/d / (1 + λ(λ+2)/d)
            # Solving for λ:
            lambda_val = -1 + np.sqrt(1 + d * rho / (1 - rho))
            a = 1
            b = lambda_val
        
        # Apply common-shock transformation
        X_bar = np.mean(X, axis=1, keepdims=True)  # Shape: (n_samples, 1)
        Y = a * X + b * X_bar
        
        return Y
    
    def generate_gaussian_copula(
        self,
        n_samples: int,
        d: int,
        correlation_matrix: Optional[np.ndarray] = None,
        rho: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate multivariate vectors with Gaussian copula dependence.
        
        Returns:
            Array of shape (n_samples, d)
        """
        # Set up correlation matrix
        if correlation_matrix is None:
            if rho is None:
                raise ValueError("Must provide either correlation_matrix or rho")
            
            # Create equi-correlation matrix
            min_rho = -1 / (d - 1)
            if not (min_rho < rho < 1):
                raise ValueError(f"Correlation rho must be in ({min_rho:.3f}, 1), got {rho}")
            
            correlation_matrix = (1 - rho) * np.eye(d) + rho * np.ones((d, d))
        else:
            correlation_matrix = np.asarray(correlation_matrix)
            if correlation_matrix.shape != (d, d):
                raise ValueError(f"Correlation matrix must have shape ({d}, {d}), "
                               f"got {correlation_matrix.shape}")
            
            # Check if matrix is valid (symmetric positive definite)
            if not np.allclose(correlation_matrix, correlation_matrix.T):
                raise ValueError("Correlation matrix must be symmetric")
            
            eigenvalues = np.linalg.eigvalsh(correlation_matrix)
            if np.min(eigenvalues) <= 0:
                # Project to nearest positive definite if needed
                warnings.warn("Correlation matrix is not positive definite, projecting to nearest PD")
                correlation_matrix = self._nearest_positive_definite(correlation_matrix)
        
        # Step 1: Generate independent samples from base distribution
        X = self.base_sampler((n_samples, d))
        
        # Step 2: Estimate empirical CDF and map to uniforms
        # For better accuracy with small samples, use rank-based pseudo-observations
        U = np.zeros_like(X)
        for j in range(d):
            ranks = np.argsort(np.argsort(X[:, j])) + 1
            U[:, j] = (ranks - 0.5) / n_samples
        
        # Step 3: Map to independent standard normals
        W = norm.ppf(U)
        
        # Step 4: Impose correlation structure
        L = np.linalg.cholesky(correlation_matrix)
        Z = W @ L.T  # Shape: (n_samples, d)
        
        # Step 5: Map back to original marginal distribution
        # Using empirical quantile function
        Y = np.zeros_like(Z)
        for j in range(d):
            # Get sorted values for this dimension
            X_sorted = np.sort(X[:, j])
            
            # Map from normal to uniform
            U_target = norm.cdf(Z[:, j])
            
            # Map from uniform to empirical quantiles
            # Linear interpolation between sorted values
            grid = (np.arange(1, n_samples + 1) - 0.5) / n_samples
            Y[:, j] = np.interp(U_target, grid, X_sorted)
        
        return Y

    def generate_copula_general(
        self,
        n_samples: int,
        d: int,
        cond_inv: Optional[Callable[[int, float, np.ndarray], float]] = None,
        copula: str = 'gaussian',
        correlation_matrix: Optional[np.ndarray] = None,
        rho: Optional[float] = None,
        clayton_theta: Optional[float] = None,
        clayton_tau: Optional[float] = None,
        ref_sample: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate samples with general copula dependence while preserving the marginal of the base distribution.
        Returns:
            Array of shape (n_samples, d)
        """
        # 1) draw i.i.d. base samples X
        X = self.base_sampler((n_samples, d))

        # build empirical CDF/quantile from ref_sample if provided, else from X itself (per-dimension)
        # Step 1.5: Map X -> U via PIT (per-dimension empirical CDF)
        U = np.zeros_like(X, dtype=float)
        # If ref_sample is provided, use the same ref for all dimensions; otherwise use each column of X
        if ref_sample is not None:
            ref = np.sort(np.asarray(ref_sample).reshape(-1))
            grid_ref = (np.arange(1, ref.size + 1) - 0.5) / ref.size
            for j in range(d):
                U[:, j] = np.interp(X[:, j], ref, grid_ref, left=grid_ref[0], right=grid_ref[-1])
        else:
            for j in range(d):
                col = X[:, j]
                order = np.argsort(col)
                ranks = np.empty_like(order, dtype=float)
                ranks[order] = np.arange(1, n_samples + 1)
                U[:, j] = (ranks - 0.5) / n_samples

        U = np.clip(U, 1e-12, 1 - 1e-12)

        # 2) impose dependence: choose cond_inv
        if cond_inv is None:
            if copula.lower() == 'gaussian':
                # reuse existing gaussian path efficiently (vectorised)
                # Map U -> standard normals
                W = norm.ppf(U)
                # Build correlation
                if correlation_matrix is None:
                    if rho is None:
                        raise ValueError("For gaussian copula, provide correlation_matrix or rho")
                    min_rho = -1 / (d - 1)
                    if not (min_rho < rho < 1):
                        raise ValueError(f"rho must be in ({min_rho:.3f}, 1) for dimension {d}")
                    correlation_matrix = (1 - rho) * np.eye(d) + rho * np.ones((d, d))
                else:
                    correlation_matrix = np.asarray(correlation_matrix)
                    if correlation_matrix.shape != (d, d):
                        raise ValueError(f"correlation_matrix must be ({d},{d})")
                    if not np.allclose(correlation_matrix, correlation_matrix.T):
                        raise ValueError("correlation_matrix must be symmetric")
                    eig = np.linalg.eigvalsh(correlation_matrix)
                    if np.min(eig) <= 0:
                        warnings.warn("Correlation not PD; projecting to nearest PD")
                        correlation_matrix = self._nearest_positive_definite(correlation_matrix)
                L = np.linalg.cholesky(correlation_matrix)
                Z = W @ L.T
                V = norm.cdf(Z)
            elif copula.lower() == 'clayton':
                if clayton_theta is None:
                    if clayton_tau is None:
                        raise ValueError("Provide clayton_theta or clayton_tau for Clayton copula")
                    clayton_theta = theta_from_tau_clayton(float(clayton_tau))
                cond_inv_auto = clayton_cond_inv(float(clayton_theta))
                V = inverse_rosenblatt_batch(U, cond_inv_auto)
            else:
                raise ValueError("Unsupported copula. Use 'gaussian', 'clayton', or provide cond_inv.")
        else:
            V = inverse_rosenblatt_batch(U, cond_inv)

        # 3) map back to original marginal via empirical quantiles (per-dimension)
        Y = np.zeros_like(V, dtype=float)
        if ref_sample is not None:
            # use shared ref_sample quantile for all dimensions
            ref = np.sort(np.asarray(ref_sample).reshape(-1))
            grid_ref = (np.arange(1, ref.size + 1) - 0.5) / ref.size
            for j in range(d):
                Y[:, j] = np.interp(V[:, j], grid_ref, ref)
        else:
            for j in range(d):
                col_sorted = np.sort(X[:, j])
                grid = (np.arange(1, n_samples + 1) - 0.5) / n_samples
                Y[:, j] = np.interp(V[:, j], grid, col_sorted)

        return np.asarray(Y)
    
    def _nearest_positive_definite(self, A: np.ndarray) -> np.ndarray:
        """Find the nearest positive definite matrix to A."""
        # Symmetric part
        B = (A + A.T) / 2
        
        # Eigenvalue decomposition
        eigval, eigvec = np.linalg.eigh(B)
        
        # Make positive definite by clipping eigenvalues
        eigval[eigval < 1e-8] = 1e-8
        
        # Reconstruct
        return eigvec @ np.diag(eigval) @ eigvec.T


def generate_ar1_correlation_matrix(d: int, rho: float) -> np.ndarray:
    """
    Generate AR(1) correlation matrix where Corr(X_i, X_j) = rho^|i-j|.
        
    Returns:
        Correlation matrix of shape (d, d)
    """
    if not -1 < rho < 1:
        raise ValueError(f"AR parameter rho must be in (-1, 1), got {rho}")
    
    R = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            R[i, j] = rho ** abs(i - j)
    
    return R


def generate_block_correlation_matrix(
    d: int,
    block_sizes: list,
    within_block_corr: float,
    between_block_corr: float = 0.0
) -> np.ndarray:
    """
    Generate block correlation matrix.
    Returns:
        Correlation matrix of shape (d, d)
    """
    if sum(block_sizes) != d:
        raise ValueError(f"Block sizes must sum to dimension {d}, got sum {sum(block_sizes)}")
    
    R = np.full((d, d), between_block_corr)
    
    # Fill in blocks
    start = 0
    for block_size in block_sizes:
        end = start + block_size
        # Within block correlation
        R[start:end, start:end] = within_block_corr
        # Diagonal elements
        for i in range(start, end):
            R[i, i] = 1.0
        start = end
    
    return R