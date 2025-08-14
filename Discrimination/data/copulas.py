"""
Copula utilities: inverse–Rosenblatt transform and specific copula helpers.

This module provides:
- inverse_rosenblatt_batch: convert i.i.d. uniforms to dependent uniforms via a
  user-supplied conditional inverse CDF function (the copula).
- Clayton copula conditional inverse and parameter helpers.
"""

from __future__ import annotations

import numpy as np
from typing import Callable


def inverse_rosenblatt_single(U: np.ndarray, cond_inv: Callable[[int, float, np.ndarray], float]) -> np.ndarray:
    """
    Apply the inverse–Rosenblatt transform to a single vector of independent uniforms U. 
    See https://vinecopulib.github.io/rvinecopulib/reference/rosenblatt.html for more details.  

    Args:
        U: Array of shape (d,) with values in (0, 1)
        cond_inv: Function (k, u, prev) -> v that returns the conditional inverse CDF
                  for component k given previous components prev=v_1..v_{k-1}.

    Returns:
        V: Array of shape (d,) such that V has the target copula dependence.
    """
    d = U.size
    V = np.empty(d, dtype=float)
    # First component is identity for most copulas in the Rosenblatt ordering
    V[0] = float(np.clip(U[0], 1e-12, 1 - 1e-12))
    for k in range(1, d):  # k is 0-based here; cond_inv expects 1-based index
        V[k] = cond_inv(k + 1, float(np.clip(U[k], 1e-12, 1 - 1e-12)), V[:k])
    return V


def inverse_rosenblatt_batch(U: np.ndarray, cond_inv: Callable[[int, float, np.ndarray], float]) -> np.ndarray:
    """
    Batched inverse–Rosenblatt
    """
    U = np.asarray(U)
    if U.ndim == 1:
        return inverse_rosenblatt_single(U, cond_inv)
    n, d = U.shape
    V = np.empty_like(U, dtype=float)
    for i in range(n):
        V[i] = inverse_rosenblatt_single(U[i], cond_inv)
    return V


def clayton_cond_inv(theta: float) -> Callable[[int, float, np.ndarray], float]:
    """
    Conditional inverse for the d-dimensional Clayton copula with parameter theta.
    """
    if theta == 0:
        raise ValueError("theta=0 corresponds to independence; choose another copula/parameter")

    def _cond_inv(k: int, u: float, prev: np.ndarray) -> float:
        # k is 1-based per the mathematical convention
        u = float(np.clip(u, 1e-12, 1 - 1e-12))
        if k == 1:
            return u
        v_prev = np.asarray(prev, dtype=float)
        t = float(np.sum(v_prev ** (-theta) - 1.0))
        alpha = 1.0 / theta + (k - 1)
        s = (1.0 + t) * (u ** (-1.0 / alpha) - 1.0)
        v = (1.0 + s) ** (-1.0 / theta)
        return float(np.clip(v, 1e-12, 1 - 1e-12))

    return _cond_inv


def theta_from_tau_clayton(tau: float) -> float:
    if not (0.0 <= tau < 1.0):
        raise ValueError("tau must be in [0, 1) for the Clayton copula")
    return 2.0 * tau / (1.0 - tau)


