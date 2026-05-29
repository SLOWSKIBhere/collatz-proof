"""
Shared tensor utilities for Part D tests.
"""

import numpy as np


def construct_kinematic_tensor(lengths: np.ndarray, log_values: np.ndarray) -> tuple:
    """
    Build the t×2 kinematic tensor and compute its condition number.

    Parameters
    ----------
    lengths : np.ndarray
        Block lengths (m_i).
    log_values : np.ndarray
        log2 of starting values (V_i).

    Returns
    -------
    sigma1 : float
        First singular value.
    sigma2 : float
        Second singular value.
    kappa : float
        Condition number σ₁/σ₂.
    """
    if len(lengths) < 2:
        return None, None, None
    T = np.column_stack([lengths.astype(np.float64), log_values.astype(np.float64)])
    T_centered = T - np.mean(T, axis=0)
    _, S, _ = np.linalg.svd(T_centered, full_matrices=False)
    sigma1 = S[0]
    sigma2 = S[1] if len(S) > 1 else 0
    kappa = sigma1 / sigma2 if sigma2 > 1e-10 else np.inf
    return sigma1, sigma2, kappa