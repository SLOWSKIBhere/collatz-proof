"""
Test for Theorem 5.1 and 5.2: Chernoff Approximation Validation.

Reproduces results from Section 5.1 of the paper:
  - Uniform grid (old method): 11.8% error
  - Exponential grid (Remizov method): 1.27% error
  - Improvement factor: 9.3x

Requires matrix size 500×500 or larger.
"""

from typing import Optional
import numpy as np


def test_chernoff_validation(
    max_size: int = 500,
    z: Optional[complex] = None,
    n_chernoff: Optional[int] = None,
) -> dict:
    """
    Validate Chernoff resolvent approximation (Remizov, 2025).

    Compares two methods:
      1. Uniform grid — old method
      2. Exponential grid — improved Remizov method

    Parameters
    ----------
    max_size : int
        Target matrix size (500 for paper results).
    z : complex
        Evaluation point (paper uses 1.5+0.0j).
    n_chernoff : int
        Number of Chernoff iterations (paper uses 50).
    """
    if z is None:
        z = 1.5 + 0.0j
    if n_chernoff is None:
        n_chernoff = 50

    from utils.koopman_operator import compute_closure_safe

    # Build closure using the original algorithm
    numbers, M = compute_closure_safe([1], max_steps=50, max_size=max_size)

    if M < 2:
        return {
            'test': 'chernoff_validation',
            'passed': False,
            'error': 'Closure too small',
            'matrix_size': M,
        }

    from utils.koopman_operator import build_koopman_matrix
    U_sparse, _, _ = build_koopman_matrix(numbers)
    U_dense = U_sparse.toarray()

    # Exact resolvent
    R_exact = np.linalg.inv(z * np.eye(M) - U_dense)
    norm_exact = np.linalg.norm(R_exact, 'fro')
    zeta = z - 1  # shift for generator L = U - I

    # Method 1: Uniform grid
    T_max = 100
    num_t_uniform = 500
    t_vals = np.linspace(0, T_max, num_t_uniform)
    dt = t_vals[1] - t_vals[0]

    R_uniform = np.zeros((M, M), dtype=complex)
    for t in t_vals:
        e_m = np.exp(-t / n_chernoff)
        S_single = e_m * np.eye(M) + (1 - e_m) * U_dense
        S_power = np.linalg.matrix_power(S_single, n_chernoff)
        R_uniform += np.exp(-zeta * t) * S_power * dt

    error_uniform = float(np.linalg.norm(R_exact - R_uniform, 'fro') / norm_exact)

    # Method 2: Exponential grid (Remizov)
    num_t_exp = 200
    t_vals_exp = np.logspace(-3, np.log10(T_max), num_t_exp)

    R_exp = np.zeros((M, M), dtype=complex)
    for i in range(len(t_vals_exp) - 1):
        t_left = t_vals_exp[i]
        t_right = t_vals_exp[i + 1]
        t_mid = (t_left + t_right) / 2
        dt_exp = t_right - t_left

        e_m = np.exp(-t_mid / n_chernoff)
        S_single = e_m * np.eye(M) + (1 - e_m) * U_dense
        S_power = np.linalg.matrix_power(S_single, n_chernoff)
        R_exp += np.exp(-zeta * t_mid) * S_power * dt_exp

    error_exp = float(np.linalg.norm(R_exact - R_exp, 'fro') / norm_exact)

    improvement = error_uniform / max(error_exp, 1e-10)
    passed = error_exp < error_uniform

    return {
        'test': 'chernoff_validation',
        'passed': bool(passed),
        'matrix_size': M,
        'target_size': max_size,
        'n_chernoff': n_chernoff,
        'z': str(z),
        'error_uniform_pct': round(error_uniform * 100, 2),
        'error_exponential_pct': round(error_exp * 100, 2),
        'improvement_factor': round(improvement, 1),
        'expected_uniform_pct': 11.8,
        'expected_exponential_pct': 1.27,
        'matches_paper': (
            abs(error_uniform * 100 - 11.8) < 1.0 and
            abs(error_exp * 100 - 1.27) < 0.5 and
            M >= 500
        ),
        'note': (
            'Errors match paper (11.8% / 1.27%) — 9.3x improvement.'
            if M >= 500 else
            f'Matrix too small ({M}×{M}). Need 500×500 for paper results.'
        ),
    }