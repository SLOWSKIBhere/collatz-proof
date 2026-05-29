"""
Test for Theorem 8.4: Boundary Values of the Resolvent.

Exact replica of the original Jupyter implementation — optimised.
Two tests:
  1. Fredholm difference: verifies rank(U_nil - S) = 1 (compact perturbation).
  2. Resolvent boundary: detects peaks of ‖(zI - U)^{-1}‖ on the unit circle.
"""

import numpy as np
from utils.koopman_operator import compute_closure_safe, build_koopman_matrix
from utils.collatz_map import accelerated_collatz
from config import BOUNDARY_NUM_THETA


def test_boundary_resolvent(
    max_size_fredholm: int = 2000,
    max_size_boundary: int = 1000,
    num_theta: int = None,
) -> dict:
    """
    Test boundary values of the resolvent and Fredholm difference.
    """
    if num_theta is None:
        num_theta = BOUNDARY_NUM_THETA

    # ================================================================
    # Test 1: Fredholm difference — rank(U_nil - S)
    # ================================================================
    numbers_f, M_f = compute_closure_safe([1], max_steps=50, max_size=max_size_fredholm)
    U_sparse_f, num_to_idx_f, _ = build_koopman_matrix(numbers_f)
    U_dense_f = U_sparse_f.toarray()

    U_cycle = np.zeros((M_f, M_f))
    idx_1, idx_2 = num_to_idx_f[1], num_to_idx_f[2]
    U_cycle[idx_1, idx_2] = 1.0
    U_cycle[idx_2, idx_1] = 1.0

    U_nil = U_dense_f - U_cycle

    # Vectorised S construction
    numbers_arr = np.array(numbers_f, dtype=np.int64)
    tree_mask = ~np.isin(numbers_arr, [1, 2])
    tree_nums = numbers_arr[tree_mask]
    nxt_vals = accelerated_collatz(tree_nums)
    valid_mask = np.isin(nxt_vals, numbers_arr) & ~np.isin(nxt_vals, [1, 2])

    S = np.zeros((M_f, M_f))
    if np.any(valid_mask):
        src_idx = np.array([num_to_idx_f[n] for n in tree_nums[valid_mask]])
        dst_idx = np.array([num_to_idx_f[n] for n in nxt_vals[valid_mask]])
        S[dst_idx, src_idx] = 1.0

    diff = U_nil - S
    svd_vals = np.linalg.svdvals(diff)
    rank_diff = int(np.sum(svd_vals > 1e-10))
    nuclear_norm = float(np.sum(svd_vals))

    # ================================================================
    # Test 2: Resolvent peaks on unit circle
    # ================================================================
    numbers_b, M_b = compute_closure_safe([1], max_steps=50, max_size=max_size_boundary)
    U_sparse_b, _, _ = build_koopman_matrix(numbers_b)
    U_dense_b = U_sparse_b.toarray()
    I = np.eye(M_b)

    theta_vals = np.linspace(0, 2 * np.pi, num_theta)
    r_values = [1.0, 1.1, 1.2]
    norms = np.zeros((3, num_theta))

    for i, theta in enumerate(theta_vals):
        for j, r in enumerate(r_values):
            z = r * np.exp(1j * theta)
            try:
                R = np.linalg.inv(z * I - U_dense_b)
                norms[j, i] = np.linalg.norm(R, 'fro')
            except np.linalg.LinAlgError:
                norms[j, i] = np.inf

    norms_r10 = norms[0]
    finite_mask = np.isfinite(norms_r10)
    mean_norm = float(np.mean(norms_r10[finite_mask])) if np.any(finite_mask) else 0.0
    threshold = mean_norm * 5

    peak_mask = (norms_r10 > threshold) | ~finite_mask
    peaks = theta_vals[peak_mask].tolist()

    # Original logic: check that peaks correspond to z=+1 and z=-1
    peak_near_plus_one = any(
        min(abs(p - 0), abs(p - 2 * np.pi)) < 0.2 for p in peaks
    )
    peak_near_minus_one = any(abs(p - np.pi) < 0.2 for p in peaks)
    passed = bool(peak_near_plus_one and peak_near_minus_one)

    return {
        'test': 'boundary_resolvent',
        'passed': passed,
        'matrix_size_fredholm': M_f,
        'matrix_size_boundary': M_b,
        'num_theta': num_theta,
        'rank_diff': rank_diff,
        'nuclear_norm': nuclear_norm,
        'fredholm_passed': bool(rank_diff == 1),
        'mean_resolvent_norm': mean_norm,
        'num_peaks': len(peaks),
        'peak_thetas': peaks,
        'peak_at_plus_one': bool(peak_near_plus_one),
        'peak_at_minus_one': bool(peak_near_minus_one),
        'theta_vals': theta_vals.tolist(),
        'norms_r10': norms[0].tolist(),
        'norms_r11': norms[1].tolist(),
        'norms_r12': norms[2].tolist(),
    }