"""
Part D: Cauchy Root Criterion for Generating Function R(z) = Σ κ_k z^k.
"""

import numpy as np
from utils.collatz_map import compute_trajectory_with_blocks
from utils.tensor_utils import construct_kinematic_tensor

def compute_root_distribution(kappa_sequence: np.ndarray) -> dict:
    """
    Compute roots of generating function R(z) = Σ κ_k z^k
    and check whether they fall inside or outside the unit circle.
    """
    kappa = kappa_sequence[:min(len(kappa_sequence), 20)]
    n = len(kappa)
    if n < 2:
        return {'all_inside': False, 'all_outside': True, 'n_roots': 0}

    # Reverse coefficients for np.roots (highest power first)
    coeffs = kappa[::-1]
    roots = np.roots(coeffs)
    moduli = np.abs(roots)

    all_inside = np.all(moduli < 1.0)
    all_outside = np.all(moduli > 1.0)

    return {
        'all_inside': bool(all_inside),
        'all_outside': bool(all_outside),
        'n_roots': int(len(roots)),
        'max_modulus': float(np.max(moduli)),
        'min_modulus': float(np.min(moduli)),
        'mean_modulus': float(np.mean(moduli)),
    }


def test_cauchy_criterion(max_n: int = 10**12, sample_size: int = 20,
                          min_blocks: int = 5, seed: int = 42) -> dict:
    """
    Test the Cauchy root criterion on real trajectories.
    """
    rng = np.random.default_rng(seed)
    starts = rng.integers(3, max_n + 1, size=sample_size, dtype=np.int64)

    results = []
    for n0 in starts:
        lengths, log_values = compute_trajectory_with_blocks(int(n0), max_steps=50000)
        if len(lengths) < min_blocks + 3:
            continue

        kappa_sequence = []
        for i in range(min_blocks, len(lengths) + 1):
            _, _, kappa = construct_kinematic_tensor(lengths[:i], log_values[:i])
            if kappa is not None and kappa < 1e8:
                kappa_sequence.append(kappa)

        if len(kappa_sequence) >= 3:
            root_info = compute_root_distribution(np.array(kappa_sequence))
            root_info['start_n'] = int(n0)
            results.append(root_info)

    # Convergent trajectories should have roots outside the unit circle
    outside_count = sum(1 for r in results if r['all_outside'])
    inside_count = sum(1 for r in results if r['all_inside'])

    passed = outside_count > inside_count

    return {
        'test': 'cauchy_criterion',
        'passed': bool(passed),
        'trajectories_tested': int(len(results)),
        'roots_all_outside': int(outside_count),
        'roots_all_inside': int(inside_count),
        'mixed_or_edge': int(len(results) - outside_count - inside_count),
    }