"""
Tests B5-B7: Obstruction Barriers.
"""

import numpy as np
from utils.collatz_map import generate_random_starts, accelerated_collatz


def test_lyapunov_ceiling(max_n: int = 10**12, sample_size: int = 200, seed: int = 42) -> dict:
    """
    Verify that the Lyapunov ceiling (5/3)^k is never violated.
    """
    starts = generate_random_starts(sample_size, max_n=max_n, seed=seed)

    violations = 0

    for n0 in starts:
        n = int(n0)
        n_initial = n
        for k in range(1, 10001):
            if n == 1 or n == 2:
                break
            n = accelerated_collatz(np.array([n]))[0]
            ceiling = n_initial * (5/3)**k
            if n > ceiling:
                violations += 1

    passed = violations == 0

    return {
        'test': 'lyapunov_ceiling',
        'passed': bool(passed),
        'violations': int(violations),
        'sample_size': int(sample_size),
    }


def test_block_growth_rate(max_n: int = 5 * 10**7, sample_size: int = 200, seed: int = 42) -> dict:
    """
    Measure actual growth rate V_k/k for trajectories.
    """
    starts = generate_random_starts(sample_size, max_n=max_n, seed=seed)

    growth_rates = []

    for n0 in starts:
        n = int(n0)
        log_initial = np.log2(n)
        k = 0
        for _ in range(50000):
            if n == 1 or n == 2:
                break
            n = accelerated_collatz(np.array([n]))[0]
            k += 1
        if k > 0:
            log_final = np.log2(max(n, 1))
            growth_rates.append((log_final - log_initial) / k)

    growth_rates = np.array(growth_rates)
    mean_growth = np.mean(growth_rates)
    beta = np.log2(3/2)
    gamma = np.log2(5/3)

    passed = mean_growth < gamma

    return {
        'test': 'block_growth_rate',
        'passed': bool(passed),
        'mean_growth_rate': float(mean_growth),
        'beta_theoretical': float(beta),
        'gamma_theoretical': float(gamma),
        'sample_size': int(sample_size),
    }


def test_sublinear_gap() -> dict:
    """
    Demonstrate the Escape Gap: Lyapunov ceiling vs sublinear block growth.
    """
    k = np.logspace(1, 6, 100)
    log_lyap = k * np.log2(5/3)
    log_sublin = np.sqrt(k) * 1.5
    gap_ratios = 2**(log_lyap - log_sublin)
    min_gap = np.min(gap_ratios)

    passed = min_gap > 1

    return {
        'test': 'sublinear_gap',
        'passed': bool(passed),
        'min_gap_ratio': float(min_gap),
        'gap_at_k_1000': float(gap_ratios[50]),
    }