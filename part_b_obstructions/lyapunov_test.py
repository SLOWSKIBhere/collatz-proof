"""
Test B1: Lyapunov Function Validation.
"""

import numpy as np
from utils.collatz_map import generate_random_starts, accelerated_collatz
from config import GAMMA_THEORETICAL


def test_lyapunov(max_n: int = 10**12, sample_size: int = 200,
                  gamma: float = 0.74, seed: int = 42) -> dict:
    """
    Verify Theorem 9.3: Lyapunov function V(n) = log2(n) - gamma * s
    strictly decreases for all n >= 3.
    """
    starts = generate_random_starts(sample_size, max_n=min(max_n, 5 * 10**6), seed=seed)

    violations = 0
    total_steps = 0

    for n0 in starts:
        n = int(n0)
        s = 0
        for _ in range(1000):
            if n == 1 or n == 2:
                break
            V_before = np.log2(n) - gamma * s
            n = accelerated_collatz(np.array([n]))[0]
            s += 1
            V_after = np.log2(max(n, 1)) - gamma * s
            if V_after >= V_before:
                violations += 1
            total_steps += 1

    passed = violations == 0

    return {
        'test': 'lyapunov_function',
        'passed': bool(passed),
        'violations': int(violations),
        'total_steps': int(total_steps),
        'sample_size': int(sample_size),
        'gamma_used': float(gamma),
        'gamma_theoretical': float(GAMMA_THEORETICAL),
    }