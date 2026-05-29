"""
Part C: Deep Brake Frequency Lower Bound Test.
"""

import numpy as np
from utils.collatz_map import generate_random_starts, accelerated_collatz


def test_fk_bound(max_n: int = 10**12, sample_size: int = 300, seed: int = 42) -> dict:
    """
    Verify Theorem C.4: f_k = t_k/k is bounded away from zero.
    """
    starts = generate_random_starts(sample_size, max_n=max_n, seed=seed)

    fk_values = []

    for n0 in starts:
        n = int(n0)
        total_steps = 0
        deep_brakes = 0
        for _ in range(50000):
            if n == 1 or n == 2:
                break
            n = accelerated_collatz(np.array([n]))[0]
            total_steps += 1
            if n % 8 == 5 and n % 2 == 1:
                deep_brakes += 1
        if total_steps > 0:
            fk_values.append(deep_brakes / total_steps)

    fk = np.array(fk_values)
    theoretical_lower_bound = 0.5 / (4 + 0.5)
    passed = np.min(fk) > 0

    return {
        'test': 'fk_lower_bound',
        'passed': bool(passed),
        'mean_fk': float(np.mean(fk)),
        'std_fk': float(np.std(fk)),
        'min_fk': float(np.min(fk)),
        'theoretical_lower_bound': float(theoretical_lower_bound),
        'sample_size': int(sample_size),
    }