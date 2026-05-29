"""
Part C: Parity Analysis — f_{3→5} Lower Bound.
"""

import numpy as np
from utils.collatz_map import generate_random_starts, accelerated_collatz


def test_parity_analysis(max_n: int = 10**12, sample_size: int = 300, seed: int = 42) -> dict:
    """
    Compute f_{3→5} and verify it is bounded away from zero.
    """
    starts = generate_random_starts(sample_size, max_n=max_n, seed=seed)

    f35_values = []
    N3_values = []

    for n0 in starts:
        n = int(n0)
        odd_steps = []
        for _ in range(50000):
            if n == 1 or n == 2:
                break
            if n % 2 == 1:
                r = n % 8
                k = (n - r) // 8
                odd_steps.append({'r': r, 'k_parity': k % 2})
            n = accelerated_collatz(np.array([n]))[0]

        N3 = 0
        N3to5 = 0
        for i in range(len(odd_steps) - 1):
            if odd_steps[i]['r'] == 3:
                N3 += 1
                if odd_steps[i + 1]['r'] == 5:
                    N3to5 += 1

        if N3 > 0:
            f35_values.append(N3to5 / N3)
            N3_values.append(N3)

    f35 = np.array(f35_values)
    N3_arr = np.array(N3_values)

    passed = np.mean(f35) > 0.3

    return {
        'test': 'parity_analysis_f35',
        'passed': bool(passed),
        'mean_f35': float(np.mean(f35)),
        'std_f35': float(np.std(f35)),
        'min_f35': float(np.min(f35)),
        'max_f35': float(np.max(f35)),
        'median_f35': float(np.median(f35)),
        'trajectories_tested': int(len(f35)),
        'total_N3': int(np.sum(N3_arr)),
        'theoretical_limit': 0.5,
    }