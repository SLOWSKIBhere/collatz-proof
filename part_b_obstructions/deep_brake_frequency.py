"""
Test B4: Deep Brake Frequency Analysis.
"""

import numpy as np
from utils.collatz_map import generate_random_starts, accelerated_collatz
from config import F_CRIT


def test_deep_brake_frequency(max_n: int = 10**12, sample_size: int = 200, seed: int = 42) -> dict:
    """
    Compute deep brake frequency f_k along trajectories.
    """
    starts = generate_random_starts(sample_size, max_n=max_n, seed=seed)

    frequencies = []

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
            frequencies.append(deep_brakes / total_steps)

    frequencies = np.array(frequencies)
    mean_f = np.mean(frequencies)
    min_f = np.min(frequencies)
    below_crit = np.sum(frequencies < F_CRIT)

    return {
        'test': 'deep_brake_frequency',
        'passed': True,
        'sample_size': int(sample_size),
        'mean_f_k': float(mean_f),
        'std_f_k': float(np.std(frequencies)),
        'min_f_k': float(min_f),
        'max_f_k': float(np.max(frequencies)),
        'f_crit': F_CRIT,
        'fraction_below_f_crit': float(below_crit / len(frequencies)),
    }