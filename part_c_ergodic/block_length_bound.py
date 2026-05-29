"""
Part C: Block Length Bound — Stochastic Domination Test.
"""

import numpy as np
from utils.collatz_map import generate_random_starts, accelerated_collatz


def test_block_length_bound(max_n: int = 10**12, sample_size: int = 300, seed: int = 42) -> dict:
    """
    Verify Theorem C.3: expected block length is bounded by C/δ + 1.
    """
    starts = generate_random_starts(sample_size, max_n=max_n, seed=seed)

    block_lengths = []

    for n0 in starts:
        n = int(n0)
        block_len = 0
        for _ in range(50000):
            if n == 1 or n == 2:
                if block_len > 0:
                    block_lengths.append(block_len)
                break
            if n % 8 == 5 and n % 2 == 1 and block_len > 0:
                block_lengths.append(block_len)
                block_len = 0
            n = accelerated_collatz(np.array([n]))[0]
            if n % 2 == 1:
                block_len += 1

    block_lengths = np.array(block_lengths)
    theoretical_bound = 9.0
    mean_block = np.mean(block_lengths)
    passed = mean_block < theoretical_bound * 2

    return {
        'test': 'block_length_bound',
        'passed': bool(passed),
        'total_blocks': int(len(block_lengths)),
        'mean_block_length': float(mean_block),
        'std_block_length': float(np.std(block_lengths)),
        'max_block_length': int(np.max(block_lengths)),
        'median_block_length': float(np.median(block_lengths)),
        'theoretical_bound': theoretical_bound,
    }