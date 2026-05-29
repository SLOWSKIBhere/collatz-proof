"""
Tests B2-B3: Modular Lemmas (Lemma 9.1, 9.4, 9.5).
"""

import numpy as np
from utils.collatz_map import generate_random_starts, accelerated_collatz
from config import C_LOG_BOUND


def test_lemma5_deep_brakes(max_n: int = 10**12, sample_size: int = 200, seed: int = 42) -> dict:
    """
    Lemma 9.4 (Inevitability of Deep Brakes).
    """
    starts = generate_random_starts(sample_size, max_n=max_n, seed=seed)

    all_have_deep_brakes = True
    deep_brake_counts = []

    for n0 in starts:
        n = int(n0)
        deep_brakes = 0
        for _ in range(50000):
            if n == 1 or n == 2:
                break
            n = accelerated_collatz(np.array([n]))[0]
            if n % 8 == 5 and n % 2 == 1:
                deep_brakes += 1
        deep_brake_counts.append(deep_brakes)
        if deep_brakes == 0 and n != 1 and n != 2:
            all_have_deep_brakes = False

    deep_brake_counts = np.array(deep_brake_counts)
    passed = bool(all_have_deep_brakes)

    return {
        'test': 'lemma5_deep_brakes',
        'passed': passed,
        'sample_size': int(sample_size),
        'mean_deep_brakes': float(np.mean(deep_brake_counts)),
        'min_deep_brakes': int(np.min(deep_brake_counts)),
        'max_deep_brakes': int(np.max(deep_brake_counts)),
        'trajectories_without_brakes': int(np.sum(deep_brake_counts == 0)),
    }


def test_lemma6_log_bound(max_n: int = 10**12, sample_size: int = 200, seed: int = 42) -> dict:
    """
    Lemma 9.5 (Logarithmic Bound on Mixed Blocks).
    """
    starts = generate_random_starts(sample_size, max_n=max_n, seed=seed)

    violations = 0
    block_lengths = []
    log_bounds = []

    for n0 in starts:
        n = int(n0)
        block_len = 0
        start_log = np.log2(max(n, 1))
        for _ in range(50000):
            if n == 1 or n == 2:
                break
            if n % 8 == 5 and n % 2 == 1 and block_len > 0:
                expected_max = start_log + C_LOG_BOUND
                if block_len > expected_max:
                    violations += 1
                block_lengths.append(block_len)
                log_bounds.append(expected_max)
                block_len = 0
                start_log = np.log2(max(n, 1))
            n = accelerated_collatz(np.array([n]))[0]
            if n % 2 == 1 and not (n % 8 == 5 and block_len > 0):
                block_len += 1

    block_lengths = np.array(block_lengths)
    passed = violations == 0 and len(block_lengths) > 0

    return {
        'test': 'lemma6_log_bound',
        'passed': bool(passed),
        'violations': int(violations),
        'total_blocks': int(len(block_lengths)),
        'mean_block_length': float(np.mean(block_lengths)) if len(block_lengths) > 0 else 0,
        'max_block_length': int(np.max(block_lengths)) if len(block_lengths) > 0 else 0,
        'C_used': C_LOG_BOUND,
    }