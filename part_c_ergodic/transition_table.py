"""
Part C: Full Transition Table for Odd Residues Modulo 8.
"""

import numpy as np
from utils.collatz_map import generate_random_starts, accelerated_collatz


def compute_transition_table(max_n: int = 10**12, sample_size: int = 500, seed: int = 42) -> dict:
    """
    Compute the complete transition table for odd residues modulo 8.
    """
    starts = generate_random_starts(sample_size, max_n=max_n, seed=seed)

    transitions = np.zeros((8, 8), dtype=np.int64)
    transitions_from = np.zeros(8, dtype=np.int64)
    k_parity_transitions = np.zeros((8, 2, 8), dtype=np.int64)

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

        for i in range(len(odd_steps) - 1):
            r_from = odd_steps[i]['r']
            r_to = odd_steps[i + 1]['r']
            kp = odd_steps[i]['k_parity']
            transitions[r_from, r_to] += 1
            transitions_from[r_from] += 1
            k_parity_transitions[r_from, kp, r_to] += 1

    prob_matrix = np.zeros((8, 8))
    for r in [1, 3, 5, 7]:
        total = transitions_from[r]
        if total > 0:
            prob_matrix[r] = transitions[r] / total

    return {
        'test': 'transition_table',
        'transitions': transitions.tolist(),
        'probabilities': prob_matrix.tolist(),
        'total_odd_steps': int(transitions_from.sum()),
    }