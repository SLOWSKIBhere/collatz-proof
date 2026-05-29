"""
Test for Theorem 3.1: Spectral Structure of the Koopman Operator.

Verifies that for any initial set {1..N_init}, the Koopman operator
has exactly two non-zero eigenvalues: +1 and -1.
"""

import numpy as np
from utils.koopman_operator import bidirectional_closure, build_koopman_matrix


def test_structural_theorem(
    n_values: list = None,
    max_steps: int = 10,
    max_size: int = 50000,
) -> dict:
    """
    Verify the structural theorem for multiple N_init values.

    Parameters
    ----------
    n_values : list
        N_init values to test (default: [3, 5, 7, 10, 15, 20]).
    max_steps : int
        Maximum closure steps.
    max_size : int
        Maximum closure size.
    """
    if n_values is None:
        n_values = [3, 5, 7, 10, 15, 20]

    results = []
    all_passed = True
    all_eigenvalues = []

    for N_init in n_values:
        closure = bidirectional_closure(
            set(range(1, N_init + 1)), max_steps, max_size
        )

        if len(closure) > max_size:
            results.append({
                'N_init': N_init,
                'size': len(closure),
                'passed': None,
                'note': 'Too large for dense matrix',
            })
            continue

        U, mapping, reverse_mapping = build_koopman_matrix(closure)
        M = U.shape[0]
        U_dense = U.toarray()

        eigenvalues = np.linalg.eigvals(U_dense)
        all_eigenvalues.extend(eigenvalues.tolist())

        nonzero = eigenvalues[np.abs(eigenvalues) > 1e-10]
        n_nonzero = len(nonzero)
        has_plus_one = np.any(np.abs(eigenvalues - 1.0) < 1e-10)
        has_minus_one = np.any(np.abs(eigenvalues + 1.0) < 1e-10)
        rank = np.linalg.matrix_rank(U_dense)

        passed = (n_nonzero == 2) and has_plus_one and has_minus_one
        if not passed:
            all_passed = False

        results.append({
            'N_init': N_init,
            'size': M,
            'rank': rank,
            'n_nonzero': n_nonzero,
            'has_plus_one': bool(has_plus_one),
            'has_minus_one': bool(has_minus_one),
            'passed': bool(passed),
            'non_zero_eigenvalues': [complex(z) for z in nonzero],
        })

    return {
        'test': 'structural_theorem',
        'passed': bool(all_passed),
        'results': results,
        'total_tests': len(results),
        'passed_count': sum(1 for r in results if r.get('passed')),
        'eigenvalues': all_eigenvalues,
        'conclusion': (
            'Exactly 2 non-zero eigenvalues (+1 and -1) for all N_init.'
            if all_passed else
            'FAILED: unexpected eigenvalues detected.'
        ),
    }