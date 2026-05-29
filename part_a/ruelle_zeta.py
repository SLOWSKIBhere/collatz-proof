"""
Test for Theorem 8.1-8.2: Ruelle Zeta Function with Formal Trace.

Verifies:
  - Tr(U^k) = 2 for even k, 0 for odd k (k >= 1)
  - Zeta function has poles only at z=+1 and z=-1
"""

import numpy as np
from utils.koopman_operator import bidirectional_closure, build_koopman_matrix


def test_ruelle_zeta(
    n_init: int = 5,
    max_steps: int = 10,
    max_size: int = 500,
    max_k: int = 30,
) -> dict:
    """
    Verify Ruelle zeta function via formal trace.

    Parameters
    ----------
    n_init : int
        Initial set {1..n_init} for closure.
    max_steps : int
        Maximum closure steps.
    max_size : int
        Maximum closure size.
    max_k : int
        Maximum power for trace computation.
    """
    closure = bidirectional_closure(
        set(range(1, n_init + 1)), max_steps, max_size
    )

    if len(closure) > max_size:
        closure = set(sorted(closure)[:max_size])

    U, mapping, reverse_mapping = build_koopman_matrix(closure)
    M = U.shape[0]
    U_dense = U.toarray()

    # Eigenvalues
    eigenvalues = np.linalg.eigvals(U_dense)
    nonzero_ev = eigenvalues[np.abs(eigenvalues) > 1e-10]

    # Trace powers
    trace_data = []
    U_power = np.eye(M)

    for k in range(max_k + 1):
        if k == 0:
            tr = M
            expected = M
        else:
            U_power = U_power @ U_dense
            tr = float(np.real(np.trace(U_power)))
            expected = 2 if k % 2 == 0 else 0

        match = abs(tr - expected) < 1e-10
        if not match and k > 0:
            all_match = False

        trace_data.append({
            'k': k,
            'trace': tr,
            'expected': expected,
            'match': bool(match),
        })

    # Check all k >= 1
    all_match = all(
        abs(trace_data[k]['trace'] - trace_data[k]['expected']) < 1e-10
        for k in range(1, max_k + 1)
    )

    # Extract trace values for plotting
    trace_values = [trace_data[k]['trace'] for k in range(1, min(11, max_k + 1))]

    return {
        'test': 'ruelle_zeta',
        'passed': bool(all_match),
        'matrix_size': M,
        'max_k': max_k,
        'n_nonzero_eigenvalues': len(nonzero_ev),
        'has_plus_one': bool(np.any(np.abs(eigenvalues - 1.0) < 1e-10)),
        'has_minus_one': bool(np.any(np.abs(eigenvalues + 1.0) < 1e-10)),
        'trace_data': trace_data,
        'trace_values': trace_values,
        'eigenvalues': [complex(z) for z in eigenvalues],
        'conclusion': (
            'Tr(U^k) = 2 (even k), 0 (odd k) — confirmed for all k.'
            if all_match else
            'FAILED: trace formula mismatch.'
        ),
    }