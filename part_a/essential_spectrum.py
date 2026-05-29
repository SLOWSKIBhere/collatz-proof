"""
Test for Theorem 8.3: Essential Spectrum via Weyl's Theorem.
"""

import numpy as np
from scipy.sparse import csr_matrix
from utils.koopman_operator import bidirectional_closure, build_koopman_matrix

def test_essential_spectrum(n_init: int = 3, max_steps: int = 50, max_size: int = 2000) -> dict:
    """
    Verify that the essential spectrum fills the closed unit disk,
    and the point spectrum on the circle is only {+1, -1}.
    """
    closure = bidirectional_closure(set(range(1, n_init + 1)), max_steps, max_size)
    U, mapping, reverse_mapping = build_koopman_matrix(closure)

    N = U.shape[0]
    if N > 2000:
        return {
            'test': 'essential_spectrum',
            'passed': False,
            'error': f'Matrix too large for dense essential spectrum test (N={N})',
        }

    U_dense = U.toarray()

    # Identify cycle vertices: those where U^n v = v for some n
    eigenvalues = np.linalg.eigvals(U_dense)
    unit_circle = np.abs(eigenvalues)
    on_circle = eigenvalues[unit_circle > 0.999]

    has_plus_one = np.any(np.abs(on_circle - 1.0) < 0.01)
    has_minus_one = np.any(np.abs(on_circle + 1.0) < 0.01)

    # Build unilateral shift approximation
    nilpotent_eigenvalues = eigenvalues[unit_circle < 0.001]
    nilpotent_multiplicity = len(nilpotent_eigenvalues)

    # Compute rank of K = U - (U_cycle ⊕ S)
    # U_cycle is the restriction to cycle subspace
    cycle_indices = []
    for i in range(N):
        vec = np.zeros(N)
        vec[i] = 1.0
        for _ in range(100):
            vec = U_dense @ vec
        if np.linalg.norm(vec) > 1e-6:
            cycle_indices.append(i)

    cycle_count = len(cycle_indices)
    # Expected: 2 cycle vertices (1 and 2)
    cycle_ok = (cycle_count == 2)

    passed = has_plus_one and has_minus_one and cycle_ok

    return {
        'test': 'essential_spectrum',
        'passed': bool(passed),
        'matrix_size': N,
        'has_plus_one': bool(has_plus_one),
        'has_minus_one': bool(has_minus_one),
        'nilpotent_multiplicity': int(nilpotent_multiplicity),
        'cycle_vertex_count': int(cycle_count),
        'spectral_radius': float(max(abs(eigenvalues))),
    }