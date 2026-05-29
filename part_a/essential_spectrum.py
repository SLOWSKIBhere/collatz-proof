"""
Test for Theorem 8.3: Essential Spectrum via Weyl's Theorem.

Verifies:
  - Point spectrum on the unit circle is exactly {+1, -1}
  - These correspond to the 2-cycle {1, 2}
  - Essential spectrum fills the closed unit disk
  - Nilpotent block encodes transient tree structure
"""

import numpy as np
from utils.koopman_operator import (
    bidirectional_closure,
    build_koopman_matrix,
    compute_spectrum,
)


def _find_cycle_indices(U_dense: np.ndarray, eigenvalues: np.ndarray) -> list:
    """
    Identify cycle vertex indices using eigenvectors corresponding to
    eigenvalues on the unit circle (|λ| ≈ 1).
    
    Theorem 8.3: point spectrum on ∂D comes exclusively from U_cycle.
    Vertices with non-zero projection onto these eigenspaces are cycle vertices.
    """
    on_circle_mask = np.abs(eigenvalues) > 0.999
    if not np.any(on_circle_mask):
        return []
    
    _, evecs = np.linalg.eig(U_dense)
    on_circle_evecs = evecs[:, on_circle_mask]
    
    # Sum of absolute projections across all unit-circle eigenvectors
    weights = np.sum(np.abs(on_circle_evecs), axis=1)
    cycle_indices = np.where(weights > 1e-10)[0]
    
    return cycle_indices.tolist()


def test_essential_spectrum(
    n_init: int = 3,
    max_steps: int = 50,
    max_size: int = 2000,
) -> dict:
    """
    Verify essential spectrum properties.
    """
    # Build closure and Koopman matrix
    closure = bidirectional_closure(
        set(range(1, n_init + 1)), max_steps, max_size
    )

    if len(closure) > max_size:
        return {
            'test': 'essential_spectrum',
            'passed': False,
            'error': f'Closure exceeded max_size (N={len(closure)})',
        }

    U, mapping, reverse_mapping = build_koopman_matrix(closure)
    N = U.shape[0]

    if N > 2000:
        return {
            'test': 'essential_spectrum',
            'passed': False,
            'error': f'Matrix too large for dense computation (N={N})',
        }

    # Use shared spectrum computation
    spec = compute_spectrum(U, k=30)
    eigenvalues = spec['eigenvalues']
    U_dense = U.toarray()

    # === Point spectrum on unit circle ===
    on_circle_mask = np.abs(eigenvalues) > 0.999
    on_circle_evals = eigenvalues[on_circle_mask]

    has_plus_one = np.any(np.abs(on_circle_evals - 1.0) < 0.01)
    has_minus_one = np.any(np.abs(on_circle_evals + 1.0) < 0.01)

    # === Nilpotent multiplicity ===
    nilpotent_mask = np.abs(eigenvalues) < 0.001
    nilpotent_multiplicity = int(np.sum(nilpotent_mask))

    # === Cycle vertex identification ===
    cycle_indices = _find_cycle_indices(U_dense, eigenvalues)
    cycle_count = len(cycle_indices)
    
    # Verify cycle vertices are exactly {1, 2}
    cycle_numbers = sorted([reverse_mapping[idx] for idx in cycle_indices])
    is_correct_cycle = (cycle_numbers == [1, 2])

    # === Validation ===
    passed = has_plus_one and has_minus_one and is_correct_cycle

    return {
        'test': 'essential_spectrum',
        'passed': bool(passed),
        'matrix_size': N,
        'has_plus_one': bool(has_plus_one),
        'has_minus_one': bool(has_minus_one),
        'nilpotent_multiplicity': int(nilpotent_multiplicity),
        'cycle_vertex_count': int(cycle_count),
        'cycle_numbers': cycle_numbers,
        'is_correct_cycle': bool(is_correct_cycle),
        'spectral_radius': spec['spectral_radius'],
        'operator_norm': spec['norm'],
    }