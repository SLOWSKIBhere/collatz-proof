"""
Koopman operator construction for the accelerated Collatz map.
Optimised: vectorised operations, set comprehensions, minimal Python loops.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


def bidirectional_closure(initial_set: set, max_steps: int = 50, max_size: int = 50000) -> set:
    """
    Compute the bidirectional closure of a set under the accelerated Collatz map.
    """
    closure = set(initial_set)
    queue_forward = set(initial_set)
    queue_backward = set(initial_set)

    for _ in range(max_steps):
        if len(closure) >= max_size:
            break

        # Forward expansion (vectorised logic via set comprehension)
        new_forward = {
            n // 2 if n % 2 == 0 else (3 * n + 1) // 2
            for n in queue_forward
        } - closure
        closure.update(new_forward)
        queue_forward = new_forward

        if len(closure) >= max_size:
            break

        # Backward expansion
        new_backward = set()
        for n in queue_backward:
            new_backward.add(2 * n)
            if (2 * n - 1) % 3 == 0:
                pre_odd = (2 * n - 1) // 3
                if pre_odd % 2 == 1:
                    new_backward.add(pre_odd)
        new_backward -= closure
        closure.update(new_backward)
        queue_backward = new_backward

        if not queue_forward and not queue_backward:
            break

    return closure


def compute_closure_safe(start_numbers, max_steps=100, max_size=50000):
    """
    Bidirectional closure matching the original Jupyter implementation.
    Optimised with set comprehensions.
    """
    numbers = set(start_numbers)
    queue_forward = set(start_numbers)
    queue_backward = set(start_numbers)

    for _ in range(max_steps):
        if len(numbers) >= max_size:
            break

        next_forward = {
            n // 2 if n % 2 == 0 else (3 * n + 1) // 2
            for n in queue_forward
        } - numbers
        numbers.update(next_forward)
        queue_backward.update(next_forward)
        queue_forward = next_forward

        if len(numbers) >= max_size:
            break

        next_backward = set()
        for n in queue_backward:
            next_backward.add(2 * n)
            if (2 * n - 1) % 3 == 0:
                m = (2 * n - 1) // 3
                if m % 2 == 1 and m > 0:
                    next_backward.add(m)
        next_backward -= numbers
        numbers.update(next_backward)
        queue_forward.update(next_backward)
        queue_backward = next_backward

        if not queue_forward and not queue_backward:
            break

    sorted_numbers = sorted(numbers)
    return sorted_numbers, len(sorted_numbers)


def build_koopman_matrix(closure) -> tuple:
    """
    Build the Koopman matrix U for the given closed set.
    Vectorised where possible.
    """
    elements = sorted(closure) if isinstance(closure, set) else list(closure)
    N = len(elements)

    # Build mapping via dictionary comprehension
    mapping = {n: i for i, n in enumerate(elements)}
    reverse_mapping = {i: n for n, i in mapping.items()}

    # Pre-compute next values for all elements
    elements_arr = np.array(elements, dtype=np.int64)
    even_mask = elements_arr % 2 == 0
    nxt_arr = np.empty(N, dtype=np.int64)
    nxt_arr[even_mask] = elements_arr[even_mask] // 2
    nxt_arr[~even_mask] = (3 * elements_arr[~even_mask] + 1) // 2

    # Get column indices via dictionary lookup
    col_indices = np.array([mapping.get(nxt, -1) for nxt in nxt_arr], dtype=np.int64)
    valid = col_indices >= 0

    # Build sparse matrix directly in CSR format
    row_ind = col_indices[valid]
    col_ind = np.arange(N, dtype=np.int64)[valid]
    data = np.ones(valid.sum(), dtype=np.float64)

    U = csr_matrix((data, (row_ind, col_ind)), shape=(N, N))
    return U, mapping, reverse_mapping


def compute_spectrum(U: csr_matrix, k: int = 30) -> dict:
    """
    Compute spectral properties of the Koopman matrix.
    """
    N = U.shape[0]

    if N <= 2000:
        U_dense = U.toarray()
        eigenvalues = np.linalg.eigvals(U_dense)
    else:
        # Arnoldi iteration for largest eigenvalues (vectorised)
        eigenvalues = np.zeros(min(N, 100), dtype=complex)
        for i in range(min(N, 100)):
            v = np.random.randn(N)
            v /= np.linalg.norm(v)
            for _ in range(50):
                v = U.T @ (U @ v)
                v /= np.linalg.norm(v)
            eigenvalues[i] = (v @ (U @ v)) / (v @ v)

    # Trace powers — keep as loop (small k, sparse matrix)
    trace_powers = []
    Uk = U.copy()
    for i in range(1, k + 1):
        trace_powers.append(float(Uk.diagonal().sum()))
        if i < k:
            Uk = Uk @ U

    norm_val = float(np.sqrt((U.T @ U).diagonal().max()))
    spectral_radius = float(max(abs(eigenvalues)))

    return {
        'eigenvalues': eigenvalues,
        'trace_powers': trace_powers,
        'norm': norm_val,
        'spectral_radius': spectral_radius,
    }