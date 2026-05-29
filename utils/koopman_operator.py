"""
Koopman operator construction for the accelerated Collatz map.
Optimised: vectorised operations, minimal Python loops, unified backend.
"""

import numpy as np
from scipy.sparse import csr_matrix
from utils.collatz_map import accelerated_collatz


def bidirectional_closure(initial_set: set, max_steps: int = 50, max_size: int = 50000) -> set:
    """
    Compute the bidirectional closure of a set under the accelerated Collatz map.
    Uses vectorised accelerated_collatz for forward expansion.
    """
    closure = set(initial_set)
    queue_forward = set(initial_set)
    queue_backward = set(initial_set)

    for _ in range(max_steps):
        if len(closure) >= max_size:
            break

        # Forward expansion — vectorised
        if queue_forward:
            fwd_arr = np.fromiter(queue_forward, dtype=np.int64, count=len(queue_forward))
            nxt_arr = accelerated_collatz(fwd_arr)
            new_forward = set(nxt_arr.tolist()) - closure
            closure.update(new_forward)
            queue_forward = new_forward
        else:
            new_forward = set()

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
    Uses vectorised accelerated_collatz for forward expansion.
    """
    numbers = set(start_numbers)
    queue_forward = set(start_numbers)
    queue_backward = set(start_numbers)

    for _ in range(max_steps):
        if len(numbers) >= max_size:
            break

        # Forward — vectorised
        if queue_forward:
            fwd_arr = np.fromiter(queue_forward, dtype=np.int64, count=len(queue_forward))
            nxt_fwd = accelerated_collatz(fwd_arr)
            next_forward = set(nxt_fwd.tolist()) - numbers
        else:
            next_forward = set()

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


def _build_lookup(elements: list, mapping: dict) -> np.ndarray:
    """
    Build a vectorised lookup array for O(1) mapping of Collatz numbers
    to matrix indices.
    """
    max_val = max(elements)
    lookup = np.full(max_val + 1, -1, dtype=np.int64)
    for num, idx in mapping.items():
        lookup[num] = idx
    return lookup


def build_koopman_matrix(closure) -> tuple:
    """
    Build the Koopman matrix U for the given closed set.
    Fully vectorised using unified backend and array lookup.
    """
    elements = sorted(closure) if isinstance(closure, set) else list(closure)
    N = len(elements)

    # Build mapping
    mapping = {n: i for i, n in enumerate(elements)}
    reverse_mapping = {i: n for n, i in mapping.items()}

    # Compute all next values via unified backend
    elements_arr = np.array(elements, dtype=np.int64)
    nxt_arr = accelerated_collatz(elements_arr)

    # Vectorised lookup: build lookup array, mask invalid
    lookup = _build_lookup(elements, mapping)
    in_range = nxt_arr <= lookup.size - 1
    col_indices = np.where(in_range, lookup[nxt_arr], -1)
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
        eigenvalues = np.zeros(min(N, 100), dtype=complex)
        for i in range(min(N, 100)):
            v = np.random.randn(N)
            v /= np.linalg.norm(v)
            for _ in range(50):
                v = U.T @ (U @ v)
                v /= np.linalg.norm(v)
            eigenvalues[i] = (v @ (U @ v)) / (v @ v)

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