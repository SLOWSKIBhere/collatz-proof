"""
Optimised Collatz functions with multi-backend support.

Supports (in order of priority):
  1. JAX (GPU + JIT via jit/vmap)
  2. Numba (CPU JIT via njit)
  3. CuPy (GPU without JIT)
  4. NumPy (pure Python fallback)

Selection is automatic based on available libraries.
Set FORCE_BACKEND in config to override.
"""

import numpy as np
from config import FORCE_BACKEND

# ======================================================================
# Backend detection
# ======================================================================

_available = {
    'jax': False,
    'numba': False,
    'cupy': False,
    'numpy': True,
}
_backend = 'numpy'

try:
    import jax.numpy as jnp
    from jax import jit, vmap, lax
    _available['jax'] = True
except ImportError:
    pass

try:
    from numba import njit, prange
    _available['numba'] = True
except ImportError:
    pass

try:
    import cupy as cp
    _available['cupy'] = True
except ImportError:
    pass

if FORCE_BACKEND and FORCE_BACKEND in _available and _available[FORCE_BACKEND]:
    _backend = FORCE_BACKEND
elif _available['jax']:
    import os
    os.environ.setdefault('JAX_ENABLE_X64', 'True')
    import jax
    jax.config.update('jax_enable_x64', True)
    _backend = 'jax'
elif _available['numba']:
    _backend = 'numba'
elif _available['cupy']:
    _backend = 'cupy'
else:
    _backend = 'numpy'

print(f"[collatz_map] Using backend: {_backend}")


# ======================================================================
# Core functions — NumPy (always available as fallback)
# ======================================================================

def _accelerated_collatz_numpy(n: np.ndarray) -> np.ndarray:
    """NumPy implementation (vectorised)."""
    n = np.asarray(n, dtype=np.int64)
    result = np.empty_like(n)
    even_mask = (n % 2 == 0)
    result[even_mask] = n[even_mask] // 2
    result[~even_mask] = (3 * n[~even_mask] + 1) // 2
    return result


def _compute_trajectory_numpy(n0: int, max_steps: int = 5000) -> np.ndarray:
    """NumPy trajectory (loop)."""
    trajectory = [n0]
    n = n0
    for _ in range(max_steps):
        if n == 1 or n == 2:
            break
        n = n // 2 if n % 2 == 0 else (3 * n + 1) // 2
        trajectory.append(n)
    return np.array(trajectory, dtype=np.int64)


def _nu2_numpy(x: np.ndarray) -> np.ndarray:
    """NumPy 2-adic valuation (vectorised loop)."""
    x = np.asarray(x, dtype=np.int64)
    result = np.zeros_like(x)
    mask = x > 0
    while np.any(mask):
        divisible = (x[mask] % 2 == 0)
        result[mask] += divisible.astype(np.int64)
        x[mask] = np.where(divisible, x[mask] // 2, x[mask])
        mask = mask & divisible
    return result


# ======================================================================
# JAX backend
# ======================================================================

if _backend == 'jax':

    @jit
    def _collatz_step_jax(n):
        """Single Collatz step, JIT-compiled."""
        return jnp.where(n % 2 == 0, n // 2, (3 * n + 1) // 2)

    _accelerated_collatz_vmap = vmap(_collatz_step_jax)

    def _accelerated_collatz_jax(n: np.ndarray) -> np.ndarray:
        """JAX implementation with vmap."""
        n_jax = jnp.asarray(n, dtype=jnp.int64)
        result = _accelerated_collatz_vmap(n_jax)
        return np.asarray(result, dtype=np.int64)

    def _compute_trajectory_jax(n0: int, max_steps: int = 5000) -> np.ndarray:
        """JAX trajectory via lax.scan."""
        def body_fn(n, _):
            nxt = jnp.where(n % 2 == 0, n // 2, (3 * n + 1) // 2)
            return nxt, nxt

        init = jnp.asarray(n0, dtype=jnp.int64)
        _, trajectory = lax.scan(body_fn, init, None, length=max_steps)
        trajectory = jnp.concatenate([init[None], trajectory])
        mask = (trajectory == 1) | (trajectory == 2)
        idx = jnp.argmax(mask)
        trajectory = jnp.where(jnp.arange(len(trajectory)) <= idx, trajectory, 0)
        return np.asarray(trajectory[trajectory > 0], dtype=np.int64)

    def _nu2_jax(x: np.ndarray) -> np.ndarray:
        """JAX 2-adic valuation via lax.while_loop."""
        x_jax = jnp.asarray(x, dtype=jnp.int64)
        result = jnp.zeros_like(x_jax)

        def cond_fn(state):
            _, _, mask = state
            return jnp.any(mask)

        def body_fn(state):
            x_arr, res, mask = state
            divisible = (x_arr % 2 == 0) & mask
            res = res + divisible.astype(jnp.int64)
            x_arr = jnp.where(divisible, x_arr // 2, x_arr)
            mask = mask & divisible
            return x_arr, res, mask

        _, result, _ = lax.while_loop(cond_fn, body_fn, (x_jax, result, x_jax > 0))
        return np.asarray(result, dtype=np.int64)

    accelerated_collatz = _accelerated_collatz_jax
    compute_trajectory = _compute_trajectory_jax
    nu2 = _nu2_jax


# ======================================================================
# Numba backend
# ======================================================================

elif _backend == 'numba':

    @njit(parallel=True)
    def _accelerated_collatz_numba(n: np.ndarray) -> np.ndarray:
        """Numba-parallelised Collatz step."""
        n_out = np.empty_like(n)
        for i in prange(len(n)):
            if n[i] % 2 == 0:
                n_out[i] = n[i] // 2
            else:
                n_out[i] = (3 * n[i] + 1) // 2
        return n_out

    @njit
    def _compute_trajectory_numba(n0: int, max_steps: int = 5000) -> np.ndarray:
        """Numba-JIT trajectory."""
        trajectory = np.zeros(max_steps + 1, dtype=np.int64)
        trajectory[0] = n0
        n = n0
        idx = 1
        for _ in range(max_steps):
            if n == 1 or n == 2:
                break
            n = n // 2 if n % 2 == 0 else (3 * n + 1) // 2
            trajectory[idx] = n
            idx += 1
        return trajectory[:idx]

    @njit(parallel=True)
    def _nu2_numba(x: np.ndarray) -> np.ndarray:
        """Numba-parallelised 2-adic valuation."""
        result = np.zeros_like(x)
        for i in prange(len(x)):
            val = x[i]
            cnt = 0
            while val > 0 and val % 2 == 0:
                val //= 2
                cnt += 1
            result[i] = cnt
        return result

    accelerated_collatz = _accelerated_collatz_numba
    compute_trajectory = _compute_trajectory_numba
    nu2 = _nu2_numba


# ======================================================================
# CuPy backend
# ======================================================================

elif _backend == 'cupy':

    def _accelerated_collatz_cupy(n: np.ndarray) -> np.ndarray:
        """CuPy GPU implementation."""
        n_cp = cp.asarray(n, dtype=cp.int64)
        even_mask = (n_cp % 2 == 0)
        result = cp.empty_like(n_cp)
        result[even_mask] = n_cp[even_mask] // 2
        result[~even_mask] = (3 * n_cp[~even_mask] + 1) // 2
        return cp.asnumpy(result)

    accelerated_collatz = _accelerated_collatz_cupy
    compute_trajectory = _compute_trajectory_numpy  # CuPy can't accelerate while-loop
    nu2 = _nu2_numpy  # CuPy can't accelerate while-loop easily


# ======================================================================
# NumPy fallback
# ======================================================================

else:
    accelerated_collatz = _accelerated_collatz_numpy
    compute_trajectory = _compute_trajectory_numpy
    nu2 = _nu2_numpy


# ======================================================================
# Common functions (backend-independent)
# ======================================================================

def compute_trajectory_with_blocks(n0: int, max_steps: int = 50000):
    """
    Compute trajectory and extract block structure.
    Uses compute_trajectory internally.
    """
    n = n0
    lengths = []
    log_values = []
    steps = 0
    prev_was_deep_brake = True
    block_len = 0

    while steps < max_steps:
        if prev_was_deep_brake:
            log_values.append(np.log2(max(n, 1)))
            block_len = 0
            prev_was_deep_brake = False

        if n == 1 or n == 2:
            break

        if n % 2 == 0:
            n //= 2
            steps += 1
        else:
            n = (3 * n + 1) // 2
            steps += 1
            block_len += 1
            if n % 8 == 5:
                lengths.append(block_len)
                prev_was_deep_brake = True
                block_len = 0

    if not prev_was_deep_brake and block_len > 0:
        lengths.append(block_len)

    return np.array(lengths, dtype=np.int64), np.array(log_values)


def is_deep_brake(n: np.ndarray) -> np.ndarray:
    """Check if n is a deep brake (≡ 5 mod 8 and odd)."""
    n = np.asarray(n, dtype=np.int64)
    return (n % 2 == 1) & (n % 8 == 5)


def generate_random_starts(n_samples: int, max_n: int = 10**12, seed: int = 42) -> np.ndarray:
    """Generate random odd starting values."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(3, max_n + 1, size=n_samples, dtype=np.int64)
    starts = starts | 1
    return starts