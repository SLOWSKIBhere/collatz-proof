"""
Tier 3: Internal constants.
NOT exposed to command line — these are either theorems from the paper
or numerical method settings that users should not need to change.
"""

import numpy as np

# ======================================================================
# Backend selection
# ======================================================================
# Set to 'jax', 'numba', 'cupy', 'numpy', or None (auto-detect).
# Auto-detection order: JAX > Numba > CuPy > NumPy.
FORCE_BACKEND = None  # None = auto-detect

# ======================================================================
# Theorem constants (from proven results)
# ======================================================================
GAMMA_THEORETICAL = np.log2(5/3)          # ≈ 0.737 — Theorem 11 (Lyapunov)
F_CRIT = 0.292                             # ISCO — critical brake frequency
C_LOG_BOUND = 10                           # Constant in Lemma 6: m ≤ log₂(n) + C

# ======================================================================
# Numerical methods
# ======================================================================
# Chernoff
CHERNOFF_N = 50                            # Number of Chernoff iterations
CHERNOFF_T_MAX = 100                       # Upper integration limit
CHERNOFF_NUM_T = 200                       # Exponential grid points
CHERNOFF_Z = 1.5 + 0.0j                    # Resolvent evaluation point

# FEAST
FEAST_POINTS = 16                          # Quadrature points
FEAST_VECTORS = 12                         # Subspace vectors
FEAST_ITERS = 2                            # Refinement iterations

# Boundary Resolvent
BOUNDARY_NUM_THETA = 100                   # Points on unit circle

# Closure
MAX_STEPS_CLOSURE = 50                     # Max closure steps
DENSE_LIMIT = 2000                         # Threshold for dense operations
SPARSE_LIMIT = 50000                       # Threshold for sparse operations

# Trajectories
MAX_STEPS_TRAJECTORY = 5000                # Max trajectory steps
BURN_IN = 20                               # Burn-in for asymptotic tests

# Tensor analysis
MIN_BLOCKS_DEFAULT = 5                     # Minimum blocks for SVD
SIM_CV_DEFAULT = 0.85                      # CV for divergent simulation
SIM_K_VALUES = [100, 1000, 10000, 100000]  # k values for simulated divergence