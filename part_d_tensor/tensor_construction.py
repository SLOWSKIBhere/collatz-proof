"""
Part D: Kinematic Tensor Construction and SVD.
"""

import numpy as np
from utils.collatz_map import compute_trajectory_with_blocks
from utils.tensor_utils import construct_kinematic_tensor

def test_tensor_construction(max_n: int = 10**12, sample_size: int = 200,
                             min_blocks: int = 5, seed: int = 42) -> dict:
    """
    Build kinematic tensors for random trajectories and measure κ.
    """
    rng = np.random.default_rng(seed)
    starts = rng.integers(3, max_n + 1, size=sample_size, dtype=np.int64)

    kappa_values = []
    sigma_ratios = []

    for n0 in starts:
        lengths, log_values = compute_trajectory_with_blocks(int(n0), max_steps=50000)
        if len(lengths) >= min_blocks:
            s1, s2, kappa = construct_kinematic_tensor(lengths, log_values)
            if kappa is not None and kappa < 1e8:
                kappa_values.append(kappa)
                sigma_ratios.append(s1 / max(s2, 1e-10))

    kappa_arr = np.array(kappa_values)
    passed = len(kappa_arr) > 10

    return {
        'test': 'tensor_construction',
        'passed': bool(passed),
        'trajectories_with_blocks': int(len(kappa_arr)),
        'mean_kappa': float(np.mean(kappa_arr)) if len(kappa_arr) > 0 else 0,
        'std_kappa': float(np.std(kappa_arr)) if len(kappa_arr) > 0 else 0,
        'min_kappa': float(np.min(kappa_arr)) if len(kappa_arr) > 0 else 0,
        'max_kappa': float(np.max(kappa_arr)) if len(kappa_arr) > 0 else 0,
        'min_blocks_required': min_blocks,
    }