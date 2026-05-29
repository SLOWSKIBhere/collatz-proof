"""
Part D: Tensor Evolution — Tracking κ as trajectory progresses.
"""

import numpy as np
from utils.collatz_map import compute_trajectory_with_blocks
from utils.tensor_utils import construct_kinematic_tensor

def simulate_divergent_trajectory(total_steps: int = 50000, f_type: str = 'sqrt') -> tuple:
    """
    Generate a simulated divergent trajectory for comparison.
    """
    if f_type == 'sqrt':
        t = int(np.sqrt(total_steps))
    elif f_type == 'log':
        t = max(1, int(np.log2(total_steps)))
    else:
        t = max(1, int(np.log(total_steps)))

    lengths = np.random.exponential(total_steps / t, size=t).astype(int)
    lengths = np.clip(lengths, 1, None)
    actual_steps = np.sum(lengths)
    if actual_steps == 0:
        return np.array([1]), np.array([10.0])
    scale = total_steps / actual_steps
    lengths = (lengths * scale).astype(int)
    lengths = np.clip(lengths, 1, None)

    V = 10.0
    log_values = []
    for m in lengths:
        log_values.append(V)
        V += 0.585 * m - 1.415 + np.random.normal(0, 0.1)

    return lengths, np.array(log_values)


def compute_kappa_evolution(lengths: np.ndarray, log_values: np.ndarray,
                            min_blocks: int = 5) -> list:
    """Compute κ as blocks accumulate."""
    t = len(lengths)
    kappa_evo = []
    for i in range(min_blocks, t + 1):
        _, _, kappa = construct_kinematic_tensor(lengths[:i], log_values[:i])
        if kappa is not None and kappa < 1e8:
            kappa_evo.append(kappa)
    return kappa_evo


def test_tensor_evolution(max_n: int = 10**12, sample_size: int = 50,
                          min_blocks: int = 5, seed: int = 42) -> dict:
    """
    Track κ evolution for real convergent and simulated divergent trajectories.
    """
    # Real trajectories
    rng = np.random.default_rng(seed)
    starts = rng.integers(3, max_n + 1, size=sample_size, dtype=np.int64)

    real_kappa_trends = []
    for n0 in starts:
        lengths, log_values = compute_trajectory_with_blocks(int(n0), max_steps=50000)
        if len(lengths) >= min_blocks + 5:
            kappa_evo = compute_kappa_evolution(lengths, log_values, min_blocks)
            if len(kappa_evo) >= 3:
                trend = np.polyfit(range(len(kappa_evo)), kappa_evo, 1)[0]
                real_kappa_trends.append(trend)

    # Simulated divergent trajectories
    sim_kappa_trends = []
    for _ in range(sample_size):
        lengths, log_values = simulate_divergent_trajectory(
            total_steps=np.random.randint(10000, 100000), f_type='sqrt'
        )
        if len(lengths) >= min_blocks + 5:
            kappa_evo = compute_kappa_evolution(lengths, log_values, min_blocks)
            if len(kappa_evo) >= 3:
                trend = np.polyfit(range(len(kappa_evo)), kappa_evo, 1)[0]
                sim_kappa_trends.append(trend)

    real_trends = np.array(real_kappa_trends)
    sim_trends = np.array(sim_kappa_trends)

    passed = np.mean(real_trends) < 0 and np.mean(sim_trends) > 0

    return {
        'test': 'tensor_evolution',
        'passed': bool(passed),
        'real_mean_kappa_trend': float(np.mean(real_trends)) if len(real_trends) > 0 else 0,
        'real_std_kappa_trend': float(np.std(real_trends)) if len(real_trends) > 0 else 0,
        'sim_mean_kappa_trend': float(np.mean(sim_trends)) if len(sim_trends) > 0 else 0,
        'sim_std_kappa_trend': float(np.std(sim_trends)) if len(sim_trends) > 0 else 0,
        'real_trajectories': int(len(real_trends)),
        'sim_trajectories': int(len(sim_trends)),
    }