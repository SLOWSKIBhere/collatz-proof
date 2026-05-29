"""
Command-line argument parser.
Tier 1: general parameters (exposed in CLI).
Tier 2: test-specific parameters (grouped in argparse).
"""

import argparse
import os


def get_full_parser(description: str) -> argparse.ArgumentParser:
    """
    Build the full argument parser with grouped parameters.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # === Tier 1: General ===
    general = parser.add_argument_group('General Configuration')
    general.add_argument('--max-n', type=int, default=10**12,
                         help='Upper bound for random sampling (default: 10^12)')
    general.add_argument('--sample-size', type=int, default=200,
                         help='Number of random trajectories (default: 200)')
    general.add_argument('--seed', type=int, default=42,
                         help='Random seed for reproducibility (default: 42)')
    general.add_argument('--numbers', type=str, default=None,
                         help="Comma-separated specific numbers, e.g. '27,31,255'")

    # === Tier 1: Test selection ===
    test_sel = parser.add_argument_group('Test Selection')
    test_sel.add_argument('--test', type=str, default='all',
                          choices=[
                              'all',
                              'part_a', 'structural', 'ruelle', 'essential', 'boundary', 'chernoff',
                              'part_b', 'lyapunov', 'lemmas', 'frequency', 'obstructions',
                              'part_c', 'transition', 'parity', 'block_bound', 'fk_bound',
                              'part_d', 'tensor_constr', 'tensor_evol', 'tensor_cauchy',
                          ],
                          help='Which test to run (default: all)')

    # === Tier 2: Lyapunov & Modular ===
    lya = parser.add_argument_group('Lyapunov & Modular Settings')
    lya.add_argument('--gamma', type=float, default=0.74,
                     help='Lyapunov constant (default: 0.74, theory: 0.737)')
    lya.add_argument('--target-v2', type=int, default=10,
                     help='Target nu_2 for deep brake search (default: 10)')

    # === Tier 2: Tensor SVD ===
    tensor = parser.add_argument_group('Tensor SVD Settings')
    tensor.add_argument('--min-blocks', type=int, default=5,
                        help='Minimum blocks for tensor analysis (default: 5)')
    tensor.add_argument('--sim-cv', type=float, default=0.85,
                        help='CV for divergent simulation (default: 0.85)')

    # === Tier 2: Closure & Matrix ===
    closure = parser.add_argument_group('Closure & Matrix Settings')
    closure.add_argument('--max-closure-size', type=int, default=50000,
                         help='Max closure size (default: 50000, limited by RAM)')
    closure.add_argument('--n-init', type=int, default=3,
                         help='Initial set {1..n_init} for closure (default: 3)')

    # === Output & Visualisation ===
    output = parser.add_argument_group('Output & Visualization')
    output.add_argument('--no-plot', action='store_true',
                        help='Disable plot generation')
    output.add_argument('--fig-dir', type=str, default='./figures',
                        help='Directory for figures (default: ./figures)')
    output.add_argument('--output', type=str, default=None,
                        help='Path for JSON/CSV results')
    output.add_argument('--quiet', action='store_true',
                        help='Only final result, no intermediate output')

    return parser


def parse_numbers(numbers_str: str) -> list[int] | None:
    """Parse comma-separated string of numbers."""
    if numbers_str is None:
        return None
    return [int(x.strip()) for x in numbers_str.split(',') if x.strip()]