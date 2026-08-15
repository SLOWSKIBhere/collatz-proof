#!/usr/bin/env python3
"""
Main entry point for the Collatz conjecture experimental code repository.

Two modes:
  python main.py              → Interactive menu
  python main.py --test all   → CLI with argparse
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from utils.argparser import get_full_parser, parse_numbers
from utils.test_runner import TestRunner
from utils.visualizer import Visualizer
from utils.results import save_results


class Args:
    """Minimal args container for interactive mode."""
    def __init__(self):
        self.test = 'all'
        self.max_n = 10**12
        self.sample_size = 200
        self.seed = 42
        self.numbers = None
        self.gamma = 0.74
        self.no_plot = False
        self.fig_dir = './figures'
        self.output = './data/results.json'
        self.quiet = False
        self.n_init = 3
        self.max_closure_size = 50000
        self.min_blocks = 5
        self.target_v2 = 10
        self.sim_cv = 0.85


# ======================================================================
# INTERACTIVE MENU
# ======================================================================

def interactive_menu():
    """Interactive console menu for selecting and running tests."""
    print("\n" + "=" * 60)
    print("  COLLATZ CONJECTURE — EXPERIMENTAL CODE")
    print("  Finite tests do not constitute a proof")
    print("=" * 60)

    args = Args()
    runner = TestRunner(args)

    while True:
        print("\nSelect a part to run:")
        print("  1. Part A — Spectral Resolution of Cycles")
        print("  2. Part B — Obstruction Theory")
        print("  3. Part C — 2-Adic Ergodic Proof (Divergence)")
        print("  4. Part D — Kinematic Tensor SVD Diagnostics")
        print("  5. Run ALL tests")
        print("  6. Generate paper figures")
        print("  7. Configure parameters (max_n, sample_size, seed, gamma)")
        print("  0. Exit")
        print("-" * 60)
        print(f"  Current: max_n={args.max_n}, sample={args.sample_size}, "
              f"seed={args.seed}, gamma={args.gamma}")

        choice = input("\nYour choice [0-7]: ").strip()

        if choice == '0':
            print("Exiting.")
            break
        elif choice == '5':
            runner.run_all(verbose=not args.quiet)
        elif choice == '6':
            _generate_paper_figures()
        elif choice == '7':
            _configure_parameters(args)
        elif choice in ('1', '2', '3', '4'):
            part_map = {'1': 'Part A', '2': 'Part B', '3': 'Part C', '4': 'Part D'}
            part = part_map[choice]
            _run_part_interactive(runner, args, part)
        else:
            print("Invalid choice. Please enter 0-7.")


def _configure_parameters(args):
    """Interactively configure test parameters."""
    print("\n--- Parameter Configuration ---")
    max_n_str = input(f"  Max n for random sampling [current: {args.max_n}]: ").strip()
    if max_n_str:
        args.max_n = int(max_n_str)
    sample_str = input(f"  Sample size [current: {args.sample_size}]: ").strip()
    if sample_str:
        args.sample_size = int(sample_str)
    seed_str = input(f"  Random seed [current: {args.seed}]: ").strip()
    if seed_str:
        args.seed = int(seed_str)
    gamma_str = input(f"  Lyapunov gamma [current: {args.gamma}]: ").strip()
    if gamma_str:
        args.gamma = float(gamma_str)
    print(f"  Updated: max_n={args.max_n}, sample={args.sample_size}, "
          f"seed={args.seed}, gamma={args.gamma}")


def _run_part_interactive(runner, args, part):
    """Run tests for a specific part with user selection."""
    tests = runner.list_tests()
    part_tests = [(k, desc) for k, desc, p in tests if p == part]

    print(f"\n--- {part} Tests ---")
    for i, (key, desc) in enumerate(part_tests, 1):
        print(f"  {i}. {key:<15s} — {desc}")
    print(f"  {len(part_tests)+1}. ALL")

    choice = input(f"\nSelect test [1-{len(part_tests)+1}, default: ALL]: ").strip()
    if not choice or choice == str(len(part_tests) + 1):
        runner.run_all(part=part, verbose=not args.quiet)
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(part_tests):
                key = part_tests[idx][0]
                result = runner.run(key, verbose=not args.quiet)
                if result:
                    _handle_result(args, key, result)
        except (ValueError, IndexError):
            print("Invalid selection.")


def _handle_result(args, key, result):
    """Save results to data/."""
    os.makedirs('data', exist_ok=True)
    save_results(result, f'data/{key}.json')


def _generate_paper_figures():
    """Generate the five paper figures."""
    from scripts.generate_figures import (
        generate_spectrum_contours, generate_chernoff_convergence,
        generate_mod8_transition_graph, generate_lyapunov_gap,
        generate_tensor_detector,
    )
    print("\nGenerating paper figures...")
    os.makedirs('figures', exist_ok=True)
    generate_spectrum_contours()
    generate_chernoff_convergence()
    generate_mod8_transition_graph()
    generate_lyapunov_gap()
    generate_tensor_detector()
    print("  All figures saved to figures/")


# ======================================================================
# MAIN
# ======================================================================

def main():
    os.makedirs('data', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    # Interactive mode (no CLI arguments)
    if len(sys.argv) == 1:
        interactive_menu()
        return

    # CLI mode
    parser = get_full_parser(
        "Collatz Conjecture: Experimental Code Repository\n"
        "WARNING: finite computations do not prove the conjecture.\n"
        "Accompanies the paper:\n"
        "  'A Complete Proof of the Collatz Conjecture\n"
        "   via Spectral Analysis and 2-Adic Ergodic Dynamics'"
    )
    args = parser.parse_args()

    runner = TestRunner(args)
    runner.run_all(verbose=not args.quiet)

    if not args.quiet:
        print(f"\n  Results saved to: data/")
        if not args.no_plot:
            print(f"  Figures saved to: {args.fig_dir}/")


if __name__ == "__main__":
    main()
