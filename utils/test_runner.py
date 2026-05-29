"""
Test Runner — unified interface for all Collatz tests.
"""

from typing import Callable, Dict, Optional, Any
import os
import numpy as np


class TestRunner:
    """
    Registry and executor for all Collatz conjecture tests.

    Usage:
        runner = TestRunner(args)
        runner.run('structural')
        runner.run_all()
        runner.list_tests()
    """

    def __init__(self, args):
        self.args = args
        self._registry: Dict[str, Dict] = {}
        self._results: Dict[str, Any] = {}
        self._build_registry()

    def _register(self, key: str, func: Callable, description: str,
                  part: str = '', **kwargs):
        """Register a test function with metadata."""
        self._registry[key] = {
            'func': func,
            'description': description,
            'part': part,
            'kwargs': kwargs,
        }

    def _build_registry(self):
        """Build the complete test registry."""
        a = self.args

        # Part A
        self._register('structural',
                    lambda: _import_and_call('part_a.structural_theorem', 'test_structural_theorem',
                                                n_values=[3, 5, 7, 10, 15, 20],
                                                max_steps=10, max_size=a.max_closure_size),
                    'Structural Theorem (Theorem 3.1)', 'Part A')
        self._register('ruelle',
                       lambda: _import_and_call('part_a.ruelle_zeta', 'test_ruelle_zeta',
                                                n_init=a.n_init, max_steps=50, max_size=min(a.max_closure_size, 2000)),
                       'Ruelle Zeta Function (Theorem 8.1-8.2)', 'Part A')
        self._register('essential',
                       lambda: _import_and_call('part_a.essential_spectrum', 'test_essential_spectrum',
                                                n_init=a.n_init, max_steps=50, max_size=min(a.max_closure_size, 2000)),
                       'Essential Spectrum (Theorem 8.3)', 'Part A')
        self._register('boundary',
                    lambda: _import_and_call('part_a.boundary_resolvent', 'test_boundary_resolvent',
                                                max_size_fredholm=min(a.max_closure_size, 2000),
                                                max_size_boundary=min(a.max_closure_size, 2000)),
                    'Boundary Resolvent (Theorem 8.4)', 'Part A')
        self._register('chernoff',
                    lambda: _import_and_call('part_a.chernoff_validation', 'test_chernoff_validation',
                                                max_size=500, z=1.5+0.0j, n_chernoff=50),
                    'Chernoff Approximations (Theorem 5.1-5.2)', 'Part A')

        # Part B
        self._register('lyapunov',
                       lambda: _import_and_call('part_b_obstructions.lyapunov_test', 'test_lyapunov',
                                                max_n=a.max_n, sample_size=a.sample_size,
                                                gamma=a.gamma, seed=a.seed),
                       'Lyapunov Function (Theorem 9.3)', 'Part B')
        self._register('lemmas',
                       lambda: _run_modular_lemmas(a),
                       'Modular Lemmas 5 & 6', 'Part B')
        self._register('frequency',
                       lambda: _import_and_call('part_b_obstructions.deep_brake_frequency', 'test_deep_brake_frequency',
                                                max_n=a.max_n, sample_size=a.sample_size, seed=a.seed),
                       'Deep Brake Frequency (Lemma 9.2)', 'Part B')
        self._register('obstructions',
                       lambda: _run_obstruction_barriers(a),
                       'Obstruction Barriers', 'Part B')

        # Part C
        self._register('transition',
                       lambda: _import_and_call('part_c_ergodic.transition_table', 'compute_transition_table',
                                                max_n=a.max_n, sample_size=a.sample_size, seed=a.seed),
                       'Full Transition Table mod 8', 'Part C')
        self._register('parity',
                       lambda: _import_and_call('part_c_ergodic.parity_analysis', 'test_parity_analysis',
                                                max_n=a.max_n, sample_size=a.sample_size, seed=a.seed),
                       'f_{3→5} Parity Analysis (Theorem C.2)', 'Part C')
        self._register('block_bound',
                       lambda: _import_and_call('part_c_ergodic.block_length_bound', 'test_block_length_bound',
                                                max_n=a.max_n, sample_size=a.sample_size, seed=a.seed),
                       'Block Length Bound (Theorem C.3)', 'Part C')
        self._register('fk_bound',
                       lambda: _import_and_call('part_c_ergodic.deep_brake_frequency_test', 'test_fk_bound',
                                                max_n=a.max_n, sample_size=a.sample_size, seed=a.seed),
                       'Deep Brake Frequency Lower Bound (Theorem C.4)', 'Part C')

        # Part D
        self._register('tensor_constr',
                       lambda: _import_and_call('part_d_tensor.tensor_construction', 'test_tensor_construction',
                                                max_n=a.max_n, sample_size=a.sample_size,
                                                min_blocks=a.min_blocks, seed=a.seed),
                       'Kinematic Tensor Construction', 'Part D')
        self._register('tensor_evol',
                       lambda: _import_and_call('part_d_tensor.tensor_evolution', 'test_tensor_evolution',
                                                max_n=a.max_n, sample_size=a.sample_size,
                                                min_blocks=a.min_blocks, seed=a.seed),
                       'Condition Number Evolution', 'Part D')
        self._register('tensor_cauchy',
                       lambda: _import_and_call('part_d_tensor.cauchy_criterion', 'test_cauchy_criterion',
                                                max_n=a.max_n, sample_size=a.sample_size,
                                                min_blocks=a.min_blocks, seed=a.seed),
                       'Cauchy Root Criterion', 'Part D')

    def list_tests(self) -> list:
        """Return list of (key, description, part) for all registered tests."""
        return [(k, v['description'], v['part']) for k, v in self._registry.items()]

    def get_tests_by_part(self, part: str) -> list:
        """Return test keys for a given part."""
        return [k for k, v in self._registry.items() if v['part'] == part]

    def run(self, key: str, verbose: bool = True) -> Optional[dict]:
        """Run a single registered test by key."""
        if key not in self._registry:
            print(f"Unknown test: {key}")
            return None

        entry = self._registry[key]
        if verbose:
            print(f"\n{'='*70}")
            print(f"  {entry['part']}: {entry['description']}")
            print(f"{'='*70}")

        try:
            result = entry['func']()
            self._results[key] = result

            # Auto-save to data/
            if self.args.output and result:
                from utils.results import save_results
                os.makedirs('data', exist_ok=True)
                save_results(result, f'data/{key}.json')

            # Auto-visualise
            if not self.args.no_plot and result:
                from utils.visualizer import Visualizer
                viz = Visualizer(result, fig_dir=self.args.fig_dir, test_name=key)
                if verbose:
                    print(viz.table())
                viz.plot()

            if verbose and isinstance(result, dict):
                passed = result.get('passed', 'N/A')
                print(f"  Passed: {passed}")
            return result
        except Exception as e:
            print(f"  ERROR: {e}")
            return {'test': key, 'passed': False, 'error': str(e)}

    def run_all(self, part: Optional[str] = None, verbose: bool = True) -> dict:
        """Run all tests, optionally filtered by part."""
        if part:
            keys = self.get_tests_by_part(part)
        else:
            keys = list(self._registry.keys())

        for key in keys:
            self.run(key, verbose=verbose)

        if not self.args.quiet:
            print(f"\n{'='*70}")
            print(f"  ALL TESTS COMPLETE")
            print(f"{'='*70}")

        return self._results

    def get_results(self) -> dict:
        """Return all accumulated test results."""
        return self._results


# ======================================================================
# Helper functions
# ======================================================================

def _import_and_call(module_path: str, func_name: str, **kwargs):
    """Import a function and call it with given kwargs."""
    import importlib
    mod = importlib.import_module(module_path)
    func = getattr(mod, func_name)
    return func(**kwargs)


def _run_modular_lemmas(args):
    from part_b_obstructions.modular_lemmas import (
        test_lemma5_deep_brakes, test_lemma6_log_bound,
    )
    r5 = test_lemma5_deep_brakes(max_n=args.max_n, sample_size=args.sample_size, seed=args.seed)
    r6 = test_lemma6_log_bound(max_n=args.max_n, sample_size=args.sample_size, seed=args.seed)
    return {'lemma5': r5, 'lemma6': r6}


def _run_obstruction_barriers(args):
    from part_b_obstructions.obstruction_tests import (
        test_lyapunov_ceiling, test_block_growth_rate, test_sublinear_gap,
    )
    r_ceil = test_lyapunov_ceiling(max_n=args.max_n, sample_size=args.sample_size, seed=args.seed)
    r_grow = test_block_growth_rate(
        max_n=min(args.max_n, 5 * 10**7), sample_size=args.sample_size, seed=args.seed,
    )
    r_gap = test_sublinear_gap()
    return {'ceiling': r_ceil, 'growth': r_grow, 'gap': r_gap}