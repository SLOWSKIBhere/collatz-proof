"""
Visualizer — unified plotting and table rendering for all Collatz test results.

Each test has a dedicated plot method matched by test key.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Any, Optional


class Visualizer:
    """
    Automatic visualisation of Collatz test results.

    Usage:
        viz = Visualizer(result, fig_dir='./figures', test_name='parity')
        viz.plot()   # generates the correct plot for this test
        viz.table()  # prints formatted data table
    """

    def __init__(self, result: dict, fig_dir: str = './figures',
                 test_name: str = 'unknown'):
        self.result = result
        self.fig_dir = fig_dir
        self.test_name = test_name
        os.makedirs(fig_dir, exist_ok=True)

    def plot(self) -> Optional[str]:
        """Dispatch to the correct plot method based on test name."""
        dispatch = {
            'structural': self._plot_spectrum,
            'ruelle': self._plot_ruelle,
            'essential': self._plot_spectrum,
            'boundary': self._plot_boundary,
            'chernoff': self._plot_chernoff,
            'lyapunov': self._plot_lyapunov,
            'lemmas': self._plot_lemmas,
            'frequency': self._plot_frequency,
            'obstructions': self._plot_obstructions,
            'transition': self._plot_transition_heatmap,
            'parity': self._plot_parity,
            'block_bound': self._plot_block_bound,
            'fk_bound': self._plot_fk_bound,
            'tensor_constr': self._plot_tensor_constr,
            'tensor_evol': self._plot_tensor_evol,
            'tensor_cauchy': self._plot_tensor_cauchy,
        }
        plot_func = dispatch.get(self.test_name, self._plot_generic)
        return plot_func()

    def table(self) -> str:
        """Generate a formatted text table of key-value pairs."""
        r = self.result

        # Handle list of dicts (e.g., structural results, trace data)
        for list_key in ['results', 'trace_data']:
            if list_key in r and isinstance(r[list_key], list) and len(r[list_key]) > 0:
                if isinstance(r[list_key][0], dict):
                    return self._table_from_list(r[list_key])

        # Flatten nested results
        flat = {}
        for k, v in r.items():
            if isinstance(v, dict) and 'passed' in v:
                for sub_k, sub_v in v.items():
                    flat[f'{k}.{sub_k}'] = sub_v
            elif not isinstance(v, (dict, list, np.ndarray)):
                flat[k] = v

        numeric = {k: v for k, v in flat.items()
                   if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_))
                   and not k.startswith('_')}

        if not numeric:
            return "(no scalar fields to tabulate)"

        max_key_len = max(len(k) for k in numeric)
        lines = [f"\n  {'Key':<{max_key_len+2}} Value",
                 f"  {'-'*max_key_len}  {'-'*20}"]
        for k, v in numeric.items():
            if isinstance(v, bool):
                v_str = '✓' if v else '✗'
            elif isinstance(v, float):
                v_str = f'{v:.6g}'
            else:
                v_str = str(v)
            lines.append(f"  {k:<{max_key_len+2}} {v_str}")

        return '\n'.join(lines)

    def _table_from_list(self, rows: list) -> str:
        """Format a list of dicts as a text table."""
        if not rows:
            return "(empty table)"
        keys = [k for k in rows[0].keys() if not k.startswith('_')]
        max_lens = {
            k: max(len(str(k)), max(len(str(row.get(k, ''))) for row in rows))
            for k in keys
        }

        header = '  ' + '  '.join(f'{str(k):<{max_lens[k]}}' for k in keys)
        sep = '  ' + '  '.join('-' * max_lens[k] for k in keys)
        lines = [header, sep]
        for row in rows:
            line = '  ' + '  '.join(f'{str(row.get(k, "")):<{max_lens[k]}}' for k in keys)
            lines.append(line)
        return '\n'.join(lines)

    # ==================================================================
    # Plot methods — one per test
    # ==================================================================

    def _save_and_close(self, fig, suffix: str) -> str:
        path = f'{self.fig_dir}/{self.test_name}_{suffix}.pdf'
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_spectrum(self) -> str:
        r = self.result
        fig, ax = plt.subplots(figsize=(7, 7))
        if 'eigenvalues' in r:
            raw = r['eigenvalues']
            # Явно указываем dtype для массива комплексных чисел
            ev = np.array([complex(x) for x in raw], dtype=np.complex128)
            ax.scatter(ev.real, ev.imag, s=2, c='#1f77b4', alpha=0.6)
        theta = np.linspace(0, 2*np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=1, alpha=0.5)
        ax.scatter([1], [0], s=100, c='red', zorder=5)
        ax.scatter([-1], [0], s=100, c='red', zorder=5)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axhline(0, color='gray', lw=0.5, alpha=0.4)
        ax.axvline(0, color='gray', lw=0.5, alpha=0.4)
        ax.set_xlabel('Re(z)')
        ax.set_ylabel('Im(z)')
        ax.set_title(f'Spectrum — {self.test_name}')
        return self._save_and_close(fig, 'spectrum')

    def _plot_ruelle(self) -> str:
        r = self.result
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Panel 1: Tr(U^k) vs k
        if 'trace_data' in r:
            trace_data = r['trace_data']
            k_vals = [td['k'] for td in trace_data]
            traces = [td['trace'] for td in trace_data]
            expected = [td['expected'] for td in trace_data]
            ax1.plot(k_vals, traces, 'o-', markersize=4, lw=1.5, label='Tr(U^k)')
            ax1.plot(k_vals, expected, 'r--', lw=1, alpha=0.7, label='Expected')
        ax1.set_xlabel('k')
        ax1.set_ylabel('Tr(U^k)')
        ax1.set_title('Trace of Koopman Powers')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Spectrum
        if 'eigenvalues' in r:
            raw = r['eigenvalues']
            # Явно указываем dtype для массива комплексных чисел
            ev = np.array([complex(x) for x in raw], dtype=np.complex128)
            # Исправляем создание круга - используем plt.Circle вместо plt.Circle
            circle = Circle((0, 0), 1, fill=False, color='gray', ls='--', alpha=0.5)
            ax2.add_artist(circle)
            ax2.scatter(ev.real, ev.imag, alpha=0.5, s=20, c='blue')
            ax2.scatter([1, -1], [0, 0], color='red', s=100, marker='*', label='+1, -1')
        ax2.axhline(y=0, color='gray', alpha=0.3)
        ax2.axvline(x=0, color='gray', alpha=0.3)
        ax2.set_xlabel('Re(λ)')
        ax2.set_ylabel('Im(λ)')
        ax2.set_title('Spectrum of U')
        ax2.set_aspect('equal')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(f'Ruelle Zeta Function — {self.test_name}')
        return self._save_and_close(fig, 'ruelle')

    def _plot_boundary(self) -> str:
        r = self.result
        fig, ax = plt.subplots(figsize=(10, 6))

        theta = np.array(r.get('theta_vals', []))
        norms_r10 = np.array(r.get('norms_r10', []))
        norms_r11 = np.array(r.get('norms_r11', []))
        norms_r12 = np.array(r.get('norms_r12', []))

        if len(theta) > 0:
            if len(norms_r10) > 0:
                ax.semilogy(theta, norms_r10, 'b-', lw=1, alpha=0.7, label='r = 1.0')
            if len(norms_r11) > 0:
                ax.semilogy(theta, norms_r11, 'g-', lw=1, alpha=0.5, label='r = 1.1')
            if len(norms_r12) > 0:
                ax.semilogy(theta, norms_r12, 'r-', lw=1, alpha=0.3, label='r = 1.2')

        for peak in r.get('peak_thetas', []):
            ax.axvline(x=peak, color='red', ls='--', alpha=0.5)

        ax.axvline(x=0, color='orange', ls=':', alpha=0.5, label='z=+1')
        ax.axvline(x=np.pi, color='purple', ls=':', alpha=0.5, label='z=-1')

        ax.set_xlabel('θ (radians)')
        ax.set_ylabel('‖(zI - U)⁻¹‖')
        M = r.get('matrix_size_boundary', r.get('matrix_size', '?'))
        ax.set_title(f'Boundary Resolvent Peaks (M={M})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return self._save_and_close(fig, 'boundary')

    def _plot_chernoff(self) -> str:
        r = self.result
        fig, ax = plt.subplots(figsize=(8, 5))
        methods = ['Uniform', 'Exponential']
        errors = [r.get('error_uniform_pct', 0), r.get('error_exponential_pct', 0)]
        colors = ['#d62728', '#2ca02c']
        ax.bar(methods, errors, color=colors, alpha=0.7)
        ax.set_ylabel('Relative error (%)')
        ax.set_title(f'Chernoff Approximation — {self.test_name}')
        return self._save_and_close(fig, 'chernoff')

    def _plot_lyapunov(self) -> str:
        r = self.result
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, f"Lyapunov Test\nViolations: {r.get('violations', 0)}\n"
                f"Total steps: {r.get('total_steps', 0)}\n"
                f"Passed: {r.get('passed', False)}",
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.set_title(f'Lyapunov Function — {self.test_name}')
        return self._save_and_close(fig, 'summary')

    def _plot_lemmas(self) -> str:
        r = self.result
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        l5 = r.get('lemma5', {})
        ax1.text(0.5, 0.5, f"Lemma 5: Deep Brakes\nPassed: {l5.get('passed')}\n"
                f"Mean brakes: {l5.get('mean_deep_brakes', 0):.1f}\n"
                f"Without brakes: {l5.get('trajectories_without_brakes', 0)}",
                transform=ax1.transAxes, ha='center', va='center', fontsize=12)
        ax1.set_title('Lemma 5')
        l6 = r.get('lemma6', {})
        ax2.text(0.5, 0.5, f"Lemma 6: Log Bound\nPassed: {l6.get('passed')}\n"
                f"Violations: {l6.get('violations', 0)}\n"
                f"Mean block: {l6.get('mean_block_length', 0):.1f}",
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        ax2.set_title('Lemma 6')
        fig.suptitle(f'Modular Lemmas — {self.test_name}')
        return self._save_and_close(fig, 'lemmas')

    def _plot_frequency(self) -> str:
        r = self.result
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = ['Mean f_k', 'Min f_k', 'Max f_k']
        values = [r.get('mean_f_k', 0), r.get('min_f_k', 0), r.get('max_f_k', 0)]
        ax.bar(labels, values, color='#ff7f0e', alpha=0.7)
        if 'f_crit' in r:
            ax.axhline(y=r['f_crit'], color='red', ls='--', label=f"f_crit = {r['f_crit']}")
        ax.set_ylabel('Frequency')
        ax.set_title(f'Deep Brake Frequency — {self.test_name}')
        ax.legend()
        return self._save_and_close(fig, 'frequency')

    def _plot_obstructions(self) -> str:
        r = self.result
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ceil = r.get('ceiling', {})
        ax1.text(0.5, 0.5, f"Lyapunov Ceiling\nPassed: {ceil.get('passed')}\n"
                f"Violations: {ceil.get('violations', 0)}",
                transform=ax1.transAxes, ha='center', va='center', fontsize=12)
        ax1.set_title('Ceiling')
        grow = r.get('growth', {})
        ax2.text(0.5, 0.5, f"Growth Rate\nPassed: {grow.get('passed')}\n"
                f"Mean growth: {grow.get('mean_growth_rate', 0):.4f}\n"
                f"β={grow.get('beta_theoretical', 0):.4f} γ={grow.get('gamma_theoretical', 0):.4f}",
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        ax2.set_title('Growth Rate')
        fig.suptitle(f'Obstruction Barriers — {self.test_name}')
        return self._save_and_close(fig, 'obstructions')

    def _plot_transition_heatmap(self) -> str:
        r = self.result
        fig, ax = plt.subplots(figsize=(7, 6))
        if 'probabilities' in r:
            prob = np.asarray(r['probabilities'])
            residues = [1, 3, 5, 7]
            im = ax.imshow(prob[1:8, 1:8], cmap='Blues', vmin=0, vmax=1)
            ax.set_xticks(range(4))
            ax.set_xticklabels([str(r) for r in residues])
            ax.set_yticks(range(4))
            ax.set_yticklabels([str(r) for r in residues])
            ax.set_xlabel('To residue')
            ax.set_ylabel('From residue')
            for i in range(4):
                for j in range(4):
                    if prob[residues[i], residues[j]] > 0:
                        color = 'white' if prob[residues[i], residues[j]] > 0.5 else 'black'
                        ax.text(j, i, f'{prob[residues[i], residues[j]]:.2f}',
                                ha='center', va='center', color=color, fontsize=9)
            plt.colorbar(im, ax=ax)
        ax.set_title(f'Transition Probabilities — {self.test_name}')
        return self._save_and_close(fig, 'transitions')

    def _plot_parity(self) -> str:
        r = self.result
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = ['Mean f₃₅', 'Min f₃₅', 'Max f₃₅']
        values = [r.get('mean_f35', 0), r.get('min_f35', 0), r.get('max_f35', 0)]
        ax.bar(labels, values, color='#1f77b4', alpha=0.7)
        if 'theoretical_limit' in r:
            ax.axhline(y=r['theoretical_limit'], color='red', ls='--',
                      label=f"Theoretical limit = {r['theoretical_limit']}")
        ax.set_ylabel('Frequency')
        ax.set_title(f'f_{{3→5}} Parity Analysis — {self.test_name}')
        ax.legend()
        return self._save_and_close(fig, 'parity')

    def _plot_block_bound(self) -> str:
        r = self.result
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = ['Mean', 'Median', 'Max', 'Bound']
        values = [r.get('mean_block_length', 0), r.get('median_block_length', 0),
                  r.get('max_block_length', 0), r.get('theoretical_bound', 0)]
        colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
        ax.bar(labels, values, color=colors, alpha=0.7)
        ax.set_ylabel('Block length')
        ax.set_title(f'Block Length Bound — {self.test_name}')
        return self._save_and_close(fig, 'block_bound')

    def _plot_fk_bound(self) -> str:
        r = self.result
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = ['Mean f_k', 'Min f_k']
        values = [r.get('mean_fk', 0), r.get('min_fk', 0)]
        ax.bar(labels, values, color='#1f77b4', alpha=0.7)
        if 'theoretical_lower_bound' in r:
            ax.axhline(y=r['theoretical_lower_bound'], color='green', ls='--',
                      label=f"Lower bound = {r['theoretical_lower_bound']:.4f}")
        ax.set_ylabel('Frequency')
        ax.set_title(f'f_k Lower Bound — {self.test_name}')
        ax.legend()
        return self._save_and_close(fig, 'fk_bound')

    def _plot_tensor_constr(self) -> str:
        r = self.result
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = ['Mean κ', 'Min κ', 'Max κ']
        values = [r.get('mean_kappa', 0), r.get('min_kappa', 0), r.get('max_kappa', 0)]
        ax.bar(labels, values, color='#1f77b4', alpha=0.7)
        ax.set_ylabel('Condition number κ')
        ax.set_title(f'Kinematic Tensor Construction — {self.test_name}')
        return self._save_and_close(fig, 'tensor_constr')

    def _plot_tensor_evol(self) -> str:
        r = self.result
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.bar(['Real', 'Simulated'],
                [r.get('real_mean_kappa_trend', 0), r.get('sim_mean_kappa_trend', 0)],
                color=['#2ca02c', '#d62728'], alpha=0.7)
        ax1.axhline(y=0, color='gray', lw=0.5)
        ax1.set_ylabel('Mean κ trend')
        ax1.set_title('Trend Direction')
        ax2.bar(['Real', 'Simulated'],
                [r.get('real_trajectories', 0), r.get('sim_trajectories', 0)],
                color=['#2ca02c', '#d62728'], alpha=0.7)
        ax2.set_ylabel('Count')
        ax2.set_title('Trajectories Tested')
        fig.suptitle(f'κ Evolution — {self.test_name}')
        return self._save_and_close(fig, 'tensor_evol')

    def _plot_tensor_cauchy(self) -> str:
        r = self.result
        fig, ax = plt.subplots(figsize=(8, 5))
        counts = [r.get('roots_all_outside', 0), r.get('roots_all_inside', 0),
                  r.get('mixed_or_edge', 0)]
        ax.bar(['Outside', 'Inside', 'Mixed/Edge'], counts,
               color=['#2ca02c', '#d62728', '#7f7f7f'], alpha=0.7)
        ax.set_ylabel('Number of trajectories')
        ax.set_title(f'Cauchy Root Distribution — {self.test_name}')
        return self._save_and_close(fig, 'cauchy')

    def _plot_generic(self) -> Optional[str]:
        """Fallback: summary text plot."""
        r = self.result
        fig, ax = plt.subplots(figsize=(8, 5))
        passed = r.get('passed', 'N/A')
        text = f"Test: {self.test_name}\nPassed: {passed}\n\nKey values:\n"
        for k, v in r.items():
            if not isinstance(v, (dict, list, np.ndarray)) and not k.startswith('_'):
                text += f"  {k}: {v}\n"
        ax.text(0.5, 0.5, text, transform=ax.transAxes, ha='center', va='center',
                fontsize=11, family='monospace')
        ax.set_title(f'Test Result — {self.test_name}')
        return self._save_and_close(fig, 'summary')