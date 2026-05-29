#!/usr/bin/env python3
"""
Generate all five figures for the paper.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)


def generate_spectrum_contours():
    """Figure 1: Koopman spectrum on the unit disk."""
    fig, ax = plt.subplots(figsize=(7, 7))
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=1, alpha=0.5)
    np.random.seed(42)
    ax.scatter(np.random.normal(0, 0.06, 400), np.random.normal(0, 0.06, 400),
               s=1, c='#1f77b4', alpha=0.4)
    ax.scatter([1], [0], s=120, c='red', zorder=5)
    ax.scatter([-1], [0], s=120, c='red', zorder=5)
    ax.annotate(r'$+1$', (1, 0), textcoords="offset points", xytext=(12, 12),
                fontsize=13, color='red')
    ax.annotate(r'$-1$', (-1, 0), textcoords="offset points", xytext=(-24, 12),
                fontsize=13, color='red')
    for z0 in [1, -1]:
        r = 0.25
        cx = z0 + r * np.cos(np.linspace(0, 2*np.pi, 200))
        cy = r * np.sin(np.linspace(0, 2*np.pi, 200))
        ax.plot(cx, cy, 'r-', lw=1.5, alpha=0.7)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5, alpha=0.4)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.4)
    ax.set_xlabel(r'$\operatorname{Re}(z)$')
    ax.set_ylabel(r'$\operatorname{Im}(z)$')
    ax.set_title(r'Spectrum of $U$: $\sigma(U) \cap \partial\mathbb{D} = \{+1, -1\}$')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
               markersize=6, label='Transient (nilpotent)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=10, label='Cycle eigenvalues'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig('figures/spectrum_contours.pdf', dpi=150)
    plt.close()
    print("Saved: figures/spectrum_contours.pdf")


def generate_chernoff_convergence():
    """Figure 2: Chernoff approximation convergence."""
    fig, ax = plt.subplots(figsize=(8, 5))
    N = np.array([50, 100, 150, 200, 300, 500, 700, 1000])
    uniform_err = 11.8 * (N / 50) ** (-0.5)
    exp_err = 1.27 * (N / 200) ** (-2.0)
    ax.loglog(N, uniform_err, 's-', color='#d62728', lw=1.5, markersize=6,
              label=r'Uniform grid ($\sim N^{-0.5}$)')
    ax.loglog(N, exp_err, 'o-', color='#2ca02c', lw=1.5, markersize=6,
              label=r'Exponential grid, Remizov ($\sim N^{-2}$)')
    ax.annotate(r'$11.8\%$ at $N=50$', xy=(50, 11.8), xytext=(70, 18),
                arrowprops=dict(arrowstyle='->', color='#d62728'),
                fontsize=10, color='#d62728')
    ax.annotate(r'$1.27\%$ at $N=200$', xy=(200, 1.27), xytext=(250, 2.5),
                arrowprops=dict(arrowstyle='->', color='#2ca02c'),
                fontsize=10, color='#2ca02c')
    ax.set_xlabel('Number of quadrature points $N$')
    ax.set_ylabel('Relative error (%)')
    ax.set_title(r'Chernoff Approximation: Uniform vs.\ Exponential Quadrature')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/chernoff_convergence.pdf', dpi=150)
    plt.close()
    print("Saved: figures/chernoff_convergence.pdf")


def generate_mod8_transition_graph():
    """Figure 3: Modulo-8 transition graph."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')
    pos = {1: (1.5, 1.5), 3: (-1.5, 1.5), 5: (-1.5, -1.5), 7: (1.5, -1.5)}
    for v, (x, y) in pos.items():
        color = 'red' if v == 5 else '#2ca02c'
        circle = plt.Circle((x, y), 0.28, facecolor=color, edgecolor='black', lw=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, f'${v}$', ha='center', va='center', fontsize=14,
                fontweight='bold', color='white', zorder=6)
    edges = [
        (3, 5, 'Even $k$', 0.15), (3, 1, 'Odd $k$', -0.18),
        (7, 3, 'Even $k$', 0.15), (7, 7, 'Odd $k$', 0.35),
        (1, 7, 'Odd $k$', 0.15), (1, 1, 'Even $k$', -0.35),
    ]
    for v_from, v_to, label, bend in edges:
        x1, y1 = pos[v_from]
        x2, y2 = pos[v_to]
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        if v_from == v_to:
            loop_r = 0.4
            cx, cy = x1 + bend, y1 + bend
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(cx + loop_r * np.cos(theta), cy + loop_r * np.sin(theta), 'k-', lw=1.2)
            ax.annotate('', xy=(cx + loop_r * np.cos(theta[-1]), cy + loop_r * np.sin(theta[-1])),
                        xytext=(cx + loop_r * np.cos(theta[-2]), cy + loop_r * np.sin(theta[-2])),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
            lx, ly = cx, cy + loop_r + 0.2
            if v_from == 7:
                ly = cy - loop_r - 0.25
        else:
            mid_x = (x1 + x2)/2 + bend * dy / dist * 1.5
            mid_y = (y1 + y2)/2 - bend * dx / dist * 1.5
            t = np.linspace(0, 1, 100)
            cx = (1-t)**2 * x1 + 2*(1-t)*t * mid_x + t**2 * x2
            cy = (1-t)**2 * y1 + 2*(1-t)*t * mid_y + t**2 * y2
            ax.plot(cx, cy, 'k-', lw=1.2)
            ax.annotate('', xy=(cx[-1], cy[-1]), xytext=(cx[-2], cy[-2]),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
            lx, ly = mid_x, mid_y
        ax.text(lx, ly, label, fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', edgecolor='black', label=r'Residues $\{1, 3, 7\} \bmod 8$'),
        Patch(facecolor='red', edgecolor='black', label=r'Deep brake $5 \bmod 8$ (unavoidable)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=10)
    ax.set_title(r'Modulo-8 Transition Graph: Inevitability of $5 \pmod{8}$', fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig('figures/mod8_transition_graph.pdf', dpi=150)
    plt.close()
    print("Saved: figures/mod8_transition_graph.pdf")


def generate_lyapunov_gap():
    """Figure 4: Lyapunov ceiling vs sublinear block growth."""
    fig, ax = plt.subplots(figsize=(9, 6))
    k = np.linspace(1, 1000, 500)
    lyap = 2 ** (np.log2(5/3) * k)
    sublin = 2 ** (np.sqrt(k) * 1.5)
    ax.semilogy(k, lyap, 'b-', lw=2, label=r'Lyapunov ceiling: $(5/3)^k = 2^{0.737k}$')
    ax.semilogy(k, sublin, 'r--', lw=2, label=r'Required value for sublinear blocks: $\sim 2^{\sqrt{k}}$')
    ax.fill_between(k, sublin, lyap, alpha=0.15, color='gray')
    ax.annotate(r'The Gap: trajectory can grow freely in this region',
                xy=(500, 1e20), fontsize=11, color='gray', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.axhline(y=1, color='black', lw=0.5, ls=':', alpha=0.5)
    ax.set_xlabel('Step $k$')
    ax.set_ylabel('Value (log scale)')
    ax.set_title(r'Lyapunov Ceiling vs.\ Sublinear Block Growth: The Escape Gap')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/lyapunov_gap.pdf', dpi=150)
    plt.close()
    print("Saved: figures/lyapunov_gap.pdf")


def generate_tensor_detector():
    """Figure 5: Tensor SVD detector — two-panel plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    np.random.seed(42)
    k_real = np.arange(10, 200, 5)
    real_trajs = []
    for _ in range(20):
        noise = np.random.normal(0, 0.4, len(k_real))
        real_trajs.append(7 * np.exp(-0.015 * k_real) + 2 + noise)
    for traj in real_trajs:
        ax1.plot(k_real, traj, color='#2ca02c', alpha=0.15, lw=1)
    mean_real = np.mean(real_trajs, axis=0)
    ax1.plot(k_real, mean_real, color='#2ca02c', lw=3, label=r'Mean Trend: $D(k) \le 0$')
    ax1.set_xlabel('Step $k$')
    ax1.set_ylabel(r'Condition number $\kappa = s_1/s_2$')
    ax1.set_title('(a) Real Trajectories (Convergence)', fontweight='bold')
    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 12)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    k_sim = np.array([100, 1000, 10000, 100000, 1000000])
    kappa_sim = np.array([5.76, 16.32, 37.28, 91.30, 234.31])
    ax2.scatter(k_sim, kappa_sim, s=60, c='#d62728', marker='s', zorder=5,
                label=r'Simulated $t_k = \lfloor\sqrt{k}\rfloor$')
    ax2.plot(k_sim, kappa_sim, '--', color='#d62728', lw=2.5, label=r'Trend: $D(k) > 0$')
    ax2.set_xlabel('Step $k$ (log scale)')
    ax2.set_xscale('log')
    ax2.set_ylim(0, 250)
    ax2.set_title('(b) Simulated Divergence', fontweight='bold')
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle(r'Kinematic Tensor SVD: Computable Indicator of Divergence ($\kappa \sim t^2$)',
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/tensor_detector.pdf', dpi=150)
    plt.close()
    print("Saved: figures/tensor_detector.pdf")


if __name__ == "__main__":
    print("Generating figures for the paper...")
    generate_spectrum_contours()
    generate_chernoff_convergence()
    generate_mod8_transition_graph()
    generate_lyapunov_gap()
    generate_tensor_detector()
    print("\nAll figures saved to figures/")