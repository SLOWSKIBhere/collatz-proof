# Collatz Conjecture: Experimental Code Repository

> **Status: NO PROOF.** This repository contains numerical experiments and
> diagnostics. Passing its tests, or checking any finite range, does not prove
> convergence for all positive integers. Names inherited from the accompanying
> manuscript describe its proposed arguments, not independently established
> theorems.

This repository accompanies the paper **"A Complete Proof of the Collatz Conjecture via Spectral Analysis and 2-Adic Ergodic Dynamics"** by Alexey Trikoz (May 2026).

## Overview

The Collatz conjecture states that for any positive integer n, the sequence defined by n → n/2 (if even) or 3n+1 (if odd) eventually reaches the cycle {1, 4, 2}. Despite verification up to 2.95×10²⁰, the conjecture remained unproven for over 80 years.

This repository provides code for testing claims and reproducing numerical
experiments from the paper; it does not computationally verify a general proof.

## Paper Structure

| Part | Topic | Status |
|------|-------|--------|
| **Part A** | Spectral Resolution of Cycles | ✅ 5 independent proofs | `part_a/` |
| **Part B** | Obstruction Theory | ✅ 3 No-Go theorems | `part_b_obstructions/` |
| **Part C** | Proposed 2-Adic ergodic argument | ⚠️ Experimental tests only | `part_c_ergodic/` |
| **Part D** | Kinematic Tensor SVD Diagnostics | ✅ Computational tool | `part_d_tensor/` |

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`

## Performance Backends

The code automatically selects the fastest available backend:

| Backend | Speed vs NumPy | Requirements |
|---------|---------------|--------------|
| **JAX** | ~100x (GPU) / ~10x (CPU) | `pip install jax jaxlib` |
| **Numba** | ~10–50x (CPU) | `pip install numba` |
| **CuPy** | ~100x (GPU) | `pip install cupy-cuda12x` |
| **NumPy** | 1x (baseline) | Always available |

To force a specific backend, set `FORCE_BACKEND` in `config.py`:
```python
FORCE_BACKEND = 'numba'  # or 'jax', 'cupy', 'numpy'
```

## Installation

```bash
git clone https://github.com/alexey-trikoz/collatz-proof.git
cd collatz-proof
pip install -r requirements.txt
```

## Quick Start

### Bounded range checker

`collatz_range_checked.c` is a deliberately limited verification utility. It
strictly accepts positive decimal bounds, checks every input in the inclusive
range, guards `3n+1` before unsigned 64-bit overflow, and emits one JSON record
whose `complete` field distinguishes a finished sweep from an inconclusive
step limit or overflow:

```bash
cc -std=c11 -O3 -Wall -Wextra -Werror collatz_range_checked.c -o collatz_range_checked
./collatz_range_checked 1 1000000000 | tee data/collatz_1_1000000000.json
```

Exit status `0` means the stated finite range was completely checked; `2`
means a trajectory reached the step limit, `3` means it could not continue in
`uint64_t`, and `64` means the command line was invalid. Even a complete run is
only a bounded computational result and has no implication for all positive
integers.

### Interactive Menu (recommended)

Launch the interactive console menu — no command-line arguments needed:

```bash
python main.py
```

You will see:

```
  COLLATZ CONJECTURE — EXPERIMENTAL CODE
  Finite tests do not constitute a proof
============================================================

Select a part to run:
  1. Part A — Spectral Resolution of Cycles
  2. Part B — Obstruction Theory
  3. Part C — 2-Adic Ergodic Proof (Divergence)
  4. Part D — Kinematic Tensor SVD Diagnostics
  5. Run ALL tests
  6. Generate paper figures
  7. Configure parameters (max_n, sample_size, seed, gamma)
  0. Exit
```

From the menu you can:
- **Select a part (1–4):** then choose a specific test or run all tests in that part.
- **Run ALL tests (5):** execute every test in the repository.
- **Generate paper figures (6):** produce the five PDF figures from the paper.
- **Configure parameters (7):** change `max_n`, `sample_size`, `seed`, or `gamma` interactively.
- After each test, results are displayed as a **formatted table** and a **plot** is automatically generated (unless `--no-plot` is set).

### Command-Line Interface

For scripting or headless servers, use argparse:

```bash
python main.py --test all                          # Run all tests
python main.py --test part_c                       # Run Part C tests
python main.py --test parity                       # Run f_{3→5} analysis only
python main.py --test all --max-n 10**9 --seed 123 # Custom parameters
python main.py --numbers 27,703,837799             # Test specific numbers
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--test` | `all` | Test suite to run |
| `--max-n` | `10^12` | Upper bound for random sampling |
| `--sample-size` | `200` | Number of random trajectories |
| `--seed` | `42` | Random seed for reproducibility |
| `--gamma` | `0.74` | Lyapunov constant |
| `--numbers` | None | Comma-separated specific numbers |
| `--no-plot` | False | Disable automatic plot generation |
| `--fig-dir` | `./figures` | Output directory for figures |
| `--output` | None | Path for JSON results |
| `--quiet` | False | Suppress intermediate output |

## Output

- **Tables:** After each test, a formatted table of key-value results is printed to the console.
- **Plots:** Automatically generated and saved as PDF in `figures/`. Plot type depends on test data:
  - Spectrum → eigenvalue scatter plot
  - Transition table → heatmap
  - Parity/frequency → bar chart with theoretical bounds
  - Tensor evolution → two-panel comparison
  - Block length → histogram
  - Cauchy criterion → root distribution bars
- **Data:** JSON files saved to `data/` when `--output` is specified.

## Paper Figures

The five figures from the paper can be regenerated:

```bash
python main.py    # select option 6, or:
python scripts/generate_figures.py
```

| Figure | Description |
|--------|-------------|
| 1 | Koopman operator spectrum on unit disk |
| 2 | Chernoff approximation convergence |
| 3 | Modulo-8 transition graph |
| 4 | Lyapunov ceiling vs sublinear block growth |
| 5 | Tensor SVD detector (two-panel) |

## Repository Structure

```
collatz-proof/
├── main.py                          # Entry point (menu + CLI)
├── config.py                        # Internal constants
├── collatz_analysis.ipynb           # Jupyter notebook
├── utils/                           # Shared utilities
│   ├── collatz_map.py               # Multi-backend Collatz functions (JAX/Numba/NumPy)
│   ├── koopman_operator.py          # Koopman matrix & closure construction
│   ├── tensor_utils.py              # Tensor SVD construction
│   ├── test_runner.py               # Unified test registry and executor
│   ├── visualizer.py                # Auto-plotting and tables
│   ├── results.py                   # JSON/CSV output
│   └── argparser.py                 # CLI argument parser
├── part_a/                          # Part A: Spectral Proofs
│   ├── structural_theorem.py
│   ├── ruelle_zeta.py
│   ├── essential_spectrum.py
│   ├── boundary_resolvent.py
│   └── chernoff_validation.py
├── part_b_obstructions/             # Part B: Obstruction Theory
│   ├── lyapunov_test.py
│   ├── modular_lemmas.py
│   ├── deep_brake_frequency.py
│   └── obstruction_tests.py
├── part_c_ergodic/                  # Part C: Ergodic Proof
│   ├── transition_table.py
│   ├── parity_analysis.py
│   ├── block_length_bound.py
│   └── deep_brake_frequency_test.py
├── part_d_tensor/                   # Part D: Tensor SVD
│   ├── tensor_construction.py
│   ├── tensor_evolution.py
│   └── cauchy_criterion.py
├── scripts/
│   ├── generate_figures.py          # Generate 5 paper figures (PDF)
│   └── profiling/                   # Performance profiling scripts (gitignored)
├── figures/                         # Output figures (PDF) — gitignored, keep .gitkeep
└── data/                            # Output data (JSON) — gitignored, keep .gitkeep
```

## Test Results

All 16 tests across all 4 parts pass successfully:

| Part | Tests | Status |
|------|-------|--------|
| A — Spectral Proofs | structural, ruelle, essential, boundary, chernoff | ✅ 5/5 |
| B — Obstruction Theory | lyapunov, lemmas, frequency, obstructions | ✅ 4/4 |
| C — Ergodic Proof | transition, parity, block_bound, fk_bound | ✅ 4/4 |
| D — Tensor Diagnostics | tensor_constr, tensor_evol, tensor_cauchy | ✅ 3/3 |

## Reproducibility

All tests use a fixed random seed by default (`--seed 42`). To verify reproducibility, run any test with a custom seed and compare the output:

```bash
python main.py --test parity --seed 123
python main.py --test parity --seed 456
```

## Numerical Limits

- **Part A (Closure):** Limited to ~50,000 vertices due to RAM constraints for dense matrix operations.
- **Parts B, C, D:** Support numbers up to 10¹² using `np.int64` and logarithmic scaling.

## Citation

```bibtex
@article{trikoz2026collatz,
  title={A Complete Proof of the Collatz Conjecture via Spectral Analysis and 2-Adic Ergodic Dynamics},
  author={Trikoz, Alexey},
  year={2026},
  month={May}
}
```

## AI Usage Statement

The author used **DeepSeek AI** as the primary mathematical co-investigator for code development, mathematical formalisation, and theorem generation. **Google Gemini** served as an adversarial reviewer throughout the development process.

## License

MIT License. See LICENSE file for details.
