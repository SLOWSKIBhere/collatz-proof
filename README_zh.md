# 考拉兹猜想：实验代码仓库

> **状态：尚无证明（NO PROOF）。** 本仓库包含数值实验和诊断测试。测试通过或
> 验证任意有限范围，都不能证明所有正整数均收敛。

本仓库附属于论文 **"A Complete Proof of the Collatz Conjecture via Spectral Analysis and Modular Dynamics"**（Alexey Trikoz，2026年5月）。

## 概述

考拉兹猜想断言：对于任何正整数 n，序列 n → n/2（若为偶数）或 3n+1（若为奇数）最终必然到达循环 {1, 4, 2}。尽管已验证到 2.95×10²⁰，该猜想在80多年间始终未被证明。

本仓库提供论文中所有定理的完整计算验证代码。

## 论文结构

| 部分 | 主题 | 状态 | 代码 |
|------|------|------|------|
| **A部分** | 循环的谱解析 | ✅ 5个独立证明 | `part_a/` |
| **B部分** | 障碍理论 | ✅ 3个No-Go定理 | `part_b_obstructions/` |
| **C部分** | 2-adic遍历证明（发散性） | ✅ 完整证明 | `part_c_ergodic/` |
| **D部分** | 运动学张量SVD诊断 | ✅ 计算工具 | `part_d_tensor/` |

## 测试结果

全部4个部分共16个测试均通过：

| 部分 | 测试 | 状态 |
|------|------|------|
| A — 谱证明 | structural, ruelle, essential, boundary, chernoff | ✅ 5/5 |
| B — 障碍理论 | lyapunov, lemmas, frequency, obstructions | ✅ 4/4 |
| C — 遍历证明 | transition, parity, block_bound, fk_bound | ✅ 4/4 |
| D — 张量诊断 | tensor_constr, tensor_evol, tensor_cauchy | ✅ 3/3 |

## 系统要求

- Python 3.9+
- 依赖项见 `requirements.txt`

## 性能后端

代码自动选择最快的可用后端：

| 后端 | 相对NumPy加速 | 安装方式 |
|------|-------------|----------|
| **JAX** | ~100x (GPU) / ~10x (CPU) | `pip install jax jaxlib` |
| **Numba** | ~10–50x (CPU) | `pip install numba` |
| **CuPy** | ~100x (GPU) | `pip install cupy-cuda12x` |
| **NumPy** | 1x (基线) | 始终可用 |

如需强制指定后端，请在 `config.py` 中设置 `FORCE_BACKEND`。

## 安装

```bash
git clone https://github.com/alexey-trikoz/collatz-proof.git
cd collatz-proof
pip install -r requirements.txt
```

## 快速开始

### 交互式菜单（推荐）

启动控制台菜单 — 无需命令行参数：

```bash
python main.py
```

您将看到：

```
  COLLATZ CONJECTURE — COMPLETE PROOF
  Code Repository
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

从菜单中可以：
- **选择部分（1–4）：** 然后选择具体测试或运行该部分全部测试。
- **运行全部测试（5）：** 执行仓库中的每个测试。
- **生成论文插图（6）：** 生成论文中的5个PDF图表。
- **配置参数（7）：** 修改 `max_n`、`sample_size`、`seed` 或 `gamma`。
- 每个测试完成后，结果以 **格式化表格** 显示，并自动生成 **图表**。

### 命令行界面

适用于脚本或无显示器服务器：

```bash
python main.py --test all                          # 运行全部测试
python main.py --test part_c                       # 运行C部分测试
python main.py --test parity                       # 仅运行 f_{3→5} 分析
python main.py --test all --max-n 10**9 --seed 123 # 自定义参数
python main.py --numbers 27,703,837799             # 测试特定数字
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--test` | `all` | 要运行的测试套件 |
| `--max-n` | `10^12` | 随机采样的上限 |
| `--sample-size` | `200` | 随机轨迹数量 |
| `--seed` | `42` | 可复现的随机种子 |
| `--gamma` | `0.74` | Lyapunov常数 |
| `--numbers` | None | 逗号分隔的特定数字 |
| `--no-plot` | False | 禁用图表生成 |
| `--fig-dir` | `./figures` | 图表输出目录 |
| `--output` | None | JSON结果输出路径 |
| `--quiet` | False | 抑制中间输出 |

## 输出

- **表格：** 每个测试后，格式化的键值对表格打印到控制台。
- **图表：** 自动生成并保存为 `figures/` 中的PDF文件。图表类型取决于测试数据。
- **数据：** 当指定 `--output` 时，JSON文件保存到 `data/`。

## 论文插图

论文中的五张图表可以重新生成：

```bash
python main.py    # 选择第6项，或：
python scripts/generate_figures.py
```

| 图表 | 描述 |
|------|------|
| 1 | 单位圆盘上的Koopman算子谱 |
| 2 | Chernoff逼近收敛性 |
| 3 | 模8转移图 |
| 4 | Lyapunov上界 vs 次线性块增长 |
| 5 | 张量SVD检测器（双面板） |

## 仓库结构

```
collatz-proof/
├── main.py                          # 入口点（菜单 + CLI）
├── config.py                        # 内部常量
├── collatz_analysis.ipynb           # Jupyter笔记本
├── requirements.txt                 # Python依赖
├── LICENSE                          # MIT许可证
├── .zenodo.json                     # Zenodo元数据
├── utils/                           # 共享工具
│   ├── collatz_map.py               # 多后端Collatz函数（JAX/Numba/NumPy）
│   ├── koopman_operator.py          # Koopman矩阵与闭包构造
│   ├── tensor_utils.py              # 张量SVD构造
│   ├── test_runner.py               # 统一测试注册与执行器
│   ├── visualizer.py                # 自动绘图与表格
│   ├── results.py                   # JSON/CSV输出
│   └── argparser.py                 # CLI参数解析器
├── part_a/                          # A部分：谱证明
│   ├── structural_theorem.py
│   ├── ruelle_zeta.py
│   ├── essential_spectrum.py
│   ├── boundary_resolvent.py
│   └── chernoff_validation.py
├── part_b_obstructions/             # B部分：障碍理论
│   ├── lyapunov_test.py
│   ├── modular_lemmas.py
│   ├── deep_brake_frequency.py
│   └── obstruction_tests.py
├── part_c_ergodic/                  # C部分：遍历证明
│   ├── transition_table.py
│   ├── parity_analysis.py
│   ├── block_length_bound.py
│   └── deep_brake_frequency_test.py
├── part_d_tensor/                   # D部分：张量SVD
│   ├── tensor_construction.py
│   ├── tensor_evolution.py
│   └── cauchy_criterion.py
├── scripts/
│   ├── generate_figures.py          # 生成论文图表（PDF）
│   └── profiling/                   # 性能分析脚本（gitignored）
├── figures/                         # 输出图表（PDF）— gitignored，仅保留 .gitkeep
└── data/                            # 输出数据（JSON）— gitignored，仅保留 .gitkeep
```

## 可复现性

所有测试默认使用固定随机种子（`--seed 42`）。要验证可复现性，请使用不同种子运行测试并比较结果：

```bash
python main.py --test parity --seed 123
python main.py --test parity --seed 456
```

## 数值限制

- **A部分（闭包）：** 受RAM限制，约50,000个顶点。
- **B、C、D部分：** 通过 `np.int64` 和对数缩放，支持最大10¹²的数字。

## 引用

```bibtex
@article{trikoz2026collatz,
  title={A Complete Proof of the Collatz Conjecture via Spectral Analysis and Modular Dynamics},
  author={Trikoz, Alexey},
  year={2026},
  month={May}
}
```

## AI使用声明

作者使用 **DeepSeek AI** 作为主要数学合作研究者，参与代码开发、数学形式化和定理生成。**Google Gemini** 在整个开发过程中担任对抗性审阅者。

## 许可证

MIT License。详见 LICENSE 文件。
