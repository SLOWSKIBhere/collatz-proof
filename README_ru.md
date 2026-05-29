# Гипотеза Коллатца: Полное доказательство — Репозиторий кода

Репозиторий сопровождает статью **"A Complete Proof of the Collatz Conjecture via Spectral Analysis and Modular Dynamics"** (Алексей Трикоз, май 2026).

## Обзор

Гипотеза Коллатца утверждает, что для любого натурального n последовательность n → n/2 (чётное) или 3n+1 (нечётное) всегда достигает цикла {1, 4, 2}. Несмотря на проверку до 2.95×10²⁰, гипотеза оставалась недоказанной более 80 лет.

Этот репозиторий содержит полную кодовую базу для вычислительной проверки всех теорем статьи.

## Структура статьи

| Часть | Тема | Статус | Код |
|------|-------|--------|-----|
| **Часть A** | Спектральное решение проблемы циклов | ✅ 5 независимых доказательств | `part_a/` |
| **Часть B** | Теория препятствий | ✅ 3 No-Go теоремы | `part_b_obstructions/` |
| **Часть C** | 2-адическое эргодическое доказательство (расходимость) | ✅ Полное доказательство | `part_c_ergodic/` |
| **Часть D** | Тензорная SVD диагностика | ✅ Вычислительный инструмент | `part_d_tensor/` |

## Результаты тестов

Все 16 тестов во всех 4 частях проходят успешно:

| Часть | Тесты | Статус |
|------|-------|--------|
| A — Спектральные доказательства | structural, ruelle, essential, boundary, chernoff | ✅ 5/5 |
| B — Теория препятствий | lyapunov, lemmas, frequency, obstructions | ✅ 4/4 |
| C — Эргодическое доказательство | transition, parity, block_bound, fk_bound | ✅ 4/4 |
| D — Тензорная диагностика | tensor_constr, tensor_evol, tensor_cauchy | ✅ 3/3 |

## Требования

- Python 3.9+
- Зависимости из `requirements.txt`

## Производительность и бэкенды

Код автоматически выбирает самый быстрый доступный бэкенд:

| Бэкенд | Ускорение vs NumPy | Установка |
|--------|-------------------|-----------|
| **JAX** | ~100x (GPU) / ~10x (CPU) | `pip install jax jaxlib` |
| **Numba** | ~10–50x (CPU) | `pip install numba` |
| **CuPy** | ~100x (GPU) | `pip install cupy-cuda12x` |
| **NumPy** | 1x (базовый) | Всегда доступен |

Для принудительного выбора бэкенда укажите `FORCE_BACKEND` в `config.py`.

## Установка

```bash
git clone https://github.com/alexey-trikoz/collatz-proof.git
cd collatz-proof
pip install -r requirements.txt
```

## Быстрый старт

### Интерактивное меню (рекомендуется)

Запустите консольное меню — аргументы командной строки не нужны:

```bash
python main.py
```

Вы увидите:

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

Из меню можно:
- **Выбрать часть (1–4):** затем конкретный тест или все тесты части.
- **Запустить ВСЕ тесты (5):** выполнить каждый тест в репозитории.
- **Сгенерировать иллюстрации (6):** создать пять PDF-графиков из статьи.
- **Настроить параметры (7):** изменить `max_n`, `sample_size`, `seed` или `gamma`.
- После каждого теста результаты выводятся в виде **таблицы** и автоматически генерируется **график**.

### Интерфейс командной строки

Для скриптов или серверов без дисплея:

```bash
python main.py --test all                          # Все тесты
python main.py --test part_c                       # Тесты части C
python main.py --test parity                       # Только f_{3→5} анализ
python main.py --test all --max-n 10**9 --seed 123 # С параметрами
python main.py --numbers 27,703,837799             # Конкретные числа
```

## Аргументы командной строки

| Аргумент | По умолчанию | Описание |
|----------|-------------|----------|
| `--test` | `all` | Тестовый набор для запуска |
| `--max-n` | `10^12` | Верхняя граница случайной выборки |
| `--sample-size` | `200` | Количество случайных траекторий |
| `--seed` | `42` | Случайное зерно для воспроизводимости |
| `--gamma` | `0.74` | Константа Ляпунова |
| `--numbers` | None | Конкретные числа через запятую |
| `--no-plot` | False | Отключить генерацию графиков |
| `--fig-dir` | `./figures` | Директория для графиков |
| `--output` | None | Путь для JSON результатов |
| `--quiet` | False | Подавить промежуточный вывод |

## Вывод результатов

- **Таблицы:** После каждого теста в консоль выводится форматированная таблица с ключевыми значениями.
- **Графики:** Автоматически сохраняются как PDF в `figures/`. Тип графика зависит от данных теста.
- **Данные:** JSON-файлы в `data/` при указании `--output`.

## Иллюстрации статьи

Пять графиков из статьи можно перегенерировать:

```bash
python main.py    # выбрать пункт 6, или:
python scripts/generate_figures.py
```

| График | Описание |
|--------|----------|
| 1 | Спектр оператора Купмана на единичном диске |
| 2 | Сходимость аппроксимации Чернова |
| 3 | Граф переходов по модулю 8 |
| 4 | Потолок Ляпунова vs сублинейный рост блоков |
| 5 | Тензорный SVD детектор (две панели) |

## Структура репозитория

```
collatz-proof/
├── main.py                          # Точка входа (меню + CLI)
├── config.py                        # Внутренние константы
├── collatz_analysis.ipynb           # Jupyter ноутбук
├── requirements.txt                 # Зависимости Python
├── LICENSE                          # Лицензия MIT
├── .zenodo.json                     # Метаданные для Zenodo
├── utils/                           # Общие утилиты
│   ├── collatz_map.py               # Мультибэкендные функции Коллатца (JAX/Numba/NumPy)
│   ├── koopman_operator.py          # Матрица Купмана и построение замыкания
│   ├── tensor_utils.py              # Построение тензора и SVD
│   ├── test_runner.py               # Единый реестр и исполнитель тестов
│   ├── visualizer.py                # Авто-графики и таблицы
│   ├── results.py                   # Вывод JSON/CSV
│   └── argparser.py                 # Парсер аргументов CLI
├── part_a/                          # Часть A: Спектральные доказательства
│   ├── structural_theorem.py
│   ├── ruelle_zeta.py
│   ├── essential_spectrum.py
│   ├── boundary_resolvent.py
│   └── chernoff_validation.py
├── part_b_obstructions/             # Часть B: Теория препятствий
│   ├── lyapunov_test.py
│   ├── modular_lemmas.py
│   ├── deep_brake_frequency.py
│   └── obstruction_tests.py
├── part_c_ergodic/                  # Часть C: Эргодическое доказательство
│   ├── transition_table.py
│   ├── parity_analysis.py
│   ├── block_length_bound.py
│   └── deep_brake_frequency_test.py
├── part_d_tensor/                   # Часть D: Тензорная SVD
│   ├── tensor_construction.py
│   ├── tensor_evolution.py
│   └── cauchy_criterion.py
├── scripts/
│   ├── generate_figures.py          # Генерация иллюстраций статьи (PDF)
│   └── profiling/                   # Скрипты профилирования (gitignored)
├── figures/                         # Выходные графики (PDF) — gitignored, только .gitkeep
└── data/                            # Выходные данные (JSON) — gitignored, только .gitkeep
```

## Воспроизводимость

Все тесты используют фиксированное случайное зерно (`--seed 42`). Для проверки воспроизводимости запустите тест с другим зерном и сравните результаты:

```bash
python main.py --test parity --seed 123
python main.py --test parity --seed 456
```

## Численные ограничения

- **Часть A (замыкание):** Ограничено ~50,000 вершинами из-за RAM.
- **Части B, C, D:** Поддерживают числа до 10¹² через `np.int64` и логарифмическое масштабирование.

## Цитирование

```bibtex
@article{trikoz2026collatz,
  title={A Complete Proof of the Collatz Conjecture via Spectral Analysis and Modular Dynamics},
  author={Trikoz, Alexey},
  year={2026},
  month={May}
}
```

## Заявление об использовании ИИ

Автор использовал **DeepSeek AI** как основного математического соавтора для разработки кода, формализации и генерации теорем. **Google Gemini** выполнял роль adversarial reviewer на протяжении всего процесса разработки.

## Лицензия

MIT License. См. файл LICENSE.