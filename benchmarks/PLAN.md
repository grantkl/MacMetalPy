# Plan: macmetalpy vs NumPy Performance Benchmarks

## Context
macmetalpy now has ~59 GPU-native Metal kernel operations (elementwise, reductions, predicates, bitwise, matmul). We need a benchmark suite to measure actual GPU speedup vs NumPy across different operation types and array sizes, identifying where Metal shines and where data-transfer overhead dominates.

## Approach
Single standalone benchmark script `benchmarks/bench_vs_numpy.py` using `time.perf_counter` (no external deps). Prints a formatted table with op name, array size, NumPy time, macmetalpy time, and speedup ratio.

## File
- **New**: `benchmarks/bench_vs_numpy.py`

## Design

### Structure
```
benchmarks/
  bench_vs_numpy.py    # standalone script, no deps beyond numpy + macmetalpy
```

### Benchmark categories (7 groups, ~30 individual ops)

1. **Elementwise unary** — sqrt, exp, log, sin, cos, abs, square, negative, floor, ceil
2. **Elementwise binary** — add, multiply, divide, power, maximum, minimum, floor_divide, fmod
3. **Reductions** — sum, max, min, mean, std, var, prod (full-array + axis)
4. **Matmul / dot** — matmul (square matrices), dot (1-D), vdot
5. **Comparisons + logic** — greater, equal, logical_and, logical_or
6. **Predicates** — isnan, isinf, isfinite
7. **Prefix scans** — cumsum, cumprod

### Array sizes
- Small: 1,000
- Medium: 100,000
- Large: 1,000,000
- XL: 10,000,000

For matmul: 64x64, 256x256, 1024x1024, 2048x2048

### Timing methodology
- **Warmup**: 3 runs discarded (ensures Metal shader compilation is cached)
- **Measured**: 10 runs, report median time
- **Synchronization**: For macmetalpy, call `.get()` after the operation to force GPU completion before stopping the timer (ensures fair comparison — we measure actual compute, not just dispatch)
- **Data creation**: Pre-create both NumPy and macmetalpy arrays outside the timing loop
- **dtype**: float32 (Metal's native type, fair comparison)

### Output format
```
macmetalpy vs NumPy Performance Benchmarks
==========================================
Platform: Apple M3 Max | dtype: float32

--- Elementwise Unary ---
Operation          Size         NumPy (ms)   Metal (ms)   Speedup
sqrt               1,000        0.003        0.012        0.25x
sqrt               100,000      0.031        0.015        2.07x
sqrt               1,000,000    0.312        0.042        7.43x
sqrt               10,000,000   3.120        0.380        8.21x
...

--- Summary ---
GPU wins: 24/30 operations at size >= 100K
Average speedup at 1M elements: 5.2x
Average speedup at 10M elements: 8.7x
Crossover point: ~10K-50K elements (below this, NumPy wins due to transfer overhead)
```

### Implementation details
- `bench_one(name, np_func, cp_func, sizes, warmup=3, repeats=10)` — core timing function
- Returns list of `(name, size, np_time, cp_time, speedup)` tuples
- `bench_matmul(sizes)` — special case for 2D matrix sizes
- `print_table(results, header)` — formatted output
- `main()` — runs all groups, prints combined report with summary stats
- Script is runnable: `python benchmarks/bench_vs_numpy.py`
- Optional `--sizes small,medium,large,xl` CLI arg to select size tiers
- Optional `--group elementwise,reductions,...` CLI arg to select categories

## Verification
```bash
# Run full benchmark suite
.venv/bin/python benchmarks/bench_vs_numpy.py

# Run specific group
.venv/bin/python benchmarks/bench_vs_numpy.py --group matmul

# Run only large sizes
.venv/bin/python benchmarks/bench_vs_numpy.py --sizes large,xl
```
