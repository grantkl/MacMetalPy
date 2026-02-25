# macmetalpy vs NumPy Benchmark Report

**Platform**: arm  
**Python**: 3.11.14  
**Date**: 2026-02-24 06:30  
**Warmup**: 2 runs  

## Category Summary

| Category | small speedup | medium speedup | large speedup | Avg Speedup |
|---|---|---|---|---|
| arithmetic_f64 | 0.43x | 0.92x | 0.96x | **0.77x** |
| bitwise | 0.32x | 0.85x | 0.91x | **0.69x** |
| casting_f64 | 0.28x | 0.92x | 0.98x | **0.73x** |
| comparison | 0.29x | 0.90x | 0.91x | **0.70x** |
| comparison_f64 | 0.34x | 0.91x | 0.92x | **0.72x** |
| complex_ops | 0.45x | 0.61x | 0.57x | **0.54x** |
| creation | 0.45x | 1.39x | 6.45x | **2.76x** |
| creation_extra | 0.63x | 0.77x | 0.83x | **0.74x** |
| creation_f64 | 0.37x | 2.87x | 27.45x | **10.23x** |
| fft | 0.81x | 0.89x | 0.84x | **0.85x** |
| fft_extra | 0.78x | 0.79x | 0.66x | **0.74x** |
| indexing | 0.41x | 0.95x | 1.00x | **0.79x** |
| indexing_extra | 0.69x | 0.95x | 1.40x | **1.00x** |
| io | 0.92x | 0.94x | - | **0.93x** |
| linalg | 0.74x | 0.96x | - | **0.85x** |
| linalg_extra | 0.73x | 0.88x | 0.72x | **0.79x** |
| logic | 0.46x | 0.87x | 0.94x | **0.75x** |
| manipulation | 0.36x | 0.54x | 0.57x | **0.49x** |
| manipulation_extra | 0.50x | 0.83x | 3.77x | **1.70x** |
| math | 0.56x | 0.90x | 0.91x | **0.79x** |
| math_ext | 0.70x | 0.87x | 0.80x | **0.79x** |
| misc | 0.64x | 0.93x | 0.94x | **0.83x** |
| nan_extra | 0.80x | 0.91x | 0.91x | **0.87x** |
| nan_ops | 0.65x | 0.79x | 0.81x | **0.75x** |
| poly | 0.92x | 0.92x | 0.90x | **0.91x** |
| poly_extra | 0.59x | - | - | **0.59x** |
| random | 1.37x | 1.91x | 1.91x | **1.73x** |
| reduction | 0.47x | 0.89x | 0.97x | **0.77x** |
| reduction_extra | 0.66x | 0.89x | 0.93x | **0.82x** |
| reduction_f64 | 0.44x | 0.80x | 0.84x | **0.69x** |
| set_extra | 0.88x | 0.93x | 0.82x | **0.88x** |
| set_ops | 0.91x | 0.91x | 0.65x | **0.82x** |
| sorting | 0.74x | 0.87x | 0.84x | **0.82x** |
| sorting_extra | 0.78x | 0.99x | 0.94x | **0.91x** |
| stats | 0.81x | 0.91x | 0.85x | **0.86x** |
| trig | 0.73x | 0.95x | 0.94x | **0.88x** |
| trig_f64 | 0.68x | 0.98x | 1.03x | **0.90x** |
| ufuncs_extra | 0.58x | 1.19x | 1.25x | **1.00x** |
| window | 0.85x | 0.92x | 0.86x | **0.88x** |

## Top 10: macmetalpy Fastest vs NumPy

| Rank | API | Category | Size | Speedup |
|------|-----|----------|------|---------|
| 1 | array_f64 | creation_f64 | large | **106.92x** |
| 2 | copyto | manipulation_extra | large | **87.63x** |
| 3 | array | creation | large | **56.14x** |
| 4 | put_along_axis | indexing_extra | large | **11.15x** |
| 5 | fabs | ufuncs_extra | large | **10.10x** |
| 6 | fabs | ufuncs_extra | medium | **9.17x** |
| 7 | array_f64 | creation_f64 | medium | **9.09x** |
| 8 | copyto | manipulation_extra | medium | **7.21x** |
| 9 | array | creation | medium | **5.55x** |
| 10 | randn | random | medium | **3.51x** |

## Top 10: macmetalpy Slowest vs NumPy

| Rank | API | Category | Size | Speedup |
|------|-----|----------|------|---------|
| 1 | asarray | creation_extra | large | **0.07x** |
| 2 | asanyarray | creation_extra | medium | **0.09x** |
| 3 | asarray | creation_extra | small | **0.09x** |
| 4 | asarray | creation_extra | medium | **0.09x** |
| 5 | asanyarray | creation_extra | small | **0.09x** |
| 6 | asanyarray | creation_extra | large | **0.09x** |
| 7 | flip | manipulation | small | **0.11x** |
| 8 | flip | manipulation | large | **0.13x** |
| 9 | flip | manipulation | medium | **0.13x** |
| 10 | divide | math | small | **0.15x** |

## Category Aggregates

| Category | Benchmarks | GPU Wins | Avg Speedup | Median Speedup | Min | Max |
|----------|------------|----------|-------------|----------------|-----|-----|
| arithmetic_f64 | 18 | 0 | 0.77x | 0.91x | 0.22x | 0.98x |
| bitwise | 27 | 0 | 0.69x | 0.87x | 0.21x | 0.95x |
| casting_f64 | 6 | 1 | 0.73x | 0.92x | 0.26x | 1.02x |
| comparison | 12 | 0 | 0.70x | 0.90x | 0.29x | 0.96x |
| comparison_f64 | 12 | 0 | 0.72x | 0.90x | 0.33x | 0.97x |
| complex_ops | 15 | 0 | 0.54x | 0.49x | 0.15x | 0.96x |
| creation | 30 | 5 | 2.76x | 0.74x | 0.19x | 56.14x |
| creation_extra | 51 | 2 | 0.74x | 0.87x | 0.07x | 1.29x |
| creation_f64 | 12 | 4 | 10.23x | 0.81x | 0.30x | 106.92x |
| fft | 18 | 0 | 0.85x | 0.86x | 0.66x | 0.92x |
| fft_extra | 18 | 0 | 0.74x | 0.74x | 0.45x | 0.96x |
| indexing | 15 | 2 | 0.79x | 0.91x | 0.18x | 1.26x |
| indexing_extra | 59 | 2 | 1.00x | 0.88x | 0.27x | 11.15x |
| io | 4 | 0 | 0.93x | 0.92x | 0.90x | 0.97x |
| linalg | 20 | 4 | 0.85x | 0.92x | 0.33x | 1.11x |
| linalg_extra | 27 | 2 | 0.79x | 0.88x | 0.24x | 1.03x |
| logic | 48 | 0 | 0.75x | 0.88x | 0.25x | 0.97x |
| manipulation | 33 | 0 | 0.49x | 0.51x | 0.11x | 0.93x |
| manipulation_extra | 81 | 2 | 1.70x | 0.58x | 0.17x | 87.63x |
| math | 42 | 0 | 0.79x | 0.89x | 0.15x | 0.97x |
| math_ext | 32 | 0 | 0.79x | 0.84x | 0.28x | 0.97x |
| misc | 31 | 2 | 0.83x | 0.91x | 0.33x | 1.26x |
| nan_extra | 27 | 1 | 0.87x | 0.88x | 0.67x | 1.20x |
| nan_ops | 15 | 0 | 0.75x | 0.78x | 0.49x | 0.91x |
| poly | 7 | 1 | 0.91x | 0.92x | 0.78x | 1.03x |
| poly_extra | 7 | 0 | 0.59x | 0.48x | 0.25x | 0.96x |
| random | 15 | 11 | 1.73x | 1.97x | 0.56x | 3.51x |
| reduction | 39 | 2 | 0.77x | 0.86x | 0.31x | 1.38x |
| reduction_extra | 27 | 0 | 0.82x | 0.86x | 0.47x | 0.99x |
| reduction_f64 | 24 | 0 | 0.69x | 0.76x | 0.30x | 0.96x |
| set_extra | 3 | 0 | 0.88x | 0.88x | 0.82x | 0.93x |
| set_ops | 15 | 0 | 0.82x | 0.89x | 0.50x | 0.95x |
| sorting | 15 | 0 | 0.82x | 0.86x | 0.67x | 0.91x |
| sorting_extra | 9 | 1 | 0.91x | 0.96x | 0.74x | 1.03x |
| stats | 24 | 0 | 0.86x | 0.88x | 0.60x | 0.97x |
| trig | 45 | 0 | 0.88x | 0.93x | 0.67x | 0.97x |
| trig_f64 | 18 | 6 | 0.90x | 0.97x | 0.42x | 1.15x |
| ufuncs_extra | 102 | 6 | 1.00x | 0.91x | 0.17x | 10.10x |
| window | 15 | 0 | 0.88x | 0.89x | 0.79x | 0.94x |

## Optimization Guidance

### Where macmetalpy is slower than NumPy

- **ufuncs_extra**: 96/102 benchmarks slower (avg 1.00x)
- **manipulation_extra**: 79/81 benchmarks slower (avg 1.70x)
- **indexing_extra**: 57/59 benchmarks slower (avg 1.00x)
- **creation_extra**: 49/51 benchmarks slower (avg 0.74x)
- **logic**: 48/48 benchmarks slower (avg 0.75x)
- **trig**: 45/45 benchmarks slower (avg 0.88x)
- **math**: 42/42 benchmarks slower (avg 0.79x)
- **reduction**: 37/39 benchmarks slower (avg 0.77x)
- **manipulation**: 33/33 benchmarks slower (avg 0.49x)
- **math_ext**: 32/32 benchmarks slower (avg 0.79x)
- **misc**: 29/31 benchmarks slower (avg 0.83x)
- **bitwise**: 27/27 benchmarks slower (avg 0.69x)
- **reduction_extra**: 27/27 benchmarks slower (avg 0.82x)
- **nan_extra**: 26/27 benchmarks slower (avg 0.87x)
- **creation**: 25/30 benchmarks slower (avg 2.76x)
- **linalg_extra**: 25/27 benchmarks slower (avg 0.79x)
- **reduction_f64**: 24/24 benchmarks slower (avg 0.69x)
- **stats**: 24/24 benchmarks slower (avg 0.86x)
- **arithmetic_f64**: 18/18 benchmarks slower (avg 0.77x)
- **fft**: 18/18 benchmarks slower (avg 0.85x)
- **fft_extra**: 18/18 benchmarks slower (avg 0.74x)
- **linalg**: 16/20 benchmarks slower (avg 0.85x)
- **complex_ops**: 15/15 benchmarks slower (avg 0.54x)
- **nan_ops**: 15/15 benchmarks slower (avg 0.75x)
- **set_ops**: 15/15 benchmarks slower (avg 0.82x)
- **sorting**: 15/15 benchmarks slower (avg 0.82x)
- **window**: 15/15 benchmarks slower (avg 0.88x)
- **indexing**: 13/15 benchmarks slower (avg 0.79x)
- **comparison**: 12/12 benchmarks slower (avg 0.70x)
- **comparison_f64**: 12/12 benchmarks slower (avg 0.72x)
- **trig_f64**: 12/18 benchmarks slower (avg 0.90x)
- **creation_f64**: 8/12 benchmarks slower (avg 10.23x)
- **sorting_extra**: 8/9 benchmarks slower (avg 0.91x)
- **poly_extra**: 7/7 benchmarks slower (avg 0.59x)
- **poly**: 6/7 benchmarks slower (avg 0.91x)
- **casting_f64**: 5/6 benchmarks slower (avg 0.73x)
- **io**: 4/4 benchmarks slower (avg 0.93x)
- **random**: 4/15 benchmarks slower (avg 1.73x)
- **set_extra**: 3/3 benchmarks slower (avg 0.88x)

### General Observations

- **small** (1,000 elements): GPU wins 7/355, avg speedup 0.61x
- **medium** (100,000 elements): GPU wins 24/346, avg speedup 0.95x
- **large** (1,000,000 elements): GPU wins 23/317, avg speedup 1.71x

### Recommendations

1. **Small arrays (<1K)**: GPU overhead often dominates; keep data on CPU for tiny operations.
2. **Medium arrays (~100K)**: GPU starts to show benefits for compute-heavy ops (math, FFT).
3. **Large arrays (1M+)**: GPU excels; batch operations where possible to amortize transfer cost.
4. **Transfer minimization**: Chain GPU operations to avoid repeated CPU<->GPU transfers.

