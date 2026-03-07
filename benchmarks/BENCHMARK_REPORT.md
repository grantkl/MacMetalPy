# macmetalpy vs NumPy Benchmark Report

**Platform**: arm  
**Python**: 3.11.14  
**Date**: 2026-03-07 08:10  
**Warmup**: 2 runs  

## Category Summary

| Category | small speedup | medium speedup | large speedup | Avg Speedup |
|---|---|---|---|---|
| arithmetic_f64 | 0.43x | 0.93x | 1.00x | **0.79x** |
| bitwise | 0.32x | 0.87x | 0.93x | **0.71x** |
| casting_f64 | 0.28x | 0.96x | 1.02x | **0.75x** |
| comparison | 0.28x | 0.90x | 0.97x | **0.72x** |
| comparison_f64 | 0.33x | 0.91x | 0.99x | **0.74x** |
| complex_ops | 0.46x | 0.67x | 0.68x | **0.60x** |
| creation | 0.44x | 1.44x | 6.33x | **2.74x** |
| creation_extra | 0.62x | 0.78x | 0.86x | **0.75x** |
| creation_f64 | 0.40x | 2.94x | 28.02x | **10.45x** |
| fft | 0.87x | 0.98x | 0.98x | **0.94x** |
| fft_extra | 0.76x | 1.05x | 1.16x | **0.99x** |
| indexing | 0.43x | 0.95x | 1.02x | **0.80x** |
| indexing_extra | 0.69x | 0.96x | 1.41x | **1.01x** |
| io | 0.96x | 0.97x | - | **0.96x** |
| linalg | 0.79x | 1.03x | - | **0.91x** |
| linalg_extra | 0.76x | 0.93x | 0.78x | **0.84x** |
| logic | 0.46x | 0.90x | 0.96x | **0.77x** |
| manipulation | 0.37x | 0.56x | 0.62x | **0.51x** |
| manipulation_extra | 0.50x | 0.88x | 4.06x | **1.81x** |
| math | 0.59x | 1.19x | 3.30x | **1.69x** |
| math_ext | 0.67x | 0.90x | 0.83x | **0.80x** |
| misc | 0.64x | 0.94x | 0.96x | **0.84x** |
| nan_extra | 0.81x | 0.91x | 0.91x | **0.88x** |
| nan_ops | 0.66x | 0.81x | 0.92x | **0.80x** |
| poly | 0.92x | 0.96x | 0.97x | **0.95x** |
| poly_extra | 0.59x | - | - | **0.59x** |
| random | 1.53x | 2.03x | 2.04x | **1.86x** |
| reduction | 0.56x | 0.94x | 0.88x | **0.79x** |
| reduction_extra | 0.63x | 1.03x | 1.39x | **1.02x** |
| reduction_f64 | 0.40x | 0.81x | 0.96x | **0.72x** |
| set_extra | 0.95x | 0.98x | 1.04x | **0.99x** |
| set_ops | 0.92x | 0.94x | 0.99x | **0.95x** |
| sorting | 0.63x | 1.55x | 7.87x | **3.35x** |
| sorting_extra | 0.78x | 1.02x | 0.96x | **0.92x** |
| stats | 0.82x | 0.93x | 0.94x | **0.89x** |
| trig | 0.75x | 1.03x | 3.88x | **1.88x** |
| trig_f64 | 0.69x | 1.00x | 1.05x | **0.91x** |
| ufuncs_extra | 0.61x | 1.48x | 3.36x | **1.82x** |
| window | 0.86x | 0.94x | 0.94x | **0.91x** |

## Top 10: macmetalpy Fastest vs NumPy

| Rank | API | Category | Size | Speedup |
|------|-----|----------|------|---------|
| 1 | array_f64 | creation_f64 | large | **108.92x** |
| 2 | copyto | manipulation_extra | large | **93.01x** |
| 3 | array | creation | large | **54.86x** |
| 4 | searchsorted | sorting | large | **32.50x** |
| 5 | logaddexp | ufuncs_extra | large | **11.52x** |
| 6 | logaddexp2 | ufuncs_extra | large | **11.35x** |
| 7 | put_along_axis | indexing_extra | large | **11.15x** |
| 8 | floor_divide | math | large | **11.12x** |
| 9 | fabs | ufuncs_extra | large | **10.42x** |
| 10 | fmod | ufuncs_extra | large | **9.91x** |

## Top 10: macmetalpy Slowest vs NumPy

| Rank | API | Category | Size | Speedup |
|------|-----|----------|------|---------|
| 1 | searchsorted | sorting | small | **0.01x** |
| 2 | asarray | creation_extra | large | **0.07x** |
| 3 | asanyarray | creation_extra | medium | **0.09x** |
| 4 | asanyarray | creation_extra | small | **0.09x** |
| 5 | asarray | creation_extra | small | **0.09x** |
| 6 | asanyarray | creation_extra | large | **0.09x** |
| 7 | asarray | creation_extra | medium | **0.09x** |
| 8 | flip | manipulation | small | **0.11x** |
| 9 | flip | manipulation | large | **0.13x** |
| 10 | flip | manipulation | medium | **0.13x** |

## Category Aggregates

| Category | Benchmarks | GPU Wins | Avg Speedup | Median Speedup | Min | Max |
|----------|------------|----------|-------------|----------------|-----|-----|
| arithmetic_f64 | 18 | 3 | 0.79x | 0.92x | 0.20x | 1.03x |
| bitwise | 27 | 0 | 0.71x | 0.89x | 0.21x | 0.98x |
| casting_f64 | 6 | 2 | 0.75x | 0.96x | 0.26x | 1.02x |
| comparison | 12 | 0 | 0.72x | 0.90x | 0.28x | 0.97x |
| comparison_f64 | 12 | 0 | 0.74x | 0.92x | 0.32x | 0.99x |
| complex_ops | 15 | 0 | 0.60x | 0.74x | 0.17x | 0.98x |
| creation | 30 | 5 | 2.74x | 0.83x | 0.19x | 54.86x |
| creation_extra | 51 | 2 | 0.75x | 0.88x | 0.07x | 1.55x |
| creation_f64 | 12 | 4 | 10.45x | 0.97x | 0.30x | 108.92x |
| fft | 18 | 1 | 0.94x | 0.97x | 0.77x | 1.01x |
| fft_extra | 18 | 5 | 0.99x | 0.93x | 0.53x | 1.67x |
| indexing | 15 | 3 | 0.80x | 0.90x | 0.18x | 1.27x |
| indexing_extra | 59 | 2 | 1.01x | 0.89x | 0.29x | 11.15x |
| io | 4 | 0 | 0.96x | 0.97x | 0.95x | 0.97x |
| linalg | 20 | 6 | 0.91x | 0.97x | 0.41x | 1.17x |
| linalg_extra | 27 | 7 | 0.84x | 0.95x | 0.29x | 1.07x |
| logic | 48 | 0 | 0.77x | 0.90x | 0.25x | 0.98x |
| manipulation | 33 | 0 | 0.51x | 0.50x | 0.11x | 0.97x |
| manipulation_extra | 81 | 2 | 1.81x | 0.62x | 0.17x | 93.01x |
| math | 42 | 11 | 1.69x | 0.93x | 0.33x | 11.12x |
| math_ext | 32 | 2 | 0.80x | 0.92x | 0.13x | 1.18x |
| misc | 31 | 1 | 0.84x | 0.90x | 0.33x | 1.26x |
| nan_extra | 27 | 1 | 0.88x | 0.88x | 0.66x | 1.23x |
| nan_ops | 15 | 0 | 0.80x | 0.84x | 0.51x | 0.96x |
| poly | 7 | 1 | 0.95x | 0.96x | 0.80x | 1.01x |
| poly_extra | 7 | 0 | 0.59x | 0.48x | 0.25x | 0.97x |
| random | 15 | 11 | 1.86x | 2.02x | 0.69x | 3.71x |
| reduction | 39 | 3 | 0.79x | 0.87x | 0.15x | 2.96x |
| reduction_extra | 27 | 3 | 1.02x | 0.86x | 0.14x | 6.50x |
| reduction_f64 | 24 | 1 | 0.72x | 0.82x | 0.25x | 1.00x |
| set_extra | 3 | 1 | 0.99x | 0.98x | 0.95x | 1.04x |
| set_ops | 15 | 1 | 0.95x | 0.96x | 0.89x | 1.08x |
| sorting | 15 | 4 | 3.35x | 0.91x | 0.01x | 32.50x |
| sorting_extra | 9 | 2 | 0.92x | 0.98x | 0.75x | 1.05x |
| stats | 24 | 0 | 0.89x | 0.90x | 0.62x | 0.99x |
| trig | 45 | 12 | 1.88x | 0.97x | 0.69x | 8.29x |
| trig_f64 | 18 | 7 | 0.91x | 0.99x | 0.42x | 1.16x |
| ufuncs_extra | 102 | 24 | 1.82x | 0.96x | 0.20x | 11.52x |
| window | 15 | 0 | 0.91x | 0.94x | 0.81x | 0.98x |

## Optimization Guidance

### Where macmetalpy is slower than NumPy

- **manipulation_extra**: 79/81 benchmarks slower (avg 1.81x)
- **ufuncs_extra**: 78/102 benchmarks slower (avg 1.82x)
- **indexing_extra**: 57/59 benchmarks slower (avg 1.01x)
- **creation_extra**: 49/51 benchmarks slower (avg 0.75x)
- **logic**: 48/48 benchmarks slower (avg 0.77x)
- **reduction**: 36/39 benchmarks slower (avg 0.79x)
- **manipulation**: 33/33 benchmarks slower (avg 0.51x)
- **trig**: 33/45 benchmarks slower (avg 1.88x)
- **math**: 31/42 benchmarks slower (avg 1.69x)
- **math_ext**: 30/32 benchmarks slower (avg 0.80x)
- **misc**: 30/31 benchmarks slower (avg 0.84x)
- **bitwise**: 27/27 benchmarks slower (avg 0.71x)
- **nan_extra**: 26/27 benchmarks slower (avg 0.88x)
- **creation**: 25/30 benchmarks slower (avg 2.74x)
- **reduction_extra**: 24/27 benchmarks slower (avg 1.02x)
- **stats**: 24/24 benchmarks slower (avg 0.89x)
- **reduction_f64**: 23/24 benchmarks slower (avg 0.72x)
- **linalg_extra**: 20/27 benchmarks slower (avg 0.84x)
- **fft**: 17/18 benchmarks slower (avg 0.94x)
- **arithmetic_f64**: 15/18 benchmarks slower (avg 0.79x)
- **complex_ops**: 15/15 benchmarks slower (avg 0.60x)
- **nan_ops**: 15/15 benchmarks slower (avg 0.80x)
- **window**: 15/15 benchmarks slower (avg 0.91x)
- **linalg**: 14/20 benchmarks slower (avg 0.91x)
- **set_ops**: 14/15 benchmarks slower (avg 0.95x)
- **fft_extra**: 13/18 benchmarks slower (avg 0.99x)
- **comparison**: 12/12 benchmarks slower (avg 0.72x)
- **comparison_f64**: 12/12 benchmarks slower (avg 0.74x)
- **indexing**: 12/15 benchmarks slower (avg 0.80x)
- **sorting**: 11/15 benchmarks slower (avg 3.35x)
- **trig_f64**: 11/18 benchmarks slower (avg 0.91x)
- **creation_f64**: 8/12 benchmarks slower (avg 10.45x)
- **poly_extra**: 7/7 benchmarks slower (avg 0.59x)
- **sorting_extra**: 7/9 benchmarks slower (avg 0.92x)
- **poly**: 6/7 benchmarks slower (avg 0.95x)
- **casting_f64**: 4/6 benchmarks slower (avg 0.75x)
- **io**: 4/4 benchmarks slower (avg 0.96x)
- **random**: 4/15 benchmarks slower (avg 1.86x)
- **set_extra**: 2/3 benchmarks slower (avg 0.99x)

### General Observations

- **small** (1,000 elements): GPU wins 6/355, avg speedup 0.62x
- **medium** (100,000 elements): GPU wins 46/346, avg speedup 1.04x
- **large** (1,000,000 elements): GPU wins 75/317, avg speedup 2.37x

### Recommendations

1. **Small arrays (<1K)**: GPU overhead often dominates; keep data on CPU for tiny operations.
2. **Medium arrays (~100K)**: GPU starts to show benefits for compute-heavy ops (math, FFT).
3. **Large arrays (1M+)**: GPU excels; batch operations where possible to amortize transfer cost.
4. **Transfer minimization**: Chain GPU operations to avoid repeated CPU<->GPU transfers.

