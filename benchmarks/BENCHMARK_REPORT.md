# macmetalpy vs NumPy Benchmark Report

**Platform**: arm  
**Python**: 3.11.14  
**Date**: 2026-02-21 20:47  
**Warmup**: 2 runs  

## Category Summary

| Category | small speedup | Avg Speedup |
|---|---|---|
| bitwise | 0.36x | **0.36x** |
| comparison | 0.37x | **0.37x** |
| complex_ops | 0.45x | **0.45x** |
| creation | 0.56x | **0.56x** |
| creation_extra | 0.66x | **0.66x** |
| fft | 0.90x | **0.90x** |
| fft_extra | 0.75x | **0.75x** |
| indexing | 0.46x | **0.46x** |
| indexing_extra | 0.72x | **0.72x** |
| io | 1.00x | **1.00x** |
| linalg | 0.80x | **0.80x** |
| linalg_extra | 0.79x | **0.79x** |
| logic | 0.46x | **0.46x** |
| manipulation | 0.40x | **0.40x** |
| manipulation_extra | 0.53x | **0.53x** |
| math | 0.56x | **0.56x** |
| math_ext | 0.71x | **0.71x** |
| misc | 0.66x | **0.66x** |
| nan_extra | 0.84x | **0.84x** |
| nan_ops | 0.69x | **0.69x** |
| poly | 0.97x | **0.97x** |
| poly_extra | 0.67x | **0.67x** |
| random | 0.87x | **0.87x** |
| reduction | 0.51x | **0.51x** |
| reduction_extra | 0.66x | **0.66x** |
| set_extra | 0.97x | **0.97x** |
| set_ops | 0.93x | **0.93x** |
| sorting | 0.68x | **0.68x** |
| sorting_extra | 0.80x | **0.80x** |
| stats | 0.84x | **0.84x** |
| trig | 0.74x | **0.74x** |
| ufuncs_extra | 0.58x | **0.58x** |
| window | 0.88x | **0.88x** |

## Top 10: macmetalpy Fastest vs NumPy

| Rank | API | Category | Size | Speedup |
|------|-----|----------|------|---------|
| 1 | float_power | ufuncs_extra | small | **1.53x** |
| 2 | fabs | ufuncs_extra | small | **1.45x** |
| 3 | zeros_like | creation | small | **1.08x** |
| 4 | save_load | io | small | **1.07x** |
| 5 | i0 | math_ext | small | **1.04x** |
| 6 | eigvals | linalg_extra | small | **1.01x** |
| 7 | eig | linalg | small | **1.00x** |
| 8 | put_along_axis | indexing_extra | small | **1.00x** |
| 9 | cond | linalg_extra | small | **0.99x** |
| 10 | pinv | linalg_extra | small | **0.99x** |

## Top 10: macmetalpy Slowest vs NumPy

| Rank | API | Category | Size | Speedup |
|------|-----|----------|------|---------|
| 1 | searchsorted | sorting | small | **0.01x** |
| 2 | asarray | creation_extra | small | **0.08x** |
| 3 | flip | manipulation | small | **0.15x** |
| 4 | asanyarray | creation_extra | small | **0.17x** |
| 5 | divide | math | small | **0.17x** |
| 6 | imag | complex_ops | small | **0.18x** |
| 7 | concatenate | manipulation | small | **0.18x** |
| 8 | asfortranarray | manipulation_extra | small | **0.18x** |
| 9 | minimum | ufuncs_extra | small | **0.18x** |
| 10 | maximum | ufuncs_extra | small | **0.19x** |

## Category Aggregates

| Category | Benchmarks | GPU Wins | Avg Speedup | Median Speedup | Min | Max |
|----------|------------|----------|-------------|----------------|-----|-----|
| bitwise | 9 | 0 | 0.36x | 0.27x | 0.23x | 0.76x |
| comparison | 4 | 0 | 0.37x | 0.37x | 0.36x | 0.38x |
| complex_ops | 5 | 0 | 0.45x | 0.40x | 0.18x | 0.76x |
| creation | 10 | 1 | 0.56x | 0.52x | 0.21x | 1.08x |
| creation_extra | 18 | 0 | 0.66x | 0.76x | 0.08x | 0.99x |
| fft | 6 | 0 | 0.90x | 0.90x | 0.85x | 0.95x |
| fft_extra | 6 | 0 | 0.75x | 0.79x | 0.55x | 0.88x |
| indexing | 5 | 0 | 0.46x | 0.43x | 0.20x | 0.77x |
| indexing_extra | 19 | 1 | 0.72x | 0.81x | 0.24x | 1.00x |
| io | 2 | 1 | 1.00x | 1.00x | 0.92x | 1.07x |
| linalg | 10 | 1 | 0.80x | 0.87x | 0.48x | 1.00x |
| linalg_extra | 12 | 1 | 0.79x | 0.86x | 0.35x | 1.01x |
| logic | 16 | 0 | 0.46x | 0.36x | 0.25x | 0.94x |
| manipulation | 11 | 0 | 0.40x | 0.33x | 0.15x | 0.64x |
| manipulation_extra | 27 | 0 | 0.53x | 0.63x | 0.18x | 0.89x |
| math | 14 | 0 | 0.56x | 0.73x | 0.17x | 0.83x |
| math_ext | 11 | 1 | 0.71x | 0.81x | 0.29x | 1.04x |
| misc | 11 | 0 | 0.66x | 0.64x | 0.35x | 0.91x |
| nan_extra | 9 | 0 | 0.84x | 0.84x | 0.68x | 0.96x |
| nan_ops | 5 | 0 | 0.69x | 0.64x | 0.55x | 0.91x |
| poly | 3 | 0 | 0.97x | 0.98x | 0.94x | 0.99x |
| poly_extra | 7 | 0 | 0.67x | 0.52x | 0.43x | 0.99x |
| random | 5 | 0 | 0.87x | 0.91x | 0.74x | 0.97x |
| reduction | 13 | 0 | 0.51x | 0.49x | 0.32x | 0.72x |
| reduction_extra | 9 | 0 | 0.66x | 0.62x | 0.43x | 0.87x |
| set_extra | 1 | 0 | 0.97x | 0.97x | 0.97x | 0.97x |
| set_ops | 5 | 0 | 0.93x | 0.91x | 0.89x | 0.98x |
| sorting | 5 | 0 | 0.68x | 0.85x | 0.01x | 0.92x |
| sorting_extra | 3 | 0 | 0.80x | 0.80x | 0.78x | 0.81x |
| stats | 9 | 0 | 0.84x | 0.89x | 0.62x | 0.94x |
| trig | 15 | 0 | 0.74x | 0.74x | 0.67x | 0.83x |
| ufuncs_extra | 34 | 2 | 0.58x | 0.57x | 0.18x | 1.53x |
| window | 5 | 0 | 0.88x | 0.84x | 0.81x | 0.98x |

## Failed Benchmarks

| API | Category | Size | Error |
|-----|----------|------|-------|
| mask_indices | indexing_extra | small | TypeError: Unsupported dtype for Metal: object |

## Optimization Guidance

### Where macmetalpy is slower than NumPy

- **ufuncs_extra**: 32/34 benchmarks slower (avg 0.58x)
- **manipulation_extra**: 27/27 benchmarks slower (avg 0.53x)
- **creation_extra**: 18/18 benchmarks slower (avg 0.66x)
- **indexing_extra**: 18/19 benchmarks slower (avg 0.72x)
- **logic**: 16/16 benchmarks slower (avg 0.46x)
- **trig**: 15/15 benchmarks slower (avg 0.74x)
- **math**: 14/14 benchmarks slower (avg 0.56x)
- **reduction**: 13/13 benchmarks slower (avg 0.51x)
- **linalg_extra**: 11/12 benchmarks slower (avg 0.79x)
- **manipulation**: 11/11 benchmarks slower (avg 0.40x)
- **misc**: 11/11 benchmarks slower (avg 0.66x)
- **math_ext**: 10/11 benchmarks slower (avg 0.71x)
- **bitwise**: 9/9 benchmarks slower (avg 0.36x)
- **creation**: 9/10 benchmarks slower (avg 0.56x)
- **linalg**: 9/10 benchmarks slower (avg 0.80x)
- **nan_extra**: 9/9 benchmarks slower (avg 0.84x)
- **reduction_extra**: 9/9 benchmarks slower (avg 0.66x)
- **stats**: 9/9 benchmarks slower (avg 0.84x)
- **poly_extra**: 7/7 benchmarks slower (avg 0.67x)
- **fft**: 6/6 benchmarks slower (avg 0.90x)
- **fft_extra**: 6/6 benchmarks slower (avg 0.75x)
- **complex_ops**: 5/5 benchmarks slower (avg 0.45x)
- **indexing**: 5/5 benchmarks slower (avg 0.46x)
- **nan_ops**: 5/5 benchmarks slower (avg 0.69x)
- **random**: 5/5 benchmarks slower (avg 0.87x)
- **set_ops**: 5/5 benchmarks slower (avg 0.93x)
- **sorting**: 5/5 benchmarks slower (avg 0.68x)
- **window**: 5/5 benchmarks slower (avg 0.88x)
- **comparison**: 4/4 benchmarks slower (avg 0.37x)
- **poly**: 3/3 benchmarks slower (avg 0.97x)
- **sorting_extra**: 3/3 benchmarks slower (avg 0.80x)
- **io**: 1/2 benchmarks slower (avg 1.00x)
- **set_extra**: 1/1 benchmarks slower (avg 0.97x)

### General Observations

- **small** (1,000 elements): GPU wins 8/324, avg speedup 0.64x

### Recommendations

1. **Small arrays (<1K)**: GPU overhead often dominates; keep data on CPU for tiny operations.
2. **Medium arrays (~100K)**: GPU starts to show benefits for compute-heavy ops (math, FFT).
3. **Large arrays (1M+)**: GPU excels; batch operations where possible to amortize transfer cost.
4. **Transfer minimization**: Chain GPU operations to avoid repeated CPU<->GPU transfers.

