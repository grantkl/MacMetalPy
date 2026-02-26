# macmetalpy vs NumPy Benchmark Report

**Platform**: arm  
**Python**: 3.11.14  
**Date**: 2026-02-25 21:27  
**Warmup**: 2 runs  

## Category Summary

| Category | small speedup | medium speedup | large speedup | Avg Speedup |
|---|---|---|---|---|
| arithmetic_f64 | 0.42x | 0.92x | 0.95x | **0.76x** |
| bitwise | 0.32x | 0.84x | 0.87x | **0.68x** |
| casting_f64 | 0.28x | 0.93x | 0.98x | **0.73x** |
| comparison | 0.29x | 0.89x | 0.87x | **0.68x** |
| comparison_f64 | 0.34x | 0.91x | 0.90x | **0.72x** |
| complex_ops | 0.45x | 0.62x | 0.56x | **0.54x** |
| creation | 0.37x | 1.43x | 6.08x | **2.63x** |
| creation_extra | 0.63x | 0.77x | 0.84x | **0.74x** |
| creation_f64 | 0.39x | 2.94x | 28.35x | **10.56x** |
| fft | 0.81x | 0.84x | 0.78x | **0.81x** |
| fft_extra | 0.75x | 1.02x | 0.86x | **0.87x** |
| indexing | 0.41x | 0.95x | 0.98x | **0.78x** |
| indexing_extra | 0.68x | 0.95x | 1.40x | **1.01x** |
| io | 0.87x | 0.86x | - | **0.87x** |
| linalg | 0.74x | 0.95x | - | **0.84x** |
| linalg_extra | 0.73x | 0.87x | 0.70x | **0.79x** |
| logic | 0.45x | 0.88x | 0.94x | **0.76x** |
| manipulation | 0.33x | 0.52x | 0.56x | **0.47x** |
| manipulation_extra | 0.49x | 0.84x | 3.75x | **1.70x** |
| math | 0.52x | 1.14x | 3.13x | **1.60x** |
| math_ext | 0.67x | 0.90x | 0.79x | **0.79x** |
| misc | 0.64x | 0.93x | 0.94x | **0.83x** |
| nan_extra | 0.80x | 0.90x | 0.88x | **0.86x** |
| nan_ops | 0.64x | 0.77x | 0.85x | **0.75x** |
| poly | 0.92x | 0.90x | 0.82x | **0.89x** |
| poly_extra | 0.59x | - | - | **0.59x** |
| random | 1.50x | 1.91x | 1.90x | **1.77x** |
| reduction | 0.56x | 0.93x | 0.97x | **0.82x** |
| reduction_extra | 0.63x | 1.02x | 1.56x | **1.07x** |
| reduction_f64 | 0.44x | 0.80x | 0.82x | **0.69x** |
| set_extra | 0.94x | 0.92x | 0.76x | **0.87x** |
| set_ops | 0.88x | 0.88x | 0.59x | **0.79x** |
| sorting | 0.61x | 1.48x | 5.39x | **2.49x** |
| sorting_extra | 0.76x | 0.99x | 0.92x | **0.89x** |
| stats | 0.81x | 0.91x | 0.85x | **0.86x** |
| trig | 0.73x | 1.02x | 3.60x | **1.78x** |
| trig_f64 | 0.69x | 0.99x | 1.03x | **0.90x** |
| ufuncs_extra | 0.58x | 1.50x | 3.85x | **1.98x** |
| window | 0.85x | 0.90x | 0.83x | **0.86x** |

## Top 10: macmetalpy Fastest vs NumPy

| Rank | API | Category | Size | Speedup |
|------|-----|----------|------|---------|
| 1 | array_f64 | creation_f64 | large | **110.90x** |
| 2 | copyto | manipulation_extra | large | **87.51x** |
| 3 | array | creation | large | **53.57x** |
| 4 | searchsorted | sorting | large | **21.67x** |
| 5 | logaddexp | ufuncs_extra | large | **14.69x** |
| 6 | floor_divide | math | large | **13.85x** |
| 7 | fmod | ufuncs_extra | large | **13.21x** |
| 8 | put_along_axis | indexing_extra | large | **11.19x** |
| 9 | logaddexp2 | ufuncs_extra | large | **10.77x** |
| 10 | nextafter | ufuncs_extra | large | **10.76x** |

## Top 10: macmetalpy Slowest vs NumPy

| Rank | API | Category | Size | Speedup |
|------|-----|----------|------|---------|
| 1 | searchsorted | sorting | small | **0.01x** |
| 2 | asarray | creation_extra | large | **0.07x** |
| 3 | asanyarray | creation_extra | medium | **0.09x** |
| 4 | asanyarray | creation_extra | large | **0.09x** |
| 5 | asanyarray | creation_extra | small | **0.09x** |
| 6 | asarray | creation_extra | small | **0.09x** |
| 7 | asarray | creation_extra | medium | **0.09x** |
| 8 | zeros | creation | small | **0.10x** |
| 9 | flip | manipulation | small | **0.11x** |
| 10 | flip | manipulation | large | **0.13x** |

## Category Aggregates

| Category | Benchmarks | GPU Wins | Avg Speedup | Median Speedup | Min | Max |
|----------|------------|----------|-------------|----------------|-----|-----|
| arithmetic_f64 | 18 | 0 | 0.76x | 0.91x | 0.19x | 0.98x |
| bitwise | 27 | 0 | 0.68x | 0.87x | 0.20x | 0.92x |
| casting_f64 | 6 | 0 | 0.73x | 0.93x | 0.26x | 0.99x |
| comparison | 12 | 0 | 0.68x | 0.84x | 0.28x | 0.96x |
| comparison_f64 | 12 | 0 | 0.72x | 0.90x | 0.33x | 0.96x |
| complex_ops | 15 | 0 | 0.54x | 0.52x | 0.15x | 0.96x |
| creation | 30 | 5 | 2.63x | 0.73x | 0.10x | 53.57x |
| creation_extra | 51 | 2 | 0.74x | 0.87x | 0.07x | 1.51x |
| creation_f64 | 12 | 3 | 10.56x | 0.76x | 0.31x | 110.90x |
| fft | 18 | 0 | 0.81x | 0.84x | 0.66x | 0.88x |
| fft_extra | 18 | 4 | 0.87x | 0.86x | 0.46x | 1.42x |
| indexing | 15 | 2 | 0.78x | 0.89x | 0.18x | 1.26x |
| indexing_extra | 59 | 4 | 1.01x | 0.88x | 0.27x | 11.19x |
| io | 4 | 0 | 0.87x | 0.87x | 0.76x | 0.96x |
| linalg | 20 | 4 | 0.84x | 0.89x | 0.41x | 1.09x |
| linalg_extra | 27 | 2 | 0.79x | 0.88x | 0.28x | 1.03x |
| logic | 48 | 0 | 0.76x | 0.90x | 0.25x | 0.97x |
| manipulation | 33 | 0 | 0.47x | 0.45x | 0.11x | 0.91x |
| manipulation_extra | 81 | 2 | 1.70x | 0.59x | 0.17x | 87.51x |
| math | 42 | 10 | 1.60x | 0.89x | 0.33x | 13.85x |
| math_ext | 32 | 3 | 0.79x | 0.84x | 0.14x | 1.16x |
| misc | 31 | 1 | 0.83x | 0.90x | 0.33x | 1.26x |
| nan_extra | 27 | 1 | 0.86x | 0.87x | 0.68x | 1.22x |
| nan_ops | 15 | 0 | 0.75x | 0.81x | 0.51x | 0.88x |
| poly | 7 | 1 | 0.89x | 0.86x | 0.79x | 1.02x |
| poly_extra | 7 | 0 | 0.59x | 0.48x | 0.25x | 0.97x |
| random | 15 | 11 | 1.77x | 1.98x | 0.69x | 3.54x |
| reduction | 39 | 3 | 0.82x | 0.89x | 0.36x | 1.38x |
| reduction_extra | 27 | 2 | 1.07x | 0.86x | 0.42x | 6.60x |
| reduction_f64 | 24 | 0 | 0.69x | 0.74x | 0.30x | 0.95x |
| set_extra | 3 | 0 | 0.87x | 0.92x | 0.76x | 0.94x |
| set_ops | 15 | 0 | 0.79x | 0.86x | 0.46x | 0.95x |
| sorting | 15 | 4 | 2.49x | 0.84x | 0.01x | 21.67x |
| sorting_extra | 9 | 1 | 0.89x | 0.96x | 0.74x | 1.03x |
| stats | 24 | 0 | 0.86x | 0.88x | 0.60x | 0.97x |
| trig | 45 | 12 | 1.78x | 0.96x | 0.69x | 7.88x |
| trig_f64 | 18 | 7 | 0.90x | 0.97x | 0.42x | 1.15x |
| ufuncs_extra | 102 | 24 | 1.98x | 0.94x | 0.19x | 14.69x |
| window | 15 | 0 | 0.86x | 0.85x | 0.80x | 0.95x |

## Optimization Guidance

### Where macmetalpy is slower than NumPy

- **manipulation_extra**: 79/81 benchmarks slower (avg 1.70x)
- **ufuncs_extra**: 78/102 benchmarks slower (avg 1.98x)
- **indexing_extra**: 55/59 benchmarks slower (avg 1.01x)
- **creation_extra**: 49/51 benchmarks slower (avg 0.74x)
- **logic**: 48/48 benchmarks slower (avg 0.76x)
- **reduction**: 36/39 benchmarks slower (avg 0.82x)
- **manipulation**: 33/33 benchmarks slower (avg 0.47x)
- **trig**: 33/45 benchmarks slower (avg 1.78x)
- **math**: 32/42 benchmarks slower (avg 1.60x)
- **misc**: 30/31 benchmarks slower (avg 0.83x)
- **math_ext**: 29/32 benchmarks slower (avg 0.79x)
- **bitwise**: 27/27 benchmarks slower (avg 0.68x)
- **nan_extra**: 26/27 benchmarks slower (avg 0.86x)
- **creation**: 25/30 benchmarks slower (avg 2.63x)
- **linalg_extra**: 25/27 benchmarks slower (avg 0.79x)
- **reduction_extra**: 25/27 benchmarks slower (avg 1.07x)
- **reduction_f64**: 24/24 benchmarks slower (avg 0.69x)
- **stats**: 24/24 benchmarks slower (avg 0.86x)
- **arithmetic_f64**: 18/18 benchmarks slower (avg 0.76x)
- **fft**: 18/18 benchmarks slower (avg 0.81x)
- **linalg**: 16/20 benchmarks slower (avg 0.84x)
- **complex_ops**: 15/15 benchmarks slower (avg 0.54x)
- **nan_ops**: 15/15 benchmarks slower (avg 0.75x)
- **set_ops**: 15/15 benchmarks slower (avg 0.79x)
- **window**: 15/15 benchmarks slower (avg 0.86x)
- **fft_extra**: 14/18 benchmarks slower (avg 0.87x)
- **indexing**: 13/15 benchmarks slower (avg 0.78x)
- **comparison**: 12/12 benchmarks slower (avg 0.68x)
- **comparison_f64**: 12/12 benchmarks slower (avg 0.72x)
- **sorting**: 11/15 benchmarks slower (avg 2.49x)
- **trig_f64**: 11/18 benchmarks slower (avg 0.90x)
- **creation_f64**: 9/12 benchmarks slower (avg 10.56x)
- **sorting_extra**: 8/9 benchmarks slower (avg 0.89x)
- **poly_extra**: 7/7 benchmarks slower (avg 0.59x)
- **casting_f64**: 6/6 benchmarks slower (avg 0.73x)
- **poly**: 6/7 benchmarks slower (avg 0.89x)
- **io**: 4/4 benchmarks slower (avg 0.87x)
- **random**: 4/15 benchmarks slower (avg 1.77x)
- **set_extra**: 3/3 benchmarks slower (avg 0.87x)

### General Observations

- **small** (1,000 elements): GPU wins 6/355, avg speedup 0.60x
- **medium** (100,000 elements): GPU wins 41/346, avg speedup 1.02x
- **large** (1,000,000 elements): GPU wins 61/317, avg speedup 2.30x

### Recommendations

1. **Small arrays (<1K)**: GPU overhead often dominates; keep data on CPU for tiny operations.
2. **Medium arrays (~100K)**: GPU starts to show benefits for compute-heavy ops (math, FFT).
3. **Large arrays (1M+)**: GPU excels; batch operations where possible to amortize transfer cost.
4. **Transfer minimization**: Chain GPU operations to avoid repeated CPU<->GPU transfers.

