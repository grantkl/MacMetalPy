# macmetalpy vs NumPy Benchmark Report

**Platform**: arm  
**Python**: 3.11.14  
**Date**: 2026-02-23 21:21  
**Warmup**: 2 runs  

## Category Summary

| Category | small speedup | medium speedup | large speedup | Avg Speedup |
|---|---|---|---|---|
| arithmetic_f64 | 0.44x | 0.96x | 1.02x | **0.81x** |
| bitwise | 0.34x | 0.89x | 0.96x | **0.73x** |
| casting_f64 | 0.30x | 0.97x | 1.04x | **0.77x** |
| comparison | 0.32x | 0.93x | 1.02x | **0.76x** |
| comparison_f64 | 0.35x | 0.94x | 1.01x | **0.76x** |
| complex_ops | 0.43x | 0.66x | 0.68x | **0.59x** |
| creation | 0.52x | 1.98x | 10.73x | **4.41x** |
| creation_extra | 0.67x | 0.82x | 0.87x | **0.78x** |
| creation_f64 | 0.43x | 3.04x | 28.55x | **10.67x** |
| fft | 0.91x | 1.01x | 0.99x | **0.97x** |
| fft_extra | 0.78x | 0.91x | 1.02x | **0.90x** |
| indexing | 0.46x | 1.00x | 1.06x | **0.84x** |
| indexing_extra | 0.71x | 0.99x | 1.47x | **1.05x** |
| io | 0.99x | 1.02x | - | **1.01x** |
| linalg | 0.79x | 0.96x | - | **0.88x** |
| linalg_extra | 0.76x | 0.89x | 0.79x | **0.82x** |
| logic | 0.44x | 0.89x | 0.97x | **0.77x** |
| manipulation | 0.41x | 0.59x | 0.63x | **0.54x** |
| manipulation_extra | 0.53x | 0.91x | 4.32x | **1.92x** |
| math | 0.63x | 1.26x | 3.04x | **1.64x** |
| math_ext | 0.69x | 0.97x | 1.00x | **0.88x** |
| misc | 0.66x | 0.97x | 1.00x | **0.87x** |
| nan_extra | 0.86x | 0.98x | 0.99x | **0.94x** |
| nan_ops | 0.67x | 0.84x | 0.98x | **0.83x** |
| poly | 0.97x | 0.99x | 0.99x | **0.98x** |
| poly_extra | 0.66x | - | - | **0.66x** |
| random | 0.82x | 0.95x | 0.95x | **0.91x** |
| reduction | 0.51x | 0.96x | 1.03x | **0.84x** |
| reduction_extra | 0.68x | 1.06x | 1.43x | **1.05x** |
| reduction_f64 | 0.45x | 0.85x | 1.00x | **0.77x** |
| set_extra | 1.06x | 1.04x | 1.11x | **1.07x** |
| set_ops | 0.98x | 1.01x | 1.02x | **1.00x** |
| sorting | 0.68x | 1.60x | 6.21x | **2.83x** |
| sorting_extra | 0.82x | 1.07x | 1.01x | **0.97x** |
| stats | 0.84x | 0.96x | 0.98x | **0.92x** |
| trig | 0.78x | 1.05x | 3.44x | **1.76x** |
| trig_f64 | 0.71x | 1.02x | 1.08x | **0.94x** |
| ufuncs_extra | 0.62x | 1.48x | 2.69x | **1.60x** |
| window | 0.88x | 0.97x | 0.96x | **0.94x** |

## Top 10: macmetalpy Fastest vs NumPy

| Rank | API | Category | Size | Speedup |
|------|-----|----------|------|---------|
| 1 | array_f64 | creation_f64 | large | **110.91x** |
| 2 | copyto | manipulation_extra | large | **99.27x** |
| 3 | array | creation | large | **98.25x** |
| 4 | searchsorted | sorting | large | **24.43x** |
| 5 | put_along_axis | indexing_extra | large | **11.66x** |
| 6 | fabs | ufuncs_extra | large | **10.57x** |
| 7 | arccos | trig | large | **10.43x** |
| 8 | floor_divide | math | large | **10.32x** |
| 9 | array | creation | medium | **10.17x** |
| 10 | arcsin | trig | large | **10.10x** |

## Top 10: macmetalpy Slowest vs NumPy

| Rank | API | Category | Size | Speedup |
|------|-----|----------|------|---------|
| 1 | frombuffer | creation_extra | large | **0.00x** |
| 2 | searchsorted | sorting | small | **0.01x** |
| 3 | frombuffer | creation_extra | medium | **0.03x** |
| 4 | asarray | creation_extra | large | **0.10x** |
| 5 | flip | manipulation | small | **0.12x** |
| 6 | asanyarray | creation_extra | large | **0.13x** |
| 7 | flip | manipulation | large | **0.13x** |
| 8 | flip | manipulation | medium | **0.14x** |
| 9 | asanyarray | creation_extra | medium | **0.14x** |
| 10 | asarray | creation_extra | small | **0.14x** |

## Category Aggregates

| Category | Benchmarks | GPU Wins | Avg Speedup | Median Speedup | Min | Max |
|----------|------------|----------|-------------|----------------|-----|-----|
| arithmetic_f64 | 18 | 7 | 0.81x | 0.95x | 0.22x | 1.03x |
| bitwise | 27 | 3 | 0.73x | 0.90x | 0.21x | 1.02x |
| casting_f64 | 6 | 2 | 0.77x | 0.97x | 0.29x | 1.04x |
| comparison | 12 | 2 | 0.76x | 0.92x | 0.32x | 1.05x |
| comparison_f64 | 12 | 3 | 0.76x | 0.94x | 0.35x | 1.01x |
| complex_ops | 15 | 2 | 0.59x | 0.67x | 0.17x | 1.03x |
| creation | 30 | 8 | 4.41x | 0.91x | 0.20x | 98.25x |
| creation_extra | 51 | 11 | 0.78x | 0.93x | 0.00x | 1.63x |
| creation_f64 | 12 | 5 | 10.67x | 0.97x | 0.30x | 110.91x |
| fft | 18 | 9 | 0.97x | 1.00x | 0.86x | 1.01x |
| fft_extra | 18 | 5 | 0.90x | 0.91x | 0.54x | 1.17x |
| indexing | 15 | 6 | 0.84x | 0.99x | 0.21x | 1.28x |
| indexing_extra | 59 | 17 | 1.05x | 0.94x | 0.22x | 11.66x |
| io | 4 | 3 | 1.01x | 1.01x | 0.98x | 1.02x |
| linalg | 20 | 4 | 0.88x | 0.93x | 0.37x | 1.01x |
| linalg_extra | 27 | 6 | 0.82x | 0.95x | 0.30x | 1.03x |
| logic | 48 | 5 | 0.77x | 0.92x | 0.21x | 1.02x |
| manipulation | 33 | 3 | 0.54x | 0.56x | 0.12x | 1.05x |
| manipulation_extra | 81 | 6 | 1.92x | 0.65x | 0.15x | 99.27x |
| math | 42 | 23 | 1.64x | 1.01x | 0.16x | 10.32x |
| math_ext | 32 | 8 | 0.88x | 0.96x | 0.30x | 1.19x |
| misc | 31 | 10 | 0.87x | 0.92x | 0.33x | 1.41x |
| nan_extra | 27 | 4 | 0.94x | 0.94x | 0.71x | 1.33x |
| nan_ops | 15 | 0 | 0.83x | 0.87x | 0.51x | 0.99x |
| poly | 7 | 2 | 0.98x | 0.99x | 0.89x | 1.03x |
| poly_extra | 7 | 0 | 0.66x | 0.58x | 0.31x | 0.99x |
| random | 15 | 1 | 0.91x | 0.96x | 0.64x | 1.00x |
| reduction | 39 | 8 | 0.84x | 0.95x | 0.35x | 1.55x |
| reduction_extra | 27 | 6 | 1.05x | 0.95x | 0.42x | 4.93x |
| reduction_f64 | 24 | 6 | 0.77x | 0.84x | 0.31x | 1.07x |
| set_extra | 3 | 3 | 1.07x | 1.06x | 1.04x | 1.11x |
| set_ops | 15 | 10 | 1.00x | 1.01x | 0.94x | 1.12x |
| sorting | 15 | 8 | 2.83x | 1.01x | 0.01x | 24.43x |
| sorting_extra | 9 | 5 | 0.97x | 1.01x | 0.80x | 1.15x |
| stats | 24 | 4 | 0.92x | 0.92x | 0.62x | 1.03x |
| trig | 45 | 16 | 1.76x | 0.99x | 0.69x | 10.43x |
| trig_f64 | 18 | 11 | 0.94x | 1.02x | 0.48x | 1.18x |
| ufuncs_extra | 102 | 29 | 1.60x | 0.98x | 0.18x | 10.57x |
| window | 15 | 0 | 0.94x | 0.96x | 0.82x | 1.00x |

## Optimization Guidance

### Where macmetalpy is slower than NumPy

- **manipulation_extra**: 75/81 benchmarks slower (avg 1.92x)
- **ufuncs_extra**: 73/102 benchmarks slower (avg 1.60x)
- **logic**: 43/48 benchmarks slower (avg 0.77x)
- **indexing_extra**: 42/59 benchmarks slower (avg 1.05x)
- **creation_extra**: 40/51 benchmarks slower (avg 0.78x)
- **reduction**: 31/39 benchmarks slower (avg 0.84x)
- **manipulation**: 30/33 benchmarks slower (avg 0.54x)
- **trig**: 29/45 benchmarks slower (avg 1.76x)
- **bitwise**: 24/27 benchmarks slower (avg 0.73x)
- **math_ext**: 24/32 benchmarks slower (avg 0.88x)
- **nan_extra**: 23/27 benchmarks slower (avg 0.94x)
- **creation**: 22/30 benchmarks slower (avg 4.41x)
- **linalg_extra**: 21/27 benchmarks slower (avg 0.82x)
- **misc**: 21/31 benchmarks slower (avg 0.87x)
- **reduction_extra**: 21/27 benchmarks slower (avg 1.05x)
- **stats**: 20/24 benchmarks slower (avg 0.92x)
- **math**: 19/42 benchmarks slower (avg 1.64x)
- **reduction_f64**: 18/24 benchmarks slower (avg 0.77x)
- **linalg**: 16/20 benchmarks slower (avg 0.88x)
- **nan_ops**: 15/15 benchmarks slower (avg 0.83x)
- **window**: 15/15 benchmarks slower (avg 0.94x)
- **random**: 14/15 benchmarks slower (avg 0.91x)
- **complex_ops**: 13/15 benchmarks slower (avg 0.59x)
- **fft_extra**: 13/18 benchmarks slower (avg 0.90x)
- **arithmetic_f64**: 11/18 benchmarks slower (avg 0.81x)
- **comparison**: 10/12 benchmarks slower (avg 0.76x)
- **comparison_f64**: 9/12 benchmarks slower (avg 0.76x)
- **fft**: 9/18 benchmarks slower (avg 0.97x)
- **indexing**: 9/15 benchmarks slower (avg 0.84x)
- **creation_f64**: 7/12 benchmarks slower (avg 10.67x)
- **poly_extra**: 7/7 benchmarks slower (avg 0.66x)
- **sorting**: 7/15 benchmarks slower (avg 2.83x)
- **trig_f64**: 7/18 benchmarks slower (avg 0.94x)
- **poly**: 5/7 benchmarks slower (avg 0.98x)
- **set_ops**: 5/15 benchmarks slower (avg 1.00x)
- **casting_f64**: 4/6 benchmarks slower (avg 0.77x)
- **sorting_extra**: 4/9 benchmarks slower (avg 0.97x)
- **io**: 1/4 benchmarks slower (avg 1.01x)

### General Observations

- **small** (1,000 elements): GPU wins 9/355, avg speedup 0.63x
- **medium** (100,000 elements): GPU wins 92/346, avg speedup 1.06x
- **large** (1,000,000 elements): GPU wins 160/317, avg speedup 2.42x

### Recommendations

1. **Small arrays (<1K)**: GPU overhead often dominates; keep data on CPU for tiny operations.
2. **Medium arrays (~100K)**: GPU starts to show benefits for compute-heavy ops (math, FFT).
3. **Large arrays (1M+)**: GPU excels; batch operations where possible to amortize transfer cost.
4. **Transfer minimization**: Chain GPU operations to avoid repeated CPU<->GPU transfers.

