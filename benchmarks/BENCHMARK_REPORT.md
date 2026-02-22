# macmetalpy vs NumPy Benchmark Report

**Platform**: arm  
**Python**: 3.11.14  
**Date**: 2026-02-22 07:56  
**Warmup**: 2 runs  

## Category Summary

| Category | small speedup | medium speedup | large speedup | Avg Speedup |
|---|---|---|---|---|
| arithmetic_f64 | 0.48x | 0.96x | 0.99x | **0.81x** |
| bitwise | 0.36x | 0.89x | 0.95x | **0.73x** |
| casting_f64 | 0.29x | 0.92x | 0.97x | **0.73x** |
| comparison | 0.33x | 0.94x | 0.99x | **0.75x** |
| comparison_f64 | 0.37x | 0.94x | 0.99x | **0.77x** |
| complex_ops | 0.45x | 0.65x | 0.67x | **0.59x** |
| creation | 0.55x | 1.91x | 10.89x | **4.45x** |
| creation_extra | 0.67x | 0.79x | 0.82x | **0.76x** |
| creation_f64 | 0.41x | 3.07x | 28.22x | **10.57x** |
| fft | 0.91x | 1.00x | 1.00x | **0.97x** |
| fft_extra | 0.78x | 0.91x | 0.92x | **0.87x** |
| indexing | 0.48x | 1.00x | 1.05x | **0.84x** |
| indexing_extra | 0.72x | 1.01x | 1.55x | **1.09x** |
| io | 0.98x | 0.99x | - | **0.98x** |
| linalg | 0.81x | 0.97x | - | **0.89x** |
| linalg_extra | 0.78x | 0.89x | 0.78x | **0.83x** |
| logic | 0.45x | 0.88x | 0.98x | **0.77x** |
| manipulation | 0.41x | 0.58x | 0.63x | **0.54x** |
| manipulation_extra | 0.53x | 0.93x | 4.44x | **1.97x** |
| math | 0.54x | 1.18x | 2.74x | **1.49x** |
| math_ext | 0.71x | 0.95x | 0.98x | **0.88x** |
| misc | 0.66x | 0.93x | 0.98x | **0.85x** |
| nan_extra | 0.83x | 0.93x | 0.98x | **0.91x** |
| nan_ops | 0.70x | 0.85x | 0.96x | **0.84x** |
| poly | 0.94x | 0.99x | 0.99x | **0.97x** |
| poly_extra | 0.68x | - | - | **0.68x** |
| random | 0.85x | 0.97x | 0.97x | **0.93x** |
| reduction | 0.52x | 0.90x | 0.98x | **0.80x** |
| reduction_extra | 0.68x | 1.04x | 1.49x | **1.07x** |
| reduction_f64 | 0.46x | 0.83x | 0.97x | **0.75x** |
| set_extra | 0.96x | 1.00x | 1.00x | **0.99x** |
| set_ops | 0.93x | 0.99x | 1.01x | **0.98x** |
| sorting | 0.68x | 1.62x | 4.60x | **2.30x** |
| sorting_extra | 0.78x | 0.98x | 0.99x | **0.91x** |
| stats | 0.85x | 0.96x | 0.97x | **0.92x** |
| trig | 0.71x | 1.04x | 3.45x | **1.73x** |
| trig_f64 | 0.72x | 0.99x | 0.99x | **0.90x** |
| ufuncs_extra | 0.56x | 1.49x | 2.96x | **1.67x** |
| window | 0.88x | 0.96x | 0.96x | **0.93x** |

## Top 10: macmetalpy Fastest vs NumPy

| Rank | API | Category | Size | Speedup |
|------|-----|----------|------|---------|
| 1 | array_f64 | creation_f64 | large | **109.92x** |
| 2 | copyto | manipulation_extra | large | **102.74x** |
| 3 | array | creation | large | **100.28x** |
| 4 | searchsorted | sorting | large | **16.73x** |
| 5 | put_along_axis | indexing_extra | large | **12.77x** |
| 6 | fabs | ufuncs_extra | large | **10.39x** |
| 7 | arccos | trig | large | **10.34x** |
| 8 | array | creation | medium | **10.23x** |
| 9 | arcsin | trig | large | **10.08x** |
| 10 | heaviside | ufuncs_extra | large | **9.87x** |

## Top 10: macmetalpy Slowest vs NumPy

| Rank | API | Category | Size | Speedup |
|------|-----|----------|------|---------|
| 1 | frombuffer | creation_extra | large | **0.00x** |
| 2 | searchsorted | sorting | small | **0.01x** |
| 3 | frombuffer | creation_extra | medium | **0.03x** |
| 4 | asanyarray | creation_extra | small | **0.08x** |
| 5 | asfortranarray | manipulation_extra | large | **0.08x** |
| 6 | asfortranarray | manipulation_extra | medium | **0.08x** |
| 7 | asanyarray | creation_extra | medium | **0.11x** |
| 8 | asarray | creation_extra | large | **0.12x** |
| 9 | flip | manipulation | small | **0.13x** |
| 10 | flip | manipulation | large | **0.14x** |

## Category Aggregates

| Category | Benchmarks | GPU Wins | Avg Speedup | Median Speedup | Min | Max |
|----------|------------|----------|-------------|----------------|-----|-----|
| arithmetic_f64 | 18 | 1 | 0.81x | 0.95x | 0.24x | 1.00x |
| bitwise | 27 | 0 | 0.73x | 0.90x | 0.23x | 1.00x |
| casting_f64 | 6 | 0 | 0.73x | 0.92x | 0.26x | 0.99x |
| comparison | 12 | 1 | 0.75x | 0.93x | 0.33x | 1.00x |
| comparison_f64 | 12 | 0 | 0.77x | 0.94x | 0.36x | 1.00x |
| complex_ops | 15 | 0 | 0.59x | 0.72x | 0.17x | 1.00x |
| creation | 30 | 7 | 4.45x | 0.89x | 0.23x | 100.28x |
| creation_extra | 51 | 2 | 0.76x | 0.91x | 0.00x | 1.01x |
| creation_f64 | 12 | 3 | 10.57x | 0.97x | 0.27x | 109.92x |
| fft | 18 | 6 | 0.97x | 1.00x | 0.86x | 1.01x |
| fft_extra | 18 | 0 | 0.87x | 0.90x | 0.57x | 1.00x |
| indexing | 15 | 3 | 0.84x | 0.99x | 0.21x | 1.26x |
| indexing_extra | 56 | 9 | 1.09x | 0.96x | 0.23x | 12.77x |
| io | 4 | 0 | 0.98x | 0.99x | 0.96x | 1.00x |
| linalg | 20 | 3 | 0.89x | 0.97x | 0.48x | 1.00x |
| linalg_extra | 27 | 5 | 0.83x | 0.96x | 0.30x | 1.02x |
| logic | 48 | 0 | 0.77x | 0.92x | 0.23x | 1.00x |
| manipulation | 33 | 0 | 0.54x | 0.59x | 0.13x | 0.99x |
| manipulation_extra | 81 | 3 | 1.97x | 0.67x | 0.08x | 102.74x |
| math | 42 | 11 | 1.49x | 0.97x | 0.17x | 8.87x |
| math_ext | 32 | 2 | 0.88x | 0.97x | 0.30x | 1.00x |
| misc | 31 | 4 | 0.85x | 0.94x | 0.33x | 1.11x |
| nan_extra | 27 | 0 | 0.91x | 0.95x | 0.69x | 0.99x |
| nan_ops | 15 | 0 | 0.84x | 0.92x | 0.56x | 0.98x |
| poly | 7 | 1 | 0.97x | 0.99x | 0.86x | 1.00x |
| poly_extra | 7 | 0 | 0.68x | 0.60x | 0.35x | 0.99x |
| random | 15 | 2 | 0.93x | 0.95x | 0.69x | 1.01x |
| reduction | 39 | 2 | 0.80x | 0.93x | 0.35x | 1.00x |
| reduction_extra | 27 | 3 | 1.07x | 0.96x | 0.44x | 5.52x |
| reduction_f64 | 24 | 1 | 0.75x | 0.84x | 0.33x | 1.01x |
| set_extra | 3 | 1 | 0.99x | 1.00x | 0.96x | 1.00x |
| set_ops | 15 | 4 | 0.98x | 0.99x | 0.90x | 1.07x |
| sorting | 15 | 5 | 2.30x | 0.99x | 0.01x | 16.73x |
| sorting_extra | 9 | 0 | 0.91x | 0.97x | 0.75x | 1.00x |
| stats | 24 | 1 | 0.92x | 0.95x | 0.63x | 1.00x |
| trig | 45 | 12 | 1.73x | 0.98x | 0.64x | 10.34x |
| trig_f64 | 18 | 5 | 0.90x | 0.98x | 0.50x | 1.01x |
| ufuncs_extra | 102 | 27 | 1.67x | 0.95x | 0.17x | 10.39x |
| window | 15 | 0 | 0.93x | 0.96x | 0.81x | 0.99x |

## Failed Benchmarks

| API | Category | Size | Error |
|-----|----------|------|-------|
| mask_indices | indexing_extra | small | TypeError: Unsupported dtype for Metal: object |
| mask_indices | indexing_extra | medium | TypeError: Unsupported dtype for Metal: object |
| mask_indices | indexing_extra | large | TypeError: Unsupported dtype for Metal: object |

## Optimization Guidance

### Where macmetalpy is slower than NumPy

- **manipulation_extra**: 78/81 benchmarks slower (avg 1.97x)
- **ufuncs_extra**: 75/102 benchmarks slower (avg 1.67x)
- **creation_extra**: 49/51 benchmarks slower (avg 0.76x)
- **logic**: 48/48 benchmarks slower (avg 0.77x)
- **indexing_extra**: 47/56 benchmarks slower (avg 1.09x)
- **reduction**: 37/39 benchmarks slower (avg 0.80x)
- **manipulation**: 33/33 benchmarks slower (avg 0.54x)
- **trig**: 33/45 benchmarks slower (avg 1.73x)
- **math**: 31/42 benchmarks slower (avg 1.49x)
- **math_ext**: 30/32 benchmarks slower (avg 0.88x)
- **bitwise**: 27/27 benchmarks slower (avg 0.73x)
- **misc**: 27/31 benchmarks slower (avg 0.85x)
- **nan_extra**: 27/27 benchmarks slower (avg 0.91x)
- **reduction_extra**: 24/27 benchmarks slower (avg 1.07x)
- **creation**: 23/30 benchmarks slower (avg 4.45x)
- **reduction_f64**: 23/24 benchmarks slower (avg 0.75x)
- **stats**: 23/24 benchmarks slower (avg 0.92x)
- **linalg_extra**: 22/27 benchmarks slower (avg 0.83x)
- **fft_extra**: 18/18 benchmarks slower (avg 0.87x)
- **arithmetic_f64**: 17/18 benchmarks slower (avg 0.81x)
- **linalg**: 17/20 benchmarks slower (avg 0.89x)
- **complex_ops**: 15/15 benchmarks slower (avg 0.59x)
- **nan_ops**: 15/15 benchmarks slower (avg 0.84x)
- **window**: 15/15 benchmarks slower (avg 0.93x)
- **random**: 13/15 benchmarks slower (avg 0.93x)
- **trig_f64**: 13/18 benchmarks slower (avg 0.90x)
- **comparison_f64**: 12/12 benchmarks slower (avg 0.77x)
- **fft**: 12/18 benchmarks slower (avg 0.97x)
- **indexing**: 12/15 benchmarks slower (avg 0.84x)
- **comparison**: 11/12 benchmarks slower (avg 0.75x)
- **set_ops**: 11/15 benchmarks slower (avg 0.98x)
- **sorting**: 10/15 benchmarks slower (avg 2.30x)
- **creation_f64**: 9/12 benchmarks slower (avg 10.57x)
- **sorting_extra**: 9/9 benchmarks slower (avg 0.91x)
- **poly_extra**: 7/7 benchmarks slower (avg 0.68x)
- **casting_f64**: 6/6 benchmarks slower (avg 0.73x)
- **poly**: 6/7 benchmarks slower (avg 0.97x)
- **io**: 4/4 benchmarks slower (avg 0.98x)
- **set_extra**: 2/3 benchmarks slower (avg 0.99x)

### General Observations

- **small** (1,000 elements): GPU wins 3/354, avg speedup 0.63x
- **medium** (100,000 elements): GPU wins 41/345, avg speedup 1.05x
- **large** (1,000,000 elements): GPU wins 80/316, avg speedup 2.42x

### Recommendations

1. **Small arrays (<1K)**: GPU overhead often dominates; keep data on CPU for tiny operations.
2. **Medium arrays (~100K)**: GPU starts to show benefits for compute-heavy ops (math, FFT).
3. **Large arrays (1M+)**: GPU excels; batch operations where possible to amortize transfer cost.
4. **Transfer minimization**: Chain GPU operations to avoid repeated CPU<->GPU transfers.

