# macmetalpy vs NumPy Benchmark Report

**Platform**: arm  
**Python**: 3.11.14  
**Date**: 2026-02-21 21:18  
**Warmup**: 2 runs  

## Category Summary

| Category | xlarge speedup | Avg Speedup |
|---|---|---|
| bitwise | 0.62x | **0.62x** |
| comparison | 1.17x | **1.17x** |
| complex_ops | 0.67x | **0.67x** |
| creation | 146.07x | **146.07x** |
| creation_extra | 0.77x | **0.77x** |
| fft_extra | 0.91x | **0.91x** |
| indexing_extra | 5.71x | **5.71x** |
| linalg_extra | 0.79x | **0.79x** |
| logic | 0.74x | **0.74x** |
| manipulation_extra | 52.44x | **52.44x** |
| math | 5.79x | **5.79x** |
| math_ext | 0.98x | **0.98x** |
| misc | 2.04x | **2.04x** |
| nan_extra | 3.23x | **3.23x** |
| reduction | 1.29x | **1.29x** |
| reduction_extra | 1.86x | **1.86x** |
| set_extra | 1.02x | **1.02x** |
| sorting_extra | 0.98x | **0.98x** |
| stats | 1.25x | **1.25x** |
| trig | 7.95x | **7.95x** |
| ufuncs_extra | 5.29x | **5.29x** |
| window | 0.95x | **0.95x** |

## Top 10: macmetalpy Fastest vs NumPy

| Rank | API | Category | Size | Speedup |
|------|-----|----------|------|---------|
| 1 | array | creation | xlarge | **1451.44x** |
| 2 | copyto | manipulation_extra | xlarge | **1398.82x** |
| 3 | put_along_axis | indexing_extra | xlarge | **83.75x** |
| 4 | floor_divide | math | xlarge | **26.67x** |
| 5 | logaddexp | ufuncs_extra | xlarge | **24.27x** |
| 6 | logaddexp2 | ufuncs_extra | xlarge | **22.59x** |
| 7 | heaviside | ufuncs_extra | xlarge | **20.21x** |
| 8 | nextafter | ufuncs_extra | xlarge | **18.94x** |
| 9 | arccos | trig | xlarge | **18.42x** |
| 10 | fmod | ufuncs_extra | xlarge | **17.30x** |

## Top 10: macmetalpy Slowest vs NumPy

| Rank | API | Category | Size | Speedup |
|------|-----|----------|------|---------|
| 1 | frombuffer | creation_extra | xlarge | **0.00x** |
| 2 | fill_diagonal | indexing_extra | xlarge | **0.01x** |
| 3 | asarray | creation_extra | xlarge | **0.09x** |
| 4 | asfortranarray | manipulation_extra | xlarge | **0.10x** |
| 5 | asanyarray | creation_extra | xlarge | **0.12x** |
| 6 | imag | complex_ops | xlarge | **0.18x** |
| 7 | flipud | manipulation_extra | xlarge | **0.21x** |
| 8 | real | complex_ops | xlarge | **0.22x** |
| 9 | fliplr | manipulation_extra | xlarge | **0.26x** |
| 10 | swapaxes | manipulation_extra | xlarge | **0.26x** |

## Category Aggregates

| Category | Benchmarks | GPU Wins | Avg Speedup | Median Speedup | Min | Max |
|----------|------------|----------|-------------|----------------|-----|-----|
| bitwise | 9 | 1 | 0.62x | 0.56x | 0.37x | 1.06x |
| comparison | 4 | 3 | 1.17x | 1.19x | 0.90x | 1.41x |
| complex_ops | 5 | 0 | 0.67x | 0.99x | 0.18x | 1.00x |
| creation | 10 | 5 | 146.07x | 1.00x | 0.31x | 1451.44x |
| creation_extra | 16 | 4 | 0.77x | 0.94x | 0.00x | 1.04x |
| fft_extra | 6 | 2 | 0.91x | 0.95x | 0.72x | 1.01x |
| indexing_extra | 17 | 4 | 5.71x | 0.98x | 0.01x | 83.75x |
| linalg_extra | 3 | 2 | 0.79x | 1.02x | 0.28x | 1.07x |
| logic | 16 | 7 | 0.74x | 0.77x | 0.32x | 1.24x |
| manipulation_extra | 27 | 5 | 52.44x | 0.63x | 0.10x | 1398.82x |
| math | 14 | 8 | 5.79x | 3.98x | 0.41x | 26.67x |
| math_ext | 10 | 3 | 0.98x | 0.99x | 0.38x | 1.85x |
| misc | 9 | 4 | 2.04x | 1.00x | 0.29x | 7.24x |
| nan_extra | 9 | 8 | 3.23x | 3.83x | 0.99x | 5.57x |
| reduction | 13 | 5 | 1.29x | 0.66x | 0.40x | 4.22x |
| reduction_extra | 9 | 3 | 1.86x | 0.99x | 0.38x | 10.40x |
| set_extra | 1 | 1 | 1.02x | 1.02x | 1.02x | 1.02x |
| sorting_extra | 3 | 1 | 0.98x | 0.97x | 0.97x | 1.00x |
| stats | 6 | 3 | 1.25x | 1.02x | 0.77x | 2.05x |
| trig | 15 | 15 | 7.95x | 5.68x | 2.79x | 18.42x |
| ufuncs_extra | 34 | 16 | 5.29x | 0.98x | 0.35x | 24.27x |
| window | 5 | 0 | 0.95x | 0.95x | 0.91x | 0.98x |

## Failed Benchmarks

| API | Category | Size | Error |
|-----|----------|------|-------|
| mask_indices | indexing_extra | xlarge | TypeError: Unsupported dtype for Metal: object |
| select | indexing_extra | xlarge | RecursionError: maximum recursion depth exceeded |

## Optimization Guidance

### Where macmetalpy is slower than NumPy

- **manipulation_extra**: 22/27 benchmarks slower (avg 52.44x)
- **ufuncs_extra**: 18/34 benchmarks slower (avg 5.29x)
- **indexing_extra**: 13/17 benchmarks slower (avg 5.71x)
- **creation_extra**: 12/16 benchmarks slower (avg 0.77x)
- **logic**: 9/16 benchmarks slower (avg 0.74x)
- **bitwise**: 8/9 benchmarks slower (avg 0.62x)
- **reduction**: 8/13 benchmarks slower (avg 1.29x)
- **math_ext**: 7/10 benchmarks slower (avg 0.98x)
- **math**: 6/14 benchmarks slower (avg 5.79x)
- **reduction_extra**: 6/9 benchmarks slower (avg 1.86x)
- **complex_ops**: 5/5 benchmarks slower (avg 0.67x)
- **creation**: 5/10 benchmarks slower (avg 146.07x)
- **misc**: 5/9 benchmarks slower (avg 2.04x)
- **window**: 5/5 benchmarks slower (avg 0.95x)
- **fft_extra**: 4/6 benchmarks slower (avg 0.91x)
- **stats**: 3/6 benchmarks slower (avg 1.25x)
- **sorting_extra**: 2/3 benchmarks slower (avg 0.98x)
- **comparison**: 1/4 benchmarks slower (avg 1.17x)
- **linalg_extra**: 1/3 benchmarks slower (avg 0.79x)
- **nan_extra**: 1/9 benchmarks slower (avg 3.23x)

### General Observations


### Recommendations

1. **Small arrays (<1K)**: GPU overhead often dominates; keep data on CPU for tiny operations.
2. **Medium arrays (~100K)**: GPU starts to show benefits for compute-heavy ops (math, FFT).
3. **Large arrays (1M+)**: GPU excels; batch operations where possible to amortize transfer cost.
4. **Transfer minimization**: Chain GPU operations to avoid repeated CPU<->GPU transfers.

