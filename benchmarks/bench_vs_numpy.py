#!/usr/bin/env python3
"""macmetalpy vs NumPy Performance Benchmark Runner.

Orchestrates running all benchmarks across size tiers, collects statistics,
and produces console output, JSON results, and a markdown report.

Usage:
    python benchmarks/bench_vs_numpy.py
    python benchmarks/bench_vs_numpy.py --sizes small,medium,large
    python benchmarks/bench_vs_numpy.py --repeat 20
    python benchmarks/bench_vs_numpy.py --filter math
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from statistics import mean, median

# ---------------------------------------------------------------------------
# Size tiers
# ---------------------------------------------------------------------------

SIZE_TIERS = {
    "small": 1_000,
    "medium": 100_000,
    "large": 1_000_000,
    "xlarge": 10_000_000,
}

DEFAULT_REPEATS = {
    "small": 50,
    "medium": 30,
    "large": 10,
    "xlarge": 5,
}

WARMUP_RUNS = 2

# ---------------------------------------------------------------------------
# Single-benchmark execution (runs in a worker process)
# ---------------------------------------------------------------------------


def _run_single_benchmark(bench_entry: dict, size_name: str, size: int, repeats: int,
                          cached_numpy: dict | None = None):
    """Run one benchmark at one size tier.

    If *cached_numpy* is provided and contains a matching entry, numpy timings
    are taken from the cache instead of being re-measured.

    Returns a result dict with timing statistics, or an error dict on failure.
    """
    name = bench_entry["name"]
    category = bench_entry["category"]
    func = bench_entry["func"]

    # Look up cached numpy result (keyed by "category/name@size_name")
    np_cache_hit = None
    if cached_numpy is not None:
        cache_key = f"{category}/{name}@{size_name}"
        np_cache_hit = cached_numpy.get(cache_key)

    try:
        # Warm-up
        for _ in range(WARMUP_RUNS):
            func(size)

        mp_times = []
        np_times = []
        for _ in range(repeats):
            mp_t, np_t = func(size)
            mp_times.append(mp_t)
            np_times.append(np_t)

        mp_median = median(mp_times)
        mp_mean = mean(mp_times)
        mp_min = min(mp_times)

        if np_cache_hit is not None:
            np_median = np_cache_hit["np_median"]
            np_mean = np_cache_hit["np_mean"]
            np_min = np_cache_hit["np_min"]
        else:
            np_median = median(np_times)
            np_mean = mean(np_times)
            np_min = min(np_times)

        speedup = np_median / mp_median if mp_median > 0 else float("inf")

        return {
            "name": name,
            "category": category,
            "size_name": size_name,
            "size": size,
            "mp_median": mp_median,
            "mp_mean": mp_mean,
            "mp_min": mp_min,
            "np_median": np_median,
            "np_mean": np_mean,
            "np_min": np_min,
            "speedup": speedup,
            "error": None,
        }
    except Exception as exc:
        return {
            "name": name,
            "category": category,
            "size_name": size_name,
            "size": size,
            "mp_median": None,
            "mp_mean": None,
            "mp_min": None,
            "np_median": None,
            "np_mean": None,
            "np_min": None,
            "speedup": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def _print_header():
    chip = platform.processor() or platform.machine()
    print()
    print("=" * 100)
    print("  macmetalpy vs NumPy Performance Benchmarks")
    print("=" * 100)
    print(f"  Platform : {chip}")
    print(f"  Python   : {sys.version.split()[0]}")
    print(f"  dtype    : float32")
    print(f"  Warmup   : {WARMUP_RUNS} runs")
    print("=" * 100)
    print()


def _print_table(results: list[dict]):
    """Print formatted results table to console."""
    hdr = (
        f"{'Category':<18} {'API':<22} {'Size':<8} "
        f"{'macmetalpy(ms)':>14} {'numpy(ms)':>14} {'Speedup':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        if r["error"]:
            print(
                f"{r['category']:<18} {r['name']:<22} {r['size_name']:<8} "
                f"{'ERROR':>14} {'':>14} {'':>10}  ({r['error']})"
            )
        else:
            marker = " *" if r["speedup"] >= 1.0 else ""
            print(
                f"{r['category']:<18} {r['name']:<22} {r['size_name']:<8} "
                f"{r['mp_median']*1000:>14.3f} {r['np_median']*1000:>14.3f} "
                f"{r['speedup']:>9.2f}x{marker}"
            )


def _print_summary(results: list[dict]):
    """Print summary statistics."""
    valid = [r for r in results if r["error"] is None]
    failed = [r for r in results if r["error"] is not None]

    print()
    print("=" * 100)
    print("  SUMMARY")
    print("=" * 100)

    if not valid:
        print("  No successful benchmarks.")
        return

    gpu_wins = sum(1 for r in valid if r["speedup"] >= 1.0)
    total = len(valid)
    print(f"  Total benchmarks: {total}  |  GPU wins: {gpu_wins}  |  Failed: {len(failed)}")
    print()

    # By size tier
    print("  By Size Tier:")
    for tier in ("small", "medium", "large"):
        tier_results = [r for r in valid if r["size_name"] == tier]
        if not tier_results:
            continue
        speeds = [r["speedup"] for r in tier_results]
        tier_wins = sum(1 for s in speeds if s >= 1.0)
        avg_speed = mean(speeds)
        med_speed = median(speeds)
        print(
            f"    {tier:>8}: avg {avg_speed:6.2f}x  median {med_speed:6.2f}x  "
            f"GPU wins {tier_wins}/{len(tier_results)}"
        )

    # By category
    print()
    print("  By Category:")
    categories = sorted(set(r["category"] for r in valid))
    for cat in categories:
        cat_results = [r for r in valid if r["category"] == cat]
        speeds = [r["speedup"] for r in cat_results]
        avg_speed = mean(speeds)
        cat_wins = sum(1 for s in speeds if s >= 1.0)
        print(
            f"    {cat:<22}: avg {avg_speed:6.2f}x  GPU wins {cat_wins}/{len(cat_results)}"
        )

    print()


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def _save_json(results: list[dict], path: str):
    """Save results to a JSON file."""
    # Convert any inf/nan to string for JSON compatibility
    clean = []
    for r in results:
        entry = dict(r)
        for key in ("speedup", "mp_median", "mp_mean", "mp_min", "np_median", "np_mean", "np_min"):
            v = entry.get(key)
            if v is not None and (v == float("inf") or v != v):  # inf or nan
                entry[key] = str(v)
        clean.append(entry)

    with open(path, "w") as f:
        json.dump(
            {
                "platform": platform.processor() or platform.machine(),
                "python": sys.version.split()[0],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "warmup_runs": WARMUP_RUNS,
                "results": clean,
            },
            f,
            indent=2,
        )
    print(f"  Results saved to {path}")


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------


def _generate_report(results: list[dict], path: str):
    """Generate a markdown benchmark report."""
    valid = [r for r in results if r["error"] is None]
    failed = [r for r in results if r["error"] is not None]

    lines = []
    a = lines.append

    chip = platform.processor() or platform.machine()
    a("# macmetalpy vs NumPy Benchmark Report")
    a("")
    a(f"**Platform**: {chip}  ")
    a(f"**Python**: {sys.version.split()[0]}  ")
    a(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}  ")
    a(f"**Warmup**: {WARMUP_RUNS} runs  ")
    a("")

    # ---- Category summary table ----
    a("## Category Summary")
    a("")
    categories = sorted(set(r["category"] for r in valid))
    size_names = sorted(set(r["size_name"] for r in valid), key=lambda s: SIZE_TIERS.get(s, 0))

    a("| Category | " + " | ".join(f"{s} speedup" for s in size_names) + " | Avg Speedup |")
    a("|" + "---|" * (len(size_names) + 2))

    for cat in categories:
        cat_results = [r for r in valid if r["category"] == cat]
        cols = []
        all_speeds = []
        for sn in size_names:
            tier_res = [r for r in cat_results if r["size_name"] == sn]
            if tier_res:
                avg = mean(r["speedup"] for r in tier_res)
                cols.append(f"{avg:.2f}x")
                all_speeds.extend(r["speedup"] for r in tier_res)
            else:
                cols.append("-")
        overall = mean(all_speeds) if all_speeds else 0
        a(f"| {cat} | " + " | ".join(cols) + f" | **{overall:.2f}x** |")

    # ---- Top 10 fastest ----
    a("")
    a("## Top 10: macmetalpy Fastest vs NumPy")
    a("")
    a("| Rank | API | Category | Size | Speedup |")
    a("|------|-----|----------|------|---------|")
    by_speedup = sorted(valid, key=lambda r: r["speedup"], reverse=True)
    for i, r in enumerate(by_speedup[:10], 1):
        a(f"| {i} | {r['name']} | {r['category']} | {r['size_name']} | **{r['speedup']:.2f}x** |")

    # ---- Top 10 slowest ----
    a("")
    a("## Top 10: macmetalpy Slowest vs NumPy")
    a("")
    a("| Rank | API | Category | Size | Speedup |")
    a("|------|-----|----------|------|---------|")
    by_slowest = sorted(valid, key=lambda r: r["speedup"])
    for i, r in enumerate(by_slowest[:10], 1):
        a(f"| {i} | {r['name']} | {r['category']} | {r['size_name']} | **{r['speedup']:.2f}x** |")

    # ---- Category-level aggregates ----
    a("")
    a("## Category Aggregates")
    a("")
    a("| Category | Benchmarks | GPU Wins | Avg Speedup | Median Speedup | Min | Max |")
    a("|----------|------------|----------|-------------|----------------|-----|-----|")

    for cat in categories:
        cat_results = [r for r in valid if r["category"] == cat]
        speeds = [r["speedup"] for r in cat_results]
        wins = sum(1 for s in speeds if s >= 1.0)
        a(
            f"| {cat} | {len(speeds)} | {wins} | "
            f"{mean(speeds):.2f}x | {median(speeds):.2f}x | "
            f"{min(speeds):.2f}x | {max(speeds):.2f}x |"
        )

    # ---- Failed benchmarks ----
    if failed:
        a("")
        a("## Failed Benchmarks")
        a("")
        a("| API | Category | Size | Error |")
        a("|-----|----------|------|-------|")
        for r in failed:
            a(f"| {r['name']} | {r['category']} | {r['size_name']} | {r['error']} |")

    # ---- Optimization guidance ----
    a("")
    a("## Optimization Guidance")
    a("")

    # Find categories where GPU loses most often
    a("### Where macmetalpy is slower than NumPy")
    a("")
    slow_cats = []
    for cat in categories:
        cat_results = [r for r in valid if r["category"] == cat]
        speeds = [r["speedup"] for r in cat_results]
        losses = sum(1 for s in speeds if s < 1.0)
        if losses > 0:
            slow_cats.append((cat, losses, len(speeds), mean(speeds)))
    slow_cats.sort(key=lambda x: x[1], reverse=True)
    for cat, losses, total, avg in slow_cats:
        a(f"- **{cat}**: {losses}/{total} benchmarks slower (avg {avg:.2f}x)")

    a("")
    a("### General Observations")
    a("")

    # Size-based observations
    for tier in ("small", "medium", "large"):
        tier_results = [r for r in valid if r["size_name"] == tier]
        if tier_results:
            speeds = [r["speedup"] for r in tier_results]
            wins = sum(1 for s in speeds if s >= 1.0)
            a(
                f"- **{tier}** ({SIZE_TIERS.get(tier, '?'):,} elements): "
                f"GPU wins {wins}/{len(speeds)}, avg speedup {mean(speeds):.2f}x"
            )

    a("")
    a("### Recommendations")
    a("")
    a("1. **Small arrays (<1K)**: GPU overhead often dominates; keep data on CPU for tiny operations.")
    a("2. **Medium arrays (~100K)**: GPU starts to show benefits for compute-heavy ops (math, FFT).")
    a("3. **Large arrays (1M+)**: GPU excels; batch operations where possible to amortize transfer cost.")
    a("4. **Transfer minimization**: Chain GPU operations to avoid repeated CPU<->GPU transfers.")
    a("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report saved to {path}")


# ---------------------------------------------------------------------------
# NumPy result caching
# ---------------------------------------------------------------------------

_NUMPY_CACHE_FILE = "numpy_cache.json"


def _save_numpy_cache_raw(cache: dict, path: str):
    """Write a numpy cache dict directly to file."""
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"  NumPy cache saved to {path} ({len(cache)} entries)")


def _load_numpy_cache(path: str, quiet: bool = False) -> dict | None:
    """Load cached numpy timings from file. Returns None if file doesn't exist."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        cache = json.load(f)
    if not quiet:
        print(f"  Loaded NumPy cache from {path} ({len(cache)} entries)")
    return cache


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def _collect_benchmarks(filter_category: str | None = None):
    """Import and collect benchmarks from bench_core and bench_advanced modules."""
    benchmarks = []

    # Import bench_core
    try:
        from bench_core import BENCHMARKS as core_benchmarks
        benchmarks.extend(core_benchmarks)
    except ImportError:
        try:
            # Try relative import path
            bench_dir = os.path.dirname(os.path.abspath(__file__))
            if bench_dir not in sys.path:
                sys.path.insert(0, bench_dir)
            from bench_core import BENCHMARKS as core_benchmarks
            benchmarks.extend(core_benchmarks)
        except ImportError as e:
            print(f"  WARNING: Could not import bench_core: {e}")

    # Import bench_advanced
    try:
        from bench_advanced import BENCHMARKS as advanced_benchmarks
        benchmarks.extend(advanced_benchmarks)
    except ImportError:
        try:
            bench_dir = os.path.dirname(os.path.abspath(__file__))
            if bench_dir not in sys.path:
                sys.path.insert(0, bench_dir)
            from bench_advanced import BENCHMARKS as advanced_benchmarks
            benchmarks.extend(advanced_benchmarks)
        except ImportError as e:
            print(f"  WARNING: Could not import bench_advanced: {e}")

    # Import bench_expanded
    try:
        from bench_expanded import BENCHMARKS as expanded_benchmarks
        benchmarks.extend(expanded_benchmarks)
    except ImportError:
        try:
            bench_dir = os.path.dirname(os.path.abspath(__file__))
            if bench_dir not in sys.path:
                sys.path.insert(0, bench_dir)
            from bench_expanded import BENCHMARKS as expanded_benchmarks
            benchmarks.extend(expanded_benchmarks)
        except ImportError as e:
            print(f"  WARNING: Could not import bench_expanded: {e}")

    # Import bench_float64
    try:
        from bench_float64 import BENCHMARKS as float64_benchmarks
        benchmarks.extend(float64_benchmarks)
    except ImportError:
        try:
            bench_dir = os.path.dirname(os.path.abspath(__file__))
            if bench_dir not in sys.path:
                sys.path.insert(0, bench_dir)
            from bench_float64 import BENCHMARKS as float64_benchmarks
            benchmarks.extend(float64_benchmarks)
        except ImportError as e:
            print(f"  WARNING: Could not import bench_float64: {e}")

    if filter_category:
        filter_cats = {c.strip().lower() for c in filter_category.split(",")}
        benchmarks = [b for b in benchmarks if b["category"].lower() in filter_cats]

    return benchmarks


def main():
    parser = argparse.ArgumentParser(
        description="macmetalpy vs NumPy benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python benchmarks/bench_vs_numpy.py\n"
            "  python benchmarks/bench_vs_numpy.py --sizes small,medium\n"
            "  python benchmarks/bench_vs_numpy.py --repeat 20 --filter math\n"
            "  python benchmarks/bench_vs_numpy.py --sizes large --filter linalg,fft\n"
        ),
    )
    parser.add_argument(
        "--sizes",
        default="small,medium,large",
        help="Comma-separated size tiers: small, medium, large (default: small,medium,large)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="Override repeat count for all sizes (0 = use per-tier defaults)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Comma-separated category names to run (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel worker processes (default: 3)",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Run benchmarks serially instead of in parallel",
    )
    parser.add_argument(
        "--numpy-cache",
        action="store_true",
        help="Use cached NumPy results instead of re-timing (from numpy_cache.json)",
    )
    args = parser.parse_args()

    bench_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve sizes
    selected_tiers = [t.strip() for t in args.sizes.split(",") if t.strip()]
    tier_configs = []
    for tier in selected_tiers:
        if tier not in SIZE_TIERS:
            print(f"  WARNING: Unknown size tier '{tier}', skipping.")
            continue
        repeats = args.repeat if args.repeat > 0 else DEFAULT_REPEATS.get(tier, 30)
        tier_configs.append((tier, SIZE_TIERS[tier], repeats))

    if not tier_configs:
        print("ERROR: No valid size tiers selected.")
        sys.exit(1)

    _print_header()

    for tier, size, repeats in tier_configs:
        print(f"  {tier:>8}: {size:>12,} elements, {repeats} repeats")
    print()

    # Collect benchmarks
    benchmarks = _collect_benchmarks(filter_category=args.filter)
    if not benchmarks:
        print("ERROR: No benchmarks found. Ensure bench_core.py and bench_advanced.py exist.")
        sys.exit(1)

    print(f"  Collected {len(benchmarks)} benchmark(s)")
    categories = sorted(set(b["category"] for b in benchmarks))
    print(f"  Categories: {', '.join(categories)}")
    print()

    # Build work items: (bench_entry, size_name, size, repeats)
    work_items = []
    for bench in benchmarks:
        bench_sizes = bench.get("sizes", ["small", "medium", "large", "xlarge"])
        for tier, size, repeats in tier_configs:
            if tier in bench_sizes:
                work_items.append((bench, tier, size, repeats))

    total_work = len(work_items)
    print(f"  Running {total_work} benchmark jobs...")

    # Load numpy cache if requested
    cached_numpy = None
    if args.numpy_cache:
        cache_path = os.path.join(bench_dir, _NUMPY_CACHE_FILE)
        cached_numpy = _load_numpy_cache(cache_path)
        if cached_numpy is None:
            print(f"  WARNING: No numpy cache found at {cache_path}, will time numpy normally.")
        else:
            print(f"  Using cached NumPy timings (re-timing macmetalpy only)")
    print()

    # Execute benchmarks
    all_results = []
    completed = 0

    if args.serial or args.workers <= 1:
        # Serial execution
        for bench, tier, size, repeats in work_items:
            completed += 1
            sys.stdout.write(
                f"\r  [{completed}/{total_work}] {bench['category']}/{bench['name']} @ {tier}    "
            )
            sys.stdout.flush()
            result = _run_single_benchmark(bench, tier, size, repeats,
                                           cached_numpy=cached_numpy)
            all_results.append(result)
            gc.collect()  # Free Metal buffers between benchmarks
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_info = {}
            for bench, tier, size, repeats in work_items:
                future = executor.submit(_run_single_benchmark, bench, tier, size, repeats,
                                         cached_numpy)
                future_to_info[future] = (bench["category"], bench["name"], tier)

            for future in as_completed(future_to_info):
                completed += 1
                cat, name, tier = future_to_info[future]
                sys.stdout.write(f"\r  [{completed}/{total_work}] {cat}/{name} @ {tier}    ")
                sys.stdout.flush()
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as exc:
                    all_results.append({
                        "name": name,
                        "category": cat,
                        "size_name": tier,
                        "size": 0,
                        "mp_median": None,
                        "mp_mean": None,
                        "mp_min": None,
                        "np_median": None,
                        "np_mean": None,
                        "np_min": None,
                        "speedup": None,
                        "error": f"Process error: {exc}",
                    })

    print("\r" + " " * 80)
    print()

    # Sort results by category, name, size for consistent display
    size_order = {"small": 0, "medium": 1, "large": 2}
    all_results.sort(key=lambda r: (r["category"], r["name"], size_order.get(r["size_name"], 9)))

    # Print results table
    _print_table(all_results)

    # Print summary
    _print_summary(all_results)

    # Save JSON
    json_path = os.path.join(bench_dir, "results.json")
    _save_json(all_results, json_path)

    # Always save/update numpy cache (merge with existing)
    np_cache_path = os.path.join(bench_dir, _NUMPY_CACHE_FILE)
    if not args.numpy_cache:
        # Only update cache when we actually timed numpy (not using cached values)
        existing_cache = _load_numpy_cache(np_cache_path, quiet=True) or {}
        for r in all_results:
            if r["error"] is not None or r["np_median"] is None:
                continue
            key = f"{r['category']}/{r['name']}@{r['size_name']}"
            existing_cache[key] = {
                "np_median": r["np_median"],
                "np_mean": r["np_mean"],
                "np_min": r["np_min"],
            }
        _save_numpy_cache_raw(existing_cache, np_cache_path)

    # Generate markdown report
    report_path = os.path.join(bench_dir, "BENCHMARK_REPORT.md")
    _generate_report(all_results, report_path)

    print()


if __name__ == "__main__":
    main()
