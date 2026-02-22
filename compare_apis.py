#!/usr/bin/env python3
"""Compare macmetalpy and numpy catalogs, generating API_COMPARISON_REPORT.md."""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- severity classification ----------

HIGH_PARAMS = {
    "out", "dtype", "axis", "keepdims", "ddof",
    "return_index", "return_inverse", "return_counts",
    "endpoint", "base", "copy", "compute_uv", "hermitian",
    "UPLO",
}

MEDIUM_PARAMS = {
    "order", "casting", "where", "initial", "kind",
    "mode", "sorter", "equal_nan", "bias", "rowvar",
    "fweights", "aweights", "norm",
}

LOW_PARAMS = {
    "subok", "like", "signature", "extobj", "args", "kwargs",
}

# Items that should not be counted as real gaps (dunder operator naming differences)
NDARRAY_OPERATOR_PARAM_IGNORE = {"value", "key", "mod", "self"}

# Not applicable / legacy / internal items
NOT_APPLICABLE_PREFIXES = (
    "CLIP", "DataSource", "ERR_", "FPE_", "FLOATING_POINT",
    "MAXDIMS", "MAY_SHARE", "RAISE", "RankWarning", "SHIFT_",
    "UFUNC_", "WRAP",
)

NOT_APPLICABLE_NAMES = {
    "add_docstring", "add_newdoc", "add_newdoc_ufunc",
    "busday_count", "busday_offset", "busdaycalendar",
    "byte_bounds", "bytes_", "char", "character", "chararray",
    "compat", "ctypeslib", "datetime64", "datetime_as_string",
    "datetime_data", "deprecate", "deprecate_with_doc", "disp",
    "dtypes", "emath", "errstate", "exceptions", "flexible",
    "format_parser", "fromfile", "fromregex", "genfromtxt",
    "get_array_wrap", "get_include", "get_printoptions",
    "getbufsize", "geterr", "geterrcall", "index_exp",
    "info", "is_busday", "iterable", "lib", "little_endian",
    "loadtxt", "lookfor", "ma", "may_share_memory", "memmap",
    "ndenumerate", "ndindex", "nested_iters", "numarray",
    "oldnumeric", "poly", "poly1d", "polyadd", "polyder",
    "polydiv", "polyfit", "polyint", "polymul", "polynomial",
    "polysub", "polyval", "printoptions", "rec", "recarray",
    "recfromcsv", "recfromtxt", "record", "require", "roots",
    "safe_eval", "savetxt", "set_printoptions", "set_string_function",
    "setbufsize", "seterr", "seterrcall", "shares_memory",
    "show_config", "show_runtime", "source", "str_", "test",
    "testing", "timedelta64", "tracemalloc_domain", "typecodes",
    "version", "void", "who",
}

LEGACY_NAMES = {
    "ALLOW_THREADS", "BUFSIZE", "False_", "Inf", "Infinity",
    "NAN", "NINF", "NZERO", "NaN", "PINF", "PZERO",
    "ScalarType", "True_", "alltrue", "asmatrix", "bmat",
    "cast", "compare_chararrays", "cumproduct", "find_common_type",
    "infty", "issubclass_", "issubsctype", "mat", "matrix",
    "maximum_sctype", "nbytes", "object_", "product", "row_stack",
    "sctypeDict", "sctypes", "sometrue", "string_", "trapz",
    "unicode_",
}


def classify_severity(param_name):
    if param_name in HIGH_PARAMS:
        return "HIGH"
    if param_name in MEDIUM_PARAMS:
        return "MEDIUM"
    return "LOW"


def is_not_applicable(name):
    if name in NOT_APPLICABLE_NAMES:
        return True
    for pfx in NOT_APPLICABLE_PREFIXES:
        if name.startswith(pfx):
            return True
    return False


def load_catalog(filename):
    path = os.path.join(SCRIPT_DIR, filename)
    with open(path) as f:
        return json.load(f)


def compare_params(np_params, mmp_params):
    """Return list of param names in np but not mmp."""
    if np_params is None or mmp_params is None:
        return []
    np_set = set(np_params)
    mmp_set = set(mmp_params)
    return sorted(np_set - mmp_set - {"self"})


def generate_report():
    np_cat = load_catalog("numpy_catalog.json")
    mmp_cat = load_catalog("macmetalpy_catalog.json")

    np_version = np_cat.get("numpy_version", "unknown")
    np_top = np_cat["top_level"]
    mmp_top = mmp_cat["top_level"]

    lines = []
    lines.append("# NumPy vs MacMetalPy API Comparison Report\n")
    lines.append(f"NumPy version analyzed: {np_version}\n")
    lines.append("---\n")

    # ---------- Section 1: Top-Level Coverage ----------
    np_names = set(np_top.keys())
    mmp_names = set(mmp_top.keys())
    common = sorted(np_names & mmp_names)
    missing = sorted(np_names - mmp_names)
    extra = sorted(mmp_names - np_names)

    lines.append("## 1. Top-Level Function/Class/Constant Coverage\n")
    lines.append(f"- **NumPy top-level items**: {len(np_names)}")
    lines.append(f"- **MacMetalPy top-level items**: {len(mmp_names)}")
    lines.append(f"- **In common**: {len(common)}")
    lines.append(f"- **Missing from MacMetalPy**: {len(missing)}")
    lines.append(f"- **Extra in MacMetalPy (not in NumPy)**: {len(extra)}\n")

    # Classify missing items
    relevant_missing = [n for n in missing if not is_not_applicable(n) and n not in LEGACY_NAMES]
    na_missing = [n for n in missing if is_not_applicable(n)]
    legacy_missing = [n for n in missing if n in LEGACY_NAMES]

    lines.append(f"### 1.1 Missing: RELEVANT (a GPU library should have these)\n")
    lines.append(f"**{len(relevant_missing)} items:**\n")
    lines.append("| Name | Type | Notes |")
    lines.append("|------|------|-------|")
    for name in relevant_missing:
        info = np_top[name]
        note = ""
        if info["type"] == "class":
            note = "class/dtype"
        elif info["type"] == "ufunc":
            note = "ufunc"
        elif info.get("signature"):
            note = f"`{info['signature']}`"
        lines.append(f"| `{name}` | {info['type']} | {note} |")
    lines.append("")

    lines.append("### 1.2 Missing: NOT APPLICABLE (strings, datetime, I/O, matrix, etc.)\n")
    lines.append(f"**{len(na_missing)} items** (collapsed for brevity):\n")
    lines.append("<details>")
    lines.append("<summary>Click to expand</summary>\n")
    lines.append("| Name | Type |")
    lines.append("|------|------|")
    for name in na_missing:
        lines.append(f"| `{name}` | {np_top[name]['type']} |")
    lines.append("\n</details>\n")

    lines.append("### 1.3 Missing: LEGACY / DEPRECATED\n")
    lines.append(f"**{len(legacy_missing)} items** (collapsed for brevity):\n")
    lines.append("<details>")
    lines.append("<summary>Click to expand</summary>\n")
    lines.append("| Name | Type |")
    lines.append("|------|------|")
    for name in legacy_missing:
        lines.append(f"| `{name}` | {np_top[name]['type']} |")
    lines.append("\n</details>\n")

    lines.append("### 1.4 Extra in MacMetalPy (not in NumPy)\n")
    lines.append(f"**{len(extra)} items:**\n")
    lines.append("| Name | Type |")
    lines.append("|------|------|")
    for name in extra:
        lines.append(f"| `{name}` | {mmp_top[name]['type']} |")
    lines.append("\n---\n")

    # ---------- Section 2: Submodule Coverage ----------
    lines.append("## 2. Submodule Coverage\n")

    for idx, submod_name in enumerate(["linalg", "fft", "random"], 1):
        np_sub = np_cat["submodules"].get(submod_name, {})
        mmp_sub = mmp_cat["submodules"].get(submod_name, {})
        np_snames = set(np_sub.keys())
        mmp_snames = set(mmp_sub.keys())
        s_common = sorted(np_snames & mmp_snames)
        s_missing = sorted(np_snames - mmp_snames)
        s_extra = sorted(mmp_snames - np_snames)

        lines.append(f"### 2.{idx} numpy.{submod_name} vs macmetalpy.{submod_name}\n")
        lines.append(f"- **NumPy items**: {len(np_snames)}")
        lines.append(f"- **MacMetalPy items**: {len(mmp_snames)}")
        lines.append(f"- **In common**: {len(s_common)}")
        lines.append(f"- **Missing from MacMetalPy**: {len(s_missing)}\n")

        if s_missing:
            lines.append("| Function | Type |")
            lines.append("|----------|------|")
            for name in s_missing:
                lines.append(f"| `{name}` | {np_sub[name]['type']} |")
            lines.append("")

        if s_extra:
            lines.append("#### Extra in MacMetalPy (not in NumPy):\n")
            lines.append("| Function | Type |")
            lines.append("|----------|------|")
            for name in s_extra:
                lines.append(f"| `{name}` | {mmp_sub[name]['type']} |")
            lines.append("")

        # Parameter gaps in submodule
        sub_gaps = []
        for name in s_common:
            np_params = np_sub[name].get("params")
            mmp_params = mmp_sub[name].get("params")
            missing_params = compare_params(np_params, mmp_params)
            if missing_params:
                sub_gaps.append((name, missing_params))

        if sub_gaps:
            lines.append(f"#### Parameter gaps in common {submod_name} functions:\n")
            lines.append("| Function | Missing Parameters |")
            lines.append("|----------|-------------------|")
            for name, params in sub_gaps:
                lines.append(f"| `{name}` | {', '.join(f'`{p}`' for p in params)} |")
            lines.append("")

        lines.append("")

    lines.append("---\n")

    # ---------- Section 3: ndarray ----------
    lines.append("## 3. ndarray Method and Property Coverage\n")

    np_methods = np_cat["ndarray_methods"]
    mmp_methods = mmp_cat["ndarray_methods"]
    np_props = np_cat["ndarray_properties"]
    mmp_props = mmp_cat["ndarray_properties"]

    nm_names = set(np_methods.keys())
    mm_names = set(mmp_methods.keys())
    m_common = sorted(nm_names & mm_names)
    m_missing = sorted(nm_names - mm_names)
    m_extra = sorted(mm_names - nm_names)

    lines.append("### 3.1 ndarray Methods\n")
    lines.append(f"- **NumPy ndarray methods**: {len(nm_names)}")
    lines.append(f"- **MacMetalPy ndarray methods**: {len(mm_names)}")
    lines.append(f"- **In common**: {len(m_common)}")
    lines.append(f"- **Missing from MacMetalPy**: {len(m_missing)}\n")

    if m_missing:
        lines.append("| Method | Type | Relevance |")
        lines.append("|--------|------|-----------|")
        for name in m_missing:
            info = np_methods[name]
            relevance = "internal" if name.startswith("__") else "method"
            lines.append(f"| `{name}` | {info['type']} | {relevance} |")
        lines.append("")

    if m_extra:
        lines.append("#### Extra ndarray methods in MacMetalPy:\n")
        lines.append("| Method | Type |")
        lines.append("|--------|------|")
        for name in m_extra:
            lines.append(f"| `{name}` | {mmp_methods[name]['type']} |")
        lines.append("")

    # ndarray properties
    np_pnames = set(np_props.keys())
    mmp_pnames = set(mmp_props.keys())
    p_common = sorted(np_pnames & mmp_pnames)
    p_missing = sorted(np_pnames - mmp_pnames)

    lines.append("### 3.2 ndarray Properties\n")
    lines.append(f"- **NumPy ndarray properties**: {len(np_pnames)}")
    lines.append(f"- **MacMetalPy ndarray properties**: {len(mmp_pnames)}")
    lines.append(f"- **Missing from MacMetalPy**: {len(p_missing)}\n")

    if p_missing:
        lines.append("| Property | Notes |")
        lines.append("|----------|-------|")
        for name in p_missing:
            lines.append(f"| `{name}` | |")
        lines.append("")

    # ndarray method param gaps
    ndarray_param_gaps = []
    for name in m_common:
        np_params = np_methods[name].get("params")
        mmp_params = mmp_methods[name].get("params")
        missing_params = compare_params(np_params, mmp_params)
        # Filter out operator naming diffs
        missing_params = [p for p in missing_params if p not in NDARRAY_OPERATOR_PARAM_IGNORE]
        if missing_params:
            ndarray_param_gaps.append((name, missing_params))

    lines.append("### 3.3 Parameter Gaps in Common ndarray Methods\n")
    if ndarray_param_gaps:
        lines.append("| Method | Missing Parameters |")
        lines.append("|--------|-------------------|")
        for name, params in ndarray_param_gaps:
            lines.append(f"| `{name}` | {', '.join(f'`{p}`' for p in params)} |")
    else:
        lines.append("No parameter gaps detected (excluding operator naming differences).")
    lines.append("")
    lines.append("---\n")

    # ---------- Section 4: Parameter-Level Comparison ----------
    lines.append("## 4. Parameter-Level Comparison (Top-Level Functions)\n")
    lines.append("This is the most important section. For every function that exists in BOTH")
    lines.append("NumPy and MacMetalPy, we compare parameter lists and report missing parameters.\n")

    # Gather all gaps
    all_gaps = []  # (func_name, [(param, severity)])
    uninspectable_np = 0
    uninspectable_mmp = 0
    total_high = 0
    total_med = 0
    total_low = 0
    total_missing_params = 0

    for name in common:
        np_info = np_top[name]
        mmp_info = mmp_top[name]

        if np_info["type"] not in ("function", "ufunc"):
            continue
        if mmp_info["type"] not in ("function",):
            continue

        np_params = np_info.get("params")
        mmp_params = mmp_info.get("params")

        if np_params is None:
            uninspectable_np += 1
            continue
        if mmp_params is None:
            uninspectable_mmp += 1
            continue

        missing_params = compare_params(np_params, mmp_params)
        if missing_params:
            gap_items = []
            for p in missing_params:
                sev = classify_severity(p)
                gap_items.append((p, sev))
                if sev == "HIGH":
                    total_high += 1
                elif sev == "MEDIUM":
                    total_med += 1
                else:
                    total_low += 1
                total_missing_params += 1
            all_gaps.append((name, gap_items))

    funcs_with_gaps = len(all_gaps)

    lines.append(f"- **Functions compared**: {len(common)}")
    lines.append(f"- **Functions with parameter gaps**: {funcs_with_gaps}")
    lines.append(f"- **Total missing parameters**: {total_missing_params}")
    lines.append(f"  - HIGH severity: {total_high}")
    lines.append(f"  - MEDIUM severity: {total_med}")
    lines.append(f"  - LOW severity: {total_low}")
    lines.append(f"- **NumPy functions with uninspectable signatures**: {uninspectable_np}")
    lines.append(f"- **MacMetalPy functions with uninspectable signatures**: {uninspectable_mmp}\n")

    # 4.1 HIGH severity
    high_gaps = [(name, [(p, s) for p, s in items if s == "HIGH"]) for name, items in all_gaps]
    high_gaps = [(n, g) for n, g in high_gaps if g]

    lines.append("### 4.1 HIGH Severity: Missing Commonly-Used Parameters\n")
    lines.append("These are parameters that real-world code frequently uses: `out=`, `dtype=`")
    lines.append("on reductions, `axis=`, `keepdims=`, `ddof=`, `return_index/inverse/counts`, etc.\n")

    if high_gaps:
        lines.append("| Function | Missing Parameter(s) | NumPy Signature |")
        lines.append("|----------|---------------------|-----------------|")
        for name, gap_items in high_gaps:
            params_str = ", ".join(f"`{p}`" for p, _ in gap_items)
            sig = np_top[name].get("signature", "") or ""
            sig_display = sig[:80] + "..." if len(sig) > 80 else sig
            lines.append(f"| `{name}` | {params_str} | `{sig_display}` |")
    else:
        lines.append("**No HIGH severity parameter gaps remain!**")
    lines.append("")

    # 4.2 MEDIUM severity
    med_gaps = [(name, [(p, s) for p, s in items if s == "MEDIUM"]) for name, items in all_gaps]
    med_gaps = [(n, g) for n, g in med_gaps if g]

    lines.append("### 4.2 MEDIUM Severity: Missing Moderately-Used Parameters\n")
    lines.append("Parameters like `order=`, `casting=`, `where=`, `initial=`, `kind=`, `mode=`, etc.\n")

    if med_gaps:
        lines.append("| Function | Missing Parameter(s) | NumPy Signature |")
        lines.append("|----------|---------------------|-----------------|")
        for name, gap_items in med_gaps:
            params_str = ", ".join(f"`{p}`" for p, _ in gap_items)
            sig = np_top[name].get("signature", "") or ""
            sig_display = sig[:80] + "..." if len(sig) > 80 else sig
            lines.append(f"| `{name}` | {params_str} | `{sig_display}` |")
    else:
        lines.append("**No MEDIUM severity parameter gaps remain!**")
    lines.append("")

    # 4.3 LOW severity
    low_gaps = [(name, [(p, s) for p, s in items if s == "LOW"]) for name, items in all_gaps]
    low_gaps = [(n, g) for n, g in low_gaps if g]

    lines.append("### 4.3 LOW Severity: Missing Rarely-Used Parameters\n")
    lines.append("Parameters like `subok=`, `like=`, `signature=`, `extobj=`, etc.\n")

    if low_gaps:
        lines.append("<details>")
        lines.append(f"<summary>Click to expand ({len(low_gaps)} functions affected)</summary>\n")
        lines.append("| Function | Missing Parameter(s) |")
        lines.append("|----------|---------------------|")
        for name, gap_items in low_gaps:
            params_str = ", ".join(f"`{p}`" for p, _ in gap_items)
            lines.append(f"| `{name}` | {params_str} |")
        lines.append("\n</details>\n")
    else:
        lines.append("**No LOW severity parameter gaps remain!**")
        lines.append("")

    # ---------- Section 5: Summary ----------
    lines.append("---\n")
    lines.append("## 5. Summary\n")

    total_np_funcs = len([n for n in np_names if np_top[n]["type"] in ("function", "ufunc")])
    common_funcs = len([n for n in common if np_top[n]["type"] in ("function", "ufunc")])
    if total_np_funcs > 0:
        func_coverage = common_funcs / total_np_funcs * 100
    else:
        func_coverage = 0

    total_params_in_np = 0
    matched_params = 0
    for name in common:
        np_info = np_top[name]
        mmp_info = mmp_top[name]
        if np_info["type"] not in ("function", "ufunc"):
            continue
        if mmp_info["type"] not in ("function",):
            continue
        np_params = np_info.get("params")
        mmp_params = mmp_info.get("params")
        if np_params is None or mmp_params is None:
            continue
        np_set = set(np_params) - {"self"}
        mmp_set = set(mmp_params) - {"self"}
        total_params_in_np += len(np_set)
        matched_params += len(np_set & mmp_set)

    if total_params_in_np > 0:
        param_coverage = matched_params / total_params_in_np * 100
    else:
        param_coverage = 0

    lines.append(f"- **Function coverage**: {common_funcs}/{total_np_funcs} ({func_coverage:.1f}%)")
    lines.append(f"- **Parameter coverage**: {matched_params}/{total_params_in_np} ({param_coverage:.1f}%)")
    lines.append(f"- **HIGH severity gaps remaining**: {total_high}")
    lines.append(f"- **MEDIUM severity gaps remaining**: {total_med}")
    lines.append(f"- **LOW severity gaps remaining**: {total_low}")
    lines.append("")

    # Write report
    report_path = os.path.join(SCRIPT_DIR, "API_COMPARISON_REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n=== API Comparison Summary ===")
    print(f"Function coverage: {common_funcs}/{total_np_funcs} ({func_coverage:.1f}%)")
    print(f"Parameter coverage: {matched_params}/{total_params_in_np} ({param_coverage:.1f}%)")
    print(f"HIGH severity gaps: {total_high}")
    print(f"MEDIUM severity gaps: {total_med}")
    print(f"LOW severity gaps: {total_low}")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    generate_report()
