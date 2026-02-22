"""Enumerate all NumPy 2 public APIs across top-level, linalg, fft, and random modules."""

import inspect
import numpy as np
import numpy.linalg
import numpy.fft
import numpy.random

MODULES = [
    ("numpy (top-level)", np),
    ("numpy.linalg", np.linalg),
    ("numpy.fft", np.fft),
    ("numpy.random", np.random),
]


def classify(module, name):
    """Classify a name in a module as function, class, or constant/other."""
    obj = getattr(module, name)
    if inspect.isclass(obj):
        return "class"
    elif callable(obj):
        return "function"
    else:
        return "constant"


def enumerate_module(module):
    """Return dict with keys 'functions', 'classes', 'constants' listing public names."""
    functions = []
    classes = []
    constants = []

    for name in sorted(dir(module)):
        # Skip private/dunder names
        if name.startswith("_"):
            continue
        category = classify(module, name)
        if category == "function":
            functions.append(name)
        elif category == "class":
            classes.append(name)
        else:
            constants.append(name)

    return {"functions": functions, "classes": classes, "constants": constants}


def main():
    lines = []
    lines.append(f"NumPy version: {np.__version__}")
    lines.append("")

    total_functions = 0
    total_classes = 0
    total_constants = 0

    for section_name, module in MODULES:
        result = enumerate_module(module)
        lines.append(f"=== {section_name} ===")

        lines.append("Functions:")
        for name in result["functions"]:
            lines.append(f"  {name}")
        lines.append("")

        lines.append("Classes:")
        for name in result["classes"]:
            lines.append(f"  {name}")
        lines.append("")

        lines.append("Constants:")
        for name in result["constants"]:
            lines.append(f"  {name}")
        lines.append("")

        nf = len(result["functions"])
        nc = len(result["classes"])
        nk = len(result["constants"])
        lines.append(f"  Subtotal: {nf} functions, {nc} classes, {nk} constants ({nf + nc + nk} total)")
        lines.append("")
        lines.append("")

        total_functions += nf
        total_classes += nc
        total_constants += nk

    lines.append("=== SUMMARY ===")
    lines.append(f"Total functions: {total_functions}")
    lines.append(f"Total classes:   {total_classes}")
    lines.append(f"Total constants: {total_constants}")
    lines.append(f"Grand total:     {total_functions + total_classes + total_constants}")

    output = "\n".join(lines) + "\n"
    output_path = "benchmarks/numpy2_apis_v2.txt"
    with open(output_path, "w") as f:
        f.write(output)
    print(f"Wrote {len(lines)} lines to {output_path}")
    print(f"Grand total: {total_functions + total_classes + total_constants} public APIs")


if __name__ == "__main__":
    main()
