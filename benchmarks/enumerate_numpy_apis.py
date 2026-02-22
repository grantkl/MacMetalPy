"""
Enumerate all public functions, classes, and constants in NumPy 2 and its key submodules.
Outputs a grouped listing with signatures where available.
"""

import importlib
import inspect
import sys
import types


MODULES = [
    "numpy",
    "numpy.linalg",
    "numpy.fft",
    "numpy.random",
    "numpy.exceptions",
    "numpy.lib",
    "numpy.char",
    "numpy.polynomial",
    "numpy.testing",
    "numpy.ma",
]


def get_public_names(mod):
    """Return sorted list of public names (not starting with _)."""
    if hasattr(mod, "__all__"):
        names = [n for n in mod.__all__ if not n.startswith("_")]
    else:
        names = [n for n in dir(mod) if not n.startswith("_")]
    return sorted(set(names))


def classify(obj):
    """Classify an object as function, class, constant, or module."""
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "function"
    if isinstance(obj, types.ModuleType):
        return "module"
    return "constant"


def get_signature(obj):
    """Try to get the signature string for a callable."""
    try:
        sig = inspect.signature(obj)
        return str(sig)
    except (ValueError, TypeError):
        return "(…)"


def enumerate_module(mod_name):
    """Enumerate all public APIs in a module and return a structured dict."""
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        return {"error": str(e)}

    names = get_public_names(mod)
    functions = []
    classes = []
    constants = []

    for name in names:
        try:
            obj = getattr(mod, name)
        except AttributeError:
            continue

        kind = classify(obj)

        if kind == "module":
            continue  # skip submodule references

        if kind == "class":
            sig = get_signature(obj)
            classes.append((name, sig))
        elif kind == "function":
            sig = get_signature(obj)
            functions.append((name, sig))
        else:
            # constant
            val_repr = repr(obj)
            if len(val_repr) > 120:
                val_repr = val_repr[:120] + "..."
            constants.append((name, type(obj).__name__, val_repr))

    return {
        "functions": functions,
        "classes": classes,
        "constants": constants,
    }


def format_output(mod_name, data):
    lines = []
    lines.append("=" * 80)
    lines.append(f"MODULE: {mod_name}")
    lines.append("=" * 80)

    if "error" in data:
        lines.append(f"  ERROR: {data['error']}")
        lines.append("")
        return "\n".join(lines)

    # Functions
    lines.append(f"\n  Functions ({len(data['functions'])}):")
    lines.append("  " + "-" * 40)
    for name, sig in data["functions"]:
        lines.append(f"    {name}{sig}")

    # Classes
    lines.append(f"\n  Classes ({len(data['classes'])}):")
    lines.append("  " + "-" * 40)
    for name, sig in data["classes"]:
        lines.append(f"    {name}{sig}")

    # Constants
    lines.append(f"\n  Constants ({len(data['constants'])}):")
    lines.append("  " + "-" * 40)
    for name, typ, val in data["constants"]:
        lines.append(f"    {name}: {typ} = {val}")

    lines.append("")
    return "\n".join(lines)


def main():
    import numpy as np

    output_lines = []
    output_lines.append(f"NumPy Public API Enumeration")
    output_lines.append(f"NumPy version: {np.__version__}")
    output_lines.append(f"Python version: {sys.version}")
    output_lines.append(f"Modules inspected: {len(MODULES)}")
    output_lines.append("")

    total_functions = 0
    total_classes = 0
    total_constants = 0

    for mod_name in MODULES:
        data = enumerate_module(mod_name)
        output_lines.append(format_output(mod_name, data))
        if "error" not in data:
            total_functions += len(data["functions"])
            total_classes += len(data["classes"])
            total_constants += len(data["constants"])

    # Summary
    output_lines.append("=" * 80)
    output_lines.append("SUMMARY")
    output_lines.append("=" * 80)
    output_lines.append(f"  Total functions: {total_functions}")
    output_lines.append(f"  Total classes:   {total_classes}")
    output_lines.append(f"  Total constants: {total_constants}")
    output_lines.append(f"  Grand total:     {total_functions + total_classes + total_constants}")
    output_lines.append("")

    result = "\n".join(output_lines)

    output_path = "/Users/grantklepzig/git/macmetalpy/benchmarks/numpy2_apis.txt"
    with open(output_path, "w") as f:
        f.write(result)

    print(f"Output written to {output_path}")
    print(f"Total: {total_functions} functions, {total_classes} classes, {total_constants} constants")


if __name__ == "__main__":
    main()
