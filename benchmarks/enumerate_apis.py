#!/usr/bin/env python3
"""Enumerate all public APIs in macmetalpy and its submodules."""

import sys
import inspect
import types

# Modules to inspect
MODULE_NAMES = [
    "macmetalpy",
    "macmetalpy.linalg",
    "macmetalpy.fft",
    "macmetalpy.random",
]

def get_public_names(mod):
    """Return sorted list of public names (not starting with _)."""
    if hasattr(mod, "__all__"):
        return sorted(mod.__all__)
    return sorted(n for n in dir(mod) if not n.startswith("_"))


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
    """Try to get the signature string; return '' on failure."""
    try:
        return str(inspect.signature(obj))
    except (ValueError, TypeError):
        return ""


def main():
    lines = []
    lines.append("=" * 72)
    lines.append("macmetalpy Public API Catalog")
    lines.append("=" * 72)
    lines.append("")

    for mod_name in MODULE_NAMES:
        try:
            mod = __import__(mod_name, fromlist=["__name__"])
        except ImportError as e:
            lines.append(f"ERROR importing {mod_name}: {e}")
            lines.append("")
            continue

        names = get_public_names(mod)

        lines.append("-" * 72)
        lines.append(f"Module: {mod_name}  ({len(names)} public names)")
        lines.append("-" * 72)

        # Check __all__
        if hasattr(mod, "__all__"):
            lines.append(f"  __all__ defined: yes ({len(mod.__all__)} entries)")
        else:
            lines.append(f"  __all__ defined: no")
        lines.append("")

        functions = []
        classes = []
        constants = []
        modules = []

        for name in names:
            obj = getattr(mod, name)
            kind = classify(obj)
            if kind == "function":
                sig = get_signature(obj)
                functions.append((name, sig))
            elif kind == "class":
                sig = get_signature(obj)
                classes.append((name, sig))
            elif kind == "module":
                modules.append(name)
            else:
                constants.append((name, type(obj).__name__, repr(obj)[:80]))

        if classes:
            lines.append(f"  Classes ({len(classes)}):")
            for name, sig in classes:
                lines.append(f"    {name}{sig}")
            lines.append("")

        if functions:
            lines.append(f"  Functions ({len(functions)}):")
            for name, sig in functions:
                lines.append(f"    {name}{sig}")
            lines.append("")

        if constants:
            lines.append(f"  Constants ({len(constants)}):")
            for name, tname, val in constants:
                lines.append(f"    {name} ({tname}) = {val}")
            lines.append("")

        if modules:
            lines.append(f"  Submodules ({len(modules)}):")
            for name in modules:
                lines.append(f"    {name}")
            lines.append("")

    # Also scan __init__.py for __all__ exports
    lines.append("=" * 72)
    lines.append("__all__ exports from __init__.py files")
    lines.append("=" * 72)
    lines.append("")
    for mod_name in MODULE_NAMES:
        try:
            mod = __import__(mod_name, fromlist=["__name__"])
        except ImportError:
            continue
        if hasattr(mod, "__all__"):
            lines.append(f"{mod_name}.__all__: {sorted(mod.__all__)}")
        else:
            lines.append(f"{mod_name}.__all__: NOT DEFINED (using dir() fallback)")
        lines.append("")

    output = "\n".join(lines)
    print(output)

    # Write to file
    out_path = "/Users/grantklepzig/git/macmetalpy/benchmarks/macmetalpy_apis.txt"
    with open(out_path, "w") as f:
        f.write(output + "\n")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
