"""Lazy kernel fusion engine for elementwise operations.

Builds a DAG of elementwise ops, then compiles and dispatches a single
fused Metal kernel that evaluates the entire graph in one GPU pass.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from ._dtypes import METAL_TYPE_NAMES

__all__ = [
    "_UNARY_EXPR",
    "_BINARY_EXPR",
    "InputNode",
    "UnaryOpNode",
    "BinaryOpNode",
    "materialize",
    "_MAX_DEPTH",
]

_MAX_DEPTH = 16

_MSL_HEADER = """\
#include <metal_stdlib>
using namespace metal;
kernel void _sync() {}
"""

# ---------------------------------------------------------------------------
# FusionNode hierarchy
# ---------------------------------------------------------------------------


class FusionNode:
    __slots__ = ("_shape", "_dtype", "_depth")


class InputNode(FusionNode):
    __slots__ = ("_buffer",)

    def __init__(self, buffer, shape, dtype):
        self._buffer = buffer
        self._shape = shape
        self._dtype = dtype
        self._depth = 0


class UnaryOpNode(FusionNode):
    __slots__ = ("_op_name", "_input")

    def __init__(self, op_name, input_node, shape, dtype):
        self._op_name = op_name
        self._input = input_node
        self._shape = shape
        self._dtype = dtype
        self._depth = input_node._depth + 1


class BinaryOpNode(FusionNode):
    __slots__ = ("_op_name", "_lhs", "_rhs")

    def __init__(self, op_name, lhs, rhs, shape, dtype):
        self._op_name = op_name
        self._lhs = lhs
        self._rhs = rhs
        self._shape = shape
        self._dtype = dtype
        self._depth = max(lhs._depth, rhs._depth) + 1


# ---------------------------------------------------------------------------
# MSL expression tables
#
# Each value is either:
#   - a str template with {x}/{a}/{b} for the operand SSA names and {mt} for
#     the Metal type name
#   - a callable (metal_type: str) -> str  for type-dependent expressions
#
# Multi-statement ops use {stmts} + {result} pattern: the compiler emits
# the statements block and then assigns the result.
# ---------------------------------------------------------------------------

def _get_unary_expr(op_name: str, metal_type: str) -> tuple[str, str]:
    """Return (statements, result_expr) for a unary op.

    statements may be empty for single-expression ops.  For multi-statement
    ops (like cbrt, sign) it contains intermediate lines ending with ';\\n'.
    """
    is_float = metal_type in ("float", "half")

    if op_name == "abs_op":
        if is_float:
            return ("", "fabs({x})")
        return ("", "abs({x})")

    if op_name == "sign_op":
        return (
            "float _sv{uid} = static_cast<float>({x});\n",
            "isnan(_sv{uid}) ? ({mt})(NAN) : ((_sv{uid} > 0.0f) ? ({mt})(1) : ((_sv{uid} < 0.0f) ? ({mt})(-1) : ({mt})(0)))"
        )

    if op_name == "sin_op":
        return (
            "float _sv{uid} = static_cast<float>({x});\n",
            "(isnan(_sv{uid}) || isinf(_sv{uid})) ? ({mt})(NAN) : ({mt})sin(_sv{uid})"
        )

    if op_name == "cos_op":
        return (
            "float _sv{uid} = static_cast<float>({x});\n",
            "(isnan(_sv{uid}) || isinf(_sv{uid})) ? ({mt})(NAN) : ({mt})cos(_sv{uid})"
        )

    if op_name == "tanh_op":
        return (
            "float _sv{uid} = static_cast<float>({x});\n",
            "isinf(_sv{uid}) ? ({mt})(_sv{uid} > 0.0f ? 1.0f : -1.0f) : ({mt})tanh(_sv{uid})"
        )

    if op_name == "cbrt_op":
        return (
            "float _sv{uid} = static_cast<float>({x});\n",
            "copysign(pow(fabs(_sv{uid}), 1.0f / 3.0f), _sv{uid})"
        )

    if op_name == "reciprocal_op":
        return ("", "({mt})(1.0f / static_cast<float>({x}))")

    # Simple cast-through-float unary ops
    _SIMPLE_FLOAT_CAST = {
        "sqrt_op":     "sqrt(static_cast<float>({x}))",
        "exp_op":      "exp(static_cast<float>({x}))",
        "log_op":      "log(static_cast<float>({x}))",
        "floor_op":    "floor(static_cast<float>({x}))",
        "ceil_op":     "ceil(static_cast<float>({x}))",
        "tan_op":      "tan(static_cast<float>({x}))",
        "asin_op":     "asin(static_cast<float>({x}))",
        "acos_op":     "acos(static_cast<float>({x}))",
        "atan_op":     "atan(static_cast<float>({x}))",
        "sinh_op":     "sinh(static_cast<float>({x}))",
        "cosh_op":     "cosh(static_cast<float>({x}))",
        "log2_op":     "log2(static_cast<float>({x}))",
        "log10_op":    "log10(static_cast<float>({x}))",
        "exp2_op":     "exp2(static_cast<float>({x}))",
        "expm1_op":    "exp(static_cast<float>({x})) - 1.0f",
        "log1p_op":    "log(1.0f + static_cast<float>({x}))",
        "rint_op":     "rint(static_cast<float>({x}))",
        "trunc_op":    "trunc(static_cast<float>({x}))",
        "asinh_op":    "asinh(static_cast<float>({x}))",
        "acosh_op":    "acosh(static_cast<float>({x}))",
        "atanh_op":    "atanh(static_cast<float>({x}))",
        "degrees_op":  "static_cast<float>({x}) * 57.29577951308232f",
        "radians_op":  "static_cast<float>({x}) * 0.017453292519943295f",
    }

    if op_name in _SIMPLE_FLOAT_CAST:
        return ("", _SIMPLE_FLOAT_CAST[op_name])

    # Non-cast simple ops
    if op_name == "neg_op" or op_name == "negative_op":
        return ("", "-{x}")

    if op_name == "square_op":
        return ("", "{x} * {x}")

    if op_name == "positive_op":
        return ("", "{x}")

    raise ValueError(f"Unknown unary op for fusion: {op_name}")


def _get_binary_expr(op_name: str, metal_type: str) -> tuple[str, str]:
    """Return (statements, result_expr) for a binary op.

    Same convention as _get_unary_expr.
    """
    is_float = metal_type in ("float", "half")

    if op_name == "add_op":
        return ("", "{a} + {b}")
    if op_name == "sub_op":
        return ("", "{a} - {b}")
    if op_name == "mul_op":
        return ("", "{a} * {b}")
    if op_name == "div_op":
        return ("", "{a} / {b}")

    if op_name == "pow_op":
        if is_float:
            return ("", "pow({a}, {b})")
        # Integer pow uses a loop — cannot fuse as a single expression
        return None  # Signal: do not fuse

    if op_name == "floor_divide_op":
        if is_float:
            return ("", "floor({a} / {b})")
        return ("", "{a} / {b}")

    if op_name == "max_op":
        return (
            "float _av{uid} = static_cast<float>({a}), _bv{uid} = static_cast<float>({b});\n",
            "(isnan(_av{uid}) || isnan(_bv{uid})) ? ({mt})(NAN) : ((_av{uid} > _bv{uid}) ? {a} : {b})"
        )

    if op_name == "min_op":
        return (
            "float _av{uid} = static_cast<float>({a}), _bv{uid} = static_cast<float>({b});\n",
            "(isnan(_av{uid}) || isnan(_bv{uid})) ? ({mt})(NAN) : ((_av{uid} < _bv{uid}) ? {a} : {b})"
        )

    if op_name == "fmax_op":
        return (
            "float _av{uid} = static_cast<float>({a}), _bv{uid} = static_cast<float>({b});\n",
            "isnan(_av{uid}) ? {b} : (isnan(_bv{uid}) ? {a} : ((_av{uid} > _bv{uid}) ? {a} : {b}))"
        )

    if op_name == "fmin_op":
        return (
            "float _av{uid} = static_cast<float>({a}), _bv{uid} = static_cast<float>({b});\n",
            "isnan(_av{uid}) ? {b} : (isnan(_bv{uid}) ? {a} : ((_av{uid} < _bv{uid}) ? {a} : {b}))"
        )

    if op_name == "atan2_op":
        return ("", "atan2(static_cast<float>({a}), static_cast<float>({b}))")

    if op_name == "hypot_op":
        return (
            "float _av{uid} = static_cast<float>({a}), _bv{uid} = static_cast<float>({b});\n",
            "sqrt(_av{uid} * _av{uid} + _bv{uid} * _bv{uid})"
        )

    if op_name == "fmod_op":
        return ("", "fmod(static_cast<float>({a}), static_cast<float>({b}))")

    if op_name == "copysign_op":
        return ("", "copysign(static_cast<float>({a}), static_cast<float>({b}))")

    if op_name == "nextafter_op":
        return ("", "nextafter(static_cast<float>({a}), static_cast<float>({b}))")

    if op_name == "logaddexp_op":
        return (
            "float _av{uid} = static_cast<float>({a}), _bv{uid} = static_cast<float>({b});\n"
            "float _mx{uid} = (_av{uid} > _bv{uid}) ? _av{uid} : _bv{uid};\n",
            "_mx{uid} + log(exp(_av{uid} - _mx{uid}) + exp(_bv{uid} - _mx{uid}))"
        )

    if op_name == "logaddexp2_op":
        return (
            "float _av{uid} = static_cast<float>({a}), _bv{uid} = static_cast<float>({b});\n"
            "float _mx{uid} = (_av{uid} > _bv{uid}) ? _av{uid} : _bv{uid};\n",
            "_mx{uid} + log2(exp2(_av{uid} - _mx{uid}) + exp2(_bv{uid} - _mx{uid}))"
        )

    if op_name == "heaviside_op":
        return (
            "float _sv{uid} = static_cast<float>({a});\n",
            "(_sv{uid} < 0.0f) ? ({mt})0 : ((_sv{uid} == 0.0f) ? {b} : ({mt})1)"
        )

    raise ValueError(f"Unknown binary op for fusion: {op_name}")


# Public expression table references (for introspection / testing)
_UNARY_EXPR = _get_unary_expr
_BINARY_EXPR = _get_binary_expr

# Sets of fuseable op names — used by ndarray.py for ``op_name in ...`` checks
_FUSEABLE_UNARY_OPS = frozenset({
    "sqrt_op", "exp_op", "log_op", "sin_op", "cos_op", "tan_op",
    "asin_op", "acos_op", "atan_op", "sinh_op", "cosh_op", "tanh_op",
    "log2_op", "log10_op", "exp2_op", "expm1_op", "log1p_op",
    "cbrt_op", "reciprocal_op", "rint_op", "trunc_op",
    "asinh_op", "acosh_op", "atanh_op",
    "degrees_op", "radians_op",
    "abs_op", "sign_op", "floor_op", "ceil_op",
    "neg_op", "negative_op", "square_op", "positive_op",
})

_FUSEABLE_BINARY_OPS = frozenset({
    "add_op", "sub_op", "mul_op", "div_op",
    "pow_op",  # float-only; _get_binary_expr returns None for int
    "max_op", "min_op", "fmax_op", "fmin_op",
    "atan2_op", "hypot_op", "floor_divide_op", "fmod_op",
    "copysign_op", "nextafter_op",
    "logaddexp_op", "logaddexp2_op", "heaviside_op",
})


# ---------------------------------------------------------------------------
# Graph-to-MSL compiler
# ---------------------------------------------------------------------------

def _compile_fusion_graph(root, metal_type):
    """Compile a fusion graph into MSL shader source.

    Parameters
    ----------
    root : FusionNode
        The root of the fusion DAG.
    metal_type : str
        The Metal type name (e.g. "float", "half", "int").

    Returns
    -------
    (shader_src, input_buffers, cache_key) where:
        shader_src : str — complete MSL source
        input_buffers : list[_Buffer] — ordered input buffers
        cache_key : tuple — hashable topology key (does NOT include buffer identity)
    """
    # Post-order traversal with deduplication by id()
    visited = OrderedDict()  # id(node) -> (ssa_name, node)
    input_buffers = []  # ordered list of _Buffer
    input_map = {}  # id(buffer) -> input index
    stmts = []  # list of MSL statement strings
    cache_parts = []  # for building cache_key
    counter = 0

    def _visit(node):
        nonlocal counter
        nid = id(node)
        if nid in visited:
            return visited[nid][0]

        if isinstance(node, InputNode):
            bid = id(node._buffer)
            if bid not in input_map:
                input_map[bid] = len(input_buffers)
                input_buffers.append(node._buffer)
            idx = input_map[bid]
            ssa = f"t{counter}"
            counter += 1
            stmts.append(f"    {metal_type} {ssa} = in{idx}[id];")
            cache_parts.append(("input", idx))
            visited[nid] = (ssa, node)
            return ssa

        if isinstance(node, UnaryOpNode):
            x_ssa = _visit(node._input)
            expr_result = _get_unary_expr(node._op_name, metal_type)
            ssa = f"t{counter}"
            uid = counter
            counter += 1

            stmt_tmpl, result_tmpl = expr_result
            # Format templates with operand names
            fmt = {"x": x_ssa, "mt": metal_type, "uid": uid}
            if stmt_tmpl:
                formatted_stmts = stmt_tmpl.format(**fmt)
                for line in formatted_stmts.strip().split("\n"):
                    stmts.append(f"    {line}")
            result_expr = result_tmpl.format(**fmt)
            stmts.append(f"    {metal_type} {ssa} = {result_expr};")
            cache_parts.append(("unary", node._op_name))
            visited[nid] = (ssa, node)
            return ssa

        if isinstance(node, BinaryOpNode):
            a_ssa = _visit(node._lhs)
            b_ssa = _visit(node._rhs)
            expr_result = _get_binary_expr(node._op_name, metal_type)
            if expr_result is None:
                raise ValueError(
                    f"Cannot fuse binary op {node._op_name} for type {metal_type}"
                )
            ssa = f"t{counter}"
            uid = counter
            counter += 1

            stmt_tmpl, result_tmpl = expr_result
            fmt = {"a": a_ssa, "b": b_ssa, "mt": metal_type, "uid": uid}
            if stmt_tmpl:
                formatted_stmts = stmt_tmpl.format(**fmt)
                for line in formatted_stmts.strip().split("\n"):
                    stmts.append(f"    {line}")
            result_expr = result_tmpl.format(**fmt)
            stmts.append(f"    {metal_type} {ssa} = {result_expr};")
            cache_parts.append(("binary", node._op_name))
            visited[nid] = (ssa, node)
            return ssa

        raise TypeError(f"Unknown node type: {type(node)}")

    root_ssa = _visit(root)

    # Build buffer parameter list
    n_inputs = len(input_buffers)
    params = []
    for i in range(n_inputs):
        params.append(f"device {metal_type} *in{i} [[buffer({i})]]")
    params.append(f"device {metal_type} *out [[buffer({n_inputs})]]")
    params.append("uint id [[thread_position_in_grid]]")

    param_str = ",\n                     ".join(params)
    body = "\n".join(stmts)

    shader_src = (
        _MSL_HEADER
        + f"kernel void fused_op({param_str}) {{\n"
        + body + "\n"
        + f"    out[id] = {root_ssa};\n"
        + "}\n"
    )

    cache_key = (metal_type, tuple(cache_parts))
    return shader_src, input_buffers, cache_key


# ---------------------------------------------------------------------------
# FusedKernelCache — thread-safe singleton
# ---------------------------------------------------------------------------

class _FusedKernelCache:
    __slots__ = ("_cache", "_lock")

    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            return self._cache.get(key)

    def put(self, key, shader_src):
        with self._lock:
            self._cache[key] = shader_src

    def clear(self):
        with self._lock:
            self._cache.clear()


_kernel_cache = _FusedKernelCache()


# ---------------------------------------------------------------------------
# materialize — compile and dispatch the fusion graph
# ---------------------------------------------------------------------------

def materialize(arr):
    """Compile and dispatch the fusion graph, materializing the result.

    After this call, ``arr._buffer`` holds the computed data and
    ``arr._fusion_node`` is set to None.
    """
    from ._metal_backend import MetalBackend

    node = arr._fusion_node
    metal_type = METAL_TYPE_NAMES[arr._dtype]

    shader_src, input_buffers, cache_key = _compile_fusion_graph(node, metal_type)

    # Check the cache
    cached = _kernel_cache.get(cache_key)
    if cached is not None:
        shader_src = cached
    else:
        _kernel_cache.put(cache_key, shader_src)

    backend = MetalBackend()
    out_buf = backend.create_buffer(arr.size, arr._dtype)
    backend.execute_kernel(shader_src, "fused_op", arr.size, input_buffers + [out_buf])

    arr._buffer = out_buf
    arr._fusion_node = None
    arr._np_data = None
