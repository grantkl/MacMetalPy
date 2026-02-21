"""Metal Shading Language (MSL) kernel source generators."""

from __future__ import annotations

__all__ = [
    "elementwise_shader", "reduction_shader", "matmul_shader",
    "comparison_shader", "boolean_shader", "where_shader", "clip_shader",
    "predicate_shader", "axis_reduction_shader",
]

_MSL_HEADER = """\
#include <metal_stdlib>
using namespace metal;
kernel void _sync() {}
"""


def elementwise_shader(metal_type: str) -> str:
    """Return MSL source with 11 elementwise kernels parameterised on *metal_type*."""
    # For pow: floats use metal_stdlib pow, ints use manual loop
    is_float = metal_type in ("float", "half")

    if is_float:
        pow_body = f"out[id] = pow(a[id], b[id]);"
        abs_body = f"out[id] = fabs(a[id]);"
        floor_divide_body = f"out[id] = floor(a[id] / b[id]);"
    else:
        pow_body = (
            f"{metal_type} base = a[id];\n"
            f"    {metal_type} exp_val = b[id];\n"
            f"    {metal_type} result = 1;\n"
            f"    for ({metal_type} i = 0; i < exp_val; i++) {{\n"
            f"        result *= base;\n"
            f"    }}\n"
            f"    out[id] = result;"
        )
        abs_body = f"out[id] = abs(a[id]);"
        floor_divide_body = f"out[id] = a[id] / b[id];"

    return (
        _MSL_HEADER
        + f"""
kernel void add_op(device {metal_type} *a [[buffer(0)]],
                   device {metal_type} *b [[buffer(1)]],
                   device {metal_type} *out [[buffer(2)]],
                   uint id [[thread_position_in_grid]]) {{
    out[id] = a[id] + b[id];
}}

kernel void sub_op(device {metal_type} *a [[buffer(0)]],
                   device {metal_type} *b [[buffer(1)]],
                   device {metal_type} *out [[buffer(2)]],
                   uint id [[thread_position_in_grid]]) {{
    out[id] = a[id] - b[id];
}}

kernel void mul_op(device {metal_type} *a [[buffer(0)]],
                   device {metal_type} *b [[buffer(1)]],
                   device {metal_type} *out [[buffer(2)]],
                   uint id [[thread_position_in_grid]]) {{
    out[id] = a[id] * b[id];
}}

kernel void div_op(device {metal_type} *a [[buffer(0)]],
                   device {metal_type} *b [[buffer(1)]],
                   device {metal_type} *out [[buffer(2)]],
                   uint id [[thread_position_in_grid]]) {{
    out[id] = a[id] / b[id];
}}

kernel void pow_op(device {metal_type} *a [[buffer(0)]],
                   device {metal_type} *b [[buffer(1)]],
                   device {metal_type} *out [[buffer(2)]],
                   uint id [[thread_position_in_grid]]) {{
    {pow_body}
}}

kernel void neg_op(device {metal_type} *a [[buffer(0)]],
                   device {metal_type} *out [[buffer(1)]],
                   uint id [[thread_position_in_grid]]) {{
    out[id] = -a[id];
}}

kernel void abs_op(device {metal_type} *a [[buffer(0)]],
                   device {metal_type} *out [[buffer(1)]],
                   uint id [[thread_position_in_grid]]) {{
    {abs_body}
}}

kernel void sqrt_op(device {metal_type} *a [[buffer(0)]],
                    device {metal_type} *out [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {{
    out[id] = sqrt(static_cast<float>(a[id]));
}}

kernel void exp_op(device {metal_type} *a [[buffer(0)]],
                   device {metal_type} *out [[buffer(1)]],
                   uint id [[thread_position_in_grid]]) {{
    out[id] = exp(static_cast<float>(a[id]));
}}

kernel void log_op(device {metal_type} *a [[buffer(0)]],
                   device {metal_type} *out [[buffer(1)]],
                   uint id [[thread_position_in_grid]]) {{
    out[id] = log(static_cast<float>(a[id]));
}}

kernel void fill_scalar(device {metal_type} *out [[buffer(0)]],
                        device {metal_type} *val [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {{
    out[id] = val[0];
}}

kernel void sign_op(device {metal_type} *a [[buffer(0)]],
                    device {metal_type} *out [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {{
    float v = static_cast<float>(a[id]);
    out[id] = isnan(v) ? ({metal_type})(NAN) : ((v > 0.0f) ? ({metal_type})(1) : ((v < 0.0f) ? ({metal_type})(-1) : ({metal_type})(0)));
}}

kernel void floor_op(device {metal_type} *a [[buffer(0)]],
                     device {metal_type} *out [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {{
    out[id] = floor(static_cast<float>(a[id]));
}}

kernel void ceil_op(device {metal_type} *a [[buffer(0)]],
                    device {metal_type} *out [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {{
    out[id] = ceil(static_cast<float>(a[id]));
}}

kernel void sin_op(device {metal_type} *a [[buffer(0)]],
                   device {metal_type} *out [[buffer(1)]],
                   uint id [[thread_position_in_grid]]) {{
    float v = static_cast<float>(a[id]);
    out[id] = (isnan(v) || isinf(v)) ? ({metal_type})(NAN) : ({metal_type})sin(v);
}}

kernel void cos_op(device {metal_type} *a [[buffer(0)]],
                   device {metal_type} *out [[buffer(1)]],
                   uint id [[thread_position_in_grid]]) {{
    float v = static_cast<float>(a[id]);
    out[id] = (isnan(v) || isinf(v)) ? ({metal_type})(NAN) : ({metal_type})cos(v);
}}

kernel void tan_op(device {metal_type} *a [[buffer(0)]],
                   device {metal_type} *out [[buffer(1)]],
                   uint id [[thread_position_in_grid]]) {{
    out[id] = tan(static_cast<float>(a[id]));
}}

kernel void asin_op(device {metal_type} *a [[buffer(0)]],
                    device {metal_type} *out [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {{
    out[id] = asin(static_cast<float>(a[id]));
}}

kernel void acos_op(device {metal_type} *a [[buffer(0)]],
                    device {metal_type} *out [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {{
    out[id] = acos(static_cast<float>(a[id]));
}}

kernel void atan_op(device {metal_type} *a [[buffer(0)]],
                    device {metal_type} *out [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {{
    out[id] = atan(static_cast<float>(a[id]));
}}

kernel void sinh_op(device {metal_type} *a [[buffer(0)]],
                    device {metal_type} *out [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {{
    out[id] = sinh(static_cast<float>(a[id]));
}}

kernel void cosh_op(device {metal_type} *a [[buffer(0)]],
                    device {metal_type} *out [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {{
    out[id] = cosh(static_cast<float>(a[id]));
}}

kernel void tanh_op(device {metal_type} *a [[buffer(0)]],
                    device {metal_type} *out [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {{
    float v = static_cast<float>(a[id]);
    out[id] = isinf(v) ? ({metal_type})(v > 0.0f ? 1.0f : -1.0f) : ({metal_type})tanh(v);
}}

kernel void log2_op(device {metal_type} *a [[buffer(0)]],
                    device {metal_type} *out [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {{
    out[id] = log2(static_cast<float>(a[id]));
}}

kernel void log10_op(device {metal_type} *a [[buffer(0)]],
                     device {metal_type} *out [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {{
    out[id] = log10(static_cast<float>(a[id]));
}}

kernel void square_op(device {metal_type} *a [[buffer(0)]],
                      device {metal_type} *out [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {{
    out[id] = a[id] * a[id];
}}

kernel void negative_op(device {metal_type} *a [[buffer(0)]],
                        device {metal_type} *out [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {{
    out[id] = -a[id];
}}

kernel void max_op(device {metal_type} *a [[buffer(0)]],
                   device {metal_type} *b [[buffer(1)]],
                   device {metal_type} *out [[buffer(2)]],
                   uint id [[thread_position_in_grid]]) {{
    float av = static_cast<float>(a[id]), bv = static_cast<float>(b[id]);
    out[id] = (isnan(av) || isnan(bv)) ? ({metal_type})(NAN) : ((av > bv) ? a[id] : b[id]);
}}

kernel void min_op(device {metal_type} *a [[buffer(0)]],
                   device {metal_type} *b [[buffer(1)]],
                   device {metal_type} *out [[buffer(2)]],
                   uint id [[thread_position_in_grid]]) {{
    float av = static_cast<float>(a[id]), bv = static_cast<float>(b[id]);
    out[id] = (isnan(av) || isnan(bv)) ? ({metal_type})(NAN) : ((av < bv) ? a[id] : b[id]);
}}

kernel void fmax_op(device {metal_type} *a [[buffer(0)]],
                    device {metal_type} *b [[buffer(1)]],
                    device {metal_type} *out [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {{
    float av = static_cast<float>(a[id]), bv = static_cast<float>(b[id]);
    out[id] = isnan(av) ? b[id] : (isnan(bv) ? a[id] : ((av > bv) ? a[id] : b[id]));
}}

kernel void fmin_op(device {metal_type} *a [[buffer(0)]],
                    device {metal_type} *b [[buffer(1)]],
                    device {metal_type} *out [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {{
    float av = static_cast<float>(a[id]), bv = static_cast<float>(b[id]);
    out[id] = isnan(av) ? b[id] : (isnan(bv) ? a[id] : ((av < bv) ? a[id] : b[id]));
}}

// ── new unary math ops ──────────────────────────────────────

kernel void exp2_op(device {metal_type} *a [[buffer(0)]],
                    device {metal_type} *out [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {{
    out[id] = exp2(static_cast<float>(a[id]));
}}

kernel void expm1_op(device {metal_type} *a [[buffer(0)]],
                     device {metal_type} *out [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {{
    out[id] = exp(static_cast<float>(a[id])) - 1.0f;
}}

kernel void log1p_op(device {metal_type} *a [[buffer(0)]],
                     device {metal_type} *out [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {{
    out[id] = log(1.0f + static_cast<float>(a[id]));
}}

kernel void cbrt_op(device {metal_type} *a [[buffer(0)]],
                    device {metal_type} *out [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {{
    float v = static_cast<float>(a[id]);
    out[id] = copysign(pow(fabs(v), 1.0f / 3.0f), v);
}}

kernel void reciprocal_op(device {metal_type} *a [[buffer(0)]],
                          device {metal_type} *out [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {{
    out[id] = ({metal_type})(1.0f / static_cast<float>(a[id]));
}}

kernel void rint_op(device {metal_type} *a [[buffer(0)]],
                    device {metal_type} *out [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {{
    out[id] = rint(static_cast<float>(a[id]));
}}

kernel void trunc_op(device {metal_type} *a [[buffer(0)]],
                     device {metal_type} *out [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {{
    out[id] = trunc(static_cast<float>(a[id]));
}}

kernel void positive_op(device {metal_type} *a [[buffer(0)]],
                        device {metal_type} *out [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {{
    out[id] = a[id];
}}

kernel void asinh_op(device {metal_type} *a [[buffer(0)]],
                     device {metal_type} *out [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {{
    out[id] = asinh(static_cast<float>(a[id]));
}}

kernel void acosh_op(device {metal_type} *a [[buffer(0)]],
                     device {metal_type} *out [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {{
    out[id] = acosh(static_cast<float>(a[id]));
}}

kernel void atanh_op(device {metal_type} *a [[buffer(0)]],
                     device {metal_type} *out [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {{
    out[id] = atanh(static_cast<float>(a[id]));
}}

kernel void degrees_op(device {metal_type} *a [[buffer(0)]],
                       device {metal_type} *out [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {{
    out[id] = static_cast<float>(a[id]) * 57.29577951308232f;
}}

kernel void radians_op(device {metal_type} *a [[buffer(0)]],
                       device {metal_type} *out [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {{
    out[id] = static_cast<float>(a[id]) * 0.017453292519943295f;
}}

// ── new binary math ops ─────────────────────────────────────

kernel void atan2_op(device {metal_type} *a [[buffer(0)]],
                     device {metal_type} *b [[buffer(1)]],
                     device {metal_type} *out [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {{
    out[id] = atan2(static_cast<float>(a[id]), static_cast<float>(b[id]));
}}

kernel void hypot_op(device {metal_type} *a [[buffer(0)]],
                     device {metal_type} *b [[buffer(1)]],
                     device {metal_type} *out [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {{
    float av = static_cast<float>(a[id]), bv = static_cast<float>(b[id]);
    out[id] = sqrt(av * av + bv * bv);
}}

kernel void floor_divide_op(device {metal_type} *a [[buffer(0)]],
                             device {metal_type} *b [[buffer(1)]],
                             device {metal_type} *out [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {{
    {floor_divide_body}
}}

kernel void fmod_op(device {metal_type} *a [[buffer(0)]],
                    device {metal_type} *b [[buffer(1)]],
                    device {metal_type} *out [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {{
    out[id] = fmod(static_cast<float>(a[id]), static_cast<float>(b[id]));
}}

kernel void copysign_op(device {metal_type} *a [[buffer(0)]],
                        device {metal_type} *b [[buffer(1)]],
                        device {metal_type} *out [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {{
    out[id] = copysign(static_cast<float>(a[id]), static_cast<float>(b[id]));
}}

kernel void nextafter_op(device {metal_type} *a [[buffer(0)]],
                         device {metal_type} *b [[buffer(1)]],
                         device {metal_type} *out [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {{
    out[id] = nextafter(static_cast<float>(a[id]), static_cast<float>(b[id]));
}}

kernel void logaddexp_op(device {metal_type} *a [[buffer(0)]],
                         device {metal_type} *b [[buffer(1)]],
                         device {metal_type} *out [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {{
    float av = static_cast<float>(a[id]), bv = static_cast<float>(b[id]);
    float mx = (av > bv) ? av : bv;
    out[id] = mx + log(exp(av - mx) + exp(bv - mx));
}}

kernel void logaddexp2_op(device {metal_type} *a [[buffer(0)]],
                          device {metal_type} *b [[buffer(1)]],
                          device {metal_type} *out [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {{
    float av = static_cast<float>(a[id]), bv = static_cast<float>(b[id]);
    float mx = (av > bv) ? av : bv;
    out[id] = mx + log2(exp2(av - mx) + exp2(bv - mx));
}}

kernel void heaviside_op(device {metal_type} *a [[buffer(0)]],
                         device {metal_type} *b [[buffer(1)]],
                         device {metal_type} *out [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {{
    float v = static_cast<float>(a[id]);
    out[id] = (v < 0.0f) ? ({metal_type})0 : ((v == 0.0f) ? b[id] : ({metal_type})1);
}}
"""
    )


def reduction_shader(metal_type: str) -> str:
    """Return MSL source with 3 parallel-reduction kernels parameterised on *metal_type*."""
    return (
        _MSL_HEADER
        + f"""
kernel void reduce_sum(device {metal_type} *input [[buffer(0)]],
                       device {metal_type} *output [[buffer(1)]],
                       device uint *n_elements [[buffer(2)]],
                       uint tid [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint gid [[threadgroup_position_in_grid]],
                       uint group_size [[threads_per_threadgroup]]) {{
    threadgroup {metal_type} shared_data[1024];
    uint n = n_elements[0];
    shared_data[lid] = (tid < n) ? input[tid] : 0;
    // Zero out slots beyond group_size to handle non-power-of-2 threadgroups
    for (uint i = lid + group_size; i < 1024u; i += group_size) {{
        shared_data[i] = 0;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 512u; s > 0; s >>= 1) {{
        if (lid < s) {{
            shared_data[lid] += shared_data[lid + s];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (lid == 0) {{
        output[gid] = shared_data[0];
    }}
}}

kernel void reduce_max(device {metal_type} *input [[buffer(0)]],
                       device {metal_type} *output [[buffer(1)]],
                       device uint *n_elements [[buffer(2)]],
                       uint tid [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint gid [[threadgroup_position_in_grid]],
                       uint group_size [[threads_per_threadgroup]]) {{
    threadgroup {metal_type} shared_data[1024];
    uint n = n_elements[0];
    shared_data[lid] = (tid < n) ? input[tid] : input[0];
    for (uint i = lid + group_size; i < 1024u; i += group_size) {{
        shared_data[i] = input[0];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 512u; s > 0; s >>= 1) {{
        if (lid < s) {{
            float a = static_cast<float>(shared_data[lid]);
            float b = static_cast<float>(shared_data[lid + s]);
            if (isnan(b) || (!isnan(a) && b > a)) {{
                shared_data[lid] = shared_data[lid + s];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (lid == 0) {{
        output[gid] = shared_data[0];
    }}
}}

kernel void reduce_min(device {metal_type} *input [[buffer(0)]],
                       device {metal_type} *output [[buffer(1)]],
                       device uint *n_elements [[buffer(2)]],
                       uint tid [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint gid [[threadgroup_position_in_grid]],
                       uint group_size [[threads_per_threadgroup]]) {{
    threadgroup {metal_type} shared_data[1024];
    uint n = n_elements[0];
    shared_data[lid] = (tid < n) ? input[tid] : input[0];
    for (uint i = lid + group_size; i < 1024u; i += group_size) {{
        shared_data[i] = input[0];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 512u; s > 0; s >>= 1) {{
        if (lid < s) {{
            float a = static_cast<float>(shared_data[lid]);
            float b = static_cast<float>(shared_data[lid + s]);
            if (isnan(b) || (!isnan(a) && b < a)) {{
                shared_data[lid] = shared_data[lid + s];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (lid == 0) {{
        output[gid] = shared_data[0];
    }}
}}
"""
    )


def matmul_shader(metal_type: str) -> str:
    """Return MSL source with a matrix-multiply kernel parameterised on *metal_type*."""
    return (
        _MSL_HEADER
        + f"""
kernel void matmul_op(device {metal_type} *A [[buffer(0)]],
                      device {metal_type} *B [[buffer(1)]],
                      device {metal_type} *C [[buffer(2)]],
                      device uint *dims [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {{
    uint M = dims[0];
    uint N = dims[1];
    uint K = dims[2];

    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    {metal_type} sum = 0;
    for (uint i = 0; i < K; i++) {{
        sum += A[row * K + i] * B[i * N + col];
    }}
    C[row * N + col] = sum;
}}
"""
    )


def comparison_shader(metal_type: str) -> str:
    """Return MSL source with 6 comparison kernels parameterised on *metal_type*.

    Input buffers use *metal_type*; output is always ``int`` (0/1 for bool).
    """
    return (
        _MSL_HEADER
        + f"""
kernel void lt_op(device {metal_type} *a [[buffer(0)]],
                  device {metal_type} *b [[buffer(1)]],
                  device int *out [[buffer(2)]],
                  uint id [[thread_position_in_grid]]) {{
    out[id] = (a[id] < b[id]) ? 1 : 0;
}}

kernel void le_op(device {metal_type} *a [[buffer(0)]],
                  device {metal_type} *b [[buffer(1)]],
                  device int *out [[buffer(2)]],
                  uint id [[thread_position_in_grid]]) {{
    out[id] = (a[id] <= b[id]) ? 1 : 0;
}}

kernel void gt_op(device {metal_type} *a [[buffer(0)]],
                  device {metal_type} *b [[buffer(1)]],
                  device int *out [[buffer(2)]],
                  uint id [[thread_position_in_grid]]) {{
    out[id] = (a[id] > b[id]) ? 1 : 0;
}}

kernel void ge_op(device {metal_type} *a [[buffer(0)]],
                  device {metal_type} *b [[buffer(1)]],
                  device int *out [[buffer(2)]],
                  uint id [[thread_position_in_grid]]) {{
    out[id] = (a[id] >= b[id]) ? 1 : 0;
}}

kernel void eq_op(device {metal_type} *a [[buffer(0)]],
                  device {metal_type} *b [[buffer(1)]],
                  device int *out [[buffer(2)]],
                  uint id [[thread_position_in_grid]]) {{
    out[id] = (a[id] == b[id]) ? 1 : 0;
}}

kernel void ne_op(device {metal_type} *a [[buffer(0)]],
                  device {metal_type} *b [[buffer(1)]],
                  device int *out [[buffer(2)]],
                  uint id [[thread_position_in_grid]]) {{
    out[id] = (a[id] != b[id]) ? 1 : 0;
}}
"""
    )


def boolean_shader() -> str:
    """Return MSL source with 3 boolean logic kernels operating on ``int`` buffers."""
    return (
        _MSL_HEADER
        + """
kernel void and_op(device int *a [[buffer(0)]],
                   device int *b [[buffer(1)]],
                   device int *out [[buffer(2)]],
                   uint id [[thread_position_in_grid]]) {
    out[id] = a[id] & b[id];
}

kernel void or_op(device int *a [[buffer(0)]],
                  device int *b [[buffer(1)]],
                  device int *out [[buffer(2)]],
                  uint id [[thread_position_in_grid]]) {
    out[id] = a[id] | b[id];
}

kernel void not_op(device int *a [[buffer(0)]],
                   device int *out [[buffer(1)]],
                   uint id [[thread_position_in_grid]]) {
    out[id] = a[id] ? 0 : 1;
}

kernel void xor_op(device int *a [[buffer(0)]],
                   device int *b [[buffer(1)]],
                   device int *out [[buffer(2)]],
                   uint id [[thread_position_in_grid]]) {
    out[id] = (a[id] != 0) != (b[id] != 0) ? 1 : 0;
}

kernel void lshift_op(device int *a [[buffer(0)]],
                      device int *b [[buffer(1)]],
                      device int *out [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    out[id] = a[id] << b[id];
}

kernel void rshift_op(device int *a [[buffer(0)]],
                      device int *b [[buffer(1)]],
                      device int *out [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    out[id] = a[id] >> b[id];
}

kernel void bit_invert_op(device int *a [[buffer(0)]],
                          device int *out [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    out[id] = ~a[id];
}

kernel void bit_and_op(device int *a [[buffer(0)]],
                       device int *b [[buffer(1)]],
                       device int *out [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    out[id] = a[id] & b[id];
}

kernel void bit_or_op(device int *a [[buffer(0)]],
                      device int *b [[buffer(1)]],
                      device int *out [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    out[id] = a[id] | b[id];
}

kernel void bit_xor_op(device int *a [[buffer(0)]],
                       device int *b [[buffer(1)]],
                       device int *out [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    out[id] = a[id] ^ b[id];
}
"""
    )


def where_shader(metal_type: str) -> str:
    """Return MSL source with a ternary where kernel.

    Condition is ``int`` (0/1); data buffers use *metal_type*.
    """
    return (
        _MSL_HEADER
        + f"""
kernel void where_op(device int *cond [[buffer(0)]],
                     device {metal_type} *x [[buffer(1)]],
                     device {metal_type} *y [[buffer(2)]],
                     device {metal_type} *out [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {{
    out[id] = cond[id] ? x[id] : y[id];
}}
"""
    )


def clip_shader(metal_type: str) -> str:
    """Return MSL source with a clip kernel (3 inputs: data, lo, hi)."""
    return (
        _MSL_HEADER
        + f"""
kernel void clip_op(device {metal_type} *a [[buffer(0)]],
                    device {metal_type} *lo [[buffer(1)]],
                    device {metal_type} *hi [[buffer(2)]],
                    device {metal_type} *out [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {{
    {metal_type} val = a[id];
    out[id] = (val < lo[id]) ? lo[id] : ((val > hi[id]) ? hi[id] : val);
}}
"""
    )


def predicate_shader(metal_type: str) -> str:
    """Return MSL source with unary predicate kernels (typed input → int output)."""
    return (
        _MSL_HEADER
        + f"""
kernel void isnan_op(device {metal_type} *a [[buffer(0)]],
                     device int *out [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {{
    out[id] = isnan(static_cast<float>(a[id])) ? 1 : 0;
}}

kernel void isinf_op(device {metal_type} *a [[buffer(0)]],
                     device int *out [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {{
    out[id] = isinf(static_cast<float>(a[id])) ? 1 : 0;
}}

kernel void isfinite_op(device {metal_type} *a [[buffer(0)]],
                        device int *out [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {{
    out[id] = isfinite(static_cast<float>(a[id])) ? 1 : 0;
}}

kernel void signbit_op(device {metal_type} *a [[buffer(0)]],
                       device int *out [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {{
    out[id] = signbit(static_cast<float>(a[id])) ? 1 : 0;
}}
"""
    )


def axis_reduction_shader(metal_type: str) -> str:
    """Return MSL source with axis-aware reduction kernels.

    These operate on 2-D (outer, inner) layout.  The caller transposes and
    reshapes the N-D array so the reduction axis is last (inner dimension).
    Each thread reduces one row of ``inner`` elements.
    """
    return (
        _MSL_HEADER
        + f"""
kernel void reduce_sum_axis(device {metal_type} *input [[buffer(0)]],
                            device {metal_type} *output [[buffer(1)]],
                            device uint *dims [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {{
    uint outer = dims[0], inner = dims[1];
    if (id >= outer) return;
    {metal_type} acc = 0;
    uint base = id * inner;
    for (uint j = 0; j < inner; j++) acc += input[base + j];
    output[id] = acc;
}}

kernel void reduce_max_axis(device {metal_type} *input [[buffer(0)]],
                            device {metal_type} *output [[buffer(1)]],
                            device uint *dims [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {{
    uint outer = dims[0], inner = dims[1];
    if (id >= outer) return;
    uint base = id * inner;
    {metal_type} mx = input[base];
    for (uint j = 1; j < inner; j++) {{
        {metal_type} v = input[base + j];
        if (v > mx) mx = v;
    }}
    output[id] = mx;
}}

kernel void reduce_min_axis(device {metal_type} *input [[buffer(0)]],
                            device {metal_type} *output [[buffer(1)]],
                            device uint *dims [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {{
    uint outer = dims[0], inner = dims[1];
    if (id >= outer) return;
    uint base = id * inner;
    {metal_type} mn = input[base];
    for (uint j = 1; j < inner; j++) {{
        {metal_type} v = input[base + j];
        if (v < mn) mn = v;
    }}
    output[id] = mn;
}}

kernel void reduce_prod_axis(device {metal_type} *input [[buffer(0)]],
                             device {metal_type} *output [[buffer(1)]],
                             device uint *dims [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {{
    uint outer = dims[0], inner = dims[1];
    if (id >= outer) return;
    {metal_type} acc = 1;
    uint base = id * inner;
    for (uint j = 0; j < inner; j++) acc *= input[base + j];
    output[id] = acc;
}}

kernel void argmax_axis(device {metal_type} *input [[buffer(0)]],
                        device int *output [[buffer(1)]],
                        device uint *dims [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {{
    uint outer = dims[0], inner = dims[1];
    if (id >= outer) return;
    uint base = id * inner;
    int best = 0;
    {metal_type} best_val = input[base];
    for (uint j = 1; j < inner; j++) {{
        {metal_type} v = input[base + j];
        if (v > best_val) {{ best_val = v; best = (int)j; }}
    }}
    output[id] = best;
}}

kernel void argmin_axis(device {metal_type} *input [[buffer(0)]],
                        device int *output [[buffer(1)]],
                        device uint *dims [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {{
    uint outer = dims[0], inner = dims[1];
    if (id >= outer) return;
    uint base = id * inner;
    int best = 0;
    {metal_type} best_val = input[base];
    for (uint j = 1; j < inner; j++) {{
        {metal_type} v = input[base + j];
        if (v < best_val) {{ best_val = v; best = (int)j; }}
    }}
    output[id] = best;
}}

kernel void prefix_sum(device {metal_type} *input [[buffer(0)]],
                       device {metal_type} *output [[buffer(1)]],
                       device uint *dims [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {{
    uint outer = dims[0], inner = dims[1];
    if (id >= outer) return;
    uint base = id * inner;
    output[base] = input[base];
    for (uint j = 1; j < inner; j++)
        output[base + j] = output[base + j - 1] + input[base + j];
}}

kernel void prefix_prod(device {metal_type} *input [[buffer(0)]],
                        device {metal_type} *output [[buffer(1)]],
                        device uint *dims [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {{
    uint outer = dims[0], inner = dims[1];
    if (id >= outer) return;
    uint base = id * inner;
    output[base] = input[base];
    for (uint j = 1; j < inner; j++)
        output[base + j] = output[base + j - 1] * input[base + j];
}}
"""
    )
