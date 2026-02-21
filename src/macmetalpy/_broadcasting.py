"""Shape broadcasting utilities following NumPy broadcasting rules."""

from __future__ import annotations

__all__ = ["broadcast_shapes", "broadcast_strides", "needs_broadcast"]


def broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """Compute the broadcast shape of one or more shapes.

    Follows the standard NumPy broadcasting rules:
    1. Shapes are right-aligned.
    2. Dimensions of size 1 are stretched to match the other.
    3. Dimensions must be equal or one of them must be 1.

    Raises
    ------
    ValueError
        If shapes are not broadcast-compatible.
    """
    if not shapes:
        return ()

    ndim = max(len(s) for s in shapes)
    result: list[int] = []

    for axis in range(-1, -ndim - 1, -1):
        dims: list[int] = []
        for s in shapes:
            idx = len(s) + axis  # maps negative axis to positive index
            if idx >= 0:
                dims.append(s[idx])
            else:
                dims.append(1)

        # All non-1 values must agree
        non_one = {d for d in dims if d != 1}
        if len(non_one) > 1:
            raise ValueError(
                f"Shape mismatch: cannot broadcast shapes {shapes}"
            )
        result.append(max(dims))

    result.reverse()
    return tuple(result)


def broadcast_strides(
    shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    strides: tuple[int, ...],
) -> tuple[int, ...]:
    """Compute element-strides for *shape* broadcast into *target_shape*.

    Dimensions that are stretched (size 1 -> larger) get stride 0.
    Leading dimensions added by broadcasting also get stride 0.
    """
    ndim_diff = len(target_shape) - len(shape)
    new_strides: list[int] = [0] * ndim_diff

    for orig_size, target_size, stride in zip(shape, target_shape[ndim_diff:], strides):
        if orig_size == target_size:
            new_strides.append(stride)
        elif orig_size == 1:
            new_strides.append(0)
        else:
            raise ValueError(
                f"Cannot broadcast dimension {orig_size} to {target_size}"
            )

    return tuple(new_strides)


def needs_broadcast(shape1: tuple[int, ...], shape2: tuple[int, ...]) -> bool:
    """Return True if two shapes require broadcasting to operate together."""
    return shape1 != shape2
