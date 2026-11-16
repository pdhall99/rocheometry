"""Saddle geometry shared by surface sampling and section construction."""

from math import sqrt

from rocheometry._roche.field import RocheField
from rocheometry._roche.potential import CriticalPoint


def _neck_width(
    field: RocheField,
    depth: float,
    saddle: CriticalPoint,
    l1: CriticalPoint | None = None,
) -> float:
    """Return the saddle's axial neck width as an angle at the component."""
    x = saddle.position[0]
    offset = depth if l1 is None else field._depth_offset(depth, l1.position[0], x)
    return sqrt(2 * abs(offset) / field.hessian(x)[0][0]) / abs(x)
