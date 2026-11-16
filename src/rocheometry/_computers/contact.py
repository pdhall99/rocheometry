"""Shared bounds for detached surfaces and watershed envelope parts."""

from math import sqrt

from rocheometry import Fitted
from rocheometry._computers.checks import require_depth
from rocheometry._computers.critical_point import CriticalPointComputer
from rocheometry._computers.watershed import WatershedComputer
from rocheometry._roche.field import RocheField
from rocheometry._roche.opening import _admissibility_gap, _coverage_gap
from rocheometry._roche.potential import CriticalPoint


def default_outer() -> CriticalPointComputer:
    from rocheometry._computers.defaults import default_outer_computer

    return default_outer_computer()


def default_watershed() -> WatershedComputer:

    return Fitted()._computers().watershed


def admissibility_depth(
    field: RocheField, l1: CriticalPoint, outer: CriticalPointComputer
) -> float:
    """Return the exact first-opening gap rounded inward, a per-field constant."""
    return _admissibility_gap(
        field,
        l1.position[0],
        outer(field).position[0],
        outer(field.reciprocal()).position[0],
    )


def coverage_depth(
    field: RocheField, l1: CriticalPoint, outer: CriticalPointComputer
) -> float:
    """Field-only nearest-rounded depth used by the gravity-line construction."""
    return _coverage_gap(
        field,
        l1.position[0],
        outer(field).position[0],
        outer(field.reciprocal()).position[0],
    )


def require_part_depth(
    field: RocheField,
    depth: float,
    l1: CriticalPoint,
    outer: CriticalPointComputer,
    *,
    allow_l2: bool = True,
) -> float:
    """Bound contact parts by depth, without snapping or an interior clamp."""
    depth = require_depth(depth)
    if depth > 0:
        limit = admissibility_depth(field, l1, outer)
        if depth > limit or (depth == limit and not allow_l2):
            raise ValueError(f"depth {depth!r} must be below the L2 gap ({limit!r})")
    return depth


def _geometry_depth(depth: float, l1: CriticalPoint) -> float:
    """Use the cone only when both axial neck offsets round onto L1.

    This is a representation test, not a level band. The surface retains its
    positive depth and contact regime; only its unrepresentable geometry
    uses the critical limit.
    """
    if depth > 0:
        radius = sqrt(depth) * sqrt(2 / l1.hessian[0][0])
        x = l1.position[0]
        if x + radius == x and x - radius == x:
            return 0.0
    return depth
