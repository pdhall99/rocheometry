"""Shared canonical residual anchor selection; no change to ray arithmetic."""

from rocheometry._computers.critical_point import CriticalPointComputer
from rocheometry._roche.field import RocheField
from rocheometry._roche.potential import CriticalPoint


def _residual_anchor(
    field: RocheField,
    depth: float,
    direction_x: float,
    l1: CriticalPoint,
    outer_computer: CriticalPointComputer,
) -> tuple[CriticalPoint, float]:
    """Return the radius solver's anchor and U(anchor)-requested_level.

    Canonical callers use their own outer saddle only on backward contact rays
    when its depth offset is smaller than abs(depth). All other rays use L1.
    Non-canonical callers retain their original residual rather than using this
    canonical-field rule. No direction normalization is needed for this sign.
    """
    if depth > 0:
        outer = outer_computer(field)
        if direction_x < 0:
            offset = field._depth_offset(depth, l1.position[0], outer.position[0])
            if abs(offset) < abs(depth):
                return outer, offset
    return l1, depth
