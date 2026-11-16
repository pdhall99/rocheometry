"""Exact and rounded outer-opening gaps for a canonical field."""

from fractions import Fraction
from functools import lru_cache
from math import nextafter

from rocheometry._roche.field import RocheField


@lru_cache(maxsize=4096)
def _exact_opening_gap(
    field: RocheField, l1_x: float, outer_x: float, opposite_outer_x: float
) -> Fraction:
    opening = max(
        field._axis_potential_exact(outer_x),
        field.reciprocal()._axis_potential_exact(opposite_outer_x),
    )
    return field._axis_potential_exact(l1_x) - opening


def _round_admissibility_gap(exact: Fraction) -> float:
    rounded = float(exact)
    return nextafter(rounded, 0.0) if rounded > exact else rounded


@lru_cache(maxsize=4096)
def _admissibility_gap(
    field: RocheField, l1_x: float, outer_x: float, opposite_outer_x: float
) -> float:
    """Largest representable closed-side depth: bounds, F and limiting fill."""
    return _round_admissibility_gap(
        _exact_opening_gap(field, l1_x, outer_x, opposite_outer_x)
    )


@lru_cache(maxsize=4096)
def _coverage_gap(
    field: RocheField, l1_x: float, outer_x: float, opposite_outer_x: float
) -> float:
    """Nearest-rounded field scale for watershed coverage and family labels.

    This approximation scale must not depend on how a public bound is rounded.
    """
    return float(_exact_opening_gap(field, l1_x, outer_x, opposite_outer_x))
