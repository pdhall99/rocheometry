"""Checks composed into computer graphs before results are shared."""

from math import isfinite
from typing import cast

from rocheometry._roche.potential import (
    Potential,
    PotentialConvention,
    require_potential,
)
from rocheometry._validation import (
    _normalize_finite_float,
    _normalize_finite_positive_float,
)


def require_positive(value: object) -> float:
    """Require a finite, positive scalar computer result."""
    if type(value) is float and isfinite(value) and value > 0.0:
        return value
    return _normalize_finite_positive_float(cast(float, value), name="computer result")


def require_jacobi_potential(value: object, *, name: str = "potential") -> Potential:
    """Require a typed canonical Jacobi potential."""
    potential = require_potential(value, name=name)
    if potential.convention is not PotentialConvention.JACOBI:
        raise ValueError(
            f"{name}: expected a Jacobi potential, got {potential.convention.value}; "
            "a Kopal value names a binary's component, so select it through "
            "BinaryGeometry.component(i) or convert it with convert_potential()"
        )
    return potential


def require_depth(value: object) -> float:
    """Require a finite depth, without a critical snapping band."""
    return _normalize_finite_float(cast(float, value), name="depth")


def require_closed_depth(depth: float) -> float:
    """Require a detached or exactly critical depth."""
    depth = require_depth(depth)
    if depth > 0:
        raise ValueError("closed-surface computers need depth <= 0 (L1)")
    return depth
