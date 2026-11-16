"""Dependency-free finite-interval mesh generation."""

import reprlib
from math import cos, pi
from typing import Literal, TypeAlias, get_args

from rocheometry._validation import (
    _normalize_finite_float,
    require_choice,
    require_instance,
    require_int,
)

IntervalMeshKind: TypeAlias = Literal["uniform", "chebyshev-lobatto"]
"""The supported finite-interval node-spacing rules."""
INTERVAL_MESH_KINDS: tuple[IntervalMeshKind, ...] = get_args(IntervalMeshKind)
"""The accepted interval-mesh identifiers derived from `IntervalMeshKind`."""


def interval_nodes(
    kind: IntervalMeshKind,
    minimum: float,
    maximum: float,
    count: int,
) -> tuple[float, ...]:
    """Return ordered nodes on one finite closed interval.

    Args:
        kind: Spacing rule, either `uniform` or `chebyshev-lobatto`.
        minimum: Finite lower endpoint, strictly below `maximum`.
        maximum: Finite upper endpoint.
        count: Node count including endpoints; integer of at least `2`.

    Returns:
        Increasing nodes with both supplied endpoints represented exactly.
    """
    require_instance(kind, str, name="kind")
    require_choice(kind, INTERVAL_MESH_KINDS, name="kind")
    minimum = _normalize_finite_float(minimum, name="minimum")
    maximum = _normalize_finite_float(maximum, name="maximum")
    if not minimum < maximum:
        raise ValueError(
            "bounds: expected minimum < maximum, "
            f"got ({reprlib.repr(minimum)}, {reprlib.repr(maximum)})"
        )
    count = require_int(count, name="count")
    if count < 2:
        raise ValueError(
            "count: expected an integer greater than or equal to 2, "
            f"got {reprlib.repr(count)}"
        )

    if kind == "uniform":
        step = (maximum - minimum) / (count - 1)
        interior = tuple(minimum + step * index for index in range(1, count - 1))
    else:
        midpoint = 0.5 * (minimum + maximum)
        half_width = 0.5 * (maximum - minimum)
        angle_step = -pi / (count - 1)
        # Descending angles make the cosine mesh ascend from minimum to maximum.
        # Form each angle from the fixed endpoint and step to avoid drift.
        interior = tuple(
            midpoint + half_width * cos(pi + angle_step * index)
            for index in range(1, count - 1)
        )

    return (minimum, *interior, maximum)
