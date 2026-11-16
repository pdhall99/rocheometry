"""Shared component-enclosing ray brackets and scaled root coordinates."""

from __future__ import annotations

from collections.abc import Callable
from math import atan2, hypot
from sys import float_info
from typing import TYPE_CHECKING, Protocol

from rocheometry._computers.contact import _geometry_depth, require_part_depth
from rocheometry._computers.neck import (
    _neck_radius,
    _neck_surface,
    _unresolved_neck_ray,
)
from rocheometry._computers.neck_ray import _refined_neck_ray
from rocheometry._computers.surface_radius.anchor import _residual_anchor
from rocheometry._computers.watershed import WatershedComputer
from rocheometry._geometry import Point3D
from rocheometry._numerics.roots import DEFAULT_ROOT_RELATIVE_TOLERANCE
from rocheometry._roche.field import RocheField
from rocheometry._roche.potential import _potential_residual_scale

# Insensitive control: radius inner bracket seed factor.
RADIUS_INNER_BRACKET_SEED_FACTOR = 1e-12

if TYPE_CHECKING:
    from rocheometry._computers.critical_point import CriticalPointComputer

# Work guard or cache capacity: radius inverse coordinate tolerance.
RADIUS_INVERSE_COORDINATE_TOLERANCE = 16.0 * DEFAULT_ROOT_RELATIVE_TOLERANCE


class _RootSolver(Protocol):
    def __call__(
        self,
        function: Callable[[float], float],
        left: float,
        right: float,
        *,
        absolute_tolerance: float,
        relative_tolerance: float,
    ) -> float: ...


def _enclosing_ray_bracket(
    residual: Callable[[float], float],
    inner: float,
    outer: float,
    tolerance: float,
) -> tuple[float, float]:
    """Contract a geometric enclosing bracket without outward potential scans.

    The caller supplies an inside lower bound and a geometric outer bound on
    the closed component. Positive centrifugal-exterior values are never
    used to search for an upper bound. Equal returned bounds denote a root.
    """
    value = residual(outer)
    if value > tolerance:
        raise RuntimeError("L1 distance does not bracket this closed equipotential")
    if value >= 0.0:
        return outer, outer
    while 0.5 * outer > inner:
        candidate = 0.5 * outer
        value = residual(candidate)
        if value == 0.0:
            return candidate, candidate
        if value > 0.0:
            inner = candidate
            break
        outer = candidate
    return inner, outer


def _surface_radius(
    field: RocheField,
    depth: float,
    direction: Point3D,
    l1_computer: CriticalPointComputer,
    relative_tolerance: float,
    max_inner_contractions: int,
    root_solver: _RootSolver,
    *,
    inverse_radius: bool = False,
    outer_computer: CriticalPointComputer,
    watershed_computer: WatershedComputer,
    exact_anchor: bool = True,
) -> float:
    """Bracket the first radial intersection and solve at its own scale."""
    l1 = l1_computer(field)
    depth = require_part_depth(field, depth, l1, outer_computer)
    depth = _geometry_depth(depth, l1)
    scale = _potential_residual_scale(l1.potential.value) + abs(depth)
    direction_x, direction_y, direction_z = direction
    component_mass_fraction = field.component_mass
    inner_radius = min(
        l1.position[0] * RADIUS_INNER_BRACKET_SEED_FACTOR,
        0.25 * component_mass_fraction / scale,
    )
    inner_radius = max(inner_radius, 16.0 * float_info.min)

    canonical = type(field).potential is RocheField.potential
    reference_x = None
    # Protect contact and cusp residuals from cancellation. The surface
    # policy controls factoring on the other rays.
    factor = (
        exact_anchor or depth > 0 or (direction_x > 0.99 and abs(depth) < 0.001 * scale)
    )
    if canonical and factor:
        anchor, _ = _residual_anchor(field, depth, direction_x, l1, outer_computer)
        reference_x = anchor.position[0]
        residual = field._ray_residual(
            direction, depth, reference_x, l1_x=l1.position[0]
        )
    elif canonical:
        residual = field._ray_residual(direction, depth, l1_x=l1.position[0])
    else:

        def residual(radius: float) -> float:
            return (
                field.potential_difference(
                    (radius * direction_x, radius * direction_y, radius * direction_z),
                    l1.position,
                )
                + depth
            )

    for _ in range(max_inner_contractions):
        if residual(inner_radius) > 0.0:
            break
        inner_radius *= 0.1
    else:
        raise RuntimeError("could not bracket equipotential near the component centre")

    outer_radius = l1.position[0]
    if depth > 0:
        if direction_x > 0.0 and direction_y == 0.0 and direction_z == 0.0:
            return l1.position[0]
        if canonical and _unresolved_neck_ray(depth, direction, l1.position[0], field):
            return _neck_radius(field, l1.position[0], depth, direction)
        psi = atan2(direction_z, direction_y)
        rim = watershed_computer.rim(field, depth, psi)
        theta_rim = atan2(hypot(rim[1], rim[2]), rim[0])
        if atan2(hypot(direction_y, direction_z), direction_x) < theta_rim:
            return hypot(*watershed_computer.sheet_point_on_ray(field, direction))
        outer_radius = abs(outer_computer(field).position[0])
        if direction_x > 0:
            outer_radius = min(outer_radius, rim[0] / direction_x)
    inner_radius, outer_radius = _enclosing_ray_bracket(
        residual,
        inner_radius,
        outer_radius,
        relative_tolerance * scale,
    )
    if inner_radius == outer_radius:
        return outer_radius

    # Scaling keeps the root O(1), even for tiny surfaces.
    # Scaling by L1 alone would retain a unit-sized relative error floor.
    if depth > 0:
        # Reuse the same factored function at the scaled bracket endpoints.
        # Rebinding rounded ray coefficients can otherwise reverse a residual
        # within an ulp of the rim and lose the bracket.
        def scaled_residual(value: float) -> float:
            return residual(
                outer_radius / value if inverse_radius else outer_radius * value
            )
    elif type(field).potential is RocheField.potential:
        scaled_residual = field._ray_residual(
            (
                outer_radius * direction_x,
                outer_radius * direction_y,
                outer_radius * direction_z,
            ),
            depth,
            reference_x,
            inverse_radius=inverse_radius,
            l1_x=l1.position[0],
        )
    else:

        def scaled_residual(value: float) -> float:
            return residual(
                outer_radius / value if inverse_radius else outer_radius * value
            )

    left, right = (
        (1.0, outer_radius / inner_radius)
        if inverse_radius
        else (inner_radius / outer_radius, 1.0)
    )
    coordinate = root_solver(
        scaled_residual,
        left,
        right,
        absolute_tolerance=RADIUS_INVERSE_COORDINATE_TOLERANCE
        if inverse_radius
        else DEFAULT_ROOT_RELATIVE_TOLERANCE,
        relative_tolerance=0.0,
    )
    radius = outer_radius / coordinate if inverse_radius else outer_radius * coordinate
    if (
        canonical
        and direction_x > 0
        and _neck_surface(field, l1.position[0], depth) is not None
    ):
        # The structural thin-neck test also covers rays outside the local
        # Taylor certificate. Refine them with untruncated DD differences.
        radius, _ = _refined_neck_ray(field, l1.position[0], depth, direction, radius)
    return radius
