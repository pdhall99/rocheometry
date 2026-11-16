"""Volumes by Cartesian x-slices, independently of rays and of the watershed.

The library integrates volumes over directions from the component centre:
radii along rays, a polar rule clustered towards the necks. This module shares
neither. It slices the body perpendicular to the line of centres and finds
each slice's extent by one-dimensional roots of the potential difference
from L1, in the component's own frame.

It has two forms:

- `CartesianSliceSurfaceVolumeComputer` plugs into a library computer graph
  in place of the sampled volume, so tests can run selection and inversion on
  an independent volume. It integrates one panel, for detached and critical
  levels.
- `lobe_volume`, `envelope_volume` and their estimates are the study truths.
  Panels break at both centres and at the L1 point, and are graded towards L1
  on the lighter mass's Hill scale, so the rule never straddles a singularity
  and resolves the neck at every mass ratio. No dividing surface is used: a
  slice integral of a contact level measures both components together. Each
  value is evaluated at two orders and the difference reported as its own
  error, as a truth must be.

Shared with the library: the field's factored potential difference, the
location of L1 and of the outer critical points (panel breaks and brackets
only), and the Gauss-Legendre nodes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import fsum, hypot, isfinite, pi, sqrt
from sys import float_info
from typing import TYPE_CHECKING, Any

from rocheometry import Potential, Reference
from rocheometry._computers.checks import require_closed_depth, require_positive
from rocheometry._levels import _levels
from rocheometry._numerics import bisect_root
from rocheometry._numerics.quadrature import gauss_legendre
from rocheometry._numerics.roots import brent_root, brent_root_adjacent
from rocheometry._roche import RocheField
from rocheometry._roche.potential import _potential_residual_scale
from rocheometry._validation import require_finite, require_quadrature_order

if TYPE_CHECKING:
    from rocheometry._computers.critical_point import CriticalPointComputer

#: Axial and transverse orders of the two evaluations.
ORDERS = ((96, 32), (160, 48))
#: Floor on a value's relative error: eight units in the last place.
ROUNDING = 8 * 2.0**-52
#: Roundoff allowance on a slice root's residual, in potential-scale units.
CARTESIAN_INTEGRAL_RESIDUAL_TOLERANCE = 512.0 * float_info.epsilon


@dataclass(frozen=True, slots=True, kw_only=True)
class CartesianSliceSurfaceVolumeComputer:
    """Integrate critical or detached surface volume using Cartesian x-slices.

    L1 caps the root searches on the component-enclosing branch. The supplied
    potential selects the surface; slice roots use the actual supplied field.
    No directional-radius computer or angular integration is used.

    Attributes:
        l1_computer: L1 strategy bounding the component-enclosing branch.
        slice_order: Gauss-Legendre order along x; integer at least `4`.
        cross_section_order: Quadrature order within each transverse slice; integer at
            least `4`.
    """

    l1_computer: CriticalPointComputer
    slice_order: int = 160
    cross_section_order: int = 32

    def __post_init__(self) -> None:
        if not callable(self.l1_computer):
            raise TypeError("l1_computer: expected a callable computer")
        require_quadrature_order(self.slice_order, name="slice_order")
        require_quadrature_order(self.cross_section_order, name="cross_section_order")

    def __call__(self, field: RocheField, depth: float, /) -> float:
        """Return the volume of one component-enclosing equipotential.

        Args:
            field: Normalized component-local field.
            depth: Canonical Jacobi level selecting a closed component-enclosing
                surface, at or above the L1 level.

        Returns:
            Enclosed volume in separation units cubed.

        Raises:
            ValueError: The depth is positive, so the level is on the contact side of L1.
            RuntimeError: The closed branch cannot be bracketed, a slice extent
                is invalid, or a slice root solve fails to converge.
        """
        l1 = self.l1_computer(field)
        depth = require_closed_depth(depth)
        l1_distance = l1.position[0]
        scale = _potential_residual_scale(l1.potential.value) + abs(depth)
        x_back, x_point = _surface_x_bounds(field, l1_distance, depth)
        # The inward x intersection bounds every radius of the closed Roche
        # surface, tightening the critical L1 sphere for detached surfaces.
        point_squared = x_point * x_point

        def y_max(x: float) -> float:
            y_outer = sqrt(max(0.0, point_squared - x * x))

            def orbital_residual(y: float) -> float:
                return field.potential_difference((x, y, 0.0), l1.position) + depth

            # Odd slice orders can sample the singular component centre.
            return bounded_surface_root(
                orbital_residual,
                y_outer,
                scale=scale,
                inner=y_outer * 1.0e-12 if x == 0.0 else 0.0,
            )

        def z_max(x: float, y: float) -> float:
            z_outer = sqrt(max(0.0, point_squared - x * x - y * y))

            def vertical_residual(z: float) -> float:
                return field.potential_difference((x, y, z), l1.position) + depth

            return bounded_surface_root(
                vertical_residual,
                z_outer,
                scale=scale,
                inner=z_outer * 1.0e-12 if x == 0.0 and y == 0.0 else 0.0,
            )

        return require_positive(
            integrate_cross_section_volume(
                x_back,
                x_point,
                y_max,
                z_max,
                slice_order=self.slice_order,
                cross_section_order=self.cross_section_order,
            )
        )


def _surface_x_bounds(
    field: RocheField, l1_distance: float, depth: float
) -> tuple[float, float]:
    component_mass = field.component_mass
    anchor = (
        float(field._axis_potential_exact(l1_distance))
        if type(field).potential is RocheField.potential
        else field.potential(l1_distance)
    )
    scale = abs(anchor) + abs(depth)
    inner = min(
        l1_distance * 1.0e-12,
        0.25 * component_mass / _potential_residual_scale(scale),
    )
    inner = max(inner, 16.0 * float_info.min)

    def forward(radius: float) -> float:
        return (
            field.potential_difference((radius, 0.0, 0.0), (l1_distance, 0.0, 0.0))
            + depth
        )

    def backward(radius: float) -> float:
        return (
            field.potential_difference((-radius, 0.0, 0.0), (l1_distance, 0.0, 0.0))
            + depth
        )

    for _ in range(32):
        if forward(inner) > 0.0 and backward(inner) > 0.0:
            break
        inner *= 0.1
    else:
        raise RuntimeError("could not bracket equipotential near the component centre")

    x_back = -bounded_surface_root(backward, l1_distance, scale=scale, inner=inner)
    x_point = bounded_surface_root(forward, l1_distance, scale=scale, inner=inner)
    return x_back, x_point


def bounded_surface_root(
    function: Callable[[float], float],
    outer: float,
    *,
    scale: float,
    inner: float = 0.0,
) -> float:
    """Return the first surface root in a positive, component-enclosing bracket.

    A nonzero inner bound avoids the field's singularity for axial roots.
    Contract the bracket before scaled bisection to retain relative accuracy
    for small surfaces and narrow cross-sections.

    Args:
        function: Finite potential residual, positive inside the surface and
            non-positive at the outer bound within roundoff tolerance.
        outer: Non-negative upper radius bound in separation units.
        scale: Canonical potential level setting the residual roundoff scale.
        inner: Non-negative lower radius bound; a positive value avoids axial
            point-mass singularities. If `outer <= inner`, return `inner`.

    Raises:
        RuntimeError: The supplied slice bounds do not enclose the closed
            branch, or the bisection solve fails to converge.
    """
    if outer <= inner:
        return inner

    def residual(radius: float) -> float:
        return require_finite(function(radius), name="slice surface residual")

    inner_residual = residual(inner)
    outer_residual = residual(outer)
    tolerance = CARTESIAN_INTEGRAL_RESIDUAL_TOLERANCE * _potential_residual_scale(scale)

    if inner_residual < -tolerance:
        raise RuntimeError("slice origin is outside the component-enclosing branch")
    if inner_residual <= 0.0:
        return inner
    if outer_residual > tolerance:
        raise RuntimeError("outer bound does not bracket the slice surface root")
    if outer_residual >= 0.0:
        return outer

    while 0.5 * outer > inner:
        candidate = 0.5 * outer
        value = residual(candidate)
        if value == 0.0:
            return candidate
        if value > 0.0:
            inner = candidate
            break
        outer = candidate
    coordinate = bisect_root(
        lambda value: residual(outer * value),
        inner / outer,
        1.0,
        absolute_tolerance=0.0,
    )
    return outer * coordinate


def integrate_cross_section_volume(
    x_min: float,
    x_max: float,
    y_max: Callable[[float], float],
    z_max: Callable[[float, float], float],
    *,
    slice_order: int,
    cross_section_order: int,
) -> float:
    """Integrate a y/z-reflection-symmetric body from Cartesian x-slices.

    Args:
        x_min: Finite lower axial bound, strictly below `x_max`.
        x_max: Finite upper axial bound in the same length units.
        y_max: Function giving the finite non-negative y extent of a slice.
        z_max: Function giving its finite non-negative z extent at `(x, y)`.
        slice_order: Axial Gauss-Legendre order; integer at least `4`.
        cross_section_order: Transverse order; integer at least `4`.

    Returns:
        Volume in the input length units cubed, including all four reflected
        y/z quadrants.

    Raises:
        RuntimeError: An extent function returns an invalid value.
    """
    if not all(isfinite(value) for value in (x_min, x_max)) or x_min >= x_max:
        raise ValueError("x bounds must be finite and increasing")

    slice_order = require_quadrature_order(slice_order, name="slice_order")
    cross_section_order = require_quadrature_order(
        cross_section_order, name="cross_section_order"
    )
    x_nodes, x_weights = gauss_legendre(slice_order)
    u_nodes, u_weights = gauss_legendre(cross_section_order)
    x_midpoint = 0.5 * (x_min + x_max)
    x_half_width = 0.5 * (x_max - x_min)
    volume = 0.0

    for x_node, x_weight in zip(x_nodes, x_weights, strict=True):
        x = x_midpoint + x_half_width * x_node
        x_weight = x_half_width * x_weight
        ymax = y_max(x)
        if not isfinite(ymax) or ymax < 0.0:
            raise RuntimeError("cross-section y extent must be finite and non-negative")

        quadrant_area = 0.0
        for u_node, u_weight in zip(u_nodes, u_weights, strict=True):
            u = 0.5 * (u_node + 1.0)
            u_weight = 0.5 * u_weight
            y = ymax * (1.0 - u * u)
            zmax = z_max(x, y)
            if not isfinite(zmax) or zmax < 0.0:
                raise RuntimeError(
                    "cross-section z extent must be finite and non-negative"
                )
            quadrant_area += u_weight * 2.0 * ymax * u * zmax

        volume += x_weight * 4.0 * quadrant_area

    return volume


def _graph() -> Any:
    return Reference()._computers()


def _residual(field: RocheField, depth: float) -> Any:
    """U(p) - u for the level at ``depth`` below L1, positive inside.

    The difference from the L1 potential is evaluated in factored form, so the
    residual keeps its relative accuracy when one mass is tiny; an absolute
    potential would lose about as many digits as the mass ratio has decades.
    """
    l1 = (_graph().l1(field).position[0], 0.0, 0.0)
    return lambda x, y=0.0, z=0.0: field.potential_difference((x, y, z), l1) + depth


def _first_root(function: Any, outer: float, inner: float = 0.0) -> float:
    """Find the first sign change of ``function`` (positive inside) beyond ``inner``.

    The bracket is halved until its midpoint is inside, so a second branch
    beyond the first root is never found; then the root is solved to adjacent
    floats.
    """
    if function(inner) <= 0.0:
        return inner
    if function(outer) > 0.0:
        raise RuntimeError("the outer bound does not bracket the surface")
    while 0.5 * outer > inner and function(0.5 * outer) <= 0.0:
        outer *= 0.5
    return brent_root_adjacent(function, inner, outer)


def _axis_extent(residual: Any, centre: float, sign: float, upper: float) -> float:
    """Distance from a centre to the level along the axis, away from the neck.

    ``upper`` is the distance to the outer critical point. At the largest
    closed level the body reaches it, to rounding: then it is the extent.
    """
    function = lambda r: residual(centre + sign * r)  # noqa: E731
    if function(upper) >= 0.0:
        return upper
    return _first_root(function, upper, upper * 1e-12)


def _integrate(
    field: RocheField, depth: float, bounds: list[float], order: tuple[int, int]
) -> float:
    residual = _residual(field, depth)

    def ymax(x: float) -> float:
        def transverse_gradient(y: float) -> float:
            return (
                1
                - field.component_mass / hypot(x, y) ** 3
                - field.companion_mass / hypot(1 - x, y) ** 3
            )

        minimum = brent_root(transverse_gradient, 1e-30, 2.0)
        return _first_root(lambda y: residual(x, y), minimum)

    def zmax(x: float, y: float) -> float:
        return _first_root(lambda z: residual(x, y, z), 2.0)

    return fsum(
        integrate_cross_section_volume(
            a, b, ymax, zmax, slice_order=order[0], cross_section_order=order[1]
        )
        for a, b in zip(bounds[:-1], bounds[1:], strict=True)
    )


def _span(field: RocheField, depth: float, *, both: bool) -> list[float]:
    """Panel breaks: the back of the body, the centres and the L1 point."""
    graph = _graph()
    residual = _residual(field, depth)
    back = _axis_extent(residual, 0.0, -1.0, abs(graph.outer(field).position[0]))
    x1 = graph.l1(field).position[0]
    # Near L1 the body varies on the lighter mass's Hill scale: 1 - x1 on the
    # component's side when the companion is light, x1 on the companion's side
    # when the component is. Panels graded geometrically towards L1 resolve it.
    left = [x1 - 4.0**k * (1.0 - x1) for k in range(6, 0, -1)]
    right = [x1 + 4.0**k * x1 for k in range(1, 7)]
    inner = [0.0, *(b for b in left if b > 0.0), x1]
    if not both:
        return [-back, *inner]
    companion = RocheField(1 / field.qi)
    front = _axis_extent(residual, 1.0, 1.0, abs(graph.outer(companion).position[0]))
    return [-back, *inner, *(b for b in right if b < 1.0), 1.0, 1.0 + front]


def envelope_volume_at_depth(
    field: RocheField, depth: float, order: int = 192
) -> float:
    """Volume of both components together at a contact level ``depth`` below L1."""
    return _integrate(field, depth, _span(field, depth, both=True), (order, 64))


def envelope_volume(field: RocheField, level: float, order: int = 192) -> float:
    """Volume of both components together at the public contact level ``level``."""
    depth = _levels(field, _graph()).depth(Potential.jacobi(level))
    return envelope_volume_at_depth(field, depth, order)


def envelope_estimate(field: RocheField, depth: float) -> dict[str, float]:
    """``envelope_volume_at_depth`` at both orders, with its own error."""
    return estimate(
        lambda order: _integrate(field, depth, _span(field, depth, both=True), order)
    )


def lobe_volume(field: RocheField, order: tuple[int, int] = ORDERS[-1]) -> float:
    """Volume of the component's Roche lobe, sliced up to the L1 point."""
    return _integrate(field, 0.0, _span(field, 0.0, both=False), order)


def estimate(function: Any) -> dict[str, float]:
    """Return a value at the higher order and its relative error.

    The error is the change from the lower order, but never below
    ``ROUNDING``: two orders can agree to the last bit while both carry the
    rounding of a sum over thousands of slices.
    """
    low, high = (function(order) for order in ORDERS)
    return dict(value=high, error=max(abs(high / low - 1), ROUNDING))


def lobe_volume_radius(qi: float, order: tuple[int, int] = ORDERS[-1]) -> float:
    return (3 * lobe_volume(RocheField(qi), order) / (4 * pi)) ** (1 / 3)


def lobe_slope(
    qi: float, noise: float, step: float = 0.02, order: tuple[int, int] = ORDERS[-1]
) -> dict[str, float]:
    """Estimate d ln R_L / d ln qi by Richardson-extrapolated central differences.

    Five-point differences at ``step`` and ``step / 2`` in ln qi. The reported
    error is the extrapolation's change plus ``noise``, the relative error of
    one radius, divided by the smaller step: truncation and roundoff both.
    """
    from math import exp, log

    def five_point(h: float) -> float:
        r = {k: log(lobe_volume_radius(qi * exp(k * h), order)) for k in (-2, -1, 1, 2)}
        return (r[-2] - 8 * r[-1] + 8 * r[1] - r[2]) / (12 * h)

    coarse, fine = five_point(step), five_point(step / 2)
    value = fine + (fine - coarse) / 15
    return dict(value=value, error=abs(value - fine) + 3 * noise / (step / 2))
