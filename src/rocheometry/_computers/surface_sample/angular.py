"""Quadrature rays and directional radii for reflection-symmetric readers."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from math import atan2, cos, exp, expm1, fsum, hypot, log1p, pi, sin
from typing import TYPE_CHECKING

from rocheometry._computers.contact import (
    _geometry_depth,
    default_outer,
    default_watershed,
    require_part_depth,
)
from rocheometry._computers.critical_point import CriticalPointComputer
from rocheometry._computers.neck import _neck_surface
from rocheometry._computers.neck_ray import _refined_neck_ray
from rocheometry._computers.saddle import _neck_width
from rocheometry._computers.surface_sample.sample import SurfaceSample, _NeckSample
from rocheometry._computers.watershed import WatershedComputer
from rocheometry._geometry import Point3D
from rocheometry._numerics.quadrature import gauss_legendre
from rocheometry._roche.field import RocheField
from rocheometry._validation import require_quadrature_order


def _default_l1() -> CriticalPointComputer:
    from rocheometry._computers.defaults import default_l1_computer

    return default_l1_computer()


if TYPE_CHECKING:
    from rocheometry._computers.surface_radius.protocol import SurfaceRadiusComputer

# Resolution budget: default fitted polar order.
DEFAULT_FITTED_POLAR_ORDER = 64
"""The default Gauss-Legendre order in polar angle."""
# Resolution budget: default fitted azimuth order.
DEFAULT_FITTED_AZIMUTH_ORDER = 24
"""The default Gauss-Legendre order in the quarter-azimuth interval."""
# Shape multiplier verified on the holdout: saddle clustering factor.
SADDLE_CLUSTERING_FACTOR = 2.0
"""Neck widths per polar span in the smooth saddle-clustering map."""
# Shape multiplier verified on the holdout: azimuth clustering factor.
AZIMUTH_CLUSTERING_FACTOR = 1.0
"""L1-to-companion width multiplier for the exponential azimuth map."""
# Shape multiplier verified on the holdout: front power factor.
FRONT_POWER_FACTOR = 100.0
"""Neck-to-companion-scale ratio controlling detached front grading."""
# Shape multiplier verified on the holdout: rim quartic coefficient.
RIM_QUARTIC_COEFFICIENT = 0.25
"""Leading coefficient in the measured quartic rim grading polynomial."""
# Shape multiplier verified on the holdout: rim cubic factor.
RIM_CUBIC_FACTOR = 30.0
"""Inverse relative rim span at which deep contact resumes cubic grading."""


@dataclass(frozen=True, slots=True)
class GaussLegendreSurfaceSampler:
    """Sample radii on a symmetry-reduced angular Gauss-Legendre grid.

    Rays cover the quadrant `y >= 0, z >= 0`. Their weights include the
    fourfold reflection factor. Detached weights sum to `4*pi`; contact
    samples contain only envelope rays and exclude the cut. Detached polar
    grading resolves both L1 and the component's own outer saddle. Its front
    exponent rises smoothly from two towards three as the neck grows relative
    to the L1-to-companion angular scale. Contact grading blends quartic and
    cubic rim maps geometrically and resolves the outer saddle at the back.
    Clustering weakens away from saddles but does not switch off.
    Exponentially graded azimuths resolve the orbital-plane edge even on an
    extreme heavy component. The standard budget remains 64 by 24 rays.
    Readers may integrate only quantities symmetric under both y and z
    reflection. Sampling preserves polar-major, azimuth-minor ray order.

    Attributes:
        l1_computer: Critical point defining the onset of contact.
        outer_computer: Critical-point strategy defining the outer opening.
        watershed_computer: Divider used for contact parts.
        radius_computer: Directional radius strategy called at each angular node.
        polar_order: Polar-angle quadrature order; integer at least `4`.
        azimuth_order: Quarter-azimuth quadrature order; integer at least `4`.
    """

    radius_computer: SurfaceRadiusComputer
    polar_order: int = DEFAULT_FITTED_POLAR_ORDER
    azimuth_order: int = DEFAULT_FITTED_AZIMUTH_ORDER
    l1_computer: CriticalPointComputer = dataclass_field(
        default_factory=_default_l1, kw_only=True
    )
    outer_computer: CriticalPointComputer = dataclass_field(
        default_factory=default_outer, kw_only=True
    )
    watershed_computer: WatershedComputer = dataclass_field(
        default_factory=default_watershed, kw_only=True
    )

    def __post_init__(self) -> None:
        if not callable(self.radius_computer):
            raise TypeError("radius_computer: expected a callable computer")
        require_quadrature_order(self.polar_order, name="polar_order")
        require_quadrature_order(self.azimuth_order, name="azimuth_order")

    def __call__(self, field: RocheField, depth: float, /) -> SurfaceSample:
        """Return quadrature rays and their surface radii in separation units.

        Args:
            field: Normalized component-local field.
            depth: Depth below exact L1, bounded by the outer-opening gap.
        """
        l1 = self.l1_computer(field)
        depth = require_part_depth(field, depth, l1, self.outer_computer)
        requested_depth = depth
        depth = _geometry_depth(depth, l1)
        part = depth > 0
        azimuths = _azimuth_quadrature(self.azimuth_order, 1 - l1.position[0])
        rays = (
            self._part_quadrature(field, depth, azimuths)
            if part
            else _detached_quadrature(
                self.polar_order,
                azimuths,
                _neck_width(field, depth, l1),
                _neck_width(field, depth, self.outer_computer(field), l1),
                (1 - l1.position[0]) / l1.position[0],
            )
        )
        radii = tuple(
            self.radius_computer.node_radius(field, depth, d) for d, _ in rays
        )
        if _neck_surface(field, l1.position[0], requested_depth) is not None:
            # Even when the sub-ULP neck uses critical ray placement, retain
            # the physical depth through each node's root and gradient.
            geometry = tuple(
                _refined_neck_ray(field, l1.position[0], requested_depth, d, r)
                for (d, _), r in zip(rays, radii, strict=True)
            )
            return _NeckSample(
                rays,
                tuple(r for r, _ in geometry),
                part,
                origin=l1.position[0],
                depth=requested_depth,
                _gradients={
                    d: gradient
                    for (d, _), (_, gradient) in zip(rays, geometry, strict=True)
                },
            )
        return SurfaceSample(rays=rays, radii=radii, part=part)

    def _part_quadrature(
        self,
        field: RocheField,
        depth: float,
        azimuths: tuple[tuple[float, float], ...],
    ) -> tuple[tuple[Point3D, float], ...]:
        nodes, weights = gauss_legendre(self.polar_order)
        l1 = self.l1_computer(field)
        neck_width = _neck_width(field, depth, self.outer_computer(field), l1)
        spans = []
        for psi, weight in azimuths:
            rim = self.watershed_computer.rim(field, depth, psi)
            theta_rim = atan2(hypot(rim[1], rim[2]), rim[0])
            span = pi - theta_rim
            epsilon = SADDLE_CLUSTERING_FACTOR * neck_width / span
            spans.append((psi, weight, theta_rim, span, epsilon))
        rays = []
        for polar_node, polar_weight in zip(nodes, weights, strict=True):
            s = (polar_node + 1) / 2
            cubic = 3 * s**3 - 2 * s**4
            dcubic = 9 * s**2 - 8 * s**3
            a = RIM_QUARTIC_COEFFICIENT
            quartic = s**4 * (a + (5 - 2 * a) * s + (a - 4) * s * s)
            dquartic = s**3 * (4 * a + 5 * (5 - 2 * a) * s + 6 * (a - 4) * s * s)
            for psi, weight, theta_rim, span, epsilon in spans:
                # The cubic contribution keeps the first rim offsets
                # representable even at high orders and large rim angles.
                relative_rim = theta_rim / span
                mix = relative_rim + (RIM_CUBIC_FACTOR * relative_rim) ** 4
                h = (quartic + mix * cubic) / (1 + mix)
                dh = (dquartic + mix * dcubic) / (1 + mix)
                dg = (2 * (1 - h) + epsilon) / (1 + epsilon)
                # 1-g(1-h), factored to avoid subtracting two angles near pi.
                theta = theta_rim + span * h * (2 + epsilon - h) / (1 + epsilon)
                transverse = sin(theta)
                direction = (cos(theta), transverse * cos(psi), transverse * sin(psi))
                jacobian = span * dg * dh * transverse / 2
                rays.append((direction, 4 * weight * polar_weight * jacobian))
        return tuple(rays)


def _saddle_map(v: float, epsilon: float) -> tuple[float, float]:
    """Return the smooth quadratic clustering map and its derivative."""
    return v * (v + epsilon) / (1 + epsilon), (2 * v + epsilon) / (1 + epsilon)


def _detached_quadrature(
    polar_order: int,
    azimuths: tuple[tuple[float, float], ...],
    neck_width: float,
    back_neck_width: float,
    companion_neck_width: float,
) -> tuple[tuple[Point3D, float], ...]:
    """Build level-dependent, symmetry-reduced directions and weights."""
    theta_nodes, theta_weights = gauss_legendre(polar_order)
    result: list[tuple[Point3D, float]] = []
    epsilon = SADDLE_CLUSTERING_FACTOR * neck_width / pi
    back_epsilon = SADDLE_CLUSTERING_FACTOR * back_neck_width / pi
    ratio = FRONT_POWER_FACTOR * neck_width
    power = 2 + ratio / (companion_neck_width + ratio)
    polar = []
    for node, weight in zip(theta_nodes, theta_weights, strict=True):
        s = (node + 1) / 2
        front = s**power + epsilon * s
        dfront = power * s ** (power - 1) + epsilon
        back = (1 - s) * (1 - s + back_epsilon)
        dback = 2 * (1 - s) + back_epsilon
        g = front / (front + back)
        dg = (dfront * back + front * dback) / (front + back) ** 2
        theta = pi * g
        transverse = sin(theta)
        theta_weight = 0.5 * pi * dg * weight * transverse
        polar.append((theta, transverse, theta_weight))
    # Preserve the known angular measure even on deliberately small test or
    # user-selected grids. This correction vanishes at converged orders.
    normalizer = 2 / fsum(w for _, _, w in polar)
    for theta, transverse, theta_weight in polar:
        theta_weight *= normalizer
        for phi, phi_weight in azimuths:
            direction = (
                cos(theta),
                transverse * cos(phi),
                transverse * sin(phi),
            )
            result.append((direction, 4.0 * theta_weight * phi_weight))
    return tuple(result)


def _azimuth_quadrature(
    order: int, companion_width: float
) -> tuple[tuple[float, float], ...]:
    """Return orbital-plane-clustered angles and quarter-azimuth weights."""
    nodes, weights = gauss_legendre(order)
    span = pi / 2
    width = AZIMUTH_CLUSTERING_FACTOR * companion_width
    stretch = log1p(span / width)
    result = []
    for node, weight in zip(nodes, weights, strict=True):
        v = (node + 1) / 2
        result.append(
            (
                width * expm1(stretch * v),
                width * stretch * exp(stretch * v) * weight / 2,
            )
        )
    normalizer = span / fsum(w for _, w in result)
    return tuple((angle, weight * normalizer) for angle, weight in result)
