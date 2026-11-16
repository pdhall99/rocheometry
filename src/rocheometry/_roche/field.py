"""Canonical Jacobi field expressed in normalized component-local coordinates.

The selected component is at x=0 and its companion at x=1. `qi` is
M_component / M_companion. The potential is normalized by the total binary
mass in one physical centre-of-mass gauge. Component-local coordinates are
right-handed coordinate charts of that same field, not different gauges.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from math import cos, fsum, hypot, isfinite, sqrt

from rocheometry._geometry import Point2D, Point3D
from rocheometry._roche.mass import mass_fractions
from rocheometry._validation import (
    normalize_component_mass_ratio,
    normalize_inclination,
)

POTENTIAL_CONVENTION = "jacobi"
"""The serialized identifier for the canonical Jacobi potential."""


@dataclass(frozen=True, slots=True)
class RocheField:
    """A dimensionless Jacobi field in a normalized component-local frame.

    The selected component is at `x=0`, its companion at `x=1`, and their
    separation is one. Potentials are positive, normalized by the total mass,
    and use the centre-of-mass centrifugal gauge; component frames are charts
    of the same physical field. Construction performs no critical-point solve.

    Args:
        qi: Component-to-companion mass ratio. Must be finite and positive;
            real numeric inputs become floats, excluding booleans.

    Attributes:
        qi: Normalized component-to-companion mass ratio.
        component_mass: Read-only component mass fraction of the total mass.
        companion_mass: Read-only companion mass fraction of the total mass.
    """

    qi: float
    component_mass: float = field(init=False, repr=False)
    companion_mass: float = field(init=False, repr=False)
    _mass_error: float = field(init=False, repr=False, compare=False)
    _half_mass_error: float = field(init=False, repr=False, compare=False)
    _potential_constant: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        qi = normalize_component_mass_ratio(self.qi)
        component_mass, companion_mass = mass_fractions(qi)
        object.__setattr__(self, "qi", qi)
        object.__setattr__(self, "component_mass", component_mass)
        object.__setattr__(self, "companion_mass", companion_mass)
        self._set_mass_correction()

    def _set_mass_correction(self) -> None:
        error, constant = _mass_correction(self.component_mass, self.companion_mass)
        object.__setattr__(self, "_mass_error", error)
        object.__setattr__(self, "_half_mass_error", error / 2)
        object.__setattr__(self, "_potential_constant", constant)

    def reciprocal(self) -> "RocheField":
        """Return the opposite chart with the same stored masses swapped exactly.

        The reciprocal qi is a rounded label, not a recipe for recomputing
        the masses. Equality, hashing and the default frozen-slot pickle
        include both mass fractions, preserving this distinction.
        """
        value = object.__new__(RocheField)
        object.__setattr__(value, "qi", normalize_component_mass_ratio(1.0 / self.qi))
        object.__setattr__(value, "component_mass", self.companion_mass)
        object.__setattr__(value, "companion_mass", self.component_mass)
        value._set_mass_correction()
        return value

    def potential(
        self,
        x: float,
        y: float = 0.0,
        z: float = 0.0,
    ) -> float:
        """Return the total-mass-normalized positive Jacobi potential.

        Args:
            x: Finite component-local x coordinate in separation units.
            y: Finite orbital-plane y coordinate in separation units.
            z: Finite out-of-plane z coordinate in separation units.

        Raises:
            ValueError: The position coincides with either point mass, where the
                field is singular.
        """
        _component_distance_squares(x, y, z)
        return self._potential(x, y, z)

    def potential_difference(self, point: Point3D, reference: Point3D) -> float:
        """Return U(point)-U(reference), factoring nearby coordinates.

        The axis form retains small saddle and transverse offsets near L1.
        Both positions must be finite and different from either point mass.
        Subclasses with another potential may override this operation; the
        default subclass implementation subtracts their potential evaluations.

        Args:
            point: Evaluation position in the component-local chart.
            reference: Reference position in the same chart and separation units.
        """
        _component_distance_squares(*point)
        _component_distance_squares(*reference)
        if type(self) is not RocheField:
            return self.potential(*point) - self.potential(*reference)
        if (
            min(
                hypot(*point),
                hypot(point[0] - 1, point[1], point[2]),
                hypot(*reference),
                hypot(reference[0] - 1, reference[1], reference[2]),
            )
            == 0
        ):
            raise ValueError("Jacobi potential is singular at a component position")
        if reference[1] == reference[2] == 0.0 and 0 < reference[0] < 1:
            return self._axis_potential_difference(point, reference[0])
        x, y, z = point
        a, b, c = reference
        r0, r1 = hypot(x, y, z), hypot(x - 1, y, z)
        s0, s1 = hypot(a, b, c), hypot(a - 1, b, c)
        transverse = (y - b) * (y + b) + (z - c) * (z + c)
        terms = (
            -self.component_mass
            * ((x - a) * (x + a) + transverse)
            / (r0 * s0 * (r0 + s0)),
            -self.companion_mass
            * ((x - a) * (x + a - 2) + transverse)
            / (r1 * s1 * (r1 + s1)),
            0.5 * ((x - a) * (x + a - 2 * self.companion_mass) + (y - b) * (y + b)),
        )
        if self._mass_error:
            return fsum(
                (
                    *terms,
                    self._half_mass_error * ((x - a) * (x + a) + (y - b) * (y + b)),
                )
            )
        return fsum(terms)

    def _axis_potential_difference(self, point: Point3D, x1: float) -> float:
        """Retain the factored arithmetic at an interior x-axis reference."""
        x, y, z = point
        dx = x - x1
        transverse = y * y + z * z
        r0, r1 = hypot(x, y, z), hypot(1 - x, y, z)
        terms = (
            -self.component_mass * (dx * (x + x1) + transverse) / (r0 * x1 * (r0 + x1)),
            -self.companion_mass
            * (dx * (x + x1 - 2) + transverse)
            / (r1 * (1 - x1) * (r1 + 1 - x1)),
            0.5 * (dx * (x + x1 - 2 * self.companion_mass) + y * y),
        )
        if self._mass_error:
            return fsum((*terms, self._half_mass_error * (dx * (x + x1) + y * y)))
        return fsum(terms)

    def log_qi_derivative(self, x: float, y: float = 0.0, z: float = 0.0) -> float:
        """Return `partial U / partial ln(qi)` at a fixed component-local point.

        With component and companion mass fractions `m` and `c`, this is
        `m*c*(1/r_component - 1/r_companion + x - c)` in the canonical Jacobi
        gauge. Coordinates and separation stay fixed while `qi` changes.

        Args:
            x: Finite component-local x coordinate in separation units.
            y: Finite orbital-plane y coordinate in separation units.
            z: Finite out-of-plane z coordinate in separation units.

        Raises:
            ValueError: A coordinate is non-finite or the point is a singular mass.
        """
        r0, r1 = _component_distance_squares(x, y, z)
        if r0 == 0.0 or r1 == 0.0:
            raise ValueError(
                "Jacobi mass derivative is undefined at a component position"
            )
        m, c = self.component_mass, self.companion_mass
        return m * c * (1.0 / sqrt(r0) - 1.0 / sqrt(r1) + x - c)

    def _potential(self, x: float, y: float = 0.0, z: float = 0.0) -> float:
        """Evaluate trusted solver coordinates without public input checks."""
        r_component_sq = x * x + y * y + z * z
        dx_companion = x - 1.0
        r_companion_sq = dx_companion * dx_companion + y * y + z * z
        if r_component_sq == 0.0 or r_companion_sq == 0.0:
            raise ValueError("Jacobi potential is singular at a component position")
        dx_com = x - self.companion_mass
        first = self.component_mass / sqrt(r_component_sq)
        second = self.companion_mass / sqrt(r_companion_sq)
        centrifugal = 0.5 * (dx_com * dx_com + y * y)
        if self._mass_error:
            # Accumulate the mass-total correction with the large terms to limit
            # cancellation; a unit mass total uses the sequential sum below.
            return fsum(
                (
                    first,
                    second,
                    centrifugal,
                    self._half_mass_error * (x * x + y * y),
                    self._potential_constant,
                )
            )
        return first + second + centrifugal

    @lru_cache(maxsize=4096)
    def _axis_potential_exact(self, x: float) -> Fraction:
        """Exact stored-mass potential with omega squared equal to m+c."""
        m, c, a = (
            Fraction(self.component_mass),
            Fraction(self.companion_mass),
            Fraction(x),
        )
        total = m + c
        return m / abs(a) + c / abs(1 - a) + total * (a - c / total) ** 2 / 2

    @lru_cache(maxsize=4096)
    def _background_potential_exact(self) -> Fraction:
        """Finite companion/centrifugal contribution at the selected mass."""
        m, c = Fraction(self.component_mass), Fraction(self.companion_mass)
        return c + c * c / (2 * (m + c))

    @lru_cache(maxsize=4096)
    def _axis_depth_parts(self, l1_x: float, anchor_x: float) -> tuple[float, float]:
        """Per-field depth of an axial anchor, stored as a double-double."""
        if type(self).potential is not RocheField.potential:
            return self.potential_difference(
                (l1_x, 0.0, 0.0), (anchor_x, 0.0, 0.0)
            ), 0.0
        exact = self._axis_potential_exact(l1_x) - self._axis_potential_exact(anchor_x)
        high = float(exact)
        return high, float(exact - Fraction(high))

    def _depth_offset(self, depth: float, l1_x: float, anchor_x: float) -> float:
        """Return U(anchor)-u without rounding an absolute level or saddle."""
        if anchor_x == l1_x:
            return depth
        high, low = self._axis_depth_parts(l1_x, anchor_x)
        difference, error = _two_sum(depth, -high)
        return fsum((difference, error, -low))

    @lru_cache(maxsize=4096)
    def _level_value(self, l1_x: float, depth: float) -> float:
        """Round the absolute fast-path report once, without a per-level Fraction.

        The cached axis value supplies its exact integer ratio. Integer division
        rounds the difference correctly, including depths smaller than an ULP
        of the absolute level. Cache only the float report, so unfactored rays
        on one surface share this conversion without a per-level Fraction.
        No absolute level is used by factored residuals.
        """
        exact = self._axis_potential_exact(l1_x)
        numerator, denominator = depth.as_integer_ratio()
        return (exact.numerator * denominator - numerator * exact.denominator) / (
            exact.denominator * denominator
        )

    def _ray_residual(
        self,
        direction: Point3D,
        depth: float,
        reference_x: float | None = None,
        *,
        inverse_radius: bool = False,
        l1_x: float,
    ) -> Callable[[float], float]:
        """Bind trusted ray coefficients, optionally factoring a nearby reference.

        Factored differences avoid cancellation near the stationary L1 cusp.
        The reference is an ordinary point on the x-axis, not field state.
        The L1 offset is depth itself; another axial anchor uses its cached
        double-double depth. Only the unfactored branch forms an absolute level.
        """
        x, y, z = direction
        transverse_squared = y * y + z * z
        norm = sqrt(x * x + transverse_squared)
        component_mass = self.component_mass
        companion_mass = self.companion_mass
        mass_over_norm = component_mass / norm
        y_squared = y * y
        if reference_x is None:
            level = self._level_value(l1_x, depth)
            if inverse_radius:

                def inverse_residual(coordinate: float) -> float:
                    inverse = 1.0 / coordinate
                    dx_companion = coordinate - x
                    companion_distance = sqrt(
                        dx_companion * dx_companion + transverse_squared
                    )
                    dx_com = x * inverse - companion_mass
                    value = coordinate * (
                        mass_over_norm + companion_mass / companion_distance
                    ) + 0.5 * (dx_com * dx_com + y_squared * inverse * inverse)
                    if self._mass_error:
                        value += (
                            self._half_mass_error
                            * (x * x + y_squared)
                            * inverse
                            * inverse
                            + self._potential_constant
                        )
                    return value - level

                return inverse_residual

            def residual(radius: float) -> float:
                dx_companion = radius * x - 1.0
                companion_radius = sqrt(
                    dx_companion * dx_companion + radius * radius * transverse_squared
                )
                dx_com = radius * x - companion_mass
                value = (
                    mass_over_norm / radius
                    + companion_mass / companion_radius
                    + 0.5 * (dx_com * dx_com + radius * radius * y_squared)
                )
                if self._mass_error:
                    value += (
                        self._half_mass_error * radius * radius * (x * x + y_squared)
                        + self._potential_constant
                    )
                return value - level

            return residual

        reference_component = abs(reference_x)
        reference_companion = abs(1.0 - reference_x)
        offset = self._depth_offset(depth, l1_x, reference_x)
        norm_minus_x = transverse_squared / (norm + x) if x > 0 else norm - x
        norm_plus_x = transverse_squared / (norm - x) if x < 0 else norm + x

        def factored_residual(radius: float) -> float:
            if inverse_radius:
                radius = 1.0 / radius
            radial_distance = radius * norm
            gap = reference_component - radial_distance
            x_gap = (
                gap + radius * norm_minus_x
                if reference_x > 0
                else -gap - radius * norm_plus_x
            )
            companion_radius = sqrt(
                (radius * x - 1.0) ** 2 + radius * radius * transverse_squared
            )
            companion_difference = (
                -x_gap * (2.0 - reference_x - radius * x)
                - radius * radius * transverse_squared
            ) / (
                companion_radius
                * reference_companion
                * (companion_radius + reference_companion)
            )
            centrifugal_difference = (
                -x_gap * (radius * x + reference_x - 2.0 * companion_mass)
                + radius * radius * y_squared
            )
            # Smallest-first ordinary addition was measured too: sorting these
            # four terms costs more than fsum, which also preserves cone-node
            # rounding. Keep it; fitted wiring limits the all-ray extension.
            terms = (
                component_mass * gap / (radial_distance * reference_component),
                companion_mass * companion_difference,
                0.5 * centrifugal_difference,
                offset,
            )
            if self._mass_error:
                return fsum(
                    (
                        *terms,
                        self._half_mass_error
                        * (
                            -x_gap * (radius * x + reference_x)
                            + radius * radius * y_squared
                        ),
                    )
                )
            return fsum(terms)

        return factored_residual

    def gradient(
        self,
        x: float,
        y: float = 0.0,
        z: float = 0.0,
    ) -> Point3D:
        """Return the Jacobi gradient in normalized component-local coordinates.

        Args:
            x: Finite component-local x coordinate in separation units.
            y: Finite orbital-plane y coordinate in separation units.
            z: Finite out-of-plane z coordinate in separation units.

        Returns:
            The `(dU/dx, dU/dy, dU/dz)` gradient in total-mass-normalized units.

        Raises:
            ValueError: The position coincides with either singular point mass.
        """
        r_component_sq, r_companion_sq = _component_distance_squares(x, y, z)
        if r_component_sq == 0.0 or r_companion_sq == 0.0:
            raise ValueError("Jacobi gradient is undefined at a component position")

        component_mass, companion_mass = self.component_mass, self.companion_mass
        if y == 0.0 and z == 0.0 and -1.0 <= x < 1.0:
            # This factorization already contains (m+c)*x-c, including epsilon.
            return (_centreline_gradient(x, component_mass, companion_mass), 0.0, 0.0)
        r_component_cubed = r_component_sq * sqrt(r_component_sq)
        r_companion_cubed = r_companion_sq * sqrt(r_companion_sq)
        dx = (
            -component_mass * x / r_component_cubed
            - companion_mass * (x - 1.0) / r_companion_cubed
            + x
            - companion_mass
        )
        dy = (
            -component_mass * y / r_component_cubed
            - companion_mass * y / r_companion_cubed
            + y
        )
        dz = (
            -component_mass * z / r_component_cubed
            - companion_mass * z / r_companion_cubed
        )
        if self._mass_error:
            dx += self._mass_error * x
            dy += self._mass_error * y
        return (dx, dy, dz)

    def hessian(
        self,
        x: float,
        y: float = 0.0,
        z: float = 0.0,
    ) -> tuple[Point3D, Point3D, Point3D]:
        """Return the symmetric Jacobi Hessian in component-local coordinates.

        Args:
            x: Finite component-local x coordinate in separation units.
            y: Finite orbital-plane y coordinate in separation units.
            z: Finite out-of-plane z coordinate in separation units.

        Returns:
            Three rows of second derivatives, in total-mass-normalized units.

        Raises:
            ValueError: The position coincides with either singular point mass.
        """
        r0, r1 = _component_distance_squares(x, y, z)
        if r0 == 0.0 or r1 == 0.0:
            raise ValueError("Jacobi Hessian is undefined at a component position")
        d0, d1 = sqrt(r0), sqrt(r1)
        # Form m/r^3 and unit directions, rather than m/r^5. The latter
        # underflows its denominator at representable tiny-mass critical points.
        w0 = self.component_mass / r0 / d0
        w1 = self.companion_mass / r1 / d1
        x0, y0, z0 = x / d0, y / d0, z / d0
        x1, y1, z1 = (x - 1.0) / d1, y / d1, z / d1
        radial = w0 + w1
        xx = 3.0 * (w0 * x0 * x0 + w1 * x1 * x1) - radial + 1.0
        xy = 3.0 * (w0 * x0 * y0 + w1 * x1 * y1)
        xz = 3.0 * (w0 * x0 * z0 + w1 * x1 * z1)
        yy = 3.0 * (w0 * y0 * y0 + w1 * y1 * y1) - radial + 1.0
        yz = 3.0 * (w0 * y0 * z0 + w1 * y1 * z1)
        zz = 3.0 * (w0 * z0 * z0 + w1 * z1 * z1) - radial
        if self._mass_error:
            xx += self._mass_error
            yy += self._mass_error
        return ((xx, xy, xz), (xy, yy, yz), (xz, yz, zz))

    def centreline_plane_hessian(
        self,
        x: float,
        u: float = 0.0,
        inclination: float = 0.0,
    ) -> tuple[float, float, float]:
        """Return (U_xx, U_xu, U_uu) in a centreline plane.

        Args:
            x: Finite centreline coordinate in separation units.
            u: Finite transverse coordinate within the plane, in separation units.
            inclination: Finite radians in `[0, pi]`, folded into `[0, pi/2]`.
                Zero selects the orbital plane.

        Raises:
            ValueError: The position coincides with either singular point mass.
        """
        inclination = normalize_inclination(inclination)

        r_component_sq, r_companion_sq = _component_distance_squares(x, u, 0.0)
        if r_component_sq == 0.0 or r_companion_sq == 0.0:
            raise ValueError("Jacobi Hessian is undefined at a component position")

        component_mass, companion_mass = self.component_mass, self.companion_mass
        dx_companion = x - 1.0
        r_component_fifth = r_component_sq * r_component_sq * sqrt(r_component_sq)
        r_companion_fifth = r_companion_sq * r_companion_sq * sqrt(r_companion_sq)

        u_xx = (
            component_mass * (3.0 * x * x - r_component_sq) / r_component_fifth
            + companion_mass
            * (3.0 * dx_companion * dx_companion - r_companion_sq)
            / r_companion_fifth
            + 1.0
        )
        u_xu = (
            3.0 * component_mass * x * u / r_component_fifth
            + 3.0 * companion_mass * dx_companion * u / r_companion_fifth
        )
        u_uu = (
            component_mass * (3.0 * u * u - r_component_sq) / r_component_fifth
            + companion_mass * (3.0 * u * u - r_companion_sq) / r_companion_fifth
            + cos(inclination) ** 2
        )
        if self._mass_error:
            u_xx += self._mass_error
            u_uu += self._mass_error * cos(inclination) ** 2
        return u_xx, u_xu, u_uu

    def axial_third_derivatives(self, x: float) -> tuple[float, float]:
        """Return `(U_xxx, U_xyy)` on the axis, where `U_xzz = U_xyy`.

        Derivatives use component-local axes and separation units, with the
        same total-mass normalization as the potential. The signed formula
        applies on either side of both masses as well as between them.

        Args:
            x: Finite axial coordinate, different from the masses at 0 and 1.

        Raises:
            ValueError: The coordinate is nonfinite or coincides with a mass.
        """
        if not isfinite(x):
            raise ValueError("coordinates must be finite")
        if x == 0.0 or x == 1.0:
            raise ValueError("Jacobi potential is singular at a component position")
        third = fsum(
            (
                self.component_mass / (abs(x) ** 3 * x),
                self.companion_mass / (abs(x - 1.0) ** 3 * (x - 1.0)),
            )
        )
        return -6.0 * third, 3.0 * third

    def critical_branch_curvature(self, x: float, inclination: float = 0.0) -> float:
        """Return the signed curvature of a saddle's upper critical branch.

        Args:
            x: Centreline saddle with positive axial and negative transverse
                curvature. Must not coincide with either point mass.
            inclination: Finite radians in `[0, pi]`, folded into `[0, pi/2]`.

        Returns:
            Curvature along `critical_upper_tangent`, positive for a
            counterclockwise turn in the plane's `(x, u)` coordinates.
        """
        a, _, b = self.centreline_plane_hessian(x, inclination=inclination)
        u_xxx, u_xuu = self.axial_third_derivatives(x)
        slope = sqrt(-a / b)
        e = (u_xxx + 3.0 * u_xuu * slope * slope) / (6.0 * b * slope)
        return -2.0 * e / hypot(1.0, slope) ** 3

    def critical_upper_tangent(
        self,
        x: float,
        inclination: float = 0.0,
    ) -> Point2D:
        """Return the upper-branch tangent at one centreline critical point.

        Args:
            x: Centreline position of a saddle with positive axial curvature and
                negative transverse curvature. Must not coincide with a point mass.
            inclination: Finite radians in `[0, pi]`, folded into `[0, pi/2]`.

        Returns:
            The unit `(dx, du)` tangent pointing toward decreasing x and increasing
            transverse coordinate on the upper critical branch.
        """
        u_xx, _, u_uu = self.centreline_plane_hessian(
            x,
            inclination=inclination,
        )
        slope = sqrt(u_xx / -u_uu)
        norm = hypot(1.0, slope)
        return (-1.0 / norm, slope / norm)


def _two_sum(a: float, b: float) -> tuple[float, float]:
    total = a + b
    remainder = total - a
    return total, (a - (total - remainder)) + (b - remainder)


@lru_cache(maxsize=4096)
def _mass_correction(m: float, c: float) -> tuple[float, float]:
    """Recover exact epsilon with TwoSum and reject unrepresentable corrections."""
    total, remainder = _two_sum(m, c)
    error, tail = _two_sum(total - 1.0, remainder)
    if tail != 0 or not isfinite(error) or abs(error) > 2.0**-52:
        raise ValueError(
            "stored mass-pair error must be representable and at most 2**-52"
        )
    exact_total = Fraction(m) + Fraction(c)
    constant = float(-Fraction(error) * Fraction(c) ** 2 / (2 * exact_total))
    return error, constant


def _centreline_gradient(
    x: float, component_mass: float, companion_mass: float
) -> float:
    """Factor the cancelling gravitational and centrifugal centreline terms."""
    if x > 0.0:
        distance = 1.0 - x
        return companion_mass / (distance * distance) * x * (
            1.0 + distance + distance * distance
        ) - component_mass / (x * x) * distance * (1.0 + x + x * x)
    radius = -x
    return component_mass / (radius * radius) * (1.0 - radius) * (
        1.0 + radius + radius * radius
    ) - companion_mass / ((1.0 + radius) ** 2) * radius * (
        3.0 + 3.0 * radius + radius * radius
    )


def _component_distance_squares(x: float, y: float, z: float) -> tuple[float, float]:
    """Return squared distances to the selected component and its companion."""
    if not all(isfinite(value) for value in (x, y, z)):
        raise ValueError("coordinates must be finite")
    r_component_sq = x * x + y * y + z * z
    dx_companion = x - 1.0
    r_companion_sq = dx_companion * dx_companion + y * y + z * z
    return r_component_sq, r_companion_sq
