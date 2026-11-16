"""Adaptive curvature-matched cubic sections of Roche equipotentials."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from math import atan2, cos, hypot, isfinite, pi, sin, sqrt
from typing import TYPE_CHECKING

from rocheometry._bezier import BezierPath, CompoundBezierPath, CubicBezier
from rocheometry._computers.checks import (
    require_depth,
)
from rocheometry._computers.contact import (
    _geometry_depth,
    default_outer,
    default_watershed,
    require_part_depth,
)
from rocheometry._computers.neck import (
    _certified_neck_ray,
    _neck_displacement,
    _neck_field,
)
from rocheometry._computers.saddle import _neck_width
from rocheometry._computers.section_path.gauge_fixed import _neck_angle
from rocheometry._computers.surface_radius.radial import _enclosing_ray_bracket
from rocheometry._computers.watershed import WatershedComputer
from rocheometry._errors import EmptySectionError
from rocheometry._geometry import AffineTransform2D, Point2D, Point3D
from rocheometry._numerics.roots import brent_root
from rocheometry._planes import (
    AxisNormalPlane,
    CentrelinePlane,
    SectionPlane,
    _canonical_section_plane,
    _transverse_direction,
)
from rocheometry._roche.field import RocheField
from rocheometry._validation import (
    _normalize_finite_positive_float,
    require_instance,
)

# Insensitive control: section trig cleanup tolerance.
SECTION_TRIG_CLEANUP_TOLERANCE = 1e-14
# Insensitive control: section arm root relative tolerance.
SECTION_ARM_ROOT_RELATIVE_TOLERANCE = 8e-16
# Insensitive control: section ray root relative tolerance.
SECTION_RAY_ROOT_RELATIVE_TOLERANCE = 8e-15
# Insensitive control: normalized residual of the G2 arm-curvature equation.
SECTION_ARM_CURVATURE_RESIDUAL_TOLERANCE = 1e-09

if TYPE_CHECKING:
    from rocheometry._computers.critical_point.protocol import CriticalPointComputer
    from rocheometry._computers.surface_radius.protocol import SurfaceRadiusComputer

# Work guard or cache capacity: section max refinement depth.
SECTION_MAX_REFINEMENT_DEPTH = 24
# Derived: section reflect second axis.
SECTION_REFLECT_SECOND_AXIS = AffineTransform2D(yy=-1.0)
# Derived: section reflect first axis.
SECTION_REFLECT_FIRST_AXIS = AffineTransform2D(xx=-1.0)


@dataclass(frozen=True, slots=True, kw_only=True)
class TracedSectionPathComputer:
    """An adaptive cubic section strategy for centreline and axis-normal planes.

    G2 cubics interpolate exact radial knots, analytic tangents, and signed
    level-curve curvatures (de Boor, Höllig & Sabin, 1987). Their local error is
    sixth order. Inadmissible curvature arms trigger subdivision; only at the
    depth limit may an interval fall back to its angular Hermite arms.
    A detached nose panel follows the existing smooth neck slot. Inadmissible
    curvature arms there may use tangent-matched chord cubics, certified by
    the same radial-error checks, without increasing the depth limit.
    Refinement checks three interior radial intersections with a safety margin.
    The tolerance describes radial error relative to the selected surface's
    pole radius, rather than a control-point or parameter error.

    Attributes:
        radius_computer: Component-centred ray intersection strategy.
        l1_computer: L1 strategy defining contact and bounding the closed branch.
        tolerance: Finite positive relative radial error target.
        outer_computer: Strategy bounding contact levels by the outer opening.
        watershed_computer: Divider supplying contact cuts and rims.
    """

    radius_computer: SurfaceRadiusComputer
    l1_computer: CriticalPointComputer
    tolerance: float = 1.0e-5
    outer_computer: CriticalPointComputer = dataclass_field(
        default_factory=default_outer
    )
    watershed_computer: WatershedComputer = dataclass_field(
        default_factory=default_watershed
    )

    def __post_init__(self) -> None:
        for name, computer in (
            ("radius_computer", self.radius_computer),
            ("l1_computer", self.l1_computer),
            ("outer_computer", self.outer_computer),
        ):
            if not callable(computer):
                raise TypeError(f"{name}: expected a callable computer")
        for name in ("rim", "rim_at_depth", "cut_height", "sheet_point_on_ray"):
            if not callable(getattr(self.watershed_computer, name, None)):
                raise TypeError(
                    f"watershed_computer.{name}: expected a callable computer"
                )
        object.__setattr__(
            self,
            "tolerance",
            _normalize_finite_positive_float(self.tolerance, name="tolerance"),
        )

    def __call__(
        self, field: RocheField, depth: float, plane: SectionPlane, /
    ) -> CompoundBezierPath:
        """Return all closed pieces of the component part's planar outline."""
        depth = require_depth(depth)
        require_instance(plane, (CentrelinePlane, AxisNormalPlane), name="plane")
        l1 = self.l1_computer(field)
        depth = require_part_depth(
            field, depth, l1, self.outer_computer, allow_l2=False
        )
        depth = _geometry_depth(depth, l1)
        if isinstance(plane, AxisNormalPlane) and depth > 0:
            canonical, reflect_second = _canonical_section_plane(plane)
            if isinstance(canonical, CentrelinePlane):
                # The same geometric plane needs the same displacement and
                # neck panel, including at arbitrarily shallow contact.
                path = CompoundBezierPath((self._compute(field, depth, canonical),))
                return (
                    path.transformed(SECTION_REFLECT_SECOND_AXIS).reversed()
                    if reflect_second
                    else path
                )
            from rocheometry._computers.section_path.axis_contact import (
                AxisContactSection,
            )

            return AxisContactSection(self, field, depth, plane).paths()[0]
        return CompoundBezierPath((self._compute(field, depth, plane),))

    def envelope(
        self, field: RocheField, depth: float, plane: SectionPlane, /
    ) -> CompoundBezierPath:
        """Return the envelope runs, reusing the outline's segment construction."""
        l1 = self.l1_computer(field)
        depth = require_part_depth(
            field, depth, l1, self.outer_computer, allow_l2=False
        )
        depth = _geometry_depth(depth, l1)
        if isinstance(plane, AxisNormalPlane) and depth > 0:
            canonical, reflect_second = _canonical_section_plane(plane)
            if isinstance(canonical, CentrelinePlane):
                path = CompoundBezierPath(
                    (self._compute(field, depth, canonical, arc_only=True),)
                )
                return (
                    path.transformed(SECTION_REFLECT_SECOND_AXIS).reversed()
                    if reflect_second
                    else path
                )
            from rocheometry._computers.section_path.axis_contact import (
                AxisContactSection,
            )

            return CompoundBezierPath(
                AxisContactSection(self, field, depth, plane).envelope()
            )
        return CompoundBezierPath((self._compute(field, depth, plane, arc_only=True),))

    def _compute(
        self,
        field: RocheField,
        depth: float,
        plane: SectionPlane,
        /,
        *,
        arc_only: bool = False,
    ) -> BezierPath:
        """Trace a planar section, optionally omitting its contact cut.

        Args:
            field: Component-local Roche field.
            depth: Depth below exact L1, strictly below the outer-opening gap.
            plane: Centreline plane or axis-normal section, in separation units.
            arc_only: Return the open envelope arc at contact levels.

        Raises:
            EmptySectionError: The section's polar centre is not inside the surface.
            RuntimeError: The fixed refinement or ray-bracketing limit is reached.
        """
        depth = require_depth(depth)
        require_instance(plane, (CentrelinePlane, AxisNormalPlane), name="plane")
        l1 = self.l1_computer(field)
        depth = require_part_depth(
            field, depth, l1, self.outer_computer, allow_l2=False
        )
        depth = _geometry_depth(depth, l1)
        if depth > 0:
            assert isinstance(plane, CentrelinePlane)
            return self._contact(field, depth, plane, arc_only=arc_only)
        plane, reflect_second = _canonical_section_plane(plane)
        return self._trace(field, depth, plane, reflect_second)

    def _trace(
        self,
        field: RocheField,
        depth: float,
        plane: SectionPlane,
        reflect_second: bool = False,
        *,
        start: float = 0.0,
        start_point: Point2D | None = None,
    ) -> BezierPath:
        l1 = self.l1_computer(field)
        critical = depth == 0
        scale = self.radius_computer(field, depth, (0.0, 0.0, 1.0))
        if isinstance(plane, CentrelinePlane):
            origin = (0.0, 0.0, 0.0)
            first = (1.0, 0.0, 0.0)
            second = _transverse_direction(plane)

            def radius(angle: float) -> float:
                if angle == start and start_point is not None:
                    return hypot(*start_point)
                c, s = _trig(angle)
                direction = _combine(first, second, c, s)
                if angle == 0.0 and critical:
                    return l1.position[0]
                return self.radius_computer(field, depth, direction)

            initial_slope = 0.0
            if critical:
                cone = field.critical_upper_tangent(l1.position[0], plane.inclination)
                initial_slope = l1.position[0] * cone[0] / cone[1]
            end = pi
        else:
            index = ("x", "y", "z").index(plane.axis)
            origin = (
                plane.offset if index == 0 else 0.0,
                plane.offset if index == 1 else 0.0,
                plane.offset if index == 2 else 0.0,
            )
            first, second = _section_basis(plane.axis)
            if plane.offset != 0.0:
                sign = 1.0 if plane.offset > 0.0 else -1.0
                normal = (
                    sign if index == 0 else 0.0,
                    sign if index == 1 else 0.0,
                    sign if index == 2 else 0.0,
                )
                extent = self.radius_computer(field, depth, normal)
                if (
                    abs(plane.offset) >= extent
                    or field.potential_difference(origin, l1.position) + depth <= 0
                ):
                    raise EmptySectionError(
                        "section centre must be strictly inside the surface"
                    )

            def radius(angle: float) -> float:
                c, s = _trig(angle)
                direction = _combine(first, second, c, s)
                return _section_radius(
                    field,
                    depth,
                    origin,
                    direction,
                    l1.position[0],
                    scale,
                )

            initial_slope = 0.0
            end = 0.5 * pi if plane.axis == "x" else pi

        def knot(angle: float) -> _Knot:
            r = radius(angle)
            c, s = _trig(angle)
            direction = _combine(first, second, c, s)
            point = (
                origin[0] + r * direction[0],
                origin[1] + r * direction[1],
                origin[2] + r * direction[2],
            )
            if angle == start and start_point is not None:
                point = _combine(first, second, *start_point)
            gradient = field.gradient(*point)
            neck = isinstance(plane, CentrelinePlane) and _certified_neck_ray(
                depth, direction, l1.position[0], field
            )
            if neck:
                gradient = _neck_field(field, l1.position[0]).gradient(
                    _neck_displacement(field, l1.position[0], depth, direction)
                )
            corner = isinstance(plane, CentrelinePlane) and critical and angle == 0.0
            if corner:
                curvature = field.critical_branch_curvature(
                    l1.position[0], plane.inclination
                )
            else:
                g1, g2 = _dot(gradient, first), _dot(gradient, second)
                hessian = field.hessian(*point)
                h_first = (
                    _dot(hessian[0], first),
                    _dot(hessian[1], first),
                    _dot(hessian[2], first),
                )
                h_second = (
                    _dot(hessian[0], second),
                    _dot(hessian[1], second),
                    _dot(hessian[2], second),
                )
                h11 = _dot(first, h_first)
                h12 = _dot(first, h_second)
                h22 = _dot(second, h_second)
                curvature = (
                    -(h11 * g2 * g2 - 2.0 * h12 * g1 * g2 + h22 * g1 * g1)
                    / hypot(g1, g2) ** 3
                )
            if angle == 0.0:
                slope = initial_slope
            elif angle == end:
                slope = 0.0
            else:
                derivative = _combine(first, second, -s, c)
                radial_gradient = _dot(gradient, direction)
                if radial_gradient == 0.0:
                    raise RuntimeError("section radial gradient is zero away from L1")
                slope = -r * _dot(gradient, derivative) / radial_gradient
                if not isfinite(slope):
                    raise RuntimeError("section tangent is not finite")
            return _Knot(
                angle,
                start_point
                if angle == start and start_point is not None
                else (r * c, r * s),
                (slope * c - r * s, slope * s + r * c),
                curvature,
                neck,
            )

        return _trace_polar_field(
            radius,
            knot,
            start=start,
            end=end,
            scale=scale,
            tolerance=self.tolerance,
            reflect_second=reflect_second,
            front_panel=(
                _neck_angle(_neck_width(field, depth, l1), (end - start) / 4)
                if isinstance(plane, CentrelinePlane) and depth < 0
                else None
            ),
        )

    def _contact(
        self,
        field: RocheField,
        depth: float,
        plane: CentrelinePlane,
        *,
        arc_only: bool,
    ) -> BezierPath:
        from rocheometry._computers.section_path.contact import ContactCut, contact_path

        cut = ContactCut(
            field, depth, plane, self.l1_computer(field), self.watershed_computer
        )
        envelope = self._trace(
            field, depth, plane, start=cut.rim_angle, start_point=cut.point(1.0)
        )
        upper_envelope = envelope.segments[: len(envelope.segments) // 2]
        if arc_only:
            return BezierPath(envelope.segments, closed=False)
        scale = self.radius_computer(field, depth, (0.0, 0.0, 1.0))
        return contact_path(cut.adaptive(self.tolerance * scale), upper_envelope)


def _trace_polar_field(
    radius: Callable[[float], float],
    knot: Callable[[float], _Knot],
    *,
    start: float,
    end: float,
    scale: float,
    tolerance: float,
    reflect_second: bool = False,
    upper_only: bool = False,
    depth_limit: int | None = None,
    front_panel: float | None = None,
) -> BezierPath:
    """Trace a planar polar field using its exact radii, tangents and curvature."""
    accepted: list[CubicBezier] = []
    threshold = 0.2 * tolerance * scale
    # The cap guards runaway recursion, not accuracy: qi=1e8, F=1e-6 at
    # pi/3 needed depth 49. Its smallest panel was 1.4e-15 rad, still about
    # 1e8 endpoint ULPs wide. A future neck panel can avoid this refinement.
    max_depth = SECTION_MAX_REFINEMENT_DEPTH if start == 0.0 else 56
    if depth_limit is not None:
        max_depth = depth_limit

    def refine(left: _Knot, right: _Knot, depth: int) -> None:
        h = (right.angle - left.angle) / 3.0
        speed0, speed1 = hypot(*left.derivative), hypot(*right.derivative)
        t0 = (left.derivative[0] / speed0, left.derivative[1] / speed0)
        t1 = (right.derivative[0] / speed1, right.derivative[1] / speed1)
        # A displacement-resolved neck can be almost radial; the detached
        # nose also approaches a straight cone with ill-conditioned G2 arms.
        # Chord scale stays regular in both geometrically defined panels.
        nose = front_panel is not None and right.angle <= start + front_panel
        neck = (left.neck and right.neck) or nose
        chord_arm = (
            hypot(right.point[0] - left.point[0], right.point[1] - left.point[1]) / 3
        )
        arms = _g2_arms(left, right, t0, t1, chord_arm if neck else h * speed0)
        if arms is not None or depth >= SECTION_MAX_REFINEMENT_DEPTH or nose:
            d0, d1 = (
                arms
                if arms is not None
                else (chord_arm, chord_arm)
                if neck
                else (h * speed0, h * speed1)
            )
            segment = CubicBezier(
                left.point,
                (left.point[0] + d0 * t0[0], left.point[1] + d0 * t0[1]),
                (right.point[0] - d1 * t1[0], right.point[1] - d1 * t1[1]),
                right.point,
            )
            good = True
            for t in (0.25, 0.5, 0.75):
                x, y = segment.evaluate(t)
                angle = atan2(y, x)
                if (
                    not left.angle <= angle <= right.angle
                    or abs(hypot(x, y) - radius(angle)) > threshold
                ):
                    good = False
                    break
            if good:
                accepted.append(segment)
                return
        if depth == max_depth:
            raise RuntimeError("section refinement depth limit reached")
        middle = knot(0.5 * (left.angle + right.angle))
        refine(left, middle, depth + 1)
        refine(middle, right, depth + 1)

    left = knot(start)
    # The smooth neck slot resolves a detached rounded nose. Its cap tends
    # to the first ordinary bisection away from the saddle and uses the
    # same refinement cap and error checks as the other intervals.
    angles = [start + front_panel] if front_panel else []
    angles.extend(
        end if upper_only and index == 4 else start + (end - start) * index / 4.0
        for index in range(1, 5)
    )
    for angle in angles:
        right = knot(angle)
        refine(left, right, 0)
        left = right
    if upper_only:
        return BezierPath(tuple(accepted), closed=False)
    if end == 0.5 * pi:
        accepted.extend(
            segment.reversed().transformed(SECTION_REFLECT_FIRST_AXIS)
            for segment in reversed(tuple(accepted))
        )
    lower = tuple(
        segment.reversed().transformed(SECTION_REFLECT_SECOND_AXIS)
        for segment in reversed(accepted)
    )
    path = BezierPath(tuple(accepted) + lower, closed=start == 0.0)
    if reflect_second:
        return path.transformed(SECTION_REFLECT_SECOND_AXIS).reversed()
    return path


@dataclass(frozen=True, slots=True)
class _Knot:
    angle: float
    point: Point2D
    derivative: Point2D
    curvature: float
    neck: bool = False


def _cross(first: Point2D, second: Point2D) -> float:
    return first[0] * second[1] - first[1] * second[0]


def _g2_arms(
    left: _Knot, right: _Knot, t0: Point2D, t1: Point2D, h0: float
) -> tuple[float, float] | None:
    """Solve the endpoint curvature equations near the angular Hermite arms.

    Eliminate the right arm to match the left curvature exactly. Refine all
    sampled sign changes and prefer the root nearest the Hermite guess. Nearly
    circular intervals can have a double root without a sign change, so also
    allow the minimum residual when it meets the same admissibility bound.
    """
    delta = (right.point[0] - left.point[0], right.point[1] - left.point[1])
    a, b, c = _cross(t0, delta), _cross(delta, t1), _cross(t0, t1)
    if abs(c) < SECTION_TRIG_CLEANUP_TOLERANCE:
        return None
    scale = abs(b) + abs(c) * h0
    k0, k1 = 1.5 * left.curvature, 1.5 * right.curvature

    def residual(d0: float) -> float:
        d1 = (a - k0 * d0 * d0) / c
        return (k1 * d1 * d1 - b + c * d0) / scale

    lo, hi = 0.4 * h0, 2.5 * h0
    grid = [lo + (hi - lo) * i / 84 for i in range(85)]
    values = [residual(d) for d in grid]
    roots: list[float] = []
    for i in range(84):
        if values[i] == 0.0:
            roots.append(grid[i])
        elif values[i] * values[i + 1] < 0.0:
            roots.append(
                brent_root(
                    residual,
                    grid[i],
                    grid[i + 1],
                    absolute_tolerance=0.0,
                    relative_tolerance=SECTION_ARM_ROOT_RELATIVE_TOLERANCE,
                )
            )
    if values[-1] == 0.0:
        roots.append(grid[-1])
    if roots:
        best = min(roots, key=lambda d: abs(d - h0))
    else:
        i = min(range(85), key=lambda j: abs(values[j]))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, 84)]
        golden = 0.5 * (sqrt(5.0) - 1.0)
        for _ in range(100):
            xa, xb = hi - golden * (hi - lo), lo + golden * (hi - lo)
            if abs(residual(xa)) < abs(residual(xb)):
                hi = xb
            else:
                lo = xa
        best = 0.5 * (lo + hi)
    d1 = (a - k0 * best * best) / c
    if not (isfinite(best) and isfinite(d1) and best > 0.0 and d1 > 0.0):
        return None
    if not abs(residual(best)) <= SECTION_ARM_CURVATURE_RESIDUAL_TOLERANCE:
        return None
    return best, d1


def _trig(angle: float) -> tuple[float, float]:
    if angle == 0.0:
        return (1.0, 0.0)
    if angle == pi:
        return (-1.0, 0.0)
    if angle == 0.5 * pi:
        return (0.0, 1.0)
    return (cos(angle), sin(angle))


def _combine(first: Point3D, second: Point3D, c: float, s: float) -> Point3D:
    return (
        c * first[0] + s * second[0],
        c * first[1] + s * second[1],
        c * first[2] + s * second[2],
    )


def _dot(first: Point3D, second: Point3D) -> float:
    return sum(a * b for a, b in zip(first, second, strict=True))


def _section_basis(axis: str) -> tuple[Point3D, Point3D]:
    if axis == "x":
        return ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    if axis == "y":
        return ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def _section_radius(
    field: RocheField,
    depth: float,
    origin: Point3D,
    direction: Point3D,
    l1_x: float,
    scale: float,
) -> float:
    def residual(radius: float) -> float:
        point: Point3D = (
            origin[0] + radius * direction[0],
            origin[1] + radius * direction[1],
            origin[2] + radius * direction[2],
        )
        return field.potential_difference(point, (l1_x, 0.0, 0.0)) + depth

    # The same component-centred L1-radius sphere used by radial solves
    # encloses every detached/critical surface. Its intersection with this
    # in-plane ray is outside the component, before the centrifugal exterior.
    # Axis-normal origins are perpendicular to their in-plane directions.
    offset = hypot(*origin)
    if offset >= l1_x:
        raise EmptySectionError("section centre must be strictly inside the surface")
    outer = sqrt((l1_x - offset) * (l1_x + offset)) / hypot(*direction)
    inner, outer = _enclosing_ray_bracket(
        residual,
        0.0,
        outer,
        SECTION_RAY_ROOT_RELATIVE_TOLERANCE
        * (abs(float(field._axis_potential_exact(l1_x))) + abs(depth)),
    )
    if inner == outer:
        return outer
    return outer * brent_root(
        lambda t: residual(outer * t),
        inner / outer,
        1.0,
        absolute_tolerance=0.0,
        relative_tolerance=SECTION_RAY_ROOT_RELATIVE_TOLERANCE,
    )
