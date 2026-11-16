"""Fixed angular knots and chord arms for centreline section families."""

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from math import cos, hypot, pi, sin, sqrt

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
from rocheometry._computers.critical_point import CriticalPointComputer
from rocheometry._computers.neck import (
    _certified_neck_ray,
    _neck_displacement,
    _neck_field,
)
from rocheometry._computers.saddle import _neck_width
from rocheometry._computers.surface_radius import SurfaceRadiusComputer
from rocheometry._computers.watershed import WatershedComputer
from rocheometry._geometry import AffineTransform2D, Point2D, Point3D
from rocheometry._planes import (
    AxisNormalPlane,
    CentrelinePlane,
    SectionPlane,
    _canonical_section_plane,
    _transverse_direction,
)
from rocheometry._roche.field import RocheField
from rocheometry._validation import require_instance, require_int

# Shape multiplier verified on the holdout: neck slot placement multiplier.
NECK_SLOT_PLACEMENT_MULTIPLIER = 10.0
"""Scale in the geometric mean of neck width and the first base-knot angle."""


def _neck_angle(nu: float, first_angle: float) -> float:
    """Approach L1 outside its rounded tip, smoothly bounded by half a base gap."""
    scale = sqrt(NECK_SLOT_PLACEMENT_MULTIPLIER * nu * first_angle)
    bound = first_angle / 2
    return scale * (bound / hypot(scale, bound))


def _chord_cubic(p0: Point2D, t0: Point2D, p3: Point2D, t1: Point2D) -> CubicBezier:
    """Place one-third-chord arms along the supplied unit endpoint tangents."""
    arm = hypot(p3[0] - p0[0], p3[1] - p0[1]) / 3
    return CubicBezier(
        p0,
        (p0[0] + arm * t0[0], p0[1] + arm * t0[1]),
        (p3[0] - arm * t1[0], p3[1] - arm * t1[1]),
        p3,
    )


def _envelope_tangent(
    field: RocheField,
    point: Point2D,
    second: Point3D,
    c: float,
    s: float,
    *,
    depth: float = 0.0,
    x1: float = 0.5,
) -> Point2D:
    """Return the unit equipotential tangent oriented towards increasing angle."""
    gx, gy, gz = field.gradient(point[0], point[1] * second[1], point[1] * second[2])
    direction = (c, s * second[1], s * second[2])
    if _certified_neck_ray(depth, direction, x1, field):
        gx, gy, gz = _neck_field(field, x1).gradient(
            _neck_displacement(field, x1, depth, direction)
        )
    tx, ty = gy * second[1] + gz * second[2], -gx
    norm = hypot(tx, ty)
    tx, ty = tx / norm, ty / norm
    return (-tx, -ty) if -s * tx + c * ty < 0.0 else (tx, ty)


def _gauge_angles(segments_per_half: int, start: float, nu: float) -> tuple[float, ...]:
    """Shared fixed polar schedule, including its collapsing neck interval."""
    m = segments_per_half - 2
    base = tuple(
        start + (pi - start) * (u - sin(2 * pi * u) / (2 * pi))
        for u in (k / m for k in range(m + 1))
    )
    neck = base[0] + _neck_angle(nu, base[1] - base[0])
    return (base[0], neck, *base[1:])


def _gauge_upper(
    angles: tuple[float, ...],
    radius: Callable[[float], float],
    tangent: Callable[[Point2D, float, float], Point2D],
    *,
    start_point: Point2D | None = None,
    critical_point: Point2D | None = None,
    cone_tangent: Point2D | None = None,
) -> tuple[CubicBezier, ...]:
    """Interpolate any star-shaped planar field on the fixed chord-arm gauge."""
    points: list[Point2D] = []
    tangents: list[Point2D] = []
    for k, angle in enumerate(angles):
        c, s = (
            (1.0, 0.0)
            if k == 0 and angle == 0.0
            else (-1.0, 0.0)
            if k == len(angles) - 1
            else (cos(angle), sin(angle))
        )
        if angle == 0.0 and critical_point is not None:
            point = critical_point
            assert cone_tangent is not None
            unit = cone_tangent
        else:
            if k == 0 and start_point is not None:
                point = start_point
            else:
                r = radius(angle)
                point = (r * c, r * s)
            unit = tangent(point, c, s)
        points.append(point)
        tangents.append(unit)
    return tuple(
        CubicBezier(points[k], points[k], points[k], points[k])
        if k == 0 and critical_point is not None
        else _chord_cubic(points[k], tangents[k], points[k + 1], tangents[k + 1])
        for k in range(len(angles) - 1)
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class GaugeFixedSectionPathComputer:
    """Interpolate radial knots and analytic tangents with a fixed topology.

    Knot angles cluster at the front and back of the component. An extra knot
    approaches L1 more slowly than the rounded neck width, so its exact tangent
    converges to the cone direction from either side of the critical level.
    All nondegenerate segments use one third of their chord length along the
    exact endpoint tangents, including neck and rim segments. Each half
    starts with a degenerate segment on detached surfaces, or with the watershed
    cut at contact levels. A separate shrinking neck segment follows it. Both
    leading segments collapse to L1 at critical fill. The remaining knots follow the
    same angular schedule. Only centreline planes are supported.
    Control points are continuous through L1. Once the neck knot leaves its
    cap, they converge like F^(1/4).

    Attributes:
        radius_computer: Component-centred ray intersection strategy.
        l1_computer: L1 strategy defining the critical endpoint.
        outer_computer: Strategy bounding contact levels by the outer opening.
        watershed_computer: Divider supplying contact cuts and rims.
        segments_per_half: Number of upper-half cubics, an integer at least three.
    """

    radius_computer: SurfaceRadiusComputer
    l1_computer: CriticalPointComputer
    segments_per_half: int = 26
    outer_computer: CriticalPointComputer = dataclass_field(
        default_factory=default_outer
    )
    watershed_computer: WatershedComputer = dataclass_field(
        default_factory=default_watershed
    )

    def __post_init__(self) -> None:
        for name in (
            "radius_computer",
            "l1_computer",
            "outer_computer",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name}: expected a callable computer")
        for name in ("rim", "rim_at_depth", "cut_height", "sheet_point_on_ray"):
            if not callable(getattr(self.watershed_computer, name, None)):
                raise TypeError(
                    f"watershed_computer.{name}: expected a callable computer"
                )
        n = require_int(self.segments_per_half, name="segments_per_half")
        if n < 3:
            raise ValueError("segments_per_half: expected at least 3")

    def __call__(
        self, field: RocheField, depth: float, plane: SectionPlane, /
    ) -> CompoundBezierPath:
        """Return the single-piece outline in a compound path."""
        return CompoundBezierPath((self._compute(field, depth, plane),))

    def envelope(
        self, field: RocheField, depth: float, plane: SectionPlane, /
    ) -> CompoundBezierPath:
        """Return the envelope runs, reusing the outline's segment construction."""
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
        """Return an upper-half-first section, optionally omitting a contact cut."""
        depth = require_depth(depth)
        require_instance(plane, (CentrelinePlane, AxisNormalPlane), name="plane")
        l1 = self.l1_computer(field)
        depth = require_part_depth(
            field, depth, l1, self.outer_computer, allow_l2=False
        )
        depth = _geometry_depth(depth, l1)
        if depth > 0:
            if isinstance(plane, AxisNormalPlane):
                raise ValueError("gauge-fixed sections support only centreline planes")
            return self._contact(field, depth, plane, arc_only=arc_only)
        plane, reflect_second = _canonical_section_plane(plane)
        if not isinstance(plane, CentrelinePlane):
            raise ValueError("gauge-fixed sections support only centreline planes")
        critical = depth == 0
        second = _transverse_direction(plane)
        angles = _gauge_angles(
            self.segments_per_half, 0.0, _neck_width(field, depth, l1)
        )

        def radius(angle: float) -> float:
            c, s = cos(angle), sin(angle)
            if angle == 0.0:
                c, s = 1.0, 0.0
            elif angle == pi:
                c, s = -1.0, 0.0
            return self.radius_computer(field, depth, (c, s * second[1], s * second[2]))

        upper = list(
            _gauge_upper(
                angles,
                radius,
                lambda p, c, s: _envelope_tangent(field, p, second, c, s),
                critical_point=l1.position[:2] if critical else None,
                cone_tangent=field.critical_upper_tangent(
                    l1.position[0], plane.inclination
                )
                if critical
                else None,
            )
        )
        front = upper[0].p0
        upper.insert(0, CubicBezier(front, front, front, front))
        reflection = AffineTransform2D(yy=-1.0)
        lower = tuple(
            segment.reversed().transformed(reflection) for segment in reversed(upper)
        )
        path = BezierPath(tuple(upper) + lower, closed=True)
        return path.transformed(reflection).reversed() if reflect_second else path

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
        second = _transverse_direction(plane)
        l1 = self.l1_computer(field)
        angles = _gauge_angles(
            self.segments_per_half, cut.rim_angle, _neck_width(field, depth, l1)
        )

        def radius(angle: float) -> float:
            c, s = (-1.0, 0.0) if angle == pi else (cos(angle), sin(angle))
            return self.radius_computer(field, depth, (c, s * second[1], s * second[2]))

        envelope = _gauge_upper(
            angles,
            radius,
            lambda p, c, s: _envelope_tangent(
                field, p, second, c, s, depth=depth, x1=l1.position[0]
            ),
            start_point=cut.point(1.0),
        )
        return contact_path(
            () if arc_only else (cut.single(),),
            tuple(envelope),
            arc_only=arc_only,
        )
