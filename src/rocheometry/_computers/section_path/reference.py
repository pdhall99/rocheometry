"""Path-based geometry references with compatible polar-angle samples."""

from bisect import bisect_right
from dataclasses import dataclass
from math import atan2, pi

from rocheometry._bezier import BezierPath, CubicBezier, _axis_extrema
from rocheometry._computers.contact import admissibility_depth
from rocheometry._computers.defaults import (
    DEFAULT_REFERENCE_AZIMUTH_ORDER,
    DEFAULT_REFERENCE_FILL_TOLERANCE,
    DEFAULT_REFERENCE_POLAR_ORDER,
    REFERENCE_SECTION_TOLERANCE,
    _Settings,
)
from rocheometry._geometry import Point2D
from rocheometry._numerics.roots import brent_root
from rocheometry._planes import (
    ORBITAL_PLANE,
    AxisNormalPlane,
    CentrelinePlane,
    SectionPlane,
)
from rocheometry._roche.field import RocheField
from rocheometry._roche.potential import Potential, PotentialConvention
from rocheometry._roche.volume import volume_radius_from_volume
from rocheometry._types import ComponentId
from rocheometry._validation import (
    _normalize_finite_float,
    _normalize_finite_positive_float,
    normalize_mass_ratio,
    normalize_volume_radius_fill,
    require_component_id,
    require_instance,
    require_num_upper_points,
)


@dataclass(frozen=True, slots=True)
class SectionLandmarks:
    """Positive-axis, top, and negative-axis landmarks in a section's chart.

    In a centreline plane `inner` points towards the companion and is L1 at
    contact. In an x section the first chart axis instead points along y.

    Attributes:
        inner: Positive first-axis intersection.
        top: Maximum upper second-coordinate point on the path.
        back: Negative first-axis intersection.
    """

    inner: Point2D
    top: Point2D
    back: Point2D


@dataclass(frozen=True, slots=True)
class SectionReference:
    """Geometry metadata and an authoritative traced component-local path.

    Samples and landmarks are derived from `path`, without further field
    solves. `num_upper_points` preserves the polar-angle sampling interface
    used by existing fitting tools; it never changes the traced geometry.

    Attributes:
        q: Finite positive system mass ratio `M1/M2`.
        component_id: Selected component number, `1` or `2`.
        volume_radius_fill: Surface-to-lobe volume-radius ratio in `(0, 1]`.
        surface_potential: Canonical Jacobi level defining the surface.
        volume: Positive surface volume in separation units cubed.
        lobe_volume: Positive normalization volume in separation units cubed.
        path: Connected, closed traced path in the plane's separation-unit chart.
        path_tolerance: Positive radial error target divided by surface volume radius.
        plane: Component-local centreline or section plane.
        num_upper_points: Compatibility upper-half sample count, at least `2`.
    """

    q: float
    component_id: ComponentId
    volume_radius_fill: float
    surface_potential: Potential
    volume: float
    lobe_volume: float
    path: BezierPath
    path_tolerance: float
    plane: SectionPlane = ORBITAL_PLANE
    num_upper_points: int = 257

    def __post_init__(self) -> None:
        object.__setattr__(self, "q", normalize_mass_ratio(self.q))
        object.__setattr__(
            self, "component_id", require_component_id(self.component_id)
        )
        object.__setattr__(
            self,
            "volume_radius_fill",
            normalize_volume_radius_fill(self.volume_radius_fill),
        )
        require_instance(self.surface_potential, Potential, name="surface_potential")
        if self.surface_potential.convention is not PotentialConvention.JACOBI:
            raise ValueError("surface_potential: expected canonical Jacobi potential")
        for name in ("volume", "lobe_volume", "path_tolerance"):
            object.__setattr__(
                self,
                name,
                _normalize_finite_positive_float(getattr(self, name), name=name),
            )
        require_instance(self.path, BezierPath, name="path")
        if not self.path.closed or len(self.path.segments) % 2:
            raise ValueError(
                "path: expected a closed, reflection-symmetric traced section"
            )
        upper = self.path.segments[: len(self.path.segments) // 2]
        if upper[0].p0[1] != 0.0 or upper[-1].p3[1] != 0.0:
            raise ValueError("path: upper half must start and end on the first axis")
        if upper[0].p0[0] <= 0.0 or upper[-1].p3[0] >= 0.0:
            raise ValueError(
                "path: upper half must run from positive to negative first axis"
            )
        for original, reflected in zip(
            reversed(upper), self.path.segments[len(upper) :], strict=True
        ):
            for a, b in zip(
                (original.p3, original.p2, original.p1, original.p0),
                (reflected.p0, reflected.p1, reflected.p2, reflected.p3),
                strict=True,
            ):
                if b != (a[0], -a[1]):
                    raise ValueError("path: expected exact reflection symmetry")
        require_instance(self.plane, (CentrelinePlane, AxisNormalPlane), name="plane")
        object.__setattr__(
            self, "num_upper_points", require_num_upper_points(self.num_upper_points)
        )

    @property
    def upper_path(self) -> BezierPath:
        """The open upper-half path, from positive to negative first axis."""
        return BezierPath(
            self.path.segments[: len(self.path.segments) // 2], closed=False
        )

    def upper_point(self, angle: float) -> Point2D:
        """Return the path's upper-half intersection at one polar angle.

        Args:
            angle: Finite chart polar angle in radians, in `[0, pi]`.
        """
        angle = _normalize_finite_float(angle, name="angle")
        if not 0.0 <= angle <= pi:
            raise ValueError("angle: expected a value in [0, pi]")
        return _upper_point(self.path.segments[: len(self.path.segments) // 2], angle)

    @property
    def upper(self) -> tuple[Point2D, ...]:
        """Compatibility samples uniform in upper-half polar angle."""
        segments = self.path.segments[: len(self.path.segments) // 2]
        angles = tuple(atan2(segment.p3[1], segment.p3[0]) for segment in segments)
        return tuple(
            _upper_point(segments, pi * index / (self.num_upper_points - 1), angles)
            for index in range(self.num_upper_points)
        )

    @property
    def landmarks(self) -> SectionLandmarks:
        """Axis endpoints and the analytic maximum second-coordinate path point."""
        segments = self.upper_path.segments
        candidates = []
        for segment in segments:
            parameters = (
                0.0,
                1.0,
                *_axis_extrema(
                    segment.p0[1], segment.p1[1], segment.p2[1], segment.p3[1]
                ),
            )
            candidates.extend(segment.evaluate(t) for t in parameters)
        return SectionLandmarks(
            segments[0].p0, max(candidates, key=lambda point: point[1]), segments[-1].p3
        )

    @property
    def lower(self) -> tuple[Point2D, ...]:
        """The reflected compatibility samples, from negative to positive axis."""
        return tuple((x, -u) for x, u in reversed(self.upper))

    @property
    def closed_points(self) -> tuple[Point2D, ...]:
        """The closed compatibility samples with exact reflection symmetry."""
        upper = self.upper
        return upper + tuple((x, -u) for x, u in reversed(upper[1:-1])) + (upper[0],)

    @property
    def volume_radius(self) -> float:
        """The surface volume-equivalent radius in separation units."""
        return volume_radius_from_volume(self.volume)

    @property
    def lobe_volume_radius(self) -> float:
        """The lobe volume-equivalent radius in separation units."""
        return volume_radius_from_volume(self.lobe_volume)


def reference_section(
    q: float,
    component_id: ComponentId,
    volume_radius_fill: float,
    plane: SectionPlane = ORBITAL_PLANE,
    *,
    num_upper_points: int = 257,
    polar_order: int = DEFAULT_REFERENCE_POLAR_ORDER,
    azimuth_order: int = DEFAULT_REFERENCE_AZIMUTH_ORDER,
    volume_radius_fill_tolerance: float = DEFAULT_REFERENCE_FILL_TOLERANCE,
) -> SectionReference:
    """Return geometry metadata and a reference graph's traced planar path.

    Args:
        q: Finite positive system mass ratio `M1/M2`.
        component_id: Selected component, `1` or `2`.
        volume_radius_fill: Surface-to-lobe volume-radius ratio in `(0, 1]`.
        plane: Component-local centreline or axis-normal section plane.
        num_upper_points: Compatibility polar-angle sample count, at least `2`.
        polar_order: Volume polar quadrature order, at least `4`; default `128`.
        azimuth_order: Quarter-azimuth volume order, at least `4`; default `64`.
        volume_radius_fill_tolerance: Finite positive relative radius residual
            for fill inversion; default `1e-13`.

    Raises:
        EmptySectionError: A section's polar centre is not strictly inside the surface.
        RuntimeError: A numerical solve or path refinement fails to converge.
    """
    q = normalize_mass_ratio(q)
    component_id = require_component_id(component_id)
    require_instance(plane, (CentrelinePlane, AxisNormalPlane), name="plane")
    num_upper_points = require_num_upper_points(num_upper_points)
    accuracy = _Settings(
        polar_order=polar_order,
        azimuth_order=azimuth_order,
        fill_tolerance=volume_radius_fill_tolerance,
        watershed_tolerance=1e-13,
        fitted_selection=False,
        exact_anchor_everywhere=True,
        fixed_paths=False,
        path_tolerance=1e-10,
    )
    field = RocheField(q)
    if component_id == 2:
        field = field.reciprocal()
    graph = accuracy._computers()
    fill = normalize_volume_radius_fill(volume_radius_fill)
    depth = 0.0 if fill == 1.0 else graph.selection.depth(field, fill)
    l1 = graph.l1(field)
    if depth >= admissibility_depth(field, l1, graph.outer):
        raise ValueError("potential must be above L2")
    potential = Potential.jacobi(field._level_value(l1.position[0], depth))
    return SectionReference(
        q=q,
        component_id=component_id,
        volume_radius_fill=volume_radius_fill,
        surface_potential=potential,
        volume=graph.volume(field, depth),
        lobe_volume=graph.selection.fast_lobe_volume(field),
        path=graph.section(field, depth, plane).paths[0],
        path_tolerance=REFERENCE_SECTION_TOLERANCE,
        plane=plane,
        num_upper_points=num_upper_points,
    )


def reference_component_section(
    q: float,
    component_id: ComponentId,
    volume_radius_fill: float,
    *,
    num_upper_points: int = 257,
    polar_order: int = DEFAULT_REFERENCE_POLAR_ORDER,
    azimuth_order: int = DEFAULT_REFERENCE_AZIMUTH_ORDER,
    volume_radius_fill_tolerance: float = DEFAULT_REFERENCE_FILL_TOLERANCE,
) -> SectionReference:
    """Return an orbital-plane specialization of `reference_section`.

    Args:
        q: Finite positive system mass ratio `M1/M2`.
        component_id: Selected component, `1` or `2`.
        volume_radius_fill: Surface-to-lobe volume-radius ratio in `(0, 1]`.
        num_upper_points: Compatibility polar-angle sample count, at least `2`.
        polar_order: Volume polar order, at least `4`; default `128`.
        azimuth_order: Quarter-azimuth volume order, at least `4`; default `64`.
        volume_radius_fill_tolerance: Finite positive relative radius residual;
            default `1e-13`.
    """
    return reference_section(
        q,
        component_id,
        volume_radius_fill,
        ORBITAL_PLANE,
        num_upper_points=num_upper_points,
        polar_order=polar_order,
        azimuth_order=azimuth_order,
        volume_radius_fill_tolerance=volume_radius_fill_tolerance,
    )


def reference_centreline_section(
    q: float,
    component_id: ComponentId,
    volume_radius_fill: float,
    plane: CentrelinePlane = ORBITAL_PLANE,
    *,
    num_upper_points: int = 257,
    polar_order: int = DEFAULT_REFERENCE_POLAR_ORDER,
    azimuth_order: int = DEFAULT_REFERENCE_AZIMUTH_ORDER,
    volume_radius_fill_tolerance: float = DEFAULT_REFERENCE_FILL_TOLERANCE,
) -> SectionReference:
    """Return a centreline-plane specialization of `reference_section`.

    Args:
        q: Finite positive system mass ratio `M1/M2`.
        component_id: Selected component, `1` or `2`.
        volume_radius_fill: Surface-to-lobe volume-radius ratio in `(0, 1]`.
        plane: Centreline plane with an inclination in radians.
        num_upper_points: Compatibility polar-angle sample count, at least `2`.
        polar_order: Volume polar order, at least `4`; default `128`.
        azimuth_order: Quarter-azimuth volume order, at least `4`; default `64`.
        volume_radius_fill_tolerance: Finite positive relative radius residual;
            default `1e-13`.
    """
    require_instance(plane, CentrelinePlane, name="plane")
    return reference_section(
        q,
        component_id,
        volume_radius_fill,
        plane,
        num_upper_points=num_upper_points,
        polar_order=polar_order,
        azimuth_order=azimuth_order,
        volume_radius_fill_tolerance=volume_radius_fill_tolerance,
    )


def _upper_point(
    segments: tuple[CubicBezier, ...],
    angle: float,
    angles: tuple[float, ...] | None = None,
) -> Point2D:
    if angle == 0.0:
        return segments[0].p0
    if angle == pi:
        return segments[-1].p3
    if angles is None:
        angles = tuple(atan2(segment.p3[1], segment.p3[0]) for segment in segments)
    segment = segments[min(bisect_right(angles, angle), len(segments) - 1)]

    def residual(t: float) -> float:
        x, y = segment.evaluate(t)
        return atan2(y, x) - angle

    return segment.evaluate(brent_root(residual, 0.0, 1.0))
