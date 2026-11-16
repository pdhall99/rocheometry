"""Adaptive and fixed-gauge paths of the quadrature projected field."""

from dataclasses import dataclass

from rocheometry._bezier import BezierPath, CompoundBezierPath, CubicBezier
from rocheometry._computers.contact import _geometry_depth, require_part_depth
from rocheometry._computers.critical_point import CriticalPointComputer
from rocheometry._computers.saddle import _neck_width
from rocheometry._computers.section_path.gauge_fixed import _gauge_angles, _gauge_upper
from rocheometry._computers.silhouette.limb import trace_limb
from rocheometry._computers.silhouette.projected import ProjectedField
from rocheometry._computers.surface_radius import SurfaceRadiusComputer
from rocheometry._computers.watershed import WatershedComputer
from rocheometry._geometry import AffineTransform2D
from rocheometry._planes import CentrelinePlane, _require_silhouette_plane
from rocheometry._roche.field import RocheField
from rocheometry._validation import (
    _normalize_finite_positive_float,
    require_int,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _SilhouettePaths:
    """Shared projected-field dependencies and closed-outline validation."""

    radius_computer: SurfaceRadiusComputer
    l1_computer: CriticalPointComputer
    outer_computer: CriticalPointComputer
    watershed_computer: WatershedComputer

    def __post_init__(self) -> None:
        for name in ("radius_computer", "l1_computer", "outer_computer"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name}: expected a callable computer")
        for name in ("rim", "rim_at_depth", "cut_height", "sheet_point_on_ray"):
            if not callable(getattr(self.watershed_computer, name, None)):
                raise TypeError(
                    f"watershed_computer.{name}: expected a callable computer"
                )

    def _field(
        self, field: RocheField, depth: float, plane: CentrelinePlane
    ) -> ProjectedField:
        _require_silhouette_plane(plane)
        l1 = self.l1_computer(field)
        depth = require_part_depth(
            field, depth, l1, self.outer_computer, allow_l2=False
        )
        depth = _geometry_depth(depth, l1)
        return ProjectedField(
            field,
            depth,
            plane,
            self.radius_computer,
            l1,
            self.outer_computer,
            self.watershed_computer,
        )

    def __call__(
        self, field: RocheField, depth: float, plane: CentrelinePlane, /
    ) -> CompoundBezierPath:
        """Return a closed detached or critical outline, upper half first."""
        _require_silhouette_plane(plane)
        l1 = self.l1_computer(field)
        depth = require_part_depth(
            field, depth, l1, self.outer_computer, allow_l2=False
        )
        if depth > 0:
            raise NotImplementedError("use silhouette_envelope at contact levels")
        return self.envelope(field, depth, plane)

    def envelope(
        self, field: RocheField, depth: float, plane: CentrelinePlane, /
    ) -> CompoundBezierPath:
        """Return closed stellar outlines or open watershed-bounded envelope arcs."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True, kw_only=True)
class TracedSilhouettePathComputer(_SilhouettePaths):
    """Trace globally maximizing LOS branches with their exact derivatives.

    Equal-height junctions share an endpoint, retaining each branch's own
    tangent and curvature to represent the true corner.
    Tolerance bounds radial error in units of the surface's pole radius.
    """

    tolerance: float = 1e-5

    def __post_init__(self) -> None:  # type: ignore[missing-override-decorator]
        _SilhouettePaths.__post_init__(self)
        object.__setattr__(
            self,
            "tolerance",
            _normalize_finite_positive_float(self.tolerance, name="tolerance"),
        )

    def envelope(  # type: ignore[missing-override-decorator]
        self, field: RocheField, depth: float, plane: CentrelinePlane, /
    ) -> CompoundBezierPath:
        """Return closed stellar outlines or open watershed-bounded envelope arcs."""
        projected = self._field(field, depth, plane)
        path = trace_limb(projected, self.tolerance)
        return CompoundBezierPath((path,))


@dataclass(frozen=True, slots=True, kw_only=True)
class GaugeFixedSilhouettePathComputer(_SilhouettePaths):
    """Use a fixed 88-segment chord-arm gauge for sky outlines.

    Contact endpoint slots collapse at the limb crossing instead of drawing
    the cut. The neck knot and its cap are shared with gauge-fixed sections.
    Controls converge through L1 like F^(1/4) once that knot is uncapped.
    The plain section schedule uses 44 segments per half, selected on the
    dense heavy, edge-on grid using traced and pointwise global references.
    """

    segments_per_half: int = 44

    def __post_init__(self) -> None:  # type: ignore[missing-override-decorator]
        _SilhouettePaths.__post_init__(self)
        if require_int(self.segments_per_half, name="segments_per_half") < 3:
            raise ValueError("segments_per_half: expected at least 3")

    def envelope(  # type: ignore[missing-override-decorator]
        self, field: RocheField, depth: float, plane: CentrelinePlane, /
    ) -> CompoundBezierPath:
        projected = self._field(field, depth, plane)
        angles = _gauge_angles(
            self.segments_per_half,
            projected.start,
            _neck_width(field, projected.depth, projected.l1),
        )
        upper = _gauge_upper(
            angles,
            projected.radius,
            projected.tangent,
            start_point=projected.start_point,
            critical_point=projected.l1.position[:2] if projected.critical else None,
            cone_tangent=projected.cone_tangent() if projected.critical else None,
        )
        front = upper[0].p0
        upper = (CubicBezier(front, front, front, front), *upper)
        reflection = AffineTransform2D(yy=-1.0)
        lower = tuple(s.reversed().transformed(reflection) for s in reversed(upper))
        return CompoundBezierPath(
            (BezierPath(upper + lower, closed=not projected.contact),)
        )
