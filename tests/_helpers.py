"""Assemble test variants while consistently rewiring all dependent nodes."""

from collections.abc import Callable
from dataclasses import dataclass
from math import atan2, hypot, pi, sin, sqrt
from pathlib import Path

import rocheometry
from rocheometry._accuracy import _Accuracy
from rocheometry._computers.composition import ComputerSet, Memoized
from rocheometry._computers.critical_point import CriticalPointComputer
from rocheometry._computers.defaults import default_l1_computer, default_outer_computer
from rocheometry._computers.lobe_slope import (
    LobeVolumeRadiusSlopeComputer,
    SampledLobeSlopeComputer,
)
from rocheometry._computers.lobe_volume import (
    LobeVolumeComputer,
    LobeVolumeFromSurfaceComputer,
)
from rocheometry._computers.mean_curvature import (
    IntegratedMeanCurvatureComputer,
    SampledMeanCurvatureComputer,
)
from rocheometry._computers.mean_gravity import (
    MeanGravityComputer,
    SampledMeanGravityComputer,
)
from rocheometry._computers.quadrature import (
    SampledQuadratureComputer,
    SurfaceQuadratureComputer,
)
from rocheometry._computers.section_path import SectionPathComputer
from rocheometry._computers.section_path.gauge_fixed import (
    NECK_SLOT_PLACEMENT_MULTIPLIER,
    GaugeFixedSectionPathComputer,
)
from rocheometry._computers.silhouette.paths import GaugeFixedSilhouettePathComputer
from rocheometry._computers.silhouette.projected import ProjectedField
from rocheometry._computers.surface_area import SampledAreaComputer, SurfaceAreaComputer
from rocheometry._computers.surface_potential import (
    BisectionSurfacePotentialComputer,
    SurfacePotentialComputer,
)
from rocheometry._computers.surface_radius import (
    BrentSurfaceRadiusComputer,
    SurfaceRadiusComputer,
)
from rocheometry._computers.surface_sample import (
    GaussLegendreSurfaceSampler,
    SurfaceSampler,
)
from rocheometry._computers.surface_selection import (
    SurfaceSelection,
    VolumeInversionSelection,
)
from rocheometry._computers.surface_volume import (
    SampledVolumeComputer,
    SurfaceVolumeComputer,
)
from rocheometry._computers.volume_radius_fill import VolumeRadiusFillComputer
from rocheometry._computers.volume_radius_fill.from_volume import (
    VolumeRadiusFillFromVolumeComputer,
)
from rocheometry._computers.watershed import TracedWatershedComputer
from rocheometry._levels import _levels
from rocheometry._roche import RocheField

REPO_ROOT = Path(__file__).resolve().parents[1]


def projected_field(
    surface: rocheometry.ComponentSurfaceGeometry, plane: rocheometry.CentrelinePlane
) -> ProjectedField:
    """Return the library's projected field for a surface's silhouette."""
    g = surface.component
    return ProjectedField(
        g.field,
        surface._depth,
        plane,
        g._computers.radius,
        g.l1(),
        g._computers.outer,
        g._computers.watershed,
    )


def select(
    g: rocheometry.ComponentGeometry, kind: str, value: float
) -> rocheometry.ComponentSurfaceGeometry:
    """Return the surface at a volume-radius fill or, otherwise, a fill-out factor."""
    if kind == "fill":
        return g.surface(volume_radius_fill=value)
    return g.surface(fill_out_factor=value)


@dataclass(frozen=True, slots=True)
class _Wired(_Accuracy):
    """Unsupported test hook for an explicitly injected private graph."""

    graph: ComputerSet

    def _computers(self) -> ComputerSet:  # type: ignore[missing-override-decorator]
        return self.graph


def _depth(field: RocheField, potential: rocheometry.Potential) -> float:
    """Convert a test's literal public level at the same boundary as the API."""
    if type(field).potential is not RocheField.potential:
        return default_l1_computer()(field).potential.value - potential.value
    return _levels(field, rocheometry.Reference()._computers()).depth(potential)


def _opening_level(field: RocheField, outer: CriticalPointComputer) -> float:
    """Return the rounded opening report, for tests of the potential boundary."""
    opposite = field.reciprocal()
    return float(
        max(
            field._axis_potential_exact(outer(field).position[0]),
            opposite._axis_potential_exact(outer(opposite).position[0]),
        )
    )


@dataclass(frozen=True, slots=True, eq=False)
class TestSelection:
    """Assemble explicit selection variants for tests of custom computers."""

    __test__ = False
    fast_lobe_volume: LobeVolumeComputer
    depth: SurfacePotentialComputer
    fill: VolumeRadiusFillComputer


def computer_graph(
    *,
    l1_computer: CriticalPointComputer | None = None,
    outer_computer: CriticalPointComputer | None = None,
    radius_computer: SurfaceRadiusComputer | None = None,
    volume_computer: SurfaceVolumeComputer | None = None,
    sample_computer: SurfaceSampler | None = None,
    lobe_slope_computer: LobeVolumeRadiusSlopeComputer | None = None,
    area_computer: SurfaceAreaComputer | None = None,
    mean_curvature_computer: IntegratedMeanCurvatureComputer | None = None,
    mean_gravity_computer: MeanGravityComputer | None = None,
    quadrature_computer: SurfaceQuadratureComputer | None = None,
    depth_computer: SurfacePotentialComputer | None = None,
    lobe_volume_computer: LobeVolumeComputer | None = None,
    selection: SurfaceSelection | None = None,
    section_computer: SectionPathComputer | None = None,
    memoize_lobe: bool = True,
    polar_order: int = 8,
    azimuth_order: int = 6,
    maxsize: int | None = 4096,
    volume_radius_fill_tolerance: float = 2e-8,
) -> ComputerSet:
    """Build a graph from producers, explicitly sharing its injected dependencies."""
    l1 = Memoized(
        default_l1_computer() if l1_computer is None else l1_computer, maxsize=maxsize
    )
    outer = Memoized(
        default_outer_computer() if outer_computer is None else outer_computer,
        maxsize=maxsize,
    )
    watershed = TracedWatershedComputer(l1_computer=l1, outer_computer=outer)
    radius = (
        BrentSurfaceRadiusComputer(
            l1_computer=l1, outer_computer=outer, watershed_computer=watershed
        )
        if radius_computer is None
        else radius_computer
    )
    sample = Memoized(
        GaussLegendreSurfaceSampler(
            radius_computer=radius,
            l1_computer=l1,
            outer_computer=outer,
            watershed_computer=watershed,
            polar_order=polar_order,
            azimuth_order=azimuth_order,
        )
        if sample_computer is None
        else sample_computer,
        maxsize=16,
    )
    lobe_slope = Memoized(
        SampledLobeSlopeComputer(sample=sample, l1_computer=l1)
        if lobe_slope_computer is None
        else lobe_slope_computer,
        maxsize=maxsize,
    )
    area = Memoized(
        SampledAreaComputer(sample=sample) if area_computer is None else area_computer,
        maxsize=maxsize,
    )
    mean_curvature = Memoized(
        SampledMeanCurvatureComputer(sample=sample)
        if mean_curvature_computer is None
        else mean_curvature_computer,
        maxsize=maxsize,
    )
    mean_gravity = Memoized(
        SampledMeanGravityComputer(sample=sample)
        if mean_gravity_computer is None
        else mean_gravity_computer,
        maxsize=maxsize,
    )
    quadrature = Memoized(
        SampledQuadratureComputer(sample=sample)
        if quadrature_computer is None
        else quadrature_computer,
        maxsize=maxsize,
    )
    volume = (
        SampledVolumeComputer(sample=sample)
        if volume_computer is None
        else volume_computer
    )
    if selection is None:
        if lobe_volume_computer is None and depth_computer is None:
            selection = VolumeInversionSelection(
                volume_computer=volume,
                l1_computer=l1,
                outer_computer=outer,
                maxsize=maxsize,
                volume_radius_fill_tolerance=volume_radius_fill_tolerance,
            )
        else:
            normalization = Memoized(
                LobeVolumeFromSurfaceComputer(volume_computer=volume, l1_computer=l1),
                maxsize=maxsize,
            )
            lobe = (
                normalization
                if lobe_volume_computer is None
                else Memoized(lobe_volume_computer, maxsize=maxsize)
                if memoize_lobe
                else lobe_volume_computer
            )
            depth = Memoized(
                BisectionSurfacePotentialComputer(
                    volume_computer=volume,
                    l1_computer=l1,
                    lobe_volume_computer=normalization,
                    volume_radius_fill_tolerance=volume_radius_fill_tolerance,
                )
                if depth_computer is None
                else depth_computer,
                maxsize=maxsize,
            )
            fill = Memoized(
                VolumeRadiusFillFromVolumeComputer(
                    volume_computer=volume,
                    lobe_volume_computer=normalization,
                    l1_computer=l1,
                ),
                maxsize=maxsize,
            )
            selection = TestSelection(fast_lobe_volume=lobe, depth=depth, fill=fill)
    section: SectionPathComputer = (
        GaugeFixedSectionPathComputer(
            radius_computer=radius,
            l1_computer=l1,
            outer_computer=outer,
            watershed_computer=watershed,
        )
        if section_computer is None
        else section_computer
    )
    return ComputerSet(
        watershed=watershed,
        l1=l1,
        outer=outer,
        radius=radius,
        sample=sample,
        volume=volume,
        lobe_slope=lobe_slope,
        area=area,
        mean_curvature=mean_curvature,
        mean_gravity=mean_gravity,
        quadrature=quadrature,
        selection=selection,
        section=section,
        silhouette=GaugeFixedSilhouettePathComputer(
            radius_computer=radius,
            l1_computer=l1,
            outer_computer=outer,
            watershed_computer=watershed,
        ),
    )


@dataclass
class CountedL1:
    calls: int = 0

    def __call__(self, field: RocheField, /) -> rocheometry.CriticalPoint:
        self.calls += 1
        return default_l1_computer()(field)


def numerical_graph(
    *,
    polar_order: int = 32,
    azimuth_order: int = 16,
    volume_radius_fill_tolerance: float = 2e-8,
    maxsize: int | None = 4096,
) -> ComputerSet:
    """Explicit test wiring for fast volume inversion and custom cache controls."""
    return computer_graph(
        polar_order=polar_order,
        azimuth_order=azimuth_order,
        volume_radius_fill_tolerance=volume_radius_fill_tolerance,
        maxsize=maxsize,
    )


def compound_paths() -> tuple[rocheometry.BezierPath, rocheometry.BezierPath]:
    loop = rocheometry.BezierPath(
        (rocheometry.CubicBezier((0, 0), (4, 0), (4, 0), (0, 0)),)
    )
    arc = rocheometry.BezierPath(
        (rocheometry.CubicBezier((-2, 2), (-1, 4), (2, 5), (3, 6)),), closed=False
    )
    return loop, arc


def single_path(compound: rocheometry.CompoundBezierPath) -> rocheometry.BezierPath:
    """Read a known single-piece section, checking its compound topology."""
    assert isinstance(compound, rocheometry.CompoundBezierPath)
    assert len(compound) == 1
    return compound.paths[0]


def gauge_neck_is_uncapped(
    surface: rocheometry.ComponentSurfaceGeometry, plane: rocheometry.CentrelinePlane
) -> bool:
    """Check the explicit cap inequality for the default 26-segment half-gauge."""
    geometry = surface.component
    field, l1 = geometry.field, geometry.l1()
    nu = (
        sqrt(2 * abs(surface._depth) / field.hessian(l1.position[0])[0][0])
        / l1.position[0]
    )
    span = pi
    if surface.regime is rocheometry.LevelRegime.CONTACT:
        rim = geometry._computers.watershed.rim(
            field, surface._depth, plane.inclination
        )
        span -= atan2(hypot(rim[1], rim[2]), rim[0])
    phi1 = span * (1 / 24 - sin(2 * pi / 24) / (2 * pi))
    return sqrt(NECK_SLOT_PLACEMENT_MULTIPLIER * nu * phi1) < phi1 / 2


@dataclass(frozen=True, slots=True)
class WholeSection:
    """Adapt a whole-envelope test producer to the complete section protocol."""

    computer: Callable[
        [RocheField, float, rocheometry.SectionPlane], rocheometry.CompoundBezierPath
    ]

    def __call__(
        self, field: RocheField, potential: float, plane: rocheometry.SectionPlane, /
    ) -> rocheometry.CompoundBezierPath:
        return self.computer(field, potential, plane)

    def envelope(
        self, field: RocheField, potential: float, plane: rocheometry.SectionPlane, /
    ) -> rocheometry.CompoundBezierPath:
        return self.computer(field, potential, plane)


@dataclass(frozen=True, slots=True)
class RadiusDelegate:
    """Give a callable test double both explicit radius entry points."""

    computer: Callable[[RocheField, float, tuple[float, float, float]], float]

    def __call__(
        self, field: RocheField, depth: float, direction: tuple[float, float, float], /
    ) -> float:
        return self.computer(field, depth, direction)

    def node_radius(
        self, field: RocheField, depth: float, direction: tuple[float, float, float], /
    ) -> float:
        return self(field, depth, direction)
