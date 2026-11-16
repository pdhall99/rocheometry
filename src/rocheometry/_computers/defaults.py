"""One private wiring function, cached by immutable accuracy settings."""

from dataclasses import dataclass
from functools import cache, lru_cache

from rocheometry._accuracy import _Accuracy
from rocheometry._computers.composition import ComputerSet, Memoized
from rocheometry._computers.critical_point import (
    BisectionCriticalPointComputer,
    CriticalPointComputer,
)
from rocheometry._computers.lobe_slope import SampledLobeSlopeComputer
from rocheometry._computers.lobe_slope.fitted import FittedLobeSlopeComputer
from rocheometry._computers.mean_curvature import SampledMeanCurvatureComputer
from rocheometry._computers.mean_gravity import SampledMeanGravityComputer
from rocheometry._computers.quadrature import SampledQuadratureComputer
from rocheometry._computers.section_path.gauge_fixed import (
    GaugeFixedSectionPathComputer,
)
from rocheometry._computers.section_path.routed import PlaneRoutedSectionPathComputer
from rocheometry._computers.section_path.traced import TracedSectionPathComputer
from rocheometry._computers.silhouette.paths import (
    GaugeFixedSilhouettePathComputer,
    TracedSilhouettePathComputer,
)
from rocheometry._computers.surface_area import SampledAreaComputer
from rocheometry._computers.surface_potential import (
    DEFAULT_VOLUME_RADIUS_FILL_TOLERANCE,
)
from rocheometry._computers.surface_radius import BrentSurfaceRadiusComputer
from rocheometry._computers.surface_radius.policy import _UseExactAnchor
from rocheometry._computers.surface_sample import (
    DEFAULT_FITTED_AZIMUTH_ORDER,
    DEFAULT_FITTED_POLAR_ORDER,
    GaussLegendreSurfaceSampler,
)
from rocheometry._computers.surface_selection import (
    FittedSelection,
    VolumeInversionSelection,
)
from rocheometry._computers.surface_volume import SampledVolumeComputer
from rocheometry._computers.watershed import TracedWatershedComputer
from rocheometry._validation import (
    _normalize_finite_positive_float,
    normalize_volume_radius_fill_tolerance,
    require_instance,
    require_quadrature_order,
)

# Resolution budget: default reference polar order.
DEFAULT_REFERENCE_POLAR_ORDER = 128
# Resolution budget: default reference azimuth order.
DEFAULT_REFERENCE_AZIMUTH_ORDER = 64
# Resolution budget: default reference fill tolerance.
DEFAULT_REFERENCE_FILL_TOLERANCE = 1e-13
# Resolution budget: reference section tolerance.
REFERENCE_SECTION_TOLERANCE = 1e-10


@cache
def default_l1_computer() -> CriticalPointComputer:
    """Return the package-default L1 computer."""
    return BisectionCriticalPointComputer.l1()


@cache
def default_outer_computer() -> CriticalPointComputer:
    """Return the package-default outer critical-point computer."""
    return BisectionCriticalPointComputer.outer()


@dataclass(frozen=True, slots=True, kw_only=True)
class _Settings(_Accuracy):
    """Unsupported internal controls used by accuracy values and build scripts."""

    polar_order: int = DEFAULT_FITTED_POLAR_ORDER
    azimuth_order: int = DEFAULT_FITTED_AZIMUTH_ORDER
    watershed_tolerance: float = 1e-11
    fitted_selection: bool = True
    fill_tolerance: float = DEFAULT_VOLUME_RADIUS_FILL_TOLERANCE
    fixed_paths: bool = True
    path_tolerance: float = 1e-5
    # Fitted factors cusp/contact residuals and extends factoring at a noise
    # floor; reference extends everywhere. This preserves fitted sampling speed.
    exact_anchor_everywhere: bool = False

    def __post_init__(self) -> None:
        for name in ("polar_order", "azimuth_order"):
            object.__setattr__(
                self, name, require_quadrature_order(getattr(self, name), name=name)
            )
        for name in ("watershed_tolerance", "path_tolerance"):
            object.__setattr__(
                self,
                name,
                _normalize_finite_positive_float(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "fill_tolerance",
            normalize_volume_radius_fill_tolerance(self.fill_tolerance),
        )
        for name in ("fitted_selection", "fixed_paths", "exact_anchor_everywhere"):
            require_instance(getattr(self, name), bool, name=name)

    def _settings(self) -> "_Settings":  # type: ignore[missing-override-decorator]
        return self


def _build_computers(settings: _Settings) -> ComputerSet:
    """Resolve one graph per settings value, with bounded reference retention."""
    return (_cached_fitted if settings.fitted_selection else _cached_reference)(
        settings
    )


def _wire_computers(settings: _Settings) -> ComputerSet:
    l1 = Memoized(default_l1_computer())
    outer = Memoized(default_outer_computer())
    watershed = TracedWatershedComputer(
        l1_computer=l1, outer_computer=outer, tolerance=settings.watershed_tolerance
    )
    unfactored = BrentSurfaceRadiusComputer(
        l1_computer=l1,
        outer_computer=outer,
        watershed_computer=watershed,
        exact_anchor=False,
    )
    anchor = (
        True
        if settings.exact_anchor_everywhere
        else Memoized(
            _UseExactAnchor(
                l1,
                outer,
                SampledVolumeComputer(
                    sample=GaussLegendreSurfaceSampler(
                        radius_computer=unfactored,
                        l1_computer=l1,
                        outer_computer=outer,
                        watershed_computer=watershed,
                        polar_order=settings.polar_order,
                        azimuth_order=settings.azimuth_order,
                    )
                ),
            )
        )
    )
    radius = BrentSurfaceRadiusComputer(
        l1_computer=l1,
        outer_computer=outer,
        watershed_computer=watershed,
        exact_anchor=anchor,
    )
    sample = Memoized(
        GaussLegendreSurfaceSampler(
            radius_computer=radius,
            l1_computer=l1,
            outer_computer=outer,
            watershed_computer=watershed,
            polar_order=settings.polar_order,
            azimuth_order=settings.azimuth_order,
        ),
        maxsize=16,
    )
    sampled_slope = SampledLobeSlopeComputer(sample=sample, l1_computer=l1)
    lobe_slope = Memoized(
        FittedLobeSlopeComputer(sampled_slope)
        if settings.fitted_selection
        else sampled_slope
    )
    volume = Memoized(SampledVolumeComputer(sample=sample))
    area = Memoized(SampledAreaComputer(sample=sample))
    mean_curvature = Memoized(SampledMeanCurvatureComputer(sample=sample))
    mean_gravity = Memoized(SampledMeanGravityComputer(sample=sample))
    quadrature = Memoized(SampledQuadratureComputer(sample=sample), maxsize=4)
    inversion = VolumeInversionSelection(
        volume_computer=volume,
        l1_computer=l1,
        outer_computer=outer,
        volume_radius_fill_tolerance=settings.fill_tolerance,
    )
    selection = (
        FittedSelection(inversion, l1_computer=l1)
        if settings.fitted_selection
        else inversion
    )
    traced = TracedSectionPathComputer(
        radius_computer=radius,
        l1_computer=l1,
        outer_computer=outer,
        watershed_computer=watershed,
        tolerance=settings.path_tolerance,
    )
    section = (
        PlaneRoutedSectionPathComputer(
            l1_computer=l1,
            outer_computer=outer,
            centreline=GaugeFixedSectionPathComputer(
                radius_computer=radius,
                l1_computer=l1,
                outer_computer=outer,
                watershed_computer=watershed,
            ),
            section=traced,
        )
        if settings.fixed_paths
        else traced
    )
    silhouette = (
        GaugeFixedSilhouettePathComputer(
            radius_computer=radius,
            l1_computer=l1,
            outer_computer=outer,
            watershed_computer=watershed,
        )
        if settings.fixed_paths
        else TracedSilhouettePathComputer(
            radius_computer=radius,
            l1_computer=l1,
            outer_computer=outer,
            watershed_computer=watershed,
            tolerance=settings.path_tolerance,
        )
    )
    return ComputerSet(
        l1=l1,
        outer=outer,
        watershed=watershed,
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
        silhouette=silhouette,
    )


_cached_fitted = cache(_wire_computers)
_cached_reference = lru_cache(maxsize=8)(_wire_computers)
