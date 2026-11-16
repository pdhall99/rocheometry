"""Memoized fitted selection with a consistent whole-selection fallback."""

from dataclasses import dataclass, field

from rocheometry._computers.composition import Memoized
from rocheometry._computers.critical_point import CriticalPointComputer
from rocheometry._computers.fitted import FITTED_FILL_ABSOLUTE_TOLERANCE
from rocheometry._computers.lobe_volume import (
    LobeVolumeComputer,
    LobeVolumeFromRadiusComputer,
)
from rocheometry._computers.lobe_volume_radius.protocol import LobeVolumeRadiusComputer
from rocheometry._computers.packaged import packaged_lobe_volume_radius_computer
from rocheometry._computers.surface_potential.fitted.computer import (
    FittedSurfacePotentialComputer,
    SurfacePotentialMap,
)
from rocheometry._computers.surface_potential.protocol import SurfacePotentialComputer
from rocheometry._computers.surface_selection.protocol import SurfaceSelection
from rocheometry._computers.volume_radius_fill import VolumeRadiusFillComputer
from rocheometry._roche.field import RocheField


def _default_l1() -> CriticalPointComputer:
    from rocheometry._computers.defaults import default_l1_computer

    return default_l1_computer()


@dataclass(frozen=True, slots=True)
class _LobeVolume:
    fitted: LobeVolumeComputer
    fallback: LobeVolumeComputer
    domain: FittedSurfacePotentialComputer

    def __call__(self, field: RocheField, /) -> float:
        return (self.fitted if self.domain.supports(field) else self.fallback)(field)


@dataclass(frozen=True, slots=True)
class _Depth:
    fitted: FittedSurfacePotentialComputer
    fallback: SurfacePotentialComputer

    def __call__(self, field: RocheField, fill: float, /) -> float:
        return (
            self.fitted if fill <= 1 and self.fitted.supports(field) else self.fallback
        )(field, fill)


@dataclass(frozen=True, slots=True, eq=False)
class FittedSelection:
    """Fast forward selection with measured, sampled-lobe-normalized reporting.

    Each of the three computers is independently memoized.
    Construction loads no artefacts and performs no solve. Fitted endpoints use
    L1 and packaged radius calibration. Fills above one use numerical inversion.
    Reported fill always uses the fallback's measured surface and lobe volumes.

    Attributes:
        fallback: Self-consistent selection used outside either fitted domain.
        l1_computer: Critical point defining the exact fill-one endpoint.
        lobe_volume_radius_computer: Packaged radius calibration by default.
        model: Explicit map, or None for lazy packaged coefficients.
        maxsize: Capacity per result cache; None is unbounded, zero disables it.
        fast_lobe_volume: Memoized fitted or fallback lobe volume.
        depth: Memoized fitted or fallback fill-to-depth conversion.
        fill: Memoized measured fill from the numerical selection.
        volume_radius_fill_tolerance: Maximum absolute error in requested fill,
            not an inversion tolerance.
    """

    fallback: SurfaceSelection
    l1_computer: CriticalPointComputer = field(
        default_factory=_default_l1, kw_only=True
    )
    lobe_volume_radius_computer: LobeVolumeRadiusComputer = field(
        default_factory=packaged_lobe_volume_radius_computer, kw_only=True
    )
    model: SurfacePotentialMap | None = field(default=None, kw_only=True, repr=False)
    maxsize: int | None = field(default=4096, kw_only=True)
    fast_lobe_volume: LobeVolumeComputer = field(init=False)
    depth: SurfacePotentialComputer = field(init=False)
    fill: VolumeRadiusFillComputer = field(init=False)
    volume_radius_fill_tolerance: float = field(
        default=FITTED_FILL_ABSOLUTE_TOLERANCE, init=False
    )

    def __post_init__(self) -> None:
        for name in ("fast_lobe_volume", "depth", "fill"):
            if not callable(getattr(self.fallback, name, None)):
                raise TypeError(f"fallback.{name}: expected a callable computer")
        forward = FittedSurfacePotentialComputer(
            l1_computer=self.l1_computer,
            lobe_volume_radius_computer=self.lobe_volume_radius_computer,
            model=self.model,
        )
        lobe = LobeVolumeFromRadiusComputer(self.lobe_volume_radius_computer)
        object.__setattr__(
            self,
            "fast_lobe_volume",
            Memoized(
                _LobeVolume(lobe, self.fallback.fast_lobe_volume, forward),
                maxsize=self.maxsize,
            ),
        )
        object.__setattr__(
            self,
            "depth",
            Memoized(_Depth(forward, self.fallback.depth), maxsize=self.maxsize),
        )
        object.__setattr__(
            self,
            "fill",
            Memoized(self.fallback.fill, maxsize=self.maxsize),
        )
