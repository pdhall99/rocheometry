"""Surface selection with consistent lobe normalization and potential inversion."""

from dataclasses import dataclass, field

from rocheometry._computers.composition import Memoized
from rocheometry._computers.contact import default_outer
from rocheometry._computers.critical_point import CriticalPointComputer
from rocheometry._computers.lobe_volume.from_surface import (
    LobeVolumeFromSurfaceComputer,
)
from rocheometry._computers.lobe_volume.protocol import LobeVolumeComputer
from rocheometry._computers.surface_potential import (
    DEFAULT_VOLUME_RADIUS_FILL_TOLERANCE,
    BrentSurfacePotentialComputer,
    SurfacePotentialComputer,
)
from rocheometry._computers.surface_volume import SurfaceVolumeComputer
from rocheometry._computers.volume_radius_fill import VolumeRadiusFillComputer
from rocheometry._computers.volume_radius_fill.from_volume import (
    VolumeRadiusFillFromVolumeComputer,
)


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class VolumeInversionSelection:
    """A volume-based surface selection with shared lobe and potential results.

    The lobe measurement and its inversion use the same volume and L1 producers.
    Construction validates the controls and assembles caches but performs no solve.

    Args:
        volume_computer: Trial equipotential-volume strategy.
        l1_computer: Critical-point strategy defining the lobe and fill `1`.
        volume_radius_fill_tolerance: Finite positive relative radius tolerance.
        maxsize: Non-negative capacity per result cache, or `None` for unbounded
            retention. Zero disables retention.

    Attributes:
        volume_computer: Trial equipotential-volume strategy in separation cubed.
        l1_computer: Critical-point strategy defining the lobe and fill `1`.
        volume_radius_fill_tolerance: Finite positive relative radius tolerance.
        maxsize: Non-negative capacity per result cache, or `None` for unbounded
            retention. Zero disables retention.
        fast_lobe_volume: Memoized lobe-volume computer derived at construction.
        fill: Memoized measured depth-to-fill computer.
        depth: Memoized fill-to-depth computer derived at construction.
    """

    volume_computer: SurfaceVolumeComputer
    l1_computer: CriticalPointComputer
    volume_radius_fill_tolerance: float = DEFAULT_VOLUME_RADIUS_FILL_TOLERANCE
    outer_computer: CriticalPointComputer = field(default_factory=default_outer)
    maxsize: int | None = 4096
    fast_lobe_volume: LobeVolumeComputer = field(init=False)
    depth: SurfacePotentialComputer = field(init=False)
    fill: VolumeRadiusFillComputer = field(init=False)

    def __post_init__(self) -> None:
        lobe_volume = Memoized(
            LobeVolumeFromSurfaceComputer(
                volume_computer=self.volume_computer, l1_computer=self.l1_computer
            ),
            maxsize=self.maxsize,
        )
        depth = Memoized(
            BrentSurfacePotentialComputer(
                volume_computer=self.volume_computer,
                l1_computer=self.l1_computer,
                lobe_volume_computer=lobe_volume,
                outer_computer=self.outer_computer,
                volume_radius_fill_tolerance=self.volume_radius_fill_tolerance,
            ),
            maxsize=self.maxsize,
        )
        object.__setattr__(self, "fast_lobe_volume", lobe_volume)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(
            self,
            "fill",
            Memoized(
                VolumeRadiusFillFromVolumeComputer(
                    volume_computer=self.volume_computer,
                    lobe_volume_computer=lobe_volume,
                    l1_computer=self.l1_computer,
                    outer_computer=self.outer_computer,
                ),
                maxsize=self.maxsize,
            ),
        )
