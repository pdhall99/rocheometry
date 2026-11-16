"""Critical lobe-volume measurements through surface integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rocheometry._computers.checks import require_positive

if TYPE_CHECKING:
    from rocheometry._computers.critical_point import CriticalPointComputer
    from rocheometry._computers.surface_volume import SurfaceVolumeComputer
    from rocheometry._roche.field import RocheField


@dataclass(frozen=True, slots=True, kw_only=True)
class LobeVolumeFromSurfaceComputer:
    """Measure the critical L1 surface with the supplied volume computer.

    Attributes:
        volume_computer: Strategy measuring the critical surface in separation units
            cubed.
        l1_computer: Critical-point strategy supplying the lobe potential.
    """

    volume_computer: SurfaceVolumeComputer
    l1_computer: CriticalPointComputer

    def __post_init__(self) -> None:
        for name in ("volume_computer", "l1_computer"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name}: expected a callable computer")

    def __call__(self, field: RocheField, /) -> float:
        """Return the volume enclosed by the critical L1 surface."""
        return require_positive(self.volume_computer(field, 0.0))
