"""Volume-equivalent radius adapters for lobe-volume computations."""

from dataclasses import dataclass

from rocheometry._computers.checks import require_positive
from rocheometry._computers.lobe_volume_radius.protocol import LobeVolumeRadiusComputer
from rocheometry._roche.field import RocheField
from rocheometry._roche.volume import sphere_volume


@dataclass(frozen=True, slots=True)
class LobeVolumeFromRadiusComputer:
    """Compute lobe volume from a supplied volume-equivalent radius computer.

    Attributes:
        radius_computer: Lobe volume-radius strategy in separation units; its
            approximation is inherited.
    """

    radius_computer: LobeVolumeRadiusComputer

    def __post_init__(self) -> None:
        if not callable(self.radius_computer):
            raise TypeError("radius_computer: expected a callable computer")

    def __call__(self, field: RocheField, /) -> float:
        """Return the volume enclosed by the critical L1 surface."""
        return require_positive(sphere_volume(self.radius_computer(field)))
