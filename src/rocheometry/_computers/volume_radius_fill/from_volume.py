"""Volume-radius fill from consistent surface and lobe volumes."""

from dataclasses import dataclass
from dataclasses import field as dataclass_field

from rocheometry._computers.contact import default_outer, require_part_depth
from rocheometry._computers.critical_point import CriticalPointComputer
from rocheometry._computers.lobe_volume import LobeVolumeComputer
from rocheometry._computers.surface_volume import SurfaceVolumeComputer
from rocheometry._roche.field import RocheField


@dataclass(frozen=True, slots=True, kw_only=True)
class VolumeRadiusFillFromVolumeComputer:
    """Measure fill at a depth using the selection's sampled-lobe normalization.

    Attributes:
        volume_computer: Enclosed volume at the supplied depth below L1.
        lobe_volume_computer: Lobe volume defining the fill normalization.
        l1_computer: Critical point defining the exact fill-one endpoint.
    """

    volume_computer: SurfaceVolumeComputer
    lobe_volume_computer: LobeVolumeComputer
    l1_computer: CriticalPointComputer

    outer_computer: CriticalPointComputer = dataclass_field(
        default_factory=default_outer
    )

    def __call__(
        self,
        field: RocheField,
        depth: float,
        /,
    ) -> float:
        """Return the cube-root volume ratio, exactly one at L1."""
        l1 = self.l1_computer(field)
        depth = require_part_depth(field, depth, l1, self.outer_computer)
        if depth == 0:
            return 1.0
        return (
            self.volume_computer(field, depth) / self.lobe_volume_computer(field)
        ) ** (1.0 / 3.0)
