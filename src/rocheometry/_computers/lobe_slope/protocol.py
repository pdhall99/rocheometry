"""Protocol for logarithmic Roche-lobe volume-radius slopes."""

from typing import Protocol

from rocheometry._roche.field import RocheField


class LobeVolumeRadiusSlopeComputer(Protocol):
    """A callable computing `d ln(R_L)/d ln(qi)` for the critical lobe.

    `R_L` is the volume-equivalent lobe radius and
    `qi = M_component/M_companion`. The result is dimensionless.
    """

    def __call__(self, field: RocheField, /) -> float:
        """Return the lobe volume-radius slope for the supplied field."""
        ...
