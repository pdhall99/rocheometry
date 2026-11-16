"""Protocol for interchangeable mean effective gravity computations."""

from typing import Protocol

from rocheometry._roche.field import RocheField


class MeanGravityComputer(Protocol):
    """Compute area-weighted mean gravity in G*(M1+M2)/separation**2 units.

    Contact parts include only the envelope, excluding the watershed cut.
    """

    def __call__(self, field: RocheField, depth: float, /) -> float:
        """Return mean effective gravity on the selected component surface."""
        ...
