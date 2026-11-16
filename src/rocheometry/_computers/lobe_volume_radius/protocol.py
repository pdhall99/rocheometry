"""Critical Roche-lobe volume-radius computer protocol."""

from typing import Protocol

from rocheometry._roche.field import RocheField


class LobeVolumeRadiusComputer(Protocol):
    """A callable strategy for the critical lobe's volume-equivalent radius."""

    def __call__(self, field: RocheField, /) -> float:
        """Compute the critical lobe volume radius in separation units.

        Args:
            field: Normalized component-local field.

        Returns:
            The radius of a sphere with the lobe volume, in separation units.
        """
        ...
