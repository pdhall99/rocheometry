"""Critical Roche-lobe volume computer protocol."""

from typing import Protocol

from rocheometry._roche.field import RocheField


class LobeVolumeComputer(Protocol):
    """A callable strategy for the volume enclosed by the critical L1 surface."""

    def __call__(self, field: RocheField, /) -> float:
        """Compute the critical lobe volume in separation units cubed.

        Args:
            field: Normalized component-local field.

        Returns:
            Critical lobe volume in separation units cubed.
        """
        ...
