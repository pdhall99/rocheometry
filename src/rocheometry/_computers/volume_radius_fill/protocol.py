"""Contract for selecting volume-radius fill by potential."""

from typing import Protocol

from rocheometry._roche.field import RocheField


class VolumeRadiusFillComputer(Protocol):
    """A surface-to-lobe volume-radius ratio strategy.

    The input is depth below exact L1 on a detached surface or watershed part.
    The L1 potential returns exactly `1.0`.
    """

    def __call__(self, field: RocheField, depth: float, /) -> float:
        """Return the fill of the component-enclosing equipotential.

        Args:
            field: Normalized component-local Roche field.
            depth: Depth below exact L1, no greater than the outer-opening gap;
                L2 itself measures the non-selectable maximum fill.
        """
        ...
