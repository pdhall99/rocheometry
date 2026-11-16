"""Interchangeable projected-outline strategies."""

from typing import Protocol

from rocheometry._bezier import CompoundBezierPath
from rocheometry._planes import CentrelinePlane
from rocheometry._roche.field import RocheField


class SilhouettePathComputer(Protocol):
    """Produce projected silhouettes at quadrature."""

    def __call__(
        self, field: RocheField, depth: float, plane: CentrelinePlane, /
    ) -> CompoundBezierPath:
        """Return a closed detached or critical outline; reject contact levels."""
        ...

    def envelope(
        self, field: RocheField, depth: float, plane: CentrelinePlane, /
    ) -> CompoundBezierPath:
        """Return envelope limbs, open at the watershed for contact parts."""
        ...
