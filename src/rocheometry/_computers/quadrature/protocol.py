"""Protocol for interchangeable surface quadrature producers."""

from typing import Protocol

from rocheometry._roche.field import RocheField
from rocheometry._surface_quadrature import SurfaceQuadrature


class SurfaceQuadratureComputer(Protocol):
    """Produce full-surface nodes and area weights in component-local units.

    Contact parts cover only the envelope, excluding the watershed cut.
    """

    def __call__(self, field: RocheField, depth: float, /) -> SurfaceQuadrature:
        """Return the selected surface's quadrature and differential geometry."""
        ...
