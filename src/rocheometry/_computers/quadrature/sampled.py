"""Public surface quadrature read from a shared sample of solved radii."""

from dataclasses import dataclass

from rocheometry._computers.surface_sample.protocol import SurfaceSampler
from rocheometry._roche.field import RocheField
from rocheometry._surface_quadrature import SurfaceQuadrature, _quadrature_from_sample


@dataclass(frozen=True, slots=True)
class SampledQuadratureComputer:
    """Unfold shared quadrant nodes with exact local normals and curvatures.

    Node order and units follow SurfaceQuadrature. Its validation runs when
    constructing the result; the graph's quadrature node retains that value.

    Attributes:
        sample: Shared quadrature sampler for the selected equipotential.
    """

    sample: SurfaceSampler

    def __post_init__(self) -> None:
        if not callable(self.sample):
            raise TypeError("sample: expected a callable computer")

    def __call__(self, field: RocheField, depth: float, /) -> SurfaceQuadrature:
        """Return the full surface quadrature in intrinsic component units."""
        return _quadrature_from_sample(field, self.sample(field, depth))
