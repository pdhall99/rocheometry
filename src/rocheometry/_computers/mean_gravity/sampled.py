"""Mean effective gravity read from a shared quadrature sample."""

from dataclasses import dataclass
from math import fsum

from rocheometry._computers.checks import require_positive
from rocheometry._computers.surface_sample.protocol import SurfaceSampler
from rocheometry._computers.surface_sample.terms import ray_geometry
from rocheometry._roche.field import RocheField


@dataclass(frozen=True, slots=True)
class SampledMeanGravityComputer:
    """Compute area-weighted mean gravity from the shared sample.

    Quadrant weights include the fourfold y/z reflection factor. Contact parts
    cover the envelope share only, excluding the watershed cut.

    Attributes:
        sample: Shared quadrature sampler for the selected equipotential.
    """

    sample: SurfaceSampler

    def __post_init__(self) -> None:
        if not callable(self.sample):
            raise TypeError("sample: expected a callable computer")

    def __call__(self, field: RocheField, depth: float, /) -> float:
        """Return area-weighted mean gravity in G*(M1+M2)/separation**2 units."""
        sample = self.sample(field, depth)
        weights: list[float] = []
        fluxes: list[float] = []
        for (direction, weight), radius in zip(sample.rays, sample.radii, strict=True):
            local = ray_geometry(field, direction, radius, sample=sample)
            area_weight = weight * local.area
            weights.append(area_weight)
            fluxes.append(area_weight * local.gravity)
        return require_positive(fsum(fluxes) / require_positive(fsum(weights)))
