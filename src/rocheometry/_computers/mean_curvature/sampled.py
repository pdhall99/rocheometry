"""Integrated mean curvature read from a shared quadrature sample."""

from dataclasses import dataclass

from rocheometry._computers.checks import require_positive
from rocheometry._computers.surface_sample.protocol import SurfaceSampler
from rocheometry._computers.surface_sample.terms import ray_terms
from rocheometry._roche.field import RocheField


@dataclass(frozen=True, slots=True)
class SampledMeanCurvatureComputer:
    """Integrate `H*dA`, where `H` is the mean of principal curvatures.

    A sphere of radius `R` has `H = 1/R` and integral `4*pi*R`.

    Quadrant weights include the fourfold y/z reflection factor.

    Attributes:
        sample: Shared quadrature sampler for the selected equipotential.
    """

    sample: SurfaceSampler

    def __post_init__(self) -> None:
        if not callable(self.sample):
            raise TypeError("sample: expected a callable computer")

    def __call__(self, field: RocheField, depth: float, /) -> float:
        """Return integrated mean curvature in separation units.

        Args:
            field: Normalized component-local field.
            depth: Depth below exact L1; zero is critical and positive is contact.
        """
        sample = self.sample(field, depth)
        total = 0.0
        for (direction, weight), radius in zip(sample.rays, sample.radii, strict=True):
            total += weight * ray_terms(field, direction, radius, sample=sample)[1]
        return require_positive(total)
