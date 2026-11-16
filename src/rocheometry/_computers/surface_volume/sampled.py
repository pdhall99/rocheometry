"""Enclosed volume read from a shared quadrature sample."""

from dataclasses import dataclass
from math import fsum, pi

from rocheometry._computers.checks import require_positive
from rocheometry._computers.surface_sample.protocol import SurfaceSampler
from rocheometry._computers.surface_sample.terms import ray_gravity_flux
from rocheometry._roche.field import RocheField


@dataclass(frozen=True, slots=True)
class SampledVolumeComputer:
    """Integrate closed-surface rays, or recover contact part volume from flux.

    For contact samples, Gauss's law excludes the flux-free watershed cut.

    Attributes:
        sample: Quadrature sampler for the selected component-enclosing surface.
    """

    sample: SurfaceSampler

    def __post_init__(self) -> None:
        if not callable(self.sample):
            raise TypeError("sample: expected a callable computer")

    def __call__(self, field: RocheField, depth: float, /) -> float:
        """Return the enclosed volume in separation units cubed.

        Args:
            field: Normalized component-local field.
            depth: Depth below exact L1, bounded by the outer-opening gap.
        """
        sample = self.sample(field, depth)
        if sample.part:
            flux = fsum(
                weight * ray_gravity_flux(field, direction, radius, sample=sample)
                for (direction, weight), radius in zip(
                    sample.rays, sample.radii, strict=True
                )
            )
            return require_positive((4 * pi * field.component_mass - flux) / 2)
        total = 0.0
        for (_, weight), radius in zip(sample.rays, sample.radii, strict=True):
            total += weight * radius**3
        return require_positive(total / 3.0)
