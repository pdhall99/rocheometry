"""Analytic lobe-radius slopes read from a shared critical-surface sample."""

from dataclasses import dataclass

from rocheometry._computers.checks import require_positive
from rocheometry._computers.critical_point import CriticalPointComputer
from rocheometry._computers.surface_sample import SurfaceSampler
from rocheometry._roche.field import RocheField
from rocheometry._roche.shape_derivative import ray_log_qi_velocity


@dataclass(frozen=True, slots=True)
class SampledLobeSlopeComputer:
    """Integrate the mass-ratio shape derivative on the critical equipotential.

    The envelope theorem gives the level slope from `partial U/partial ln(qi)`
    at L1, since the spatial gradient vanishes there. The shared critical
    sample supplies `V` and its derivative `V_dot`; the result is `V_dot/(3*V)`.
    Quadrant weights include the fourfold y/z reflection factor.

    Attributes:
        sample: Shared quadrature sampler for the critical surface.
        l1_computer: Critical-point strategy defining the lobe level.
    """

    sample: SurfaceSampler
    l1_computer: CriticalPointComputer

    def __post_init__(self) -> None:
        for name in ("sample", "l1_computer"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name}: expected a callable computer")

    def __call__(self, field: RocheField, /) -> float:
        """Return the dimensionless `d ln(R_L)/d ln(qi)` of the critical lobe."""
        l1 = self.l1_computer(field)
        level_slope = field.log_qi_derivative(*l1.position)
        sample = self.sample(field, 0.0)
        total = 0.0
        volume_slope = 0.0
        for (direction, weight), radius in zip(sample.rays, sample.radii, strict=True):
            total += weight * radius**3
            volume_slope += (
                weight
                * radius**2
                * ray_log_qi_velocity(field, level_slope, direction, radius)
            )
        volume = total / 3.0
        return require_positive(volume_slope / (3.0 * volume))
