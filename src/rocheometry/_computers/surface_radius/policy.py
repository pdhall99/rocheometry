"""Choose one residual for an entire fitted surface, independently of its rays."""

from dataclasses import dataclass
from sys import float_info

from rocheometry._computers.critical_point import CriticalPointComputer
from rocheometry._computers.surface_volume import SurfaceVolumeComputer
from rocheometry._roche.field import RocheField
from rocheometry._roche.volume import volume_radius_from_volume


@dataclass(frozen=True, slots=True)
class _UseExactAnchor:
    """Apply 4*eps*abs(u)*R_v/m > 1e-12, with cheap certified shortcuts.

    The enclosing sphere bounds R_v from above. For a sphere of radius r < 1,
    the companion-plus-centrifugal potential is at least its central value
    minus c*r²/(2*(1-r)³): the linear terms cancel and the Hessian is bounded
    below by -c/(1-r)³. If that sphere is inside the surface, it bounds R_v
    from below. Only the transition band needs a volume solve using the existing
    contact/front-ray factoring rather than the all-ray extension.
    Memoize this producer by (field, level), so every ray uses one decision.
    """

    l1_computer: CriticalPointComputer
    outer_computer: CriticalPointComputer
    volume_computer: SurfaceVolumeComputer

    def __call__(self, field: RocheField, depth: float, /) -> bool:
        l1 = self.l1_computer(field)
        m, c, u = (
            field.component_mass,
            field.companion_mass,
            field._level_value(l1.position[0], depth),
        )
        threshold = m * (1e-12 / (4 * float_info.epsilon * abs(u)))
        if threshold == 0.0:
            return True
        l1 = self.l1_computer(field)
        bound = max(l1.position[0], abs(self.outer_computer(field).position[0]))
        if threshold >= bound:
            return False
        if threshold < min(1.0, l1.position[0]):
            lower = (
                m / threshold
                + c
                + c * c / 2
                - c * threshold**2 / (2 * (1 - threshold) ** 3)
            )
            if lower > u:
                return True
        radius = volume_radius_from_volume(self.volume_computer(field, depth))
        return radius > threshold
