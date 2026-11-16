"""Surface-potential selection by bracketed volume inversion."""

from __future__ import annotations

import reprlib
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from math import nextafter
from typing import TYPE_CHECKING

from rocheometry._computers.contact import admissibility_depth, default_outer
from rocheometry._numerics import brent_root
from rocheometry._numerics.roots import _adjacent_float_root
from rocheometry._roche.field import RocheField
from rocheometry._roche.volume import volume_radius_from_volume
from rocheometry._validation import normalize_volume_radius_fill_tolerance, require_int

if TYPE_CHECKING:
    from rocheometry._computers.critical_point import CriticalPointComputer
    from rocheometry._computers.lobe_volume.protocol import LobeVolumeComputer
    from rocheometry._computers.surface_volume.protocol import SurfaceVolumeComputer

# Resolution budget: default volume radius fill tolerance.
DEFAULT_VOLUME_RADIUS_FILL_TOLERANCE = 2.0e-8
"""The relative volume-radius residual accepted during fill inversion."""
# Work guard or cache capacity: fill inversion max iterations.
FILL_INVERSION_MAX_ITERATIONS = 80
# Work guard or cache capacity: fill inversion max bracket expansions.
FILL_INVERSION_MAX_BRACKET_EXPANSIONS = 128


@dataclass(frozen=True, slots=True, kw_only=True)
class BisectionSurfacePotentialComputer:
    """A bisection strategy for selecting surfaces by enclosed volume.

    Accept the canonical Jacobi level when the recovered volume radius differs
    from the target by at most the configured relative fill tolerance. The L1
    level is the lower bracket and is returned directly at fill `1`.
    Dependencies must use compatible lobe normalization and field conventions.

    Attributes:
        volume_computer: Trial equipotential-volume computer in separation cubed.
        l1_computer: Critical point defining fill `1` and the lower potential.
        lobe_volume_computer: Lobe volume defining the target radius scale.
        volume_radius_fill_tolerance: Finite positive relative radius tolerance.
        max_iterations: Positive maximum number of root-solve iterations.
        max_bracket_expansions: Positive maximum number of upper-bracket expansions.
    """

    volume_computer: SurfaceVolumeComputer
    l1_computer: CriticalPointComputer
    lobe_volume_computer: LobeVolumeComputer
    volume_radius_fill_tolerance: float = DEFAULT_VOLUME_RADIUS_FILL_TOLERANCE
    outer_computer: CriticalPointComputer = dataclass_field(
        default_factory=default_outer
    )
    max_iterations: int = FILL_INVERSION_MAX_ITERATIONS
    max_bracket_expansions: int = FILL_INVERSION_MAX_BRACKET_EXPANSIONS

    def __post_init__(self) -> None:
        for name in ("volume_computer", "l1_computer", "lobe_volume_computer"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name}: expected a callable dependency")
        object.__setattr__(
            self,
            "volume_radius_fill_tolerance",
            normalize_volume_radius_fill_tolerance(self.volume_radius_fill_tolerance),
        )
        for name, value in (
            ("max_iterations", self.max_iterations),
            ("max_bracket_expansions", self.max_bracket_expansions),
        ):
            value = require_int(value, name=name)
            if value < 1:
                raise ValueError(
                    f"{name}: expected a positive integer, got {reprlib.repr(value)}"
                )

    def __call__(self, field: RocheField, volume_radius_fill: float, /) -> float:
        """Return depth at the requested fill, with zero exactly at fill one."""
        return self._select_depth(field, volume_radius_fill, use_brent=False)

    def _select_depth(
        self, field: RocheField, fill: float, *, use_brent: bool
    ) -> float:
        if fill == 1.0:
            return 0.0
        l1 = self.l1_computer(field)
        lobe = self.lobe_volume_computer(field)
        target = volume_radius_from_volume(lobe) * fill
        ratios = {0.0: 1.0 / fill}

        def ratio(depth: float) -> float:
            if depth not in ratios:
                value = (
                    volume_radius_from_volume(self.volume_computer(field, depth))
                    / target
                )
                if not value > 0:
                    raise RuntimeError(
                        "surface volume has no positive representable radius"
                    )
                ratios[depth] = value
            return ratios[depth]

        excluded = None
        if fill > 1:
            lower = 0.0
            upper = admissibility_depth(field, l1, self.outer_computer)
            maximum = (self.volume_computer(field, upper) / lobe) ** (1 / 3)
            if fill >= maximum:
                raise ValueError(
                    f"volume_radius_fill {fill!r} must be below maximum fill {maximum!r}"
                )
            ratios[upper] = maximum / fill
            excluded = upper
        else:
            upper = 0.0
            step = 0.125 * abs(l1.potential.value)  # Per-field bracket scale.
            lower = -step
            for _ in range(self.max_bracket_expansions):
                if ratio(lower) <= 1:
                    break
                upper = lower
                step *= 2
                lower = -step
            else:
                raise RuntimeError(
                    "could not bracket component-enclosing equipotential depth"
                )
        bracket_lower, bracket_upper = lower, upper

        def residual(depth: float) -> float:
            nonlocal bracket_lower, bracket_upper
            r = ratio(depth)
            if r > 1:
                bracket_upper = min(bracket_upper, depth)
            else:
                bracket_lower = max(bracket_lower, depth)
            return 1 / r - 1

        if use_brent:
            root = brent_root(
                residual,
                lower,
                upper,
                absolute_tolerance=0.0,
                residual_tolerance=self.volume_radius_fill_tolerance
                / (1 + self.volume_radius_fill_tolerance),
                max_iterations=self.max_iterations,
            )
            if (
                root != excluded
                and abs(ratio(root) - 1) <= self.volume_radius_fill_tolerance
            ):
                return root
            lower, upper = bracket_lower, bracket_upper
        else:
            for _ in range(self.max_iterations):
                if nextafter(lower, upper) == upper:
                    break
                root = lower + (upper - lower) / 2
                r = ratio(root)
                if root != excluded and abs(r - 1) <= self.volume_radius_fill_tolerance:
                    return root
                if r > 1:
                    upper = root
                else:
                    lower = root
            else:
                raise RuntimeError("component volume-fill solve did not converge")
        return self._representable_depth(
            field, target, lower, upper, excluded_upper=excluded
        )

    def _representable_depth(
        self,
        field: RocheField,
        target: float,
        lower: float,
        upper: float,
        *,
        excluded_upper: float | None = None,
    ) -> float:
        """Finish a depth bracket at its representable resolution."""

        def residual(value: float) -> float:
            return (
                volume_radius_from_volume(self.volume_computer(field, value)) / target
                - 1
            )

        f_lower, f_upper = residual(lower), residual(upper)
        for _ in range(self.max_iterations):
            if nextafter(lower, upper) == upper:
                chosen = _adjacent_float_root(
                    lower, upper, f_lower, f_upper, self.volume_radius_fill_tolerance
                )
                if chosen is None:
                    break
                if chosen == excluded_upper:
                    # The opening depth is a measurement endpoint, never a selectable part.
                    chosen = lower
                    if abs(f_lower) > max(
                        self.volume_radius_fill_tolerance, abs(f_upper - f_lower)
                    ):
                        break
                return chosen
            midpoint = lower + (upper - lower) / 2
            error = residual(midpoint)
            if error > 0:
                upper, f_upper = midpoint, error
            else:
                lower, f_lower = midpoint, error
        raise RuntimeError("component volume-fill solve did not converge")
