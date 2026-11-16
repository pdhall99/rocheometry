"""Typed public Roche potential conventions and exact conversions."""

from __future__ import annotations

import reprlib
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from math import hypot
from typing import cast

from rocheometry._geometry import Point3D
from rocheometry._roche.mass import mass_fractions
from rocheometry._types import ComponentId
from rocheometry._validation import (
    _normalize_finite_float,
    normalize_mass_ratio,
    normalize_point3d,
    require_component_id,
    require_instance,
)


class PotentialConvention(StrEnum):
    """Supported dimensionless Roche potential conventions."""

    JACOBI = "jacobi"
    """The positive, total-mass-normalized centre-of-mass-gauge potential."""
    KOPAL = "kopal"
    """The component-centred potential normalized by the selected component mass."""


@dataclass(frozen=True, slots=True)
class Potential:
    """A typed dimensionless Roche potential with convention metadata.

    Jacobi values are total-mass normalized and component independent. Kopal
    values are component-centred and normalized by the identified component's mass.

    Args:
        value: Finite potential level; real numeric inputs become floats.
        convention: Supported convention or its serialized string identifier.
        component_id: Required component number (`1` or `2`) for Kopal; must
            be `None` for Jacobi.

    Attributes:
        value: Normalized finite potential level.
        convention: Resolved potential convention.
        component_id: Kopal component number, or `None` for Jacobi.
    """

    value: float
    convention: PotentialConvention
    component_id: ComponentId | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "value",
            _normalize_finite_float(self.value, name="potential.value"),
        )
        try:
            convention = PotentialConvention(self.convention)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "potential.convention: expected a supported PotentialConvention, "
                f"got {reprlib.repr(self.convention)}"
            ) from exc
        object.__setattr__(self, "convention", convention)

        if convention is PotentialConvention.JACOBI:
            if self.component_id is not None:
                raise ValueError(
                    "potential.component_id: canonical Jacobi potential is "
                    "component-independent"
                )
            return

        if self.component_id is None:
            raise ValueError(
                "potential.component_id: component-centred Kopal potential requires "
                "component_id=1 or component_id=2"
            )
        object.__setattr__(
            self,
            "component_id",
            require_component_id(self.component_id, name="potential.component_id"),
        )

    @classmethod
    def jacobi(cls, value: float) -> Potential:
        """Return one canonical total-mass-normalized Jacobi potential.

        Args:
            value: Finite dimensionless level normalized by the total binary mass.
        """
        return cls(value=value, convention=PotentialConvention.JACOBI)

    @classmethod
    def kopal(cls, value: float, *, component_id: ComponentId) -> Potential:
        """Return one component-centred, component-mass-normalized Kopal potential.

        Args:
            value: Finite dimensionless level normalized by the selected mass.
            component_id: Component number, either `1` or `2`.
        """
        return cls(
            value=value,
            convention=PotentialConvention.KOPAL,
            component_id=component_id,
        )


def require_potential(value: object, *, name: str = "potential") -> Potential:
    """Require a typed potential, saying how to make one from a bare number."""
    if isinstance(value, Potential):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        raise TypeError(
            f"{name}: expected Potential, got {type(value).__name__}; wrap the value as "
            "Potential.jacobi(value) or Potential.kopal(value, component_id=...)"
        )
    return require_instance(value, Potential, name=name)


@dataclass(frozen=True, slots=True)
class CriticalPoint:
    """A stationary Jacobi point with its potential Hessian.

    Coordinates and Hessian axes use the producer's frame. Hessian entries keep
    intrinsic G*(M1+M2)/a**3 units, where a is the separation, even when
    positions use a binary view's length scale. Eigenvalues describe the
    potential's Hessian, not surface curvatures.

    The positive Jacobi potential rises towards the components. Collinear points
    rise along the axis and fall transversely (Morse index 2); triangular
    points rise in the orbital plane and fall out of it (index 1).

    The value checks this structure, but not stationarity, and does not
    recompute the supplied derivatives.

    Args:
        position: Finite coordinates in the producer's frame and length units.
        potential: Canonical dimensionless Jacobi potential at the point.
        hessian: Symmetric three-by-three tuple of finite rows. Orbital-plane
            symmetry requires the xz and yz entries to be zero.
    """

    position: Point3D
    potential: Potential
    hessian: tuple[Point3D, Point3D, Point3D]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position",
            normalize_point3d(self.position, name="position"),
        )
        object.__setattr__(
            self,
            "potential",
            require_instance(self.potential, Potential, name="potential"),
        )
        if self.potential.convention is not PotentialConvention.JACOBI:
            raise ValueError("critical point: expected canonical Jacobi potential")
        require_instance(self.hessian, tuple, name="hessian")
        if len(self.hessian) != 3:
            raise ValueError("hessian: expected three rows")
        h = cast(
            tuple[Point3D, Point3D, Point3D],
            tuple(
                normalize_point3d(row, name=f"hessian[{i}]")
                for i, row in enumerate(self.hessian)
            ),
        )
        if any(h[i][j] != h[j][i] for i in range(3) for j in range(i)):
            raise ValueError("hessian: expected symmetric entries")
        if h[0][2] != 0.0 or h[1][2] != 0.0:
            raise ValueError(
                "hessian: expected zero xz and yz entries by orbital symmetry"
            )
        object.__setattr__(self, "hessian", h)

    @property
    def eigenvalues(self) -> tuple[float, float, float]:
        """Ascending potential-Hessian eigenvalues in G*(M1+M2)/a**3 units."""
        h = self.hessian
        a, b, d = h[0][0], h[0][1], h[1][1]
        if b == 0.0:
            values = (a, d, h[2][2])
        else:
            midpoint = a / 2 + d / 2
            radius = hypot(a / 2 - d / 2, b)
            values = (midpoint - radius, midpoint + radius, h[2][2])
        return cast(tuple[float, float, float], tuple(sorted(values)))

    @property
    def morse_index(self) -> int:
        """Number of strictly negative potential-Hessian eigenvalues."""
        return sum(value < 0.0 for value in self.eigenvalues)


def convert_potential(
    potential: Potential,
    *,
    q: float,
    convention: PotentialConvention,
    component_id: ComponentId | None = None,
) -> Potential:
    """Convert one typed potential through the canonical Jacobi convention.

    Args:
        potential: Typed source value.
        q: Public system mass ratio `M1/M2`.
        convention: Requested target convention.
        component_id: Required for a target component-centred Kopal convention and
            forbidden for the component-independent Jacobi convention.

    Returns:
        The same physical equipotential level expressed in the target convention.
    """
    potential = require_potential(potential)
    q = normalize_mass_ratio(q)

    try:
        target = PotentialConvention(convention)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "convention: expected a supported PotentialConvention, "
            f"got {reprlib.repr(convention)}"
        ) from exc

    if target is PotentialConvention.JACOBI:
        if component_id is not None:
            raise ValueError(
                "component_id: canonical Jacobi potential is component-independent"
            )
        if potential.convention is PotentialConvention.JACOBI:
            return potential
        return Potential.jacobi(float(_to_jacobi_exact(potential, q)))

    if component_id is None:
        raise ValueError(
            "component_id: target component-centred Kopal convention requires "
            "component_id=1 or component_id=2"
        )
    component_id = require_component_id(component_id)
    if (
        potential.convention is PotentialConvention.KOPAL
        and potential.component_id == component_id
    ):
        return potential

    jacobi = _to_jacobi_exact(potential, q)
    component_mass, companion_mass = _conversion_masses(q, component_id)
    kopal = (
        jacobi - companion_mass**2 / (2 * (component_mass + companion_mass))
    ) / component_mass
    return Potential.kopal(float(kopal), component_id=component_id)


def _conversion_masses(
    q: float, component_id: ComponentId
) -> tuple[Fraction, Fraction]:
    masses = tuple(map(Fraction, mass_fractions(q)))
    return (masses[0], masses[1]) if component_id == 1 else (masses[1], masses[0])


def _to_jacobi_exact(potential: Potential, q: float) -> Fraction:
    if potential.convention is PotentialConvention.JACOBI:
        return Fraction(potential.value)

    component_id = potential.component_id
    if component_id is None:  # pragma: no cover - enforced by Potential
        raise AssertionError("Kopal potential is missing its component_id")
    component_mass, companion_mass = _conversion_masses(q, component_id)
    return Fraction(potential.value) * component_mass + companion_mass**2 / (
        2 * (component_mass + companion_mass)
    )


def _potential_residual_scale(potential: float) -> float:
    """Return the natural O(1) residual scale for the Jacobi potential."""
    return max(1.0, abs(potential))
