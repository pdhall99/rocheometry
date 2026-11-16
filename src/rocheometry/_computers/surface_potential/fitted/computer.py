"""Dependency-free endpoint-constrained fill/potential maps."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from math import isfinite, sqrt
from typing import ClassVar

from rocheometry._computers.critical_point import CriticalPointComputer
from rocheometry._computers.fitted import FITTED_QI_MAX, FITTED_QI_MIN
from rocheometry._computers.lobe_volume_radius.protocol import LobeVolumeRadiusComputer
from rocheometry._json import (
    child_mapping,
    formatted_json_text,
    number,
    parse_mapping,
    sequence,
    string,
)
from rocheometry._numerics.polynomial import CoefficientMatrix, chebyshev_value
from rocheometry._roche.field import RocheField
from rocheometry._validation import (
    _normalize_finite_positive_float,
    normalize_float_tuple,
    normalize_volume_radius_fill,
    validate_keys,
)


@dataclass(frozen=True, slots=True)
class SurfacePotentialMap:
    """Chebyshev residual tensor with analytic critical and small-fill endpoints.

    Rows are ordered by degree in the compact fourth-root qi coordinate;
    columns by degree in `2*sqrt(1-fill)-1`. The fourth-root coordinate follows
    the wide-domain prototype and is affinely mapped across the stored domain.

    Attributes:
        coefficients: Nonempty rectangular tensor of finite Chebyshev coefficients.
        qi_min: Positive lower component-to-companion mass ratio.
        qi_max: Upper component-to-companion mass ratio, greater than qi_min.
        provenance: Training, reference, dependency and validation metadata.
    """

    SCHEMA: ClassVar[str] = "rocheometry.surface-potential-map/v1"
    COORDINATE: ClassVar[str] = "fourth-root-compact"
    coefficients: CoefficientMatrix
    qi_min: float = FITTED_QI_MIN
    qi_max: float = FITTED_QI_MAX
    provenance: Mapping[str, object] = field(
        default_factory=dict, compare=False, hash=False, repr=False
    )

    def __post_init__(self) -> None:
        rows = tuple(
            normalize_float_tuple(row, name="coefficients") for row in self.coefficients
        )
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("coefficients: expected a nonempty rectangular tensor")
        object.__setattr__(self, "coefficients", rows)
        for name in ("qi_min", "qi_max"):
            object.__setattr__(
                self,
                name,
                _normalize_finite_positive_float(getattr(self, name), name=name),
            )
        if self.qi_min >= self.qi_max:
            raise ValueError("qi_min must be less than qi_max")
        object.__setattr__(self, "provenance", dict(self.provenance))

    def coordinate(self, qi: float) -> float:
        """Return the compact fourth-root coordinate, rejecting extrapolation."""
        if not self.qi_min <= qi <= self.qi_max:
            raise ValueError(
                f"qi: potential map requires {self.qi_min:g} <= qi <= {self.qi_max:g}"
            )
        lo = _compact(self.qi_min)
        hi = _compact(self.qi_max)
        return 2 * (_compact(qi) - lo) / (hi - lo) - 1

    def coefficients_at(self, qi: float) -> tuple[float, ...]:
        """Contract the qi dimension once for forward or iterative inverse use."""
        z = self.coordinate(qi)
        return tuple(
            chebyshev_value(z, tuple(row[j] for row in self.coefficients))
            for j in range(len(self.coefficients[0]))
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SurfacePotentialMap:
        """Decode a validated model artefact and its provenance."""
        source = "surface-potential map"
        validate_keys(
            value,
            {"schema", "coordinate", "qi_min", "qi_max", "coefficients", "provenance"},
            source,
        )
        if string(value, "schema", source) != cls.SCHEMA:
            raise ValueError("unsupported surface-potential map schema")
        if string(value, "coordinate", source) != cls.COORDINATE:
            raise ValueError("unsupported surface-potential map coordinate")
        rows: list[tuple[float, ...]] = []
        for row in sequence(value, "coefficients", source):
            if not isinstance(row, list):
                raise TypeError("coefficients: expected an array of arrays")
            # Use the JSON numeric decoder to exclude booleans and non-finite values.
            rows.append(
                tuple(number({"value": item}, "value", "coefficients") for item in row)
            )
        return cls(
            tuple(rows),
            number(value, "qi_min", source),
            number(value, "qi_max", source),
            child_mapping(value, "provenance", source),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize coefficients, domain and reproducibility metadata."""
        return {
            "schema": self.SCHEMA,
            "coordinate": self.COORDINATE,
            "qi_min": self.qi_min,
            "qi_max": self.qi_max,
            "coefficients": [list(row) for row in self.coefficients],
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_json(cls, text: str) -> SurfacePotentialMap:
        """Deserialize one JSON model artefact."""
        return cls.from_mapping(parse_mapping(text, "surface-potential map JSON"))

    def to_json(self) -> str:
        """Serialize one JSON model artefact with a trailing newline."""
        return formatted_json_text(self.to_mapping())


def _compact(qi: float) -> float:
    root = sqrt(sqrt(qi))
    return root / (1 + root)


def _value_derivative(x: float, coefficients: tuple[float, ...]) -> tuple[float, float]:
    """Evaluate a Chebyshev series and its derivative in one recurrence."""
    value = coefficients[0]
    derivative = 0.0
    previous, current = 1.0, x
    d_previous, d_current = 0.0, 1.0
    for index in range(1, len(coefficients)):
        coefficient = coefficients[index]
        value += coefficient * current
        derivative += coefficient * d_current
        following = 2 * x * current - previous
        d_following = 2 * current + 2 * x * d_current - d_previous
        previous, current = current, following
        d_previous, d_current = d_current, d_following
    return value, derivative


def fill_times_t(
    fill: float, tinf: float, coefficients: tuple[float, ...]
) -> tuple[float, float]:
    """Return fill*t and its fill derivative, including both analytic endpoints."""
    s = sqrt(max(0.0, 1 - fill))
    residual, d_residual = _value_derivative(2 * s - 1, coefficients)
    f3 = fill * fill * fill
    t = 1 + (tinf - 1) * (1 - f3) + (1 - fill) * f3 * residual
    derivative = (
        1
        + (tinf - 1) * (1 - 4 * f3)
        + f3 * (4 - 5 * fill) * residual
        - fill * f3 * s * d_residual
    )
    return fill * t, derivative


@lru_cache(maxsize=4096)
def _fitted_parameters(
    field: RocheField, x1: float, radius: float, model: SurfacePotentialMap
) -> tuple[float, float, float, tuple[float, ...]]:
    """Contract per-field constants once without hashing injected strategies."""
    background = field._background_potential_exact()
    delta = float(field._axis_potential_exact(x1) - background)
    return (
        float(background),
        delta,
        delta * radius / field.component_mass,
        model.coefficients_at(field.qi),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class FittedSurfacePotentialComputer:
    """Forward fill-to-depth map with exact critical and small-fill limits.

    Attributes:
        l1_computer: Critical point defining the exact fill-one level.
        lobe_volume_radius_computer: Calibration of the small-fill asymptote.
        model: Explicit residual tensor, or None to load the packaged map lazily.
    """

    l1_computer: CriticalPointComputer
    lobe_volume_radius_computer: LobeVolumeRadiusComputer
    model: SurfacePotentialMap | None = None

    def __post_init__(self) -> None:
        for name in ("l1_computer", "lobe_volume_radius_computer"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name}: expected a callable computer")

    def supports(self, field: RocheField) -> bool:
        """Whether both map and radius calibration support this mass ratio."""
        lo = FITTED_QI_MIN if self.model is None else self.model.qi_min
        hi = FITTED_QI_MAX if self.model is None else self.model.qi_max
        if not lo <= field.qi <= hi:
            return False
        radius = self.lobe_volume_radius_computer
        return (
            getattr(radius, "qi_min", 0.0)
            <= field.qi
            <= getattr(radius, "qi_max", float("inf"))
        )

    def _parameters(
        self, field: RocheField
    ) -> tuple[float, float, float, tuple[float, ...]]:
        if not self.supports(field):
            raise ValueError("qi: outside fitted potential map or lobe-radius domain")
        model = self.model
        if model is None:
            from rocheometry._computers.packaged import _packaged_surface_potential_map

            model = _packaged_surface_potential_map()
        return _fitted_parameters(
            field,
            self.l1_computer(field).position[0],
            self.lobe_volume_radius_computer(field),
            model,
        )

    def __call__(self, field: RocheField, volume_radius_fill: float, /) -> float:
        """Return the endpoint-factored fitted depth, exactly zero at fill one."""
        fill = normalize_volume_radius_fill(volume_radius_fill)
        if not self.supports(field):
            raise ValueError("qi: outside fitted potential map or lobe-radius domain")
        if fill == 1:
            return 0.0
        _, delta, tinf, coefficients = self._parameters(field)
        residual = chebyshev_value(2 * sqrt(1 - fill) - 1, coefficients)
        f2 = fill * fill
        f3 = f2 * fill
        g = fill * (1 + (tinf - 1) * (1 - f3) + (1 - fill) * f3 * residual)
        if not isfinite(g) or g <= 0:
            raise RuntimeError("surface-potential map produced an invalid fill*t")
        factor = -1 + fill * (tinf - 1) * (1 + fill + f2) + f3 * fill * residual
        return delta * (1 - fill) * factor / g
