"""Dependency-free fitted volume-radius runtime models."""

from __future__ import annotations

import reprlib
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, log10, sqrt
from typing import ClassVar, Protocol

from rocheometry._json import (
    float_tuple,
    formatted_json_text,
    number,
    parse_mapping,
    string,
)
from rocheometry._numerics.polynomial import power_value
from rocheometry._roche.field import RocheField
from rocheometry._validation import (
    normalize_float_tuple,
    require_instance,
    validate_keys,
)


class Interpolator(Protocol):
    """An interpolation strategy on an ordered one-dimensional grid."""

    NAME: ClassVar[str]
    """The stable serialization identifier of an interpolation strategy."""

    def __call__(
        self,
        xs: tuple[float, ...],
        ys: tuple[float, ...],
        x: float,
    ) -> float:
        """Return the interpolated value at x.

        Args:
            xs: Strictly increasing coordinates with at least two nodes.
            ys: Values at the nodes, with the same length as `xs`.
            x: Evaluation coordinate in the closed interval `[xs[0], xs[-1]]`.
        """
        ...


@dataclass(frozen=True, slots=True)
class LinearInterpolator:
    """Piecewise-linear interpolation on an ordered grid."""

    NAME: ClassVar[str] = "linear"
    """The registered identifier for piecewise-linear interpolation."""

    def __call__(
        self,
        xs: tuple[float, ...],
        ys: tuple[float, ...],
        x: float,
    ) -> float:
        """Return the piecewise-linear interpolation at x.

        Args:
            xs: Strictly increasing coordinates with at least two nodes.
            ys: Values at the nodes, with the same length as `xs`.
            x: Evaluation coordinate in the closed interval `[xs[0], xs[-1]]`.
        """
        if len(xs) != len(ys) or len(xs) < 2:
            raise ValueError("interpolation grids must have equal length >= 2")
        if not xs[0] <= x <= xs[-1]:
            raise ValueError("interpolation coordinate lies outside the grid")
        if x == xs[-1]:
            return ys[-1]

        upper = bisect_right(xs, x)
        lower = max(0, upper - 1)
        fraction = (x - xs[lower]) / (xs[lower + 1] - xs[lower])
        return ys[lower] + fraction * (ys[lower + 1] - ys[lower])


# Resolution budget: dispatch to the linear interpolation formula.
FITTED_LOBE_INTERPOLATOR_REGISTRY: dict[str, Interpolator] = {
    LinearInterpolator.NAME: LinearInterpolator(),
}


def parse_interpolator(name: str) -> Interpolator:
    """Return one registered interpolation strategy from its stable name.

    Args:
        name: Stable interpolation strategy identifier.

    Raises:
        ValueError: No registered interpolation strategy has this name.
    """
    name = require_instance(name, str, name="interpolator")
    try:
        return FITTED_LOBE_INTERPOLATOR_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            "interpolator: expected a registered volume-radius strategy, "
            f"got {reprlib.repr(name)}"
        ) from exc


@dataclass(frozen=True, slots=True)
class InterpolatedLobeVolumeRadiusComputer:
    """A lobe-radius strategy interpolating a finite table in `log10(qi)`.

    Calls return radii in separation units and reject mass ratios outside the
    table domain. Numeric table entries are normalized to floats. Custom
    interpolators may be used in memory; serialization requires a matching
    registered interpolation strategy.

    Attributes:
        log_qis: At least two finite, strictly increasing logarithmic mass ratios.
        radii: Corresponding finite positive radii, with the same node count.
        interpolator: Callable strategy with a nonempty stable `NAME`.
    """

    MODEL_FORM: ClassVar[str] = "interpolator"
    """The serialized interpolated radius model form."""

    log_qis: tuple[float, ...]
    radii: tuple[float, ...]
    interpolator: Interpolator = LinearInterpolator()

    def __post_init__(self) -> None:
        log_qis = normalize_float_tuple(self.log_qis, name="log_qis")
        radii = normalize_float_tuple(self.radii, name="radii")

        if len(log_qis) != len(radii):
            raise ValueError("table: log_qis and radii must have the same length")
        if len(log_qis) < 2:
            raise ValueError(
                f"log_qis: expected at least two nodes, got {reprlib.repr(log_qis)}"
            )
        if any(
            not right > left
            for left, right in zip(log_qis[:-1], log_qis[1:], strict=True)
        ):
            raise ValueError(
                "log_qis: expected strictly increasing values, "
                f"got {reprlib.repr(log_qis)}"
            )
        if any(not radius > 0.0 for radius in radii):
            raise ValueError(
                f"radii: expected positive values, got {reprlib.repr(radii)}"
            )

        interpolator = self.interpolator
        if not callable(interpolator) or not hasattr(interpolator, "NAME"):
            raise TypeError(
                "interpolator: expected an interpolation strategy, "
                f"got {type(interpolator).__name__}"
            )
        name = require_instance(
            interpolator.NAME,
            str,
            name="interpolator.NAME",
        )
        if not name:
            raise ValueError("interpolator.NAME: expected a non-empty string, got ''")

        object.__setattr__(self, "log_qis", log_qis)
        object.__setattr__(self, "radii", radii)

    @property
    def log_qi_min(self) -> float:
        """The minimum supported logarithmic component mass ratio."""
        return self.log_qis[0]

    @property
    def log_qi_max(self) -> float:
        """The maximum supported logarithmic component mass ratio."""
        return self.log_qis[-1]

    @property
    def qi_min(self) -> float:
        """The minimum supported component mass ratio."""
        return 10.0**self.log_qi_min

    @property
    def qi_max(self) -> float:
        """The maximum supported component mass ratio."""
        return 10.0**self.log_qi_max

    @property
    def node_count(self) -> int:
        """The number of interpolation nodes."""
        return len(self.log_qis)

    def __call__(self, field: RocheField, /) -> float:
        """Return the critical Roche-lobe volume radius."""
        qi = field.qi
        if not self.qi_min <= qi <= self.qi_max:
            raise ValueError(
                f"volume-radius model requires {self.qi_min:g} <= qi <= {self.qi_max:g}"
            )
        return self.interpolator(self.log_qis, self.radii, log10(qi))

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> InterpolatedLobeVolumeRadiusComputer:
        """Deserialize one interpolated volume-radius model mapping.

        Args:
            value: Artefact or configuration mapping to decode.

        Raises:
            ValueError: Contents are malformed, inconsistent, or use an unsupported
                format.
            TypeError: A stored field has the wrong type.
        """
        validate_keys(
            value,
            {"model_form", "interpolator", "log_qis", "radii"},
            "volume-radius model",
        )
        model_form = string(value, "model_form", "volume-radius model")
        if model_form != cls.MODEL_FORM:
            raise ValueError(f"unsupported volume-radius model form {model_form!r}")
        return cls(
            log_qis=float_tuple(value, "log_qis", "volume-radius model"),
            radii=float_tuple(value, "radii", "volume-radius model"),
            interpolator=parse_interpolator(
                string(value, "interpolator", "volume-radius model")
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize this interpolated model to a plain mapping.

        Raises:
            ValueError: The interpolator does not match a registered strategy and
                cannot be represented by the artefact format.
        """
        name = self.interpolator.NAME
        registered = parse_interpolator(name)
        if (
            type(self.interpolator) is not type(registered)
            or self.interpolator != registered
        ):
            raise ValueError(
                f"interpolator {name!r} does not match the registered strategy"
            )
        return {
            "interpolator": name,
            "log_qis": list(self.log_qis),
            "model_form": self.MODEL_FORM,
            "radii": list(self.radii),
        }

    def to_json(self) -> str:
        """Serialize this interpolated model to JSON."""
        return formatted_json_text(self.to_mapping())

    @classmethod
    def from_json(cls, text: str) -> InterpolatedLobeVolumeRadiusComputer:
        """Deserialize and validate one interpolated volume-radius model.

        Args:
            text: Serialized JSON artefact to decode.

        Raises:
            ValueError: Contents are malformed, inconsistent, or use an unsupported
                format.
            TypeError: A stored field has the wrong type.
        """
        value = parse_mapping(text, "volume-radius JSON")
        return cls.from_mapping(value)


@dataclass(frozen=True, slots=True)
class PiecewiseSixthRootHornerLobeVolumeRadiusComputer:
    """A two-piece Horner model of the critical lobe's volume-equivalent radius.

    The compact coordinate is `s = qi**(1/6) / (1 + qi**(1/6))`.
    Each piece fits radius divided by `s**2`, using a rescaled coordinate
    with the equal-mass join at `qi=1`. Calls return separation-unit radii.
    Stored continuity metadata describes the fitted artefact; construction
    does not enforce matching derivatives across arbitrary supplied pieces.

    Attributes:
        low_coefficients: Nonempty finite power coefficients in ascending degree
            order for `qi <= 1`, evaluated at `4*s - 1`.
        high_coefficients: Nonempty finite power coefficients in ascending degree
            order for `qi > 1`, evaluated at `4*s - 3`.
    """

    MODEL_FORM: ClassVar[str] = "piecewise-sixth-root-horner"
    """The serialized two-piece Horner radius model form."""
    COORDINATE: ClassVar[str] = "sixth-root-compact"
    """The serialized compact sixth-root coordinate identifier."""
    CONTINUITY: ClassVar[str] = "C2"
    """The continuity metadata expected for the fitted equal-mass join."""
    JOIN_QI: ClassVar[float] = 1.0
    """The component mass ratio at which the two polynomial pieces meet."""

    low_coefficients: tuple[float, ...]
    high_coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        low = normalize_float_tuple(
            self.low_coefficients,
            name="low_coefficients",
        )
        high = normalize_float_tuple(
            self.high_coefficients,
            name="high_coefficients",
        )
        if not low:
            raise ValueError("low_coefficients: expected at least one value")
        if not high:
            raise ValueError("high_coefficients: expected at least one value")
        object.__setattr__(self, "low_coefficients", low)
        object.__setattr__(self, "high_coefficients", high)

    @property
    def coefficient_count(self) -> int:
        """The total stored polynomial coefficient count."""
        return len(self.low_coefficients) + len(self.high_coefficients)

    def __call__(self, field: RocheField, /) -> float:
        """Return the critical Roche-lobe volume radius.

        Args:
            field: Component-local field supplying a finite positive mass ratio.

        Raises:
            RuntimeError: The fitted coefficients produce a non-finite or non-positive
                radius.
        """
        qi = field.qi
        s = _sixth_root_coordinate(qi)
        if qi <= self.JOIN_QI:
            x = 4.0 * s - 1.0
            target = power_value(x, self.low_coefficients)
        else:
            x = 4.0 * s - 3.0
            target = power_value(x, self.high_coefficients)

        radius = s * s * target
        if not isfinite(radius) or radius <= 0.0:
            raise RuntimeError(
                "piecewise sixth-root volume-radius model produced an invalid radius"
            )
        return radius

    def logarithmic_slope(self, field: RocheField, /) -> float:
        """Differentiate the same polynomial analytically in ln(qi).

        R=s**2*P(4*s-offset) and ds/dln(qi)=s*(1-s)/6. No field
        sample, finite-difference step or additional fitted coefficients enter.
        """
        s = _sixth_root_coordinate(field.qi)
        coefficients = (
            self.low_coefficients
            if field.qi <= self.JOIN_QI
            else self.high_coefficients
        )
        x = 4 * s - (1 if field.qi <= self.JOIN_QI else 3)
        value, derivative = coefficients[-1], 0.0
        for coefficient in reversed(coefficients[:-1]):
            derivative = derivative * x + value
            value = value * x + coefficient
        return (1 - s) / 3 + (2 / 3) * s * (1 - s) * derivative / value

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> PiecewiseSixthRootHornerLobeVolumeRadiusComputer:
        """Deserialize one selected piecewise-Horner model mapping.

        Args:
            value: Artefact or configuration mapping to decode.

        Raises:
            ValueError: Contents are malformed, inconsistent, or use an unsupported
                format.
            TypeError: A stored field has the wrong type.
        """
        validate_keys(
            value,
            {
                "model_form",
                "coordinate",
                "join_qi",
                "continuity",
                "low_coefficients",
                "high_coefficients",
            },
            "volume-radius model",
        )
        model_form = string(value, "model_form", "volume-radius model")
        if model_form != cls.MODEL_FORM:
            raise ValueError(f"unsupported volume-radius model form {model_form!r}")
        coordinate = string(value, "coordinate", "volume-radius model")
        if coordinate != cls.COORDINATE:
            raise ValueError(f"unsupported volume-radius coordinate {coordinate!r}")
        join_qi = number(value, "join_qi", "volume-radius model")
        if join_qi != cls.JOIN_QI:
            raise ValueError(f"piecewise volume-radius join must be {cls.JOIN_QI:g}")
        continuity = string(value, "continuity", "volume-radius model")
        if continuity != cls.CONTINUITY:
            raise ValueError(
                f"piecewise volume-radius continuity must be {cls.CONTINUITY!r}"
            )
        return cls(
            low_coefficients=float_tuple(
                value,
                "low_coefficients",
                "volume-radius model",
            ),
            high_coefficients=float_tuple(
                value,
                "high_coefficients",
                "volume-radius model",
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize the selected piecewise-Horner model."""
        return {
            "continuity": self.CONTINUITY,
            "coordinate": self.COORDINATE,
            "high_coefficients": list(self.high_coefficients),
            "join_qi": self.JOIN_QI,
            "low_coefficients": list(self.low_coefficients),
            "model_form": self.MODEL_FORM,
        }

    def to_json(self) -> str:
        """Serialize the selected piecewise-Horner model to JSON."""
        return formatted_json_text(self.to_mapping())

    @classmethod
    def from_json(
        cls,
        text: str,
    ) -> PiecewiseSixthRootHornerLobeVolumeRadiusComputer:
        """Deserialize and validate one piecewise-Horner model.

        Args:
            text: Serialized JSON artefact to decode.

        Raises:
            ValueError: Contents are malformed, inconsistent, or use an unsupported
                format.
            TypeError: A stored field has the wrong type.
        """
        value = parse_mapping(text, "volume-radius JSON")
        return cls.from_mapping(value)


LobeVolumeRadiusModel = (
    InterpolatedLobeVolumeRadiusComputer
    | PiecewiseSixthRootHornerLobeVolumeRadiusComputer
)


def lobe_volume_radius_computer_from_mapping(
    value: Mapping[str, object],
) -> LobeVolumeRadiusModel:
    """Deserialize a supported fitted lobe-radius model mapping.

    Args:
        value: Serialized model mapping with a supported `model_form`.

    Raises:
        ValueError: The model form or stored model contents are unsupported or invalid.
        TypeError: A stored field has the wrong type.
    """
    model_form = string(value, "model_form", "volume-radius model")
    if model_form == InterpolatedLobeVolumeRadiusComputer.MODEL_FORM:
        return InterpolatedLobeVolumeRadiusComputer.from_mapping(value)
    if model_form == PiecewiseSixthRootHornerLobeVolumeRadiusComputer.MODEL_FORM:
        return PiecewiseSixthRootHornerLobeVolumeRadiusComputer.from_mapping(value)
    raise ValueError(f"unsupported volume-radius model form {model_form!r}")


def lobe_volume_radius_computer_from_json(text: str) -> LobeVolumeRadiusModel:
    """Deserialize a supported fitted lobe-radius JSON artefact.

    Args:
        text: Serialized JSON for a supported radius model.

    Raises:
        ValueError: The model form or stored model contents are unsupported or invalid.
        TypeError: A stored field has the wrong type.
    """
    value = parse_mapping(text, "volume-radius JSON")
    return lobe_volume_radius_computer_from_mapping(value)


def _sixth_root_coordinate(qi: float) -> float:
    """Return s = qi**(1/6) / (1 + qi**(1/6))."""
    q13 = qi ** (1.0 / 3.0)
    q16 = sqrt(q13)
    return q16 / (1.0 + q16)
