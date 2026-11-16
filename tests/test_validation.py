import math
from fractions import Fraction

import pytest

from rocheometry._validation import (
    normalize_inclination,
    require_choice,
    require_finite,
    require_instance,
    require_int,
    require_positive,
    validate_real,
)


@pytest.mark.parametrize(
    "inclination",
    (
        0.0,
        -0.0,
        math.pi / 6,
        math.pi / 2,
        math.pi / 2 + 1e-12,
        math.pi - 1e-12,
        math.pi,
    ),
)
def test_in_range_inclinations_preserve_previous_folding_bits(
    inclination: float,
) -> None:
    previous = abs(inclination) % math.pi
    previous = min(previous, math.pi - previous)
    assert normalize_inclination(inclination).hex() == previous.hex()


def test_out_of_range_inclination_message_suggests_degree_conversion() -> None:
    with pytest.raises(ValueError) as error:
        normalize_inclination(70)
    assert str(error.value) == (
        "inclination: expected radians in [0, pi], got 70; use math.radians() for degrees"
    )


def test_require_choice_returns_the_identical_value() -> None:
    value = "".join(("compact", "-cuberoot"))

    assert require_choice(value, ("log", "compact-cuberoot"), name="kind") is value


def test_require_choice_derives_its_error_message_from_the_choices() -> None:
    with pytest.raises(
        ValueError, match="kind: expected 'first' or 'second', got 'typo'"
    ):
        require_choice("typo", ("first", "second"), name="kind")


def test_require_instance_returns_same_value() -> None:
    value = object()

    assert require_instance(value, object, name="value") is value


def test_require_int_rejects_bool() -> None:
    with pytest.raises(TypeError, match="value: expected int, got bool"):
        require_int(True, name="value")


def test_validate_real_accepts_runtime_real() -> None:
    validate_real(Fraction(1, 2), name="value")


@pytest.mark.parametrize("value", [True, "1.0"])
def test_validate_real_rejects_wrong_types(value: object) -> None:
    with pytest.raises(TypeError, match="value: expected a real number"):
        validate_real(value, name="value")


def test_require_finite_returns_same_value() -> None:
    value = 1.5

    assert require_finite(value, name="value") is value


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_require_finite_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="value: expected a finite number"):
        require_finite(value, name="value")


def test_require_positive_returns_same_value() -> None:
    value = 1.5

    assert require_positive(value, name="value") is value


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan])
def test_require_positive_rejects_non_positive_values(value: float) -> None:
    with pytest.raises(ValueError, match="value: expected a positive number"):
        require_positive(value, name="value")
