import math
from typing import cast

import pytest

import rocheometry
from rocheometry._roche import RocheField

HESSIAN = ((17.0, 0.0, 0.0), (0.0, -7.0, 0.0), (0.0, 0.0, -8.0))


def test_canonical_potential_uses_concise_serialized_identifier() -> None:
    potential = rocheometry.Potential.jacobi(2.0)

    assert potential.convention is rocheometry.PotentialConvention.JACOBI
    assert potential.convention.value == ("jacobi")
    assert potential.component_id is None


def test_kopal_potential_requires_persistent_component() -> None:
    with pytest.raises(ValueError, match="requires component"):
        rocheometry.Potential(
            value=3.0,
            convention=rocheometry.PotentialConvention.KOPAL,
        )

    with pytest.raises(ValueError, match="component-independent"):
        rocheometry.Potential(
            value=2.0,
            convention=rocheometry.PotentialConvention.JACOBI,
            component_id=1,
        )


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_potential_rejects_nonfinite_value(value: float) -> None:
    with pytest.raises(ValueError, match="potential.value"):
        rocheometry.Potential.jacobi(value)


def test_critical_point_pairs_position_with_canonical_potential() -> None:
    potential = rocheometry.Potential.jacobi(4.0)
    point = rocheometry.CriticalPoint(
        position=(1, 2, 3),
        potential=potential,
        hessian=HESSIAN,
    )

    assert point.position == (1.0, 2.0, 3.0)
    assert all(isinstance(coordinate, float) for coordinate in point.position)
    assert point.potential is potential


@pytest.mark.parametrize("position", [None, 1.0, "abc", [1, 2, 3], {"x": 1}])
def test_critical_point_rejects_wrong_position_type(position: object) -> None:
    with pytest.raises(TypeError, match="position: expected tuple"):
        rocheometry.CriticalPoint(
            position=cast(tuple[float, float, float], position),
            potential=rocheometry.Potential.jacobi(2.0),
            hessian=HESSIAN,
        )


@pytest.mark.parametrize("position", [(), (1.0, 2.0), (1.0, 2.0, 3.0, 4.0)])
def test_critical_point_rejects_wrong_position_length(
    position: tuple[float, ...],
) -> None:
    with pytest.raises(
        ValueError, match="position: expected three Cartesian coordinates"
    ):
        rocheometry.CriticalPoint(
            position=cast(tuple[float, float, float], position),
            potential=rocheometry.Potential.jacobi(2.0),
            hessian=HESSIAN,
        )


@pytest.mark.parametrize("coordinate", [None, True, "0"])
def test_critical_point_rejects_wrong_coordinate_type(coordinate: object) -> None:
    with pytest.raises(TypeError, match=r"position\[1\]: expected a real number"):
        rocheometry.CriticalPoint(
            position=(0.0, cast(float, coordinate), 0.0),
            potential=rocheometry.Potential.jacobi(2.0),
            hessian=HESSIAN,
        )


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("coordinate", [math.inf, -math.inf, math.nan])
def test_critical_point_rejects_nonfinite_coordinates(
    index: int, coordinate: float
) -> None:
    position = [0.0, 0.0, 0.0]
    position[index] = coordinate
    with pytest.raises(
        ValueError, match=rf"position\[{index}\]: expected a finite number"
    ):
        rocheometry.CriticalPoint(
            position=cast(tuple[float, float, float], tuple(position)),
            potential=rocheometry.Potential.jacobi(2.0),
            hessian=HESSIAN,
        )


def test_documented_component_one_kopal_conversion() -> None:
    q = 0.4
    q_kopal = 1.0 / q
    omega = 4.2
    expected_u = omega / (1.0 + q_kopal) + 0.5 * (q_kopal / (1.0 + q_kopal)) ** 2

    converted = rocheometry.convert_potential(
        rocheometry.Potential.kopal(omega, component_id=1),
        q=q,
        convention=rocheometry.PotentialConvention.JACOBI,
    )

    assert converted.convention is rocheometry.PotentialConvention.JACOBI
    assert converted.component_id is None
    assert converted.value == pytest.approx(expected_u, rel=2.0e-15)


@pytest.mark.parametrize("q", [1.0e-4, 0.1, 0.4, 1.0, 2.5, 10.0, 1.0e4])
@pytest.mark.parametrize("component_id", [1, 2])
def test_kopal_round_trip_through_canonical(
    q: float,
    component_id: int,
) -> None:
    canonical = rocheometry.Potential.jacobi(2.123456789)
    kopal = rocheometry.convert_potential(
        canonical,
        q=q,
        convention=rocheometry.PotentialConvention.KOPAL,
        component_id=component_id,  # type: ignore[arg-type]
    )
    recovered = rocheometry.convert_potential(
        kopal,
        q=q,
        convention=rocheometry.PotentialConvention.JACOBI,
    )

    assert kopal.component_id == component_id
    assert recovered.value == pytest.approx(canonical.value, rel=3.0e-15)


@pytest.mark.parametrize("q", [1.0e-4, 0.2, 1.0, 5.0, 1.0e4])
@pytest.mark.parametrize("component_id", [1, 2])
def test_kopal_formula_matches_canonical_field_in_each_component_chart(
    q: float,
    component_id: int,
) -> None:
    qi = q if component_id == 1 else 1.0 / q
    q_kopal = 1.0 / qi
    x, y, z = (0.31, 0.22, -0.09)
    r_i = math.sqrt(x * x + y * y + z * z)
    dx = x - 1.0
    r_c = math.sqrt(dx * dx + y * y + z * z)
    omega = (
        1.0 / r_i
        + q_kopal / r_c
        + 0.5 * (1.0 + q_kopal) * (x * x + y * y)
        - q_kopal * x
    )

    converted = rocheometry.convert_potential(
        rocheometry.Potential.kopal(
            omega,
            component_id=component_id,  # type: ignore[arg-type]
        ),
        q=q,
        convention=rocheometry.PotentialConvention.JACOBI,
    )

    assert converted.value == pytest.approx(
        RocheField(qi).potential(x, y, z),
        rel=3.0e-15,
        abs=3.0e-15,
    )


def test_kopal_conversion_between_component_charts_preserves_level() -> None:
    q = 0.3
    first = rocheometry.Potential.kopal(4.1, component_id=1)

    second = rocheometry.convert_potential(
        first,
        q=q,
        convention=rocheometry.PotentialConvention.KOPAL,
        component_id=2,
    )
    first_u = rocheometry.convert_potential(
        first,
        q=q,
        convention=rocheometry.PotentialConvention.JACOBI,
    )
    second_u = rocheometry.convert_potential(
        second,
        q=q,
        convention=rocheometry.PotentialConvention.JACOBI,
    )

    assert second.component_id == 2
    assert second_u.value == pytest.approx(first_u.value, rel=3.0e-15)


def test_binary_conversion_helper_uses_binary_mass_ratio() -> None:
    binary = rocheometry.BinaryGeometry(q=0.4)
    canonical = rocheometry.Potential.jacobi(2.0)

    actual = binary.convert_potential(
        canonical,
        convention=rocheometry.PotentialConvention.KOPAL,
        component_id=2,
    )
    expected = rocheometry.convert_potential(
        canonical,
        q=0.4,
        convention=rocheometry.PotentialConvention.KOPAL,
        component_id=2,
    )

    assert actual == expected
