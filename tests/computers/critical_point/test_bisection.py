from decimal import Decimal, localcontext
from math import isfinite, nextafter, ulp
from sys import float_info

import pytest
from typing_extensions import override

import rocheometry
from rocheometry._computers.critical_point import (
    BisectionCriticalPointComputer,
)
from rocheometry._roche import RocheField
from rocheometry._roche.mass import mass_fractions


def _decimal_critical_distance(qi: float, *, outer: bool) -> float:
    """Independently solve the original force balance at high precision."""
    with localcontext() as context:
        context.prec = 400
        component, companion = map(Decimal.from_float, mass_fractions(qi))
        reflected = not outer and qi > 1.0
        mass = companion if reflected else component
        scale = mass ** (Decimal(1) / 3)
        lower = scale / 8
        upper = min(2 * scale, Decimal(1) if outer else Decimal("0.5"))

        def force(distance: Decimal) -> Decimal:
            x = -distance if outer else 1 - distance if reflected else distance
            derivative = (
                -component * x / abs(x) ** 3
                - companion * (x - 1) / abs(x - 1) ** 3
                + (component + companion) * x
                - companion
            )
            return -derivative if reflected or outer else derivative

        assert force(lower) < 0
        assert force(upper) >= 0
        for _ in range(160):
            midpoint = (lower + upper) / 2
            value = force(midpoint)
            if value == 0:
                return float(midpoint)
            if value < 0:
                lower = midpoint
            else:
                upper = midpoint
        return float((lower + upper) / 2)


@pytest.mark.parametrize(
    "qi",
    [
        nextafter(0.0, 1.0),
        1.0e-300,
        1.0e-120,
        1.0e-60,
        1.0e-40,
        1.0e-36,
        1.0e-24,
        1.0e-12,
        1.0e-4,
        1.0,
        1.0e4,
        1.0e12,
        1.0e24,
        1.0e36,
        1.0e40,
        1.0e46,
    ],
)
@pytest.mark.parametrize("outer", [False, True])
def test_adaptive_critical_distances_match_high_precision_force_balance(
    qi: float, outer: bool
) -> None:
    geometry = rocheometry.ComponentGeometry(qi)
    point = geometry.outer() if outer else geometry.l1()
    x = point.position[0]
    reflected = not outer and qi > 1.0
    distance = -x if outer else 1.0 - x if reflected else x
    expected = _decimal_critical_distance(qi, outer=outer)
    # Companion-side distances inherit the spacing of coordinates near x=1.
    assert distance == pytest.approx(
        expected, rel=8.0e-15, abs=ulp(x) if reflected else 0.0
    )
    assert point.potential.value == float(geometry.field._axis_potential_exact(x))


@pytest.mark.parametrize("qi", [1.0e48, 1.0e120, float_info.max])
def test_unresolvable_l1_reports_its_numerical_limit(qi: float) -> None:
    if qi == float_info.max:
        # The stored pair's exact sum error cannot itself be represented.
        with pytest.raises(ValueError, match="mass-pair error"):
            rocheometry.ComponentGeometry(qi)
        return
    geometry = rocheometry.ComponentGeometry(qi)
    with pytest.raises(ValueError, match=r"l1.*numerical resolution.*qi="):
        geometry.l1()
    assert geometry.outer().position[0] == pytest.approx(-1.0, abs=4.0e-15)
    assert isfinite(geometry.outer().potential.value)


@pytest.mark.parametrize("qi", [1.0e-40, 1.0e-24, 1.0e-4, 1.0, 1.0e4, 1.0e24, 1.0e40])
def test_extreme_l1_obeys_component_exchange(qi: float) -> None:
    point = rocheometry.ComponentGeometry(qi).l1()
    reciprocal = rocheometry.ComponentGeometry(1.0 / qi).l1()
    assert point.position[0] == pytest.approx(
        1.0 - reciprocal.position[0], rel=8.0e-15, abs=2.0 * float_info.epsilon
    )
    assert point.potential.value == pytest.approx(
        reciprocal.potential.value, rel=2.0e-15
    )


class _ShiftedGradientField(RocheField):
    @override
    def gradient(
        self, x: float, y: float = 0.0, z: float = 0.0
    ) -> tuple[float, float, float]:
        return (x + 0.6 if x < 0.0 else x - 0.75, 0.0, 0.0)


@pytest.mark.parametrize("qi", [1.0e-120, 0.3, 100.0])
def test_adaptive_brackets_use_the_actual_supplied_field(qi: float) -> None:
    field = _ShiftedGradientField(qi)
    assert BisectionCriticalPointComputer.l1()(field).position[0] == pytest.approx(
        0.75, abs=2.0e-15
    )
    assert BisectionCriticalPointComputer.outer()(field).position[0] == pytest.approx(
        -0.6, abs=2.0e-15
    )


def test_explicit_critical_brackets_remain_fixed() -> None:
    field = _ShiftedGradientField(0.3)
    computer = BisectionCriticalPointComputer(0.7, 0.9)
    assert computer(field).position[0] == pytest.approx(0.75, abs=1.0e-14)
    with pytest.raises(ValueError, match="root is not bracketed"):
        BisectionCriticalPointComputer(0.1, 0.6)(field)


def test_adaptive_and_fixed_bracket_configuration_cannot_be_mixed() -> None:
    with pytest.raises(ValueError, match="adaptive solves do not accept fixed bounds"):
        BisectionCriticalPointComputer(0.1, 0.9, branch="l1")
    with pytest.raises(ValueError, match="both lower and upper are required"):
        BisectionCriticalPointComputer(lower=0.1)
