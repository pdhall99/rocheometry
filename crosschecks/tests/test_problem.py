"""The shared field agrees across number types and with the library's problem."""

from decimal import localcontext
from fractions import Fraction

import mpmath as mp
import pytest

import rocheometry
from crosschecks.problem import (
    Binary,
    Level,
    axis_potential,
    exact_values,
    gradient,
    masses,
    potential,
)

POINTS = [(0.3, 0.2, 0.1), (-0.4, 0.0, 0.25), (1.2, -0.1, 0.0)]


@pytest.mark.parametrize("qi", [1e-6, 0.37, 1.0, 50.0])
@pytest.mark.parametrize("point", POINTS)
def test_decimal_and_mpmath_agree(qi: float, point: tuple[float, float, float]) -> None:
    binary = Binary.of(rocheometry.ComponentGeometry(qi).field)
    with localcontext() as context:
        context.prec = 100
        u, g = exact_values(binary, tuple(map(Fraction, point)))
    with mp.workdps(60):
        m, c = masses(binary, mp.mpf)
        x = tuple(map(mp.mpf, point))
        assert abs(mp.mpf(str(u)) - potential(m, c, *x)) < mp.mpf("1e-55")
        for a, b in zip(g, gradient(m, c, *x), strict=True):
            assert abs(mp.mpf(str(a)) - b) < mp.mpf("1e-55")


@pytest.mark.parametrize("point", POINTS)
def test_gradient_is_the_derivative_of_the_potential(
    point: tuple[float, float, float],
) -> None:
    m, c = masses(Binary.of(rocheometry.ComponentGeometry(0.37).field), mp.mpf)
    with mp.workdps(40):
        x = list(map(mp.mpf, point))
        for j, value in enumerate(gradient(m, c, *x)):

            def along(t: mp.mpf, j: int = j) -> mp.mpf:
                return potential(m, c, *(x[:j] + [t] + x[j + 1 :]))

            assert abs(mp.diff(along, x[j]) - value) < mp.mpf("1e-30")


def test_axis_potential_is_exact_and_matches_the_potential() -> None:
    binary = Binary.of(rocheometry.ComponentGeometry(0.37).field)
    exact = axis_potential(binary, 0.6)
    assert isinstance(exact, Fraction)
    with mp.workdps(40):
        m, c = masses(binary, mp.mpf)
        on_axis = potential(m, c, mp.mpf(0.6), mp.mpf(0), mp.mpf(0))
        assert abs(mp.mpf(exact.numerator) / exact.denominator - on_axis) < mp.mpf(
            "1e-35"
        )


def test_level_reads_the_surface_problem() -> None:
    g = rocheometry.ComponentGeometry(0.37)
    surface = g.surface(volume_radius_fill=0.8)
    level = Level.of(surface)
    assert level.binary == Binary(g.field.component_mass, g.field.companion_mass)
    assert level.l1_x == g.l1().position[0]
    # Detached levels lie above L1's potential, so their depth is negative.
    assert level.depth == surface._depth < 0
    assert level.value() == axis_potential(level.binary, level.l1_x) - Fraction(
        level.depth
    )
    # The level's potential is the surface's, to the surface's rounding.
    assert float(level.value()) == pytest.approx(surface.potential.value, rel=1e-15)
