"""The float and rational evaluators use the stored masses' own total."""

import math
import random
from fractions import Fraction

import pytest

import rocheometry
from rocheometry._levels import _levels
from rocheometry._roche.field import RocheField, _mass_correction


def exact(
    field: RocheField,
    point: tuple[Fraction, Fraction, Fraction],
    r0: Fraction,
    r1: Fraction,
) -> Fraction:
    m, c = Fraction(field.component_mass), Fraction(field.companion_mass)
    total = m + c
    x, y, _ = point
    return m / r0 + c / r1 + total * ((x - c / total) ** 2 + y * y) / 2


def test_exact_reciprocal_model_at_random_rational_points() -> None:
    rng = random.Random(2026102502)
    for _ in range(100):
        field = RocheField(10 ** rng.uniform(-8, 8))
        opposite = field.reciprocal()
        t = Fraction(rng.randrange(1, 50), 64)
        v = Fraction(rng.randrange(1, 50), 32)
        w = Fraction(rng.randrange(1, 50), 32)
        total = (1 + t * t) / (1 - t * t)
        delta = (1 - v * v) / (1 + v * v)
        r0, r1 = (total + delta) / 2, (total - delta) / 2
        x = (1 + total * delta) / 2
        rho = 2 * t * v / ((1 - t * t) * (1 + v * v))
        y = rho * (1 - w * w) / (1 + w * w)
        z = rho * 2 * w / (1 + w * w)
        assert x * x + y * y + z * z == r0 * r0
        assert (x - 1) ** 2 + y * y + z * z == r1 * r1
        assert exact(field, (x, y, z), r0, r1) == exact(
            opposite, (1 - x, -y, z), r1, r0
        )


@pytest.mark.parametrize("qi", (1e-8, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e8))
def test_exact_chart_critical_levels_and_gaps_agree_relative_to_gap(qi: float) -> None:
    g = rocheometry.ComponentGeometry(qi)
    a = _levels(g.field, g._computers)
    b = _levels(g.field.reciprocal(), g._computers)
    tolerance = Fraction.from_float(1e-20) * min(a.l1 - a.opening, b.l1 - b.opening)
    assert abs(a.l1 - b.l1) <= tolerance
    assert abs((a.l1 - a.opening) - (b.l1 - b.opening)) <= tolerance


def test_mass_error_guard_fails_loudly() -> None:
    for pair in ((1.0, 1.0), (math.nextafter(1.0, 0.0), 1e-300)):
        with pytest.raises(ValueError, match="mass-pair error"):
            _mass_correction(*pair)
