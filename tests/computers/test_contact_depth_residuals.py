"""Depth offsets keep the bits of literal levels and compensate outer gaps."""

import math
from fractions import Fraction

import pytest

import rocheometry


@pytest.mark.parametrize("qi", (1e-8, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e8))
def test_compensated_outer_offset_matches_exact_gap(qi: float) -> None:
    g = rocheometry.ComponentGeometry(qi)
    f = g.field
    x1, xo = g.l1().position[0], g.outer().position[0]
    exact = f._axis_potential_exact(x1) - f._axis_potential_exact(xo)
    high, low = f._axis_depth_parts(x1, xo)
    assert high == float(exact)
    assert low == float(exact - Fraction(high))
    for depth in (0.0, high, math.nextafter(high, 0.0), math.nextafter(high, math.inf)):
        actual = f._depth_offset(depth, x1, xo)
        bound = Fraction(math.ulp(low)) / 2 + Fraction(math.ulp(actual)) / 2
        assert abs(Fraction(actual) - (Fraction(depth) - exact)) <= bound


@pytest.mark.parametrize("qi", (1e-8, 1e-3, 1.0, 1e3, 1e8))
def test_unfactored_report_rounds_once_without_fraction_per_level(qi: float) -> None:
    g = rocheometry.ComponentGeometry(qi)
    x1 = g.l1().position[0]
    exact = g.field._axis_potential_exact(x1)
    for depth in (0.0, -1.0, 1e-20, 1e-60, 1e-200, math.nextafter(0.0, 1.0)):
        assert g.field._level_value(x1, depth) == float(exact - Fraction(depth))


@pytest.mark.parametrize("qi", (1e-8, 1e-6, 1e-3, 1e3, 1e6, 1e8))
@pytest.mark.parametrize("accuracy", (rocheometry.Fitted(), rocheometry.Reference()))
def test_twelve_d1_band_snaps_are_now_computer_contact(
    qi: float,
    accuracy: rocheometry.Fitted | rocheometry.Reference,
) -> None:
    from rocheometry._computers.contact import _geometry_depth

    g = rocheometry.ComponentGeometry(qi, accuracy=accuracy)
    s = g.surface(fill_out_factor=1e-12)
    assert s._depth > 0
    assert s.regime is rocheometry.LevelRegime.CONTACT
    assert _geometry_depth(s._depth, g.l1()) == s._depth
    rim = g._computers.watershed.rim(g.field, s._depth, math.pi / 3)
    assert math.hypot(*rim[1:]) > 0
