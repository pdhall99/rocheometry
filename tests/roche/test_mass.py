import pytest

from rocheometry._roche import component_mass_ratio, mass_fractions


@pytest.mark.parametrize("q", [1.0e-4, 0.37, 1.0, 2.7, 1.0e4])
def test_mass_fractions_normalize_component_local_ratio(q: float) -> None:
    first, second = mass_fractions(q)

    assert first + second == pytest.approx(1.0)
    assert first / second == pytest.approx(q)
    assert mass_fractions(component_mass_ratio(q, 1)) == (first, second)
    assert mass_fractions(component_mass_ratio(q, 2)) == pytest.approx((second, first))
