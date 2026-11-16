import pytest

from rocheometry import kepler_separation


def test_one_solar_mass_one_day() -> None:
    a = kepler_separation(period_days=1.0, total_mass_msun=1.0)
    assert a == pytest.approx(4.208278345829611, rel=1e-14)


def test_earth_orbit_is_one_au() -> None:
    a = kepler_separation(period_days=365.256363004, total_mass_msun=1.0 + 3.003489e-6)
    assert a == pytest.approx(1.495978707e11 / 6.957e8, rel=1e-7)


def test_scales_as_mass_third_period_two_thirds() -> None:
    base = kepler_separation(period_days=1.0, total_mass_msun=1.0)
    scaled = kepler_separation(period_days=8.0, total_mass_msun=27.0)
    assert scaled == pytest.approx(12.0 * base, rel=1e-14)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_rejects_invalid_period(bad: float) -> None:
    with pytest.raises(ValueError):
        kepler_separation(period_days=bad, total_mass_msun=1.0)
