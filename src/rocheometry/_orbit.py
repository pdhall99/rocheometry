"""Orbital scale from Kepler's third law, in nominal solar units."""

from math import pi

from rocheometry._validation import normalize_mass, normalize_period

# Derived: nominal solar mass parameter.
NOMINAL_SOLAR_MASS_PARAMETER = 1.3271244e20
"""IAU 2015 nominal solar mass parameter GM_sun (Resolution B3), in m^3 s^-2."""
# Derived: nominal solar radius.
NOMINAL_SOLAR_RADIUS = 6.957e8
"""IAU 2015 nominal solar radius (Resolution B3), in m."""
# Derived: seconds per day.
SECONDS_PER_DAY = 86400.0
# Derived: kepler solar unit factor.
KEPLER_SOLAR_UNIT_FACTOR = (
    NOMINAL_SOLAR_MASS_PARAMETER
    * SECONDS_PER_DAY**2
    / (4.0 * pi**2 * NOMINAL_SOLAR_RADIUS**3)
)  # about 74.527 R_sun^3 per (M_sun day^2)


def kepler_separation(*, period_days: float, total_mass_msun: float) -> float:
    """Return the orbital separation in nominal solar radii.

    Uses Kepler's third law, a^3 = G (M1 + M2) P^2 / (4 pi^2), with the IAU 2015
    nominal solar mass parameter and radius (Prša et al. 2016,
    https://doi.org/10.3847/0004-6256/152/2/41). Working from GM_sun rather
    than G and M_sun separately avoids the uncertainty in G. One solar mass on
    a one-day orbit gives about 4.208 R_sun.

    Args:
        period_days: Orbital period in days of 86400 s. Must be finite and positive.
        total_mass_msun: Total mass M1 + M2 in nominal solar masses. Must be
            finite and positive.
    """
    period = normalize_period(period_days, name="period_days")
    mass = normalize_mass(total_mass_msun, name="total_mass_msun")
    return (KEPLER_SOLAR_UNIT_FACTOR * mass) ** (1.0 / 3.0) * period ** (2.0 / 3.0)
