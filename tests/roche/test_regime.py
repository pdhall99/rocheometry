"""Level classification uses supplied critical values without solving geometry."""

from math import inf, nextafter

import pytest

import rocheometry
from rocheometry._roche import LevelRegime, level_regime


def test_regime_exports_and_members() -> None:
    assert rocheometry.LevelRegime is LevelRegime
    assert {member.name for member in LevelRegime} == {
        "DETACHED",
        "CRITICAL",
        "CONTACT",
        "OPEN",
    }


@pytest.mark.parametrize(
    "depth, expected",
    [
        (-0.1, LevelRegime.DETACHED),
        (-nextafter(0.0, 1.0), LevelRegime.DETACHED),
        (0.0, LevelRegime.CRITICAL),
        (nextafter(0.0, 1.0), LevelRegime.CONTACT),
        (1e-30, LevelRegime.CONTACT),
        (nextafter(0.2, 0.0), LevelRegime.CONTACT),
        (0.2, LevelRegime.OPEN),
        (nextafter(0.2, inf), LevelRegime.OPEN),
    ],
)
def test_pure_regime_boundaries(depth: float, expected: LevelRegime) -> None:
    assert level_regime(depth, 0.2) is expected
