"""Opposite charts retain the original float mass pair, including in pickles."""

import pickle
from dataclasses import FrozenInstanceError

import pytest

from rocheometry._roche.field import RocheField


@pytest.mark.parametrize("qi", (1e-8, 1e-6, 1e-3, 0.7, 1.0, 1e3, 1e6, 1e8))
def test_reciprocal_swaps_stored_masses_without_recomputing(qi: float) -> None:
    field = RocheField(qi)
    opposite = field.reciprocal()
    assert opposite.qi == 1 / field.qi
    assert opposite.component_mass == field.companion_mass
    assert opposite.companion_mass == field.component_mass
    twice = opposite.reciprocal()
    assert (twice.component_mass, twice.companion_mass) == (
        field.component_mass,
        field.companion_mass,
    )
    restored = pickle.loads(pickle.dumps(opposite))
    assert restored == opposite
    assert hash(restored) == hash(opposite)
    assert restored.component_mass == field.companion_mass
    assert restored.companion_mass == field.component_mass
    with pytest.raises(FrozenInstanceError):
        setattr(opposite, "component_mass", 0.5)


def test_equal_labels_with_different_mass_pairs_have_distinct_cache_keys() -> None:
    field = RocheField(1e-8)
    exact_opposite = field.reciprocal()
    rounded_opposite = RocheField(1 / field.qi)
    assert exact_opposite.qi == rounded_opposite.qi
    assert exact_opposite != rounded_opposite
    assert len({exact_opposite, rounded_opposite}) == 2
