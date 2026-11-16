"""Radius models can supply volume-facing calibration without rebinding fields."""

import pytest

from rocheometry._computers.lobe_volume import (
    LobeVolumeComputer,
    LobeVolumeFromRadiusComputer,
)
from rocheometry._computers.lobe_volume_radius.fitted import (
    InterpolatedLobeVolumeRadiusComputer,
)
from rocheometry._roche.field import RocheField
from rocheometry._roche.volume import sphere_volume, volume_radius_from_volume


def test_radius_adapter_is_lazy_and_passes_the_actual_field_to_its_computer() -> None:
    supplied: list[RocheField] = []

    def radius(field: RocheField, /) -> float:
        supplied.append(field)
        return 0.25 * field.qi

    volume: LobeVolumeComputer = LobeVolumeFromRadiusComputer(radius)
    assert supplied == []
    field = RocheField(0.4)
    assert volume_radius_from_volume(volume(field)) == pytest.approx(0.1)
    assert supplied == [field]
    assert supplied[0] is field


def test_radius_adapter_retains_the_underlying_models_mass_ratio_domain() -> None:
    radius = InterpolatedLobeVolumeRadiusComputer(log_qis=(-1.0, 0.0), radii=(0.2, 0.4))
    volume: LobeVolumeComputer = LobeVolumeFromRadiusComputer(radius)
    assert volume(RocheField(0.1)) == sphere_volume(0.2)
    assert volume(RocheField(1.0)) == sphere_volume(0.4)
    with pytest.raises(ValueError, match="volume-radius model requires"):
        volume(RocheField(10.0))
