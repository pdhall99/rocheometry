"""Numerical producers reject failed integrals before memoization."""

from types import ModuleType

import pytest

import rocheometry
import rocheometry._computers.mean_curvature.sampled as curvature
import rocheometry._computers.surface_area.sampled as area
import rocheometry._computers.surface_volume.sampled as volume
from rocheometry._computers.composition import Memoized
from rocheometry._computers.surface_sample import SurfaceSample
from rocheometry._roche.field import RocheField
from tests._helpers import _depth


@pytest.mark.parametrize("module", (area, curvature))
@pytest.mark.parametrize("value", (0.0, -1.0, float("nan"), float("inf")))
def test_failed_sampled_integrals_are_not_cached(
    module: ModuleType, value: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = SurfaceSample(((((0.0, 0.0, 1.0)), 1.0),), (0.1,))

    def sampler(field: RocheField, potential: float, /) -> SurfaceSample:
        return sample

    monkeypatch.setattr(
        module, "ray_terms", lambda *args, **kwargs: (value, value, 0.0)
    )
    producer = (
        area.SampledAreaComputer(sampler)
        if module is area
        else curvature.SampledMeanCurvatureComputer(sampler)
    )
    node = Memoized(producer)
    for _ in range(2):
        with pytest.raises(ValueError):
            node(
                RocheField(0.3),
                _depth(RocheField(0.3), rocheometry.Potential.jacobi(2.0)),
            )
    assert node.cache_info().misses == 2
    assert node.cache_info().currsize == 0


def test_contact_volume_rejects_failed_gravity_flux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = SurfaceSample(((((0.0, 0.0, 1.0)), 1.0),), (0.1,), part=True)
    monkeypatch.setattr(
        volume, "ray_gravity_flux", lambda *args, **kwargs: float("inf")
    )
    producer = volume.SampledVolumeComputer(lambda field, potential: sample)
    with pytest.raises(ValueError):
        producer(
            RocheField(0.3), _depth(RocheField(0.3), rocheometry.Potential.jacobi(2.0))
        )
