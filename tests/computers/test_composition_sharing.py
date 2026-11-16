"""Private graph sharing is observable through actual result-cache reuse."""

from dataclasses import replace
from typing import cast

import pytest

import rocheometry
from rocheometry._accuracy import _Accuracy
from rocheometry._computers.composition import ComputerSet, Memoized
from rocheometry._computers.surface_sample import SurfaceSample
from rocheometry._computers.surface_volume import SampledVolumeComputer
from tests._helpers import _Wired


@pytest.mark.parametrize(
    "accuracy",
    (rocheometry.Fitted(), rocheometry.Reference(polar_order=16, azimuth_order=8)),
)
def test_scalar_readers_and_quadrature_share_one_radius_sample(
    accuracy: _Accuracy,
) -> None:
    geometry = rocheometry.ComponentGeometry(0.3, accuracy=accuracy)
    graph = geometry._computers
    sample = cast(Memoized[..., SurfaceSample], graph.sample)
    sample.cache_clear()
    surface = geometry.surface(
        potential=rocheometry.Potential.jacobi(geometry.l1().potential.value * 1.1)
    )
    volume = SampledVolumeComputer(sample=graph.sample)
    volume(geometry.field, surface._depth)
    assert sample.cache_info().misses == 1
    before = sample.cache_info().hits
    surface.area()
    surface.integrated_mean_curvature()
    surface.mean_gravity()
    surface.quadrature()
    assert sample.cache_info().misses == 1
    assert sample.cache_info().hits == before + 4


def test_both_binary_components_share_the_resolved_graph() -> None:
    binary = rocheometry.BinaryGeometry(
        0.3, accuracy=rocheometry.Reference(polar_order=16, azimuth_order=8)
    )
    assert (
        binary._computers is binary.geometry1._computers is binary.geometry2._computers
    )
    assert binary.geometry1.accuracy is binary.accuracy is binary.geometry2.accuracy


def test_injected_strategies_remain_replaceable_inside_private_graph() -> None:
    graph = rocheometry.Fitted()._computers()
    graph = replace(graph, area=lambda field, potential: 12.0)
    assert (
        rocheometry.ComponentGeometry(0.3, accuracy=_Wired(graph)).lobe().area() == 12.0
    )


def test_binary_resolves_accuracy_once_even_if_factory_does_not_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def resolve(accuracy: rocheometry.Reference) -> ComputerSet:
        calls.append(accuracy)
        return replace(rocheometry.Fitted()._computers())

    monkeypatch.setattr(rocheometry.Reference, "_computers", resolve)
    accuracy = rocheometry.Reference()
    binary = rocheometry.BinaryGeometry(0.3, accuracy=accuracy)
    assert calls == [accuracy]
    assert (
        binary._computers is binary.geometry1._computers is binary.geometry2._computers
    )
