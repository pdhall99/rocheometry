"""Mean gravity and public quadrature are checked, shared, replaceable nodes."""

import pickle
from dataclasses import replace
from typing import cast

import pytest

import rocheometry
from rocheometry import Fitted
from rocheometry._computers import defaults
from rocheometry._computers.composition import Memoized
from rocheometry._computers.mean_gravity import (
    SampledMeanGravityComputer,
)
from rocheometry._computers.quadrature import SampledQuadratureComputer
from rocheometry._computers.surface_sample import SurfaceSampler
from rocheometry._roche.field import RocheField
from tests._helpers import _Wired, computer_graph


@pytest.mark.parametrize("reference", (False, True))
@pytest.mark.parametrize("level", ("detached", "critical", "contact"))
def test_packaged_readers_share_sample_and_validate_quadrature_once(
    reference: bool, level: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = (
        defaults._wire_computers(
            defaults._Settings(
                polar_order=128,
                azimuth_order=64,
                fill_tolerance=1e-13,
                watershed_tolerance=1e-13,
                fitted_selection=False,
                exact_anchor_everywhere=True,
                fixed_paths=False,
                path_tolerance=1e-10,
            )
        )
        if reference
        else defaults._wire_computers(defaults._Settings())
    )
    geometry = rocheometry.ComponentGeometry(1.0, accuracy=_Wired(graph))
    surface = (
        geometry.surface(fill_out_factor=0.5)
        if level == "contact"
        else geometry.surface(volume_radius_fill=1.0 if level == "critical" else 0.99)
    )
    validations = []
    original = rocheometry.SurfaceQuadrature.__post_init__

    def validate(value: rocheometry.SurfaceQuadrature) -> None:
        validations.append(value)
        original(value)

    monkeypatch.setattr(rocheometry.SurfaceQuadrature, "__post_init__", validate)
    quadrature = surface.quadrature()
    gravity = surface.mean_gravity()
    sample_info = cast(Memoized[..., object], graph.sample).cache_info()
    assert surface.quadrature() is quadrature
    assert surface.mean_gravity() is gravity
    assert validations == [quadrature]
    assert cast(Memoized[..., object], graph.sample).cache_info() == sample_info
    for name, maxsize in (
        ("mean_gravity", cast(Memoized[..., object], graph.area).maxsize),
        ("quadrature", 4),
    ):
        node = cast(Memoized[..., object], getattr(graph, name))
        assert node.cache_info().hits == node.cache_info().misses == 1
        assert node.cache_info().maxsize == maxsize
        assert getattr(node.computer, "sample") is graph.sample


def test_custom_nodes_change_component_and_binary_readers() -> None:
    quadrature = rocheometry.SurfaceQuadrature(
        ((0.1, 0.0, 0.0),), ((1.0, 0.0, 0.0),), (7.0,), (2.0,), (1.0,), (1.0,)
    )
    calls = []

    def mean(field: RocheField, potential: float, /) -> float:
        calls.append(("gravity", field, potential))
        return 7.0

    def nodes(field: RocheField, potential: float, /) -> rocheometry.SurfaceQuadrature:
        calls.append(("quadrature", field, potential))
        return quadrature

    graph = replace(
        Fitted()._computers(),
        mean_gravity=Memoized(mean),
        quadrature=Memoized(nodes),
    )
    component = rocheometry.ComponentGeometry(1.0, accuracy=_Wired(graph))
    surface = component.lobe()
    assert surface.mean_gravity() == 7.0
    assert surface.quadrature() is quadrature
    for frame in ("binary", "centre_of_mass"):
        for component_id in (1, 2):
            view = (
                rocheometry.BinaryGeometry(
                    1.0, separation=3.0, frame=frame, accuracy=_Wired(graph)
                )
                .component(component_id)
                .lobe()
            )
            assert view.mean_gravity() == 7.0
            assert view.quadrature().weights == (18.0,)
            assert view.quadrature().gravities == (7.0,)
    assert len(calls) == 2


@pytest.mark.parametrize("maxsize", (0, 2, None))
def test_reader_caches_follow_area_capacity_and_pickle_empty(
    maxsize: int | None,
) -> None:
    graph = computer_graph(maxsize=maxsize)
    geometry = rocheometry.ComponentGeometry(1.0, accuracy=_Wired(graph))
    first = geometry.lobe()
    original = first.quadrature()
    first.mean_gravity()
    for offset in (0.1, 0.2):
        surface = geometry.surface(
            potential=rocheometry.Potential.jacobi(first.potential.value + offset)
        )
        surface.quadrature()
        surface.mean_gravity()
    for name in ("mean_gravity", "quadrature"):
        info = cast(Memoized[..., object], getattr(graph, name)).cache_info()
        assert info.maxsize == maxsize
        assert info.currsize == (3 if maxsize is None else maxsize)
    assert (first.quadrature() is original) is (maxsize is None)
    restored = pickle.loads(pickle.dumps(graph))
    for name in ("mean_gravity", "quadrature"):
        info = getattr(restored, name).cache_info()
        assert info.maxsize == maxsize
        assert info.currsize == 0
    assert (
        rocheometry.ComponentGeometry(1.0, accuracy=_Wired(restored))
        .lobe()
        .quadrature()
        == original
    )


@pytest.mark.parametrize(
    "computer", (SampledMeanGravityComputer, SampledQuadratureComputer)
)
def test_sampled_readers_require_a_callable_sample(
    computer: type[SampledMeanGravityComputer] | type[SampledQuadratureComputer],
) -> None:
    with pytest.raises(TypeError, match="sample: expected a callable"):
        computer(sample=cast(SurfaceSampler, None))
