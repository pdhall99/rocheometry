"""Residual decisions are shared by a whole surface and depend on accuracy."""

from dataclasses import replace
from sys import float_info

import pytest

import rocheometry
from rocheometry._computers.defaults import _Settings
from rocheometry._computers.surface_radius.policy import _UseExactAnchor
from rocheometry._computers.surface_volume import SampledVolumeComputer


@pytest.mark.parametrize("qi", (1e-8, 1e-6, 1e-5, 2e-5, 3e-5, 1e-3, 1.0, 1e3))
def test_fitted_surface_decision_matches_the_volume_radius_rule(qi: float) -> None:
    geometry = rocheometry.ComponentGeometry(qi)
    surface = geometry.surface(volume_radius_fill=0.8)
    graph = geometry._computers
    decision = graph.radius.exact_anchor  # type: ignore[missing-attribute]
    expected = (
        4
        * float_info.epsilon
        * abs(surface.potential.value)
        * surface.volume_radius()
        / geometry.field.component_mass
        > 1e-12
    )
    assert decision(geometry.field, surface._depth) is expected
    # No direction enters the cached decision, even for near-axis rays.
    before = decision.cache_info()
    decision(geometry.field, surface._depth)
    assert decision.cache_info().hits == before.hits + 1


def test_reference_anchor_policy_is_an_explicit_settings_value() -> None:
    assert rocheometry.Reference()._computers().radius.exact_anchor is True  # type: ignore[missing-attribute]
    settings = _Settings()
    exact = replace(settings, exact_anchor_everywhere=True)
    assert exact != settings
    assert exact._computers() is not settings._computers()
    with pytest.raises(TypeError, match="exact_anchor_everywhere"):
        replace(settings, exact_anchor_everywhere=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("qi", (1e-8, 1e-6, 1e-3, 1.0, 1e3))
def test_certified_bounds_avoid_a_volume_prepass(qi: float) -> None:
    geometry = rocheometry.ComponentGeometry(qi)
    surface = geometry.surface(volume_radius_fill=0.8)
    graph = geometry._computers

    def forbidden_volume(field: object, potential: object) -> float:
        raise AssertionError("a certified shortcut should not sample the surface")

    policy = _UseExactAnchor(graph.l1, graph.outer, forbidden_volume)
    assert policy(geometry.field, surface._depth) is (qi < 1e-5)


def test_transition_band_measures_the_actual_volume() -> None:
    geometry = rocheometry.ComponentGeometry(2e-5)
    surface = geometry.surface(volume_radius_fill=0.8)
    graph = geometry._computers
    calls = 0
    volume = SampledVolumeComputer(sample=graph.sample)

    def counted_volume(field: object, potential: object) -> float:
        nonlocal calls
        calls += 1
        return volume(geometry.field, surface._depth)

    policy = _UseExactAnchor(graph.l1, graph.outer, counted_volume)
    assert policy(geometry.field, surface._depth) is False
    assert calls == 1
