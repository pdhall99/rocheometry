"""Surface reports measure one sample; fitted population paths never sample."""

import math
from typing import NoReturn, cast

import pytest

import rocheometry
from rocheometry._computers.composition import Memoized
from rocheometry._computers.defaults import _Settings, _wire_computers
from rocheometry._computers.surface_sample import GaussLegendreSurfaceSampler
from rocheometry._computers.surface_volume import SampledVolumeComputer
from tests._helpers import _Wired


def test_fill_reuses_surface_sample_and_caches_measured_lobe() -> None:
    graph = _wire_computers(_Settings())
    g = rocheometry.ComponentGeometry(0.3, accuracy=_Wired(graph))
    s = g.surface(volume_radius_fill=0.8)
    sample = cast(Memoized[..., object], graph.sample)
    assert sample.cache_info().misses == 0
    s.volume()
    assert sample.cache_info().misses == 1
    s.volume_radius_fill()
    assert sample.cache_info().misses == 2
    info = sample.cache_info()
    s.volume_radius_fill()
    g.lobe().volume()
    assert sample.cache_info() == info


def test_surface_reports_do_not_need_either_packaged_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rocheometry._computers import packaged

    def unexpected() -> NoReturn:
        raise AssertionError("surface measurements must not read a fitted map")

    monkeypatch.setattr(packaged, "_packaged_surface_potential_map", unexpected)
    monkeypatch.setattr(packaged, "_packaged_lobe_volume_radius_model", unexpected)
    g = rocheometry.ComponentGeometry(
        0.3, accuracy=_Wired(_wire_computers(_Settings()))
    )
    for factor in (-0.5, 0.0, 0.5):
        s = g.surface(fill_out_factor=factor)
        assert s.volume() > 0
        assert s.volume_radius() > 0
        assert s.volume_radius_fill() > 0
        assert s.area() > 0
        assert s.mean_gravity() > 0


@pytest.mark.parametrize("qi", (1e-8, 1e-3, 1.0, 1e3, 1e8))
def test_population_paths_use_no_sampler(
    qi: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("fitted population paths must not sample")

    monkeypatch.setattr(GaussLegendreSurfaceSampler, "__call__", unexpected)
    graph = _wire_computers(_Settings())
    g = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph))
    radius = g.lobe_volume_radius()
    assert radius > 0
    assert 0 < g.lobe_volume_radius_slope() < 1 / 3
    for fill in (0.3, 0.8, 0.99, 1.0):
        assert g.surface(volume_radius_fill=fill)
        assert g.surface(volume_radius=fill * radius)
        assert g.volume_radius_fill_for(fill * radius) == pytest.approx(fill, rel=1e-15)
    for component in (1, 2):
        view = rocheometry.BinaryGeometry(
            qi, separation=3, accuracy=_Wired(graph)
        ).component(component)
        radius = view.lobe_volume_radius()
        assert view.surface(volume_radius=radius * 0.8)
        assert view.volume_radius_fill_for(radius * 0.8) == pytest.approx(
            0.8, rel=1e-15
        )


@pytest.mark.parametrize("accuracy", (rocheometry.Fitted(), rocheometry.Reference()))
@pytest.mark.parametrize("kind", ("detached", "critical", "contact"))
def test_volume_and_fill_are_measured_with_one_lobe_normalization(
    accuracy: rocheometry.Fitted | rocheometry.Reference, kind: str
) -> None:
    g = rocheometry.ComponentGeometry(0.001, accuracy=accuracy)
    s = (
        g.surface(volume_radius_fill=0.8)
        if kind == "detached"
        else g.lobe()
        if kind == "critical"
        else g.surface(fill_out_factor=0.5)
    )
    measure = SampledVolumeComputer(g._computers.sample)
    assert s.volume() == measure(g.field, s._depth)
    assert s.volume_radius() == (3 * s.volume() / (4 * math.pi)) ** (1 / 3)
    assert s.volume_radius_fill() == (s.volume() / measure(g.field, 0.0)) ** (1 / 3)
    if kind == "critical":
        assert s.volume_radius_fill() == 1.0
    assert g.lobe_volume_radius() == pytest.approx(
        g.lobe().volume_radius(),
        rel=1e-6 if isinstance(accuracy, rocheometry.Fitted) else 1e-10,
    )


@pytest.mark.parametrize("qi", (1e-8, 1e-3, 1.0, 1e3, 1e8))
@pytest.mark.parametrize("accuracy", (rocheometry.Fitted(), rocheometry.Reference()))
def test_shallow_depth_never_becomes_a_critical_surface(
    qi: float, accuracy: rocheometry.Fitted | rocheometry.Reference
) -> None:
    g = rocheometry.ComponentGeometry(qi, accuracy=accuracy)
    for factor in (-1e-9, 1e-9):
        s = g.surface(fill_out_factor=factor)
        assert s.regime is (
            rocheometry.LevelRegime.CONTACT
            if factor > 0
            else rocheometry.LevelRegime.DETACHED
        )
        # The regime is exact. The measured fill is within quadrature noise of
        # one this close to L1, as volume_radius_fill documents: only beyond
        # that noise must it lie on the regime's side.
        fill = s.volume_radius_fill()
        assert abs(fill - 1) < 1e-9
        if abs(fill - 1) > 1e-14:
            assert (fill > 1) == (factor > 0)


@pytest.fixture(scope="module", params=(rocheometry.Fitted(), rocheometry.Reference()))
def continuity_domain(
    request: pytest.FixtureRequest,
) -> tuple[dict[float, rocheometry.ComponentGeometry], float]:
    """Measure the zero-depth noise independently of the rate-test depths."""
    geometries = {
        qi: rocheometry.ComponentGeometry(qi, accuracy=request.param)
        for qi in (1e-8, 1e-3, 1.0, 1e3, 1e8)
    }
    floor = max(
        abs(g._surface_at_depth(sign * 1e-30).volume_radius_fill() - 1)
        for g in geometries.values()
        for sign in (-1, 1)
    ) + math.ulp(1.0)
    return geometries, floor


@pytest.mark.slow
@pytest.mark.parametrize("qi", (1e-8, 1e-3, 1.0, 1e3, 1e8))
def test_measured_fill_is_first_order_in_depth_until_the_noise_floor(
    continuity_domain: tuple[dict[float, rocheometry.ComponentGeometry], float],
    qi: float,
) -> None:
    """Check sign, the independent critical derivative, and absence of a jump.

    The zero-depth noise comes from separate +/-1e-30 probes over the domain.
    The coarsest pair bounds genuine finite-depth nonlinearity. At finer
    depths that contribution must shrink in proportion to depth; the measured
    noise is the only depth-independent allowance. Unresolvable signs do not
    classify a surface: its depth and regime do.
    """
    from rocheometry._levels import _levels

    geometries, floor = continuity_domain
    g = geometries[qi]
    gap = _levels(g.field, g._computers).admissibility_gap
    sample = g._computers.sample(g.field, 0.0)
    slope = math.fsum(
        weight
        * radius**2
        / (
            -sum(
                a * b
                for a, b in zip(
                    g.field.gradient(*(radius * v for v in direction)), direction
                )
            )
        )
        for (direction, weight), radius in zip(sample.rays, sample.radii, strict=True)
    ) / (3 * g.lobe().volume())
    coarse = max(
        abs(
            (g._surface_at_depth(sign * 1e-3 * gap).volume_radius_fill() - 1)
            / (sign * 1e-3 * gap)
            - slope
        )
        for sign in (-1, 1)
    )
    for factor in (1e-3, 1e-6, 1e-9, 1e-12):
        for sign in (-1, 1):
            depth = sign * factor * gap
            s = g._surface_at_depth(depth)
            difference = s.volume_radius_fill() - 1
            assert s.regime is (
                rocheometry.LevelRegime.CONTACT
                if sign > 0
                else rocheometry.LevelRegime.DETACHED
            )
            if abs(slope * depth) > floor:
                assert difference * sign > 0
            assert abs(difference - slope * depth) <= coarse * abs(depth) + floor
