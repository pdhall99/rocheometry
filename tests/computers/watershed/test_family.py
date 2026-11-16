"""Fixed-family queries agree with the original traced secant implementation."""

import json
import pickle
from dataclasses import fields
from math import cos, hypot, pi
from pathlib import Path
from typing import NotRequired, TypedDict, cast
from unittest.mock import patch

import pytest

import rocheometry
from rocheometry import Fitted, Reference
from rocheometry._computers import defaults
from rocheometry._computers.composition import ComputerSet
from rocheometry._computers.contact import admissibility_depth
from rocheometry._computers.watershed import traced
from rocheometry._computers.watershed.family import _fit
from rocheometry._computers.watershed.traced import (
    WATERSHED_AZIMUTH_LINE_COUNT,
    WATERSHED_COSINE_SERIES_TERMS,
    TracedWatershedComputer,
    WatershedLine,
    _ExactWatershedComputer,
)
from rocheometry._geometry import Point3D
from rocheometry._roche.field import RocheField
from tests._helpers import _Wired
from tests.computers.test_contact_parts import FACTORS, QI

ANGLES = (0.0, 1e-6, pi / 8, pi / 4, 3 * pi / 8, pi / 2 - 1e-6, pi / 2)
DEPTH_FACTORS = (1e-10, 1e-8, 1e-6, 1e-4, 0.01, 0.5, 0.99, 1.05)


class AlphaSample(TypedDict):
    reference: bool
    qi: float
    factor: float
    potential: float
    volume: float
    area: float
    integrated_mean_curvature: float
    mean_gravity: float
    matched: NotRequired[dict[str, float]]
    converged: NotRequired[dict[str, float]]


ALPHA_SAMPLES = cast(
    list[AlphaSample],
    json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("alpha-family-integrals.json")
        .read_text()
    )["samples"],
)


def gap(qi: float) -> float:
    g = Reference()._computers()
    f = RocheField(qi)
    return admissibility_depth(f, g.l1(f), g.outer)


@pytest.fixture(scope="module", params=(1e-13, 1e-11))
def pair(
    request: pytest.FixtureRequest,
) -> tuple[TracedWatershedComputer, _ExactWatershedComputer, float]:
    g = Reference()._computers()
    tolerance = cast(float, request.param)
    return (
        TracedWatershedComputer(
            l1_computer=g.l1, outer_computer=g.outer, tolerance=tolerance
        ),
        _ExactWatershedComputer(
            l1_computer=g.l1, outer_computer=g.outer, tolerance=1e-14
        ),
        1e-12 if tolerance == 1e-13 else 1e-10,
    )


@pytest.mark.slow
@pytest.mark.parametrize("qi", QI)
def test_family_rims_match_exact_grid(
    pair: tuple[TracedWatershedComputer, _ExactWatershedComputer, float], qi: float
) -> None:
    w, exact, tolerance = pair
    field = RocheField(qi)
    # The selected default must retain tenfold margin below the query bounds.
    bound = tolerance * w.l1_computer(field).position[0] / 10
    for factor in DEPTH_FACTORS:
        for psi in ANGLES:
            depth = factor * gap(qi)
            actual, expected = (
                w.rim_at_depth(field, depth, psi),
                exact.rim_at_depth(field, depth, psi),
            )
            assert (
                hypot(*(a - b for a, b in zip(actual, expected, strict=True))) <= bound
            ), (qi, factor, psi, actual, expected)
            assert (
                abs(w.cut_height(field, hypot(*expected[1:]), psi) - expected[0])
                <= bound
            )
            ray = w.sheet_point_on_ray(field, expected)
            assert hypot(*(a - b for a, b in zip(ray, expected, strict=True))) <= bound


@pytest.mark.slow
@pytest.mark.parametrize("qi", QI)
def test_family_cut_and_ray_match_exact(
    pair: tuple[TracedWatershedComputer, _ExactWatershedComputer, float], qi: float
) -> None:
    w, exact, tolerance = pair
    field = RocheField(qi)
    bound = tolerance * w.l1_computer(field).position[0]
    for factor in (0.01, 0.5, 0.99):
        for psi in (0.0, pi / 6, pi / 3, pi / 2):
            p = exact.rim_at_depth(field, factor * gap(qi), psi)
            rho = hypot(*p[1:])
            assert (
                abs(w.cut_height(field, rho, psi) - exact.cut_height(field, rho, psi))
                <= bound
            ), (qi, factor, psi)
            for sy, sz in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                direction = (p[0], sy * p[1], sz * p[2])
                actual, expected = (
                    w.sheet_point_on_ray(field, direction),
                    exact.sheet_point_on_ray(field, direction),
                )
                assert (
                    hypot(*(a - b for a, b in zip(actual, expected, strict=True)))
                    <= bound
                ), (qi, factor, psi, direction)


@pytest.mark.parametrize("terms", (13, 17, 21))
def test_qr_recovers_cosine_harmonics_on_nonuniform_nodes(terms: int) -> None:
    angles = tuple(pi * k / 48 for k in range(25)) + tuple(
        pi / 2 * (k / 24) ** 3 for k in range(1, 24)
    )
    x = tuple(0.1 + 0.03 * cos(6 * a) for a in angles)
    r = tuple(0.2 - 0.04 * cos(10 * a) for a in angles)
    xc, rc = _fit(angles, (x, r), terms)
    for j, c in enumerate(xc):
        assert c == pytest.approx({0: 0.1, 3: 0.03}.get(j, 0), abs=1e-15)
    for j, c in enumerate(rc):
        assert c == pytest.approx({0: 0.2, 5: -0.04}.get(j, 0), abs=1e-15)


def test_qr_rejects_rank_deficient_nodes() -> None:
    with pytest.raises(ValueError, match="rank deficient"):
        _fit((0.2,) * 48, ((1.0,) * 48,), 17)


def test_qr_rejects_relatively_rank_deficient_nodes() -> None:
    angles = tuple(pi / 4 + (k - 24) * 1e-7 for k in range(48))
    with pytest.raises(ValueError, match="rank deficient"):
        _fit(angles, ((1.0,) * 48,), 3)


@pytest.mark.parametrize(
    "computer_type,node_count,depth_factor",
    ((_ExactWatershedComputer, 4, 1.05), (TracedWatershedComputer, 7, 2.0)),
)
def test_builds_each_node_once_without_tracing(
    computer_type: type[_ExactWatershedComputer], node_count: int, depth_factor: float
) -> None:
    g = Reference()._computers()
    with (
        patch.object(traced, "Memoized", wraps=traced.Memoized) as memoized,
        patch.object(traced, "_TraceLine", wraps=traced._TraceLine) as trace,
    ):
        w = computer_type(l1_computer=g.l1, outer_computer=g.outer)
    assert memoized.call_count == node_count
    trace.assert_called_once_with(g.l1, g.outer, w.tolerance, depth_factor)
    assert w.line.cache_info().currsize == 0


def test_fit_controls_and_family_query_fields_are_private_and_minimal() -> None:
    names = {item.name for item in fields(TracedWatershedComputer)}
    assert "line_count" not in names and "series_terms" not in names
    assert {item.name for item in fields(traced._FamilySheetPoint)} == {
        "l1_computer",
        "cut_section",
    }
    assert {item.name for item in fields(traced._FamilyCutHeight)} == {
        "l1_computer",
        "outer_computer",
        "section",
        "cut_section",
        "tolerance",
    }


def test_family_is_fixed_shared_across_levels_and_pickle_clears_results() -> None:
    g = Reference()._computers()
    w = TracedWatershedComputer(
        l1_computer=g.l1, outer_computer=g.outer, _line_count=65
    )
    f = RocheField(0.05)
    p = w.rim_at_depth(f, 0.5 * gap(f.qi), 0.4)
    family = w.family(f)
    assert tuple(line.alpha for line in family.lines[:65]) == tuple(
        pi * k / (2 * 64) for k in range(65)
    )
    assert len(family.lines) == 128
    assert all(
        a.alpha < b.alpha
        for a, b in zip(family.lines[65:-1], family.lines[66:], strict=True)
    )
    misses = w.line.cache_info().misses
    for factor in (0.1, 0.6, 0.9):
        w.rim_at_depth(f, factor * gap(f.qi), 0.7)
        w.rim_at_depth(RocheField(20.0), factor * gap(f.qi), 0.7)
    w.cut_height(f, hypot(*p[1:]), 0.4)
    w.sheet_point_on_ray(f, p)
    assert w.family(f) is family
    assert w.family.cache_info().misses == 1
    assert w.line.cache_info().misses == misses == 128
    restored = pickle.loads(pickle.dumps(w))
    assert restored._line_count == 65
    assert restored._series_terms == WATERSHED_COSINE_SERIES_TERMS
    assert restored.family.cache_info().currsize == 0
    assert restored.section_computer.cache_info().currsize == 0
    assert restored.rim_at_depth(f, 0.5 * gap(f.qi), 0.4) == p


def test_near_axis_ray_uses_coordinate_residuals(
    pair: tuple[TracedWatershedComputer, _ExactWatershedComputer, float],
) -> None:
    w, exact, tolerance = pair
    field = RocheField(1e-6)
    direction = (0.9999999988941625, 4.70283418641036e-5, 1.0106208131738642e-7)
    actual = w.sheet_point_on_ray(field, direction)
    expected = exact.sheet_point_on_ray(field, direction)
    x1 = w.l1_computer(field).position[0]
    assert (
        hypot(*(a - b for a, b in zip(actual, expected, strict=True))) <= tolerance * x1
    )


@pytest.mark.parametrize("count", (0, 1, 2, -1))
def test_family_rejects_small_counts(count: int) -> None:
    g = Reference()._computers()
    with pytest.raises(ValueError, match="line_count"):
        TracedWatershedComputer(
            l1_computer=g.l1, outer_computer=g.outer, _line_count=count
        )


@pytest.mark.parametrize("count", (True, 33.0))
def test_family_requires_integer_count(count: object) -> None:
    g = Reference()._computers()
    with pytest.raises(TypeError, match="line_count"):
        TracedWatershedComputer(
            l1_computer=g.l1, outer_computer=g.outer, _line_count=cast(int, count)
        )


@pytest.mark.parametrize("qi", QI)
def test_family_has_exact_quadrant_symmetry(qi: float) -> None:
    w = Reference()._computers().watershed
    f = RocheField(qi)
    d = 0.99 * gap(qi)
    assert w.rim_at_depth(f, d, 0.0)[2] == 0.0
    assert w.rim_at_depth(f, d, pi / 2)[1] == 0.0
    p = w.rim_at_depth(f, d, pi / 8)
    for psi, sy, sz in ((-pi / 8, 1, -1), (pi - pi / 8, -1, 1), (pi + pi / 8, -1, -1)):
        a = w.rim_at_depth(f, d, psi)
        assert a == pytest.approx((p[0], sy * p[1], sz * p[2]), abs=1e-15)


@pytest.fixture(scope="module", params=(False, True))
def integral_graphs(
    request: pytest.FixtureRequest,
) -> tuple[ComputerSet, ComputerSet]:
    reference = cast(bool, request.param)
    settings = defaults._Settings(
        polar_order=128 if reference else 64,
        azimuth_order=64 if reference else 24,
        watershed_tolerance=1e-13 if reference else 1e-11,
        fitted_selection=not reference,
        exact_anchor_everywhere=reference,
        fill_tolerance=1e-13 if reference else 2e-8,
        fixed_paths=not reference,
        path_tolerance=1e-10 if reference else 1e-5,
    )
    new = defaults._wire_computers(settings)
    with patch.object(defaults, "TracedWatershedComputer", _ExactWatershedComputer):
        old = defaults._wire_computers(settings)
    return old, new


@pytest.mark.slow
@pytest.mark.parametrize("qi", QI)
@pytest.mark.parametrize("factor", FACTORS)
def test_contact_integrals_stable_at_common_potential(
    integral_graphs: tuple[ComputerSet, ComputerSet], qi: float, factor: float
) -> None:
    old, new = integral_graphs
    before = rocheometry.ComponentGeometry(qi, accuracy=_Wired(old)).surface(
        fill_out_factor=factor
    )
    after = rocheometry.ComponentGeometry(qi, accuracy=_Wired(new))._surface_at_depth(
        before._depth
    )
    for reader in ("volume", "area", "integrated_mean_curvature", "mean_gravity"):
        assert getattr(after, reader)() == pytest.approx(
            getattr(before, reader)(), rel=1e-11, abs=0
        ), reader


@pytest.mark.slow
@pytest.mark.parametrize(
    "sample",
    ALPHA_SAMPLES,
    ids=lambda s: (
        f"{'reference' if s['reference'] else 'default'}-{s['qi']}-{s['factor']}"
    ),
)
def test_contact_integrals_preserve_alpha_family_and_improve_original_default(
    sample: AlphaSample,
) -> None:
    graph = Reference()._computers() if sample["reference"] else Fitted()._computers()
    surface = rocheometry.ComponentGeometry(
        sample["qi"], accuracy=_Wired(graph)
    ).surface(potential=rocheometry.Potential.jacobi(sample["potential"]))
    matched = sample.get("matched", sample)
    for reader in ("volume", "area", "integrated_mean_curvature", "mean_gravity"):
        actual = cast(float, getattr(surface, reader)())
        expected = cast(float, matched[reader])
        # Isolate representation: both families use the same integration tolerance.
        assert actual == pytest.approx(expected, rel=1e-11, abs=0), reader
        if not sample["reference"]:
            converged = sample["converged"][reader]
            original = cast(float, sample[reader])
            # Isolate integration accuracy too: identical potential and quadrature.
            assert abs(actual - converged) <= (
                abs(original - converged) + 1e-12 * abs(converged)
            ), reader


@pytest.mark.slow
@pytest.mark.parametrize(
    "ratios", ((1 - 1e-6, 1.0, 1 + 1e-6), tuple(10 ** (k / 8) for k in range(-4, 5)))
)
def test_family_varies_smoothly_with_mass_ratio(ratios: tuple[float, ...]) -> None:
    g = Reference()._computers()
    w = TracedWatershedComputer(l1_computer=g.l1, outer_computer=g.outer)
    exact = _ExactWatershedComputer(l1_computer=g.l1, outer_computer=g.outer)
    points: list[Point3D] = []
    baseline: list[Point3D] = []
    for qi in ratios:
        f = RocheField(qi)
        points.append(w.rim_at_depth(f, 0.5 * gap(qi), pi / 4))
        baseline.append(exact.rim_at_depth(f, 0.5 * gap(qi), pi / 4))
    # Compare second differences with exact tracing: any curvature from the
    # physical level convention is retained, without extra interpolation jumps.
    for i in range(1, len(ratios) - 1):
        for j in range(3):
            second = points[i + 1][j] - 2 * points[i][j] + points[i - 1][j]
            expected = baseline[i + 1][j] - 2 * baseline[i][j] + baseline[i - 1][j]
            assert abs(second - expected) <= 1e-12
    assert WATERSHED_AZIMUTH_LINE_COUNT == 25


@pytest.mark.parametrize("qi", QI)
def test_second_line_set_improves_terminal_azimuth_spread(qi: float) -> None:
    g = Reference()._computers()
    f = RocheField(min(qi, 1 / qi))
    w = cast(TracedWatershedComputer, g.watershed)
    lines = w.family(f).lines
    n = w._line_count
    assert len(lines) == 2 * n - 2
    first = lines[:n]
    second = (first[0], *lines[n:], first[-1])
    from math import atan2, log

    targets = tuple(pi * k / (2 * (n - 1)) for k in range(n))

    def error(subset: tuple[WatershedLine, ...]) -> float:
        points = tuple(line.at_log_depth(log(gap(qi))) for line in subset)
        return sum(
            (atan2(p[2], p[1]) - target) ** 2
            for p, target in zip(points, targets, strict=True)
        )

    assert error(second) < error(first)


@pytest.mark.parametrize(
    "terms,error",
    (
        (0, ValueError),
        (-1, ValueError),
        (26, ValueError),
        (48, ValueError),
        (49, ValueError),
        (True, TypeError),
        (17.0, TypeError),
    ),
)
def test_series_term_validation(terms: object, error: type[Exception]) -> None:
    g = Reference()._computers()
    with pytest.raises(error, match="series_terms"):
        TracedWatershedComputer(
            l1_computer=g.l1, outer_computer=g.outer, _series_terms=cast(int, terms)
        )


def test_family_capacity_is_independent_and_survives_pickle() -> None:
    graph = Reference()._computers()
    w = TracedWatershedComputer(
        l1_computer=graph.l1,
        outer_computer=graph.outer,
        _line_count=17,
        family_maxsize=2,
    )
    first = w.family(RocheField(0.05))
    w.family(RocheField(0.1))
    w.family(RocheField(0.2))
    assert w.family.cache_info().currsize == 2
    assert w.family(RocheField(0.05)) is not first
    assert w.rim_computer.cache_info().maxsize == 128
    assert w.depth_computer.cache_info().maxsize == 128
    restored = pickle.loads(pickle.dumps(w))
    assert restored.family_maxsize == 2
    assert restored.family.cache_info().maxsize == 2
    assert restored.family.cache_info().currsize == 0


@pytest.mark.parametrize(
    "capacity,error",
    ((-1, ValueError), (True, TypeError), (4.0, TypeError), (None, TypeError)),
)
def test_family_capacity_validation(capacity: object, error: type[Exception]) -> None:
    graph = Reference()._computers()
    with pytest.raises(error, match="family_maxsize"):
        TracedWatershedComputer(
            l1_computer=graph.l1,
            outer_computer=graph.outer,
            family_maxsize=cast(int, capacity),
        )


def test_dense_arrays_remain_unchanged_across_queries_and_eviction() -> None:
    from array import array
    from math import log

    graph = Reference()._computers()
    w = TracedWatershedComputer(
        l1_computer=graph.l1,
        outer_computer=graph.outer,
        _line_count=17,
        family_maxsize=1,
    )
    field = RocheField(0.05)
    line = w(field, pi / 8)
    assert isinstance(line.log_depths, array) and line.log_depths.typecode == "d"
    assert isinstance(line.steps.data, array) and line.steps.data.typecode == "d"
    before = line.log_depths.tobytes(), line.steps.data.tobytes()
    at = log(0.5 * gap(field.qi))
    expected = line.at_log_depth(at)
    w.family(field)
    w.family(RocheField(0.1))
    w.line.cache_clear()
    for offset in (0.0, -0.1, -1.0):
        line.at_log_depth(at + offset)
    assert (line.log_depths.tobytes(), line.steps.data.tobytes()) == before
    assert line.at_log_depth(at) == expected
