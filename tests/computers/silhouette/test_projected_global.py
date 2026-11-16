"""Stateless LOS isolation and exact reference branch junctions."""

import copy
import random
from math import cos, hypot, pi, sin, sqrt
from typing import Any

import pytest

import rocheometry
from rocheometry._computers.silhouette.chord import _Chord
from rocheometry._computers.silhouette.projected import ProjectedField
from rocheometry._geometry import Point3D
from tests._helpers import projected_field, select


def field_for(
    case: dict[str, Any],
) -> tuple[rocheometry.ComponentSurfaceGeometry, ProjectedField]:
    """Return the Reference projected field at a fitted surface's level."""
    selected = select(
        rocheometry.ComponentGeometry(case["qi"]), case["kind"], case["level"]
    )
    surface = rocheometry.ComponentGeometry(
        case["qi"], accuracy=rocheometry.Reference()
    ).surface(potential=selected.potential)
    return surface, projected_field(
        surface, rocheometry.CentrelinePlane(case["inclination"])
    )


def test_interval_bounds_include_each_monotone_piece() -> None:
    rng = random.Random(2026102403)
    chord = _Chord(
        0.99747**2 + 0.0024**2,
        (1 - 0.99747) ** 2 + 0.0024**2,
        1 / (1 + 1e-8),
        1e-8 / (1 + 1e-8),
        1.0,
        sin(4 * pi / 9) ** 2,
        0.0024 * sin(4 * pi / 9) * cos(4 * pi / 9),
        0.07,
    )
    for _ in range(200):
        a, b = sorted((rng.uniform(-0.07, 0.07), rng.uniform(-0.07, 0.07)))
        for order, function in ((0, chord.derivative), (1, chord.curvature)):
            lo, hi = chord.bounds(a, b, order)
            assert all(lo <= function(a + (b - a) * j / 100) <= hi for j in range(101))


def test_one_mass_isolation_includes_maximum_minimum_and_constrained_maximum() -> None:
    # D'' changes sign just once on w>0. This one-mass chord has the
    # max/min pair plus a constrained maximum at the right endpoint.
    chord = _Chord(0.01, 1.0, 1.0, 0.0, 1.0, 0.9, 0.01, 4.0)
    roots = [w for w in chord.stationary() if w > 0]
    assert len(roots) == 2
    assert chord.curvature(roots[0]) < 0 < chord.curvature(roots[1])
    assert chord.derivative(chord.limit) > 0
    turn = sqrt(3 * chord.a / 2)
    assert 0 < turn < chord.limit


def test_query_order_and_deliberately_bad_warm_starts_do_not_change_bits() -> None:
    _, field = field_for(dict(qi=1e8, kind="fill", level=1.0, inclination=4 * pi / 9))
    queries = [
        (r * cos(a), r * sin(a))
        for a in (0.00238, 0.00240, 0.0024165, 0.00244, 0.00246, 0.02, 0.8)
        for r in (0.99, 0.99747, 0.999)
    ]
    expected = {q: field.maximum(*q) for q in queries}
    random.Random(2026102404).shuffle(queries)
    for seed in (-1e100, 0.0, 0.003, 0.01, 1e100):
        other = copy.copy(field)
        other._points = {}
        other._maxima = {}
        other._stationary = {}
        setattr(other, "_previous_w", seed)
        assert {q: other.maximum(*q) for q in queries} == expected


def test_factored_level_upper_bounds_the_whole_interval() -> None:
    _, field = field_for(dict(qi=1e8, kind="fill", level=1.0, inclination=4 * pi / 9))
    x, t = 0.99747, 0.0024
    chord = field._chord(x, t)
    rng = random.Random(2026102405)
    for _ in range(100):
        a, b = sorted(
            (
                rng.uniform(-chord.limit, chord.limit),
                rng.uniform(-chord.limit, chord.limit),
            )
        )
        upper = chord.level_upper(
            x, t, field.c, field.s, field.l1.position[0], 0.0, a, b
        )
        assert all(
            field._residual(field._point(x, t, a + (b - a) * j / 50)) <= upper
            for j in range(51)
        )


@pytest.mark.parametrize("kind,level", (("fill", 1.0), ("F", 0.99)))
def test_limb_radii_are_bit_identical_when_queries_are_shuffled(
    kind: str, level: float
) -> None:
    _, field = field_for(dict(qi=1e8, kind=kind, level=level, inclination=4 * pi / 9))
    queries = [
        field.start + 0.001,
        0.00238,
        0.00240,
        0.0024165,
        0.00244,
        0.00246,
        0.02,
        0.8,
        pi,
    ]
    expected = {a: field.radius(a) for a in queries}
    other = copy.copy(field)
    other._points = {}
    other._maxima = {}
    other._stationary = {}
    setattr(other, "_previous_w", 1e100)
    random.Random(2026102406).shuffle(queries)
    assert {a: other.radius(a) for a in queries} == expected


def test_limb_membership_beyond_sheet_queries_uses_an_invariant_collar() -> None:
    import rocheometry
    from tests.computers.silhouette.test_projected_outlines import projected

    surface = select(
        rocheometry.ComponentGeometry(100.0, accuracy=rocheometry.Reference()),
        "F",
        0.99,
    )
    field = projected(surface, pi / 6)
    x, t = 0.8586568485757005, 0.05801226846760989
    point = field.maximum(x, t)
    # This is a valid limb point beyond the neck cylinder. The proof of its
    # membership must not extend or weaken the direct watershed bounds.
    from math import atan2

    with pytest.raises(
        ValueError, match="transverse radius lies beyond the traced watershed"
    ):
        field.watershed.cut_height(
            field.field, hypot(point[1], point[2]), atan2(point[2], point[1])
        )
    radius = field.radius(atan2(t, x))
    assert abs(
        field.value(radius * x / hypot(x, t), radius * t / hypot(x, t))
    ) <= field._value_roundoff(x, t)


def test_adjacent_unresolved_maximum_minimum_pair_is_reported() -> None:
    from rocheometry._errors import DegenerateLineOfSightError

    # Exactly D(.5)=D'(.5)=0, using dyadic coefficients. This pair must not
    # disappear through same-sign pruning or be guessed from a warm start.
    chord = _Chord(0.75, 2.0, 1.0, 0.0, 1.0, 0.25, 0.375, 1.0)
    assert chord.derivative(0.5) == chord.curvature(0.5) == 0.0
    with pytest.raises(
        DegenerateLineOfSightError, match="degenerate line-of-sight stationary pair"
    ):
        chord.stationary()


def test_derivative_bound_keeps_numerator_cancellation_in_its_allowance() -> None:
    import mpmath as mp

    chord = _Chord(0.75, 2.0, 1.0, 0.0, 1.0, 1e-30, 1e-16, 1.0)
    w = sqrt(0.375)
    lower, upper = chord.bounds(w, w, 1)
    with mp.workdps(60):
        v = mp.mpf(w)
        exact = mp.mpf(chord.s2) + (2 * v * v - mp.mpf(chord.a)) / (
            mp.mpf(chord.a) + v * v
        ) ** mp.mpf("2.5")
        assert mp.mpf(lower) <= exact <= mp.mpf(upper)


def test_signed_zero_and_numpy_keys_cannot_poison_the_cache() -> None:
    import struct

    import numpy as np

    _, field = field_for(dict(qi=1.0, kind="fill", level=1.0, inclination=pi / 3))

    def bits(point: Point3D) -> bytes:
        return b"".join(struct.pack(">d", v) for v in point)

    expected = bits(field.maximum(0.0, 0.2))
    other = copy.copy(field)
    other._maxima = {}
    other._stationary = {}
    other.maximum(-0.0, np.float64(0.2))
    assert bits(other.maximum(0.0, 0.2)) == expected
    assert bits(other.maximum(np.float64(0.0), 0.2)) == expected


def test_empty_bounding_chord_is_outside_not_a_centrifugal_maximum() -> None:
    _, field = field_for(dict(qi=1.0, kind="fill", level=1.0, inclination=pi / 3))
    assert field.value(2 * field.outer, 0.0) == -float("inf")
    with pytest.raises(
        ValueError, match="line of sight does not meet the component bound"
    ):
        field.maximum(2 * field.outer, 0.0)


def test_strict_concavity_retains_the_original_root_bracket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chord = _Chord(1.0, 2.0, 0.5, 0.2, 1.0, 0.1, 0.1, 0.5)
    left, right = -chord.limit, chord.limit
    assert chord.bounds(left, right, 1)[1] < 0
    assert chord.derivative(left) > 0 > chord.derivative(right)
    expected = chord.root(left, right)
    original = _Chord.root
    brackets = []

    def refine(self: _Chord, a: float, b: float) -> float:
        brackets.append((a, b))
        return original(self, a, b)

    def unnecessary_lower_bound(w: float) -> float:
        raise AssertionError("a strictly concave chord needs no pruning lower bound")

    monkeypatch.setattr(_Chord, "root", refine)
    assert chord.stationary(attained_lower=unnecessary_lower_bound) == (expected,)
    assert brackets == [(left, right)]


def test_positive_level_bound_never_evaluates_a_pruning_lower_bound() -> None:
    chord = _Chord(0.01, 1.0, 1.0, 0.0, 1.0, 0.9, 0.01, 4.0)
    expected = chord.stationary()

    def unnecessary_lower_bound(w: float) -> float:
        raise AssertionError("the original pruning rule cannot prune U-u >= 0")

    assert (
        chord.stationary(
            attained_lower=unnecessary_lower_bound, level_upper=lambda a, b: 1.0
        )
        == expected
    )
