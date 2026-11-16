"""Branch crossings and structurally visible short secondary limbs."""

import random
from math import atan2, cos, sin
from types import SimpleNamespace
from typing import cast

import pytest

from rocheometry._computers.silhouette import limb
from rocheometry._computers.silhouette.chord import _curvature_term_bounds
from rocheometry._computers.silhouette.projected import ProjectedField
from rocheometry._geometry import Point2D
from rocheometry._numerics.roots import DEFAULT_ROOT_ABSOLUTE_TOLERANCE, brent_root


@pytest.mark.parametrize("amplitude", (1.0, 1e-100, 1e-300))
def test_hidden_branch_peak_finds_both_crossings_without_a_height_cutoff(
    monkeypatch: pytest.MonkeyPatch, amplitude: float
) -> None:
    def margin(angle: float) -> tuple[float, float]:
        return amplitude * (1 / 16 - (angle - 0.5) ** 2), amplitude * (1 - 2 * angle)

    monkeypatch.setattr(
        limb,
        "_score",
        lambda field, angle, points: (-margin(angle)[0], -margin(angle)[1]),
    )
    monkeypatch.setattr(limb, "_margin", lambda field, angle, primary: margin(angle))

    def crossing(field: ProjectedField, left: float, right: float) -> limb._Junction:
        angle = brent_root(lambda a: margin(a)[0], left, right)
        return limb._Junction(
            angle, (cos(angle), sin(angle)), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
        )

    monkeypatch.setattr(limb, "_junction", crossing)
    points = ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    found = limb._hidden_junctions(cast(ProjectedField, None), 0.0, 1.0, points, points)
    assert [j.angle for j in found] == pytest.approx(
        [0.25, 0.75], rel=0, abs=DEFAULT_ROOT_ABSOLUTE_TOLERANCE
    )


def test_touch_without_a_sign_change_does_not_create_a_corner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        limb, "_score", lambda field, a, points: ((a - 0.5) ** 2, 2 * a - 1)
    )
    monkeypatch.setattr(
        limb, "_margin", lambda field, a, primary: (-((a - 0.5) ** 2), 1 - 2 * a)
    )
    points = ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert (
        limb._hidden_junctions(cast(ProjectedField, None), 0.0, 1.0, points, points)
        == []
    )


def test_single_maximum_is_smooth_without_testing_tangent_jump() -> None:
    field = SimpleNamespace(s=0.0, c=1.0, _residual=lambda p: 0.0)
    point = (0.0, 0.0, 0.0)
    assert (
        limb._hidden_junctions(
            cast(ProjectedField, field), 0.1, 0.2, (point,), (point,)
        )
        == []
    )


def test_curvature_bounds_enclose_rectangles_cut_by_the_bounding_sphere() -> None:
    rng = random.Random(2026102701)
    for _ in range(100):
        lo = 10 ** rng.uniform(-6, 0)
        hi = lo * rng.uniform(1, 10)
        zl = lo * rng.uniform(0, 2)
        zh = hi * rng.uniform(2, 5)
        cap = rng.uniform(lo + zl, hi + zh)
        lower, upper = _curvature_term_bounds(lo, hi, zl, zh, cap)
        for i in range(21):
            for j in range(21):
                a, z = lo + (hi - lo) * i / 20, zl + (zh - zl) * j / 20
                if a + z <= cap:
                    value = (2 * z - a) / (a + z) ** 2.5
                    assert lower <= value <= upper


def test_short_corner_arm_preserves_the_analytic_tangent_in_absolute_coordinates() -> (
    None
):
    point = (0.9946351551549991, 0.005446177784304905)
    tangent = (-0.26839825159289926, 0.9931953470652779)
    control = (point[0] - 4.9014071690223204e-9, point[1] + 1.8137431194963083e-8)

    def angle(p: Point2D) -> float:
        dx, dy = p[0] - point[0], p[1] - point[1]
        return abs(
            atan2(dx * tangent[1] - dy * tangent[0], dx * tangent[0] + dy * tangent[1])
        )

    aligned = limb._aligned_arm(point, control, tangent)
    assert angle(control) > 1e-9
    assert angle(aligned) < 1e-9
    assert aligned[0] == control[0]
    assert limb._aligned_arm(point, point, tangent) == point
