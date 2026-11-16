"""Independent basin, symmetry, ordering and ray checks for the watershed."""

from math import atan2, hypot, log, pi
from typing import cast

import pytest

import rocheometry
from rocheometry import Fitted, Reference
from rocheometry._computers.watershed import TracedWatershedComputer
from rocheometry._geometry import Point3D
from rocheometry._roche.field import RocheField
from tests._helpers import _depth

QI = (1e-6, 1e-3, 0.05, 1.0, 20.0, 1e3, 1e6)


def level(qi: float, factor: float = 0.5) -> rocheometry.Potential:
    g = Reference()._computers()
    f = RocheField(min(qi, 1 / qi))
    u1 = g.l1(f).potential.value
    u2 = max(g.outer(f).potential.value, g.outer(RocheField(1 / f.qi)).potential.value)
    return rocheometry.Potential.jacobi(u1 - factor * (u1 - u2))


def test_watershed_set_wiring_and_cache() -> None:
    for graph, tol in (
        (Reference()._computers(), 1e-13),
        (Fitted()._computers(), 1e-11),
    ):
        assert isinstance(graph.watershed, TracedWatershedComputer)
        assert graph.watershed.tolerance == tol
        a = graph.watershed(RocheField(0.05), 0.3)
        assert graph.watershed(RocheField(0.05), 0.3) is a
        assert len(a.log_depths) == len(a.steps) + 1


@pytest.mark.parametrize("qi", QI)
def test_rim_mirror_l1_limit_and_ray_consistency(qi: float) -> None:
    graph = Reference()._computers()
    w = graph.watershed
    field = RocheField(qi)
    l1 = graph.l1(field)
    for psi in (0.0, pi / 6, pi / 3, pi / 2):
        u = level(qi)
        p = w.rim(field, _depth(field, u), psi)
        mirror = w.rim(RocheField(1 / qi), _depth(RocheField(1 / qi), u), psi)
        assert mirror == pytest.approx((1 - p[0], p[1], p[2]), abs=1e-10, rel=0)
        assert field.potential(*p) == pytest.approx(u.value, abs=1e-14, rel=0)
        assert w.sheet_point_on_ray(field, p) == pytest.approx(p, abs=1e-10, rel=0)
        near = w.rim(
            field,
            _depth(field, rocheometry.Potential.jacobi(l1.potential.value - 1e-10)),
            psi,
        )
        assert abs(near[0] - l1.position[0]) <= 1e-4 * l1.position[0]


@pytest.mark.parametrize("qi", QI)
def test_lines_are_radially_increasing_and_angularly_ordered(qi: float) -> None:
    graph = Reference()._computers()
    w = cast(TracedWatershedComputer, graph.watershed)
    field = RocheField(qi)
    angles = [k * pi / 16 for k in range(9)]
    lines = [w(field, a) for a in angles]
    final = log(lines[0].u_l1 - level(qi, 1).value)
    depths = [
        lines[0].log_depths[0] + (final - lines[0].log_depths[0]) * k / 32
        for k in range(33)
    ]
    previous = [0.0] * len(lines)
    previous_angles = [(0.0, 0.0)] * len(lines)
    for s in depths:
        points = [line.at_log_depth(s) for line in lines]
        radii = [hypot(p[1], p[2]) for p in points]
        assert all(r > p for r, p in zip(radii, previous, strict=True))
        azimuths = [atan2(p[2], p[1]) for p in points]
        assert all(a < b for a, b in zip(azimuths, azimuths[1:]))
        polar_angles = [
            (atan2(r, p[0]), atan2(r, 1 - p[0]))
            for r, p in zip(radii, points, strict=True)
        ]
        assert all(
            a[0] > b[0] and a[1] > b[1]
            for a, b in zip(polar_angles, previous_angles, strict=True)
        )
        previous_angles = polar_angles
        previous = radii


def uphill_basin(field: RocheField, p: Point3D) -> int:
    """Independently classify a point by climbing the normalized gradient."""
    graph = Reference()._computers()

    def derivative(
        t: float, position: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        g = field.gradient(*position)
        # Arc length prevents the point-mass singularity from controlling dt.
        norm = hypot(*g)
        return g[0] / norm, g[1] / norm, g[2] / norm

    radius = min(graph.l1(field).position[0], 1 - graph.l1(field).position[0]) * 1e-3
    q = p
    for _ in range(10000):
        r0, r1 = hypot(*q), hypot(q[0] - 1, q[1], q[2])
        if min(r0, r1) < radius:
            return 0 if r0 < r1 else 1
        step = 0.02 * min(r0, r1)
        k1 = derivative(0, q)
        k2 = derivative(
            0, cast(Point3D, tuple(q[j] + step * k1[j] / 2 for j in range(3)))
        )
        k3 = derivative(
            0, cast(Point3D, tuple(q[j] + step * k2[j] / 2 for j in range(3)))
        )
        k4 = derivative(0, cast(Point3D, tuple(q[j] + step * k3[j] for j in range(3))))
        q = cast(
            Point3D,
            tuple(
                q[j] + step * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6
                for j in range(3)
            ),
        )
    raise AssertionError("uphill classifier did not reach either star")


@pytest.mark.parametrize("qi", QI)
@pytest.mark.slow
def test_displacements_climb_to_opposite_stars(qi: float) -> None:

    graph = Reference()._computers()
    field = RocheField(qi)

    for psi in (0.0, pi / 4, pi / 2):
        p = graph.watershed.rim(field, _depth(field, level(qi)), psi)
        delta = 1e-3 * graph.l1(field).position[0]
        assert uphill_basin(field, (p[0] - delta, p[1], p[2])) == 0
        assert uphill_basin(field, (p[0] + delta, p[1], p[2])) == 1


def test_line_cache_is_bounded_and_pickles_without_results() -> None:
    import pickle

    graph = Reference()._computers()
    computer = TracedWatershedComputer(
        l1_computer=graph.l1, outer_computer=graph.outer, maxsize=2
    )
    field = RocheField(20.0)
    line = computer(field, 0.4)
    assert computer(field, 0.4) is line
    computer(field, 0.5)
    computer(field, 0.6)
    assert computer.line.cache_info().currsize == 2
    assert computer(field, 0.4) is not line
    level = max(
        graph.outer(field).potential.value,
        graph.outer(RocheField(1 / field.qi)).potential.value,
    )
    restored = pickle.loads(pickle.dumps(computer))
    assert restored.line.cache_info().currsize == 0
    assert restored.rim_computer.cache_info().currsize == 0
    assert restored.depth_computer.cache_info().currsize == 0
    assert restored(field, 0.4).point(level) == pytest.approx(
        computer(field, 0.4).point(level), abs=1e-13
    )


def test_critical_rim_is_exact_l1_in_every_quadrant() -> None:
    graph = Reference()._computers()
    field = RocheField(0.05)
    l1 = graph.l1(field)
    for k in range(9):
        assert graph.watershed.rim(field, 0.0, k * pi / 4) == l1.position
