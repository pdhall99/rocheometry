"""Critical Hessians agree with potential differences and frame changes."""

from dataclasses import replace
from decimal import Decimal, localcontext
from math import fsum, hypot, sqrt
from typing import cast

import pytest

import rocheometry
from rocheometry import Fitted
from rocheometry._computers.critical_point import (
    BisectionCriticalPointComputer,
    CriticalPointComputer,
)
from rocheometry._frames import component_frame, transform_vector
from rocheometry._geometry import Point3D
from rocheometry._roche import RocheField
from rocheometry._types import BinaryFrame, ComponentId
from tests._helpers import _Wired, computer_graph

QI = (1e-8, 1e-3, 1.0, 1e3, 1e8)


def _check_structure(
    point: rocheometry.CriticalPoint, *, triangular: bool, trace: float = 2
) -> None:
    assert isinstance(point, rocheometry.CriticalPoint)
    h = point.hessian
    assert fsum(h[i][i] for i in range(3)) == pytest.approx(trace, rel=0, abs=1e-12)
    assert all(h[i][j] == h[j][i] for i in range(3) for j in range(3))
    assert h[0][2] == h[1][2] == 0
    assert tuple(sorted(point.eigenvalues)) == point.eigenvalues
    assert fsum(point.eigenvalues) == pytest.approx(trace, rel=0, abs=1e-12)
    if triangular:
        assert point.eigenvalues[0] < 0 < point.eigenvalues[1] <= point.eigenvalues[2]
        assert h[2][2] < 0
        assert point.morse_index == 1
    else:
        assert all(h[i][j] == 0 for i in range(3) for j in range(3) if i != j)
        assert h[0][0] > 0 > h[1][1]
        assert h[2][2] < 0
        assert point.morse_index == 2


@pytest.mark.parametrize("qi", QI)
def test_local_points_and_direct_producers_have_hessians(qi: float) -> None:
    g = rocheometry.ComponentGeometry(qi)
    for point, triangular in (
        (g.l1(), False),
        (g.outer(), False),
        (g.triangular(), True),
    ):
        _check_structure(point, triangular=triangular)
        assert point.hessian == g.field.hessian(*point.position)
    assert BisectionCriticalPointComputer.l1()(g.field) == g.l1()
    assert BisectionCriticalPointComputer.outer()(g.field) == g.outer()


@pytest.mark.parametrize("q", QI)
@pytest.mark.parametrize("frame", ("binary", "centre_of_mass"))
@pytest.mark.parametrize("separation", (0.25, 1.0, 12.0))
def test_all_five_binary_points_have_invariant_spectrum_and_laplace_trace(
    q: float, frame: BinaryFrame, separation: float
) -> None:
    b = rocheometry.BinaryGeometry(q, frame=frame, separation=separation)
    intrinsic = rocheometry.BinaryGeometry(q)
    for i, point in enumerate((b.l1(), b.l2(), b.l3(), b.l4(), b.l5()), 1):
        _check_structure(point, triangular=i >= 4)
        original = getattr(intrinsic, f"l{i}")()
        assert point.hessian == original.hessian
        assert point.eigenvalues == original.eigenvalues
        assert point.morse_index == original.morse_index


def _decimal_potential(qi: float, position: tuple[Decimal, ...]) -> Decimal:
    """Evaluate the defining potential independently at high precision.

    Double differences lose digits near tiny Hill radii; high precision keeps
    the fourth-order stencil meaningful without using analytic derivatives.
    """
    x, y, z = position
    q = Decimal.from_float(qi)
    m0, m1 = q / (1 + q), 1 / (1 + q)
    r0 = (x * x + y * y + z * z).sqrt()
    r1 = ((x - 1) ** 2 + y * y + z * z).sqrt()
    return m0 / r0 + m1 / r1 + ((x - m1) ** 2 + y * y) / 2


def _difference(qi: float, point: Point3D, i: int, j: int) -> float:
    with localcontext() as context:
        context.prec = 60
        p = tuple(Decimal.from_float(x) for x in point)
        distance = min(hypot(*point), hypot(point[0] - 1, point[1], point[2]))
        step = Decimal.from_float(distance) * Decimal("0.0001")

        def value(a: int, b: int = 0) -> Decimal:
            coords = list(p)
            coords[i] += a * step
            coords[j] += b * step
            return _decimal_potential(qi, tuple(coords))

        if i == j:
            return float(
                (-value(2) + 16 * value(1) - 30 * value(0) + 16 * value(-1) - value(-2))
                / (12 * step * step)
            )
        stencil = ((-2, 1), (-1, -8), (1, 8), (2, -1))
        return float(
            sum(wa * wb * value(a, b) for a, wa in stencil for b, wb in stencil)
            / (144 * step * step)
        )


@pytest.mark.parametrize("q", QI)
def test_every_hessian_entry_matches_fourth_order_potential_differences(
    q: float,
) -> None:
    g = rocheometry.ComponentGeometry(q)
    b = rocheometry.BinaryGeometry(q)
    for point in (
        g.l1(),
        g.outer(),
        g.triangular(),
        b.l1(),
        b.l2(),
        b.l3(),
        b.l4(),
        b.l5(),
    ):
        for i in range(3):
            for j in range(3):
                assert point.hessian[i][j] == pytest.approx(
                    _difference(q, point.position, i, j), rel=1e-7, abs=1e-9
                ), (q, point.position, i, j)


@pytest.mark.parametrize("q", QI)
@pytest.mark.parametrize("frame", ("binary", "centre_of_mass"))
@pytest.mark.parametrize("component_id", (1, 2))
@pytest.mark.parametrize("separation", (0.25, 1.0, 3.5, 12.0))
def test_view_hessian_only_rotates_in_both_frames(
    q: float,
    frame: BinaryFrame,
    component_id: ComponentId,
    separation: float,
) -> None:
    binary = rocheometry.BinaryGeometry(q, separation=separation, frame=frame)
    component = binary.component(component_id)
    local = component.intrinsic_geometry
    # Columns of R are transformed local unit vectors; translations are ignored.
    basis: tuple[Point3D, ...] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    columns = tuple(
        transform_vector(v, source=component_frame(component_id), target=frame, q=q)
        for v in basis
    )
    for original, viewed in zip(
        (local.l1(), local.outer(), local.triangular()),
        (component.l1(), component.outer(), component.triangular()),
        strict=True,
    ):
        assert viewed.potential == original.potential
        for i in range(3):
            for j in range(3):
                expected = fsum(
                    columns[k][i] * original.hessian[k][other] * columns[other][j]
                    for k in range(3)
                    for other in range(3)
                )
                assert viewed.hessian[i][j] == expected
        assert viewed.morse_index == original.morse_index
        assert viewed.eigenvalues == original.eigenvalues
    # Both x and y reverse for component 2: their signs cancel in xy.
    h = component.triangular().hessian
    assert h[0][1] == local.triangular().hessian[0][1]
    assert (
        h
        == rocheometry.BinaryGeometry(q, separation=separation, frame="binary")
        .component(component_id)
        .triangular()
        .hessian
    )


def test_equal_mass_mirrors_and_triangular_cross_entry_signs() -> None:
    equal = rocheometry.BinaryGeometry(1)
    assert equal.l2().hessian == equal.l3().hessian
    assert equal.l4().hessian == equal.l5().hessian
    assert equal.component(1).l1().hessian == equal.component(2).l1().hessian
    unequal = rocheometry.BinaryGeometry(0.4)
    assert unequal.l4().hessian[0][1] == pytest.approx(-unequal.l5().hessian[0][1])
    assert unequal.l4().hessian[0][1] != 0


@pytest.mark.parametrize("role", ("l1", "outer"))
@pytest.mark.parametrize("memoized", (False, True))
def test_custom_noncritical_point_computers_are_rejected(
    role: str, memoized: bool
) -> None:
    def point(field: RocheField, /) -> object:
        return object()

    # Deliberately violate the protocol to check the runtime error, including
    # custom graphs with and without a memoized critical-point producer.
    producer = cast(CriticalPointComputer, point)
    if memoized:
        graph = computer_graph(
            l1_computer=producer if role == "l1" else None,
            outer_computer=producer if role == "outer" else None,
        )
    else:
        graph = (
            replace(Fitted()._computers(), l1=producer)
            if role == "l1"
            else replace(Fitted()._computers(), outer=producer)
        )
    geometry = rocheometry.ComponentGeometry(1, accuracy=_Wired(graph))
    for _ in range(2):
        with pytest.raises(TypeError, match=rf"{role}.*expected CriticalPoint"):
            getattr(geometry, role)()
        view = rocheometry.BinaryGeometry(1, accuracy=_Wired(graph)).component(2)
        with pytest.raises(TypeError, match=rf"{role}.*expected CriticalPoint"):
            getattr(view, role)()


@pytest.mark.parametrize("qi", QI)
def test_l1_cone_half_angle_matches_critical_section_tangent(qi: float) -> None:
    geometry = rocheometry.ComponentGeometry(qi)
    critical = geometry.l1()
    path = geometry.lobe().section().paths[0]
    first = next(
        segment
        for segment in path.segments
        if segment.p0 == critical.position[:2] and segment.p1 != segment.p0
    )
    dx, dy = first.p1[0] - first.p0[0], first.p1[1] - first.p0[1]
    assert abs(dy / dx) == pytest.approx(
        sqrt(critical.hessian[0][0] / abs(critical.hessian[1][1])), rel=1e-9
    )


@pytest.mark.parametrize("qi", QI)
def test_triangular_point_is_the_closed_form_equilateral_point(qi: float) -> None:
    g = rocheometry.ComponentGeometry(qi)
    point = g.triangular()
    assert point.position == (0.5, sqrt(3.0) / 2.0, 0.0)
    x, y, z = point.position
    assert hypot(x, y) == pytest.approx(1.0, abs=1e-16)
    assert hypot(x - 1.0, y) == pytest.approx(1.0, abs=1e-16)
    force = g.field.gradient(x, y, z)
    assert max(map(abs, force)) <= 4e-16
    assert point.potential.value == pytest.approx(g.field.potential(x, y, z), rel=4e-16)


@pytest.mark.parametrize(
    "accuracy", (rocheometry.Fitted(), rocheometry.Reference()), ids=repr
)
@pytest.mark.parametrize("qi", tuple(10.0**k for k in range(-8, 9)))
def test_morse_indices_at_every_decade(
    qi: float, accuracy: rocheometry.Fitted | rocheometry.Reference
) -> None:
    g = rocheometry.ComponentGeometry(qi, accuracy=accuracy)
    assert (g.l1().morse_index, g.outer().morse_index) == (2, 2)
    assert g.triangular().morse_index == 1
