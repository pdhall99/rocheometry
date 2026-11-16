from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
from math import acos, cos, exp, floor, fsum, log2, pi, sin, sqrt

import pytest

from rocheometry import ComponentGeometry
from rocheometry._roche import RocheField


@pytest.mark.parametrize("qi", (1e-8, 1e-6, 1e-3, 1.0, 1e8))
@pytest.mark.parametrize("outer", (False, True))
def test_axis_level_offset_rounds_only_after_subtracting_level(
    qi: float, outer: bool
) -> None:
    geometry = ComponentGeometry(qi)
    point = geometry.outer() if outer else geometry.l1()
    field = geometry.field
    x, level = point.position[0], point.potential.value
    with localcontext() as context:
        context.prec = 80
        a, m, c, u = map(
            Decimal.from_float, (x, field.component_mass, field.companion_mass, level)
        )
        expected = float(
            m / abs(a) + c / abs(1 - a) + (m + c) * (a - c / (m + c)) ** 2 / 2
        )
    assert float(field._axis_potential_exact(x)) == expected
    hits = field._axis_potential_exact.cache_info().hits
    assert float(RocheField(qi)._axis_potential_exact(x)) == expected
    assert field._axis_potential_exact.cache_info().hits == hits + 1


@pytest.mark.parametrize("qi", [1e-6, 1.0, 1e6])
def test_mass_fractions_are_read_only(qi: float) -> None:
    field = RocheField(qi)
    assert field.component_mass / field.companion_mass == pytest.approx(qi)
    assert field.component_mass + field.companion_mass == 1.0
    for name in ("component_mass", "companion_mass"):
        with pytest.raises(FrozenInstanceError):
            setattr(field, name, 0.0)


@pytest.mark.parametrize("qi", [1e-6, 1.0, 1e6])
def test_axial_third_derivatives_match_gradient_differences_at_l1(qi: float) -> None:
    geometry = ComponentGeometry(qi)
    _check_axial_third_derivatives(geometry.field, geometry.l1().position[0])


@pytest.mark.parametrize("x", [-0.4, 0.3, 1.4])
def test_axial_third_derivatives_use_signed_distances(x: float) -> None:
    _check_axial_third_derivatives(RocheField(0.37), x)


def _check_axial_third_derivatives(field: RocheField, x: float) -> None:
    u_xxx, u_xyy = field.axial_third_derivatives(x)
    # A dyadic step keeps the equal-mass stencil exactly symmetric about L1.
    step = 2.0 ** floor(log2(2e-3 * min(abs(x), abs(x - 1))))
    for axis, expected in enumerate((u_xxx, u_xyy, u_xyy)):

        def gradient_x(offset: int) -> float:
            point = [x, 0.0, 0.0]
            point[axis] += offset * step
            return field.gradient(*point)[0]

        # Fourth-order second difference of U_x, independent of the Hessian.
        measured = fsum(
            weight * gradient_x(offset)
            for offset, weight in ((-2, -1), (-1, 16), (0, -30), (1, 16), (2, -1))
        ) / (12 * step**2)
        assert measured == pytest.approx(expected, rel=1e-8, abs=0.0)


@pytest.mark.parametrize("x", [0.0, 1.0, float("inf"), float("nan")])
def test_axial_third_derivatives_reject_singular_or_nonfinite_x(x: float) -> None:
    with pytest.raises(ValueError):
        RocheField(0.37).axial_third_derivatives(x)


def _scaled_potential(qi: float, x: float, y: float, z: float) -> float:
    r1 = sqrt(x * x + y * y + z * z)
    dx2 = x - 1.0
    r2 = sqrt(dx2 * dx2 + y * y + z * z)
    return qi / r1 + 1.0 / r2 + 0.5 * (1.0 + qi) * (x * x + y * y) - x


def _scaled_gradient(qi: float, x: float, y: float, z: float) -> tuple[float, ...]:
    r1_sq = x * x + y * y + z * z
    dx2 = x - 1.0
    r2_sq = dx2 * dx2 + y * y + z * z
    r1_cubed = r1_sq * sqrt(r1_sq)
    r2_cubed = r2_sq * sqrt(r2_sq)
    return (
        -qi * x / r1_cubed - dx2 / r2_cubed + (1.0 + qi) * x - 1.0,
        -qi * y / r1_cubed - y / r2_cubed + (1.0 + qi) * y,
        -qi * z / r1_cubed - z / r2_cubed,
    )


def _scaled_centreline_plane_hessian(
    qi: float,
    x: float,
    u: float,
    inclination: float,
) -> tuple[float, ...]:
    r1_sq = x * x + u * u
    dx2 = x - 1.0
    r2_sq = dx2 * dx2 + u * u
    r1_fifth = r1_sq * r1_sq * sqrt(r1_sq)
    r2_fifth = r2_sq * r2_sq * sqrt(r2_sq)
    return (
        qi * (3.0 * x * x - r1_sq) / r1_fifth
        + (3.0 * dx2 * dx2 - r2_sq) / r2_fifth
        + (1.0 + qi),
        3.0 * qi * x * u / r1_fifth + 3.0 * dx2 * u / r2_fifth,
        qi * (3.0 * u * u - r1_sq) / r1_fifth
        + (3.0 * u * u - r2_sq) / r2_fifth
        + (1.0 + qi) * cos(inclination) ** 2,
    )


def test_analytic_gradient_matches_central_difference() -> None:
    field = RocheField(0.37)
    point = (0.21, 0.17, -0.13)
    step = 1.0e-6
    expected = field.gradient(*point)

    finite_difference = []
    for axis in range(3):
        plus = list(point)
        minus = list(point)
        plus[axis] += step
        minus[axis] -= step
        derivative = (field.potential(*plus) - field.potential(*minus)) / (2.0 * step)
        finite_difference.append(derivative)

    assert tuple(finite_difference) == pytest.approx(
        expected,
        rel=2.0e-9,
        abs=2.0e-9,
    )


@pytest.mark.parametrize("inclination", [pi / 2, pi / 3, 0.0])
def test_centreline_plane_hessian_matches_gradient_difference(
    inclination: float,
) -> None:
    field = RocheField(0.37)
    x = 0.21
    u = 0.17
    step = 1.0e-6
    cos_i = cos(inclination)
    sin_i = sin(inclination)

    def plane_gradient(x_value: float, u_value: float) -> tuple[float, float]:
        gradient_x, gradient_y, gradient_z = field.gradient(
            x_value,
            u_value * cos_i,
            -u_value * sin_i,
        )
        gradient_u = gradient_y * cos_i - gradient_z * sin_i
        return gradient_x, gradient_u

    left = plane_gradient(x - step, u)
    right = plane_gradient(x + step, u)
    below = plane_gradient(x, u - step)
    above = plane_gradient(x, u + step)

    finite_xx = (right[0] - left[0]) / (2.0 * step)
    finite_ux = (right[1] - left[1]) / (2.0 * step)
    finite_xu = (above[0] - below[0]) / (2.0 * step)
    finite_uu = (above[1] - below[1]) / (2.0 * step)

    u_xx, u_xu, u_uu = field.centreline_plane_hessian(
        x,
        u,
        inclination,
    )
    assert u_xx == pytest.approx(finite_xx, rel=2.0e-9, abs=2.0e-9)
    assert u_xu == pytest.approx(finite_xu, rel=2.0e-9, abs=2.0e-9)
    assert u_xu == pytest.approx(finite_ux, rel=2.0e-9, abs=2.0e-9)
    assert u_uu == pytest.approx(finite_uu, rel=2.0e-9, abs=2.0e-9)


@pytest.mark.parametrize("inclination", [pi / 2, pi / 4, 0.0])
def test_critical_tangent_lies_on_quadratic_cone(
    inclination: float,
) -> None:
    field = RocheField(1.0)
    critical_x = 0.5
    tangent_x, tangent_u = field.critical_upper_tangent(critical_x, inclination)
    u_xx, u_xu, u_uu = field.centreline_plane_hessian(
        critical_x,
        inclination=inclination,
    )

    quadratic = (
        u_xx * tangent_x * tangent_x
        + 2.0 * u_xu * tangent_x * tangent_u
        + u_uu * tangent_u * tangent_u
    )
    assert quadratic == pytest.approx(0.0, abs=2.0e-12)
    assert tangent_x * tangent_x + tangent_u * tangent_u == pytest.approx(1.0)
    assert tangent_x < 0.0
    assert tangent_u > 0.0


def test_centreline_plane_hessian_rejects_nonphysical_lambda() -> None:
    with pytest.raises(ValueError, match="inclination"):
        RocheField(1.0).centreline_plane_hessian(0.5, inclination=float("nan"))


@pytest.mark.parametrize("qi", [1.0e-4, 0.1, 1.0, 10.0, 1.0e4])
def test_jacobi_potential_is_affine_rescaling_of_scaled_field(qi: float) -> None:
    x, y, z = (0.31, 0.22, -0.09)
    jacobi = RocheField(qi).potential(x, y, z)
    recovered_scaled = (1.0 + qi) * jacobi - 0.5 / (1.0 + qi)
    assert recovered_scaled == pytest.approx(
        _scaled_potential(qi, x, y, z),
        rel=2.0e-15,
        abs=2.0e-15,
    )


@pytest.mark.parametrize("qi", [1.0e-4, 0.1, 1.0, 10.0, 1.0e4])
def test_jacobi_gradient_is_scaled_scaled_gradient(qi: float) -> None:
    point = (0.31, 0.22, -0.09)
    scale = 1.0 + qi
    recovered_scaled = tuple(scale * value for value in RocheField(qi).gradient(*point))
    assert recovered_scaled == pytest.approx(
        _scaled_gradient(qi, *point),
        rel=3.0e-15,
        abs=3.0e-15,
    )


@pytest.mark.parametrize("qi", [1.0e-4, 0.1, 1.0, 10.0, 1.0e4])
@pytest.mark.parametrize("inclination", [pi / 2, acos(sqrt(0.37)), 0.0])
def test_jacobi_hessian_is_scaled_hessian(
    qi: float,
    inclination: float,
) -> None:
    x, u = (0.31, 0.22)
    scale = 1.0 + qi
    recovered_scaled = tuple(
        scale * value
        for value in RocheField(qi).centreline_plane_hessian(x, u, inclination)
    )
    assert recovered_scaled == pytest.approx(
        _scaled_centreline_plane_hessian(qi, x, u, inclination),
        rel=4.0e-15,
        abs=4.0e-15,
    )


@pytest.mark.parametrize("qi", [0.1, 0.7, 2.0, 10.0])
def test_jacobi_potential_obeys_component_exchange_symmetry(qi: float) -> None:
    x, y, z = (0.31, 0.22, -0.09)
    original = RocheField(qi).potential(x, y, z)
    swapped = RocheField(1.0 / qi).potential(1.0 - x, -y, z)
    assert swapped == pytest.approx(original, rel=2.0e-15, abs=2.0e-15)


@pytest.mark.parametrize("qi", [0.1, 0.7, 2.0, 10.0])
def test_jacobi_gradient_obeys_component_exchange_symmetry(qi: float) -> None:
    x, y, z = (0.31, 0.22, -0.09)
    original = RocheField(qi).gradient(x, y, z)
    swapped = RocheField(1.0 / qi).gradient(1.0 - x, -y, z)
    expected = (-original[0], -original[1], original[2])
    assert swapped == pytest.approx(expected, rel=3.0e-15, abs=3.0e-15)


@pytest.mark.parametrize("qi", [0.1, 0.7, 2.0, 10.0])
def test_orbital_hessian_obeys_component_exchange_symmetry(qi: float) -> None:
    x, u = (0.31, 0.22)
    original = RocheField(qi).centreline_plane_hessian(x, u, 1.0)
    swapped = RocheField(1.0 / qi).centreline_plane_hessian(1.0 - x, -u, 1.0)
    assert swapped == pytest.approx(original, rel=4.0e-15, abs=4.0e-15)


def test_equal_mass_l1_has_canonical_jacobi_value() -> None:
    field = RocheField(1.0)
    assert field.potential(0.5) == pytest.approx(2.0, abs=2.0e-15)


@pytest.mark.parametrize(
    "point", [(0.21, 0.17, -0.12), (-0.4, 0.3, 0.2), (0.8, -0.23, 0.31)]
)
def test_hessian_is_symmetric_and_matches_gradient_differences(
    point: tuple[float, float, float],
) -> None:
    field = RocheField(0.37)
    hessian = field.hessian(*point)
    scale = max(abs(value) for row in hessian for value in row)
    step = 1e-6
    for j in range(3):
        plus, minus = list(point), list(point)
        plus[j] += step
        minus[j] -= step
        upper, lower = field.gradient(*plus), field.gradient(*minus)
        for i in range(3):
            assert hessian[i][j] == hessian[j][i]
            assert (
                abs(hessian[i][j] - (upper[i] - lower[i]) / (2 * step)) <= 1e-6 * scale
            )


@pytest.mark.parametrize("x", [0.0, 1.0])
def test_hessian_and_branch_curvature_reject_point_masses(x: float) -> None:
    field = RocheField(0.37)
    with pytest.raises(ValueError, match="component position"):
        field.hessian(x)
    with pytest.raises(ValueError, match="component position"):
        field.critical_branch_curvature(x)


@pytest.mark.parametrize(
    "point",
    [(float("nan"), 0.1, 0.2), (0.1, float("inf"), 0.2), (0.1, 0.2, -float("inf"))],
)
def test_hessian_rejects_nonfinite_coordinates(
    point: tuple[float, float, float],
) -> None:
    with pytest.raises(ValueError, match="finite"):
        RocheField(0.37).hessian(*point)


@pytest.mark.parametrize("qi", [0.1, 1.0, 10.0])
@pytest.mark.parametrize(
    "point", [(0.21, 0.17, -0.12), (-0.4, 0.3, 0.2), (0.8, -0.23, 0.31)]
)
def test_log_qi_derivative_matches_fixed_point_potential_difference(
    qi: float, point: tuple[float, float, float]
) -> None:
    step = 1e-5
    upper = RocheField(qi * exp(step)).potential(*point)
    lower = RocheField(qi * exp(-step)).potential(*point)
    assert RocheField(qi).log_qi_derivative(*point) == pytest.approx(
        (upper - lower) / (2 * step), rel=1e-8, abs=0
    )


@pytest.mark.parametrize("x", [0.0, 1.0])
def test_log_qi_derivative_rejects_point_masses(x: float) -> None:
    with pytest.raises(ValueError, match="component position"):
        RocheField(0.3).log_qi_derivative(x)


@pytest.mark.parametrize(
    "point",
    [(float("nan"), 0.1, 0.2), (0.1, float("inf"), 0.2), (0.1, 0.2, -float("inf"))],
)
def test_log_qi_derivative_rejects_nonfinite_coordinates(
    point: tuple[float, float, float],
) -> None:
    with pytest.raises(ValueError, match="finite"):
        RocheField(0.3).log_qi_derivative(*point)


@pytest.mark.parametrize("dx", (-1e-7, -1e-10, -1e-12))
def test_potential_difference_retains_neck_precision(dx: float) -> None:
    from decimal import Decimal, localcontext
    from sys import float_info

    field = RocheField(1.0)
    reference = (0.5, 0.0001, 0.2)
    point = (0.5 + dx, 0.0001 + dx / 10, 0.2)
    with localcontext() as context:
        context.prec = 60

        def potential(p: tuple[float, float, float]) -> Decimal:
            x, y, z = map(Decimal.from_float, p)
            half = Decimal(".5")
            return (
                half / (x * x + y * y + z * z).sqrt()
                + half / ((1 - x) ** 2 + y * y + z * z).sqrt()
                + half * ((x - half) ** 2 + y * y)
            )

        exact = float(potential(point) - potential(reference))
    assert field.potential_difference(point, reference) == pytest.approx(
        exact, rel=0.0, abs=16 * float_info.epsilon * abs(dx)
    )
