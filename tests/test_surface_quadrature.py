"""Public quadrature exposes differential geometry and integral identities."""

import math
import pickle
from dataclasses import FrozenInstanceError, asdict, replace
from math import cos, fsum, hypot, pi, sin, sqrt
from typing import Any, cast

import pytest
from typing_extensions import override

import rocheometry
from rocheometry import Fitted, Reference
from rocheometry._computers.composition import ComputerSet
from rocheometry._frames import component_frame, transform_vector
from rocheometry._geometry import Point3D
from rocheometry._numerics.quadrature import gauss_legendre
from rocheometry._roche.field import RocheField
from rocheometry._types import BinaryFrame, ComponentId
from tests._helpers import RadiusDelegate, _Wired, computer_graph

QI = (1e-3, 1.0, 1e3)


def test_frozen_value_lengths_equality_pickle_and_repr() -> None:
    q = rocheometry.SurfaceQuadrature(
        ((1, 0, 0),), ((1, 0, 0),), (2,), (3,), (1,), (1,)
    )
    assert len(q) == 1
    assert pickle.loads(pickle.dumps(q)) == q
    assert hash(pickle.loads(pickle.dumps(q))) == hash(q)
    assert q != replace(q, gravities=(3,))
    assert repr(q) == (
        "SurfaceQuadrature(positions=((1.0, 0.0, 0.0),), normals=((1.0, 0.0, 0.0),), "
        "gravities=(2.0,), weights=(3.0,), mean_curvatures=(1.0,), gaussian_curvatures=(1.0,))"
    )
    with pytest.raises(FrozenInstanceError):
        setattr(q, "weights", (1,))
    for name in (
        "normals",
        "gravities",
        "weights",
        "mean_curvatures",
        "gaussian_curvatures",
    ):
        with pytest.raises(ValueError, match="match positions"):
            replace(q, **{name: ()})
        with pytest.raises(TypeError, match="expected tuple"):
            payload: dict[str, Any] = dict(asdict(q))
            payload[name] = []
            rocheometry.SurfaceQuadrature(**payload)
    with pytest.raises(ValueError, match="unit vectors"):
        replace(q, normals=((0, 0, 0),))
    with pytest.raises(ValueError, match="finite"):
        replace(q, gaussian_curvatures=(float("nan"),))
    with pytest.raises(ValueError, match="positive"):
        replace(q, weights=(0,))
    # Concave regions can have either sign of curvature.
    assert replace(
        q, mean_curvatures=(-1,), gaussian_curvatures=(-1,)
    ).gaussian_curvatures == (-1,)
    assert len(rocheometry.SurfaceQuadrature((), (), (), (), (), ())) == 0


@pytest.fixture(scope="module", params=(False, True), ids=("default", "reference"))
def graph(request: pytest.FixtureRequest) -> ComputerSet:
    return Reference()._computers() if request.param else Fitted()._computers()


def _surface(
    qi: float, case: str, graph: ComputerSet
) -> rocheometry.ComponentSurfaceGeometry:
    c = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph))
    return (
        c.surface(fill_out_factor=0.5)
        if case == "contact"
        else c.surface(volume_radius_fill=1.0 if case == "critical" else 0.8)
    )


@pytest.mark.parametrize("qi", QI)
@pytest.mark.parametrize("case", ("detached", "critical", "contact"))
def test_reader_consistency_and_node_geometry(
    graph: ComputerSet, qi: float, case: str
) -> None:
    s = _surface(qi, case, graph)
    q = s.quadrature()
    sample = graph.sample(s.component.field, s._depth)
    assert len(q) == 4 * len(sample.radii)
    area = fsum(q.weights)
    assert area == pytest.approx(s.area(), rel=1e-13)
    assert fsum(
        w * h for w, h in zip(q.weights, q.mean_curvatures, strict=True)
    ) == pytest.approx(s.integrated_mean_curvature(), rel=1e-13)
    assert fsum(
        w * g for w, g in zip(q.weights, q.gravities, strict=True)
    ) / area == pytest.approx(s.mean_gravity(), rel=1e-13)
    field = s.component.field
    for i, (p, n, g, w, h, k) in enumerate(
        zip(
            q.positions,
            q.normals,
            q.gravities,
            q.weights,
            q.mean_curvatures,
            q.gaussian_curvatures,
            strict=True,
        )
    ):
        assert field.potential(*p) == pytest.approx(s.potential.value, rel=1e-12)
        assert hypot(*n) == pytest.approx(1, rel=1e-14)
        assert fsum(x * v for x, v in zip(p, n, strict=True)) / hypot(*p) > 0
        assert g == pytest.approx(hypot(*field.gradient(*p)), rel=1e-14)
        if i % 4 == 0:
            radius = sample.radii[i // 4]
            direction = sample.rays[i // 4][0]
            assert p == tuple(radius * d for d in direction)
            for j, (sy, sz) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
                assert q.positions[i + j] == (p[0], sy * p[1], sz * p[2])
                assert q.normals[i + j] == (n[0], sy * n[1], sz * n[2])
                assert (
                    q.gravities[i + j],
                    q.weights[i + j],
                    q.mean_curvatures[i + j],
                    q.gaussian_curvatures[i + j],
                ) == (g, w, h, k)


@pytest.mark.parametrize("qi", QI)
@pytest.mark.parametrize("fill", (0.5, 0.99, 0.9999))
def test_closed_detached_integral_identities(qi: float, fill: float) -> None:
    s = rocheometry.ComponentGeometry(
        qi, accuracy=_Wired(Reference()._computers())
    ).surface(volume_radius_fill=fill)
    q = s.quadrature()
    area = fsum(q.weights)
    for axis in range(3):
        assert (
            abs(fsum(w * n[axis] for w, n in zip(q.weights, q.normals, strict=True)))
            <= 1e-12 * area
        )
    volume = (
        fsum(
            w * fsum(x * v for x, v in zip(p, n, strict=True))
            for w, p, n in zip(q.weights, q.positions, q.normals, strict=True)
        )
        / 3
    )
    assert volume == pytest.approx(s.volume(), rel=1e-9)


@pytest.mark.parametrize("qi", (1e-6, 1e-3, 0.05, 1.0, 20.0, 1e3, 1e4, 1e6))
@pytest.mark.parametrize(
    "fill,bound",
    ((0.5, 1e-6), (0.99, 1e-6), (0.999, 5e-6), (0.9999, 3e-5), (0.99999, 5e-5)),
)
def test_gauss_bonnet_with_measured_near_critical_bounds(
    qi: float, fill: float, bound: float
) -> None:
    # The smooth closed-surface identity excludes the conical critical lobe.
    assert 0 < fill < 1
    q = rocheometry.ComponentGeometry(qi).surface(volume_radius_fill=fill).quadrature()
    assert (
        abs(
            fsum(w * k for w, k in zip(q.weights, q.gaussian_curvatures, strict=True))
            - 4 * pi
        )
        <= bound
    )


def _cone_apex_angle(hessian: tuple[Point3D, Point3D, Point3D]) -> float:
    """Intrinsic total angle from the Hessian cone's spherical generator curve.

    With A=U_xx, B=-U_yy, C=-U_zz, a=sqrt(A/B), b=sqrt(A/C), a generator
    is r(psi)=(-1,a*cos(psi),b*sin(psi)). Its unit-vector curve has speed
    |r cross r'|/|r|**2. Integrating its spherical arc length gives the cone's
    total angle, independently of the sampled surface's Gaussian curvature.
    """
    a = sqrt(hessian[0][0] / -hessian[1][1])
    b = sqrt(hessian[0][0] / -hessian[2][2])
    nodes, weights = gauss_legendre(64)
    terms = []
    for node, weight in zip(nodes, weights, strict=True):
        psi = pi * (node + 1) / 4
        c, s = cos(psi), sin(psi)
        speed = sqrt(a * a * b * b + b * b * c * c + a * a * s * s) / (
            1 + a * a * c * c + b * b * s * s
        )
        terms.append(weight * speed)
    return pi * fsum(terms)  # Four quadrants, each with Jacobian pi/4.


def test_circular_cone_apex_angle_matches_analytic_circumference() -> None:
    # tan(half-angle)=2; the unit-sphere link has radius sin(half-angle)=2/sqrt(5).
    assert _cone_apex_angle(((4, 0, 0), (0, -1, 0), (0, 0, -1))) == pytest.approx(
        4 * pi / sqrt(5), rel=1e-14
    )


@pytest.mark.parametrize("qi", (1e-6, 1e-3, 0.05, 1.0, 20.0, 1e3, 1e6))
def test_critical_gauss_bonnet_includes_hessian_cone_deficit(qi: float) -> None:
    component = rocheometry.ComponentGeometry(
        qi, accuracy=_Wired(Reference()._computers())
    )
    surface = component.lobe()
    assert surface.regime is rocheometry.LevelRegime.CRITICAL
    assert surface.volume_radius_fill() == 1
    deficit = 2 * pi - _cone_apex_angle(component.l1().hessian)
    assert 0 < deficit < 2 * pi
    q = surface.quadrature()
    actual = fsum(w * k for w, k in zip(q.weights, q.gaussian_curvatures, strict=True))
    # Critical clustering resolves the regular cone tip extremely closely;
    # cancellation in K near the apex sets this measured absolute tolerance.
    assert actual == pytest.approx(4 * pi - deficit, rel=0, abs=1e-5)


@pytest.mark.parametrize("qi", (1e-6, 1e-3, 0.05, 1.0, 20.0, 1e3, 1e6))
@pytest.mark.parametrize("case", ("detached", "critical", "contact"))
def test_reference_gravity_gauss_law(qi: float, case: str) -> None:
    # Select once at default accuracy; reference integration at that potential
    # isolates Gauss's law from the separate extreme-ratio fill inversion.
    potential = _surface(qi, case, Fitted()._computers()).potential
    s = rocheometry.ComponentGeometry(
        qi, accuracy=_Wired(Reference()._computers())
    ).surface(potential=potential)
    q = s.quadrature()
    flux = fsum(w * g for w, g in zip(q.weights, q.gravities, strict=True))
    expected = 4 * pi * s.component.field.component_mass - 2 * s.volume()
    assert flux == pytest.approx(expected, rel=1e-9)
    assert s.mean_gravity() == pytest.approx(expected / s.area(), rel=1e-9)


@pytest.mark.parametrize("component_id", (1, 2))
@pytest.mark.parametrize("frame", ("binary", "centre_of_mass"))
@pytest.mark.parametrize("case", ("detached", "critical", "contact"))
def test_binary_view_transforms_and_measurement_units(
    component_id: ComponentId, frame: BinaryFrame, case: str
) -> None:
    binary = rocheometry.BinaryGeometry(0.4, separation=3.5, frame=frame)
    component = binary.component(component_id)
    intrinsic = _surface(component.intrinsic_geometry.qi, case, binary._computers)
    view = component.surface(potential=intrinsic.potential)
    local = view.intrinsic_geometry.quadrature()
    q = view.quadrature()
    a = binary.separation
    assert q.positions == tuple(component._output_position(p) for p in local.positions)
    assert q.normals == tuple(
        transform_vector(
            n, source=component_frame(component_id), target=frame, q=binary.q
        )
        for n in local.normals
    )
    assert q.gravities == local.gravities
    assert q.weights == tuple(w * a**2 for w in local.weights)
    assert q.mean_curvatures == tuple(h / a for h in local.mean_curvatures)
    assert q.gaussian_curvatures == tuple(k / a / a for k in local.gaussian_curvatures)
    area = fsum(q.weights)
    assert area == pytest.approx(view.area(), rel=1e-13)
    assert fsum(
        w * h for w, h in zip(q.weights, q.mean_curvatures, strict=True)
    ) == pytest.approx(view.integrated_mean_curvature(), rel=1e-13)
    assert fsum(
        w * g for w, g in zip(q.weights, q.gravities, strict=True)
    ) / area == pytest.approx(view.mean_gravity(), rel=1e-13)


def test_quadrature_and_readers_share_solved_radii() -> None:
    calls = 0
    reference = Reference()._computers()

    def radius(field: RocheField, potential: float, direction: Point3D, /) -> float:
        nonlocal calls
        calls += 1
        return reference.radius(field, potential, direction)

    g = computer_graph(
        radius_computer=RadiusDelegate(radius), polar_order=8, azimuth_order=6
    )
    s = rocheometry.ComponentGeometry(0.4, accuracy=_Wired(g)).lobe()
    assert s.area() > 0
    assert len(s.quadrature()) == 4 * 8 * 6
    assert s.integrated_mean_curvature() > 0
    assert s.mean_gravity() > 0
    assert calls == 8 * 6


def test_sphere_curvature_conventions_and_gaussian_formula() -> None:
    from rocheometry._computers.surface_sample import SurfaceSample
    from rocheometry._roche.field import RocheField
    from rocheometry._surface_quadrature import _quadrature_from_sample

    class SphericalField(RocheField):
        # U=1/r+r^2/3 is radial with Laplacian 2 away from the origin.
        @override
        def gradient(self, x: float, y: float = 0.0, z: float = 0.0) -> Point3D:
            r = hypot(x, y, z)
            factor = -1 / r**3 + 2 / 3
            return factor * x, factor * y, factor * z

        @override
        def hessian(
            self, x: float, y: float = 0.0, z: float = 0.0
        ) -> tuple[Point3D, Point3D, Point3D]:
            r = hypot(x, y, z)
            normal = (x / r, y / r, z / r)
            return cast(
                tuple[Point3D, Point3D, Point3D],
                tuple(
                    tuple(
                        (3 * normal[i] * normal[j] - (i == j)) / r**3
                        + (2 / 3 if i == j else 0)
                        for j in range(3)
                    )
                    for i in range(3)
                ),
            )

    radius = 0.25
    q = _quadrature_from_sample(
        SphericalField(1), SurfaceSample((((0.6, 0.8, 0), 4 * pi),), (radius,))
    )
    assert q.mean_curvatures == pytest.approx((1 / radius,) * 4, rel=1e-14)
    assert q.gaussian_curvatures == pytest.approx((1 / radius**2,) * 4, rel=1e-14)
    assert fsum(q.weights) == pytest.approx(4 * pi * radius**2, rel=1e-14)


@pytest.mark.slow
@pytest.mark.parametrize(
    "accuracy", (rocheometry.Fitted(), rocheometry.Reference()), ids=repr
)
@pytest.mark.parametrize("qi", tuple(10.0**k for k in range(-8, 9)))
def test_quadrature_nodes_are_outward_and_mirrored_at_every_decade(
    qi: float, accuracy: rocheometry.Fitted | rocheometry.Reference
) -> None:
    # Each group of four nodes is one node and its y and z reflections, exactly,
    # and every normal points away from the component centre.
    geometry = rocheometry.ComponentGeometry(qi, accuracy=accuracy)
    for surface in (
        *(geometry.surface(volume_radius_fill=f) for f in (0.3, 0.8, 1.0)),
        geometry.surface(fill_out_factor=0.5),
    ):
        q = surface.quadrature()
        assert len(q.positions) % 4 == 0
        for j in range(0, len(q.positions), 4):
            p, n = q.positions[j], q.normals[j]
            for k, (sy, sz) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
                assert q.positions[j + k] == (p[0], sy * p[1], sz * p[2])
                assert q.normals[j + k] == (n[0], sy * n[1], sz * n[2])
        assert all(
            math.fsum(a * b for a, b in zip(n, p, strict=True)) > 0
            for n, p in zip(q.normals, q.positions, strict=True)
        )
