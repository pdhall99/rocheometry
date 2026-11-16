"""Radius selectors use each graph's lobe normalization and the view's units."""

from fractions import Fraction
from typing import cast

import pytest
from crosschecks.cartesian import CartesianSliceSurfaceVolumeComputer

import rocheometry
from rocheometry import Fitted, Reference
from rocheometry._computers.composition import ComputerSet
from rocheometry._computers.defaults import default_l1_computer
from rocheometry._computers.surface_selection import FittedSelection
from rocheometry._roche import RocheField
from rocheometry._types import BinaryFrame, ComponentId
from tests._helpers import _Wired, computer_graph


@pytest.fixture(scope="module", params=["default", "reference", "cartesian"])
def selection_computers(request: pytest.FixtureRequest) -> ComputerSet:
    if request.param == "default":
        return Fitted()._computers()
    if request.param == "reference":
        return Reference()._computers()
    return computer_graph(
        volume_computer=CartesianSliceSurfaceVolumeComputer(
            l1_computer=default_l1_computer()
        )
    )


def _check_radius_selection(
    geometry: rocheometry.ComponentGeometry | rocheometry.BinaryComponentView,
) -> None:
    lobe = geometry.lobe()
    lobe_radius = geometry.lobe_volume_radius()
    assert lobe.volume_radius() == pytest.approx(lobe_radius, rel=1e-6)
    assert geometry.volume_radius_fill_for(lobe_radius) == 1.0
    assert geometry.surface(volume_radius=lobe_radius) == lobe
    assert (
        geometry.surface(volume_radius=lobe_radius).potential == geometry.l1().potential
    )

    for fill in (0.1, 0.8, 0.99999):
        radius = fill * lobe_radius
        surface = geometry.surface(volume_radius=radius)
        intrinsic = (
            geometry.intrinsic_geometry
            if isinstance(geometry, rocheometry.BinaryComponentView)
            else geometry
        )
        fitted = isinstance(intrinsic._computers.selection, FittedSelection)
        if fitted:
            assert abs(surface.volume_radius() - radius) / lobe_radius <= 2e-6
            assert (
                abs(
                    surface.volume_radius_fill()
                    - geometry.volume_radius_fill_for(radius)
                )
                <= 2e-6
            )
        else:
            assert surface.volume_radius() == pytest.approx(radius, rel=2.0e-8, abs=0.0)
            assert surface.volume_radius_fill() == pytest.approx(
                geometry.volume_radius_fill_for(radius), rel=2e-8
            )
        if isinstance(geometry, rocheometry.BinaryComponentView):
            assert isinstance(surface, rocheometry.BinaryComponentSurfaceView)
            assert surface.component is geometry
            assert surface.intrinsic_geometry.volume_radius() == pytest.approx(
                surface.volume_radius() / geometry.binary.separation, rel=1e-15
            )
        else:
            assert isinstance(surface, rocheometry.ComponentSurfaceGeometry)
            assert surface.component is geometry


@pytest.mark.parametrize("qi", [0.03, 0.3, 1.0, 4.0])
def test_component_surface_radius_selection(
    selection_computers: ComputerSet, qi: float
) -> None:
    _check_radius_selection(
        rocheometry.ComponentGeometry(qi, accuracy=_Wired(selection_computers))
    )


@pytest.mark.parametrize("separation", [0.1, 1.0, 7.3, 12.0])
@pytest.mark.parametrize("frame", ["binary", "centre_of_mass"])
@pytest.mark.parametrize("component_id", [1, 2])
def test_binary_surface_radius_selection_in_view_units(
    selection_computers: ComputerSet,
    separation: float,
    frame: BinaryFrame,
    component_id: ComponentId,
) -> None:
    component = rocheometry.BinaryGeometry(
        0.3, separation=separation, frame=frame, accuracy=_Wired(selection_computers)
    ).component(component_id)
    _check_radius_selection(component)


@pytest.fixture(params=["component", "binary"])
def selector_geometry(
    request: pytest.FixtureRequest,
) -> rocheometry.ComponentGeometry | rocheometry.BinaryComponentView:
    def unexpected(field: RocheField, /) -> float:
        raise AssertionError(
            "invalid selectors must be rejected before computing the lobe"
        )

    graph = computer_graph(lobe_volume_computer=unexpected)
    if request.param == "component":
        return rocheometry.ComponentGeometry(0.3, accuracy=_Wired(graph))
    return rocheometry.BinaryGeometry(
        0.3, separation=7.3, accuracy=_Wired(graph)
    ).component(2)


@pytest.mark.parametrize(
    "selectors",
    [
        {},
        {"volume_radius_fill": None},
        {"volume_radius": None},
        {"volume_radius_fill": None, "volume_radius": None},
        {"volume_radius_fill": 0.8, "volume_radius": 0.1},
        {"volume_radius_fill": 0.0, "volume_radius": -1.0},
    ],
)
def test_surface_requires_exactly_one_selector(
    selector_geometry: rocheometry.ComponentGeometry | rocheometry.BinaryComponentView,
    selectors: dict[str, float | None],
) -> None:
    with pytest.raises(
        TypeError,
        match="surface: pass exactly one of volume_radius_fill, volume_radius, potential or fill_out_factor",
    ):
        selector_geometry.surface(**selectors)  # type: ignore[arg-type]


@pytest.mark.parametrize("radius", [True, False, "0.1", 0.1j, None])
def test_radius_selectors_reject_wrong_types_before_solving(
    selector_geometry: rocheometry.ComponentGeometry | rocheometry.BinaryComponentView,
    radius: object,
) -> None:
    with pytest.raises(TypeError, match="volume_radius: expected a real number"):
        selector_geometry.volume_radius_fill_for(cast(float, radius))
    if radius is not None:
        with pytest.raises(TypeError, match="volume_radius: expected a real number"):
            selector_geometry.surface(volume_radius=cast(float, radius))


@pytest.mark.parametrize(
    "radius", [0.0, -0.1, float("nan"), float("inf"), -float("inf"), 10**400]
)
def test_radius_selectors_reject_invalid_values_before_solving(
    selector_geometry: rocheometry.ComponentGeometry | rocheometry.BinaryComponentView,
    radius: float,
) -> None:
    with pytest.raises(ValueError, match="volume_radius:"):
        selector_geometry.volume_radius_fill_for(radius)
    with pytest.raises(ValueError, match="volume_radius:"):
        selector_geometry.surface(volume_radius=radius)


def test_radius_selector_normalizes_real_inputs() -> None:
    radius = Fraction(1, 10)
    surface = rocheometry.ComponentGeometry(0.3).surface(
        volume_radius=cast(float, radius)
    )
    assert surface.volume_radius() == pytest.approx(0.1, rel=2.0e-8, abs=0.0)


@pytest.mark.parametrize("component_id", [1, 2])
@pytest.mark.parametrize("convention_component", [None, 1, 2])
@pytest.mark.parametrize("fill", [0.8, 1.0])
def test_binary_potential_selection_converts_conventions(
    component_id: ComponentId, convention_component: ComponentId | None, fill: float
) -> None:
    binary = rocheometry.BinaryGeometry(0.3, separation=7.3, frame="centre_of_mass")
    component = binary.component(component_id)
    expected = component.surface(volume_radius_fill=fill)
    potential = binary.convert_potential(
        expected.potential,
        convention=(
            rocheometry.PotentialConvention.JACOBI
            if convention_component is None
            else rocheometry.PotentialConvention.KOPAL
        ),
        component_id=convention_component,
    )
    surface = component.surface(potential=potential)
    assert surface.potential == binary.convert_potential(
        potential, convention=rocheometry.PotentialConvention.JACOBI
    )
    assert surface.potential.value == pytest.approx(expected.potential.value, abs=5e-16)
    assert surface.volume_radius_fill() == pytest.approx(fill, rel=2e-8)
    assert surface == component.surface(potential=potential)
    assert hash(surface) == hash(component.surface(potential=potential))
    # A rounded potential report can lose bits of the stored depth, so it
    # need not preserve identity with the original fill-selected surface.
    if convention_component is None and fill == 1.0:
        assert surface.potential == component.l1().potential
        assert surface.volume_radius_fill() == 1.0


@pytest.mark.parametrize("kind", ["intrinsic", "binary"])
def test_open_side_potential_is_rejected(kind: str) -> None:
    geometry = rocheometry.ComponentGeometry(0.3)
    potential = rocheometry.Potential.jacobi(geometry.outer().potential.value - 0.1)
    with pytest.raises(ValueError, match="must be above L2"):
        if kind == "intrinsic":
            geometry.surface(potential=potential)
        else:
            rocheometry.BinaryGeometry(0.3).component(1).surface(potential=potential)


@pytest.mark.parametrize("selector", ["volume_radius_fill", "volume_radius"])
def test_potential_cannot_be_combined_with_another_selector(selector: str) -> None:
    geometry = rocheometry.ComponentGeometry(0.3)
    potential = geometry.l1().potential
    for component in (geometry, rocheometry.BinaryGeometry(0.3).component(1)):
        with pytest.raises(TypeError, match="exactly one"):
            component.surface(potential=potential, **{selector: 0.8})  # type: ignore[arg-type]
