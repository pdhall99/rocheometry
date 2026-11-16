"""Surfaces are identified by depth, and levels convert to depth exactly."""

import ast
import math
import pickle
from fractions import Fraction
from pathlib import Path

import pytest

import rocheometry
from rocheometry._accuracy import _Accuracy
from rocheometry._levels import _levels
from rocheometry._roche.mass import mass_fractions
from rocheometry._types import ComponentId


@pytest.mark.parametrize("accuracy", (rocheometry.Fitted(), rocheometry.Reference()))
@pytest.mark.parametrize("qi", (1e-8, 1e-6, 1.0, 1e6, 1e8))
@pytest.mark.parametrize("factor", (1e-12, 1e-6, 0.01, 0.5, 0.99))
def test_factor_round_trip_uses_depth(
    accuracy: _Accuracy, qi: float, factor: float
) -> None:
    g = rocheometry.ComponentGeometry(qi, accuracy=accuracy)
    surface = g.surface(fill_out_factor=factor)
    assert abs(surface.fill_out_factor() - factor) <= 2 * math.ulp(factor)
    restored = pickle.loads(pickle.dumps(surface))
    assert restored == surface
    assert hash(restored) == hash(surface)
    assert restored._depth == surface._depth
    assert restored.component._computers is accuracy._computers()


@pytest.mark.parametrize("q", (1e-8, 1e-6, 1e-3, 0.7, 1.0, 1e3, 1e6, 1e8))
def test_binary_parts_share_depth_and_preserve_reciprocal_pair(q: float) -> None:
    binary = rocheometry.BinaryGeometry(q, separation=12, frame="centre_of_mass")
    one, two = binary.component(1), binary.component(2)
    for factor in (1e-12, 1e-6, 0.01, 0.5, 0.99):
        a = one.surface(fill_out_factor=factor)
        b = two.surface(fill_out_factor=factor)
        assert a._depth == b._depth
        assert a.fill_out_factor() == b.fill_out_factor()
        assert a.potential == b.potential
        restored = pickle.loads(pickle.dumps(b))
        assert restored == b
        assert hash(restored) == hash(b)
        assert restored.intrinsic_geometry.component.field == (
            one.intrinsic_geometry.field.reciprocal()
        )
        assert restored.intrinsic_geometry.component._computers is (binary._computers)


@pytest.mark.parametrize("q", (1e-8, 1e-3, 0.7, 4.378680301162732, 1e8))
@pytest.mark.parametrize("component_id", (1, 2))
def test_kopal_conversion_uses_exact_swapped_mass_pair(
    q: float, component_id: ComponentId
) -> None:
    m, c = map(Fraction, mass_fractions(q))
    if component_id == 2:
        m, c = c, m
    offset = c * c / (2 * (m + c))
    source = rocheometry.Potential.kopal(3.25, component_id=component_id)
    converted = rocheometry.convert_potential(
        source, q=q, convention=rocheometry.PotentialConvention.JACOBI
    )
    assert converted.value == float(Fraction(source.value) * m + offset)
    back = rocheometry.convert_potential(
        converted,
        q=q,
        convention=rocheometry.PotentialConvention.KOPAL,
        component_id=component_id,
    )
    assert back.value == float((Fraction(converted.value) - offset) / m)
    binary = rocheometry.BinaryGeometry(q)
    source = rocheometry.convert_potential(
        rocheometry.Potential.jacobi(binary.l1().potential.value + 1),
        q=q,
        convention=rocheometry.PotentialConvention.KOPAL,
        component_id=component_id,
    )
    a = binary.component(1).surface(potential=source)
    b = binary.component(2).surface(potential=source)
    assert a._depth == b._depth
    exact_depth = _levels(binary.geometry1.field, binary._computers).depth(source)
    assert a._depth == exact_depth


@pytest.mark.parametrize("q", (1e-8, 1e-6, 1e-3, 0.7, 1.0, 1e3, 1e6, 1e8))
@pytest.mark.parametrize("component_id", (1, 2))
@pytest.mark.parametrize("factor", (-1.0, 0.01, 0.5, 0.99))
def test_kopal_depth_rounds_once_and_jacobi_report_obeys_absolute_bound(
    q: float, component_id: ComponentId, factor: float
) -> None:
    binary = rocheometry.BinaryGeometry(q)
    level = binary.component(1).surface(fill_out_factor=factor).potential
    kopal = binary.convert_potential(
        level,
        convention=rocheometry.PotentialConvention.KOPAL,
        component_id=component_id,
    )
    m, c = map(Fraction, mass_fractions(q))
    if component_id == 2:
        m, c = c, m
    exact_jacobi = Fraction(kopal.value) * m + c * c / (2 * (m + c))
    levels = _levels(binary.geometry1.field, binary._computers)
    expected = float(levels.l1 - exact_jacobi)
    first = binary.component(1).surface(potential=kopal)
    second = binary.component(2).surface(potential=kopal)
    assert first._depth == second._depth == expected
    jacobi = binary.convert_potential(
        kopal, convention=rocheometry.PotentialConvention.JACOBI
    )
    assert jacobi.value == float(exact_jacobi)
    materialized = binary.component(1).surface(potential=jacobi)
    # Absolute units: half a ULP of the level can be many ULPs of depth.
    bound = Fraction(math.ulp(jacobi.value)) / 2 + Fraction(math.ulp(expected))
    assert abs(Fraction(first._depth) - Fraction(materialized._depth)) <= bound


@pytest.mark.parametrize("accuracy", (rocheometry.Fitted(), rocheometry.Reference()))
@pytest.mark.parametrize("q", (1e-8, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e8))
def test_opening_rounding_keeps_depth_without_a_computer_clamp(
    accuracy: _Accuracy, q: float
) -> None:
    binary = rocheometry.BinaryGeometry(q, accuracy=accuracy)
    factor = math.nextafter(1.0, 0.0)
    for component_id in (1, 2):
        component = binary.component(component_id)
        surface = component.surface(fill_out_factor=factor)
        g = surface.intrinsic_geometry.component
        levels = _levels(g.field, g._computers)
        assert surface._depth < levels.admissibility_gap
        assert surface.potential == levels.opening_report
        assert surface.intrinsic_geometry._depth == surface._depth
        assert not hasattr(surface.intrinsic_geometry, "_computer_potential")
        assert surface.regime is rocheometry.LevelRegime.CONTACT
        assert surface.fill_out_factor() == surface._depth / levels.admissibility_gap
        assert abs(surface.fill_out_factor() - factor) <= 2 * math.ulp(factor)
        assert component.surface(fill_out_factor=surface.fill_out_factor()) == surface
        with pytest.raises(ValueError, match="above L2"):
            component.surface(potential=surface.potential)
        restored = pickle.loads(pickle.dumps(surface))
        assert restored == surface
        assert hash(restored) == hash(surface)
        assert restored._depth == surface._depth


def test_internal_field_construction_has_no_rounded_reciprocal() -> None:
    root = Path(__file__).parents[1] / "src/rocheometry"
    calls = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "RocheField":
                calls.append((path.relative_to(root).as_posix(), ast.unparse(node)))
    assert calls
    for path, call in calls:
        assert call in (
            "RocheField(q)",
            "RocheField(qi)",
            "RocheField(self.qi)",
            "RocheField(float(q))",
            "RocheField(row.qi)",
        ), (
            path,
            call,
        )
