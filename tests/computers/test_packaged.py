import subprocess
import sys

import rocheometry
import rocheometry._computers
from rocheometry._computers.packaged import (
    load_packaged_lobe_volume_radius_computer,
    packaged_lobe_volume_computer,
    packaged_lobe_volume_radius_computer,
)
from rocheometry._roche.volume import sphere_volume


def test_package_imports() -> None:
    assert rocheometry.__doc__ is not None


def test_public_names_reserve_map_level_vocabulary() -> None:
    for module in (rocheometry, rocheometry._computers):
        assert not [
            name
            for name in dir(module)
            if not name.startswith("_") and "contour" in name.lower()  # Map-level term.
        ]


def test_component_geometry_construction_loads_no_fitted_model() -> None:
    code = """
import sys
import rocheometry
from rocheometry._computers.packaged import packaged_lobe_volume_computer, packaged_lobe_volume_radius_computer

packaged_lobe_volume_computer()
packaged_lobe_volume_radius_computer()
rocheometry.ComponentGeometry(1.0)
rocheometry.Reference()._computers()


for name in (
    "rocheometry._computers.section_path.fitted",
    "rocheometry._computers.lobe_volume_radius.fitted",
):
    if name in sys.modules:
        raise SystemExit("geometry construction imported fitted model: " + name)
"""

    subprocess.run([sys.executable, "-c", code], check=True)


def test_traced_path_does_not_import_geometry_reference_helpers() -> None:
    code = """
import sys
import rocheometry
from rocheometry._computers.section_path.traced import TracedSectionPathComputer

computers = rocheometry.Reference(polar_order=8, azimuth_order=6)._computers()
computer = TracedSectionPathComputer(radius_computer=computers.radius, l1_computer=computers.l1)
computer(rocheometry.ComponentGeometry(0.3).field, computers.selection.depth(rocheometry.ComponentGeometry(0.3).field, 0.8), rocheometry.ORBITAL_PLANE).paths[0].sample(9)
assert "rocheometry._computers.section_path.reference" not in sys.modules

from rocheometry._computers.section_path import SectionReference, reference_component_section
from rocheometry._computers.section_path.reference import SectionReference as Reference
assert SectionReference is Reference
assert callable(reference_component_section)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_packaged_models_load_once_on_first_use() -> None:
    code = """
from unittest.mock import patch
import rocheometry
import rocheometry._computers.packaged as packaged

with patch.object(packaged, "load_packaged_lobe_volume_radius_computer",
                  wraps=packaged.load_packaged_lobe_volume_radius_computer) as radius_loader:
    radius = packaged.packaged_lobe_volume_radius_computer()
    volume = packaged.packaged_lobe_volume_computer()
    geometry = rocheometry.ComponentGeometry(1.0)

    assert radius_loader.call_count == 0
    volume(geometry.field)
    radius(geometry.field)
    assert radius_loader.call_count == 1

    geometry.surface(volume_radius_fill=0.8).section()
    other = rocheometry.ComponentGeometry(0.4).surface(volume_radius_fill=0.9)
    other.volume_radius()
    other.section()
    assert radius_loader.call_count == 1
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_packaged_adapters_match_loaded_models() -> None:
    radius = packaged_lobe_volume_radius_computer()
    volume = packaged_lobe_volume_computer()
    fitted_radius = load_packaged_lobe_volume_radius_computer()

    for qi in (0.25, 1.0, 4.0):
        geometry = rocheometry.ComponentGeometry(qi)
        assert radius(geometry.field) == fitted_radius(geometry.field)
        assert volume(geometry.field) == sphere_volume(fitted_radius(geometry.field))


def test_binary_construction_loads_no_fitted_model() -> None:
    code = """
import sys
import rocheometry

rocheometry.BinaryGeometry(
    1.0,
    separation=2.0,
    frame="centre_of_mass",
)

for name in (
    "rocheometry._computers.section_path.fitted",
    "rocheometry._computers.lobe_volume_radius.fitted",
):
    if name in sys.modules:
        raise SystemExit("binary construction imported fitted model: " + name)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_runtime_section_does_not_import_fitting_support_package() -> None:
    code = """
import sys
import rocheometry

rocheometry.ComponentGeometry(1.0).surface(volume_radius_fill=0.9).section()
forbidden = (
    "rocheometry._computers.section_path.fitted",
    "rocheometry._computers.section_path.fitted.config",
    "rocheometry._computers.section_path.fitted.training",
    "rocheometry._computers.section_path.fitted.fit",
    "rocheometry._computers.section_path.fitted.bezier",
    "rocheometry._computers.section_path.fitted.release",
    "rocheometry._computers.lobe_volume_radius.fitted.config",
    "rocheometry._computers.lobe_volume_radius.fitted.fit",
    "rocheometry._computers.lobe_volume_radius.fitted.provenance",
    "numpy",
    "scipy",
    "yaml",
)
loaded = sorted(name for name in forbidden if name in sys.modules)
if loaded:
    raise SystemExit("runtime imported fitting modules: " + ", ".join(loaded))
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_surface_potential_map_loads_once_and_reference_graph_stays_unfitted() -> None:
    code = """
from unittest.mock import patch
import rocheometry
import rocheometry._computers.packaged as packaged

with patch.object(packaged, "load_packaged_surface_potential_map",
                  wraps=packaged.load_packaged_surface_potential_map) as loader:
    graph = rocheometry.Fitted()._computers()
    geometry = rocheometry.ComponentGeometry(0.3)
    assert loader.call_count == 0
    reference = rocheometry.Reference(polar_order=8, azimuth_order=6)._computers()
    reference.selection.depth(reference_field := geometry.field, 0.8)
    reference.selection.fill(reference_field, 0.0)
    graph.selection.depth(rocheometry.ComponentGeometry(1e9).field, 0.8)
    assert loader.call_count == 0
    surface = geometry.surface(volume_radius_fill=0.8)
    surface.volume_radius_fill()
    surface.section()
    rocheometry.ComponentGeometry(2.0).surface(volume_radius_fill=0.6).section()
    assert loader.call_count == 1
    for suffix in ("config", "fit", "provenance"):
        import sys
        assert "rocheometry._computers.surface_potential.fitted." + suffix not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
