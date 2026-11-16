import xml.etree.ElementTree as ET

import numpy as np
import pytest

import rocheometry
from rocheometry import svg
from tests._helpers import single_path

pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MatplotlibPath  # noqa: E402

from rocheometry.matplotlib import add_binary_lobes, path_patch  # noqa: E402


def test_path_patch_preserves_native_cubic_segments() -> None:
    path = single_path(rocheometry.BinaryGeometry(q=0.25).component(1).lobe().section())
    converted = path_patch(path).get_path()
    codes = converted.codes

    assert codes is not None
    vertices_array = np.asarray(converted.vertices)
    codes_array = np.asarray(codes)
    assert len(vertices_array) == 3 * len(path.segments) + 2
    assert len(codes_array) == 3 * len(path.segments) + 2
    assert codes_array[0] == MatplotlibPath.MOVETO
    assert sum(code == MatplotlibPath.CURVE4 for code in codes_array) == 3 * len(
        path.segments
    )
    assert codes_array[-1] == MatplotlibPath.CLOSEPOLY

    assert tuple(vertices_array[0]) == pytest.approx(path.segments[0].p0)
    assert tuple(vertices_array[1]) == pytest.approx(path.segments[0].p1)
    assert tuple(vertices_array[2]) == pytest.approx(path.segments[0].p2)
    assert tuple(vertices_array[3]) == pytest.approx(path.segments[0].p3)


def test_path_patch_defaults_to_an_outline_like_svg() -> None:
    path = single_path(rocheometry.BinaryGeometry(q=1.0).component(1).lobe().section())

    assert path_patch(path).get_fill() is False
    assert path_patch(path, fill=True).get_fill() is True


def test_path_patch_wraps_native_path_and_forwards_properties() -> None:
    path = single_path(rocheometry.BinaryGeometry(q=1.0).component(1).lobe().section())
    patch = path_patch(path, fill=False, linewidth=1.5)
    codes = patch.get_path().codes

    assert isinstance(patch, PathPatch)
    assert patch.get_fill() is False
    assert patch.get_linewidth() == pytest.approx(1.5)
    assert codes is not None
    codes_array = np.asarray(codes)
    assert sum(code == MatplotlibPath.CURVE4 for code in codes_array) == 3 * len(
        path.segments
    )


def test_matplotlib_adapters_reject_wrong_path_type() -> None:
    with pytest.raises(TypeError, match="path: expected BezierPath"):
        path_patch(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("count", (0, 1, 2))
def test_compound_concatenates_native_paths(count: int) -> None:
    from tests._helpers import compound_paths

    items = compound_paths()[:count]
    converted = path_patch(rocheometry.CompoundBezierPath(items)).get_path()
    individual = tuple(path_patch(p).get_path() for p in items)
    if individual:
        np.testing.assert_array_equal(
            converted.vertices, np.concatenate([p.vertices for p in individual])
        )
        np.testing.assert_array_equal(
            converted.codes, np.concatenate([np.asarray(p.codes) for p in individual])
        )
    else:
        assert np.asarray(converted.vertices).shape == (0, 2)
    patch = path_patch(rocheometry.CompoundBezierPath(items), fill=False, linewidth=2)
    assert isinstance(patch, PathPatch)
    assert not patch.get_fill()
    assert patch.get_linewidth() == 2
    np.testing.assert_array_equal(patch.get_path().vertices, converted.vertices)
    np.testing.assert_array_equal(patch.get_path().codes, converted.codes)


@pytest.mark.parametrize("count", (0, 1, 2))
def test_svg_and_matplotlib_have_identical_cubic_vertices(count: int) -> None:
    from tests._helpers import compound_paths

    path = rocheometry.CompoundBezierPath(compound_paths()[:count])
    svg_patch = ET.fromstring(svg.path_element(path, precision=17))
    tokens = svg_patch.attrib["d"].split()
    coords = np.asarray(
        [float(token) for token in tokens if token not in ("M", "C", "Z")]
    ).reshape(-1, 2)
    converted = path_patch(path, fill=False).get_path()
    vertices = np.asarray(converted.vertices)
    if converted.codes is None:
        assert tokens == []
        assert vertices.shape == (0, 2)
        return
    codes = np.asarray(converted.codes)
    mpl_coords = vertices[codes != MatplotlibPath.CLOSEPOLY]

    np.testing.assert_array_equal(coords, mpl_coords)
    assert tokens.count("M") == int(sum(codes == MatplotlibPath.MOVETO))
    assert tokens.count("Z") == int(sum(codes == MatplotlibPath.CLOSEPOLY))


def test_add_binary_lobes_matches_svg_bounds_and_paths() -> None:
    binary = rocheometry.BinaryGeometry(0.25, separation=4.0, frame="centre_of_mass")
    figure, ax = plt.subplots()
    try:
        patches = add_binary_lobes(ax, binary, margin=0.1, linewidth=2)
        assert len(patches) == 2
        assert all(patch in ax.patches for patch in patches)
        assert all(not patch.get_fill() for patch in patches)
        assert all(patch.get_linewidth() == 2 for patch in patches)
        assert ax.get_aspect() == 1.0

        view = svg.document(binary, margin=0.1, precision=17, y_axis="up")
        element = ET.fromstring(view)
        x0, y0, width, height = map(float, element.attrib["viewBox"].split())
        assert ax.get_xlim() == pytest.approx((x0, x0 + width))
        assert ax.get_ylim() == pytest.approx((-y0 - height, -y0))

        for component, patch in zip(
            (binary.component(1), binary.component(2)), patches, strict=True
        ):
            reference = component.lobe().section()
            converted = path_patch(reference).get_path()
            np.testing.assert_array_equal(patch.get_path().vertices, converted.vertices)
            np.testing.assert_array_equal(patch.get_path().codes, converted.codes)
    finally:
        plt.close(figure)


def test_add_binary_lobes_validates_axes_binary_and_margin() -> None:
    binary = rocheometry.BinaryGeometry(1.0)
    fig, ax = plt.subplots()
    try:
        with pytest.raises(TypeError, match="ax"):
            add_binary_lobes(object(), binary)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="binary"):
            add_binary_lobes(ax, object())  # type: ignore[arg-type]
        for margin in (-1.0, float("inf"), True):
            with pytest.raises((TypeError, ValueError), match="margin"):
                add_binary_lobes(ax, binary, margin=margin)  # type: ignore[arg-type]
        assert len(ax.patches) == 0
    finally:
        plt.close(fig)
