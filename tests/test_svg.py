import math
import xml.etree.ElementTree as ET
from typing import Literal

import pytest

import rocheometry
from rocheometry import svg
from tests._helpers import single_path


def _path_d(
    path: rocheometry.BezierPath | rocheometry.CompoundBezierPath,
    *,
    precision: int = 12,
) -> str:
    return ET.fromstring(svg.path_element(path, precision=precision)).attrib["d"]


def test_path_element_preserves_native_cubic_segments() -> None:
    path = single_path(rocheometry.BinaryGeometry(q=1.0).component(1).lobe().section())
    data = _path_d(path)

    assert data.startswith("M ")
    assert data.count(" C ") == 52
    assert data.endswith(" Z")
    assert "nan" not in data.lower()
    assert "inf" not in data.lower()


def test_path_element_leaves_open_path_open() -> None:
    segment = rocheometry.CubicBezier(
        p0=(0.0, 0.0),
        p1=(1.0, 0.0),
        p2=(1.0, 1.0),
        p3=(2.0, 1.0),
    )
    path = rocheometry.BezierPath((segment,), closed=False)

    data = _path_d(path)

    assert data.startswith("M 0 0 C ")
    assert not data.endswith(" Z")


def test_document_serializes_binary_geometry_as_valid_svg() -> None:
    binary = rocheometry.BinaryGeometry(
        0.25,
        separation=4.0,
        frame="centre_of_mass",
    )

    text = svg.document(binary, margin=0.0)
    root = ET.fromstring(text)

    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    xmin, ymin, xmax, ymax = binary.lobe_bounds()
    view_box = tuple(float(value) for value in root.attrib["viewBox"].split())
    assert view_box == pytest.approx((xmin, ymin, xmax - xmin, ymax - ymin))

    paths = list(root)
    assert len(paths) == 2
    assert paths[0].attrib["class"] == "rocheometry-component-1"
    assert paths[1].attrib["class"] == "rocheometry-component-2"
    for path in paths:
        assert path.attrib["stroke-width"] == "1"
        assert path.attrib["vector-effect"] == "non-scaling-stroke"
    assert paths[0].attrib["d"] == _path_d(
        single_path(binary.component(1).lobe().section())
    )
    assert paths[1].attrib["d"] == _path_d(
        single_path(binary.component(2).lobe().section())
    )


@pytest.mark.parametrize("margin", [-1.0, math.inf, math.nan])
def test_document_rejects_invalid_margin(margin: float) -> None:
    with pytest.raises(ValueError, match="margin"):
        svg.document(
            rocheometry.BinaryGeometry(1.0),
            margin=margin,
        )


@pytest.mark.parametrize("precision", [0, 18])
def test_svg_rejects_invalid_precision(precision: int) -> None:
    path = single_path(rocheometry.BinaryGeometry(q=1.0).component(1).lobe().section())

    with pytest.raises(ValueError, match="precision"):
        _path_d(path, precision=precision)


def test_svg_rejects_wrong_precision_type() -> None:
    path = single_path(rocheometry.BinaryGeometry(q=1.0).component(1).lobe().section())

    with pytest.raises(TypeError, match="precision: expected int"):
        _path_d(path, precision=True)


def test_document_rejects_bool_margin() -> None:
    with pytest.raises(TypeError, match="margin: expected a real number"):
        svg.document(
            rocheometry.BinaryGeometry(1.0),
            margin=True,
        )


def test_svg_rejects_wrong_path_type() -> None:
    with pytest.raises(TypeError, match="path: expected BezierPath"):
        _path_d(object())  # type: ignore[arg-type]


def test_svg_rejects_wrong_binary_type() -> None:
    with pytest.raises(TypeError, match="binary: expected BinaryGeometry"):
        svg.document(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("count", (0, 1, 2))
def test_compound_path_element_preserves_subpath_commands(count: int) -> None:
    from tests._helpers import compound_paths

    items = compound_paths()[:count]
    compound = rocheometry.CompoundBezierPath(items)
    data = _path_d(compound, precision=7)
    assert data == " ".join(_path_d(p, precision=7) for p in items)
    assert data.count("M ") == count
    assert data.count(" Z") == sum(p.closed for p in items)


def test_empty_compound_still_validates_precision() -> None:
    with pytest.raises(ValueError, match="precision"):
        _path_d(rocheometry.CompoundBezierPath(), precision=0)


def test_path_element_is_unfilled_and_escapes_xml_attributes() -> None:
    path = single_path(rocheometry.BinaryGeometry(1.0).component(1).lobe().section())
    element = ET.fromstring(
        svg.path_element(
            path,
            precision=17,
            class_name='star & "companion"',
            stroke="red & blue",
            linewidth=1.25,
        )
    )
    assert element.tag == "path"
    assert element.attrib["class"] == 'star & "companion"'
    assert element.attrib["d"] == _path_d(path, precision=17)
    assert element.attrib["stroke"] == "red & blue"
    assert element.attrib["fill"] == "none"
    assert element.attrib["stroke-width"] == "1.25"
    assert "vector-effect" not in element.attrib


def test_document_paths_accepts_open_closed_and_empty_compound_paths() -> None:
    first = rocheometry.BezierPath(
        (rocheometry.CubicBezier((1, 2), (2, 3), (3, 3), (4, 2)),),
        closed=False,
    )
    second = rocheometry.BezierPath(
        (rocheometry.CubicBezier((6, -3), (7, -2), (7, -4), (6, -3)),),
        closed=True,
    )
    paths = (
        first,
        rocheometry.CompoundBezierPath((second,)),
        rocheometry.CompoundBezierPath(),
    )
    names = ("first", "second", None)

    axes: tuple[Literal["down", "up"], ...] = ("down", "up")
    for axis in axes:
        text = svg.document_paths(
            paths, margin=0, precision=17, classes=names, y_axis=axis
        )
        root = ET.fromstring(text)
        drawn = list(root)
        assert len(drawn) == 3
        assert [p.attrib.get("class") for p in drawn] == list(names)
        assert [p.attrib["d"] for p in drawn] == [
            _path_d(path, precision=17) for path in paths
        ]
        assert drawn[0].attrib["d"].startswith("M ")
        assert not drawn[0].attrib["d"].endswith(" Z")
        assert drawn[1].attrib["d"].endswith(" Z")
        assert drawn[2].attrib["d"] == ""

        xmin = min(first.bounds[0], second.bounds[0])
        ymin = min(first.bounds[1], second.bounds[1])
        xmax = max(first.bounds[2], second.bounds[2])
        ymax = max(first.bounds[3], second.bounds[3])
        expected_y = ymin if axis == "down" else -ymax
        assert tuple(map(float, root.attrib["viewBox"].split())) == pytest.approx(
            (xmin, expected_y, xmax - xmin, ymax - ymin)
        )
        for drawn_path in drawn:
            assert drawn_path.attrib["stroke-width"] == "1"
            assert drawn_path.attrib["vector-effect"] == "non-scaling-stroke"
            assert drawn_path.attrib.get("transform") == (
                None if axis == "down" else "scale(1,-1)"
            )


@pytest.mark.parametrize("linewidth", (0.0, 2.5))
def test_documents_use_pixel_linewidth_without_changing_bounds(
    linewidth: float,
) -> None:
    binary = rocheometry.BinaryGeometry(1.0)
    path = binary.component(1).lobe().section()
    for default_text, custom_text in (
        (svg.document(binary), svg.document(binary, linewidth=linewidth)),
        (svg.document_paths((path,)), svg.document_paths((path,), linewidth=linewidth)),
    ):
        default = ET.fromstring(default_text)
        custom = ET.fromstring(custom_text)
        assert custom.attrib["viewBox"] == default.attrib["viewBox"]
        for element in custom:
            assert float(element.attrib["stroke-width"]) == linewidth
            assert element.attrib["vector-effect"] == "non-scaling-stroke"


@pytest.mark.parametrize("linewidth", (-1.0, math.inf, -math.inf, math.nan))
def test_documents_reject_invalid_linewidth(linewidth: float) -> None:
    binary = rocheometry.BinaryGeometry(1.0)
    path = binary.component(1).lobe().section()
    with pytest.raises(ValueError, match="linewidth"):
        svg.document(binary, linewidth=linewidth)
    with pytest.raises(ValueError, match="linewidth"):
        svg.document_paths((path,), linewidth=linewidth)


def test_documents_reject_bool_linewidth() -> None:
    binary = rocheometry.BinaryGeometry(1.0)
    path = binary.component(1).lobe().section()
    with pytest.raises(TypeError, match="linewidth: expected a real number"):
        svg.document(binary, linewidth=True)
    with pytest.raises(TypeError, match="linewidth: expected a real number"):
        svg.document_paths((path,), linewidth=True)


def test_document_y_up_flips_display_but_not_cubic_coordinates() -> None:
    binary = rocheometry.BinaryGeometry(0.25, separation=4.0, frame="centre_of_mass")
    down = ET.fromstring(svg.document(binary, precision=17))
    up = ET.fromstring(svg.document(binary, precision=17, y_axis="up"))
    down_box = tuple(map(float, down.attrib["viewBox"].split()))
    up_box = tuple(map(float, up.attrib["viewBox"].split()))
    assert up_box == pytest.approx(
        (down_box[0], -down_box[1] - down_box[3], down_box[2], down_box[3])
    )
    for normal, flipped in zip(down, up, strict=True):
        assert normal.attrib["d"] == flipped.attrib["d"]
        assert normal.attrib.get("transform") is None
        assert flipped.attrib["transform"] == "scale(1,-1)"


@pytest.mark.parametrize("axis", ["left", "", None, 1])
def test_document_paths_rejects_invalid_y_axis(axis: object) -> None:
    path = single_path(rocheometry.BinaryGeometry(1.0).component(1).lobe().section())
    with pytest.raises((TypeError, ValueError), match="y_axis"):
        svg.document_paths((path,), y_axis=axis)  # type: ignore[arg-type]


def test_document_paths_rejects_missing_paths_and_mismatched_classes() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        svg.document_paths((rocheometry.CompoundBezierPath(),))
    path = single_path(rocheometry.BinaryGeometry(1.0).component(1).lobe().section())
    with pytest.raises(ValueError, match="classes"):
        svg.document_paths((path,), classes=("first", "second"))


@pytest.mark.parametrize("width", (-1.0, float("nan"), True))
def test_path_element_rejects_invalid_linewidth(width: object) -> None:
    path = single_path(rocheometry.BinaryGeometry(1.0).component(1).lobe().section())
    with pytest.raises((TypeError, ValueError), match="linewidth"):
        svg.path_element(path, linewidth=width)  # type: ignore[arg-type]
