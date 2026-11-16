"""SVG serialization for Roche geometry paths."""

from collections.abc import Iterable
from typing import Literal
from xml.sax.saxutils import quoteattr

from rocheometry._bezier import BezierPath, CompoundBezierPath
from rocheometry._binary_geometry import BinaryGeometry
from rocheometry._geometry import Point2D
from rocheometry._rendering import binary_lobe_paths, padded_bounds
from rocheometry._validation import (
    normalize_margin,
    require_choice,
    require_instance,
    require_precision,
)

__all__ = ["document", "document_paths", "path_element"]


def path_element(
    path: BezierPath | CompoundBezierPath,
    *,
    precision: int = 12,
    class_name: str | None = None,
    fill: str = "none",
    stroke: str = "currentColor",
    linewidth: float | None = None,
    transform: str | None = None,
    _non_scaling_stroke: bool = False,
) -> str:
    """Return a styled SVG path element for a Rocheometry Bézier path.

    The SVG element counterpart of matplotlib.path_patch(); both take
    a Rocheometry Bézier path and return a native drawing element.

    The default is an unfilled `currentColor` outline, matching document().
    Attribute values are XML-escaped. A transform, when supplied, is emitted
    as an SVG attribute; the underlying cubic coordinates remain unchanged.

    Args:
        path: Bézier path or compound to serialize.
        precision: Significant digits per coordinate, from 1 to 17.
        class_name: Optional CSS class.
        fill: SVG fill value, default "none".
        stroke: SVG stroke value, default "currentColor".
        linewidth: Optional non-negative stroke width in SVG coordinate units.
        transform: Optional SVG transform attribute.
    """
    data = _path_data(path, precision=precision)
    require_instance(fill, str, name="fill")
    require_instance(stroke, str, name="stroke")
    attributes = []
    if class_name is not None:
        require_instance(class_name, str, name="class_name")
        attributes.append(f"class={quoteattr(class_name)}")
    attributes.extend(
        (
            f"d={quoteattr(data)}",
            f"fill={quoteattr(fill)}",
            f"stroke={quoteattr(stroke)}",
        )
    )
    if linewidth is not None:
        width = normalize_margin(linewidth, name="linewidth")
        attributes.append(f"stroke-width={quoteattr(_format_number(width, precision))}")
    if _non_scaling_stroke:
        attributes.append('vector-effect="non-scaling-stroke"')
    if transform is not None:
        require_instance(transform, str, name="transform")
        attributes.append(f"transform={quoteattr(transform)}")
    return f"<path {' '.join(attributes)}/>"


def document(
    binary: BinaryGeometry,
    *,
    margin: float = 0.05,
    linewidth: float = 1.0,
    precision: int = 12,
    y_axis: Literal["down", "up"] = "down",
) -> str:
    """Return a standalone SVG with both critical-lobe orbital sections.

    The binary's scientific frame and separation are retained. By default,
    SVG's downward-positive display y-axis is not hidden by a transform.
    Choose y_axis="up" to display positive scientific y upwards, as Matplotlib
    does; this flips the SVG display, not the path coordinates.

    Stroke width is independent of the drawing's scale. Bounds do not include
    the stroke; with margin=0, half the stroke can be clipped.

    Args:
        binary: Binary whose two critical-lobe sections define the drawing.
        margin: Padding as a fraction of the larger lobe-bound dimension.
            Must be finite and non-negative.
        linewidth: Stroke width in screen pixels. Must be finite and non-negative.
        precision: Significant digits per coordinate, from 1 to 17.
        y_axis: Display positive y down (default) or up.

    Raises:
        RuntimeError: The computed drawing bounds have zero size.
    """
    binary = require_instance(binary, BinaryGeometry, name="binary")
    margin = normalize_margin(margin)
    linewidth = normalize_margin(linewidth, name="linewidth")
    precision = require_precision(precision)
    return document_paths(
        binary_lobe_paths(binary),
        margin=margin,
        linewidth=linewidth,
        precision=precision,
        classes=("rocheometry-component-1", "rocheometry-component-2"),
        y_axis=y_axis,
    )


def document_paths(
    paths: Iterable[BezierPath | CompoundBezierPath],
    *,
    margin: float = 0.05,
    linewidth: float = 1.0,
    precision: int = 12,
    classes: Iterable[str | None] | None = None,
    y_axis: Literal["down", "up"] = "down",
) -> str:
    """Render any ordered sequence of Bézier paths in a standalone SVG.

    All paths share the default unfilled `currentColor` outline styling;
    optional CSS classes allow callers to style them independently. Empty
    compound paths are permitted but do not contribute to the bounds. At
    least one nonempty path is needed to define a viewBox.

    The default downward-positive display orientation preserves the raw
    path coordinates. y_axis="up" flips display y using SVG transforms, so
    the resulting view has the same orientation as Matplotlib.

    Stroke width is independent of the drawing's scale. Bounds do not include
    the stroke; with margin=0, half the stroke can be clipped.

    Args:
        paths: Paths to draw, in stacking order (first at the back).
        margin: Padding as a fraction of the larger combined dimension.
        linewidth: Stroke width in screen pixels. Must be finite and non-negative.
        precision: Significant digits per coordinate, from 1 to 17.
        classes: Optional CSS class per path, or None for no classes.
        y_axis: Display positive scientific y down (default) or up.
    """
    margin = normalize_margin(margin)
    linewidth = normalize_margin(linewidth, name="linewidth")
    precision = require_precision(precision)
    require_instance(y_axis, str, name="y_axis")
    require_choice(y_axis, ("down", "up"), name="y_axis")
    paths = tuple(paths)
    if classes is None:
        class_names: tuple[str | None, ...] = (None,) * len(paths)
    else:
        class_names = tuple(classes)
        if len(class_names) != len(paths):
            raise ValueError("classes: expected one class name per path")
    for index, class_name in enumerate(class_names):
        if class_name is not None:
            require_instance(class_name, str, name=f"classes[{index}]")

    xmin, ymin, xmax, ymax = padded_bounds(paths, margin=margin)
    view_y = ymin if y_axis == "down" else -ymax
    view_box = " ".join(
        _format_number(value, precision)
        for value in (xmin, view_y, xmax - xmin, ymax - ymin)
    )
    flip = "scale(1,-1)" if y_axis == "up" else None
    elements = "".join(
        path_element(
            path,
            precision=precision,
            class_name=class_name,
            linewidth=linewidth,
            transform=flip,
            _non_scaling_stroke=True,
        )
        for path, class_name in zip(paths, class_names, strict=True)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}">{elements}</svg>'
    )


def _path_data(path: BezierPath | CompoundBezierPath, *, precision: int = 12) -> str:
    """Return SVG path-data text using native cubic `C` commands.

    Args:
        path: Bézier path or compound to serialize, in its existing coordinate frame.
        precision: Significant digits per coordinate. Must be an integer
            in `[1, 17]`; booleans are rejected.

    Returns:
        An SVG `d` attribute with a move and cubic commands for each path,
        and a closing `Z` for each closed path. Subpaths are joined by one
        space; an empty compound returns an empty string. No transform is applied.
    """
    path = require_instance(path, (BezierPath, CompoundBezierPath), name="path")
    precision = require_precision(precision)
    if isinstance(path, CompoundBezierPath):
        return " ".join(_path_data(part, precision=precision) for part in path)
    start = path.segments[0].p0
    commands = [f"M {_format_point(start, precision)}"]
    for segment in path.segments:
        commands.append(
            "C "
            f"{_format_point(segment.p1, precision)} "
            f"{_format_point(segment.p2, precision)} "
            f"{_format_point(segment.p3, precision)}"
        )
    if path.closed:
        commands.append("Z")
    return " ".join(commands)


def _format_point(point: Point2D, precision: int) -> str:
    return (
        f"{_format_number(point[0], precision)} {_format_number(point[1], precision)}"
    )


def _format_number(value: float, precision: int) -> str:
    if value == 0.0:
        value = 0.0
    return format(value, f".{precision}g")
