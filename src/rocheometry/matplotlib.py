"""Optional Matplotlib adapters for Rocheometry Bézier paths.

Importing this module requires the `plot` extra; the core package does not
import Matplotlib.
"""

from typing import Any

from matplotlib.axes import Axes
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MatplotlibPath

from rocheometry._bezier import BezierPath, CompoundBezierPath
from rocheometry._binary_geometry import BinaryGeometry
from rocheometry._rendering import binary_lobe_paths, padded_bounds
from rocheometry._validation import require_instance

__all__ = ["add_binary_lobes", "path_patch"]


def path_patch(path: BezierPath | CompoundBezierPath, **kwargs: Any) -> PathPatch:
    """Create a Matplotlib `PathPatch` for a Rocheometry Bézier path.

    The Matplotlib patch counterpart of svg.path_element(); both take
    a Rocheometry Bézier path and return a native drawing element, and both
    default to an unfilled outline.

    Args:
        path: Planar Bézier path or compound to convert.
        **kwargs: Patch options forwarded to `matplotlib.patches.PathPatch`,
            such as edge colour, line width, or fill=True with a face colour.
    """
    return PathPatch(_mpl_path(path), **{"fill": False, **kwargs})


def _mpl_path(path: BezierPath | CompoundBezierPath) -> MatplotlibPath:
    """Build a native Matplotlib Path for the public patch adapter.

    Cubic segments remain `CURVE4` commands, with a `MOVETO` per path and
    `CLOSEPOLY` for each closed path. An empty compound has zero vertices.

    Args:
        path: Planar Bézier path or compound to convert.
    """
    path = require_instance(path, (BezierPath, CompoundBezierPath), name="path")
    if isinstance(path, CompoundBezierPath):
        return MatplotlibPath.make_compound_path(*(_mpl_path(part) for part in path))
    vertices = [path.segments[0].p0]
    codes = [MatplotlibPath.MOVETO]

    for segment in path.segments:
        vertices.extend((segment.p1, segment.p2, segment.p3))
        codes.extend((MatplotlibPath.CURVE4,) * 3)

    if path.closed:
        vertices.append(path.segments[0].p0)
        codes.append(MatplotlibPath.CLOSEPOLY)

    return MatplotlibPath(vertices, codes)


def add_binary_lobes(
    ax: Axes,
    binary: BinaryGeometry,
    *,
    margin: float = 0.05,
    **kwargs: Any,
) -> tuple[PathPatch, PathPatch]:
    """Add both critical-lobe sections to axes and frame them with equal aspect.

    Uses the same Bézier curves and padded bounds as SVG document(binary).
    The axes keep their usual upward-positive scientific y direction.
    Existing artists remain present; the axes limits and aspect are updated.

    Args:
        ax: Existing Matplotlib axes to draw on.
        binary: Binary whose two critical-lobe sections are drawn.
        margin: Finite non-negative padding as a fraction of the larger
            combined lobe dimension (default 0.05).
        **kwargs: PathPatch properties applied to both components.
            Paths default to unfilled outlines; pass fill=True to override.

    Returns:
        Two PathPatch objects in component order, both added to ax.
    """
    ax = require_instance(ax, Axes, name="ax")
    paths = binary_lobe_paths(binary)
    xmin, ymin, xmax, ymax = padded_bounds(paths, margin=margin)
    patches = (path_patch(paths[0], **kwargs), path_patch(paths[1], **kwargs))
    for patch in patches:
        ax.add_patch(patch)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    return patches
