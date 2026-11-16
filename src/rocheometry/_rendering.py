"""Shared, backend-independent drawing composition and bounds."""

from collections.abc import Iterable

from rocheometry._bezier import BezierPath, CompoundBezierPath
from rocheometry._binary_geometry import BinaryGeometry
from rocheometry._geometry import Bounds2D
from rocheometry._validation import normalize_margin, require_instance

Path = BezierPath | CompoundBezierPath


def binary_lobe_paths(
    binary: BinaryGeometry,
) -> tuple[CompoundBezierPath, CompoundBezierPath]:
    """Return the two critical orbital-plane sections in the binary's output frame."""
    binary = require_instance(binary, BinaryGeometry, name="binary")
    return (
        binary.component(1).lobe().section(),
        binary.component(2).lobe().section(),
    )


def padded_bounds(paths: Iterable[Path], *, margin: float = 0.05) -> Bounds2D:
    """Return exact cubic-path bounds expanded by a fraction of the larger span.

    Empty compound paths do not contribute to bounds. At least one nonempty
    path is required. Coordinates are not transformed or resampled.
    """
    margin = normalize_margin(margin)
    bounds: list[Bounds2D] = []
    for index, path in enumerate(paths):
        path = require_instance(
            path, (BezierPath, CompoundBezierPath), name=f"paths[{index}]"
        )
        if isinstance(path, CompoundBezierPath) and not path:
            continue
        bounds.append(path.bounds)

    if not bounds:
        raise ValueError("paths: expected at least one nonempty path")

    xmin = min(b[0] for b in bounds)
    ymin = min(b[1] for b in bounds)
    xmax = max(b[2] for b in bounds)
    ymax = max(b[3] for b in bounds)
    padding = margin * max(xmax - xmin, ymax - ymin)
    x0, y0, x1, y1 = (
        xmin - padding,
        ymin - padding,
        xmax + padding,
        ymax + padding,
    )
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("drawing bounds have zero size")
    return (x0, y0, x1, y1)
