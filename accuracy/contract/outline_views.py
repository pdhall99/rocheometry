"""Fitted silhouette views measured against Reference at the same depth."""

from time import perf_counter
from typing import Any
from unittest.mock import patch

from crosschecks.distance.bezier import _PathDistance
from crosschecks.distance.curves import sample, two_way_distance

import rocheometry
from accuracy.harness.common import select, silhouette_segments
from rocheometry._computers.silhouette import limb


def upper_half(path: rocheometry.CompoundBezierPath) -> rocheometry.CompoundBezierPath:
    """Reduce only after checking every lower control is its exact reflection."""
    pieces = []
    for part in path.paths:
        n = len(part.segments) // 2
        if 2 * n != len(part.segments):
            return path
        for first, second in zip(
            part.segments[:n], reversed(part.segments[n:]), strict=True
        ):
            a = (first.p0, first.p1, first.p2, first.p3)
            b = (second.p3, second.p2, second.p1, second.p0)
            if any(p != (q[0], -q[1]) for p, q in zip(a, b, strict=True)):
                return path
        pieces.append(rocheometry.BezierPath(part.segments[:n], closed=False))
    return rocheometry.CompoundBezierPath(tuple(pieces))


def evaluate(case: dict[str, Any]) -> dict[str, Any]:
    started = perf_counter()
    surface = select(
        rocheometry.ComponentGeometry(case["qi"]), case["kind"], case["level"]
    )
    ref = rocheometry.ComponentGeometry(
        case["qi"], accuracy=rocheometry.Reference()
    )._surface_at_depth(surface._depth)
    plane = rocheometry.CentrelinePlane(case["inclination"])
    junctions = {}
    original = limb._junction

    def capture(f: Any, *args: Any, **kwargs: Any) -> limb._Junction:
        junction = original(f, *args, **kwargs)
        junctions[junction.angle] = junction
        return junction

    with patch.object(limb, "_junction", capture):
        truth = upper_half(ref.silhouette_envelope(plane))
    path = surface.silhouette_envelope(plane)
    assert len(path.paths[0].segments) == 2 * silhouette_segments(surface.component)
    rv = ref.volume_radius()
    distances = two_way_distance(
        upper_half(path),
        truth,
        count=101,
        corners=[j.point for j in junctions.values()],
    )
    probes = sample(upper_half(path), 101)
    index = min(
        range(len(probes)),
        key=lambda j: sum(
            (a - b) ** 2 for a, b in zip(probes[j], distances["forward_point"])
        ),
    )
    corners = [
        dict(
            angle=j.angle,
            point=j.point,
            error=min(_PathDistance(p)(j.point) for p in path.paths) / rv,
        )
        for j in junctions.values()
    ]
    return dict(
        case=case,
        error=distances["value"] / rv,
        forward_error=distances["forward"] / rv,
        reverse_error=distances["reverse"] / rv,
        worst_point=distances["forward_point"],
        worst_segment=index // 101,
        corner_error=max((c["error"] for c in corners), default=0.0),
        corners=corners,
        certified_checks=distances["checks"],
        seconds=perf_counter() - started,
    )
