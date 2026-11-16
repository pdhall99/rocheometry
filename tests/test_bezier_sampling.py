"""Arc-length sampling accounts for speed, segment lengths, and degeneracy."""

from math import hypot
from typing import cast

import pytest

from rocheometry import AffineTransform2D, BezierPath, CubicBezier
from rocheometry._bezier import SampleSpacing


@pytest.mark.parametrize("scale", [1e-15, 1.0, 1e15])
def test_nonuniform_line_has_uniform_arc_length_samples(scale: float) -> None:
    line = CubicBezier((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (scale, 0.0))
    path = BezierPath((line,), closed=False)
    assert path.sample(5) == path.sample(5, spacing="parameter")
    for index, point in enumerate(path.sample(9, spacing="arc_length")):
        assert point == pytest.approx(
            (scale * index / 8, 0.0), rel=3e-13, abs=scale * 1e-14
        )


def test_arc_length_spacing_crosses_unequal_segments() -> None:
    path = BezierPath(
        (
            CubicBezier((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
            CubicBezier((1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)),
        ),
        closed=False,
    )
    assert path.sample(5)[2] == (1.0, 0.0)
    for index, point in enumerate(path.sample(5, spacing="arc_length")):
        assert point == pytest.approx((index, 0.0), abs=1e-13)


def test_curved_samples_have_equal_independently_integrated_lengths() -> None:
    segment = CubicBezier((0.0, 0.0), (3.0, 5.0), (-2.0, 5.0), (1.0, 0.0))
    path = BezierPath((segment,), closed=False)
    dense = tuple(segment.evaluate(index / 20000) for index in range(20001))
    prefix = [0.0]
    for a, b in zip(dense, dense[1:], strict=False):
        prefix.append(prefix[-1] + hypot(b[0] - a[0], b[1] - a[1]))
    for index, sample in enumerate(path.sample(17, spacing="arc_length")):
        nearest = min(
            range(len(dense)),
            key=lambda j: hypot(sample[0] - dense[j][0], sample[1] - dense[j][1]),
        )
        assert prefix[nearest] / prefix[-1] == pytest.approx(index / 16, abs=7e-5)


def test_arc_length_sampling_skips_degenerate_segments_and_closes_exactly() -> None:
    point = CubicBezier((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    loop = CubicBezier((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0))
    path = BezierPath((point, loop, point))
    samples = path.sample(13, spacing="arc_length")
    assert samples[0] == samples[-1] == (0.0, 0.0)
    assert samples == BezierPath((loop,)).sample(13, spacing="arc_length")
    assert BezierPath((point,)).sample(4, spacing="arc_length") == ((0.0, 0.0),) * 4


def test_arc_length_sampling_handles_stationary_reversal() -> None:
    segment = CubicBezier((0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0))
    samples = BezierPath((segment,)).sample(5, spacing="arc_length")
    for point, expected in zip(samples, (0.0, 0.375, 0.75, 0.375, 0.0), strict=True):
        assert point == pytest.approx((expected, 0.0), abs=1e-11)


def test_arc_length_sampling_commutes_with_uniform_scaling() -> None:
    path = BezierPath(
        (CubicBezier((0.0, 0.0), (3.0, 5.0), (-2.0, 5.0), (1.0, 0.0)),), closed=False
    )
    transform = AffineTransform2D.scale(2.5)
    a = path.transformed(transform).sample(17, spacing="arc_length")
    b = tuple(transform.apply(p) for p in path.sample(17, spacing="arc_length"))
    for first, second in zip(a, b, strict=True):
        assert first == pytest.approx(second, abs=1e-12)


def test_sampling_rejects_unknown_spacing() -> None:
    path = BezierPath((CubicBezier((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)),))
    with pytest.raises(ValueError, match="spacing"):
        path.sample(spacing=cast(SampleSpacing, "distance"))
    with pytest.raises(TypeError, match="spacing"):
        path.sample(spacing=cast(SampleSpacing, None))
