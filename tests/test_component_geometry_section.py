import math

import pytest

import rocheometry
from tests._helpers import single_path


def test_section_returns_closed_fifty_two_segment_path() -> None:
    path = single_path(
        rocheometry.ComponentGeometry(0.37).surface(volume_radius_fill=0.8).section()
    )

    assert isinstance(path, rocheometry.BezierPath)
    assert path.closed
    assert len(path.segments) == 52
    assert path.segments[-1].p3 == path.segments[0].p0


def test_contact_section_starts_with_a_degenerate_segment() -> None:
    path = single_path(rocheometry.ComponentGeometry(1.0).lobe().section())
    first = path.segments[0]
    assert first.p0 == first.p1 == first.p2 == first.p3


@pytest.mark.parametrize("q", [1.0e-5, 1.0e5])
def test_section_traces_outside_model_domain(q: float) -> None:
    geometry = rocheometry.ComponentGeometry(q)
    path = single_path(geometry.surface(volume_radius_fill=0.8).section())
    assert path.closed


@pytest.mark.parametrize("volume_radius_fill", [0.0, -0.1, math.inf, math.nan])
def test_section_rejects_invalid_volume_radius_fill(volume_radius_fill: float) -> None:
    with pytest.raises(ValueError, match="volume_radius_fill"):
        single_path(
            rocheometry.ComponentGeometry(1.0)
            .surface(volume_radius_fill=volume_radius_fill)
            .section()
        )
