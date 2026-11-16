"""Pickles hold inputs and settings, and rebuild from them."""

import pickle
import warnings

import pytest

import rocheometry


def _values() -> dict[str, object]:
    geometry = rocheometry.ComponentGeometry(
        0.3, accuracy=rocheometry.Reference(polar_order=16, azimuth_order=8)
    )
    binary = rocheometry.BinaryGeometry(
        0.3, separation=2.0, frame="centre_of_mass", accuracy=geometry.accuracy
    )
    return dict(
        component=geometry,
        binary=binary,
        surface=geometry._surface_at_depth(1e-20),
        view=binary.component(2),
        view_surface=rocheometry.BinaryComponentSurfaceView._from_depth(
            binary.component(2), 1e-20
        ),
    )


@pytest.mark.parametrize(
    "kind", ["component", "binary", "surface", "view", "view_surface"]
)
@pytest.mark.parametrize("protocol", [2, pickle.HIGHEST_PROTOCOL])
def test_pickle_round_trips_silently(kind: str, protocol: int) -> None:
    value = _values()[kind]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        payload = pickle.dumps(value, protocol=protocol)
        restored = pickle.loads(payload)
    assert restored == value
    assert not caught
