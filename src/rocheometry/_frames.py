"""Coordinate-frame transforms for normalized Roche geometry."""

from typing import Literal

from rocheometry._geometry import AffineTransform2D, Point3D
from rocheometry._types import ComponentId, Frame
from rocheometry._validation import (
    normalize_mass_ratio,
    normalize_point3d,
    require_component_id,
    require_frame,
)


def component_frame(component_id: ComponentId) -> Literal["component1", "component2"]:
    """Return the normalized local frame for component 1 or 2.

    Args:
        component_id: Component number, either `1` or `2`.
    """
    component_id = require_component_id(component_id)
    return "component1" if component_id == 1 else "component2"


def frame_transform(
    *,
    source: Frame,
    target: Frame,
    q: float | None = None,
) -> AffineTransform2D:
    """Return an orbital-plane transform after validating frame names and q.

    q is required when changing to or from the centre-of-mass frame. Identity
    transforms need no mass ratio; a supplied q must be finite and positive.

    Args:
        source: Source chart: `binary`, `centre_of_mass`, `component1`, or `component2`.
        target: Destination chart using the same unit separation.
        q: System mass ratio `M1/M2`. Must be finite and positive when given;
            required for changes involving `centre_of_mass`, otherwise may be `None`.
    """
    source = require_frame(source, name="source")
    target = require_frame(target, name="target")
    if q is not None:
        q = normalize_mass_ratio(q)
    if source == target:
        return AffineTransform2D.identity()
    return _to_binary_transform(source, q).then(_from_binary_transform(target, q))


def transform(
    position: Point3D,
    *,
    source: Frame,
    target: Frame,
    q: float | None = None,
) -> Point3D:
    """Transform a normalized position between Roche coordinate frames.

    q uses the system convention M1/M2 and is required only when changing to or
    from the centre-of-mass frame. Component frames place the named component
    at the origin and its companion on positive x.

    Orbital-plane coordinates use `frame_transform`; z is unchanged.

    Args:
        position: Three-coordinate tuple of finite real numbers, in separation
            units. Coordinates are converted to floats.
        source: Source chart: `binary`, `centre_of_mass`, `component1`, or `component2`.
        target: Destination chart using the same unit separation.
        q: System mass ratio `M1/M2`. Must be finite and positive when given;
            required for changes involving `centre_of_mass`, otherwise may be `None`.
    """
    x, y, z = normalize_point3d(position, name="position")
    x, y = frame_transform(source=source, target=target, q=q).apply((x, y))
    return (x, y, z)


def transform_vector(
    vector: Point3D,
    *,
    source: Frame,
    target: Frame,
    q: float | None = None,
) -> Point3D:
    """Transform a finite vector between frames, ignoring translations.

    Args:
        vector: Three-coordinate tuple of finite real numbers. Coordinates
            are converted to floats; translations are ignored and z is unchanged.
        source: Source chart: `binary`, `centre_of_mass`, `component1`, or `component2`.
        target: Destination chart using the same unit separation.
        q: Optional finite positive system mass ratio `M1/M2`. Required for
            changes involving `centre_of_mass`, as for position transforms.
    """
    x, y, z = normalize_point3d(vector, name="vector")
    x, y = frame_transform(source=source, target=target, q=q).apply_vector((x, y))
    return (x, y, z)


def _to_binary_transform(
    source: Frame,
    q: float | None,
) -> AffineTransform2D:
    if source in {"binary", "component1"}:
        return AffineTransform2D.identity()
    if source == "component2":
        return AffineTransform2D(xx=-1.0, yy=-1.0, tx=1.0)

    if q is None:
        raise ValueError("q is required for centre-of-mass frame transforms")
    return AffineTransform2D.translation(1.0 / (1.0 + q), 0.0)


def _from_binary_transform(
    target: Frame,
    q: float | None,
) -> AffineTransform2D:
    if target in {"binary", "component1"}:
        return AffineTransform2D.identity()
    if target == "component2":
        return AffineTransform2D(xx=-1.0, yy=-1.0, tx=1.0)

    if q is None:
        raise ValueError("q is required for centre-of-mass frame transforms")
    return AffineTransform2D.translation(-1.0 / (1.0 + q), 0.0)
