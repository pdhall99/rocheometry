"""Immutable directional samples of a component-enclosing surface."""

from dataclasses import dataclass, field

from rocheometry._geometry import Point3D
from rocheometry._validation import _normalize_finite_positive_float


@dataclass(frozen=True, slots=True)
class SurfaceSample:
    """Surface radii paired with weighted unit quadrature directions.

    Rays cover one quadrant, `y >= 0, z >= 0`, with weights multiplied by four
    to represent the full sphere. Readers may integrate only quantities
    symmetric under both `y -> -y` and `z -> -z`. Construction checks matching lengths and finite positive radii before
    a sample can be shared.

    Attributes:
        rays: Unit directions and solid-angle weights. Detached samples cover
            the full solid angle; contact samples cover the envelope share.
        part: Whether rays cover only the envelope share of a contact part.
        radii: Surface radii in separation units, in the same order as `rays`.
    """

    rays: tuple[tuple[Point3D, float], ...]
    radii: tuple[float, ...]
    part: bool = False

    def __post_init__(self) -> None:
        if len(self.rays) != len(self.radii):
            raise ValueError("sample: rays and radii must have equal lengths")
        for radius in self.radii:
            _normalize_finite_positive_float(radius, name="sample radius")


@dataclass(frozen=True, slots=True, kw_only=True)
class _NeckSample(SurfaceSample):
    """Retain the origin/depth needed to reconstruct unrounded neck geometry."""

    origin: float
    depth: float

    _gradients: dict[Point3D, Point3D] = field(
        default_factory=dict, repr=False, compare=False, hash=False
    )
