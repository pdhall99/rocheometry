"""Small immutable geometry primitives."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rocheometry._validation import _normalize_finite_float, require_finite

Point2D = tuple[float, float]
"""A planar `(x, y)` coordinate pair."""
Point3D = tuple[float, float, float]

#: A queried point or direction: any three numbers, such as a tuple, a list or a
#: NumPy array.
PointLike = Iterable[float]
"""A spatial `(x, y, z)` coordinate triple."""
Bounds2D = tuple[float, float, float, float]
"""Axis-aligned bounds ordered `(xmin, ymin, xmax, ymax)`."""


@dataclass(frozen=True, slots=True)
class AffineTransform2D:
    """An immutable planar affine map with finite matrix and translation entries.

    The map sends `(x, y)` to `(xx*x + xy*y + tx, yx*x + yy*y + ty)`.
    Real numeric entries are converted to floats; singular maps are allowed.
    Translations use the same units as input positions.

    Attributes:
        xx: Coefficient of input x in output x.
        xy: Coefficient of input y in output x.
        yx: Coefficient of input x in output y.
        yy: Coefficient of input y in output y.
        tx: Translation along output x.
        ty: Translation along output y.
    """

    xx: float = 1.0
    xy: float = 0.0
    yx: float = 0.0
    yy: float = 1.0
    tx: float = 0.0
    ty: float = 0.0

    def __post_init__(self) -> None:
        for name in ("xx", "xy", "yx", "yy", "tx", "ty"):
            object.__setattr__(
                self,
                name,
                _normalize_finite_float(getattr(self, name), name=name),
            )

    @classmethod
    def identity(cls) -> AffineTransform2D:
        """Return the identity transform."""
        return cls()

    @classmethod
    def translation(cls, x: float, y: float) -> AffineTransform2D:
        """Return a translation transform.

        Args:
            x: Finite displacement along x, in input position units.
            y: Finite displacement along y, in input position units.
        """
        return cls(tx=x, ty=y)

    @classmethod
    def scale(
        cls,
        x: float,
        y: float | None = None,
    ) -> AffineTransform2D:
        """Return an axis-aligned scale transform.

        Args:
            x: Finite scale factor along x; negative values reflect the axis.
            y: Finite scale factor along y, or `None` to use `x` on both axes.
        """
        y = x if y is None else y
        return cls(xx=x, yy=y)

    def apply(self, point: Point2D) -> Point2D:
        """Transform one finite point supplied by the caller.

        Args:
            point: Position with two finite coordinates, in the transform's units.
        """
        x, y = point
        require_finite(x, name="point[0]")
        require_finite(y, name="point[1]")
        return (
            self.xx * x + self.xy * y + self.tx,
            self.yx * x + self.yy * y + self.ty,
        )

    def apply_vector(self, vector: Point2D) -> Point2D:
        """Transform one finite vector, ignoring translation.

        Args:
            vector: Vector with two finite coordinates.
        """
        x, y = vector
        require_finite(x, name="vector[0]")
        require_finite(y, name="vector[1]")
        return (
            self.xx * x + self.xy * y,
            self.yx * x + self.yy * y,
        )

    def then(self, other: AffineTransform2D) -> AffineTransform2D:
        """Return the transform that applies this transform, then the other.

        Args:
            other: Map to apply after this map, so the result is `other(self(point))`.
        """
        return AffineTransform2D(
            xx=other.xx * self.xx + other.xy * self.yx,
            xy=other.xx * self.xy + other.xy * self.yy,
            yx=other.yx * self.xx + other.yy * self.yx,
            yy=other.yx * self.xy + other.yy * self.yy,
            tx=other.xx * self.tx + other.xy * self.ty + other.tx,
            ty=other.yx * self.tx + other.yy * self.ty + other.ty,
        )

    def inverse(self) -> AffineTransform2D:
        """Return the inverse transform.

        Raises:
            ValueError: The map is singular and has no inverse, or its inverse
                cannot be represented with finite entries.
        """
        determinant = self.xx * self.yy - self.xy * self.yx
        if determinant == 0.0:
            raise ValueError("singular affine transform has no inverse")

        inverse_xx = self.yy / determinant
        inverse_xy = -self.xy / determinant
        inverse_yx = -self.yx / determinant
        inverse_yy = self.xx / determinant
        return AffineTransform2D(
            xx=inverse_xx,
            xy=inverse_xy,
            yx=inverse_yx,
            yy=inverse_yy,
            tx=-(inverse_xx * self.tx + inverse_xy * self.ty),
            ty=-(inverse_yx * self.tx + inverse_yy * self.ty),
        )
