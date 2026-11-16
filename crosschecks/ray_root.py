"""Independent 100-digit ray-root gradients for checking quadrature nodes.

The rational axis offset and factored differences prevent cancellation of an
absolute level near 1.5. No normal, gradient or radius producer supplies the
answer; the sampled radius is only a Newton starting point.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
from math import ulp

import rocheometry
from crosschecks.problem import Level, axis_potential, masses
from rocheometry._geometry import Point3D


class RayRoot:
    def __init__(self, surface: rocheometry.ComponentSurfaceGeometry) -> None:
        level = Level.of(surface)
        self.depth = level.depth
        self.x1 = level.l1_x
        self.outer = surface.component.outer().position[0]
        self.m, self.c = masses(level.binary, Decimal)
        self.own_depth = axis_potential(level.binary, self.x1) - axis_potential(
            level.binary, self.outer
        )

    def normal(self, direction: Point3D, radius: float) -> tuple[float, ...]:
        return self.geometry(direction, radius)[1]

    def geometry(
        self, direction: Point3D, radius: float
    ) -> tuple[float, tuple[float, ...], float]:
        with localcontext() as context:
            context.prec = 100
            m, c = self.m, self.c
            total = m + c
            dx, dy, dz = map(Decimal.from_float, direction)
            transverse = dy * dy + dz * dz
            norm = (dx * dx + transverse).sqrt()
            anchor = self.x1
            offset = Fraction(self.depth)
            if dx < 0 and abs(offset - self.own_depth) < abs(offset):
                anchor = self.outer
                offset -= self.own_depth
            a = Decimal.from_float(anchor)
            offset = Decimal(offset.numerator) / Decimal(offset.denominator)
            ac, bc = abs(a), abs(1 - a)
            minus = transverse / (norm + dx) if dx > 0 else norm - dx
            plus = transverse / (norm - dx) if dx < 0 else norm + dx

            def evaluate(
                r: Decimal,
            ) -> tuple[Decimal, tuple[Decimal, Decimal, Decimal], Decimal | int]:
                distance = r * norm
                gap = ac - distance
                xgap = gap + r * minus if a > 0 else -gap - r * plus
                x, y, z = r * dx, r * dy, r * dz
                companion = ((x - 1) ** 2 + y * y + z * z).sqrt()
                companion_difference = (-xgap * (2 - a - x) - r * r * transverse) / (
                    companion * bc * (companion + bc)
                )
                centrifugal = (
                    -xgap * (total * (x + a) / 2 - c) + total * r * r * dy * dy / 2
                )
                residual = (
                    m * gap / (distance * ac)
                    + c * companion_difference
                    + centrifugal
                    + offset
                )
                gradient = (
                    -m * x / distance**3 - c * (x - 1) / companion**3 + total * x - c,
                    -m * y / distance**3 - c * y / companion**3 + total * y,
                    -m * z / distance**3 - c * z / companion**3,
                )
                derivative = sum(g * d for g, d in zip(gradient, (dx, dy, dz)))
                return residual, gradient, derivative

            r = Decimal.from_float(radius)
            previous = None
            # A rounded start can lie on the other side of a nearly double
            # root. Move inward in representable starting-radius steps until
            # the local star-facing branch has negative radial derivative.
            step = Decimal.from_float(ulp(radius))
            for _ in range(256):
                residual, gradient, derivative = evaluate(r)
                if residual >= 0 and derivative < 0:
                    break
                candidate = r - step
                r = candidate if candidate > 0 else r / 2
                step *= 2
            else:
                raise RuntimeError("exact ray root did not find the star-facing branch")
            lower = upper = None
            for _ in range(256):
                residual, gradient, derivative = evaluate(r)
                if not residual:
                    break
                if residual > 0:
                    lower = r
                else:
                    upper = r
                if lower is not None and upper is not None:
                    if context.next_plus(lower) >= upper:
                        r = min((lower, upper), key=lambda v: abs(evaluate(v)[0]))
                        break
                updated = r - residual / derivative
                if updated == r:
                    break
                if (
                    lower is not None
                    and upper is not None
                    and (not lower < updated < upper or updated == previous)
                ):
                    updated = lower + (upper - lower) / 2
                previous, r = r, updated
            else:
                raise RuntimeError("100-digit factored ray root did not converge")
            _, gradient, derivative = evaluate(r)
            if derivative >= 0:
                raise RuntimeError("exact ray root left the star-facing branch")
            magnitude = sum((v * v for v in gradient), Decimal(0)).sqrt()
            self.last_radius = str(r)
            return (
                float(r),
                tuple(float(-v / magnitude) for v in gradient),
                float(magnitude),
            )
