"""Watershed cuts and upper-half assembly for contact sections."""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from math import atan2, hypot, pi, sqrt

from rocheometry._bezier import BezierPath, CubicBezier
from rocheometry._computers.neck import _neck_field, _unresolved_neck_position
from rocheometry._computers.watershed import WatershedComputer
from rocheometry._geometry import AffineTransform2D, Point2D
from rocheometry._planes import CentrelinePlane, _transverse_direction
from rocheometry._roche.field import RocheField
from rocheometry._roche.potential import CriticalPoint

# Insensitive control: contact plane parallel tolerance.
CONTACT_PLANE_PARALLEL_TOLERANCE = 1e-14
# Insensitive control: contact plane seed depth factor.
CONTACT_PLANE_SEED_DEPTH_FACTOR = 1e-12

# Derived: contact plane reflect y.
CONTACT_PLANE_REFLECT_Y = AffineTransform2D(yy=-1.0)


def contact_path(
    cut: tuple[CubicBezier, ...],
    envelope: tuple[CubicBezier, ...],
    *,
    arc_only: bool = False,
) -> BezierPath:
    """Join an upper cut and envelope, then reflect in reverse traversal order."""
    upper = cut + envelope
    lower = tuple(
        s.reversed().transformed(CONTACT_PLANE_REFLECT_Y) for s in reversed(upper)
    )
    return BezierPath(upper + lower, closed=not arc_only)


@dataclass(frozen=True, slots=True)
class ContactCut:
    """A plane's watershed graph, parametrized by normalized square-root depth."""

    field: RocheField
    depth: float
    plane: CentrelinePlane
    l1: CriticalPoint
    watershed: WatershedComputer
    _points: dict[float, Point2D] = dataclass_field(
        default_factory=dict, compare=False, repr=False
    )

    @property
    def sigma(self) -> float:
        return sqrt(self.depth)

    @property
    def rim_angle(self) -> float:
        return atan2(self.point(1.0)[1], self.point(1.0)[0])

    def point(self, parameter: float) -> Point2D:
        """Evaluate the exact graph; extend evenly/oddly across the L1 endpoint."""
        if parameter in self._points:
            return self._points[parameter]
        if parameter == 0:
            return (self.l1.position[0], 0.0)
        if abs(parameter) == 1:
            p = self.watershed.rim(self.field, self.depth, self.plane.inclination)
        else:
            p = self.watershed.rim_at_depth(
                self.field,
                (parameter * self.sigma) ** 2,
                self.plane.inclination,
            )
        point = (p[0], hypot(p[1], p[2]) * (1 if parameter > 0 else -1))
        self._points[parameter] = point
        return point

    def derivative(self, parameter: float) -> Point2D:
        """Differentiate in square-root depth, with the exact vertical L1 limit."""
        second = _transverse_direction(self.plane)
        if parameter == 0:
            h = self.field.hessian(self.l1.position[0])
            return (
                0.0,
                self.sigma
                * sqrt(2 / (-h[1][1] * second[1] ** 2 - h[2][2] * second[2] ** 2)),
            )
        x, rho = self.point(parameter)
        if _unresolved_neck_position(self.l1.position[0], abs(rho)):
            model = _neck_field(self.field, self.l1.position[0])
            y, z = rho * second[1], rho * second[2]
            offset = model.sheet_offset(y, z)
            gx, gy, gz = model.gradient(((offset, 0.0), (y, 0.0), (z, 0.0)))
            shift = model.sheet_offset(0.0, 0.0)
            xr = 2 * (offset - shift) / rho
            dr = (
                -2
                * parameter
                * self.depth
                / (gx * xr + gy * second[1] + gz * second[2])
            )
            return xr * dr, dr
        if self.plane.inclination in (0.0, pi / 2):
            x, t = self.point(parameter)
            g = self.field.gradient(x, t * second[1], t * second[2])
            factor = -2 * parameter * self.sigma**2 / sum(v * v for v in g)
            return (factor * g[0], factor * (g[1] * second[1] + g[2] * second[2]))
        # Fourth-order central differences also handle the endpoint by using
        # the dense gravity lines just beyond the requested surface level.
        h = min(1e-3, abs(parameter) / 8)
        a, b = self.point(parameter - 2 * h), self.point(parameter - h)
        c, d = self.point(parameter + h), self.point(parameter + 2 * h)
        return (
            (a[0] - 8 * b[0] + 8 * c[0] - d[0]) / (12 * h),
            (a[1] - 8 * b[1] + 8 * c[1] - d[1]) / (12 * h),
        )

    def hermite(self, left: float, right: float) -> CubicBezier:
        p0, p3 = self.point(left), self.point(right)
        d0, d1 = self.derivative(left), self.derivative(right)
        h = (right - left) / 3
        return CubicBezier(
            p0,
            (p0[0] + h * d0[0], p0[1] + h * d0[1]),
            (p3[0] - h * d1[0], p3[1] - h * d1[1]),
            p3,
        )

    def single(self) -> CubicBezier:
        """Interpolate the endpoints, endpoint tangents, and midpoint in sigma."""
        p0, p3 = self.point(0.0), self.point(1.0)
        middle = self.watershed.rim(self.field, self.depth / 4, self.plane.inclination)
        mid = (middle[0], hypot(*middle[1:]))
        v0, v1 = self.derivative(0.0), self.derivative(1.0)
        n0, n1 = hypot(*v0), hypot(*v1)
        t0, t1 = (v0[0] / n0, v0[1] / n0), (v1[0] / n1, v1[1] / n1)
        rx, ry = (
            (8 * mid[0] - 4 * (p0[0] + p3[0])) / 3,
            (8 * mid[1] - 4 * (p0[1] + p3[1])) / 3,
        )
        determinant = t0[1] * t1[0] - t0[0] * t1[1]
        if abs(determinant) < CONTACT_PLANE_PARALLEL_TOLERANCE:
            # Equal masses give a straight vertical cut and parallel tangents.
            # Preserve the total Hermite arm length, and use the remaining
            # freedom to interpolate the midpoint in square-root depth exactly.
            mean_arm = (n0 + n1) / 6
            a, b = mean_arm + ry / 2, mean_arm - ry / 2
            return CubicBezier(
                p0,
                (p0[0] + a * t0[0], p0[1] + a * t0[1]),
                (p3[0] - b * t1[0], p3[1] - b * t1[1]),
                p3,
            )
        a = (t1[0] * ry - t1[1] * rx) / determinant
        b = (t0[0] * ry - t0[1] * rx) / determinant
        return CubicBezier(
            p0,
            (p0[0] + a * t0[0], p0[1] + a * t0[1]),
            (p3[0] - b * t1[0], p3[1] - b * t1[1]),
            p3,
        )

    def adaptive(self, tolerance: float) -> tuple[CubicBezier, ...]:
        """Refine Hermite cubics against independent height inversions."""
        accepted = []

        def refine(left: float, right: float, depth: int) -> None:
            segment = self.hermite(left, right)
            for t in (0.25, 0.5, 0.75):
                x, transverse = segment.evaluate(t)
                exact = self.watershed.cut_height(
                    self.field, transverse, self.plane.inclination
                )
                if abs(x - exact) > 0.2 * tolerance:
                    break
            else:
                accepted.append(segment)
                return
            if depth == 24:
                raise RuntimeError("watershed cut refinement depth limit reached")
            middle = (left + right) / 2
            refine(left, middle, depth + 1)
            refine(middle, right, depth + 1)

        # Resolve the quadratic seed separately: below it the traced sheet
        # uses the L1 model, while above it the downhill flow relaxes onto the
        # sheet. Its small initial layer must not hide between error probes.
        seed = min(
            1.0,
            sqrt(CONTACT_PLANE_SEED_DEPTH_FACTOR * abs(self.l1.potential.value))
            / self.sigma,
        )
        refine(0.0, seed, 0)
        if seed < 1.0:
            refine(seed, 1.0, 0)
        return tuple(accepted)
