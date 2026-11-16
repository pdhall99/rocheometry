"""Adaptive axis-normal sections of watershed-divided envelope parts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import cached_property
from math import atan2, cos, fsum, hypot, pi, sin, sqrt
from typing import TYPE_CHECKING, TypeAlias

from rocheometry._bezier import BezierPath, CompoundBezierPath, CubicBezier
from rocheometry._geometry import AffineTransform2D, Point2D, Point3D
from rocheometry._numerics.roots import brent_root
from rocheometry._planes import AxisNormalPlane
from rocheometry._roche.field import RocheField

# Insensitive control: axis contact root and bracket tolerance.
AXIS_CONTACT_ROOT_AND_BRACKET_TOLERANCE = 2e-14
# Insensitive control: axis contact root relative tolerance.
AXIS_CONTACT_ROOT_RELATIVE_TOLERANCE = 8e-16
# Insensitive control: axis contact fold root absolute tolerance.
AXIS_CONTACT_FOLD_ROOT_ABSOLUTE_TOLERANCE = 2e-15
# Insensitive control: axis contact inner bracket seed.
AXIS_CONTACT_INNER_BRACKET_SEED = 1e-14

if TYPE_CHECKING:
    from rocheometry._computers.section_path.traced import TracedSectionPathComputer

# Derived: axis contact reflect x.
AXIS_CONTACT_REFLECT_X = AffineTransform2D(xx=-1.0)
# Derived: axis contact reflect y.
AXIS_CONTACT_REFLECT_Y = AffineTransform2D(yy=-1.0)
# A recursion guard, not an accuracy setting: the 53 recorded shallow-contact
# failures need depths 49--52, with accepted intervals at least 823,537 endpoint
# ULPs wide. Near-L2 radius noise is separately fixed and passes at depth 48.
# Work guard or cache capacity: axis contact max refinement depth.
AXIS_CONTACT_MAX_REFINEMENT_DEPTH = 56


@dataclass(frozen=True)
class _Slice:
    """A y or z plane's slice: its offset, its normal axis and a polar centre.

    Coordinates in the slice are ``(x, t)``, where ``t`` runs along the
    in-plane axis, ``moving``.
    """

    offset: float
    normal: int
    centre_x: float

    @property
    def moving(self) -> int:
        return 3 - self.normal

    @property
    def origin(self) -> Point3D:
        return (
            self.centre_x,
            self.offset if self.normal == 1 else 0.0,
            self.offset if self.normal == 2 else 0.0,
        )

    def space(self, x: float, t: float) -> Point3D:
        """Return the point ``(x, t)`` of the slice in three dimensions."""
        return (
            x,
            self.offset if self.normal == 1 else t,
            t if self.normal == 1 else self.offset,
        )

    def direction(self, c: float, s: float) -> Point3D:
        """Return the unit direction at polar angle (cos c, sin s) in the slice."""
        return (c, 0.0 if self.normal == 1 else s, s if self.normal == 1 else 0.0)


@dataclass
class AxisContactSection:
    """A contact slice, with separate exact envelope and sheet intersections."""

    tracer: TracedSectionPathComputer
    field: RocheField
    depth: float
    plane: AxisNormalPlane
    _heights: dict[tuple[float, float], float] = dataclass_field(default_factory=dict)
    _rims: dict[float, Point3D] = dataclass_field(default_factory=dict)
    _anchor: Point3D | None = None

    @cached_property
    def pole(self) -> float:
        return self.tracer.radius_computer(self.field, self.depth, (0.0, 0.0, 1.0))

    @cached_property
    def outer(self) -> float:
        return abs(self.tracer.outer_computer(self.field).position[0])

    def rim(self, psi: float) -> Point3D:
        if psi not in self._rims:
            self._rims[psi] = self.tracer.watershed_computer.rim(
                self.field, self.depth, psi
            )
        return self._rims[psi]

    def height(self, rho: float, psi: float) -> float:
        key = (rho, psi)
        if key not in self._heights:
            self._heights[key] = self.tracer.watershed_computer.cut_height(
                self.field, rho, psi
            )
        return self._heights[key]

    def sheet_radius(self, x: float, psi: float) -> float:
        """Intersect a fixed-x plane with the sheet using square-root depth.

        This is the same monotone height crossing as cut_height(rho, psi)=x,
        expressed in the sheet's native parameter to avoid nested inversions.
        """
        l1 = self.tracer.l1_computer(self.field)
        depth = self.depth
        points: dict[float, Point3D] = {0.0: l1.position, 1.0: self.rim(psi)}

        def point(t: float) -> Point3D:
            if t not in points:
                points[t] = self.tracer.watershed_computer.rim_at_depth(
                    self.field, depth * t * t, psi
                )
            return points[t]

        root = brent_root(
            lambda t: point(t)[0] - x,
            0.0,
            1.0,
            absolute_tolerance=AXIS_CONTACT_ROOT_AND_BRACKET_TOLERANCE,
            relative_tolerance=AXIS_CONTACT_ROOT_RELATIVE_TOLERANCE,
        )
        return hypot(*point(root)[1:])

    def value(self, point: Point3D) -> float:
        if point == (0.0, 0.0, 0.0):
            return float("inf")
        if self._anchor is not None and type(self.field) is RocheField:
            return self.field.potential_difference(point, self._anchor)
        if type(self.field) is RocheField:
            l1 = self.tracer.l1_computer(self.field)
            return self.field.potential_difference(point, l1.position) + self.depth
        return (
            self.field.potential_difference(
                point, self.tracer.l1_computer(self.field).position
            )
            + self.depth
        )

    def curvature(self, point: Point3D, first: int, second: int) -> float:
        """Signed curvature of the plane's implicit envelope curve."""
        g = self.field.gradient(*point)
        h = self.field.hessian(*point)
        a, b = g[first], g[second]
        return (
            -(
                h[first][first] * b * b
                - 2 * h[first][second] * a * b
                + h[second][second] * a * a
            )
            / hypot(a, b) ** 3
        )

    def sheet_slope(self, point: Point3D, moving: int) -> float:
        """Height-map derivative in one transverse coordinate."""
        if point[3 - moving] == 0:
            g = self.field.gradient(*point)
            return g[0] / g[moving]
        normal = self.sheet_normal(point)
        return -normal[moving] / normal[0]

    def sheet_normal(self, point: Point3D) -> Point3D:
        """Use gravity and fixed-depth azimuth variation as sheet tangents.

        Gravity supplies one tangent. Differencing the smooth rim family gives
        the other, avoiding four nested height inversions at nearby positions.
        """
        g = self.field.gradient(*point)
        l1 = self.tracer.l1_computer(self.field)
        depth = -self.field.potential_difference(point, l1.position)
        psi = atan2(point[2], point[1])
        h = 0.02
        points = tuple(
            self.tracer.watershed_computer.rim_at_depth(self.field, depth, psi + k * h)
            for k in (-2, -1, 1, 2)
        )
        tangent = tuple(
            (points[0][j] - 8 * points[1][j] + 8 * points[2][j] - points[3][j])
            / (12 * h)
            for j in range(3)
        )
        return (
            tangent[1] * g[2] - tangent[2] * g[1],
            tangent[2] * g[0] - tangent[0] * g[2],
            tangent[0] * g[1] - tangent[1] * g[0],
        )

    def empty(self) -> None:
        from rocheometry._errors import EmptySectionError

        raise EmptySectionError("section centre must be strictly inside the part")

    def envelope_radius(self, origin: Point3D, direction: Point3D, cap: float) -> float:
        """Solve the single enclosing root in a fixed bracket, without stepping out."""
        l1 = self.tracer.l1_computer(self.field)
        reference = None
        if (
            origin == (0.0, 0.0, 0.0)
            and type(self.field) is RocheField
            and direction[0] < 0
        ):
            l1 = self.tracer.l1_computer(self.field)
            outer = self.tracer.outer_computer(self.field)
            if abs(
                self.field._depth_offset(self.depth, l1.position[0], outer.position[0])
            ) < abs(self.depth):
                reference = outer

        if reference is not None:
            # Use the same depth-based, exact-anchor arithmetic as radial
            # rays. A distant rim reference loses precision near the L2 tip.
            ray_residual = self.field._ray_residual(
                direction, self.depth, reference.position[0], l1_x=l1.position[0]
            )

            def residual(t: float) -> float:
                return ray_residual(cap * t)

        else:

            def residual(t: float) -> float:
                return self.value(
                    (
                        origin[0] + cap * t * direction[0],
                        origin[1] + cap * t * direction[1],
                        origin[2] + cap * t * direction[2],
                    )
                )

        inner = AXIS_CONTACT_INNER_BRACKET_SEED if origin == (0.0, 0.0, 0.0) else 0.0
        right = 1.0
        if residual(right) > AXIS_CONTACT_ROOT_AND_BRACKET_TOLERANCE * (
            abs(self.tracer.l1_computer(self.field).potential.value) + abs(self.depth)
        ):
            # A translated plane can meet the companion beyond the neck.
            # Find the intervening potential minimum in this fixed bracket;
            # outward stepping could skip the intervening empty interval.
            a, b = inner, right
            golden = (sqrt(5.0) - 1) / 2
            c, d = b - golden * (b - a), a + golden * (b - a)
            fc, fd = residual(c), residual(d)
            for _ in range(80):
                if fc < fd:
                    b, d, fd = d, c, fc
                    c = b - golden * (b - a)
                    fc = residual(c)
                else:
                    a, c, fc = c, d, fd
                    d = a + golden * (b - a)
                    fd = residual(d)
            right = (a + b) / 2
            if residual(right) >= 0:
                raise RuntimeError("contact section envelope is not bracketed")
        if right == 1.0 and residual(right) >= 0:
            return cap
        return cap * brent_root(
            residual,
            inner,
            right,
            absolute_tolerance=0.0,
            relative_tolerance=AXIS_CONTACT_ROOT_RELATIVE_TOLERANCE,
        )

    def paths(self) -> tuple[CompoundBezierPath, tuple[BezierPath, ...]]:
        if self.plane.axis == "x":
            path, runs = self.axial(include_cut=True)
            assert path is not None
            return CompoundBezierPath((path,)), runs
        paths, runs = self.transverse(include_cut=True)
        assert paths is not None
        return paths, runs

    def envelope(self) -> tuple[BezierPath, ...]:
        """Build envelope segments without refining discarded cut segments."""
        if self.plane.axis == "x":
            return self.axial(include_cut=False)[1]
        return self.transverse(include_cut=False)[1]

    def transverse(
        self,
        *,
        include_cut: bool,
        centre_x: float = 0.0,
        star_end: float | None = None,
        back_cap: float | None = None,
    ) -> tuple[CompoundBezierPath | None, tuple[BezierPath, ...]]:
        """Trace each connected piece about an interior polar centre.

        A y or z plane meets the contact body in one closed curve per piece.
        Where the plane crosses the watershed rim, the curve has a fold: the
        cut sheet runs from the axis to the fold and the envelope from the
        fold round the back. If the neck pinches the slice, it is split into
        a star and a collar traced about separate centres. Each piece is
        traced in its upper half and mirrored.
        """
        slice_ = _Slice(
            abs(self.plane.offset), 1 if self.plane.axis == "y" else 2, centre_x
        )
        if slice_.offset >= self.outer or self.value(slice_.origin) <= 0:
            self.empty()
        axis_angle = 0.0 if slice_.normal == 1 else pi / 2
        extent = self.rim(axis_angle)[slice_.normal]
        crossed = slice_.offset < extent and star_end is None
        cut: tuple[CubicBezier, ...] = ()
        start = 0.0
        start_point: Point2D | None = None
        graph_envelope = False
        if slice_.offset == extent and self.field.qi >= 1 and star_end is None:
            start_point = self._touch_rim(slice_, axis_angle)
        if crossed:
            fold = self._fold(slice_, axis_angle)
            end_t = fold[slice_.moving]
            start_point = (fold[0], end_t)
            start = atan2(end_t, fold[0] - centre_x)
            graph_envelope = self._envelope_is_graph(slice_, fold)
            if centre_x == 0.0 and slice_.offset > 0.0:
                split = self._split_at_neck(slice_, axis_angle, include_cut)
                if split is not None:
                    return split
            if include_cut:
                cut = self._cut(slice_, fold)
        sphere_cap = sqrt(self.outer**2 - slice_.offset**2)
        if graph_envelope:
            assert start_point is not None
            env = self.transverse_graph(
                slice_.origin, slice_.moving, sphere_cap, start_point
            )
        else:
            env = self._transverse_envelope(
                slice_, start, start_point, sphere_cap, star_end, back_cap
            )
        return _close_transverse(cut, env, include_cut=include_cut, crossed=crossed)

    def _touch_rim(self, slice_: _Slice, axis_angle: float) -> Point2D:
        """Start the envelope where the plane touches the rim on the axis."""
        start_point = (self.rim(axis_angle)[0], 0.0)
        self._anchor = slice_.space(start_point[0], 0.0)
        return start_point

    def _fold(self, slice_: _Slice, axis_angle: float) -> Point3D:
        """Where the plane crosses the rim: the fold between cut and envelope."""
        offset, normal = slice_.offset, slice_.normal
        if offset == 0:
            psi = pi / 2 - axis_angle
        else:
            psi = brent_root(
                lambda a: self.rim(a)[normal] - offset,
                0.0,
                pi / 2,
                absolute_tolerance=AXIS_CONTACT_FOLD_ROOT_ABSOLUTE_TOLERANCE,
                relative_tolerance=AXIS_CONTACT_ROOT_RELATIVE_TOLERANCE,
            )
        p = self.rim(psi)
        self._anchor = (
            p[0],
            offset if normal == 1 else p[1],
            p[2] if normal == 1 else offset,
        )
        return p

    def _envelope_is_graph(self, slice_: _Slice, fold: Point3D) -> bool:
        """Whether the field points away from the centre at the fold.

        Then the envelope is not star-shaped about the centre and is traced as
        a graph over x instead.
        """
        assert self._anchor is not None
        gradient = self.field.gradient(*self._anchor)
        return (
            gradient[0] * (fold[0] - slice_.centre_x)
            + gradient[slice_.moving] * fold[slice_.moving]
            > 0
        )

    def _split_at_neck(
        self, slice_: _Slice, axis_angle: float, include_cut: bool
    ) -> tuple[CompoundBezierPath | None, tuple[BezierPath, ...]] | None:
        """Split a slice the neck pinches into a star and a collar, or None.

        Along the plane's x axis the field can dip below the level between
        the component and the cut: then the slice has two pieces, the star
        about the component centre up to the dip, and the collar beyond it.
        """
        offset = slice_.offset
        x_cut = self.height(offset, axis_angle)

        def axis_point(x: float) -> Point3D:
            return (x, slice_.origin[1], slice_.origin[2])

        if not (
            self.value(axis_point(x_cut)) >= 0
            and self.field.gradient(*axis_point(x_cut))[0] > 0
        ):
            return None
        minimum = brent_root(
            lambda x: self.field.gradient(*axis_point(x))[0],
            0.0,
            x_cut,
            absolute_tolerance=0.0,
            relative_tolerance=AXIS_CONTACT_ROOT_RELATIVE_TOLERANCE,
        )
        if not self.value(axis_point(minimum)) < 0:
            return None
        entry = brent_root(
            lambda x: self.value(axis_point(x)),
            minimum,
            x_cut,
            absolute_tolerance=0.0,
            relative_tolerance=AXIS_CONTACT_ROOT_RELATIVE_TOLERANCE,
        )
        star, star_runs = self.transverse(include_cut=include_cut, star_end=minimum)
        collar, collar_runs = self.transverse(
            include_cut=include_cut,
            centre_x=(entry + x_cut) / 2,
            back_cap=minimum,
        )
        if star is None or collar is None:
            return None, star_runs + collar_runs
        return CompoundBezierPath(star.paths + collar.paths), star_runs + collar_runs

    def _cut(self, slice_: _Slice, fold: Point3D) -> tuple[CubicBezier, ...]:
        """Trace the cut sheet from the axis to the fold, as a graph over t."""
        offset, normal, moving = slice_.offset, slice_.normal, slice_.moving
        end_t = fold[moving]

        def cut_point(t: float) -> Point2D:
            if t == end_t:
                return (fold[0], end_t)
            y, z = (offset, t) if normal == 1 else (t, offset)
            return (self.height(hypot(y, z), atan2(abs(z), abs(y))), t)

        def derivative(t: float) -> Point2D:
            if t == 0:
                return (0.0, 1.0)
            x, _ = cut_point(t)
            return (self.sheet_slope(slice_.space(x, t), moving), 1.0)

        return _adaptive(
            cut_point,
            derivative,
            lambda p: abs(p[0] - cut_point(p[1])[0]),
            0.0,
            end_t,
            self.tracer.tolerance * self.pole,
        )

    def _transverse_envelope(
        self,
        slice_: _Slice,
        start: float,
        start_point: Point2D | None,
        sphere_cap: float,
        star_end: float | None,
        back_cap: float | None,
    ) -> tuple[CubicBezier, ...]:
        """Trace the envelope in polar angle about the centre, from the start to pi.

        Rays are capped by the bounding sphere, and by the fold, the star's
        end or the collar's back where those apply.
        """
        centre_x, moving = slice_.centre_x, slice_.moving

        def radius(angle: float) -> float:
            if start_point is not None and angle == start:
                return hypot(start_point[0] - centre_x, start_point[1])
            c, s = _trig(angle)
            cap = -centre_x * c + sqrt(sphere_cap**2 - (centre_x * s) ** 2)
            if start_point is not None and c > 0:
                cap = min(cap, (start_point[0] - centre_x) / c)
            if star_end is not None and c > 0:
                cap = min(cap, (star_end - centre_x) / c)
            if back_cap is not None and c < 0:
                cap = min(cap, (back_cap - centre_x) / c)
            return self.envelope_radius(slice_.origin, slice_.direction(c, s), cap)

        def point(angle: float) -> Point2D:
            if angle == start and start_point is not None:
                return start_point
            r = radius(angle)
            c, s = _trig(angle)
            return (centre_x + r * c, r * s)

        def derivative(angle: float) -> Point2D:
            x, t = point(angle)
            p = slice_.space(x, t)
            g = self.field.gradient(*p)
            c, s = _trig(angle)
            r = hypot(x - centre_x, t)
            if g[0] == g[moving] == 0:
                # A saddle on the slice: the level curve's slope from the Hessian.
                h = self.field.hessian(*p)
                slope = -r * sqrt(-h[moving][moving] / h[0][0])
            else:
                slope = -r * (-s * g[0] + c * g[moving]) / (c * g[0] + s * g[moving])
            return (slope * c - r * s, slope * s + r * c)

        def curvature(angle: float) -> float:
            x, t = point(angle)
            p = slice_.space(x, t)
            g = self.field.gradient(*p)
            if g[0] == g[moving] == 0:
                return 0.0
            return self.curvature(p, 0, moving)

        return _adaptive(
            point,
            derivative,
            lambda p: abs(
                hypot(p[0] - centre_x, p[1]) - radius(atan2(p[1], p[0] - centre_x))
            ),
            start,
            pi,
            self.tracer.tolerance * self.pole,
            curvature=curvature,
        )

    def transverse_graph(
        self, origin: Point3D, moving: int, cap: float, rim: Point2D
    ) -> tuple[CubicBezier, ...]:
        """Follow a connected upper envelope that folds in polar angle.

        Star shape about the star does not imply star shape about a translated
        plane's origin. The envelope is still a graph in x. Splitting at its
        radial tangency gives fixed brackets for both radial branches.
        """
        normal = 3 - moving
        back = -self.envelope_radius(origin, (-1.0, 0.0, 0.0), cap)
        length = rim[0] - back
        widths: dict[float, float] = {back: 0.0, rim[0]: rim[1]}

        def space(x: float, t: float) -> Point3D:
            return (x, origin[1] if normal == 1 else t, t if normal == 1 else origin[2])

        def width(x: float) -> float:
            if x not in widths:
                if moving == 1:
                    z = origin[2]
                    upper = brent_root(
                        lambda y: (
                            1
                            - self.field.component_mass / hypot(x, y, z) ** 3
                            - self.field.companion_mass / hypot(1 - x, y, z) ** 3
                        ),
                        0.0,
                        2.0,
                    )
                else:
                    l1 = self.tracer.l1_computer(self.field)
                    x1 = l1.position[0]
                    c = self.field.companion_mass
                    dx = x1 - x
                    transverse = origin[1] ** 2
                    # U_L1-C(x,y)-depth, using the anchor's gravity and the
                    # factored centrifugal difference, never an absolute u.
                    gap = fsum(
                        (
                            self.field.component_mass / abs(x1),
                            c / abs(1 - x1),
                            0.5 * (dx * (x1 + x - 2 * c) - transverse),
                            self.field._half_mass_error * (dx * (x1 + x) - transverse),
                            -self.depth,
                        )
                    )
                    upper = 2 / gap
                widths[x] = brent_root(
                    lambda t: self.value(space(x, t)),
                    0.0,
                    upper,
                    absolute_tolerance=0.0,
                    relative_tolerance=AXIS_CONTACT_ROOT_RELATIVE_TOLERANCE,
                )
            return widths[x]

        def radial_gradient(x: float) -> float:
            t = width(x)
            g = self.field.gradient(*space(x, t))
            return g[0] * x + g[moving] * t

        fold_x = brent_root(
            radial_gradient,
            back,
            rim[0],
            absolute_tolerance=0.0,
            relative_tolerance=AXIS_CONTACT_ROOT_RELATIVE_TOLERANCE,
        )
        fold = (fold_x, width(fold_x))
        fold_angle = atan2(fold[1], fold[0])
        fold_s = 1 - sqrt((fold_x - back) / length)
        rim_angle = atan2(rim[1], rim[0])

        def point(s: float) -> Point2D:
            if s == 0:
                return rim
            if s == 1:
                return (back, 0.0)
            if s == fold_s:
                return fold
            x = back + length * (1 - s) ** 2
            return (x, width(x))

        def derivative(s: float) -> Point2D:
            x, t = point(s)
            p = space(x, t)
            g = self.field.gradient(*p)
            if s == 1:
                h = self.field.hessian(*p)
                return (0.0, -sqrt(-2 * g[0] * length / h[moving][moving]))
            dx = -2 * length * (1 - s)
            return (dx, -g[0] / g[moving] * dx)

        def curvature(s: float) -> float:
            x, t = point(s)
            return self.curvature(space(x, t), 0, moving)

        segments: tuple[CubicBezier, ...] = ()
        for a, b, entering in ((0.0, fold_s, True), (fold_s, 1.0, False)):

            def error(p: Point2D, *, entering: bool = entering) -> float:
                angle = atan2(p[1], p[0])
                if angle < fold_angle or (entering and angle > rim_angle):
                    return float("inf")
                if angle == fold_angle:
                    exact = hypot(*fold)
                else:
                    c, s = _trig(angle)
                    direction = (
                        c,
                        0.0 if normal == 1 else s,
                        s if normal == 1 else 0.0,
                    )
                    if entering:
                        low, high = fold_x / c, rim[0] / c
                        exact = brent_root(
                            lambda r: self.value(space(r * c, r * s)),
                            low,
                            high,
                            absolute_tolerance=0.0,
                            relative_tolerance=AXIS_CONTACT_ROOT_RELATIVE_TOLERANCE,
                        )
                    else:
                        upper = min(cap, fold_x / c) if c > 0 else cap
                        exact = self.envelope_radius(origin, direction, upper)
                return abs(hypot(*p) - exact)

            segments += _adaptive(
                point,
                derivative,
                error,
                a,
                b,
                self.tracer.tolerance * self.pole,
                curvature=curvature,
            )
        return segments

    def axial(
        self, *, include_cut: bool
    ) -> tuple[BezierPath | None, tuple[BezierPath, ...]]:
        """Trace quadrant runs, splitting exactly at each rim intersection.

        An x plane meets the body in one curve about the axis. Each quadrant is
        split where the curve crosses the watershed rim; each part is either
        envelope or cut sheet. The first quadrant is traced and mirrored into
        the others, and the envelope parts are joined into open runs.
        """
        x = self.plane.offset
        self._check_axial_plane(x)
        origin = (x, 0.0, 0.0)
        if abs(x) >= self.outer or self.value(origin) <= 0:
            self.empty()
        cap = sqrt(self.outer**2 - x * x)
        intervals = self._rim_crossings(x)
        quarter: list[tuple[CubicBezier | None, bool]] = []
        for left, right in zip(intervals[:-1], intervals[1:], strict=True):
            cut = self.field.qi < 1 and x > self.rim((left + right) / 2)[0]
            if cut and not include_cut:
                quarter.append((None, False))
                continue
            segments = self._axial_part(
                x, origin, cap, left, right, interior=intervals[1:-1], cut=cut
            )
            quarter.extend((s, not cut) for s in segments)
        return _close_axial(_mirror_quarter(quarter), include_cut=include_cut)

    def _check_axial_plane(self, x: float) -> None:
        """Planes beyond L1 belong to the companion, or are not supported."""
        l1_x = self.tracer.l1_computer(self.field).position[0]
        if x >= l1_x:
            if self.field.qi > 1 and x > l1_x:
                largest = max(self.rim(k * pi / 32)[0] for k in range(17))
                if x < largest:
                    raise NotImplementedError(
                        "ring-shaped section of the heavier component's part"
                    )
            self.empty()

    def _rim_crossings(self, x: float) -> list[float]:
        """Angles in the first quadrant that split it at the watershed rim.

        Only a light component's part reaches past the rim; its crossings are
        bracketed on a 17-point grid and solved.
        """
        intervals = [0.0, pi / 2]
        if self.field.qi < 1 and x > min(self.rim(0.0)[0], self.rim(pi / 2)[0]):
            grid = tuple(k * pi / 32 for k in range(17))
            for a, b in zip(grid[:-1], grid[1:], strict=True):
                fa, fb = self.rim(a)[0] - x, self.rim(b)[0] - x
                if fa == 0:
                    intervals.append(a)
                if fa * fb < 0:
                    intervals.append(
                        brent_root(
                            lambda psi: self.rim(psi)[0] - x,
                            a,
                            b,
                            absolute_tolerance=AXIS_CONTACT_FOLD_ROOT_ABSOLUTE_TOLERANCE,
                            relative_tolerance=AXIS_CONTACT_ROOT_RELATIVE_TOLERANCE,
                        )
                    )
        return sorted(set(intervals))

    def _axial_part(
        self,
        x: float,
        origin: Point3D,
        cap: float,
        left: float,
        right: float,
        *,
        interior: list[float],
        cut: bool,
    ) -> tuple[CubicBezier, ...]:
        """One part of the quadrant, envelope or cut sheet, in polar angle.

        At a rim crossing (an ``interior`` angle) the radius is the rim's,
        exactly, so neighbouring parts share their endpoint.
        """
        radii: dict[float, float] = {}

        def radius(angle: float) -> float:
            if angle not in radii:
                c, s = _trig(angle)
                if angle in interior:
                    radii[angle] = hypot(*self.rim(angle)[1:])
                elif cut:
                    radii[angle] = self.sheet_radius(x, angle)
                else:
                    radii[angle] = self.envelope_radius(origin, (0.0, c, s), cap)
            return radii[angle]

        def point(angle: float) -> Point2D:
            r = radius(angle)
            c, s = _trig(angle)
            return (r * c, r * s)

        def derivative(angle: float) -> Point2D:
            c, s = _trig(angle)
            r = radius(angle)
            if angle in (0.0, pi / 2):
                slope = 0.0
            elif cut:
                normal = self.sheet_normal((x, r * c, r * s))
                slope = (
                    -r
                    * (-s * normal[1] + c * normal[2])
                    / (c * normal[1] + s * normal[2])
                )
            else:
                g = self.field.gradient(x, r * c, r * s)
                slope = -r * (-s * g[1] + c * g[2]) / (c * g[1] + s * g[2])
            return (slope * c - r * s, slope * s + r * c)

        def error(p: Point2D) -> float:
            return abs(hypot(*p) - radius(atan2(p[1], p[0])))

        def curvature(angle: float) -> float:
            y, z = point(angle)
            return self.curvature((x, y, z), 1, 2)

        return _adaptive(
            point,
            derivative,
            error,
            left,
            right,
            self.tracer.tolerance * self.pole,
            curvature=None if cut else curvature,
        )


#: A segment of the section and whether it is envelope (True) or cut sheet.
Part: TypeAlias = tuple[CubicBezier | None, bool]


def _mirror_quarter(quarter: list[Part]) -> list[Part]:
    """Reflect the first quadrant's parts in z, then the half in y."""
    upper = quarter + [
        (
            s.reversed().transformed(AXIS_CONTACT_REFLECT_X) if s is not None else None,
            e,
        )
        for s, e in reversed(quarter)
    ]
    return upper + [
        (
            s.reversed().transformed(AXIS_CONTACT_REFLECT_Y) if s is not None else None,
            e,
        )
        for s, e in reversed(upper)
    ]


def _close_axial(
    full: list[Part], *, include_cut: bool
) -> tuple[BezierPath | None, tuple[BezierPath, ...]]:
    """Close the section, and join its envelope parts into open runs."""
    path = (
        BezierPath(tuple(s for s, _ in full if s is not None), closed=True)
        if include_cut or all(e for _, e in full)
        else None
    )
    if all(e for _, e in full):
        assert path is not None
        return path, (path,)
    if not any(e for _, e in full):
        return path, ()
    return path, _envelope_runs(full)


def _envelope_runs(full: list[Part]) -> tuple[BezierPath, ...]:
    """Join consecutive envelope parts into open runs, cut parts between."""
    # Start at the beginning of an envelope run, so no run wraps the seam.
    if full[0][1]:
        start = len(full) - 1
        while full[start - 1][1]:
            start -= 1
    else:
        start = next(i for i, (_, e) in enumerate(full) if e)
    ordered = full[start:] + full[:start]
    runs: list[BezierPath] = []
    current: list[CubicBezier] = []
    for segment, envelope in ordered:
        if envelope:
            assert segment is not None
            current.append(segment)
        elif current:
            runs.append(BezierPath(tuple(current), closed=False))
            current = []
    if current:
        runs.append(BezierPath(tuple(current), closed=False))
    return tuple(runs)


def _close_transverse(
    cut: tuple[CubicBezier, ...],
    env: tuple[CubicBezier, ...],
    *,
    include_cut: bool,
    crossed: bool,
) -> tuple[CompoundBezierPath | None, tuple[BezierPath, ...]]:
    """Mirror the upper half into the closed section and its envelope run."""
    upper = cut + env
    lower = tuple(
        s.reversed().transformed(AXIS_CONTACT_REFLECT_Y) for s in reversed(upper)
    )
    path = (
        BezierPath(upper + lower, closed=True) if include_cut or not crossed else None
    )
    if not crossed:
        assert path is not None
        return CompoundBezierPath((path,)), (path,)
    envelope_lower = tuple(
        s.reversed().transformed(AXIS_CONTACT_REFLECT_Y) for s in reversed(env)
    )
    return (CompoundBezierPath((path,)) if path is not None else None), (
        BezierPath(env + envelope_lower, closed=False),
    )


def _trig(angle: float) -> Point2D:
    if angle == 0.0:
        return (1.0, 0.0)
    if angle == pi / 2:
        return (0.0, 1.0)
    if angle == pi:
        return (-1.0, 0.0)
    return (cos(angle), sin(angle))


def _adaptive(
    point: Callable[[float], Point2D],
    derivative: Callable[[float], Point2D],
    error: Callable[[Point2D], float],
    left: float,
    right: float,
    tolerance: float,
    *,
    curvature: Callable[[float], float] | None = None,
) -> tuple[CubicBezier, ...]:
    """Refine G2 envelope cubics or Hermite cuts at independent geometric probes."""
    accepted: list[CubicBezier] = []
    knots: dict[float, tuple[Point2D, Point2D]] = {}

    def knot(t: float) -> tuple[Point2D, Point2D]:
        if t not in knots:
            knots[t] = point(t), derivative(t)
        return knots[t]

    def refine(a: float, b: float, depth: int) -> None:
        p0, d0 = knot(a)
        p3, d1 = knot(b)
        h = (b - a) / 3
        arm0, arm1 = (h * d0[0], h * d0[1]), (h * d1[0], h * d1[1])
        if curvature is not None:
            from rocheometry._computers.section_path.traced import _g2_arms, _Knot

            n0, n1 = hypot(*d0), hypot(*d1)
            t0, t1 = (d0[0] / n0, d0[1] / n0), (d1[0] / n1, d1[1] / n1)
            arms = _g2_arms(
                _Knot(a, p0, d0, curvature(a)),
                _Knot(b, p3, d1, curvature(b)),
                t0,
                t1,
                h * n0,
            )
            if arms is not None:
                arm0, arm1 = (
                    (arms[0] * t0[0], arms[0] * t0[1]),
                    (arms[1] * t1[0], arms[1] * t1[1]),
                )
        segment = CubicBezier(
            p0,
            (p0[0] + arm0[0], p0[1] + arm0[1]),
            (p3[0] - arm1[0], p3[1] - arm1[1]),
            p3,
        )
        if all(
            error(segment.evaluate(t)) <= 0.1 * tolerance for t in (0.25, 0.5, 0.75)
        ):
            accepted.append(segment)
            return
        if depth == AXIS_CONTACT_MAX_REFINEMENT_DEPTH:
            raise RuntimeError("contact section refinement depth limit reached")
        mid = (a + b) / 2
        refine(a, mid, depth + 1)
        refine(mid, b, depth + 1)

    refine(left, right, 0)
    return tuple(accepted)
