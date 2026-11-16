"""Dimensionless component-local Roche geometry."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from fractions import Fraction
from math import fsum, hypot, pi, sqrt
from typing import Literal, TypeAlias

from rocheometry._accuracy import Fitted, _Accuracy
from rocheometry._bezier import CompoundBezierPath
from rocheometry._computers.checks import require_jacobi_potential
from rocheometry._computers.composition import ComputerSet
from rocheometry._computers.contact import require_part_depth
from rocheometry._computers.ownership import belongs_to_component
from rocheometry._computers.surface_radius.anchor import _residual_anchor
from rocheometry._geometry import Point3D, PointLike
from rocheometry._levels import _levels
from rocheometry._planes import (
    ORBITAL_PLANE,
    AxisNormalPlane,
    CentrelinePlane,
    SectionPlane,
    _require_silhouette_plane,
)
from rocheometry._roche.field import RocheField
from rocheometry._roche.potential import CriticalPoint, Potential
from rocheometry._roche.regime import LevelRegime, level_regime
from rocheometry._roche.triangular import triangular_point
from rocheometry._roche.volume import volume_radius_from_volume
from rocheometry._surface_quadrature import SurfaceQuadrature
from rocheometry._validation import (
    _normalize_finite_float,
    _normalize_finite_positive_float,
    normalize_query_point,
    normalize_volume_radius,
    require_choice,
    require_exactly_one,
    require_instance,
)

RadiusName: TypeAlias = Literal["point", "back", "side", "pole"]
"""A component-local point, back, side, or pole radius direction."""

# Resolution budget: named radius directions.
NAMED_RADIUS_DIRECTIONS: dict[RadiusName, Point3D] = {
    "point": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "side": (0.0, 1.0, 0.0),
    "pole": (0.0, 0.0, 1.0),
}


@dataclass(frozen=True, slots=True)
class ComponentGeometry:
    """A component-local Roche geometry with a chosen accuracy.

    The selected component is at `x=0`, its companion at `x=1`, and their
    separation is the unit of length. Construction performs no numerical solve,
    and equal geometries share their computed results.

    Args:
        qi: Component-to-companion mass ratio. Must be finite and positive;
            real numeric inputs are converted to floats, excluding booleans.
            Ratios outside 1e-8 to 1e8 are accepted, but lie outside the
            verified range.
        accuracy: Frozen accuracy choice, Fitted() by default.

    Attributes:
        qi: Normalized component-to-companion mass ratio.
        accuracy: Settings used for all numerical computations.
    """

    qi: float
    _: KW_ONLY
    accuracy: _Accuracy = Fitted()
    _computers: ComputerSet = field(init=False, repr=False, compare=False)
    _field: RocheField = field(init=False, repr=False)

    def __post_init__(self) -> None:
        roche_field = RocheField(self.qi)
        object.__setattr__(self, "qi", roche_field.qi)
        object.__setattr__(self, "_field", roche_field)
        accuracy = require_instance(self.accuracy, _Accuracy, name="accuracy")
        object.__setattr__(self, "_computers", accuracy._computers())

    def __reduce__(self) -> tuple[object, ...]:
        """Rebuild from values, resolving the receiving process's cached graph."""
        return (_restore_component, (self.field, self.accuracy))

    @classmethod
    def _from_resolved(
        cls, roche_field: RocheField, accuracy: _Accuracy, computers: ComputerSet
    ) -> ComponentGeometry:
        """Construct a binary component with its parent's already resolved graph."""
        value = object.__new__(cls)
        object.__setattr__(value, "qi", roche_field.qi)
        object.__setattr__(value, "_field", roche_field)
        object.__setattr__(value, "accuracy", accuracy)
        object.__setattr__(value, "_computers", computers)
        return value

    @property
    def field(self) -> RocheField:
        """The normalized Roche field of this geometry."""
        return self._field

    def potential_at(self, point: PointLike) -> Potential:
        """Return the Jacobi potential at a component-local point.

        The potential is rounded once. Points are in separation units, and both
        point-mass positions raise `ValueError`.

        Args:
            point: Three finite Cartesian coordinates in this component's chart.

        Examples:
            >>> import rocheometry
            >>> geometry = rocheometry.ComponentGeometry(1.0)
            >>> geometry.potential_at((0.5, 0.0, 0.0)) == geometry.l1().potential
            True
        """
        point = normalize_query_point(point)
        l1 = self.l1()
        # Add the factored difference to the exact anchor in rationals, so the
        # report is rounded once rather than at an absolute-level addition.
        difference = self.field.potential_difference(point, l1.position)
        return Potential.jacobi(
            float(
                self.field._axis_potential_exact(l1.position[0]) + Fraction(difference)
            )
        )

    def l1(self) -> CriticalPoint:
        """Return local L1, with its Hessian in G*(M1+M2)/a**3 units."""
        return require_instance(
            self._computers.l1(self.field), CriticalPoint, name="l1 computer result"
        )

    def outer(self) -> CriticalPoint:
        """Return the local outer point and Hessian in G*(M1+M2)/a**3 units."""
        return require_instance(
            self._computers.outer(self.field),
            CriticalPoint,
            name="outer computer result",
        )

    def level_regime(self, potential: Potential) -> LevelRegime:
        """Classify a level without constructing a surface.

        The classification is exact: detached, critical, contact, or beyond the
        level at which the common envelope opens.

        Args:
            potential: Jacobi level. For a Kopal level, select through the binary.
        """
        potential = require_jacobi_potential(potential)
        levels = _levels(self.field, self._computers)
        return level_regime(levels.depth(potential), levels.admissibility_gap)

    def triangular(self) -> CriticalPoint:
        """Return local +y triangular point and Hessian in G*(M1+M2)/a**3 units."""
        return triangular_point(self.field)

    def volume_radius_fill_for(self, volume_radius: float) -> float:
        """Return the selection fill R/lobe_volume_radius(); may exceed one.

        Fitted uses the fast lobe model without sampling. The selected surface's
        measured fill can differ within its selection accuracy.

        Args:
            volume_radius: Finite positive radius in units of the separation.
        """
        radius = normalize_volume_radius(volume_radius, name="volume_radius")
        return radius / self.lobe_volume_radius()

    def max_volume_radius_fill(self) -> float:
        """Return the non-selectable part-fill supremum at the outer opening.

        It is continuous at qi=1, with a slope jump where the opening outer
        point switches between the two components.
        """
        return self._computers.selection.fill(
            self.field, _levels(self.field, self._computers).admissibility_gap
        )

    def surface(
        self,
        *,
        volume_radius_fill: float | None = None,
        volume_radius: float | None = None,
        potential: Potential | None = None,
        fill_out_factor: float | None = None,
    ) -> ComponentSurfaceGeometry:
        """Select a surface by exactly one fill, radius, potential or fill-out factor.

        Fills below one select detached surfaces, and a fill of one selects the
        critical lobe. Contact parts need a level above L2 and a fill below
        `max_volume_radius_fill()`. Fill-out factors may be any finite F < 1.

        Selection by fill is accurate to 2e-6 absolute with `Fitted()`, and to the
        solver tolerance with `Reference()`. The selected surface's fill and volume
        are measured, so they can differ from the request by that much.

        A surface is identified by its level's depth below L1. The reported
        `potential` is rounded once, so selecting a surface again by it can move
        the level slightly near L1 or L2. The reported fill-out factor is
        computed from the depth directly, which avoids that loss of precision,
        though it can still differ from a requested factor by rounding. The L2
        potential itself cannot be selected and raises `ValueError`.

        Args:
            volume_radius_fill: Volume radius as a fraction of the lobe's.
            volume_radius: Volume radius in separation units.
            potential: Jacobi level. For a Kopal level, select through the binary.
            fill_out_factor: Depth below L1 as a fraction of the depth of L2,
                negative for detached surfaces and positive for contact parts.
        """
        require_exactly_one(
            "surface",
            volume_radius_fill=volume_radius_fill,
            volume_radius=volume_radius,
            potential=potential,
            fill_out_factor=fill_out_factor,
        )
        if potential is not None:
            return self._surface_at_depth(
                _levels(self.field, self._computers).depth(
                    require_jacobi_potential(potential)
                )
            )
        if fill_out_factor is not None:
            factor = _normalize_finite_float(fill_out_factor, name="fill_out_factor")
            if factor >= 1:
                raise ValueError(f"fill_out_factor {factor!r} must be below 1 (L2)")
            return self._surface_at_depth(
                factor * _levels(self.field, self._computers).admissibility_gap
            )
        if volume_radius is not None:
            volume_radius_fill = self.volume_radius_fill_for(volume_radius)
            if volume_radius_fill > 1:
                maximum = self.max_volume_radius_fill() * self.lobe_volume_radius()
                if volume_radius >= maximum:
                    raise ValueError(
                        f"volume_radius {volume_radius!r} must be below maximum part volume radius {maximum!r}"
                    )
        assert volume_radius_fill is not None
        fill = _normalize_finite_positive_float(
            volume_radius_fill, name="volume_radius_fill"
        )
        if fill > 1:
            maximum_fill = self.max_volume_radius_fill()
            if fill >= maximum_fill:
                raise ValueError(
                    f"volume_radius_fill {fill!r} must be below maximum fill {maximum_fill!r}"
                )
        if fill == 1:
            return self._surface_at_depth(0.0)
        depth = self._computers.selection.depth(self.field, fill)
        return self._surface_at_depth(depth)

    def _surface_at_depth(self, depth: float) -> ComponentSurfaceGeometry:
        levels = _levels(self.field, self._computers)
        if depth >= levels.admissibility_gap:
            raise ValueError(
                f"potential must be above L2 ({levels.opening_report.value})"
            )
        # Validate depth directly; no absolute-level adapter or snapping band.
        require_part_depth(
            self.field,
            depth,
            self.l1(),
            self._computers.outer,
            allow_l2=False,
        )
        return ComponentSurfaceGeometry._from_depth(self, depth)

    def lobe(self) -> ComponentSurfaceGeometry:
        """Return the critical L1 surface value.

        The critical level is the saddle's exact potential at the computed L1
        position; `l1().potential` is its float representation and selects it
        exactly. All other levels retain their literal float values.
        """
        return self._surface_at_depth(0.0)

    def lobe_volume_radius_slope(self) -> float:
        """Return `d ln(R_L)/d ln(qi)` for the Roche-lobe volume radius.

        Here `qi = M_component/M_companion`. The slope against the reciprocal
        ratio `q = M_companion/M_component` is its negative. It falls from
        `1/3` as `qi -> 0` (the Hill limit) towards zero as `qi -> infinity`.
        The result is dimensionless.
        Within the calibrated domain, Fitted differentiates the packaged
        polynomial without sampling. Reference and out-of-domain fallbacks
        integrate the critical-surface shape derivative.
        """
        return self._computers.lobe_slope(self.field)

    def lobe_volume_radius(self) -> float:
        """Return the fast critical-lobe radius in separation units.

        Within the calibrated domain, Fitted uses the packaged model without
        sampling. Reference and out-of-domain fallbacks measure the critical
        lobe. In contrast, lobe().volume_radius() always measures
        the shared quadrature, just as every surface does. The two agree within
        the chosen set's accuracy. Use this method for population synthesis.
        """
        return volume_radius_from_volume(
            self._computers.selection.fast_lobe_volume(self.field)
        )


def _restore_component(
    roche_field: RocheField, accuracy: _Accuracy
) -> ComponentGeometry:
    return ComponentGeometry._from_resolved(
        roche_field, accuracy, accuracy._computers()
    )


@dataclass(frozen=True, slots=True, init=False)
class ComponentSurfaceGeometry:
    """A surface of one component, at one level.

    Construct it with `ComponentGeometry.surface()` or `lobe()`. Lengths are in
    separation units.
    """

    component: ComponentGeometry
    _depth: float

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "construct surfaces with ComponentGeometry.surface(potential=...) or lobe()"
        )

    @classmethod
    def _from_depth(
        cls, geometry: ComponentGeometry, depth: float
    ) -> ComponentSurfaceGeometry:
        value = object.__new__(cls)
        object.__setattr__(value, "component", geometry)
        object.__setattr__(value, "_depth", depth)
        return value

    def __repr__(self) -> str:
        """Show how to select this surface again: the lobe, or its fill-out factor."""
        if self._depth == 0.0:
            return f"{self.component!r}.lobe()"
        return f"{self.component!r}.surface(fill_out_factor={self.fill_out_factor()!r})"

    def __reduce__(self) -> tuple[object, ...]:
        return (
            ComponentSurfaceGeometry._from_depth,
            (self.component, self._depth),
        )

    @property
    def potential(self) -> Potential:
        """The surface's Jacobi level, rounded once."""
        g = self.component
        return _levels(g.field, g._computers).potential(self._depth)

    @property
    def qi(self) -> float:
        """The parent component-to-companion mass ratio."""
        return self.component.qi

    def volume_radius_fill(self) -> float:
        """Return the volume radius as a fraction of the lobe's, (V/V_lobe)**(1/3).

        The lobe is measured with the same quadrature, so the critical lobe's fill
        is exactly one. Near L1 a fill may round to one; `regime` gives the exact
        side.
        """
        geometry = self.component
        return geometry._computers.selection.fill(geometry.field, self._depth)

    def volume(self) -> float:
        """Return the enclosed volume in separation units cubed.

        Contact volumes include the watershed cut. The first read measures the
        surface; later reads of any measured quantity reuse that measurement.
        """
        geometry = self.component
        return geometry._computers.volume(geometry.field, self._depth)

    def volume_radius(self) -> float:
        """Return the volume radius of the selected surface."""
        return volume_radius_from_volume(self.volume())

    def area(self) -> float:
        """Return the area in separation units squared, excluding a contact part's cut."""
        geometry = self.component
        return geometry._computers.area(geometry.field, self._depth)

    def quadrature(self) -> SurfaceQuadrature:
        """Return the surface's quadrature nodes and area weights.

        Positions are in separation units; normals point outward.
        Weights are in separation units squared, H in inverse separation units and K in inverse
        separation units squared. Gravities are in G*(M1+M2)/separation**2.

        Standard nodes are in polar-major, azimuth-minor order, with the
        reflections (y, z), (y, -z), (-y, z), (-y, -z) consecutive for each node.
        Contact nodes cover only the envelope share, excluding the cut, like
        `area()`. Smooth integrands converge like `area()`; kinked ones such as
        max(0, n·v) converge more slowly, and at quadrature
        `silhouette(CentrelinePlane(i))` gives the projected boundary directly.
        Nodes carry the backward-error contract described on `SurfaceQuadrature`.
        """
        geometry = self.component
        return geometry._computers.quadrature(geometry.field, self._depth)

    def area_radius(self) -> float:
        """Return `sqrt(A/(4*pi))`, the area radius in separation units."""
        return sqrt(self.area() / (4.0 * pi))

    def integrated_mean_curvature(self) -> float:
        """Return envelope-share `M = integral(H*dA)` in separation units.

        `H` is the mean of the outward principal curvatures: a sphere of
        radius `R` has `H = 1/R` and `M = 4*pi*R`.
        The integral is continuous through L1, approaching its lobe value like
        `sqrt(F)` from contact because of the neck.
        """
        geometry = self.component
        return geometry._computers.mean_curvature(geometry.field, self._depth)

    def mean_curvature_radius(self) -> float:
        """Return `M/(4*pi)`, the mean-curvature radius in separation units."""
        return self.integrated_mean_curvature() / (4.0 * pi)

    def mean_gravity(self) -> float:
        """Return the area-weighted mean of `|gradient(U)|` on the surface.

        The mean excludes the cut on contact parts. Gravity is in units of
        G*(M1+M2)/separation**2.

        Gauss's law gives `(4*pi*m - 2*V)/A`, where `m` is the component's fraction
        of the total mass. This holds exactly for detached surfaces and for
        watershed parts, because no gravity crosses the watershed cut.
        """
        geometry = self.component
        return geometry._computers.mean_gravity(geometry.field, self._depth)

    @property
    def regime(self) -> LevelRegime:
        """The level regime of this selected surface or envelope part."""
        g = self.component
        return level_regime(
            self._depth, _levels(g.field, g._computers).admissibility_gap
        )

    def fill_out_factor(self) -> float:
        """Return (U_L1-u)/(U_L1-U_L2), independent of potential convention."""
        geometry = self.component
        return (
            self._depth / _levels(geometry.field, geometry._computers).admissibility_gap
        )

    def contains(self, point: PointLike) -> bool:
        """Return whether a point is inside this surface's region of the component.

        Contact parts end at the watershed, and detached surfaces exclude the
        companion and the exterior. The component's centre is inside. A point on the
        watershed is inside either part if it is inside the surface. Points within
        rounding error of the surface, including points very close to its
        quadrature nodes, may be classified either way.

        Args:
            point: Three finite Cartesian coordinates in this component's chart,
                in separation units.

        Examples:
            >>> import rocheometry
            >>> surface = rocheometry.ComponentGeometry(1.0).lobe()
            >>> surface.contains((0.0, 0.0, 0.0))
            True
            >>> surface.contains((1.0, 0.0, 0.0))
            False
        """
        point = normalize_query_point(point)
        if point == (0.0, 0.0, 0.0):
            return True  # The enclosed mass is interior; its potential is singular.
        geometry = self.component
        field, computers = geometry.field, geometry._computers
        l1 = computers.l1(field)
        bound = (
            abs(computers.outer(field).position[0])
            if self._depth > 0
            else l1.position[0]
        )
        # The same geometric enclosing spheres bracket the first radius roots.
        # Beyond them, a positive residual could belong to the centrifugal exterior
        # or companion and cannot describe this closed component.
        if hypot(*point) > bound or point == (1.0, 0.0, 0.0):
            return False
        anchor, offset = l1, self._depth
        if type(field).potential is RocheField.potential:
            anchor, offset = _residual_anchor(
                field, self._depth, point[0], l1, computers.outer
            )
        residual = fsum((field.potential_difference(point, anchor.position), offset))
        return residual > 0.0 and belongs_to_component(
            field, l1, computers.watershed, point
        )

    def radius(self, direction: PointLike) -> float:
        """Return the first intersection along a finite non-zero direction.

        Directions need not be unit vectors. At critical and contact levels,
        the +x boundary point is L1.

        Args:
            direction: Component-local three-coordinate tuple. Coordinates must be
                finite and the vector nonzero; it is normalized before solving.
        """
        direction = _normalized_direction(direction)
        geometry = self.component
        depth = self._depth
        if depth == 0 and direction == (1.0, 0.0, 0.0):
            return geometry.l1().position[0]
        return geometry._computers.radius(geometry.field, depth, direction)

    def named_radius(self, name: RadiusName) -> float:
        """Return the component-local point, back, side or pole radius.

        Args:
            name: Characteristic direction: `point` is `+x`, `back` is `-x`,
                `side` is `+y`, and `pole` is `+z`, in component-local coordinates.
        """
        return self.radius(NAMED_RADIUS_DIRECTIONS[_require_radius_name(name)])

    def section(self, plane: SectionPlane = ORBITAL_PLANE) -> CompoundBezierPath:
        """Return the closed outlines of this surface in a plane.

        For a contact part, each outline includes its cut on the watershed; use
        `section_envelope()` to draw the outline without it. The piece containing
        the polar origin comes first, then any other pieces in increasing x,
        towards the companion. Each outer loop starts at its front point and runs
        counter-clockwise, upper half first.

        Args:
            plane: A plane in the component's own frame. Defaults to the orbital
                plane.

        Raises:
            EmptySectionError: The polar origin is not inside this part.
            NotImplementedError: An x-normal section of the heavier component is
                ring-shaped.
        """
        geometry = self.component
        require_instance(plane, (CentrelinePlane, AxisNormalPlane), name="plane")
        return geometry._computers.section(geometry.field, self._depth, plane)

    def silhouette(self, plane: CentrelinePlane = ORBITAL_PLANE) -> CompoundBezierPath:
        """Return the closed outline of this surface seen at quadrature.

        The line of sight is normal to a plane through the centreline, as at the
        orbital phase where the components appear side by side. Unlike a section,
        the outline includes points off the plane where the line of sight grazes the
        surface. It starts at the front point, upper half first. Contact levels
        have no closed silhouette of their own: use `silhouette_envelope()`.

        Args:
            plane: A centreline plane in the component's own frame. Defaults to the
                orbital plane.

        Raises:
            TypeError: The plane is axis-normal, so it has no line of sight.
            NotImplementedError: The level is in contact; use
                `silhouette_envelope()`.
        """
        _require_silhouette_plane(plane)
        geometry = self.component
        return geometry._computers.silhouette(geometry.field, self._depth, plane)

    def silhouette_envelope(
        self, plane: CentrelinePlane = ORBITAL_PLANE
    ) -> CompoundBezierPath:
        """Return the outline of this surface seen at quadrature, without the cut.

        A detached or critical surface gives the same closed outline as
        `silhouette(plane)`. A contact part gives an open limb arc that ends where
        it crosses the watershed; the two components' arcs together draw the common
        envelope. Which component hides which is left to the caller.

        Args:
            plane: A centreline plane in the component's own frame. Defaults to the
                orbital plane.

        Raises:
            TypeError: The plane is axis-normal, so it has no line of sight.
        """
        _require_silhouette_plane(plane)
        geometry = self.component
        return geometry._computers.silhouette.envelope(
            geometry.field, self._depth, plane
        )

    def section_envelope(
        self, plane: SectionPlane = ORBITAL_PLANE
    ) -> CompoundBezierPath:
        """Return the outlines of this surface in a plane, without the watershed cut.

        Each segment is the matching segment of `section(plane)`, in the same
        order. Use this to draw a binary, and `section()` for filled shapes:

        - A detached or critical surface, or a contact slice that misses the cut,
          gives the same closed outline as `section()`.
        - A slice that is all cut gives an empty path.
        - A separated collar gives the closed loop around the component, then its
          open envelope arc.
        - Other outlines that meet the cut become open arcs.

        The two components' arcs together draw the common envelope, meeting at the
        rim to within the accuracy setting's tolerance.

        Args:
            plane: A plane in the component's own frame. Defaults to the orbital
                plane.

        Raises:
            EmptySectionError: The polar origin is not inside this part.
            NotImplementedError: An x-normal section of the heavier component is
                ring-shaped.
        """
        require_instance(plane, (CentrelinePlane, AxisNormalPlane), name="plane")
        geometry = self.component
        return geometry._computers.section.envelope(geometry.field, self._depth, plane)


def _normalized_direction(direction: PointLike) -> Point3D:
    x, y, z = normalize_query_point(direction, name="direction")
    scale = max(abs(x), abs(y), abs(z))
    if scale == 0.0:
        raise ValueError("direction: expected a non-zero vector")
    x, y, z = x / scale, y / scale, z / scale
    norm = hypot(x, y, z)
    return (x / norm, y / norm, z / norm)


def _require_radius_name(name: RadiusName) -> RadiusName:
    return require_choice(
        require_instance(name, str, name="name"), NAMED_RADIUS_DIRECTIONS, name="name"
    )
