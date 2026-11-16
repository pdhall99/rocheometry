"""Binary Roche geometry views and composition."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from functools import partial
from typing import cast

from rocheometry._accuracy import Fitted, _Accuracy
from rocheometry._bezier import CompoundBezierPath
from rocheometry._component_geometry import (
    ComponentGeometry,
    ComponentSurfaceGeometry,
    RadiusName,
    _normalized_direction,
)
from rocheometry._computers.composition import ComputerSet
from rocheometry._frames import (
    component_frame,
    frame_transform,
    transform,
    transform_vector,
)
from rocheometry._geometry import AffineTransform2D, Bounds2D, Point3D, PointLike
from rocheometry._levels import _levels
from rocheometry._planes import (
    ORBITAL_PLANE,
    AxisNormalPlane,
    CentrelinePlane,
    SectionPlane,
    _require_silhouette_plane,
)
from rocheometry._roche.field import RocheField
from rocheometry._roche.potential import (
    CriticalPoint,
    Potential,
    PotentialConvention,
    convert_potential,
)
from rocheometry._roche.regime import LevelRegime, level_regime
from rocheometry._surface_quadrature import SurfaceQuadrature
from rocheometry._types import BinaryFrame, ComponentId, Frame
from rocheometry._validation import (
    normalize_mass,
    normalize_mass_ratio,
    normalize_query_point,
    normalize_separation,
    normalize_volume_radius,
    require_binary_frame,
    require_component_id,
    require_exactly_one,
    require_instance,
)


@dataclass(frozen=True, slots=True)
class BinaryGeometry:
    """A binary of mass ratio `q = M1/M2` and its two reciprocal components.

    Construction performs no solve. View coordinates and radii are scaled by
    `separation`, and volumes by its cube; potentials remain dimensionless.
    Critical-point Hessians use view axes and intrinsic G*(M1+M2)/a**3 units.

    Args:
        q: System mass ratio `M1/M2`. Must be finite and positive;
            real numeric inputs are converted to floats, excluding booleans.
            Ratios outside 1e-8 to 1e8 are accepted, but lie outside the
            verified range.
        separation: Distance between component centres in caller-chosen length
            units. Must be finite and positive; converted to a float.
        frame: Output chart: `binary` places the centres at `x=0` and `x=1`
            before scaling; `centre_of_mass` places the centre of mass at the origin.
        accuracy: Frozen accuracy choice shared by both components.

    Attributes:
        q: Normalized system mass ratio `M1/M2`.
        separation: Separation in caller-chosen length units.
        frame: Coordinate chart for all binary views.
        accuracy: Settings used for both components.

    Raises:
        ValueError: The reciprocal component mass ratio cannot be represented
            as a finite float.
    """

    q: float
    _: KW_ONLY
    separation: float = 1.0
    frame: BinaryFrame = "binary"
    accuracy: _Accuracy = Fitted()
    _computers: ComputerSet = field(init=False, repr=False, compare=False)
    _geometry1: ComponentGeometry = field(init=False, repr=False, compare=False)
    _geometry2: ComponentGeometry = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        q = normalize_mass_ratio(self.q)
        accuracy = require_instance(self.accuracy, _Accuracy, name="accuracy")
        object.__setattr__(self, "_computers", accuracy._computers())
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "separation", normalize_separation(self.separation))
        object.__setattr__(self, "frame", require_binary_frame(self.frame))
        roche_field = RocheField(q)
        object.__setattr__(
            self,
            "_geometry1",
            ComponentGeometry._from_resolved(roche_field, accuracy, self._computers),
        )
        object.__setattr__(
            self,
            "_geometry2",
            ComponentGeometry._from_resolved(
                roche_field.reciprocal(), accuracy, self._computers
            ),
        )

    def __reduce__(self) -> tuple[object, ...]:
        """Pickle the accuracy value, never the resolved graph or components."""
        return (
            partial(
                BinaryGeometry,
                separation=self.separation,
                frame=self.frame,
                accuracy=self.accuracy,
            ),
            (self.q,),
        )

    @classmethod
    def from_masses(
        cls,
        mass1: float,
        mass2: float,
        *,
        separation: float = 1.0,
        frame: BinaryFrame = "binary",
        accuracy: _Accuracy = Fitted(),
    ) -> BinaryGeometry:
        """Build a binary from component masses in any common unit; q = mass1/mass2."""
        q = normalize_mass(mass1, name="mass1") / normalize_mass(mass2, name="mass2")
        return cls(q, separation=separation, frame=frame, accuracy=accuracy)

    @property
    def geometry1(self) -> ComponentGeometry:
        """Component 1's intrinsic geometry, with `qi = q`."""
        return self._geometry1

    @property
    def geometry2(self) -> ComponentGeometry:
        """Component 2's intrinsic geometry, with `qi = 1/q`."""
        return self._geometry2

    @property
    def centre_of_mass(self) -> Point3D:
        """The centre of mass in this binary's output frame and length units.

        It is the origin of the centre-of-mass frame, so it is exactly zero there.
        """
        return self._view_position((0.0, 0.0, 0.0), source="centre_of_mass")

    def potential_at(self, point: PointLike) -> Potential:
        """Return the Jacobi potential at a point in this binary's view.

        Coordinates use the binary's frame and length units; the potential is in
        dimensionless total-mass Jacobi units and rounded once. Point-mass
        positions raise `ValueError`.

        Args:
            point: Three finite Cartesian coordinates in the binary view.

        Examples:
            >>> import rocheometry
            >>> binary = rocheometry.BinaryGeometry(1.0, separation=2.0)
            >>> binary.potential_at((1.0, 0.0, 0.0)) == binary.geometry1.l1().potential
            True
        """
        point = normalize_query_point(point)
        local = transform(
            (
                point[0] / self.separation,
                point[1] / self.separation,
                point[2] / self.separation,
            ),
            source=self.frame,
            target="component1",
            q=self.q,
        )
        return self.geometry1.potential_at(local)

    def component(self, component_id: ComponentId) -> BinaryComponentView:
        """Return a view of component 1 or 2 in this binary's frame and scale.

        Args:
            component_id: Component number, either `1` or `2`.
        """
        return BinaryComponentView(self, component_id)

    def l1(self) -> CriticalPoint:
        """Return L1 in view axes with its Hessian in G*(M1+M2)/a**3 units."""
        return self.component(1).l1()

    def l2(self) -> CriticalPoint:
        """Return L2 in this view.

        This is the outer point of the less massive component. At equal mass,
        component 2 supplies L2.
        """
        return self._less_massive_component().outer()

    def l3(self) -> CriticalPoint:
        """Return L3 in this view.

        This is the outer point of the more massive component. At equal mass,
        component 1 supplies L3.
        """
        return self._more_massive_component().outer()

    def l4(self) -> CriticalPoint:
        """Return L4 in this view.

        This is the point leading the less massive component for an orbit
        counterclockwise about +z.

        This is the more massive component's triangular point. At equal mass,
        component 1 supplies it, matching l3.
        """
        return self._more_massive_component().triangular()

    def l5(self) -> CriticalPoint:
        """Return L5 in this view.

        This is the point trailing the less massive component for an orbit
        counterclockwise about +z.

        This is the less massive component's triangular point. At equal mass,
        component 2 supplies it, matching l2.
        """
        return self._less_massive_component().triangular()

    def convert_potential(
        self,
        potential: Potential,
        *,
        convention: PotentialConvention,
        component_id: ComponentId | None = None,
    ) -> Potential:
        """Convert a typed potential using this binary mass ratio.

        Args:
            potential: Typed source value.
            convention: Target potential convention.
            component_id: Required for a target Kopal potential; must be `None`
                for the component-independent Jacobi convention.
        """
        return convert_potential(
            potential,
            q=self.q,
            convention=convention,
            component_id=component_id,
        )

    def lobe_bounds(self) -> Bounds2D:
        """Return the combined orbital-plane bounds of both critical-lobe sections.

        The tuple is `(xmin, ymin, xmax, ymax)` in this binary's frame and length
        units. Both sections are computed when first needed.
        """
        first = self.component(1).lobe().section().bounds
        second = self.component(2).lobe().section().bounds
        return (
            min(first[0], second[0]),
            min(first[1], second[1]),
            max(first[2], second[2]),
            max(first[3], second[3]),
        )

    def _view_transform(self, source: Frame) -> AffineTransform2D:
        """Map normalized orbital-plane coordinates in `source` into this view."""
        return frame_transform(source=source, target=self.frame, q=self.q).then(
            AffineTransform2D.scale(self.separation)
        )

    def _view_position(self, position: Point3D, *, source: Frame) -> Point3D:
        x, y = self._view_transform(source).apply((position[0], position[1]))
        return (x, y, self.separation * position[2])

    def _more_massive_component(self) -> BinaryComponentView:
        """Return the more massive component, component 1 at equal mass."""
        return self.component(1 if self.q >= 1.0 else 2)

    def _less_massive_component(self) -> BinaryComponentView:
        """Return the less massive component, component 2 at equal mass."""
        return self.component(2 if self.q >= 1.0 else 1)


@dataclass(frozen=True, slots=True)
class BinaryComponentView:
    """A component view in its parent binary's coordinate frame and scale.

    View positions use the binary's separation as their length unit.
    Critical-point Hessians rotate to the view axes and keep the intrinsic
    G*(M1+M2)/a**3 units. Creating a view performs no solve.

    Attributes:
        binary: Parent binary geometry.
        component_id: Component number, either `1` or `2`.
    """

    binary: BinaryGeometry
    component_id: ComponentId

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "binary",
            require_instance(self.binary, BinaryGeometry, name="binary"),
        )
        object.__setattr__(
            self,
            "component_id",
            require_component_id(self.component_id, name="component_id"),
        )

    @property
    def intrinsic_geometry(self) -> ComponentGeometry:
        """The normalized component-local geometry underlying this view."""
        return (
            self.binary.geometry1 if self.component_id == 1 else self.binary.geometry2
        )

    @property
    def position(self) -> Point3D:
        """The component centre in the binary's output frame and length units."""
        return self._output_position((0.0, 0.0, 0.0))

    def l1(self) -> CriticalPoint:
        """Return this component's L1 in the binary view."""
        return self._critical_view(self.intrinsic_geometry.l1())

    def outer(self) -> CriticalPoint:
        """Return this component's outer critical point in the binary view."""
        return self._critical_view(self.intrinsic_geometry.outer())

    def level_regime(self, potential: Potential) -> LevelRegime:
        """Classify a level in any potential convention, without constructing a surface."""
        levels = _levels(self.binary.geometry1.field, self.binary._computers)
        return level_regime(levels.depth(potential), levels.admissibility_gap)

    def triangular(self) -> CriticalPoint:
        """Return this component's triangular point in the binary view."""
        return self._critical_view(self.intrinsic_geometry.triangular())

    def volume_radius_fill_for(self, volume_radius: float) -> float:
        """Return the selection fill R/lobe_volume_radius() in view units.

        Fitted uses the fast model without sampling below fill one. Reported
        surface fill is measured and can differ within selection accuracy.

        Args:
            volume_radius: Finite positive radius in the binary's length units.
        """
        radius = normalize_volume_radius(volume_radius, name="volume_radius")
        return radius / self.lobe_volume_radius()

    def lobe_volume_radius(self) -> float:
        """Return the fast lobe-radius model in the binary's length units.

        Fitted uses packaged coefficients without sampling; Reference measures
        the lobe. lobe().volume_radius() always measures the critical surface
        with the shared quadrature. Both agree within the set's accuracy.
        """
        return self.binary.separation * self.intrinsic_geometry.lobe_volume_radius()

    def lobe_volume_radius_slope(self) -> float:
        """Return `d ln(R_L)/d ln(qi)` for this component's lobe radius.

        Here `qi = M_component/M_companion`. The result is dimensionless, so it
        does not depend on the binary's length units.
        """
        return self.intrinsic_geometry.lobe_volume_radius_slope()

    def max_volume_radius_fill(self) -> float:
        """Return the dimensionless, non-selectable fill supremum.

        It is continuous at equal masses, with different one-sided slopes.
        """
        return self.intrinsic_geometry.max_volume_radius_fill()

    def surface(
        self,
        *,
        volume_radius_fill: float | None = None,
        volume_radius: float | None = None,
        potential: Potential | None = None,
        fill_out_factor: float | None = None,
    ) -> BinaryComponentSurfaceView:
        """Select a surface by exactly one fill, radius, potential or fill-out factor.

        Radii use the binary's length units; fills and fill-out factors are
        dimensionless.

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
            volume_radius: Volume radius in the binary's length units.
            potential: Level in any potential convention.
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
            # Kopal values use the binary's component numbering, which only
            # component 1's local field shares.
            levels = _levels(self.binary.geometry1.field, self.binary._computers)
            return BinaryComponentSurfaceView._from_depth(self, levels.depth(potential))
        if fill_out_factor is not None:
            chosen = self.binary.geometry1.surface(fill_out_factor=fill_out_factor)
            return BinaryComponentSurfaceView._from_depth(self, chosen._depth)
        if volume_radius is not None:
            volume_radius_fill = self.volume_radius_fill_for(volume_radius)
            if volume_radius_fill > 1:
                maximum = self.max_volume_radius_fill() * self.lobe_volume_radius()
                if volume_radius >= maximum:
                    raise ValueError(
                        f"volume_radius {volume_radius!r} must be below maximum part volume radius {maximum!r}"
                    )
            volume_radius = None
        surface = self.intrinsic_geometry.surface(
            volume_radius_fill=volume_radius_fill,
            volume_radius=volume_radius,
            potential=potential,
            fill_out_factor=fill_out_factor,
        )
        return BinaryComponentSurfaceView._from_depth(self, surface._depth)

    def lobe(self) -> BinaryComponentSurfaceView:
        """Return this component's critical L1 surface view."""
        return BinaryComponentSurfaceView._from_depth(self, 0.0)

    def _critical_view(self, point: CriticalPoint) -> CriticalPoint:
        """Map positions to the view and rotate the intrinsic Hessian axes.

        Hessians retain G*(M1+M2)/a**3 units, independent of the displayed
        separation. Only the axes rotate; centre-of-mass translation has no
        effect. Component 2 uses R=diag(-1,-1,1).
        """
        # Component 2 rotates both x and y through pi; translation between
        # binary and centre-of-mass frames has no effect on the Hessian.
        signs = (1.0, 1.0, 1.0) if self.component_id == 1 else (-1.0, -1.0, 1.0)
        hessian = cast(
            tuple[Point3D, Point3D, Point3D],
            tuple(
                tuple(signs[i] * signs[j] * value for j, value in enumerate(row))
                for i, row in enumerate(point.hessian)
            ),
        )
        return CriticalPoint(
            position=self._output_position(point.position),
            potential=point.potential,
            hessian=hessian,
        )

    def _output_transform(self) -> AffineTransform2D:
        return self.binary._view_transform(component_frame(self.component_id))

    def _output_position(self, position: Point3D) -> Point3D:
        return self.binary._view_position(
            position, source=component_frame(self.component_id)
        )


@dataclass(frozen=True, slots=True, init=False)
class BinaryComponentSurfaceView:
    """A surface of one component of a binary, in the binary's frame and units.

    Construct it with `component.surface()` or `lobe()`. Its potential is
    reported in component 1's chart.
    """

    component: BinaryComponentView
    _depth: float
    _intrinsic_geometry: ComponentSurfaceGeometry = field(
        init=False, repr=False, compare=False
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("construct surfaces with component.surface() or lobe()")

    @classmethod
    def _from_depth(
        cls, component: BinaryComponentView, depth: float
    ) -> BinaryComponentSurfaceView:
        value = object.__new__(cls)
        object.__setattr__(value, "component", component)
        object.__setattr__(value, "_depth", depth)
        object.__setattr__(
            value,
            "_intrinsic_geometry",
            component.intrinsic_geometry._surface_at_depth(depth),
        )
        return value

    def __repr__(self) -> str:
        """Show how to select this surface again: the lobe, or its fill-out factor."""
        if self._depth == 0.0:
            return f"{self.component!r}.lobe()"
        return f"{self.component!r}.surface(fill_out_factor={self.fill_out_factor()!r})"

    def __reduce__(self) -> tuple[object, ...]:
        return (BinaryComponentSurfaceView._from_depth, (self.component, self._depth))

    @property
    def potential(self) -> Potential:
        """Round the common exact level once, identically for both components."""
        g = self.component.binary.geometry1
        return _levels(g.field, g._computers).potential(self._depth)

    @property
    def intrinsic_geometry(self) -> ComponentSurfaceGeometry:
        """The normalized component-local surface underlying this view."""
        return self._intrinsic_geometry

    @property
    def regime(self) -> LevelRegime:
        """The level regime of the intrinsic surface or envelope part."""
        return self.intrinsic_geometry.regime

    def fill_out_factor(self) -> float:
        """Return the convention-independent contact fill-out factor."""
        g = self.component.binary.geometry1
        return self._depth / _levels(g.field, g._computers).admissibility_gap

    def volume_radius_fill(self) -> float:
        """Return the intrinsic surface-to-lobe volume-radius ratio."""
        return self.intrinsic_geometry.volume_radius_fill()

    def volume(self) -> float:
        """Return enclosed volume in physical separation units cubed."""
        return self.intrinsic_geometry.volume() * self.component.binary.separation**3

    def volume_radius(self) -> float:
        """Return the volume radius in the binary's length units."""
        return (
            self.intrinsic_geometry.volume_radius() * self.component.binary.separation
        )

    def area(self) -> float:
        """Return envelope-share area, excluding the cut, in view length units squared."""
        return self.intrinsic_geometry.area() * self.component.binary.separation**2

    def area_radius(self) -> float:
        """Return `sqrt(A/(4*pi))`, the area radius in view length units."""
        return self.intrinsic_geometry.area_radius() * self.component.binary.separation

    def integrated_mean_curvature(self) -> float:
        """Return envelope-share integral H dA in view length units."""
        return (
            self.intrinsic_geometry.integrated_mean_curvature()
            * self.component.binary.separation
        )

    def mean_curvature_radius(self) -> float:
        """Return `M/(4*pi)`, the mean-curvature radius in view length units."""
        return (
            self.intrinsic_geometry.mean_curvature_radius()
            * self.component.binary.separation
        )

    def mean_gravity(self) -> float:
        """Return mean effective gravity in G*(M1+M2)/separation**2 units."""
        return self.intrinsic_geometry.mean_gravity()

    def quadrature(self) -> SurfaceQuadrature:
        """Return the surface's quadrature nodes and area weights in this view.

        Positions are in the binary's length units; normals point outward.
        Weights are in the binary's length units squared, H in inverse the binary's length units and K in inverse
        the binary's length units squared. Gravities are in G*(M1+M2)/separation**2.

        Standard nodes are in polar-major, azimuth-minor order, with the
        reflections (y, z), (y, -z), (-y, z), (-y, -z) consecutive for each node.
        Contact nodes cover only the envelope share, excluding the cut, like
        `area()`. Smooth integrands converge like `area()`; kinked ones such as
        max(0, n·v) converge more slowly, and at quadrature
        `silhouette(CentrelinePlane(i))` gives the projected boundary directly.
        Nodes carry the backward-error contract described on `SurfaceQuadrature`.
        """
        local = self.intrinsic_geometry.quadrature()
        binary = self.component.binary
        separation = binary.separation
        source = component_frame(self.component.component_id)
        return SurfaceQuadrature(
            positions=tuple(
                self.component._output_position(p) for p in local.positions
            ),
            normals=tuple(
                transform_vector(n, source=source, target=binary.frame, q=binary.q)
                for n in local.normals
            ),
            gravities=local.gravities,
            weights=tuple(w * separation**2 for w in local.weights),
            mean_curvatures=tuple(h / separation for h in local.mean_curvatures),
            gaussian_curvatures=tuple(
                k / separation / separation for k in local.gaussian_curvatures
            ),
        )

    def contains(self, point: PointLike) -> bool:
        """Return whether a point is inside this surface's region of the component.

        Coordinates use this binary's frame and length units. Contact parts end at
        the watershed, and the component's centre is inside. Points within rounding
        error of the surface, including points very close to its quadrature nodes,
        may be classified either way.

        Args:
            point: Three finite Cartesian coordinates in the binary view.

        Examples:
            >>> import rocheometry
            >>> binary = rocheometry.BinaryGeometry(1.0, separation=2.0)
            >>> surface = binary.component(2).lobe()
            >>> surface.contains((2.0, 0.0, 0.0))
            True
            >>> surface.contains((0.0, 0.0, 0.0))
            False
        """
        point = normalize_query_point(point)
        binary = self.component.binary
        local = transform(
            (
                point[0] / binary.separation,
                point[1] / binary.separation,
                point[2] / binary.separation,
            ),
            source=binary.frame,
            target=component_frame(self.component.component_id),
            q=binary.q,
        )
        return self.intrinsic_geometry.contains(local)

    def radius(self, direction: PointLike) -> float:
        """Return a physical radius along a direction in the binary output frame.

        The direction must contain three finite real coordinates and be non-zero;
        it need not be a unit vector. Use named_radius() for characteristic radii
        defined in the component-local frame.

        Args:
            direction: Three-coordinate tuple in the binary output frame.
                Coordinates must be finite and the vector nonzero; its magnitude
                does not affect the result.
        """
        direction = _normalized_direction(direction)
        binary = self.component.binary
        local_direction = transform_vector(
            direction,
            source=binary.frame,
            target=component_frame(self.component.component_id),
            q=binary.q,
        )
        return self.intrinsic_geometry.radius(local_direction) * binary.separation

    def named_radius(self, name: RadiusName) -> float:
        """Return a component-local characteristic radius in physical length units.

        Args:
            name: Component-local direction: `point` is `+x`, `back` is `-x`,
                `side` is `+y`, and `pole` is `+z`, regardless of output frame.
        """
        return (
            self.intrinsic_geometry.named_radius(name)
            * self.component.binary.separation
        )

    def section(self, plane: SectionPlane = ORBITAL_PLANE) -> CompoundBezierPath:
        """Return the closed outlines of this surface in a plane.

        For a contact part, each outline includes its cut on the watershed; use
        `section_envelope()` to draw the outline without it. Pieces come in the
        same order as in the component's own section.

        Paths are in the binary's frame and length units, in the plane's 2D axes.

        Args:
            plane: A plane in the binary's frame, with any offset in its length
                units from that frame's origin. Defaults to the orbital plane.

        Raises:
            EmptySectionError: The polar origin is not inside this part.
            NotImplementedError: An x-normal section of the heavier component is
                ring-shaped.
        """
        local_plane = self._local_plane(plane)
        return self.intrinsic_geometry.section(local_plane).transformed(
            self._section_transform(plane, local_plane)
        )

    def silhouette(self, plane: CentrelinePlane = ORBITAL_PLANE) -> CompoundBezierPath:
        """Return the closed outline of this surface seen at quadrature.

        The line of sight is normal to a plane through the centreline, as at the
        orbital phase where the components appear side by side. Unlike a section,
        the outline includes points off the plane where the line of sight grazes the
        surface. It starts at the front point. Contact levels have no closed
        silhouette of their own: use `silhouette_envelope()`.

        Paths are in the binary's frame and length units, in the plane's 2D axes.

        Args:
            plane: A centreline plane in the binary's frame. Defaults to the
                orbital plane.

        Raises:
            TypeError: The plane is axis-normal, so it has no line of sight.
            NotImplementedError: The level is in contact; use
                `silhouette_envelope()`.
        """
        _require_silhouette_plane(plane)
        return self.intrinsic_geometry.silhouette(plane).transformed(
            self._section_transform(plane, plane)
        )

    def silhouette_envelope(
        self, plane: CentrelinePlane = ORBITAL_PLANE
    ) -> CompoundBezierPath:
        """Return the outline of this surface seen at quadrature, without the cut.

        A detached or critical surface gives the same closed outline as
        `silhouette(plane)`. A contact part gives an open limb arc that ends where
        it crosses the watershed; the two components' arcs together draw the common
        envelope. Which component hides which is left to the caller.

        Paths are in the binary's frame and length units, in the plane's 2D axes.

        Args:
            plane: A centreline plane in the binary's frame. Defaults to the
                orbital plane.

        Raises:
            TypeError: The plane is axis-normal, so it has no line of sight.
        """
        _require_silhouette_plane(plane)
        return self.intrinsic_geometry.silhouette_envelope(plane).transformed(
            self._section_transform(plane, plane)
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

        Paths are in the binary's frame and length units, in the plane's 2D axes.

        Args:
            plane: A plane in the binary's frame, with any offset in its length
                units from that frame's origin. Defaults to the orbital plane.

        Raises:
            EmptySectionError: The polar origin is not inside this part.
            NotImplementedError: An x-normal section of the heavier component is
                ring-shaped.
        """
        local_plane = self._local_plane(plane)
        return self.intrinsic_geometry.section_envelope(local_plane).transformed(
            self._section_transform(plane, local_plane)
        )

    def _local_plane(self, plane: SectionPlane) -> SectionPlane:
        """Convert view-frame offsets into this component's separation units."""
        require_instance(plane, (CentrelinePlane, AxisNormalPlane), name="plane")
        if isinstance(plane, CentrelinePlane):
            return plane
        if plane.axis == "x":
            offset, _ = (
                self.component._output_transform().inverse().apply((plane.offset, 0.0))
            )
        else:
            offset = plane.offset / self.component.binary.separation
            if plane.axis == "y" and self.component.component_id == 2:
                offset = -offset
        return AxisNormalPlane(plane.axis, offset)

    def _section_transform(
        self, plane: SectionPlane, local_plane: SectionPlane
    ) -> AffineTransform2D:
        """Map the local plane chart into the view plane's own chart."""
        if isinstance(plane, CentrelinePlane):
            # Inclinations are folded by y/z symmetry. Component 2's reflected
            # chart therefore uses the same planar map at every inclination.
            # Retain its exact coefficients, avoiding a trigonometric round trip.
            return self.component._output_transform()

        def mapped(point: tuple[float, float]) -> tuple[float, float]:
            return plane.project(
                self.component._output_position(local_plane.lift(point))
            )

        origin = mapped((0.0, 0.0))
        first = mapped((1.0, 0.0))
        second = mapped((0.0, 1.0))
        return AffineTransform2D(
            xx=first[0] - origin[0],
            xy=second[0] - origin[0],
            yx=first[1] - origin[1],
            yy=second[1] - origin[1],
            tx=origin[0],
            ty=origin[1],
        )
