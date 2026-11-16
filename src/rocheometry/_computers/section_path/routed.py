"""Plane routing between interchangeable section-path strategies."""

from dataclasses import dataclass
from dataclasses import field as dataclass_field

from rocheometry._bezier import CompoundBezierPath
from rocheometry._computers.contact import (
    _geometry_depth,
    default_outer,
    require_part_depth,
)
from rocheometry._computers.critical_point import CriticalPointComputer
from rocheometry._computers.section_path.protocol import SectionPathComputer
from rocheometry._geometry import AffineTransform2D
from rocheometry._planes import (
    AxisNormalPlane,
    CentrelinePlane,
    SectionPlane,
    _canonical_section_plane,
)
from rocheometry._roche.field import RocheField


def _default_l1() -> CriticalPointComputer:
    from rocheometry._computers.defaults import default_l1_computer

    return default_l1_computer()


@dataclass(frozen=True, slots=True, kw_only=True)
class PlaneRoutedSectionPathComputer:
    """Route centreline planes and parallel sections to their own strategies.

    Attributes:
        centreline: Strategy for planes containing the line of centres,
            including zero-offset y and z sections.
        section: Strategy for all remaining axis-normal sections.
        l1_computer: Critical endpoint for plane and level validation.
        outer_computer: Outer opening bounding contact sections.
    """

    centreline: SectionPathComputer
    section: SectionPathComputer
    l1_computer: CriticalPointComputer = dataclass_field(default_factory=_default_l1)
    outer_computer: CriticalPointComputer = dataclass_field(
        default_factory=default_outer
    )

    def __post_init__(self) -> None:
        for name in ("centreline", "section", "l1_computer", "outer_computer"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name}: expected a callable computer")

    def __call__(
        self,
        field: RocheField,
        depth: float,
        plane: SectionPlane,
        /,
    ) -> CompoundBezierPath:
        """Return the section from the strategy selected by the plane."""
        l1 = self.l1_computer(field)
        depth = require_part_depth(
            field, depth, l1, self.outer_computer, allow_l2=False
        )
        depth = _geometry_depth(depth, l1)
        if isinstance(plane, AxisNormalPlane) and depth > 0:
            return self.section(field, depth, plane)
        canonical, reflect_second = _canonical_section_plane(plane)
        computer = (
            self.centreline if isinstance(canonical, CentrelinePlane) else self.section
        )
        path = computer(field, depth, canonical)
        if reflect_second:
            return path.transformed(AffineTransform2D(yy=-1.0)).reversed()
        return path

    def envelope(
        self, field: RocheField, depth: float, plane: SectionPlane, /
    ) -> CompoundBezierPath:
        """Route every envelope run, preserving the plane's traversal convention."""
        l1 = self.l1_computer(field)
        depth = require_part_depth(
            field, depth, l1, self.outer_computer, allow_l2=False
        )
        depth = _geometry_depth(depth, l1)
        if isinstance(plane, AxisNormalPlane) and depth > 0:
            return self.section.envelope(field, depth, plane)
        canonical, reflect_second = _canonical_section_plane(plane)
        computer = (
            self.centreline if isinstance(canonical, CentrelinePlane) else self.section
        )
        paths = computer.envelope(field, depth, canonical)
        if reflect_second:
            return CompoundBezierPath(
                tuple(
                    p.transformed(AffineTransform2D(yy=-1.0)).reversed()
                    for p in reversed(paths.paths)
                )
            )
        return paths
