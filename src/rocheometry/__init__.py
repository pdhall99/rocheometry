"""Compute the equipotential geometry of binary systems in the classical Roche model."""

from rocheometry._accuracy import Fitted, Reference
from rocheometry._bezier import BezierPath, CompoundBezierPath, CubicBezier
from rocheometry._binary_geometry import (
    BinaryComponentSurfaceView,
    BinaryComponentView,
    BinaryGeometry,
)
from rocheometry._component_geometry import ComponentGeometry, ComponentSurfaceGeometry
from rocheometry._errors import (
    ComponentOwnershipError,
    DegenerateLineOfSightError,
    EmptySectionError,
)
from rocheometry._geometry import AffineTransform2D
from rocheometry._orbit import kepler_separation
from rocheometry._planes import (
    ORBITAL_PLANE,
    AxisNormalPlane,
    CentrelinePlane,
    SectionPlane,
)
from rocheometry._roche.potential import (
    CriticalPoint,
    Potential,
    PotentialConvention,
    convert_potential,
)
from rocheometry._roche.regime import LevelRegime
from rocheometry._surface_quadrature import SurfaceQuadrature

__all__ = [
    "ORBITAL_PLANE",
    "AffineTransform2D",
    "AxisNormalPlane",
    "BezierPath",
    "BinaryComponentSurfaceView",
    "BinaryComponentView",
    "BinaryGeometry",
    "CentrelinePlane",
    "ComponentGeometry",
    "ComponentOwnershipError",
    "ComponentSurfaceGeometry",
    "CompoundBezierPath",
    "CriticalPoint",
    "CubicBezier",
    "DegenerateLineOfSightError",
    "EmptySectionError",
    "Fitted",
    "LevelRegime",
    "Potential",
    "PotentialConvention",
    "Reference",
    "SectionPlane",
    "SurfaceQuadrature",
    "convert_potential",
    "kepler_separation",
]


def __getattr__(name: str) -> str:
    # Read lazily, so importing the package doesn't read its metadata.
    if name == "__version__":
        from importlib.metadata import version

        return version("rocheometry")
    raise AttributeError(f"module 'rocheometry' has no attribute {name!r}")
