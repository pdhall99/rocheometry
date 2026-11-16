"""Shared surface samples, their protocol, and angular sampling strategy."""

from rocheometry._computers.surface_sample.angular import (
    DEFAULT_FITTED_AZIMUTH_ORDER,
    DEFAULT_FITTED_POLAR_ORDER,
    GaussLegendreSurfaceSampler,
)
from rocheometry._computers.surface_sample.protocol import SurfaceSampler
from rocheometry._computers.surface_sample.sample import SurfaceSample

__all__ = [
    "DEFAULT_FITTED_AZIMUTH_ORDER",
    "DEFAULT_FITTED_POLAR_ORDER",
    "GaussLegendreSurfaceSampler",
    "SurfaceSample",
    "SurfaceSampler",
]
