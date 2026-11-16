"""Gauge-fixed, traced, and plane-routed Bézier section strategies."""

from typing import TYPE_CHECKING

from rocheometry._computers.section_path.gauge_fixed import (
    GaugeFixedSectionPathComputer,
)
from rocheometry._computers.section_path.protocol import SectionPathComputer
from rocheometry._computers.section_path.routed import PlaneRoutedSectionPathComputer
from rocheometry._computers.section_path.traced import (
    TracedSectionPathComputer,
)

__all__ = [
    "GaugeFixedSectionPathComputer",
    "PlaneRoutedSectionPathComputer",
    "SectionLandmarks",
    "SectionPathComputer",
    "SectionReference",
    "TracedSectionPathComputer",
    "reference_centreline_section",
    "reference_component_section",
    "reference_section",
]


if TYPE_CHECKING:
    from rocheometry._computers.section_path.reference import (
        SectionLandmarks,
        SectionReference,
        reference_centreline_section,
        reference_component_section,
        reference_section,
    )


def __getattr__(name: str) -> object:
    if name in {
        "SectionLandmarks",
        "SectionReference",
        "reference_section",
        "reference_centreline_section",
        "reference_component_section",
    }:
        from rocheometry._computers.section_path import reference

        return getattr(reference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
