"""Protocol for depth selection, measured fill and fast lobe calibration."""

from typing import Protocol

from rocheometry._computers.lobe_volume import LobeVolumeComputer
from rocheometry._computers.surface_potential import SurfacePotentialComputer
from rocheometry._computers.volume_radius_fill import VolumeRadiusFillComputer


class SurfaceSelection(Protocol):
    """Forward depth selection and independently measured surface fill.

    `depth` returns zero at fill one, negative depth below fill one,
    and positive contact depth above it. `fill` always measures surface volume
    relative to the sampled critical lobe; it is exactly one at zero depth.
    Fitted forward selection is approximate, so its reported fill can differ
    from the request within the selection contract. Numerical selection targets
    the same measured normalization to the configured solver tolerance.

    `fast_lobe_volume` calibrates the fast geometry lobe radius: packaged with fitted
    selection, sampled with numerical selection. It need not be the measured
    normalization inside `fill`.
    """

    @property
    def fast_lobe_volume(self) -> LobeVolumeComputer:
        """The lobe volume calibrating the fast geometry radius."""
        ...

    @property
    def depth(self) -> SurfacePotentialComputer:
        """The forward fill-to-depth computer."""
        ...

    @property
    def fill(self) -> VolumeRadiusFillComputer:
        """The measured fill, normalized by the sampled lobe."""
        ...
