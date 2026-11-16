"""Lazy adapters and artefact loaders for packaged fitted computers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import TYPE_CHECKING

from rocheometry._computers.fitted import FITTED_QI_MAX, FITTED_QI_MIN
from rocheometry._computers.lobe_slope.protocol import LobeVolumeRadiusSlopeComputer
from rocheometry._computers.lobe_volume import (
    LobeVolumeComputer,
    LobeVolumeFromRadiusComputer,
)
from rocheometry._computers.lobe_volume_radius.protocol import LobeVolumeRadiusComputer
from rocheometry._roche.field import RocheField

if TYPE_CHECKING:
    from rocheometry._computers.lobe_volume_radius.fitted.computer import (
        LobeVolumeRadiusModel,
    )
    from rocheometry._computers.surface_potential.fitted.computer import (
        SurfacePotentialMap,
    )


@dataclass(frozen=True, slots=True)
class PackagedLobeVolumeRadiusComputer:
    """A lazy adapter for the packaged fitted critical lobe-radius model.

    Construction performs no artefact load. The shared fitted model is loaded
    on first call and retained for later calls. Radius depends only on the
    supplied field's mass ratio and is expressed in separation units.
    """

    @property
    def qi_min(self) -> float:
        """Lower calibrated packaged-fit bound, without loading the artefact."""
        return FITTED_QI_MIN

    @property
    def qi_max(self) -> float:
        """Upper calibrated packaged-fit bound, without loading the artefact."""
        return FITTED_QI_MAX

    def __call__(self, field: RocheField, /) -> float:
        """Return the fitted critical lobe volume-equivalent radius."""
        return _packaged_lobe_volume_radius_model()(field)


@cache
def packaged_lobe_volume_radius_computer() -> LobeVolumeRadiusComputer:
    """Return the shared lazy adapter for the packaged fitted lobe radius.

    The fitted model is loaded and cached on first use.
    """
    return PackagedLobeVolumeRadiusComputer()


@cache
def packaged_lobe_volume_computer() -> LobeVolumeComputer:
    """Return the shared lazy fitted lobe-volume adapter."""
    return LobeVolumeFromRadiusComputer(packaged_lobe_volume_radius_computer())


def load_packaged_lobe_volume_radius_computer() -> LobeVolumeRadiusModel:
    """Load the fitted volume-radius computer shipped with Rocheometry.

    Each call deserializes a new model; use the `packaged_*_computer` adapters
    when the model should be loaded once and shared.
    """
    from rocheometry._computers.lobe_volume_radius.fitted.computer import (
        lobe_volume_radius_computer_from_json,
    )

    resource = (
        files("rocheometry").joinpath("_data").joinpath("volume-radius-model.json")
    )
    return lobe_volume_radius_computer_from_json(resource.read_text(encoding="utf-8"))


@cache
def _packaged_lobe_volume_radius_model() -> LobeVolumeRadiusModel:
    """Load the packaged fitted critical volume-radius model once."""
    return load_packaged_lobe_volume_radius_computer()


def load_packaged_surface_potential_map() -> SurfacePotentialMap:
    """Deserialize the shipped endpoint-constrained surface-potential tensor."""
    from rocheometry._computers.surface_potential.fitted.computer import (
        SurfacePotentialMap,
    )

    resource = (
        files("rocheometry").joinpath("_data").joinpath("surface-potential-map.json")
    )
    return SurfacePotentialMap.from_json(resource.read_text(encoding="utf-8"))


@cache
def _packaged_surface_potential_map() -> SurfacePotentialMap:
    """Load and share the packaged surface-potential tensor on first use."""
    return load_packaged_surface_potential_map()


@cache
def _packaged_lobe_slope_computer() -> LobeVolumeRadiusSlopeComputer:
    """Resolve and validate the analytic derivative once, on first use.

    Deferred construction preserves artefact-free geometry wiring. Subsequent
    slope calls do not import or type-check the model again.
    """
    from rocheometry._computers.lobe_volume_radius.fitted.computer import (
        PiecewiseSixthRootHornerLobeVolumeRadiusComputer,
    )

    model = _packaged_lobe_volume_radius_model()
    if not isinstance(model, PiecewiseSixthRootHornerLobeVolumeRadiusComputer):
        raise TypeError("packaged radius model must provide an analytic slope")
    return model.logarithmic_slope
