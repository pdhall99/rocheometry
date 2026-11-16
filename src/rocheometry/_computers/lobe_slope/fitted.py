"""Fast slope of the packaged lobe-radius polynomial, with numerical fallback."""

from dataclasses import dataclass

from rocheometry._computers.fitted import FITTED_QI_MAX, FITTED_QI_MIN
from rocheometry._computers.lobe_slope.protocol import LobeVolumeRadiusSlopeComputer
from rocheometry._computers.packaged import _packaged_lobe_slope_computer
from rocheometry._roche.field import RocheField


@dataclass(frozen=True, slots=True)
class FittedLobeSlopeComputer:
    """Differentiate the existing radius fit in its calibrated domain."""

    fallback: LobeVolumeRadiusSlopeComputer

    def __call__(self, field: RocheField, /) -> float:
        if not FITTED_QI_MIN <= field.qi <= FITTED_QI_MAX:
            return self.fallback(field)
        return _packaged_lobe_slope_computer()(field)
