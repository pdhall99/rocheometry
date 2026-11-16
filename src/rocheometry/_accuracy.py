"""Immutable public accuracy choices for privately wired computations."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rocheometry._validation import require_quadrature_order

if TYPE_CHECKING:
    from rocheometry._computers.composition import ComputerSet
    from rocheometry._computers.defaults import _Settings


class _Accuracy:
    __slots__ = ()

    def _settings(self) -> "_Settings":
        raise NotImplementedError

    def _computers(self) -> "ComputerSet":
        from rocheometry._computers.defaults import _build_computers

        return _build_computers(self._settings())


@dataclass(frozen=True, slots=True)
class Fitted(_Accuracy):
    """Packaged fits with fast numerical fallbacks (the default)."""

    def _settings(self) -> "_Settings":  # type: ignore[missing-override-decorator]
        from rocheometry._computers.defaults import _Settings

        return _Settings()


@dataclass(frozen=True, slots=True, kw_only=True)
class Reference(_Accuracy):
    """Fully numerical reference computations.

    At default orders, integrated mean curvature and mean-curvature radius have
    a 2e-8 relative bound and every other scalar 1e-10;
    `Reference(polar_order=512, azimuth_order=192)` gives 1e-10 for all
    scalars. See [accuracy](https://pdhall99.github.io/rocheometry/accuracy/) for
    the outline bounds and the node contract.

    Args:
        polar_order: Polar quadrature order, an integer at least four.
        azimuth_order: Quarter-azimuth quadrature order, an integer at least four.
    """

    polar_order: int = 128
    azimuth_order: int = 64

    def __post_init__(self) -> None:
        for name in ("polar_order", "azimuth_order"):
            object.__setattr__(
                self, name, require_quadrature_order(getattr(self, name), name=name)
            )

    def _settings(self) -> "_Settings":  # type: ignore[missing-override-decorator]
        from rocheometry._computers.defaults import _Settings

        return _Settings(
            polar_order=self.polar_order,
            azimuth_order=self.azimuth_order,
            watershed_tolerance=1e-13,
            fitted_selection=False,
            fill_tolerance=1e-13,
            fixed_paths=False,
            path_tolerance=1e-10,
            exact_anchor_everywhere=True,
        )
