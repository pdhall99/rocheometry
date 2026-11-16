"""Ownership callers share certificates and catch a dedicated error type."""

from types import SimpleNamespace
from typing import cast

import pytest

import rocheometry
from rocheometry._computers import ownership
from rocheometry._computers.watershed import WatershedComputer


def test_certificate_raises_public_ownership_error() -> None:
    geometry = rocheometry.ComponentGeometry(1.0)
    sheet = cast(WatershedComputer, SimpleNamespace(cut_height=lambda *args: 0.5))
    with pytest.raises(rocheometry.ComponentOwnershipError):
        ownership.certify_part(geometry.field, geometry.l1(), sheet, (0.6, 0.0, 0.0))


def test_membership_catches_type_without_matching_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = rocheometry.ComponentGeometry(1.0)

    def no_cut(*args: object) -> float:
        raise ValueError("outside cut chart")

    def rejected(*args: object, **kwargs: object) -> None:
        raise rocheometry.ComponentOwnershipError("different explanatory wording")

    sheet = cast(WatershedComputer, SimpleNamespace(cut_height=no_cut))
    monkeypatch.setattr(ownership, "certify_part", rejected)
    assert not ownership.belongs_to_component(
        geometry.field, geometry.l1(), sheet, (0.6, 0.0, 0.0)
    )
