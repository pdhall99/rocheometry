"""Critical-point values retain the potential-point interface and Hessian spectrum."""

import pickle
from dataclasses import FrozenInstanceError, replace
from math import inf, nan
from typing import cast

import pytest

import rocheometry
from rocheometry._geometry import Point3D

HESSIAN = ((17.0, 0.0, 0.0), (0.0, -7.0, 0.0), (0.0, 0.0, -8.0))


def test_value_interface_equality_repr_pickle_and_immutability() -> None:
    point = rocheometry.CriticalPoint(
        (0.5, 0, 0), rocheometry.Potential.jacobi(2), HESSIAN
    )
    assert isinstance(point, rocheometry.CriticalPoint)
    assert point.position == (0.5, 0.0, 0.0)
    assert point.potential == rocheometry.Potential.jacobi(2)
    assert point.eigenvalues == (-8.0, -7.0, 17.0)
    assert point.morse_index == 2
    assert pickle.loads(pickle.dumps(point)) == point
    assert hash(pickle.loads(pickle.dumps(point))) == hash(point)
    assert point == rocheometry.CriticalPoint(point.position, point.potential, HESSIAN)
    assert point != replace(point, hessian=((18, 0, 0), (0, -7, 0), (0, 0, -8)))
    assert repr(point) == (
        "CriticalPoint(position=(0.5, 0.0, 0.0), "
        f"potential={point.potential!r}, hessian={HESSIAN!r})"
    )
    assert all(type(value) is float for row in point.hessian for value in row)
    with pytest.raises(FrozenInstanceError):
        setattr(point, "hessian", HESSIAN)


def test_coupled_block_and_strictly_negative_morse_count() -> None:
    point = rocheometry.CriticalPoint(
        (0.5, 1, 0),
        rocheometry.Potential.jacobi(2),
        ((2, 1, 0), (1, 2, 0), (0, 0, -1)),
    )
    assert point.eigenvalues == (-1, 1, 3)
    assert point.morse_index == 1
    zero = replace(point, hessian=((0, 0, 0), (0, 1, 0), (0, 0, -1)))
    assert zero.eigenvalues == (-1, 0, 1)
    assert zero.morse_index == 1


def test_discriminant_avoids_overflow_when_squaring_entries_would_overflow() -> None:
    point = rocheometry.CriticalPoint(
        (0.5, 1, 0),
        rocheometry.Potential.jacobi(2),
        ((1e200, 1e200, 0), (1e200, 1e200, 0), (0, 0, -1e200)),
    )
    assert point.eigenvalues == (-1e200, 0, 2e200)


@pytest.mark.parametrize(
    "hessian,error,match",
    [
        ([], TypeError, "hessian: expected tuple"),
        (((1, 0, 0),), ValueError, "three rows"),
        (((1, 0), (0, 1, 0), (0, 0, 1)), ValueError, r"hessian\[0\]"),
        (((1, 1, 0), (0, 1, 0), (0, 0, 1)), ValueError, "symmetric"),
        (((1, 0, 1), (0, 1, 0), (1, 0, 1)), ValueError, "orbital symmetry"),
        (((1, 0, 0), (0, 1, 1), (0, 1, 1)), ValueError, "orbital symmetry"),
        (((inf, 0, 0), (0, 1, 0), (0, 0, 1)), ValueError, "finite"),
        (((nan, 0, 0), (0, 1, 0), (0, 0, 1)), ValueError, "finite"),
        (((True, 0, 0), (0, 1, 0), (0, 0, 1)), TypeError, "real number"),
    ],
)
def test_hessian_validation(
    hessian: object, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        rocheometry.CriticalPoint(
            (0.5, 0, 0),
            rocheometry.Potential.jacobi(2),
            cast(tuple[Point3D, Point3D, Point3D], hessian),
        )


def test_critical_point_requires_canonical_potential() -> None:
    with pytest.raises(ValueError, match="canonical Jacobi"):
        rocheometry.CriticalPoint(
            (0.5, 0, 0), rocheometry.Potential.kopal(4, component_id=1), HESSIAN
        )
