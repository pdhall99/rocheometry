"""Compound paths preserve independent loops and open arcs as one value."""

import pickle

import pytest

import rocheometry
from tests._helpers import compound_paths


@pytest.mark.parametrize("value", ([], None, compound_paths()[0]))
def test_compound_requires_tuple(value: object) -> None:
    with pytest.raises(TypeError, match="paths: expected tuple"):
        rocheometry.CompoundBezierPath(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", (object(), compound_paths()[0].segments[0], ()))
def test_compound_requires_paths(value: object) -> None:
    with pytest.raises(TypeError, match=r"paths\[1\]: expected BezierPath"):
        rocheometry.CompoundBezierPath((compound_paths()[0], value))  # type: ignore[arg-type]


@pytest.mark.parametrize("count", (0, 1, 2))
def test_compound_value_and_pickle(count: int) -> None:
    items = compound_paths()[:count]
    compound = rocheometry.CompoundBezierPath(items)
    assert len(compound) == count
    assert tuple(compound) == items
    assert bool(compound) == bool(items)
    assert compound == rocheometry.CompoundBezierPath(items)
    assert hash(compound) == hash(rocheometry.CompoundBezierPath(items))
    assert pickle.loads(pickle.dumps(compound)) == compound
    assert not hasattr(compound, "sample")


def test_compound_bounds_use_exact_cubic_extrema() -> None:
    compound = rocheometry.CompoundBezierPath(compound_paths())
    assert compound.bounds == (-2.0, 0.0, 3.0, 6.0)
    assert compound_paths()[0].bounds[2] == 3.0  # Control points reach 4.
    with pytest.raises(ValueError, match="^an empty compound path has no bounds$"):
        _ = rocheometry.CompoundBezierPath().bounds


@pytest.mark.parametrize("count", (0, 1, 2))
def test_compound_transforms_and_reverses(count: int) -> None:
    items = compound_paths()[:count]
    compound = rocheometry.CompoundBezierPath(items)
    transform = rocheometry.AffineTransform2D(xx=-2, xy=0.5, yy=3, tx=5, ty=-1)
    assert compound.transformed(transform).paths == tuple(
        p.transformed(transform) for p in items
    )
    assert compound.reversed().paths == tuple(p.reversed() for p in reversed(items))
    assert compound.reversed().reversed() == compound
