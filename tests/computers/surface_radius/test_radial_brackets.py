"""Enclosing brackets retain the stellar crossing inside a geometric bound."""

from rocheometry._computers.surface_radius.radial import _enclosing_ray_bracket


def test_enclosing_bracket_does_not_search_the_centrifugal_exterior() -> None:
    evaluations = []

    def residual(radius: float) -> float:
        evaluations.append(radius)
        # A very narrow outside interval that an expanding scan can skip.
        return (1 - radius) * (1.000001 - radius)

    inner, outer = _enclosing_ray_bracket(residual, 0.0, 1.0000005, 0.0)
    assert residual(inner) > 0 > residual(outer)
    assert inner < 1 < outer
    assert all(0 < r <= 1.0000005 for r in evaluations)


def test_enclosing_bracket_returns_an_exact_contracted_root() -> None:
    assert _enclosing_ray_bracket(lambda r: 1 - r, 0.0, 2.0, 0.0) == (1.0, 1.0)
