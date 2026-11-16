"""Public exceptions share one stable module."""

import rocheometry
from rocheometry import _errors


def test_public_exception_exports() -> None:
    assert rocheometry.EmptySectionError is _errors.EmptySectionError
    assert rocheometry.DegenerateLineOfSightError is _errors.DegenerateLineOfSightError
    assert issubclass(rocheometry.EmptySectionError, ValueError)
    assert issubclass(rocheometry.DegenerateLineOfSightError, RuntimeError)
