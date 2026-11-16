"""Exceptions exposed by the public geometry API."""


class EmptySectionError(ValueError):
    """A section whose polar centre is not strictly inside the selected surface."""


class DegenerateLineOfSightError(RuntimeError):
    """An unresolved stationary pair occupies adjacent representable positions."""


class ComponentOwnershipError(RuntimeError):
    """A point lies beyond the component's side of the watershed."""
