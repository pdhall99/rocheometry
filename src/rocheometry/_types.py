"""Shared internal type aliases."""

from typing import Literal

ComponentId = Literal[1, 2]
"""The component number, either `1` or `2`."""

BinaryFrame = Literal["binary", "centre_of_mass"]
"""The binary origin or centre-of-mass output chart."""

Frame = Literal[BinaryFrame, "component1", "component2"]
"""A binary, centre-of-mass, or component-centred coordinate chart."""
