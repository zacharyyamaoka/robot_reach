"""Example arms with batched closed-form FK and IK: 1, 2, 3 and 6 DOF."""

from .planar import five_bar, pendulum, rr
from .spatial import shoulder3, ur5e

__all__ = ["pendulum", "rr", "five_bar", "shoulder3", "ur5e"]
