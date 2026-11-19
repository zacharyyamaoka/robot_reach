"""The example gallery: a 1-DOF pendulum, 2-DOF RR, 2-DOF five-bar, 3-DOF shoulder and 6-DOF UR5e.

WHY these five (Zach, 2026-11-19: "Then please show example for 1 DOF arm (pendulum) 2 DOF arm (RR)
3 DOF arm (just shoulder) and 6 DOF arm"); the five-bar is added because it is BAM's own shoulder
and the one 2-DOF arm whose joint-space picture has a "cannot close" hole.
"""

from __future__ import annotations

from .robots import five_bar, pendulum, rr, shoulder3, ur5e

# name -> (arm factory, analyze() keyword arguments)
EXAMPLES: dict[str, tuple] = {
    "pendulum": (pendulum, {}),
    "rr": (rr, {"floor_z": -0.15}),
    "five_bar": (five_bar, {}),
    "shoulder3": (shoulder3, {"floor_z": 0.0}),
    "ur5e": (ur5e, {"floor_z": 0.0}),
}
