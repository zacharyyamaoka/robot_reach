"""Status codes: the one vocabulary every view colours by.

An IK or FK function may always return *an* answer plus a code (Zach, 2026-11-19: "the IK can
basically always return an answer and like an error code"). The library adds the codes it can
check itself (joint limits, FK round trip, singularity, torque, collision), so a joint-space
picture and a Cartesian map say "why not" in the same words and the same colours.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class Code(IntEnum):
    """Why a configuration or a solution is (not) usable. `OK` is the only good one."""

    PAD = -1  # an unused solution slot (an arm with fewer branches at this target)
    OK = 0
    NO_SOLUTION = 1  # IK found nothing: the target is out of reach
    JOINT_LIMIT = 2
    SINGULAR = 3
    COLLISION = 4
    TORQUE_LIMIT = 5  # static gravity torque exceeds a joint's limit
    FK_MISMATCH = 6  # FK(IK(x)) does not land back on x: the solver lied
    FK_FAIL = 7  # FK has no answer (a parallel linkage that cannot close)


# WHY white for "nothing here" (Zach, 2026-11-19 daily note: "white to a solid color ... so it's
# clear white means empty"): empty / impossible regions are near-white, problems are the only
# saturated warm colours, OK is a solid green.
CODE_COLORS: dict[Code, tuple[int, int, int]] = {
    Code.PAD: (255, 255, 255),
    Code.OK: (46, 160, 67),
    Code.NO_SOLUTION: (242, 242, 242),
    Code.JOINT_LIMIT: (230, 230, 230),
    Code.SINGULAR: (124, 77, 255),
    Code.COLLISION: (220, 53, 69),
    Code.TORQUE_LIMIT: (245, 159, 0),
    Code.FK_MISMATCH: (214, 51, 132),
    Code.FK_FAIL: (205, 205, 205),
}

CODE_LABELS: dict[Code, str] = {
    Code.PAD: "unused slot",
    Code.OK: "OK",
    Code.NO_SOLUTION: "no IK solution",
    Code.JOINT_LIMIT: "outside joint limits",
    Code.SINGULAR: "singular",
    Code.COLLISION: "collision",
    Code.TORQUE_LIMIT: "over torque limit",
    Code.FK_MISMATCH: "FK(IK) mismatch",
    Code.FK_FAIL: "linkage cannot close",
}


def code_rgb(codes: np.ndarray) -> np.ndarray:
    """(...) int codes -> (..., 3) uint8 colours."""
    codes = np.asarray(codes)
    out = np.empty(codes.shape + (3,), dtype=np.uint8)
    for code, rgb in CODE_COLORS.items():
        out[codes == int(code)] = rgb
    return out


# Workspace levels: how many of the six pose DOF the arm controls at a point.
# WHY counted in DOF (Zach, 2026-11-19: "the number of wrist DOFs you can achieve at each of those
# points you indicate with color"): 3 = position only, 4 = straight-down approach at any yaw,
# 5 = any approach in the hemisphere, 6 = every sampled orientation (dexterous).
LEVEL_NAMES: dict[int, str] = {
    0: "unreachable",
    3: "reachable",
    4: "4-DOF",
    5: "hemisphere",
    6: "dexterous",
}

# A white-to-solid single-hue ramp (light teal -> deep blue), one step per level.
LEVEL_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 255, 255),
    3: (161, 218, 180),
    4: (65, 182, 196),
    5: (34, 94, 168),
    6: (12, 44, 132),
}


def level_rgb(levels: np.ndarray) -> np.ndarray:
    levels = np.asarray(levels)
    out = np.full(levels.shape + (3,), 255, dtype=np.uint8)
    for level, rgb in LEVEL_COLORS.items():
        out[levels == level] = rgb
    return out


def ramp_rgb(values: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """Continuous 0..vmax -> white..deep blue, for scores and solution counts."""
    values = np.asarray(values, dtype=float)
    top = float(np.nanmax(values)) if vmax is None else float(vmax)
    t = np.clip(values / top, 0.0, 1.0) if top > 0 else np.zeros_like(values)
    stops = np.array([[247, 251, 255], [161, 218, 180], [65, 182, 196], [34, 94, 168], [12, 44, 132]], float)
    x = t * (len(stops) - 1)
    i = np.clip(np.floor(x).astype(int), 0, len(stops) - 2)
    f = (x - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)
