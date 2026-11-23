"""Status codes: the one vocabulary every view colours by.

An IK or FK function may always return *an* answer plus a code (Zach, 2026-11-19: "the IK can
basically always return an answer and like an error code"). The library adds the codes it can
check from the maths alone (joint limits, the FK round trip, singularity, torque), so a joint-space
picture and a Cartesian map say "why not" in the same words and the same colours.

**Collision is your FK/IK's job, not this library's** (Zach, 2026-11-23: "reach doesn't really know
anything about collision. it just knows kinda like success or failure and then failure reasons").
Your IK returns ``SELF_COLLISION`` / ``ENVIRONMENT_COLLISION`` (or plain ``COLLISION``) for a branch,
your FK returns them per posture (``fk(q) -> (T, code)``); the library only enumerates and colours
the reasons it is handed. See `COLLISIONS`.
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
    # Returned by YOUR IK/FK (or the optional `Arm.collision` adapter), never computed here:
    SELF_COLLISION = 8  # two links of the arm touch
    ENVIRONMENT_COLLISION = 9  # the arm touches the world (a table, a wall, the optional floor plane)
    EXCLUDED = 10  # a working solution of an IK configuration the viewer's filter excludes (never stored)


# Every collision reason: `COLLISION` (unspecified), self and environment.
COLLISIONS: tuple[Code, ...] = (Code.COLLISION, Code.SELF_COLLISION, Code.ENVIRONMENT_COLLISION)


def is_collision(codes) -> np.ndarray:
    """(...) bool: the code is any collision reason (`COLLISIONS`)."""
    return np.isin(np.asarray(codes), [int(c) for c in COLLISIONS])


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
    Code.SELF_COLLISION: (176, 30, 45),
    Code.ENVIRONMENT_COLLISION: (240, 110, 90),
    Code.EXCLUDED: (236, 236, 236),
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
    Code.SELF_COLLISION: "self-collision",
    Code.ENVIRONMENT_COLLISION: "environment collision",
    Code.EXCLUDED: "excluded IK configuration",
}


def code_rgb(codes: np.ndarray) -> np.ndarray:
    """(...) int codes -> (..., 3) uint8 colours."""
    codes = np.asarray(codes)
    out = np.empty(codes.shape + (3,), dtype=np.uint8)
    for code, rgb in CODE_COLORS.items():
        out[codes == int(code)] = rgb
    return out


# Workspace levels. Internally an int per point (0, 3, 4, 5, 6: how many of the six pose DOF the arm
# controls there), shown to people by NAME only.
# WHY no DOF numbers in labels (Zach, 2026-11-23: "What do you m 3 dof+ 4 dof + ... how is hemisphere
# 5+ dof? thats not the meaning of thoose workspaces"): the names say what each one means.
# Nested: each includes the next (a dexterous point is also hemisphere, top-down and reachable).
LEVEL_NAMES: dict[int, str] = {
    0: "unreachable",
    3: "reachable",
    4: "top-down",
    5: "hemisphere",
    6: "dexterous",
}

LEVEL_MEANINGS: dict[int, str] = {
    0: "no orientation works",
    3: "at least one orientation works",
    4: "straight down along the approach axis, at every yaw",
    5: "every approach in the hemisphere around the approach axis, every roll",
    6: "every sampled orientation",
}

LEVEL_KEYS: dict[int, str] = {3: "reachable", 4: "four_dof", 5: "hemisphere", 6: "dexterous"}


# WHY these maps (Zach, 2026-11-23: "I like you can choose between viridis or magma ... start it with
# defaults but give people the option to just change"): matplotlib's perceptually uniform sequential maps,
# 17 stops each sampled from matplotlib 3.11 (no matplotlib at runtime: the core stays numpy only).
COLORMAPS: dict[str, tuple[tuple[int, int, int], ...]] = {
    'viridis': ((68, 1, 84), (72, 24, 106), (71, 45, 123), (66, 64, 134), (59, 82, 139), (51, 99, 141), (44, 114, 142), (38, 130, 142), (33, 145, 140), (31, 160, 136), (40, 174, 128), (63, 188, 115), (94, 201, 98), (132, 212, 75), (173, 220, 48), (216, 226, 25), (253, 231, 37)),
    'magma': ((0, 0, 4), (10, 8, 34), (29, 17, 71), (54, 16, 107), (81, 18, 124), (106, 28, 129), (131, 38, 129), (156, 46, 127), (183, 55, 121), (208, 65, 111), (231, 82, 99), (245, 107, 92), (252, 137, 97), (254, 167, 114), (254, 196, 136), (253, 226, 163), (252, 253, 191)),
    'plasma': ((13, 8, 135), (49, 5, 151), (76, 2, 161), (102, 0, 167), (126, 3, 168), (149, 17, 161), (170, 35, 149), (188, 53, 135), (204, 71, 120), (218, 90, 106), (230, 108, 92), (240, 128, 78), (248, 149, 64), (253, 172, 51), (253, 197, 39), (248, 223, 37), (240, 249, 33)),
    'inferno': ((0, 0, 4), (11, 7, 36), (33, 12, 74), (61, 9, 101), (87, 16, 110), (113, 25, 110), (138, 34, 106), (163, 44, 97), (188, 55, 84), (210, 70, 68), (228, 90, 49), (241, 115, 29), (249, 142, 9), (252, 172, 17), (249, 203, 53), (242, 234, 105), (252, 255, 164)),
    'cividis': ((0, 34, 78), (0, 46, 106), (26, 56, 111), (50, 67, 109), (67, 78, 108), (83, 90, 109), (97, 101, 111), (111, 112, 115), (125, 124, 120), (140, 136, 120), (155, 148, 118), (171, 160, 114), (188, 174, 108), (205, 187, 99), (222, 201, 88), (240, 216, 70), (254, 232, 56)),
}
DEFAULT_COLORMAP = "viridis"

# Categorical good / partial / bad: a traffic light (Zach, 2026-11-23: "for some like Good or Bad I like
# the traffic light").
TRAFFIC = {"good": (46, 160, 67), "partial": (240, 170, 0), "bad": (214, 40, 40)}


def colormap_rgb(values: np.ndarray, vmin: float = 0.0, vmax: float = 1.0, name: str = DEFAULT_COLORMAP,
                 reverse: bool = False) -> np.ndarray:
    """(...) values -> (..., 3) uint8 through a `COLORMAPS` map; `reverse` flips it (high = dark)."""
    stops = np.asarray(COLORMAPS[name], float)
    if reverse:
        stops = stops[::-1]
    values = np.asarray(values, float)
    span = vmax - vmin
    t = np.clip((values - vmin) / span, 0.0, 1.0) if span > 0 else np.ones_like(values)
    t = np.nan_to_num(t)
    x = t * (len(stops) - 1)
    i = np.clip(np.floor(x).astype(int), 0, len(stops) - 2)
    f = (x - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).round().astype(np.uint8)


def level_colors(name: str = DEFAULT_COLORMAP, reverse: bool = False) -> dict[int, tuple[int, int, int]]:
    """One colour per workspace level, from a colormap (reachable low ... dexterous high)."""
    rgb = colormap_rgb(np.array([0.0, 1 / 3, 2 / 3, 1.0]), name=name, reverse=reverse)
    return {k: tuple(int(c) for c in v) for k, v in zip((3, 4, 5, 6), rgb)}


LEVEL_COLORS: dict[int, tuple[int, int, int]] = {0: (255, 255, 255), **level_colors()}


def level_rgb(levels: np.ndarray, colors: dict | None = None) -> np.ndarray:
    levels = np.asarray(levels)
    out = np.full(levels.shape + (3,), 255, dtype=np.uint8)
    for level, rgb in (colors or LEVEL_COLORS).items():
        out[levels == level] = rgb
    return out


def ramp_rgb(values: np.ndarray, vmax: float | None = None, name: str = DEFAULT_COLORMAP, reverse: bool = False) -> np.ndarray:
    """Continuous 0..vmax through a colormap (default viridis), for scores and solution counts."""
    values = np.asarray(values, dtype=float)
    top = float(np.nanmax(values)) if vmax is None else float(vmax)
    return colormap_rgb(values, 0.0, top, name=name, reverse=reverse)
