"""Workspace blobs: the outer surface of a set of grid voxels.

WHY voxel surfaces, not convex hulls (Zach, 2026-11-19: a convex hull "may not account for holes
within the workspace ... but I guess you could have multiple bodies"): the boundary faces of the
occupied voxels keep every hole, cavity and separate body exactly, with no dependency.
"""

from __future__ import annotations

import numpy as np

_DIRS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])


def lattice_index(points: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Integer cell of each grid point, and the origin they are measured from."""
    origin = points.min(0)
    return np.round((points - origin) / step).astype(np.int64), origin


def voxel_surface(points: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """(V, 3) vertices and (F, 3) faces of the boundary of cubes of side `step` at `points`."""
    if len(points) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    idx, origin = lattice_index(points, step)
    occupied = {tuple(i) for i in idx}
    h = step / 2
    out_v, out_f, offset = [], [], 0
    for d in _DIRS:
        open_side = np.array([tuple(n) not in occupied for n in idx + d])
        if not open_side.any():
            continue
        centres = origin + idx[open_side] * step
        axis = int(np.nonzero(d)[0][0])
        u, v = [a for a in range(3) if a != axis]
        corners = np.zeros((4, 3))
        corners[:, axis] = d[axis] * h
        corners[:, u] = [-h, h, h, -h]
        corners[:, v] = [-h, -h, h, h]
        if (d[axis] < 0) != (axis == 1):  # outward winding (the y axis flips the u-v handedness)
            corners = corners[::-1]
        quad = (centres[:, None, :] + corners[None]).reshape(-1, 3)
        k = offset + 4 * np.arange(len(centres))[:, None]
        out_f.append(np.concatenate([k + [0, 1, 2], k + [0, 2, 3]]))
        out_v.append(quad)
        offset += len(quad)
    return np.concatenate(out_v), np.concatenate(out_f)


def count_bodies(points: np.ndarray, step: float) -> int:
    """How many separate 6-connected pieces the voxels form."""
    if len(points) == 0:
        return 0
    idx, _ = lattice_index(points, step)
    remaining = {tuple(i) for i in idx}
    bodies = 0
    while remaining:
        bodies += 1
        stack = [remaining.pop()]
        while stack:
            c = stack.pop()
            for d in _DIRS:
                n = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
                if n in remaining:
                    remaining.remove(n)
                    stack.append(n)
    return bodies
