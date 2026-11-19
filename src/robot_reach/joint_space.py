"""Joint-space maps: the 2R picture, generalised.

WHY (Zach, 2026-11-19): "in 2RR I always like the picture view as you can clearly see what's
possible or not ... You can label solution / no solution ... singularity ... or collision".

A 2-DOF arm's joint space *is* the picture. For more joints, pick two and either

- **slice**: hold the others at `q0` (the picture at one posture), or
- **sweep**: sample the others and colour each cell by the fraction of samples that are OK
  ("you need to kinda sum across" — some cells work only for some postures of the rest).

Every cell gets the same `Code` vocabulary as the Cartesian map. A 1-DOF arm gets a line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arm import Arm
from .codes import Code


@dataclass(frozen=True, eq=False)
class JointMap:
    arm_name: str
    axes: tuple[int, ...]  # joint indices on the picture's axes (1 or 2)
    names: tuple[str, ...]
    grids: tuple[np.ndarray, ...]  # 1-D value arrays per axis, rad
    code: np.ndarray  # (n,) or (n, n): the slice's code, or the sweep's most common code
    ok_fraction: np.ndarray  # same shape: 1/0 for a slice, 0..1 for a sweep
    mode: str  # "slice" | "sweep"
    q0: np.ndarray  # (dof,) posture of the other joints (slice) / reference (sweep)
    torque: np.ndarray | None = None  # (n, dof) gravity torque along a 1-D map
    manipulability: np.ndarray | None = None  # same shape as code (slice only)

    def code_counts(self) -> dict[str, int]:
        vals, counts = np.unique(self.code, return_counts=True)
        return {Code(int(v)).name: int(c) for v, c in zip(vals, counts)}


def _default_q0(arm: Arm) -> np.ndarray:
    """The arm's declared home (`meta["home"]`), else zero clipped into the limits.
    WHY a home: q = 0 is often exactly singular (a UR's wrist at q5 = 0), which would paint the
    whole slice purple and hide everything else."""
    lim = arm.limits
    home = arm.meta.get("home")
    q0 = np.zeros(arm.dof) if home is None else np.asarray(home, float)
    return np.clip(q0, lim[:, 0], lim[:, 1])


def joint_map(
    arm: Arm,
    axes: tuple[int, ...] = (0, 1),
    *,
    n: int = 181,
    mode: str = "slice",
    q0: np.ndarray | None = None,
    n_sweep: int = 48,
    floor_z: float | None = None,
    singular_tol: float = 0.02,
    w_ref: float | None = None,
    pad_deg: float = 0.0,
    seed: int = 0,
) -> JointMap:
    """Classify a grid over one or two joints. `pad_deg` extends the axes past the limits so the
    limit itself is visible as a band."""
    axes = tuple(axes)
    q0 = _default_q0(arm) if q0 is None else np.asarray(q0, float)
    lim = arm.limits
    pad = np.deg2rad(pad_deg)
    grids = tuple(np.linspace(lim[a, 0] - pad, lim[a, 1] + pad, n) for a in axes)
    mesh = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, len(axes))
    if w_ref is None and singular_tol > 0:
        from .sampling import sample_joints

        w_ref = float(np.nanpercentile(arm.manipulability(sample_joints(arm, 2000, seed)), 95))

    if mode == "slice" or arm.dof == len(axes):
        Q = np.repeat(q0[None], len(mesh), axis=0)
        Q[:, list(axes)] = mesh
        code = arm.check(Q, floor_z=floor_z, singular_tol=singular_tol, w_ref=w_ref)
        shape = tuple(len(g) for g in grids)
        torque = arm.gravity_torque(Q) if len(axes) == 1 else None
        manip = arm.manipulability(Q).reshape(shape)
        return JointMap(
            arm_name=arm.name,
            axes=axes,
            names=tuple(arm.joint_names[a] for a in axes),
            grids=grids,
            code=code.reshape(shape),
            ok_fraction=(code == Code.OK).astype(float).reshape(shape),
            mode="slice",
            q0=q0,
            torque=torque,
            manipulability=manip,
        )

    # sweep: every cell x n_sweep random postures of the other joints
    rng = np.random.default_rng(seed)
    others = [j for j in range(arm.dof) if j not in axes]
    Q = np.empty((len(mesh), n_sweep, arm.dof))
    Q[:, :, others] = lim[others, 0] + rng.random((len(mesh), n_sweep, len(others))) * (lim[others, 1] - lim[others, 0])
    Q[:, :, list(axes)] = mesh[:, None, :]
    code = arm.check(Q.reshape(-1, arm.dof), floor_z=floor_z, singular_tol=singular_tol, w_ref=w_ref)
    code = code.reshape(len(mesh), n_sweep)
    ok_fraction = (code == Code.OK).mean(axis=1)
    # the cell's colour when nothing is OK: its most common failure
    codes = np.arange(-1, 8)
    counts = np.stack([(code == c).sum(axis=1) for c in codes], axis=1)
    counts[:, codes == Code.OK] = np.where(ok_fraction > 0, 10**9, 0)[:, None]
    dominant = codes[np.argmax(counts, axis=1)].astype(np.int8)
    shape = tuple(len(g) for g in grids)
    return JointMap(
        arm_name=arm.name,
        axes=axes,
        names=tuple(arm.joint_names[a] for a in axes),
        grids=grids,
        code=dominant.reshape(shape),
        ok_fraction=ok_fraction.reshape(shape),
        mode="sweep",
        q0=q0,
    )


@dataclass(frozen=True)
class JointRanges:
    """Per joint: the limits, and the range the OK IK solutions of a reach map actually use."""

    names: tuple[str, ...]
    limits: np.ndarray  # (dof, 2)
    used: np.ndarray | None  # (dof, 2) min/max over OK solutions, NaN if none
    hist_edges: np.ndarray | None  # (dof, B+1)
    hist: np.ndarray | None  # (dof, B) counts of OK solutions per bin


def joint_ranges(arm: Arm, reach=None, bins: int = 60) -> JointRanges:
    lim = arm.limits
    if reach is None or len(reach.sol_q) == 0:
        return JointRanges(arm.joint_names, lim, None, None, None)
    q = reach.sol_q.astype(float)
    used = np.stack([q.min(0), q.max(0)], axis=1)
    edges = np.stack([np.linspace(lo, hi, bins + 1) for lo, hi in lim])
    hist = np.stack([np.histogram(q[:, j], bins=edges[j])[0] for j in range(arm.dof)])
    return JointRanges(arm.joint_names, lim, used, edges, hist)
