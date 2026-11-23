"""Joint-space maps: the 2R picture, generalised.

WHY (Zach, 2026-11-19): "in 2RR I always like the picture view as you can clearly see what's
possible or not ... You can label solution / no solution ... singularity ... or collision".

A 2-DOF arm's joint space *is* the picture. For more joints, pick two and either

- **slice**: hold the others at `q0` (the picture at one posture), or
- **sweep**: sample the others and colour each cell by the fraction of samples that are OK
  ("you need to kinda sum across" — some cells work only for some postures of the rest). The others
  are sampled uniformly over their full limits, or over a range you choose (`other_ranges`).

WHY the sweep keeps "always singular" / "always collision" apart (Zach, 2026-11-23: "Definitely show
the always collision between those two joints ... then do show the singularity ... and OK for all vs
OK for some ... kinda feels a bit like the collision matrix"): a cell that never works because the
two joints themselves collide is a different fact from one that fails only for some postures of the
rest. Worst first: singular, then collision, then the other failures, then OK for some, OK for all.

Joint labels carry the joint's place in the chain (`joint_label`: "shoulder_lift_joint (J2)"), so a
picture reads at a glance whatever the URDF called the joint (Zach, 2026-11-23).

Every cell gets the same `Code` vocabulary as the Cartesian map. A 1-DOF arm gets a line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arm import Arm
from .codes import Code


# Worst first: the failure a never-OK sweep cell is labelled with when its postures fail in mixed ways.
FAILURE_PRECEDENCE = (Code.SINGULAR, Code.SELF_COLLISION, Code.ENVIRONMENT_COLLISION, Code.COLLISION, Code.TORQUE_LIMIT, Code.FK_FAIL, Code.FK_MISMATCH, Code.JOINT_LIMIT, Code.NO_SOLUTION)


def joint_label(names, index: int) -> str:
    """``"shoulder_lift_joint (J2)"``: the URDF name plus the joint's 1-based place in the chain."""
    return f"{names[index]} (J{index + 1})"


def joint_labels(names) -> tuple[str, ...]:
    return tuple(joint_label(names, i) for i in range(len(names)))


def worst_failure(codes: np.ndarray) -> np.ndarray:
    """(cells, samples) codes -> (cells,) the worst failure present (`FAILURE_PRECEDENCE`), else NO_SOLUTION."""
    out = np.full(len(codes), int(Code.NO_SOLUTION), np.int8)
    for c in reversed(FAILURE_PRECEDENCE):
        out[np.any(codes == c, axis=1)] = c
    return out


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
    # sweep only: every sampled posture of the cell had this same `code` ("always collision"); None = unknown
    uniform: np.ndarray | None = None
    n_sweep: int = 0  # sweep only: postures of the other joints sampled per cell
    other_ranges: np.ndarray | None = None  # sweep only: (dof, 2) range each other joint was sampled over, rad

    @property
    def labels(self) -> tuple[str, ...]:
        """Axis labels with the chain index, ``("shoulder_lift_joint (J2)", "elbow_joint (J3)")``."""
        return tuple(f"{n} (J{a + 1})" for n, a in zip(self.names, self.axes))

    def sweep_text(self, joint_names=None, limits=None) -> str:
        """How a sweep was made, in words: which joints were sampled over which ranges, how many per cell.

        `joint_names` / `limits` (dof, 2) let it say "full limits" when a range is the whole limit."""
        if self.mode != "sweep":
            return ""
        if self.other_ranges is None:
            return "the other joints: sampled uniformly over their full limits (sample count not recorded)"
        parts = []
        for j, (lo, hi) in enumerate(self.other_ranges):
            if j in self.axes or not np.isfinite(lo):
                continue
            name = f"J{j + 1}" if joint_names is None else joint_label(joint_names, j)
            full = limits is not None and np.allclose([lo, hi], limits[j], atol=1e-9)
            span = f"{np.rad2deg(lo):.0f}…{np.rad2deg(hi):.0f}°"
            parts.append(f"{name} {span}" + (" (full limits)" if full else ""))
        return f"each cell = {self.n_sweep} postures of the other joints, uniform over: " + ", ".join(parts)

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
    other_ranges=None,
) -> JointMap:
    """Classify a grid over one or two joints. `pad_deg` extends the axes past the limits so the
    limit itself is visible as a band. `other_ranges` (sweep): ``{joint index: (lo, hi)}`` rad, or a
    (dof, 2) array, the range each other joint is sampled over (default its full limits; clipped to
    the limits)."""
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
    ranges = sweep_ranges(arm, axes, other_ranges)
    Q = np.empty((len(mesh), n_sweep, arm.dof))
    Q[:, :, others] = ranges[others, 0] + rng.random((len(mesh), n_sweep, len(others))) * (ranges[others, 1] - ranges[others, 0])
    Q[:, :, list(axes)] = mesh[:, None, :]
    flat = Q.reshape(-1, arm.dof)
    # WHY chunked: the singularity test builds a Jacobian per posture by FK of dof bumped copies, so a
    # 91 x 91 x 48 sweep in one call peaks above 1 GB; 20k postures at a time keeps it ~50 MB.
    code = np.concatenate([arm.check(flat[k : k + 20_000], floor_z=floor_z, singular_tol=singular_tol, w_ref=w_ref)
                           for k in range(0, len(flat), 20_000)])
    code = code.reshape(len(mesh), n_sweep)
    cell_code, ok_fraction, uniform = classify_sweep(code)
    shape = tuple(len(g) for g in grids)
    return JointMap(
        arm_name=arm.name,
        axes=axes,
        names=tuple(arm.joint_names[a] for a in axes),
        grids=grids,
        code=cell_code.reshape(shape),
        ok_fraction=ok_fraction.reshape(shape),
        mode="sweep",
        q0=q0,
        uniform=uniform.reshape(shape),
        n_sweep=int(n_sweep),
        other_ranges=ranges,
    )


def sweep_ranges(arm: Arm, axes, other_ranges=None) -> np.ndarray:
    """(dof, 2) the range each joint is sampled over in a sweep: the limits, narrowed by
    `other_ranges` (``{j: (lo, hi)}`` or (dof, 2)) and clipped to them. The picture's own axes get NaN."""
    lim = arm.limits.astype(float)
    out = lim.copy()
    if other_ranges is not None:
        items = other_ranges.items() if isinstance(other_ranges, dict) else enumerate(np.asarray(other_ranges, float))
        for j, (lo, hi) in items:
            lo, hi = sorted((float(lo), float(hi)))
            out[j] = (np.clip(lo, *lim[j]), np.clip(hi, *lim[j]))
    out[list(axes)] = np.nan
    return out


def classify_sweep(code: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cells, samples) posture codes -> per cell (code, ok_fraction, uniform).

    - code: OK when any posture works, else the worst failure present (`FAILURE_PRECEDENCE`:
      singular, then collision, ...).
    - uniform: every posture had exactly that code: "OK for all", "always singular", "always collision"."""
    code = np.asarray(code)
    ok_fraction = (code == Code.OK).mean(axis=1)
    cell = np.where(ok_fraction > 0, int(Code.OK), worst_failure(code)).astype(np.int8)
    uniform = np.all(code == cell[:, None], axis=1)
    return cell, ok_fraction, uniform


@dataclass(frozen=True)
class JointRanges:
    """Per joint: the limits, and the range the OK IK solutions of a reach map actually use."""

    names: tuple[str, ...]  # the URDF names; `joint_labels(names)` adds the chain index
    limits: np.ndarray  # (dof, 2)
    used: np.ndarray | None  # (dof, 2) min/max over OK solutions, NaN if none
    hist_edges: np.ndarray | None  # (dof, B+1)
    hist: np.ndarray | None  # (dof, B) counts of OK solutions per bin


BIN_DEG = 10.0  # joint-range histogram bin width


def range_edges(limits: np.ndarray, bin_deg: float = BIN_DEG) -> np.ndarray:
    """(dof, B+1) histogram edges: one lattice of `bin_deg` bins (aligned to 0) spanning every joint's
    limits, the same for each joint so the rows share one angle axis."""
    step = np.deg2rad(bin_deg)
    lo = np.floor(np.min(limits[:, 0]) / step - 1e-9) * step
    hi = np.ceil(np.max(limits[:, 1]) / step + 1e-9) * step
    edges = np.arange(lo, hi + step / 2, step)
    return np.tile(edges, (len(limits), 1))


def joint_ranges(arm: Arm, reach=None, bin_deg: float = BIN_DEG) -> JointRanges:
    """Per joint, the range the working IK solutions use and a histogram of their angles in `bin_deg`
    bins (Zach's pick, 2026-11-23: "B2 one histogram per joint that is clearer")."""
    lim = arm.limits
    if reach is None or len(reach.sol_q) == 0:
        return JointRanges(arm.joint_names, lim, None, None, None)
    return ranges_of(arm.joint_names, lim, reach.sol_q.astype(float), bin_deg)


def ranges_of(names, limits: np.ndarray, q: np.ndarray, bin_deg: float = BIN_DEG) -> JointRanges:
    """`JointRanges` of a set of working postures (the viewer re-runs it on a filtered map)."""
    if not len(q):
        return JointRanges(tuple(names), limits, None, None, None)
    used = np.stack([q.min(0), q.max(0)], axis=1)
    edges = range_edges(limits, bin_deg)
    hist = np.stack([np.histogram(q[:, j], bins=edges[j])[0] for j in range(q.shape[1])])
    return JointRanges(tuple(names), limits, used, edges, hist)
