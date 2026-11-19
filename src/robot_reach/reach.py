"""Cartesian reach map: every IK branch at every (position, orientation), with a code each.

Kept from bam_reach v1 (the July 2026 FRs) — these are the semantics, re-implemented batched:

- A solution counts only if IK says OK, it is inside the joint limits, FK of it lands back on the
  target (the round trip, "FR - Reach analysis preserves IK FK collision and consistency truth"),
  and nothing that was checked collides. Unchecked collision is unknown, never "free".
- Four workspace kinds are masks over ONE map ("FR - One dexterous reach map derives workspace
  masks"): reachable, 4-DOF, hemisphere, dexterous.
- Position-only arms get position-only targets ("FR - Lower DoF solvers share the reach-map
  contract"), never a false 6-DOF promise.

New in v2 (Zach, 2026-11-19: "You also need to support like having multiple solutions"): the map
keeps every IK branch (P, O, S), so the viewer can click through them, and it runs vectorised.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .arm import CHUNK, Arm
from .codes import Code
from .orientations import OrientationSet

SCHEMA = "robot_reach.reach_map.v1"


def _wrap_into_limits(q: np.ndarray, lower: np.ndarray, upper: np.ndarray, revolute: np.ndarray) -> np.ndarray:
    """Shift revolute joints by +-2pi when that lands them inside their limits."""
    q = q.copy()
    for shift in (2 * np.pi, -2 * np.pi):
        out = (q < lower) | (q > upper)
        moved = q + shift
        fix = out & revolute & (moved >= lower) & (moved <= upper)
        q[fix] = moved[fix]
    return q


def _rotation_error(Ra: np.ndarray, Rb: np.ndarray, task: str) -> np.ndarray:
    if task == "pose":
        cos = (np.trace(np.swapaxes(Ra, -1, -2) @ Rb, axis1=-2, axis2=-1) - 1) / 2
    elif task == "axis":
        cos = np.sum(Ra[..., :, 2] * Rb[..., :, 2], axis=-1)
    else:
        return np.zeros(Ra.shape[:-2])
    return np.arccos(np.clip(cos, -1.0, 1.0))


@dataclass(frozen=True, eq=False)
class ReachMap:
    arm_name: str
    joint_names: tuple[str, ...]
    task: str
    positions: np.ndarray  # (P, 3) in the arm base frame
    orientations: OrientationSet  # O orientations shared by every position
    code: np.ndarray  # (P, O, S) int8 final code per IK branch
    sol_index: np.ndarray  # (K,) flat index into (P, O, S) of every OK branch
    sol_q: np.ndarray  # (K, dof) float32 joints of every OK branch
    seconds: float = 0.0
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ shapes
    @property
    def n_positions(self) -> int:
        return self.code.shape[0]

    @property
    def n_orientations(self) -> int:
        return self.code.shape[1]

    @property
    def n_branches(self) -> int:
        return self.code.shape[2]

    # ------------------------------------------------------------------ projections
    @property
    def n_solutions(self) -> np.ndarray:
        """(P, O) count of distinct OK IK branches."""
        return np.sum(self.code == Code.OK, axis=2)

    @property
    def valid_mask(self) -> np.ndarray:
        """(P, O) at least one OK branch. Same field as bam_reach v1."""
        return self.n_solutions > 0

    @property
    def reach_scores(self) -> np.ndarray:
        """(P,) fraction of the orientation set reachable. Same field as bam_reach v1."""
        return self.valid_mask.mean(axis=1)

    @property
    def joint_solutions(self) -> np.ndarray:
        """(P, O, dof) the first OK branch per pose, NaN where none. bam_reach v1's shape, which
        robot_trajectories duck-types; use `solutions()` for every branch."""
        out = np.full((self.n_positions * self.n_orientations, len(self.joint_names)), np.nan)
        pose_of = self.sol_index // self.n_branches
        first = np.unique(pose_of, return_index=True)[1]
        out[pose_of[first]] = self.sol_q[first]
        return out.reshape(self.n_positions, self.n_orientations, -1)

    def score(self, kind: str = "dexterous") -> np.ndarray:
        """(P,) fraction of the kind's orientation subset that has an OK branch."""
        if self.task == "position" or kind == "reachable":
            return self.valid_mask.any(axis=1).astype(float)
        mask = getattr(self.orientations, kind)
        return self.valid_mask[:, mask].mean(axis=1)

    def workspace(self, kind: str = "reachable", threshold: float = 1.0) -> np.ndarray:
        """(P,) bool: points that belong to the kind. Thresholds default to 1.0 (all orientations
        of the subset), as in v1; lower it to forgive sampling at the boundary."""
        if kind == "reachable":
            return self.valid_mask.any(axis=1)
        if self.task == "position":
            return np.zeros(self.n_positions, bool)  # explicit: not claimed, not computed
        return self.score(kind) >= threshold - 1e-9

    def levels(self, threshold: float = 1.0) -> np.ndarray:
        """(P,) how many pose DOF the arm controls at each point: 0, 3, 4, 5 or 6."""
        lv = np.where(self.workspace("reachable"), 3, 0)
        if self.task != "position":
            lv = np.where(self.workspace("four_dof", threshold), 4, lv)
            lv = np.where(self.workspace("hemisphere", threshold), 5, lv)
            lv = np.where(self.workspace("dexterous", threshold), 6, lv)
        return lv

    def solutions(self, p: int) -> list[tuple[int, int, np.ndarray]]:
        """Every OK branch at position p: [(orientation index, branch index, q), ...]."""
        lo = p * self.n_orientations * self.n_branches
        hi = lo + self.n_orientations * self.n_branches
        a, b = np.searchsorted(self.sol_index, [lo, hi])
        out = []
        for flat, q in zip(self.sol_index[a:b], self.sol_q[a:b]):
            rest = flat - lo
            out.append((int(rest // self.n_branches), int(rest % self.n_branches), q.astype(float)))
        return out

    def code_counts(self) -> dict[str, int]:
        """How many branches ended on each code (PAD excluded)."""
        vals, counts = np.unique(self.code[self.code != Code.PAD], return_counts=True)
        return {Code(int(v)).name: int(c) for v, c in zip(vals, counts)}

    def target(self, p: int, o: int) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.orientations.R[o]
        T[:3, 3] = self.positions[p]
        return T

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        """Versioned, compressed, pickle-free (kept from v1)."""
        path = Path(path)
        if not path.name.endswith(".reach.npz"):
            path = path.with_name(path.name + ".reach.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema": SCHEMA,
            "arm_name": self.arm_name,
            "joint_names": list(self.joint_names),
            "task": self.task,
            "seconds": self.seconds,
            "meta": self.meta,
        }
        np.savez_compressed(
            path,
            meta=np.frombuffer(json.dumps(meta).encode(), np.uint8),
            positions=self.positions,
            R=self.orientations.R,
            tilt=self.orientations.tilt,
            roll=self.orientations.roll,
            approach=self.orientations.approach,
            code=self.code,
            sol_index=self.sol_index,
            sol_q=self.sol_q,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> ReachMap:
        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode())
            if meta.get("schema") != SCHEMA:
                raise ValueError(f"{path}: schema {meta.get('schema')!r}, expected {SCHEMA!r}")
            return cls(
                arm_name=meta["arm_name"],
                joint_names=tuple(meta["joint_names"]),
                task=meta["task"],
                positions=z["positions"],
                orientations=OrientationSet(R=z["R"], tilt=z["tilt"], roll=z["roll"], approach=z["approach"]),
                code=z["code"],
                sol_index=z["sol_index"],
                sol_q=z["sol_q"],
                seconds=float(meta.get("seconds", 0.0)),
                meta=meta.get("meta", {}),
            )


def reach_map(
    arm: Arm,
    positions: np.ndarray,
    orientations: OrientationSet | None = None,
    *,
    pos_tol: float = 1e-4,
    rot_tol: float = 1e-3,
    dedupe_tol: float = 1e-5,
    floor_z: float | None = None,
    singular_tol: float = 0.0,
    w_ref: float | None = None,
    progress=None,
) -> ReachMap:
    """IK every branch at every (position, orientation); classify each branch.

    `pos_tol` / `rot_tol` are the FK round-trip tolerances (m, rad). `singular_tol` (relative to
    `w_ref`, a typical manipulability) marks near-singular branches SINGULAR; 0 turns it off."""
    t0 = time.perf_counter()
    positions = np.asarray(positions, float).reshape(-1, 3)
    if orientations is None or arm.task == "position":
        orientations = OrientationSet.identity()
    P, O = len(positions), len(orientations)
    lower, upper = np.asarray(arm.lower, float), np.asarray(arm.upper, float)
    revolute = np.asarray(arm.meta.get("revolute", [True] * arm.dof), bool)
    torque_lim = None if arm.torque_limits is None else np.asarray(arm.torque_limits, float)

    codes_chunks, idx_chunks, q_chunks = [], [], []
    width = None
    n_targets = P * O
    for start in range(0, n_targets, CHUNK):
        flat = np.arange(start, min(start + CHUNK, n_targets))
        p_idx, o_idx = flat // O, flat % O
        T = np.zeros((len(flat), 4, 4))
        T[:, :3, :3] = orientations.R[o_idx]
        T[:, :3, 3] = positions[p_idx]
        T[:, 3, 3] = 1.0
        q, code = arm.inverse(T)
        n, S, dof = q.shape
        code = code.copy()
        live = code == Code.OK
        # 1. joint limits (after trying the +-2pi twin of each revolute joint)
        if live.any():
            qq = _wrap_into_limits(q[live], lower, upper, revolute)
            q[live] = qq
            inside = np.all((qq >= lower - 1e-9) & (qq <= upper + 1e-9), axis=1)
            sub = code[live]
            sub[~inside] = Code.JOINT_LIMIT
            code[live] = sub
        # 2. FK round trip
        live = code == Code.OK
        if live.any():
            T_fk, ok = arm.forward(q[live])
            tgt = np.repeat(T[:, None], S, axis=1)[live]
            perr = np.linalg.norm(T_fk[:, :3, 3] - tgt[:, :3, 3], axis=1)
            rerr = _rotation_error(T_fk[:, :3, :3], tgt[:, :3, :3], arm.task)
            bad = ~ok
            sub = code[live]
            sub[bad] = Code.FK_FAIL
            sub[~bad & ((perr > pos_tol) | (rerr > rot_tol))] = Code.FK_MISMATCH
            code[live] = sub
        # 3. duplicates: two branches that are the same joint vector count once
        if S > 1:
            for a in range(S):
                for b in range(a + 1, S):
                    both = (code[:, a] == Code.OK) & (code[:, b] == Code.OK)
                    if both.any():
                        same = both & (np.max(np.abs(q[:, a] - q[:, b]), axis=1) < dedupe_tol)
                        code[same, b] = Code.PAD
        # 4. collision, 5. torque, 6. singularity — only on survivors
        live = code == Code.OK
        if live.any():
            hit = arm.collides(q[live], floor_z=floor_z)
            if hit is not None:
                sub = code[live]
                sub[hit] = Code.COLLISION
                code[live] = sub
        live = code == Code.OK
        if live.any() and torque_lim is not None:
            tau = arm.gravity_torque(q[live])
            if tau is not None:
                sub = code[live]
                sub[np.any(np.abs(tau) > torque_lim, axis=1)] = Code.TORQUE_LIMIT
                code[live] = sub
        live = code == Code.OK
        if live.any():
            sing = arm.is_singular(q[live], singular_tol=singular_tol, w_ref=w_ref)
            if sing is not None:
                sub = code[live]
                sub[sing] = Code.SINGULAR
                code[live] = sub

        width = S if width is None else max(width, S)
        rows, cols = np.nonzero(code == Code.OK)
        idx_chunks.append(((flat[rows]) * S, cols, S))
        q_chunks.append(q[rows, cols].astype(np.float32))
        codes_chunks.append(code)
        if progress is not None:
            progress(min(start + CHUNK, n_targets), n_targets)

    S = width or 1
    code_all = np.full((n_targets, S), int(Code.PAD), np.int8)
    row = 0
    sol_index, sol_q = [], []
    for code, (base, cols, s), q in zip(codes_chunks, idx_chunks, q_chunks):
        n = len(code)
        code_all[row : row + n, : code.shape[1]] = code
        # re-index flat (target * s + branch) into the global width S
        sol_index.append((base // s) * S + cols)
        sol_q.append(q)
        row += n
    sol_index = np.concatenate(sol_index) if sol_index else np.zeros(0, np.int64)
    sol_q = np.concatenate(sol_q) if sol_q else np.zeros((0, arm.dof), np.float32)
    order = np.argsort(sol_index, kind="stable")
    return ReachMap(
        arm_name=arm.name,
        joint_names=arm.joint_names,
        task=arm.task,
        positions=positions,
        orientations=orientations,
        code=code_all.reshape(P, O, S),
        sol_index=sol_index[order].astype(np.int64),
        sol_q=sol_q[order],
        seconds=time.perf_counter() - t0,
        meta={"pos_tol": pos_tol, "rot_tol": rot_tol, "floor_z": floor_z, "singular_tol": singular_tol},
    )
