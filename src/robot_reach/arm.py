"""Arm: an arm is its FK and IK, passed in as pure functions.

WHY functions in, not a robot class (Zach, Jul 2026 and again 2026-11-19: "You pass in FK and IK as
pure functions"): any solver — closed form, a five-bar, Pinocchio, cuRobo, a learned model — plugs
in without this library importing it. The functions are *batched* numpy functions because speed is
the point ("super fast and easy to visualize"); `Arm.from_scalar` wraps one-pose-at-a-time
functions for convenience.

Conventions (the whole contract):

- ``fk(q)``: ``(N, dof) -> (N, 4, 4)`` homogeneous base->tool poses, or ``((N, 4, 4), ok)`` with a
  ``(N,)`` bool when FK can fail (a parallel linkage that cannot close).
- ``ik(T)``: ``(N, 4, 4) -> (q, code)`` with ``q`` ``(N, S, dof)`` and ``code`` ``(N, S)`` ints from
  `robot_reach.codes.Code`. **Return every branch** (S = the arm's solution count: 2 for an elbow,
  8 for a UR). An unused slot is ``Code.PAD``. The solver may report ``NO_SOLUTION`` or its own
  ``JOINT_LIMIT`` etc.; the library re-checks limits, the FK round trip, collision, torque and
  singularity itself, so a lazy solver cannot overclaim.
- The tool's local +Z is the approach axis (what "4-DOF" and "hemisphere" are measured against).
- Units: SI, joint angles in radians. Views show degrees.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .codes import Code

Task = Literal["pose", "axis", "position"]
"""What part of a target pose the arm is asked to match.

- ``pose``: position and full orientation (a 6-DOF arm).
- ``axis``: position and the tool +Z direction; roll is free (a 5-DOF arm).
- ``position``: position only (a 1-3 DOF arm, or a positioning shoulder).
"""

CHUNK = 20_000  # poses per vectorised IK/FK call; bounds peak memory on big maps


@dataclass(frozen=True, eq=False)
class Arm:
    name: str
    joint_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    fk: Callable[[np.ndarray], object]
    ik: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None
    task: Task = "pose"
    # Points to draw the arm as a chain (base, joints..., tool): (N, dof) -> (N, K, 3).
    skeleton: Callable[[np.ndarray], np.ndarray] | None = None
    # Point masses for the static gravity torque: (N, dof) -> (N, M, 3), with `masses` (M,) in kg.
    mass_points: Callable[[np.ndarray], np.ndarray] | None = None
    masses: tuple[float, ...] = ()
    torque_limits: tuple[float, ...] | None = None  # N·m per joint
    # Self/scene collision: (N, dof) -> (N,) bool. None = not checked (never assumed free).
    collision: Callable[[np.ndarray], np.ndarray] | None = None
    # Mechanism-specific singularity test, (N, dof) -> (N,) bool (e.g. a five-bar's transmission
    # angle). ORed with the generic manipulability test.
    singular: Callable[[np.ndarray], np.ndarray] | None = None
    # The arm's working plane for planar arms: a point and normal. Grids collapse onto it.
    plane: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    description: str = ""
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basics
    @property
    def dof(self) -> int:
        return len(self.joint_names)

    @property
    def limits(self) -> np.ndarray:
        """(dof, 2) [lower, upper]."""
        return np.stack([np.asarray(self.lower, float), np.asarray(self.upper, float)], axis=1)

    def within_limits(self, q: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        q = np.asarray(q, float)
        lo, hi = np.asarray(self.lower, float), np.asarray(self.upper, float)
        return np.all((q >= lo - tol) & (q <= hi + tol), axis=-1)

    # ------------------------------------------------------------------ FK / IK
    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, dof) -> (T (N,4,4), ok (N,)). Chunked; normalises both FK return styles."""
        q = np.atleast_2d(np.asarray(q, float))
        Ts, oks = [], []
        for start in range(0, len(q), CHUNK):
            out = self.fk(q[start : start + CHUNK])
            if isinstance(out, tuple):
                T, ok = out
                ok = np.asarray(ok, bool)
            else:
                T = out
                ok = np.all(np.isfinite(np.asarray(T, float).reshape(len(T), -1)), axis=1)
            Ts.append(np.asarray(T, float))
            oks.append(ok)
        if not Ts:
            return np.zeros((0, 4, 4)), np.zeros(0, bool)
        return np.concatenate(Ts), np.concatenate(oks)

    def inverse(self, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, 4, 4) -> (q (N, S, dof), code (N, S)). Chunked; pads every chunk to the same S."""
        if self.ik is None:
            raise ValueError(f"{self.name}: no IK function was given")
        T = np.asarray(T, float).reshape(-1, 4, 4)
        qs, codes = [], []
        for start in range(0, len(T), CHUNK):
            q, code = self.ik(T[start : start + CHUNK])
            q = np.asarray(q, float)
            code = np.asarray(code, np.int8)
            if q.ndim == 2:  # a single-branch solver returned (N, dof)
                q, code = q[:, None, :], code.reshape(-1, 1)
            qs.append(q)
            codes.append(code)
        width = max((c.shape[1] for c in codes), default=1)
        q_out = np.zeros((len(T), width, self.dof))
        c_out = np.full((len(T), width), int(Code.PAD), np.int8)
        row = 0
        for q, code in zip(qs, codes):
            n, s = code.shape
            q_out[row : row + n, :s] = q
            c_out[row : row + n, :s] = code
            row += n
        return q_out, c_out

    # ------------------------------------------------------------------ derived physics
    def jacobian(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Numeric geometric Jacobian, (N, 6, dof): rows 0-2 linear, rows 3-5 angular."""
        q = np.atleast_2d(np.asarray(q, float))
        n, dof = q.shape
        T0, _ = self.forward(q)
        bumped = np.repeat(q[:, None, :], dof, axis=1) + eps * np.eye(dof)[None]
        T1, _ = self.forward(bumped.reshape(-1, dof))
        T1 = T1.reshape(n, dof, 4, 4)
        lin = (T1[..., :3, 3] - T0[:, None, :3, 3]) / eps  # (N, dof, 3)
        dR = T1[..., :3, :3] @ np.swapaxes(T0[:, None, :3, :3], -1, -2)
        ang = np.stack([dR[..., 2, 1] - dR[..., 1, 2], dR[..., 0, 2] - dR[..., 2, 0], dR[..., 1, 0] - dR[..., 0, 1]], -1)
        ang = ang / (2 * eps)
        return np.swapaxes(np.concatenate([lin, ang], axis=-1), 1, 2)

    def manipulability(self, q: np.ndarray) -> np.ndarray:
        """Product of the Jacobian's task-relevant singular values (0 at a singularity). (N,)"""
        J = self.jacobian(q)
        if self.task == "position":
            J = J[:, :3]
        rank = min(J.shape[1], J.shape[2], self.task_dims)
        w = np.full(len(J), np.nan)  # NaN where FK failed nearby (a linkage at its closure edge)
        good = np.all(np.isfinite(J.reshape(len(J), -1)), axis=1)
        if good.any():
            w[good] = np.prod(np.linalg.svd(J[good], compute_uv=False)[:, :rank], axis=1)
        return w

    @property
    def task_dims(self) -> int:
        """How many task-space directions the arm can move in (planar arms: at most 2)."""
        base = {"pose": 6, "axis": 5, "position": 3}[self.task]
        if self.plane is not None and self.task == "position":
            base = 2
        return min(base, self.dof)

    def gravity_torque(self, q: np.ndarray, g: float = 9.81, eps: float = 1e-6) -> np.ndarray | None:
        """Static holding torque dV/dq for the point masses, (N, dof). None without masses."""
        if self.mass_points is None or not self.masses:
            return None
        q = np.atleast_2d(np.asarray(q, float))
        n, dof = q.shape
        m = np.asarray(self.masses, float)
        z0 = self.mass_points(q)[..., 2] @ m  # (N,)
        bumped = np.repeat(q[:, None, :], dof, axis=1) + eps * np.eye(dof)[None]
        z1 = (self.mass_points(bumped.reshape(-1, dof))[..., 2] @ m).reshape(n, dof)
        return g * (z1 - z0[:, None]) / eps

    def chain(self, q: np.ndarray) -> np.ndarray:
        """Points to draw, (N, K, 3). Falls back to base origin -> tool."""
        q = np.atleast_2d(np.asarray(q, float))
        if self.skeleton is not None:
            return np.asarray(self.skeleton(q), float)
        T, _ = self.forward(q)
        return np.stack([np.zeros((len(q), 3)), T[:, :3, 3]], axis=1)

    # ------------------------------------------------------------------ joint-space checks
    def check(
        self,
        q: np.ndarray,
        *,
        floor_z: float | None = None,
        singular_tol: float = 0.0,
        w_ref: float | None = None,
        torque: bool = True,
    ) -> np.ndarray:
        """Classify joint configurations, (N,) codes. First failing check wins, in this order:
        joint limit, FK fails, collision, torque, singular, else OK."""
        q = np.atleast_2d(np.asarray(q, float))
        code = np.full(len(q), int(Code.OK), np.int8)
        code[~self.within_limits(q)] = Code.JOINT_LIMIT
        _, ok = self.forward(q)
        code[(code == Code.OK) & ~ok] = Code.FK_FAIL
        live = code == Code.OK
        if live.any():
            hit = self.collides(q[live], floor_z=floor_z)
            if hit is not None:
                sub = code[live]
                sub[hit] = Code.COLLISION
                code[live] = sub
        live = code == Code.OK
        if torque and live.any() and self.torque_limits is not None:
            tau = self.gravity_torque(q[live])
            if tau is not None:
                over = np.any(np.abs(tau) > np.asarray(self.torque_limits, float), axis=1)
                sub = code[live]
                sub[over] = Code.TORQUE_LIMIT
                code[live] = sub
        live = code == Code.OK
        if live.any():
            sing = self.is_singular(q[live], singular_tol=singular_tol, w_ref=w_ref)
            if sing is not None:
                sub = code[live]
                sub[sing] = Code.SINGULAR
                code[live] = sub
        return code

    def is_singular(self, q: np.ndarray, *, singular_tol: float = 0.0, w_ref: float | None = None) -> np.ndarray | None:
        """(N,) bool from the arm's own test and/or relative manipulability; None if neither ran."""
        out = None
        if self.singular is not None:
            out = np.asarray(self.singular(q), bool)
        if singular_tol > 0 and w_ref:
            w = self.manipulability(q)
            low = ~(w >= singular_tol * w_ref)  # NaN (FK fails a hair away) counts as singular
            out = low if out is None else (out | low)
        return out

    def collides(self, q: np.ndarray, *, floor_z: float | None = None) -> np.ndarray | None:
        """(N,) bool, or None when nothing was checked (unknown is not 'free')."""
        hit = None
        if self.collision is not None:
            hit = np.asarray(self.collision(q), bool)
        if floor_z is not None:
            pts = self.chain(q)[:, 1:, :]  # the base point is fixed; only moving points can hit
            below = np.any(pts[..., 2] < floor_z - 1e-9, axis=1)
            hit = below if hit is None else (hit | below)
        return hit

    # ------------------------------------------------------------------ convenience
    @classmethod
    def from_scalar(
        cls,
        name: str,
        joint_names: Sequence[str],
        lower: Sequence[float],
        upper: Sequence[float],
        fk: Callable[[np.ndarray], object],
        ik: Callable[[np.ndarray], object] | None = None,
        **kwargs,
    ) -> Arm:
        """Wrap one-pose-at-a-time functions.

        ``fk(q) -> T`` (4x4), ``(T, ok)`` or None. ``ik(T) -> [q, ...]`` or ``[(q, code), ...]``
        (empty list = no solution)."""
        dof = len(joint_names)

        def fk_batch(Q: np.ndarray):
            Ts = np.full((len(Q), 4, 4), np.nan)
            ok = np.zeros(len(Q), bool)
            for i, q in enumerate(Q):
                out = fk(q)
                if isinstance(out, tuple):
                    out, good = out
                    if not good:
                        continue
                if out is None:
                    continue
                Ts[i] = out
                ok[i] = True
            return Ts, ok

        def ik_batch(T: np.ndarray):
            rows = []
            for pose in T:
                sols = list(ik(pose) or [])
                rows.append([(s if isinstance(s, tuple) else (s, Code.OK)) for s in sols])
            width = max(1, max((len(r) for r in rows), default=1))
            q = np.zeros((len(T), width, dof))
            code = np.full((len(T), width), int(Code.PAD), np.int8)
            for i, row in enumerate(rows):
                if not row:
                    code[i, 0] = Code.NO_SOLUTION
                for j, (qq, cc) in enumerate(row):
                    q[i, j] = qq
                    code[i, j] = int(cc)
            return q, code

        return cls(
            name=name,
            joint_names=tuple(joint_names),
            lower=tuple(float(v) for v in lower),
            upper=tuple(float(v) for v in upper),
            fk=fk_batch,
            ik=ik_batch if ik is not None else None,
            **kwargs,
        )
