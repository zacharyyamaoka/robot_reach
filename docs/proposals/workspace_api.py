"""Proposal G (prototype, not src/): the workspace a downstream consumer samples from.

The recommended shape (G2 in the report): one frozen value, `ReachableWorkspace`, that a trajectory
generator can hold without importing robot_reach (it would live in robot_msgs; numpy only), made
by robot_reach (`workspace_from_result`), with the verbs Zach asked for:

    ws = workspace_from_result(result, "hemisphere", types=UR5E_BRANCHES.allowed(elbow="up"))
    ws.contains(p) · ws.region(box=...) · ws.sample(n, rng) · ws.sample_pairs(n, rng)
    ws.nearest_ok_pose(T) · ws.voxels() · ws.save(path) / ReachableWorkspace.load(path)

Every pose `sample` returns is one the analysis actually checked (a grid point x an orientation
from the shared set) with its joint solution, so nothing is re-solved and nothing is promised that
was not computed. WHY discrete: a pose between grid points was never checked; `jitter` would need
the arm's IK again, which a consumer of a saved workspace may not have.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KINDS = ("reachable", "four_dof", "hemisphere", "dexterous")


@dataclass(frozen=True)
class Cylinder:
    """Upright cylinder in the arm base frame: a belt zone seen from above is often round-ish."""

    center_xy: tuple[float, float]
    radius: float
    z: tuple[float, float]

    def contains(self, p: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(p)
        r = np.hypot(p[:, 0] - self.center_xy[0], p[:, 1] - self.center_xy[1])
        return (r <= self.radius) & (p[:, 2] >= self.z[0]) & (p[:, 2] <= self.z[1])


@dataclass(frozen=True)
class AxisBox:
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]

    def contains(self, p: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(p)
        return np.all((p >= np.asarray(self.lo)) & (p <= np.asarray(self.hi)), axis=1)


@dataclass(frozen=True)
class Samples:
    T: np.ndarray  # (n, 4, 4) tool poses in the base frame
    q: np.ndarray  # (n, dof) a joint solution that reaches each
    branch_type: np.ndarray  # (n,) IKBranches type index of that solution
    position_index: np.ndarray  # (n,) into ws.positions
    orientation_index: np.ndarray  # (n,) into ws.R


@dataclass(frozen=True, eq=False)
class ReachableWorkspace:
    """One workspace kind of one arm, as plain arrays. Everything a sampler needs; nothing to solve."""

    arm: str
    kind: str  # reachable | four_dof | hemisphere | dexterous
    frame: str  # the arm base frame the numbers are in
    joint_names: tuple[str, ...]
    positions: np.ndarray  # (P, 3) member grid points
    step: np.ndarray  # (3,) voxel size, m
    R: np.ndarray  # (O, 3, 3) the orientations this kind asks for
    ok: np.ndarray  # (P, O) bool: an allowed-type solution exists
    sol_po: np.ndarray  # (K,) flat p * O + o of every allowed solution, sorted
    sol_q: np.ndarray  # (K, dof)
    sol_type: np.ndarray  # (K,) branch type
    settings: dict = field(default_factory=dict)  # how it was computed (F's meta), carried along

    # ---------------------------------------------------------------- membership
    def contains(self, p: np.ndarray) -> np.ndarray:
        """(N,) bool: the voxel holding each point is in this workspace."""
        p = np.atleast_2d(np.asarray(p, float))
        d = np.abs(p[:, None, :] - self.positions[None, :, :]) <= self.step / 2 + 1e-9
        return np.any(np.all(d, axis=2), axis=1)

    def region(self, shape) -> ReachableWorkspace:
        """The part of the workspace inside `shape` (anything with contains(points) -> bool)."""
        keep = np.flatnonzero(shape.contains(self.positions))
        O = len(self.R)
        p_of = self.sol_po // O
        new_index = np.full(len(self.positions), -1)
        new_index[keep] = np.arange(len(keep))
        s = np.isin(p_of, keep)
        po = new_index[p_of[s]] * O + self.sol_po[s] % O
        order = np.argsort(po, kind="stable")
        return ReachableWorkspace(
            self.arm, self.kind, self.frame, self.joint_names, self.positions[keep], self.step, self.R,
            self.ok[keep], po[order], self.sol_q[s][order], self.sol_type[s][order],
            {**self.settings, "region": repr(shape)},
        )

    # ---------------------------------------------------------------- sampling
    def _pick(self, po: np.ndarray, rng: np.random.Generator, types: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """For each flat (p, o), one solution (random among the allowed types there)."""
        lo = np.searchsorted(self.sol_po, po, "left")
        hi = np.searchsorted(self.sol_po, po, "right")
        idx = lo + (rng.random(len(po)) * (hi - lo)).astype(int)
        if types is not None:  # keep the requested type where it exists at that pose
            for i, t in enumerate(types):
                same = np.flatnonzero(self.sol_type[lo[i] : hi[i]] == t)
                if len(same):
                    idx[i] = lo[i] + same[rng.integers(len(same))]
        return self.sol_q[idx], self.sol_type[idx]

    def _pose(self, p: np.ndarray, o: np.ndarray) -> np.ndarray:
        T = np.zeros((len(p), 4, 4))
        T[:, :3, :3] = self.R[o]
        T[:, :3, 3] = self.positions[p]
        T[:, 3, 3] = 1.0
        return T

    def sample(self, n: int, rng: np.random.Generator) -> Samples:
        """n checked poses, uniform over the (position, orientation) pairs that work."""
        pairs = np.unique(self.sol_po)
        po = pairs[rng.integers(len(pairs), size=n)]
        p, o = po // len(self.R), po % len(self.R)
        q, t = self._pick(po, rng)
        return Samples(self._pose(p, o), q, t, p, o)

    def sample_pairs(self, n: int, rng: np.random.Generator, same_type: bool = True) -> tuple[Samples, Samples]:
        """A -> B pairs. same_type: B uses A's branch type where B has one (no flip of the arm
        between them; the trajectory generator decides the path)."""
        a = self.sample(n, rng)
        pairs = np.unique(self.sol_po)
        po_b = pairs[rng.integers(len(pairs), size=n)]
        if same_type:  # redraw B until it has A's type (bounded)
            for _ in range(20):
                lo = np.searchsorted(self.sol_po, po_b, "left")
                hi = np.searchsorted(self.sol_po, po_b, "right")
                has = np.array([np.any(self.sol_type[l:h] == t) for l, h, t in zip(lo, hi, a.branch_type)])
                if has.all():
                    break
                po_b[~has] = pairs[rng.integers(len(pairs), size=int((~has).sum()))]
        p, o = po_b // len(self.R), po_b % len(self.R)
        q, t = self._pick(po_b, rng, a.branch_type if same_type else None)
        return a, Samples(self._pose(p, o), q, t, p, o)

    def nearest_ok_pose(self, T: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        """The checked pose nearest a wanted one: (T_ok, q, distance m, rotation deg). Position first
        (the 8 nearest member voxels), then the orientation closest to the wanted one."""
        d = np.linalg.norm(self.positions - T[:3, 3], axis=1)
        best = None
        for p in np.argsort(d)[:8]:
            os_ = np.flatnonzero(self.ok[p])
            if not len(os_):
                continue
            cos = (np.trace(np.swapaxes(self.R[os_], 1, 2) @ T[:3, :3], axis1=1, axis2=2) - 1) / 2
            ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            o = os_[np.argmin(ang)]
            key = (round(d[p], 6), ang.min())
            if best is None or key < best[0]:
                best = (key, p, o)
        (_, ang), p, o = best
        q, _ = self._pick(np.array([p * len(self.R) + o]), np.random.default_rng(0))
        return self._pose(np.array([p]), np.array([o]))[0], q[0], float(d[p]), float(ang)

    # ---------------------------------------------------------------- export
    def voxels(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(mask (nx, ny, nz) bool, origin (3,), step (3,)): a dense grid for occupancy users."""
        origin = self.positions.min(0)
        idx = np.rint((self.positions - origin) / self.step).astype(int)
        mask = np.zeros(idx.max(0) + 1, bool)
        mask[tuple(idx.T)] = True
        return mask, origin, self.step

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        head = {"schema": "robot_msgs/reachable-workspace/0-proposal", "arm": self.arm, "kind": self.kind,
                "frame": self.frame, "joint_names": list(self.joint_names), "settings": self.settings}
        np.savez_compressed(path, head=np.frombuffer(json.dumps(head).encode(), np.uint8), positions=self.positions,
                            step=self.step, R=self.R, ok=self.ok, sol_po=self.sol_po, sol_q=self.sol_q, sol_type=self.sol_type)
        return path

    @classmethod
    def load(cls, path: str | Path) -> ReachableWorkspace:
        with np.load(path, allow_pickle=False) as z:
            h = json.loads(bytes(z["head"]).decode())
            return cls(h["arm"], h["kind"], h["frame"], tuple(h["joint_names"]), z["positions"], z["step"], z["R"],
                       z["ok"], z["sol_po"], z["sol_q"], z["sol_type"], h["settings"])

    def __repr__(self) -> str:
        return (f"ReachableWorkspace({self.arm} {self.kind}: {len(self.positions)} voxels of "
                f"{self.step[0] * 1000:.0f} mm, {len(self.R)} orientations, {len(self.sol_q):,} solutions)")


def workspace_from_result(result, kind: str, types: np.ndarray | None = None, sol_type: np.ndarray | None = None,
                          frame: str = "base_link") -> ReachableWorkspace:
    """robot_reach's side: cut one kind (and the allowed IK branch types) out of a ReachResult."""
    m = result.reach
    P, O, S = m.code.shape
    sol_type = np.zeros(len(m.sol_q), np.uint8) if sol_type is None else sol_type
    allowed = np.ones(len(m.sol_q), bool) if types is None else np.asarray(types)[sol_type]
    omask = np.ones(O, bool) if kind in ("reachable",) else getattr(m.orientations, kind)
    po_all = m.sol_index // S
    keep_sol = allowed & omask[po_all % O]
    ok = np.zeros(P * O, bool)
    ok[po_all[keep_sol]] = True
    ok = ok.reshape(P, O)[:, omask]
    member = ok.any(1) if kind == "reachable" else ok.all(1)
    remap_o = np.full(O, -1)
    remap_o[omask] = np.arange(omask.sum())
    remap_p = np.full(P, -1)
    remap_p[member] = np.arange(member.sum())
    s = keep_sol & member[po_all // O]
    po = remap_p[po_all[s] // O] * omask.sum() + remap_o[po_all[s] % O]
    order = np.argsort(po, kind="stable")
    step = float(result.config["step"])
    settings = {"source": str(result.path), "step_m": step, "orientations": int(O), "kind_orientations": int(omask.sum()),
                "allowed_types": None if types is None else np.flatnonzero(types).tolist()}
    return ReachableWorkspace(m.arm_name, kind, frame, tuple(m.joint_names), m.positions[member], np.full(3, step),
                              m.orientations.R[omask], ok[member], po[order], m.sol_q[s][order].astype(float),
                              sol_type[s][order], settings)
