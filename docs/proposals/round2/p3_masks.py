"""P3 · Numpy masks: the workspace IS a boolean array, and every question is numpy you already know.

    ok = ws.ok                                    # (P, O, C) bool: position x orientation x IK configuration
    up = ws.ok & ws.configuration(elbow="up")     # broadcasting does the filtering
    belt_hemi = ws.voxels("hemisphere", up) & ws.inside(belt)      # (P,) bool
    ws.sample(ws.poses(belt_hemi, "hemisphere", up), 3, rng, k=2)

robot_msgs: `ReachableWorkspace` (manipulation/reachable_workspace.py) stores the dense `ok` cube
(P x O x C bits) beside the sparse joint solutions. Masks are plain `np.ndarray`s: no query
language, nothing to learn but the three axes, and any mask a user builds (x > 0.4, a camera's
view frustum) composes with `&` / `|` / `~`. Several chains: a leading chain axis on one shared grid.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

import core
from core import RULE, ChainReach, PoseSamples


@dataclass(frozen=True, eq=False)
class ReachableWorkspace:
    reach: ChainReach
    ok: np.ndarray  # (P, O, C) bool: an IK solution in configuration c reaches (position p, orientation o)

    @classmethod
    def of(cls, reach: ChainReach) -> ReachableWorkspace:
        C = len(reach.configurations)
        ok = np.zeros(reach.P * reach.O * C, bool)
        ok[reach.solution_pose * C + reach.solution_configuration] = True
        return cls(reach, ok.reshape(reach.P, reach.O, C))

    # --------------------------------------------------------------- the three axes
    def configuration(self, *labels: str, **answers) -> np.ndarray:
        """(C,) bool, shaped to broadcast against ok: ws.ok & ws.configuration(elbow="up")."""
        return self.reach.configurations.allowed(*labels, **answers)

    def orientations(self, kind: str = "reachable", *, within_deg: float | None = None) -> np.ndarray:
        """(O, 1) bool: the orientations a kind asks for (optionally narrowed to a cone round straight down)."""
        o = self.reach.kind_orientations[kind].copy()
        if within_deg is not None:
            o &= self.reach.tilt_deg <= within_deg + 1e-6
        return o[:, None]

    def inside(self, zone) -> np.ndarray:
        """(P,) bool."""
        return zone.contains(self.reach.positions)

    # --------------------------------------------------------------- derived masks
    def voxels(self, kind: str, ok: np.ndarray | None = None) -> np.ndarray:
        """(P,) bool: grid points in `kind`, judged on `ok` (default: every IK configuration)."""
        ok = self.ok if ok is None else ok
        if RULE[kind] == "none":
            return np.ones(self.reach.P, bool)
        any_c = ok.any(2)[:, self.reach.kind_orientations[kind]]
        return any_c.any(1) if RULE[kind] == "any" else any_c.all(1)

    @property
    def dexterous(self) -> np.ndarray:  # attribute shortcuts return (P,) voxel masks
        return self.voxels("dexterous")

    @property
    def hemisphere(self) -> np.ndarray:
        return self.voxels("hemisphere")

    @property
    def top_down(self) -> np.ndarray:
        return self.voxels("top_down")

    @property
    def reachable(self) -> np.ndarray:
        return self.voxels("reachable")

    @property
    def full(self) -> np.ndarray:
        return self.voxels("full")

    def poses(self, voxels: np.ndarray, kind: str = "reachable", ok: np.ndarray | None = None) -> np.ndarray:
        """(P, O, C) bool: the samplable poses = ok, at those voxels, at the kind's orientations."""
        ok = self.ok if ok is None else ok
        return ok & voxels[:, None, None] & self.orientations(kind)[None]

    # --------------------------------------------------------------- sampling
    def sample(self, mask: np.ndarray, n: int, rng: np.random.Generator, *, k: int = 1, same_configuration: bool = True) -> PoseSamples:
        """Draw from the True cells of a (P, O, C) mask: k = 1 poses (n,), k > 1 groups (n, k)."""
        r = self.reach
        C = len(r.configurations)
        rows = np.flatnonzero(mask.reshape(-1)[r.solution_pose * C + r.solution_configuration])
        idx = core.draw(r.solution_pose[rows], r.solution_configuration[rows], n, k, rng, same_configuration and k > 1)
        s = core.samples(r, rows[idx])
        return replace(s, pose=s.pose[:, 0], q=s.q[:, 0], configuration=s.configuration[:, 0]) if k == 1 else s

    @property
    def nbytes(self) -> int:
        return self.ok.nbytes + self.reach.solution_q.nbytes + self.reach.solution_pose.nbytes


@dataclass(frozen=True, eq=False)
class CellMasks:
    """Several chains on one cell grid: voxel masks get a leading chain axis (Ch, G)."""

    chains: dict  # name -> ReachableWorkspace

    def __post_init__(self):
        keys = [core.cell_keys(w.reach.positions, w.reach.voxel_size) for w in self.chains.values()]
        grid = np.unique(np.concatenate(keys))
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "_where", [np.searchsorted(grid, k) for k in keys])

    def voxels(self, kind: str, ok: dict | None = None) -> np.ndarray:
        """(Ch, G) bool: chain c's `kind` voxels on the shared grid."""
        out = np.zeros((len(self.chains), len(self.grid)), bool)
        for i, (name, w) in enumerate(self.chains.items()):
            out[i, self._where[i]] = w.voxels(kind, None if ok is None else ok.get(name))
        return out

    def to_chain(self, name: str, cell_mask: np.ndarray) -> np.ndarray:
        """(G,) cell mask -> (P,) mask on that chain's own grid."""
        return cell_mask[self._where[list(self.chains).index(name)]]

    def sample_shared(self, cell_mask: np.ndarray, kind: str, n: int, rng) -> dict[str, PoseSamples]:
        ws = list(self.chains.values())
        rows = []
        for name, w in self.chains.items():
            m = w.poses(self.to_chain(name, cell_mask), kind)
            C = len(w.reach.configurations)
            rows.append(np.flatnonzero(m.reshape(-1)[w.reach.solution_pose * C + w.reach.solution_configuration]))
        picked, _ = core.shared_draw([w.reach for w in ws], rows, n, rng)
        return {w.reach.chain: core.samples(w.reach, rw) for w, rw in zip(ws, picked)}


def reachable_workspace(reach: ChainReach) -> ReachableWorkspace:
    return ReachableWorkspace.of(reach)
