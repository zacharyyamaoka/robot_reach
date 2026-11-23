"""P1 · Attribute namespace: the workspace kinds are attributes, each one a region you sample from.

    ws.dexterous.sample(3, rng)
    ws.hemisphere.inside(belt).only(elbow="up").sample_pairs(3, rng)
    cell.left.hemisphere & cell.right.hemisphere          # several chains: one more attribute level

robot_msgs: `ReachableWorkspace` (manipulation/reachable_workspace.py) holds the ChainReach arrays
and an optional IK-configuration filter; `WorkspaceRegion` is a lazy view (kind + zones + filter),
never stored. `Cell` (manipulation/cell_workspace.py) maps chain name -> ReachableWorkspace.

WHY views are lazy and the filter is re-applied at the end: "hemisphere, elbow up only" must mean
"every hemisphere orientation works WITH an elbow-up solution", so `.only()` and `.hemisphere`
commute (ws.only(elbow="up").hemisphere == ws.hemisphere.only(elbow="up")).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

import core
from core import KINDS, RULE, ChainReach, PoseSamples


@dataclass(frozen=True, eq=False)
class ReachableWorkspace:
    reach: ChainReach
    allowed: np.ndarray | None = None  # (n_configurations,) bool; None = every IK configuration

    # the happy paths: one attribute per kind
    @property
    def full(self) -> WorkspaceRegion:
        return WorkspaceRegion(self, "full")

    @property
    def reachable(self) -> WorkspaceRegion:
        return WorkspaceRegion(self, "reachable")

    @property
    def top_down(self) -> WorkspaceRegion:
        return WorkspaceRegion(self, "top_down")

    @property
    def hemisphere(self) -> WorkspaceRegion:
        return WorkspaceRegion(self, "hemisphere")

    @property
    def dexterous(self) -> WorkspaceRegion:
        return WorkspaceRegion(self, "dexterous")

    def __getitem__(self, kind: str) -> WorkspaceRegion:  # for a kind held in a variable
        if kind not in KINDS:
            raise KeyError(f"{kind!r}: kinds are {KINDS}")
        return WorkspaceRegion(self, kind)

    @property
    def configurations(self):
        return self.reach.configurations

    def only(self, *labels: str, **answers) -> ReachableWorkspace:
        a = self.reach.configurations.allowed(*labels, **answers)
        return replace(self, allowed=a if self.allowed is None else a & self.allowed)

    def __repr__(self) -> str:
        sizes = ", ".join(f"{k} {len(self[k])}" for k in KINDS)
        return f"ReachableWorkspace({self.reach.chain}: {sizes} voxels of {self.reach.voxel_size * 1000:.0f} mm)"


@dataclass(frozen=True, eq=False)
class WorkspaceRegion:
    ws: ReachableWorkspace
    kind: str
    zones: tuple = ()
    within_deg: float | None = None  # narrows the kind's orientations to a cone round straight down

    def inside(self, zone) -> WorkspaceRegion:
        return replace(self, zones=self.zones + (zone,))

    def only(self, *labels: str, **answers) -> WorkspaceRegion:
        return replace(self, ws=self.ws.only(*labels, **answers))

    def within(self, deg: float) -> WorkspaceRegion:
        """The recommended borrow from P2: a custom approach cone without a sixth attribute."""
        return replace(self, within_deg=float(deg))

    def spec(self) -> dict:
        """The region as data (P2's WorkspaceQuery fields): what a campaign logs beside its draws."""
        r = self.ws.reach
        return {"chain": r.chain, "kind": self.kind, "zones": [repr(z) for z in self.zones], "within_deg": self.within_deg,
                "configurations": None if self.ws.allowed is None else [lb for lb, a in zip(r.configurations.labels, self.ws.allowed) if a]}

    def to_sampler(self):
        """The recommended borrow from P4: hand a draw-only file to a training loop."""
        from p4_sampler import PoseSampler

        r, rows = self.ws.reach, self._rows()
        po = r.solution_pose[rows]
        return PoseSampler(r.chain, r.frame, r.joint_names, f"{r.chain} {self.spec()}", r.positions[po // r.O].astype(np.float32),
                           r.orientations[po % r.O].astype(np.float32), r.solution_q[rows], r.solution_configuration[rows],
                           r.configurations.labels, po, r.voxel_size, True, tuple(r.base_xyz))

    def as_reach_map_like(self):
        """robot_trajectories' duck type (joint_names, joint_solutions, valid_mask) + v1's reach_scores, cut to this region."""
        m = core.reach_map_like(self.ws.reach, self._allowed)
        m.valid_mask = m.valid_mask & self.voxel_mask()[:, None] & self._orient[None, :]
        m.joint_solutions = np.where(m.valid_mask[..., None], m.joint_solutions, np.nan)
        m.reach_scores = m.valid_mask.mean(1)
        return m

    # --------------------------------------------------------------- what it is
    @property
    def _allowed(self) -> np.ndarray:
        return self.ws.reach.everything() if self.ws.allowed is None else self.ws.allowed

    @property
    def _orient(self) -> np.ndarray:
        o = self.ws.reach.kind_orientations[self.kind]
        return o if self.within_deg is None else o & (self.ws.reach.tilt_deg <= self.within_deg + 1e-6)

    def _zone_mask(self) -> np.ndarray:
        m = np.ones(self.ws.reach.P, bool)
        for z in self.zones:
            m &= z.contains(self.ws.reach.positions)
        return m

    def voxel_mask(self) -> np.ndarray:
        return core.voxels(self.ws.reach, self._orient, RULE[self.kind], self._allowed) & self._zone_mask()

    @property
    def positions(self) -> np.ndarray:
        return self.ws.reach.positions[self.voxel_mask()]

    def __len__(self) -> int:
        return int(self.voxel_mask().sum())

    def contains(self, points) -> np.ndarray:
        r = self.ws.reach
        keys = core.cell_keys(np.atleast_2d(points), r.voxel_size)
        return np.isin(keys, core.cell_keys(self.positions, r.voxel_size))

    def _rows(self) -> np.ndarray:
        return core.rows(self.ws.reach, self._orient, RULE[self.kind], self._allowed, self._zone_mask())

    def count_by_configuration(self) -> dict[str, int]:
        """Poses per IK configuration that this region can hand out."""
        r = self.ws.reach
        c = np.bincount(r.solution_configuration[self._rows()], minlength=len(r.configurations))
        return {lb: int(n) for lb, n in zip(r.configurations.labels, c) if n}

    # --------------------------------------------------------------- sampling
    def _draw(self, n, k, rng, same_configuration) -> PoseSamples:
        r, rows = self.ws.reach, self._rows()
        idx = core.draw(r.solution_pose[rows], r.solution_configuration[rows], n, k, rng, same_configuration)
        return core.samples(r, rows[idx])

    def sample(self, n: int, rng: np.random.Generator) -> PoseSamples:
        s = self._draw(n, 1, rng, False)
        return replace(s, pose=s.pose[:, 0], q=s.q[:, 0], configuration=s.configuration[:, 0])

    def sample_pairs(self, n: int, rng: np.random.Generator, *, same_configuration: bool = True) -> PoseSamples:
        return self._draw(n, 2, rng, same_configuration)  # (n, 2): A, B

    def sample_waypoints(self, n: int, k: int, rng: np.random.Generator, *, same_configuration: bool = True) -> PoseSamples:
        return self._draw(n, k, rng, same_configuration)  # (n, k)

    def __and__(self, other: WorkspaceRegion) -> SharedRegion:
        return SharedRegion((self, other))

    def __repr__(self) -> str:
        r = self.ws.reach
        f = "" if self.ws.allowed is None else " · " + "|".join(np.array(r.configurations.labels)[self.ws.allowed])
        z = "".join(f" ∩ {z!r}" for z in self.zones) + ("" if self.within_deg is None else f" ∩ within {self.within_deg:g} deg")
        return f"<{r.chain}.{self.kind}{z}{f}: {len(self)} voxels, {len(self._rows()):,} IK solutions>"


@dataclass(frozen=True, eq=False)
class SharedRegion:
    """Grid points every chain's region contains (a hand-over, two arms over one belt)."""

    regions: tuple[WorkspaceRegion, ...]

    def sample(self, n: int, rng: np.random.Generator) -> dict[str, PoseSamples]:
        reaches = [g.ws.reach for g in self.regions]
        rows, _ = core.shared_draw(reaches, [g._rows() for g in self.regions], n, rng)
        return {r.chain: core.samples(r, rw) for r, rw in zip(reaches, rows)}

    def __len__(self) -> int:
        keys = [set(core.cell_keys(g.positions, g.ws.reach.voxel_size).tolist()) for g in self.regions]
        return len(set.intersection(*keys))


@dataclass(frozen=True, eq=False)
class Cell:
    """Several chains in one frame: cell.left.hemisphere, cell["right"].dexterous."""

    chains: dict

    def __getattr__(self, name: str) -> ReachableWorkspace:
        try:
            return self.__dict__["chains"][name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(self, name: str) -> ReachableWorkspace:
        return self.chains[name]


def reachable_workspace(reach: ChainReach) -> ReachableWorkspace:
    """robot_reach's producer (would be `rr.load(folder).workspace()`)."""
    return ReachableWorkspace(reach)
