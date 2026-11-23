"""P2 · Query builder: the question is a value of its own, saved in a campaign config and asked of any arm.

    q = Q.hemisphere.inside(belt).configuration(elbow="up")
    ws.sample(q, 3, rng, k=2)          # A -> B pairs
    q.to_json()                         # the campaign's sampling spec, arm-independent

robot_msgs: `WorkspaceQuery` (manipulation/workspace_query.py), a frozen, JSON-able description of
WHICH poses (kind, zones, IK configurations, an approach cone, the chain); `ReachableWorkspace`
(manipulation/reachable_workspace.py) holds the arrays and answers queries (count, contains,
sample). What sets it apart: the same query runs against the UR5e today and a UR10e tomorrow, and a
trajectory campaign can log exactly which poses it drew from.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace

import numpy as np

import core
from core import KINDS, RULE, ChainReach, PoseSamples


@dataclass(frozen=True)
class BoxZone:
    """A JSON-able zone for queries: an axis-aligned box in the chain/cell frame (lo, hi in m)."""

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    name: str = "zone"

    def contains(self, p: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(p)
        return np.all((p >= self.lo) & (p <= self.hi), axis=1)


@dataclass(frozen=True)
class WorkspaceQuery:
    kind: str = "reachable"
    zones: tuple[BoxZone, ...] = ()
    configurations: tuple[tuple[str, tuple[str, ...]], ...] = ()  # (("elbow", ("up",)),) — AND across axes
    approach_within_deg: float | None = None  # narrows the kind's orientations to this cone round straight down
    chain: str | None = None
    same_configuration: bool = True

    # builder: every method returns a new query
    def of(self, kind: str) -> WorkspaceQuery:
        if kind not in KINDS:
            raise ValueError(f"{kind!r}: kinds are {KINDS}")
        return replace(self, kind=kind)

    def __getattr__(self, kind: str) -> WorkspaceQuery:  # Q.hemisphere == Q.of("hemisphere")
        if kind in KINDS:
            return self.of(kind)
        raise AttributeError(kind)

    def inside(self, zone: BoxZone) -> WorkspaceQuery:
        return replace(self, zones=self.zones + (zone,))

    def configuration(self, **answers) -> WorkspaceQuery:
        add = tuple((k, (v,) if isinstance(v, str) else tuple(v)) for k, v in answers.items())
        return replace(self, configurations=self.configurations + add)

    def approach_within(self, deg: float) -> WorkspaceQuery:
        return replace(self, approach_within_deg=float(deg))

    def on(self, chain: str) -> WorkspaceQuery:
        return replace(self, chain=chain)

    def any_configuration(self) -> WorkspaceQuery:
        return replace(self, same_configuration=False)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> WorkspaceQuery:
        d = json.loads(text)
        return cls(d["kind"], tuple(BoxZone(tuple(z["lo"]), tuple(z["hi"]), z["name"]) for z in d["zones"]),
                   tuple((a, tuple(o)) for a, o in d["configurations"]), d["approach_within_deg"], d["chain"], d["same_configuration"])


Q = WorkspaceQuery()


@dataclass(frozen=True, eq=False)
class ReachableWorkspace:
    reach: ChainReach

    def _resolve(self, q: WorkspaceQuery):
        r = self.reach
        allowed = r.configurations.allowed(**{a: o for a, o in q.configurations}) if q.configurations else r.everything()
        orient = r.kind_orientations[q.kind].copy()
        if q.approach_within_deg is not None:
            orient &= r.tilt_deg <= q.approach_within_deg + 1e-6
        zone = np.ones(r.P, bool)
        for z in q.zones:
            zone &= z.contains(r.positions)
        return allowed, orient, RULE[q.kind], zone

    def count(self, q: WorkspaceQuery) -> dict[str, int]:
        allowed, orient, rule, zone = self._resolve(q)
        vox = core.voxels(self.reach, orient, rule, allowed) & zone
        return {"voxels": int(vox.sum()), "ik_solutions": len(core.rows(self.reach, orient, rule, allowed, zone))}

    def contains(self, q: WorkspaceQuery, points) -> np.ndarray:
        allowed, orient, rule, zone = self._resolve(q)
        vox = core.voxels(self.reach, orient, rule, allowed) & zone
        return np.isin(core.cell_keys(np.atleast_2d(points), self.reach.voxel_size),
                       core.cell_keys(self.reach.positions[vox], self.reach.voxel_size))

    def rows(self, q: WorkspaceQuery) -> np.ndarray:
        allowed, orient, rule, zone = self._resolve(q)
        return core.rows(self.reach, orient, rule, allowed, zone)

    def sample(self, q: WorkspaceQuery, n: int, rng: np.random.Generator, *, k: int = 1) -> PoseSamples:
        """k = 1: n poses (n,); k = 2: A -> B pairs (n, 2); k > 2: waypoints (n, k)."""
        rows = self.rows(q)
        r = self.reach
        idx = core.draw(r.solution_pose[rows], r.solution_configuration[rows], n, k, rng, q.same_configuration and k > 1)
        s = core.samples(r, rows[idx])
        return replace(s, pose=s.pose[:, 0], q=s.q[:, 0], configuration=s.configuration[:, 0]) if k == 1 else s


@dataclass(frozen=True, eq=False)
class Cell:
    chains: dict  # name -> ReachableWorkspace

    def sample(self, q: WorkspaceQuery, n, rng, *, k=1) -> PoseSamples:
        if q.chain is None:
            raise ValueError(f"a cell query names its chain: .on({list(self.chains)})")
        return self.chains[q.chain].sample(q, n, rng, k=k)

    def sample_shared(self, queries: list[WorkspaceQuery], n: int, rng) -> dict[str, PoseSamples]:
        """One grid point per draw that every query's chain reaches, with each chain's solution there."""
        ws = [self.chains[q.chain] for q in queries]
        rows, _ = core.shared_draw([w.reach for w in ws], [w.rows(q) for w, q in zip(ws, queries)], n, rng)
        return {w.reach.chain: core.samples(w.reach, rw) for w, rw in zip(ws, rows)}


def reachable_workspace(reach: ChainReach) -> ReachableWorkspace:
    return ReachableWorkspace(reach)
