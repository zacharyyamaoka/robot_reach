"""Workspace regions: the downstream API over a reach result (Zach's pick P1, refined 2026-11-24).

    ws = rr.load("out/ur5e").workspace()
    ws.dexterous.sample(3, rng)                                           # PoseSamples (3,)
    ws.hemisphere.inside(belt).only(elbow="up").sample_pairs(3, rng)      # PoseSamples (3, 2): A, B
    ws.regions["dexterous"] is the same region as ws.dexterous            # attributes are aliases
    ws.regions.register("steep", ws.reachable.within(30))                 # then ws.steep works too

ONE region type. A `WorkspaceRegion` is a lazy definition over one chain's checked IK solutions:
which orientations it asks for (a mask over the shared orientation set), how many of them must work
at a grid point (`rule`: none / any / all), which zones it is cut to, which IK configurations it may
use, and an optional cone round straight down. The levels (full ⊃ reachable ⊃ top_down ⊃ hemisphere ⊃
dexterous) are just the built-in entries of a registry, each the sub-region of the one before;
`ws.dexterous` is an alias for `ws.regions["dexterous"]`, and a user-registered name is looked up the
same way, on the workspace and as a sub-region of its parent.

WHY membership is re-judged under every filter (round 2): "hemisphere, elbow up only" must mean
"every hemisphere orientation works WITH an elbow-up IK solution", so `.only()` never overclaims and
commutes with picking the region (ws.only(elbow="up").hemisphere == ws.hemisphere.only(elbow="up")).

WHY every pose handed out is a grid point x an orientation IK actually solved, with that solution's
joints: nothing is re-solved here and nothing is promised that the library did not check.

Pure numpy. robot_msgs only at the edge: `PoseSamples.poses()` / `.joint_states()`. Multi-chain is
kept open (a workspace names its chain, frame and base placement; regions never assume one arm), but
only one chain is implemented. robot_msgs home, when it is added there (not in this task):
`robot_msgs.manipulation.ReachableWorkspace` (the arrays below) + `WorkspaceRegion.spec()` as data.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

from .configurations import IKConfigurations
from .sampling import Box, Cylinder

RULES = ("none", "any", "all")
SCHEMA = "robot_reach.region.v1"
SAMPLER_SCHEMA = "robot_reach.pose_sampler.v1"


# ------------------------------------------------------------------------------------------ results
@dataclass(frozen=True, eq=False)
class PoseSamples:
    """Drawn poses, shape (n,) or (n, k) for k waypoints, and the checked IK solution of each."""

    chain: str
    frame: str
    joint_names: tuple[str, ...]
    pose: np.ndarray  # (..., 4, 4) tool pose in `frame`
    q: np.ndarray  # (..., dof) rad: an IK solution the library checked reaches it
    configuration: np.ndarray  # (...,) IK configuration index
    configuration_names: tuple[str, ...]  # index -> "shoulder front · elbow up · wrist J5+"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.configuration.shape

    def __len__(self) -> int:
        return len(self.configuration)

    def configuration_name(self, *index) -> str:
        return self.configuration_names[int(self.configuration[index])]

    def poses(self) -> list:
        """robot_msgs `Pose` per sample, flattened in C order."""
        from robot_msgs.geometry import Pose

        return [Pose.from_matrix(T) for T in self.pose.reshape(-1, 4, 4)]

    def joint_states(self) -> list:
        """robot_msgs `JointState` per sample, flattened in C order."""
        from robot_msgs.sensor import JointState

        return [JointState(name=self.joint_names, position=q) for q in self.q.reshape(-1, self.q.shape[-1])]


class NearestPose(NamedTuple):
    pose: np.ndarray  # (4, 4) the checked pose closest to the target
    q: np.ndarray  # (dof,) its IK solution, rad
    distance_m: float  # position error to the target
    angle_deg: float  # orientation error to the target
    configuration: str  # the IK configuration's name


# ------------------------------------------------------------------------------------------ the arrays
@dataclass(frozen=True, eq=False)
class ReachableWorkspace:
    """One chain's checked reach as plain arrays, plus the named-region registry.

    Build it with `ReachResult.workspace()` (or `ReachableWorkspace.from_reach_map`)."""

    chain: str
    frame: str
    joint_names: tuple[str, ...]
    base_xyz: np.ndarray  # (3,) where the chain's base sits in `frame`
    positions: np.ndarray  # (P, 3) every grid point analysed, in `frame`, m
    voxel_size: float  # m
    rotations: np.ndarray  # (O, 3, 3) the orientation set shared by every point
    tilt_deg: np.ndarray  # (O,) angle of the tool's +Z from straight down
    solution_pose: np.ndarray  # (K,) p * O + o, sorted
    solution_q: np.ndarray  # (K, dof) rad
    solution_configuration: np.ndarray  # (K,) index into `configurations`
    configurations: IKConfigurations
    allowed: np.ndarray | None = None  # (n_configurations,) bool; None = every IK configuration
    _store: dict = field(default_factory=dict, repr=False)  # name -> (WorkspaceRegion, parent); shared by .only() copies

    # ---------------------------------------------------------------- producers
    @classmethod
    def from_reach_map(cls, m, *, voxel_size: float | None = None, frame: str = "base_link",
                       base_xyz=(0.0, 0.0, 0.0)) -> ReachableWorkspace:
        if m is None:
            raise ValueError("this result has no reach map (a curve workspace): nothing to sample")
        S = m.n_branches
        po = (m.sol_index // S).astype(np.int64)
        configs = m.configurations or IKConfigurations(tuple(m.joint_names), ())
        cfg = np.zeros(len(po), np.uint8) if m.sol_config is None else np.asarray(m.sol_config, np.uint8)
        positions = np.asarray(m.positions, float) + np.asarray(base_xyz, float)
        ws = cls(m.arm_name, frame, tuple(m.joint_names), np.asarray(base_xyz, float), positions,
                 float(voxel_size or _infer_step(positions)), np.asarray(m.orientations.R, float), np.degrees(m.orientations.tilt),
                 po, np.asarray(m.sol_q, float), cfg, configs)
        ws._register_builtins(oriented=m.task != "position")
        return ws

    def _register_builtins(self, oriented: bool) -> None:
        O, t = len(self.rotations), self.tilt_deg
        every = np.ones(O, bool)
        built = [("full", every, "none", None), ("reachable", every, "any", "full")]
        if oriented:  # WHY: a position-only arm claims no orientation workspace (as ReachMap.workspace)
            built += [("top_down", t < 1e-4, "all", "reachable"), ("hemisphere", t <= 90 + 1e-4, "all", "top_down"),
                      ("dexterous", every, "all", "hemisphere")]
        for name, orient, rule, parent in built:
            self._store[name] = (WorkspaceRegion(self, name, orient, rule), parent)

    # ---------------------------------------------------------------- the registry
    @property
    def regions(self) -> Regions:
        """Every named region, dict-like: ws.regions["dexterous"], list(ws.regions), .register(...)."""
        return Regions(self)

    def region(self, name: str) -> WorkspaceRegion:
        return self.regions[name]

    # the built-in aliases (typed for editors); registered names resolve through __getattr__
    @property
    def full(self) -> WorkspaceRegion:
        return self.region("full")

    @property
    def reachable(self) -> WorkspaceRegion:
        return self.region("reachable")

    @property
    def top_down(self) -> WorkspaceRegion:
        return self.region("top_down")

    @property
    def hemisphere(self) -> WorkspaceRegion:
        return self.region("hemisphere")

    @property
    def dexterous(self) -> WorkspaceRegion:
        return self.region("dexterous")

    def __getattr__(self, name: str) -> WorkspaceRegion:
        store = self.__dict__.get("_store")
        if name.startswith("_") or store is None or name not in store:
            raise AttributeError(name)
        return self.region(name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self._store))

    # ---------------------------------------------------------------- filters
    def only(self, allowed: np.ndarray | None = None, **choices) -> ReachableWorkspace:
        """The same workspace, using only some IK configurations: ws.only(elbow="up")."""
        a = _allowed_of(self.configurations, allowed, choices)
        return replace(self, allowed=a if self.allowed is None else a & self.allowed)

    def from_spec(self, spec: dict) -> WorkspaceRegion:
        """Rebuild a region from its `spec()` (box and cylinder zones; other shapes are logged only)."""
        g = self.region(spec["region"])
        if spec.get("within_deg") is not None:
            g = g.within(spec["within_deg"])
        for z in spec.get("zones", ()):
            g = g.inside(_zone_from_json(z))
        if spec.get("configurations") is not None:
            names = [self.configurations.name(i) for i in range(self.configurations.n_configurations)]
            g = g.only(np.isin(names, spec["configurations"]))
        return g

    def __repr__(self) -> str:
        sizes = ", ".join(f"{n} {len(self.region(n))}" for n in self._store)
        return f"ReachableWorkspace({self.chain}: {sizes} grid points of {self.voxel_size * 1000:.0f} mm)"


class Regions(Mapping):
    """The named regions of one workspace, bound to it (so ws.only(...).regions carries the filter)."""

    def __init__(self, ws: ReachableWorkspace):
        self._ws = ws

    def __getitem__(self, name: str) -> WorkspaceRegion:
        try:
            region, _ = self._ws._store[name]
        except KeyError:
            raise KeyError(f"no region {name!r}; regions are {list(self._ws._store)}") from None
        return replace(region, ws=self._ws)

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._ws._store))

    def __len__(self) -> int:
        return len(self._ws._store)

    def parent(self, name: str) -> str | None:
        return self._ws._store[name][1]

    def children(self, name: str) -> list[str]:
        return [n for n, (_, p) in self._ws._store.items() if p == name]

    def register(self, name: str, region: WorkspaceRegion, *, parent: str | None = None) -> WorkspaceRegion:
        """Name a region so ws.<name>, ws.regions[name] and <parent>.<name> find it. `parent`
        defaults to the region it was cut from. Re-registering a name replaces it."""
        if not name.isidentifier() or name.startswith("_") or hasattr(type(self._ws), name) or hasattr(WorkspaceRegion, name):
            raise ValueError(f"{name!r} is not a free attribute name")
        if parent is not None and parent not in self._ws._store:
            raise KeyError(f"no parent region {parent!r}")
        named = replace(region, name=name)
        self._ws._store[name] = (named, region.name if parent is None else parent)
        return replace(named, ws=self._ws)

    def __repr__(self) -> str:
        return "Regions(" + ", ".join(f"{n} ⊂ {p}" if p else n for n, (_, p) in self._ws._store.items()) + ")"


# ------------------------------------------------------------------------------------------ the region
@dataclass(frozen=True, eq=False)
class WorkspaceRegion:
    """A lazy region of one chain's workspace. Filters return regions; nothing is computed until asked."""

    ws: ReachableWorkspace
    name: str
    orient: np.ndarray  # (O,) bool: the orientations this region asks for
    rule: str  # "none": every grid point · "any": one asked orientation works · "all": every one works
    zones: tuple = ()  # each has contains(points (N, 3)) -> (N,) bool, in the workspace's frame
    allowed: np.ndarray | None = None  # (n_configurations,) bool, AND-ed with the workspace's
    within_deg: float | None = None  # narrows `orient` to a cone round straight down

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError(f"rule {self.rule!r}; rules are {RULES}")

    # ---------------------------------------------------------------- filters (each returns a region)
    def inside(self, zone) -> WorkspaceRegion:
        """Cut to a zone: anything with contains(points) in the workspace's frame (rr.Box, rr.Cylinder)."""
        return replace(self, zones=self.zones + (zone,))

    def only(self, allowed: np.ndarray | None = None, **choices) -> WorkspaceRegion:
        """Use only some IK configurations: .only(elbow="up"), .only(wrist=("J5+", "J5−")). Membership is
        re-judged: a grid point stays only if the region's orientations work WITH those configurations."""
        a = _allowed_of(self.ws.configurations, allowed, choices)
        return replace(self, allowed=a if self.allowed is None else a & self.allowed)

    def within(self, deg: float) -> WorkspaceRegion:
        """Only the orientations whose tool +Z is within `deg` of straight down (a custom approach cone)."""
        deg = float(deg)
        return replace(self, within_deg=deg if self.within_deg is None else min(deg, self.within_deg))

    # ---------------------------------------------------------------- sub-regions
    @property
    def parent(self) -> str | None:
        return self.ws._store[self.name][1] if self.name in self.ws._store else None

    @property
    def regions(self) -> dict[str, WorkspaceRegion]:
        """The registered sub-regions of this one, each carrying this region's zones and filters."""
        return {n: self.region(n) for n in self.ws.regions.children(self.name)}

    def region(self, name: str) -> WorkspaceRegion:
        """A registered region, cut by this region's zones, IK configurations and cone."""
        sub = self.ws.regions[name]
        allowed = sub.allowed if self.allowed is None else (self.allowed if sub.allowed is None else sub.allowed & self.allowed)
        within = [d for d in (sub.within_deg, self.within_deg) if d is not None]
        return replace(sub, zones=self.zones + sub.zones, allowed=allowed, within_deg=min(within) if within else None)

    def __getattr__(self, name: str) -> WorkspaceRegion:
        ws = self.__dict__.get("ws")
        if name.startswith("_") or ws is None or name not in ws._store:
            raise AttributeError(name)
        return self.region(name)

    # ---------------------------------------------------------------- identity
    def _key(self) -> tuple:
        a = self._allowed
        return (id(self.ws._store), self.ws.chain, self.rule, self._orient.tobytes(), a.tobytes(), tuple(map(repr, self.zones)))

    def __eq__(self, other) -> bool:
        """Same chain, same membership definition (orientations, rule, zones, IK configurations)."""
        return isinstance(other, WorkspaceRegion) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    # ---------------------------------------------------------------- what it is
    @property
    def _allowed(self) -> np.ndarray:
        a = np.ones(self.ws.configurations.n_configurations, bool)
        for x in (self.ws.allowed, self.allowed):
            if x is not None:
                a = a & x
        return a

    @property
    def _orient(self) -> np.ndarray:
        o = np.asarray(self.orient, bool)
        return o if self.within_deg is None else o & (self.ws.tilt_deg <= self.within_deg + 1e-6)

    @cached_property
    def _ok_po(self) -> np.ndarray:
        """(P, O) bool: an allowed IK solution exists at that pose."""
        ws = self.ws
        keep = self._allowed[ws.solution_configuration]
        ok = np.zeros(len(ws.positions) * len(ws.rotations), bool)
        ok[ws.solution_pose[keep]] = True
        return ok.reshape(len(ws.positions), len(ws.rotations))

    @cached_property
    def voxel_mask(self) -> np.ndarray:
        """(P,) bool: the grid points in this region."""
        P, orient = len(self.ws.positions), self._orient
        if self.rule == "none":
            m = np.ones(P, bool)
        else:
            ok = self._ok_po[:, orient]
            m = ok.any(1) if self.rule == "any" else ok.all(1) & orient.any()
        for z in self.zones:
            m = m & np.asarray(z.contains(self.ws.positions), bool)
        return m

    @cached_property
    def _rows(self) -> np.ndarray:
        """Indices of the IK solutions this region may hand out (allowed configuration, at a member
        grid point, at one of the asked orientations)."""
        ws = self.ws
        O = len(ws.rotations)
        p, o = ws.solution_pose // O, ws.solution_pose % O
        return np.flatnonzero(self._allowed[ws.solution_configuration] & self.voxel_mask[p] & self._orient[o])

    @property
    def positions(self) -> np.ndarray:
        return self.ws.positions[self.voxel_mask]

    def __len__(self) -> int:
        """Grid points in the region."""
        return int(self.voxel_mask.sum())

    @property
    def n_solutions(self) -> int:
        return len(self._rows)

    def contains(self, points) -> np.ndarray:
        """(N,) bool: the point falls in a grid voxel of this region (nearest grid point within half a voxel)."""
        pts = np.atleast_2d(np.asarray(points, float))
        ws = self.ws
        origin = ws.positions.min(0) if len(ws.positions) else np.zeros(3)
        mine = _pack(np.floor((self.positions - origin) / ws.voxel_size + 0.5).astype(np.int64))
        return np.isin(_pack(np.floor((pts - origin) / ws.voxel_size + 0.5).astype(np.int64)), mine)

    def count_by_configuration(self) -> dict[str, int]:
        """IK solutions per IK configuration this region can hand out."""
        c = np.bincount(self.ws.solution_configuration[self._rows], minlength=self.ws.configurations.n_configurations)
        return {self.ws.configurations.name(i): int(n) for i, n in enumerate(c) if n}

    # ---------------------------------------------------------------- sampling
    def _samples(self, row_idx: np.ndarray) -> PoseSamples:
        ws = self.ws
        O = len(ws.rotations)
        po = ws.solution_pose[row_idx]
        T = np.zeros((*row_idx.shape, 4, 4))
        T[..., :3, :3] = ws.rotations[po % O]
        T[..., :3, 3] = ws.positions[po // O]
        T[..., 3, 3] = 1.0
        return PoseSamples(ws.chain, ws.frame, ws.joint_names, T, ws.solution_q[row_idx].astype(float),
                           ws.solution_configuration[row_idx], _names(ws.configurations))

    def _draw(self, n: int, k: int, rng: np.random.Generator, same_configuration: bool) -> PoseSamples:
        rows = self._rows
        idx = _draw(self.ws.solution_pose[rows], self.ws.solution_configuration[rows], n, k, rng, same_configuration)
        return self._samples(rows[idx])

    def sample(self, n: int, rng: np.random.Generator) -> PoseSamples:
        """(n,) poses, uniform over (grid point, orientation) poses, each with one checked IK solution."""
        s = self._draw(n, 1, rng, False)
        return replace(s, pose=s.pose[:, 0], q=s.q[:, 0], configuration=s.configuration[:, 0])

    def sample_pairs(self, n: int, rng: np.random.Generator, *, same_configuration: bool = True) -> PoseSamples:
        """(n, 2): A, B, two distinct poses; by default both in one IK configuration."""
        return self._draw(n, 2, rng, same_configuration)

    def sample_waypoints(self, n: int, k: int, rng: np.random.Generator, *, same_configuration: bool = True) -> PoseSamples:
        """(n, k) distinct poses per draw."""
        return self._draw(n, k, rng, same_configuration)

    def nearest_ok_pose(self, pose, *, m_per_rad: float = 0.1) -> NearestPose:
        """The checked pose in this region closest to `pose` (4x4, or (3,) position: orientation ignored).
        Cost = position error + m_per_rad x orientation error (default: 10 cm per radian)."""
        rows = self._rows
        if not len(rows):
            raise ValueError(f"region {self.name!r} is empty")
        ws, O = self.ws, len(self.ws.rotations)
        T = np.asarray(pose, float)
        po = ws.solution_pose[rows]
        d = np.linalg.norm(ws.positions[po // O] - (T[:3, 3] if T.shape == (4, 4) else T), axis=1)
        if T.shape == (4, 4):
            cos = (np.einsum("ij,kij->k", T[:3, :3], ws.rotations) - 1) / 2  # trace(T^T R) per orientation
            ang = np.arccos(np.clip(cos, -1, 1))[po % O]
        else:
            ang = np.zeros(len(rows))
        i = int(np.argmin(d + m_per_rad * ang))
        s = self._samples(rows[i : i + 1])
        return NearestPose(s.pose[0], s.q[0], float(d[i]), float(np.degrees(ang[i])), s.configuration_name(0))

    # ---------------------------------------------------------------- hand-offs
    def spec(self) -> dict:
        """The region as JSON-able data: what a campaign logs beside its draws (`ws.from_spec` rebuilds it)."""
        a = self._allowed
        return {"schema": SCHEMA, "chain": self.ws.chain, "frame": self.ws.frame, "region": self.name, "parent": self.parent,
                "rule": self.rule, "within_deg": self.within_deg, "zones": [_zone_to_json(z) for z in self.zones],
                "configurations": None if a.all() else [self.ws.configurations.name(i) for i in np.flatnonzero(a)],
                "grid_points": len(self), "ik_solutions": self.n_solutions}

    def to_sampler(self, *, same_configuration: bool = True) -> PoseSampler:
        """A draw-only copy of this region's eligible poses for a training loop (saveable, no workspace inside)."""
        ws, rows = self.ws, self._rows
        O = len(ws.rotations)
        po = ws.solution_pose[rows]
        return PoseSampler(ws.chain, ws.frame, ws.joint_names, json.dumps(self.spec(), ensure_ascii=False),
                           ws.positions[po // O].copy(), ws.rotations[po % O].copy(), ws.solution_q[rows].copy(),
                           ws.solution_configuration[rows].copy(), _names(ws.configurations), po.copy(), same_configuration)

    def as_reach_map_like(self) -> SimpleNamespace:
        """robot_trajectories' duck type (joint_names, joint_solutions (P, O, dof) NaN where none,
        valid_mask (P, O), reach_scores (P,)), cut to this region."""
        ws, rows = self.ws, self._rows
        P, O = len(ws.positions), len(ws.rotations)
        po = ws.solution_pose[rows]
        first = np.r_[True, np.diff(po) != 0] if len(po) else np.zeros(0, bool)
        js = np.full((P * O, len(ws.joint_names)), np.nan)
        js[po[first]] = ws.solution_q[rows[first]]
        valid = np.zeros(P * O, bool)
        valid[po] = True
        valid = valid.reshape(P, O)
        return SimpleNamespace(joint_names=ws.joint_names, joint_solutions=js.reshape(P, O, -1), valid_mask=valid,
                               reach_scores=valid.mean(1), positions=ws.positions, orientations=ws.rotations)

    def __repr__(self) -> str:
        bits = "".join(f" ∩ {z!r}" for z in self.zones)
        if self.within_deg is not None:
            bits += f" ∩ within {self.within_deg:g}°"
        a = self._allowed
        if not a.all():
            bits += " · " + " | ".join(self.ws.configurations.name(i) for i in np.flatnonzero(a))
        return f"<{self.ws.chain}.{self.name}{bits}: {len(self)} grid points, {self.n_solutions:,} IK solutions>"


# ------------------------------------------------------------------------------------------ draw-only hand-off
@dataclass(frozen=True, eq=False)
class PoseSampler:
    """The eligible poses only (position, rotation, IK solution, IK configuration per row): the smallest
    thing a trajectory generator can hold. `region.to_sampler()` makes one; `save` / `load` move it."""

    chain: str
    frame: str
    joint_names: tuple[str, ...]
    description: str  # the region's spec, as JSON
    position: np.ndarray  # (M, 3) m
    rotation: np.ndarray  # (M, 3, 3)
    q: np.ndarray  # (M, dof) rad, checked
    configuration: np.ndarray  # (M,)
    configuration_names: tuple[str, ...]
    pose_key: np.ndarray  # (M,) equal for rows at the same pose (several IK configurations)
    same_configuration: bool = True  # the policy for pairs / waypoints

    def __len__(self) -> int:
        return len(np.unique(self.pose_key))

    def _samples(self, idx: np.ndarray) -> PoseSamples:
        T = np.zeros((*idx.shape, 4, 4))
        T[..., :3, :3] = self.rotation[idx]
        T[..., :3, 3] = self.position[idx]
        T[..., 3, 3] = 1.0
        return PoseSamples(self.chain, self.frame, self.joint_names, T, self.q[idx].astype(float), self.configuration[idx],
                           self.configuration_names)

    def sample(self, n: int, rng: np.random.Generator) -> PoseSamples:
        return self._samples(_draw(self.pose_key, self.configuration, n, 1, rng, False)[:, 0])

    def sample_pairs(self, n: int, rng: np.random.Generator) -> PoseSamples:
        return self.sample_waypoints(n, 2, rng)

    def sample_waypoints(self, n: int, k: int, rng: np.random.Generator) -> PoseSamples:
        return self._samples(_draw(self.pose_key, self.configuration, n, k, rng, self.same_configuration))

    def save(self, path: str | Path) -> Path:
        head = {"schema": SAMPLER_SCHEMA, "chain": self.chain, "frame": self.frame, "joint_names": list(self.joint_names),
                "description": self.description, "configuration_names": list(self.configuration_names),
                "same_configuration": self.same_configuration}
        path = Path(path)
        with open(path, "wb") as f:  # WHY a file handle: np.savez would append ".npz" to any other suffix
            np.savez_compressed(f, head=np.frombuffer(json.dumps(head).encode(), np.uint8), position=self.position,
                                rotation=self.rotation, q=self.q, configuration=self.configuration, pose_key=self.pose_key)
        return path

    @classmethod
    def load(cls, path: str | Path) -> PoseSampler:
        with np.load(path, allow_pickle=False) as z:
            h = json.loads(bytes(z["head"]).decode())
            if h.get("schema") != SAMPLER_SCHEMA:
                raise ValueError(f"{path}: schema {h.get('schema')!r}, expected {SAMPLER_SCHEMA}")
            return cls(h["chain"], h["frame"], tuple(h["joint_names"]), h["description"], z["position"], z["rotation"], z["q"],
                       z["configuration"], tuple(h["configuration_names"]), z["pose_key"], h["same_configuration"])

    def __repr__(self) -> str:
        return f"PoseSampler({self.chain}: {len(self):,} poses, {len(self.q):,} IK solutions)"


# ------------------------------------------------------------------------------------------ helpers
def _names(configs: IKConfigurations) -> tuple[str, ...]:
    return tuple(configs.name(i) or "any" for i in range(configs.n_configurations))


def _allowed_of(configs: IKConfigurations, allowed, choices: dict) -> np.ndarray:
    known = {q.name for q in configs.questions}
    unknown = set(choices) - known
    if unknown:  # WHY: IKConfigurations.allowed ignores unknown names; a typo must not silently allow everything
        raise ValueError(f"no IK configuration question {sorted(unknown)}; this chain has {sorted(known) or 'none'}")
    a = configs.allowed(**choices)
    if allowed is not None:
        allowed = np.asarray(allowed, bool)
        if allowed.shape != a.shape:
            raise ValueError(f"allowed has shape {allowed.shape}, expected {a.shape}")
        a = a & allowed
    return a


def _pack(g: np.ndarray) -> np.ndarray:
    g = g + (1 << 20)
    return (g[:, 0] << 42) | (g[:, 1] << 21) | g[:, 2]


def _infer_step(positions: np.ndarray) -> float:
    gaps = [np.diff(np.unique(np.round(positions[:, i], 9))) for i in range(3)]
    gaps = np.concatenate([g[g > 1e-9] for g in gaps]) if len(positions) > 1 else np.zeros(0)
    return float(gaps.min()) if len(gaps) else 1.0


def _draw(pose_key: np.ndarray, configuration: np.ndarray, n: int, k: int, rng: np.random.Generator,
          same_configuration: bool) -> np.ndarray:
    """(n, k) row indices: n draws of k DISTINCT poses each, uniform over poses (not solutions).
    same_configuration: all k poses of a draw share one IK configuration (picked in proportion to its
    pose count), so A -> B never asks the arm to change posture family on the way."""
    if len(pose_key) == 0:
        raise ValueError("nothing to sample: the region is empty")
    out = np.empty((n, k), np.int64)
    if same_configuration:
        order = np.lexsort((pose_key, configuration))
        ck = configuration[order].astype(np.int64) * (int(pose_key.max()) + 1) + pose_key[order]
        first = np.r_[True, np.diff(ck) != 0]
        u_row, u_cfg = order[first], configuration[order][first].astype(np.int64)
        counts = np.bincount(u_cfg, minlength=int(u_cfg.max()) + 1)
        start = np.r_[0, np.cumsum(counts)[:-1]]
        w = np.where(counts >= k, counts, 0).astype(float)
        if w.sum() == 0:
            raise ValueError(f"no IK configuration has {k} distinct poses here")
        cfgs = rng.choice(len(counts), size=n, p=w / w.sum())
        for i, c in enumerate(cfgs):
            out[i] = u_row[start[c] + rng.choice(counts[c], k, replace=False)]
    else:
        order = np.argsort(pose_key, kind="stable")
        keys, lo, cnt = np.unique(pose_key[order], return_index=True, return_counts=True)
        if len(keys) < k:
            raise ValueError(f"only {len(keys)} poses, asked for {k}")
        for i in range(n):
            pick = rng.choice(len(keys), k, replace=False)
            out[i] = order[lo[pick] + (rng.random(k) * cnt[pick]).astype(int)]
    return out


def _zone_to_json(z) -> dict:
    if isinstance(z, Box):
        return {"box": {"center": list(map(float, z.center)), "size": list(map(float, z.size))}}
    if isinstance(z, Cylinder):
        return {"cylinder": {"center": list(map(float, z.center)), "radius": float(z.radius), "height": float(z.height)}}
    return {"repr": repr(z)}


def _zone_from_json(d: dict):
    if "box" in d:
        return Box(tuple(d["box"]["center"]), tuple(d["box"]["size"]))
    if "cylinder" in d:
        c = d["cylinder"]
        return Cylinder(tuple(c["center"]), c["radius"], c["height"])
    raise ValueError(f"cannot rebuild zone {d}: only box and cylinder zones round-trip")
