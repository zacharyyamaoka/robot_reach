"""P4 · Sampler-first: downstream never sees a workspace, only a ready-to-draw set of checked poses.

    s = sampler(reach, "hemisphere", inside=belt, configurations=dict(elbow="up"))   # robot_reach side
    s.save("belt.poses.npz")                                                         # the hand-off
    s.pairs(3, rng) · s.waypoints(3, 5, rng) · next(s.stream(rng, k=2))              # trajectory side

robot_msgs: `PoseSampler` (manipulation/pose_sampler.py): the eligible poses only (position,
orientation, IK solution, IK configuration per row) plus the drawing policy. It is the smallest
thing a trajectory generator can hold, and it does exactly one job. Several chains:
`CoupledSampler` draws one pose per chain per draw, coupled by a rule ("same_position").
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

import core
from core import RULE, ChainReach, PoseSamples


@dataclass(frozen=True, eq=False)
class PoseSampler:
    chain: str
    frame: str
    joint_names: tuple[str, ...]
    description: str  # what it was cut from, in words ("ur5e hemisphere ∩ belt · elbow=up")
    position: np.ndarray  # (M, 3) m
    rotation: np.ndarray  # (M, 3, 3)
    q: np.ndarray  # (M, dof) rad, checked
    configuration: np.ndarray  # (M,) uint8
    configuration_labels: tuple[str, ...]
    pose_key: np.ndarray  # (M,) which rows are the same pose (several IK configurations at one pose)
    voxel_size: float
    same_configuration: bool = True  # the policy for pairs / waypoints
    base_xyz: tuple = (0.0, 0.0, 0.0)  # where the chain's base sits in `frame`

    def __len__(self) -> int:
        return len(np.unique(self.pose_key))

    def _samples(self, idx: np.ndarray) -> PoseSamples:
        T = np.zeros((*idx.shape, 4, 4))
        T[..., :3, :3] = self.rotation[idx]
        T[..., :3, 3] = self.position[idx]
        T[..., 3, 3] = 1
        return PoseSamples(self.chain, T, self.q[idx].astype(float), self.configuration[idx], self.configuration_labels, np.asarray(self.base_xyz, float))

    def poses(self, n: int, rng: np.random.Generator) -> PoseSamples:
        idx = core.draw(self.pose_key, self.configuration, n, 1, rng, False)[:, 0]
        return self._samples(idx)

    def pairs(self, n: int, rng: np.random.Generator) -> PoseSamples:
        return self.waypoints(n, 2, rng)

    def waypoints(self, n: int, k: int, rng: np.random.Generator) -> PoseSamples:
        return self._samples(core.draw(self.pose_key, self.configuration, n, k, rng, self.same_configuration))

    def stream(self, rng: np.random.Generator, k: int = 1, batch: int = 256) -> Iterator[PoseSamples]:
        """Endless draws for a training loop: one (k,) group per next()."""
        while True:
            s = self.waypoints(batch, k, rng)
            for i in range(batch):
                yield replace(s, pose=s.pose[i], q=s.q[i], configuration=s.configuration[i])

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.position, self.rotation, self.q, self.configuration, self.pose_key))

    def save(self, path: str | Path) -> Path:
        head = {"schema": "robot_msgs/pose-sampler/0-proposal", "chain": self.chain, "frame": self.frame,
                "joint_names": list(self.joint_names), "description": self.description, "labels": list(self.configuration_labels),
                "voxel_size": self.voxel_size, "same_configuration": self.same_configuration, "base_xyz": list(self.base_xyz)}
        np.savez_compressed(path, head=np.frombuffer(json.dumps(head).encode(), np.uint8), position=self.position,
                            rotation=self.rotation, q=self.q, configuration=self.configuration, pose_key=self.pose_key)
        return Path(path)

    @classmethod
    def load(cls, path: str | Path) -> PoseSampler:
        with np.load(path, allow_pickle=False) as z:
            h = json.loads(bytes(z["head"]).decode())
            return cls(h["chain"], h["frame"], tuple(h["joint_names"]), h["description"], z["position"], z["rotation"], z["q"],
                       z["configuration"], tuple(h["labels"]), z["pose_key"], h["voxel_size"], h["same_configuration"], tuple(h["base_xyz"]))

    def __repr__(self) -> str:
        return f"PoseSampler({self.description}: {len(self):,} poses, {len(self.q):,} IK solutions, {self.nbytes / 1e6:.1f} MB)"


def sampler(reach: ChainReach, kind: str = "reachable", *, inside=None, configurations: dict | None = None,
            within_deg: float | None = None, same_configuration: bool = True) -> PoseSampler:
    """robot_reach's producer: resolve the question once, hand over only the answer."""
    allowed = reach.configurations.allowed(**(configurations or {}))
    orient = reach.kind_orientations[kind].copy()
    if within_deg is not None:
        orient &= reach.tilt_deg <= within_deg + 1e-6
    zone = None if inside is None else inside.contains(reach.positions)
    rows = core.rows(reach, orient, RULE[kind], allowed, zone)
    po = reach.solution_pose[rows]
    words = f"{reach.chain} {kind}" + (f" ∩ {inside!r}" if inside is not None else "") + \
            ("".join(f" · {a}={v}" for a, v in configurations.items()) if configurations else "")
    return PoseSampler(reach.chain, reach.frame, reach.joint_names, words, reach.positions[po // reach.O].astype(np.float32),
                       reach.orientations[po % reach.O].astype(np.float32), reach.solution_q[rows], reach.solution_configuration[rows],
                       reach.configurations.labels, po, reach.voxel_size, same_configuration, tuple(reach.base_xyz))


@dataclass(frozen=True, eq=False)
class CoupledSampler:
    """One pose per chain per draw. couple="same_position": every chain at the same grid point (hand-over)."""

    samplers: dict  # name -> PoseSampler
    couple: str = "same_position"

    def draw(self, n: int, rng) -> dict[str, PoseSamples]:
        ss = list(self.samplers.values())
        if self.couple == "independent":
            return {s.chain: s.poses(n, rng) for s in ss}
        keys = [core.cell_keys(s.position.astype(float), s.voxel_size) for s in ss]
        common = keys[0]
        for k in keys[1:]:
            common = np.intersect1d(common, k)
        pick = common[rng.integers(len(common), size=n)]
        out = {}
        for s, k in zip(ss, keys):
            order = np.argsort(k, kind="stable")
            lo, hi = np.searchsorted(k[order], pick, "left"), np.searchsorted(k[order], pick, "right")
            out[s.chain] = s._samples(order[lo + (rng.random(n) * (hi - lo)).astype(int)])
        return out
