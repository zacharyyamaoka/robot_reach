"""The pure-numpy inside every round-2 proposal shares; the five proposals differ only in the surface.

`ChainReach` is what robot_reach hands over for ONE kinematic chain: grid points, the shared
orientation set, and every checked IK solution with its IK configuration. Nothing in here imports
robot_reach or robot_msgs; `from_result` (the producer) only reads a ReachResult's arrays, and
`Zone` takes any shape with `contains(points)` (robot_msgs' `SolidPrimitive` at the interface).

Workspace kinds are named by which orientations must work, never by a DOF count (Zach, 2026-11-23:
"how is hemisphere 5+ dof? thats not the meaning of thoose workspaces"):

    full        every grid point analysed, reachable or not
    reachable   at least one orientation works
    top_down    pointing straight down works at every roll (today's "four_dof")
    hemisphere  every approach within 90 deg of straight down works, every roll
    dexterous   every sampled orientation works

Every pose a sampler returns is a grid point x an orientation of the shared set that IK actually
solved, with that solution's joints: nothing is re-solved, nothing promised that was not checked.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ik_configurations import IKConfigurations

KINDS = ("full", "reachable", "top_down", "hemisphere", "dexterous")
RULE = {"full": "none", "reachable": "any", "top_down": "all", "hemisphere": "all", "dexterous": "all"}


@dataclass(frozen=True)
class Zone:
    """A region of the cell: a shape (anything with contains(points) in its own frame, e.g.
    robot_msgs.shape.SolidPrimitive) placed at `center` in the chain's frame (axis-aligned here)."""

    shape: object
    center: tuple[float, float, float]
    name: str = "zone"

    def contains(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(self.shape.contains(np.atleast_2d(points) - np.asarray(self.center)), bool)

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, eq=False)
class ChainReach:
    """One chain's checked reach, as plain arrays (the robot_msgs message would hold exactly these)."""

    chain: str
    frame: str
    joint_names: tuple[str, ...]
    base_xyz: np.ndarray  # (3,) where the chain's base sits in `frame`
    positions: np.ndarray  # (P, 3) every grid point analysed, in `frame`, m
    voxel_size: float  # m
    orientations: np.ndarray  # (O, 3, 3) one orientation set shared by every point
    tilt_deg: np.ndarray  # (O,) angle from straight down
    kind_orientations: dict  # kind -> (O,) bool: which orientations that kind asks for
    solution_pose: np.ndarray  # (K,) p * O + o, sorted
    solution_q: np.ndarray  # (K, dof) rad
    solution_configuration: np.ndarray  # (K,) uint8, index into `configurations`
    configurations: IKConfigurations

    @property
    def P(self) -> int:
        return len(self.positions)

    @property
    def O(self) -> int:
        return len(self.orientations)

    def everything(self) -> np.ndarray:
        return np.ones(len(self.configurations), bool)


def from_result(result, configurations: IKConfigurations, solution_configuration: np.ndarray, frame: str = "base_link") -> ChainReach:
    """robot_reach's side: read a ReachResult's arrays (and each solution's IK configuration)."""
    m = result.reach
    P, O, S = m.code.shape
    po = (m.sol_index // S).astype(np.int64)
    assert np.all(np.diff(po) >= 0), "sol_index must be sorted"
    o = m.orientations
    kinds = {"full": np.ones(O, bool), "reachable": np.ones(O, bool), "top_down": np.asarray(o.four_dof, bool),
             "hemisphere": np.asarray(o.hemisphere, bool), "dexterous": np.asarray(o.dexterous, bool)}
    return ChainReach(m.arm_name, frame, tuple(m.joint_names), np.zeros(3), m.positions.astype(float), float(result.config["step"]),
                      o.R.astype(float), np.degrees(o.tilt), kinds, po, m.sol_q, np.asarray(solution_configuration, np.uint8), configurations)


def placed(reach: ChainReach, chain: str, base_xyz, frame: str = "cell") -> ChainReach:
    """The same chain bolted down at `base_xyz` in a shared cell frame (the multi-chain stand-in)."""
    xyz = np.asarray(base_xyz, float)
    return replace(reach, chain=chain, frame=frame, base_xyz=xyz, positions=reach.positions + xyz,
                   configurations=replace(reach.configurations, chain=chain))


# ------------------------------------------------------------------------------------- membership
def ok_po(reach: ChainReach, allowed: np.ndarray) -> np.ndarray:
    """(P, O) bool: an IK solution in an allowed configuration exists at that pose."""
    keep = np.asarray(allowed, bool)[reach.solution_configuration]
    ok = np.zeros(reach.P * reach.O, bool)
    ok[reach.solution_pose[keep]] = True
    return ok.reshape(reach.P, reach.O)


def voxels(reach: ChainReach, orient: np.ndarray, rule: str, allowed: np.ndarray) -> np.ndarray:
    """(P,) bool: grid points in the workspace whose orientation set is `orient` under `rule`."""
    if rule == "none":
        return np.ones(reach.P, bool)
    ok = ok_po(reach, allowed)[:, orient]
    return ok.any(1) if rule == "any" else ok.all(1) & orient.any()


def rows(reach: ChainReach, orient: np.ndarray, rule: str, allowed: np.ndarray, voxel_mask: np.ndarray | None = None) -> np.ndarray:
    """Indices of the solutions a sampler may hand out: allowed configuration, inside the workspace
    (and `voxel_mask`), at one of the orientations the workspace asks for."""
    member = voxels(reach, orient, rule, allowed)
    if voxel_mask is not None:
        member = member & voxel_mask
    p, o = reach.solution_pose // reach.O, reach.solution_pose % reach.O
    return np.flatnonzero(np.asarray(allowed, bool)[reach.solution_configuration] & member[p] & orient[o])


# ------------------------------------------------------------------------------------- sampling
def draw(pose_key: np.ndarray, configuration: np.ndarray, n: int, k: int, rng: np.random.Generator,
         same_configuration: bool = True) -> np.ndarray:
    """(n, k) row indices: n draws of k DISTINCT poses each, uniform over poses (not solutions).

    same_configuration: all k poses of a draw share one IK configuration (picked in proportion to
    how many poses it has), so A -> B never asks the arm to change posture family on the way."""
    if len(pose_key) == 0:
        raise ValueError("nothing to sample: the workspace is empty")
    out = np.empty((n, k), np.int64)
    if same_configuration:
        order = np.lexsort((pose_key, configuration))
        ck = configuration[order].astype(np.int64) * (pose_key.max() + 1) + pose_key[order]
        first = np.r_[True, np.diff(ck) != 0]
        u_row, u_cfg = order[first], configuration[order][first]
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


DRAWN: list = []


@dataclass(frozen=True, eq=False)
class PoseSamples:
    """Drawn poses (shape (n,) or (n, k) for k waypoints) and the checked IK solution of each."""

    chain: str
    pose: np.ndarray  # (..., 4, 4) tool pose in the chain's frame
    q: np.ndarray  # (..., dof) rad: an IK solution that reaches it (verified by the library)
    configuration: np.ndarray  # (...,) index
    labels: tuple[str, ...]  # configuration index -> "front/up/J5+"
    base_xyz: np.ndarray  # (3,) so a checker can undo the cell placement

    def __post_init__(self):
        DRAWN.append(self)  # WHY: lets the runner FK-check every pose a snippet drew, inline ones too

    @property
    def shape(self) -> tuple[int, ...]:
        return self.configuration.shape

    def configuration_label(self, *index) -> str:
        return self.labels[int(self.configuration[index])]

    def lines(self, max_rows: int = 3) -> list[str]:
        """Human-readable: position mm, tilt from down (deg), configuration; for k > 1 the largest joint move."""
        out = []
        P = self.pose.reshape(-1, *self.pose.shape[-2:]) if self.configuration.ndim == 1 else self.pose
        if self.configuration.ndim == 1:
            for i in range(min(max_rows, len(P))):
                out.append(f"{_xyz(P[i])}  {_tilt(P[i])}  {self.labels[int(self.configuration[i])]}")
        else:
            for i in range(min(max_rows, len(P))):
                steps = " -> ".join(_xyz(P[i, j]) for j in range(P.shape[1]))
                d = np.abs(np.degrees(np.diff(self.q[i], axis=0)))
                d = np.minimum(d, 360 - d % 360).max(axis=0)  # per joint, the largest step along the tour
                j = int(np.argmax(d))
                out.append(f"{steps}  [{self.labels[int(self.configuration[i, 0])]}, largest joint step {d[j]:.0f} deg (J{j + 1})]")
        return out


def _xyz(T: np.ndarray) -> str:
    return "(" + ", ".join(f"{v:+.2f}" for v in T[:3, 3]) + ")"


def _tilt(T: np.ndarray) -> str:
    return f"tilt {np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1))):3.0f} deg"


def samples(reach: ChainReach, row_idx: np.ndarray) -> PoseSamples:
    """Rows of `reach` -> PoseSamples, keeping the shape of `row_idx`."""
    po = reach.solution_pose[row_idx]
    T = np.zeros((*row_idx.shape, 4, 4))
    T[..., :3, :3] = reach.orientations[po % reach.O]
    T[..., :3, 3] = reach.positions[po // reach.O]
    T[..., 3, 3] = 1.0
    return PoseSamples(reach.chain, T, reach.solution_q[row_idx].astype(float), reach.solution_configuration[row_idx],
                       reach.configurations.labels, reach.base_xyz)


# ------------------------------------------------------------------------------------- several chains
def cell_keys(positions: np.ndarray, voxel_size: float) -> np.ndarray:
    """Integer key of each grid point in the cell frame. WHY: chains placed a whole number of voxels
    apart share one grid, so 'both arms reach this point' is a key intersection, not a nearest search."""
    g = np.rint(positions / voxel_size).astype(np.int64) + (1 << 20)
    return (g[:, 0] << 42) | (g[:, 1] << 21) | g[:, 2]


def shared_draw(reaches: list[ChainReach], row_sets: list[np.ndarray], n: int, rng: np.random.Generator) -> tuple[list[np.ndarray], int]:
    """n grid points every chain reaches (through its own rows), and one row per chain at each."""
    keys = [cell_keys(r.positions[r.solution_pose[rs] // r.O], r.voxel_size) for r, rs in zip(reaches, row_sets)]
    common = keys[0]
    for k in keys[1:]:
        common = np.intersect1d(common, k)
    common = np.unique(common)
    if not len(common):
        raise ValueError("no grid point is reached by every chain")
    pick = common[rng.integers(len(common), size=n)]
    out = []
    for rs, k in zip(row_sets, keys):
        order = np.argsort(k, kind="stable")
        lo = np.searchsorted(k[order], pick, "left")
        hi = np.searchsorted(k[order], pick, "right")
        out.append(rs[order[lo + (rng.random(n) * (hi - lo)).astype(int)]])
    return out, len(common)


# ------------------------------------------------------------------------------------- robot_trajectories
def reach_map_like(reach: ChainReach, allowed: np.ndarray):
    """What robot_trajectories' ReachSet.from_reach_map duck-types today (bam_reach v1 field names):
    joint_names, joint_solutions (P, O, J) = first allowed solution or NaN, valid_mask (P, O),
    reach_scores (P,) = fraction of orientations that work."""
    from types import SimpleNamespace

    keep = np.flatnonzero(np.asarray(allowed, bool)[reach.solution_configuration])
    po = reach.solution_pose[keep]
    first = np.r_[True, np.diff(po) != 0]
    js = np.full((reach.P * reach.O, len(reach.joint_names)), np.nan, np.float32)
    js[po[first]] = reach.solution_q[keep[first]]
    valid = ok_po(reach, allowed)
    return SimpleNamespace(joint_names=reach.joint_names, joint_solutions=js.reshape(reach.P, reach.O, -1),
                           valid_mask=valid, reach_scores=valid.mean(1))
