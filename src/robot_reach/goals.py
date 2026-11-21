"""Solve for the best rest pose of a use case: a `PoseGoal` (a few settings) -> gates, objectives, a
Pareto set, the pick and a NamedPose-ready record. Any actuated rigid body that is an `Arm`.

Zach, 2026-11-21: *"you have a robot arm. and then we basically like define a workspace where you
want to do picking and then like zones that you need to stay clear of. and then within that zone you
can pick the ideal position ... there is just a few things you can set here that can help us solve
for any use case on any actuated rigid body."* His four cases, as goals:

- belt empty: hover just above the belt (``clearances={"plane": 0.020}``, held, energy first);
- machine off: rest even on the belt at zero energy (``clearances={"plane": 0.0}``, ``torque="limp"``);
- picking, but not this arm: well clear of the belt, low energy (``{"plane": 0.120}``, held);
- ready: the wrist where it picks everything quickly (a `Workspace`, ``weights={"readiness": ...}``).

A goal's settings (the contract):

1. ``clearances``: per scene zone the least distance in m — ``"plane"`` (the support plane; its
   label, e.g. ``"belt"``, works too), each keep-out by name, ``"lane"`` (how far inside its lane the
   arm stays, toward a neighbour). ``0`` = touching allowed, ``None`` = ignored.
2. ``torque``: ``"held"`` (energy = copper W at the pose) or ``"limp"`` (torque off: the pose must be
   a passive rest — a potential minimum through the arm's own closure, gravity pressing joints into
   their stops, static friction, or a support it may touch — and settle within ``settle_deg``;
   energy = 0).
3. ``workspace``: the task poses the arm must be ready to reach; ``readiness`` = the time from the
   pose to each (trapezoid per joint under the arm's velocity / acceleration limits, the slowest
   joint sets it), worst and mean over the workspace.
4. ``weights`` over the objectives ``energy`` (W), ``readiness`` (s), ``compactness`` (the hand's
   reach from the reference line, m) and ``occlusion`` (the fraction of the workspace a `Camera` loses
   behind the arm).

Then gates first (clearances, limits with ``stop_margin_deg``, closure, self-collision, torque
limits when held, a passive rest when limp, your own `Gate`s), then a weighted score over objectives
normalised over the poses that pass (or your ``anchors``). `solve` samples the arm's joint space
(closed chains through their FK), refines the best few locally and returns the Pareto set, the pick
and its record.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .codes import Code
from .rest import Gate, RestReport, RestStudy, default_q, wrap_into_limits

OBJECTIVES = ("energy", "readiness", "compactness", "occlusion")
UNITS = {"energy": "W", "readiness": "s", "compactness": "m", "occlusion": ""}
DEFAULT_VELOCITY = 1.0  # rad/s, when the arm states none (assumption, reported)
DEFAULT_ACCELERATION = 2.0  # rad/s^2
G = 9.81


# =========================================================================== the settings
@dataclass(frozen=True, eq=False)
class Workspace:
    """The task poses the arm must be ready to reach.

    Give them as tool poses (`targets`, (K, 4, 4) in the arm frame; `task` says what counts: "pose",
    "axis" — position and the tool +z — or "position"), solved once by the arm's IK or a numeric IK,
    or as joint configurations your own planner chose (`joint_targets`, (K, dof) or (K, S, dof) with
    S branches each; NaN rows = unreachable). `region` (P, 3) are the points a camera must see
    (default: the targets' positions). `stat` is the readiness number scored: "mean" or "worst"."""

    name: str = "workspace"
    targets: np.ndarray | None = None
    task: str = "position"
    joint_targets: np.ndarray | None = None
    region: np.ndarray | None = None
    stat: Literal["mean", "worst"] = "mean"
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class Camera:
    """A pinhole looking from `position` along `look` (default: at the workspace region's centre).
    `resolution_deg` is one bin of the occlusion z-buffer; `margin_bins` grows the arm's silhouette (a
    perception mask's margin)."""

    position: tuple[float, float, float]
    look: tuple[float, float, float] | None = None
    resolution_deg: float = 0.25
    margin_bins: int = 1
    name: str = "camera"


@dataclass(frozen=True, eq=False)
class PoseGoal:
    """One use case (module docstring). Besides the four settings: `settle_deg` (limp), the stop
    margin every joint keeps, `search` (a joint held at a value or kept in an interval: the search
    space, not the use case), extra `gates`, optional `anchors` ({objective: (best, worst)}), a
    `camera` for occlusion, and `friction` (limp: count the arm's static friction, or bound without).
    A limp pose must also survive a `kick_deg` nudge of any one joint either way: the arm falls back
    to the rest, or friction keeps it where the nudge left it — it ends within max(settle, kick) of
    the pose (an unstable balance — a pendulum standing up on its friction — tips over: not a rest)."""

    name: str
    clearances: Mapping[str, float | None] = field(default_factory=dict)
    torque: Literal["held", "limp"] = "held"
    workspace: Workspace | None = None
    weights: Mapping[str, float] = field(default_factory=lambda: {"energy": 1.0})
    description: str = ""
    camera: Camera | None = None
    settle_deg: float = 3.0
    kick_deg: float = 3.0
    stop_margin_deg: float = 0.0
    search: Mapping[str, float | tuple[float, float]] = field(default_factory=dict)
    gates: tuple[Gate, ...] = ()
    anchors: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    friction: bool = True

    def __post_init__(self):
        if self.torque not in ("held", "limp"):
            raise ValueError(f"PoseGoal.torque is 'held' or 'limp', not {self.torque!r}")
        for k in self.weights:
            if k not in OBJECTIVES:
                raise ValueError(f"PoseGoal.weights: {k!r} is not one of {OBJECTIVES}")
        if self.weights.get("readiness") and self.workspace is None:
            raise ValueError(f"{self.name}: readiness is weighted but there is no workspace")
        if self.weights.get("occlusion") and self.camera is None:
            raise ValueError(f"{self.name}: occlusion is weighted but there is no camera")

    def settings(self) -> dict:
        """The goal's settings as plain data (the record's why, the viewer's panel)."""
        return {
            "clearances_mm": {k: (None if v is None else round(1e3 * v, 1)) for k, v in self.clearances.items()},
            "torque": self.torque,
            "settle_deg": self.settle_deg if self.torque == "limp" else None,
            "kick_deg": self.kick_deg if self.torque == "limp" else None,
            "friction": self.friction if self.torque == "limp" else None,
            "workspace": None if self.workspace is None else {"name": self.workspace.name, "poses": _n_targets(self.workspace),
                                                              "readiness": self.workspace.stat},
            "weights": dict(self.weights),
            "stop_margin_deg": self.stop_margin_deg,
            "search": {k: (round(math.degrees(v), 2) if not isinstance(v, (tuple, list)) else [round(math.degrees(x), 2) for x in v])
                       for k, v in self.search.items()},
            "gates": [g.name for g in self.gates],
            "camera": None if self.camera is None else self.camera.name,
            "anchors": {k: list(v) for k, v in self.anchors.items()},
        }


def _n_targets(ws: Workspace) -> int:
    if ws.joint_targets is not None:
        return len(ws.joint_targets)
    return 0 if ws.targets is None else len(ws.targets)


# =========================================================================== readiness
def drive_limits(arm) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(velocity, acceleration) per joint and the assumptions made where the arm states none."""
    notes = []
    v = np.asarray(arm.velocity_limits, float) if arm.velocity_limits is not None else None
    a = np.asarray(arm.acceleration_limits, float) if arm.acceleration_limits is not None else None
    if v is None:
        v = np.full(arm.dof, DEFAULT_VELOCITY)
        notes.append(f"no velocity_limits on the arm: {DEFAULT_VELOCITY} per joint assumed")
    if a is None:
        a = np.full(arm.dof, DEFAULT_ACCELERATION)
        notes.append(f"no acceleration_limits on the arm: {DEFAULT_ACCELERATION} per joint assumed")
    return v, a, notes


def trapezoid_time(dq: np.ndarray, vmax: np.ndarray, amax: np.ndarray) -> np.ndarray:
    """Rest-to-rest time per joint for moves `dq` (..., dof): a trapezoid, or a triangle when short."""
    d = np.abs(dq)
    tri = d <= vmax**2 / amax
    return np.where(tri, 2.0 * np.sqrt(d / amax), d / vmax + vmax / amax)


def readiness(q: np.ndarray, targets: np.ndarray, valid: np.ndarray, vmax, amax) -> tuple[np.ndarray, float, float]:
    """Seconds from `q` to each workspace pose: per target the fastest of its branches, per branch the
    slowest joint. Returns (times (K,), mean, worst) over the reachable targets."""
    t = trapezoid_time(targets - q, vmax, amax).max(axis=-1)  # (K, S)
    t = np.where(valid, t, np.inf).min(axis=1)
    reach = np.isfinite(t)
    if not reach.any():
        return t, math.nan, math.nan
    return t, float(np.mean(t[reach])), float(np.max(t[reach]))


def workspace_configurations(arm, ws: Workspace, seeds: Sequence[np.ndarray] = ()) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(K, S, dof) joint configurations of the workspace and their validity (K, S): the given ones,
    else the arm's IK, else a numeric IK (urdf.numeric_ik through the arm's FK and Jacobian) from
    `seeds` + samples. Every branch is re-checked (limits, closure, collision)."""
    notes = []
    if ws.joint_targets is not None:
        Q = np.asarray(ws.joint_targets, float)
        if Q.ndim == 2:
            Q = Q[:, None, :]
        notes.append("workspace: joint configurations given by the caller (its planner)")
    else:
        if ws.targets is None:
            raise ValueError(f"workspace {ws.name!r} has neither targets nor joint_targets")
        T = np.asarray(ws.targets, float).reshape(-1, 4, 4)
        if arm.ik is not None:
            Q, code = arm.inverse(T)
            notes.append("workspace: the arm's own IK, every branch")
        else:
            from .urdf import into_limits, numeric_ik

            home = seeds[0] if len(seeds) else default_q(arm)
            ik = numeric_ik(lambda q: (arm.forward(q)[0], arm.jacobian(q)), arm.lower, arm.upper, task=ws.task,
                            n_seeds=8, home=home, pos_tol=1e-6, rot_tol=1e-5, max_iter=60)
            Q, code = ik(T)
            Q = into_limits(Q.reshape(-1, arm.dof), np.asarray(arm.lower), np.asarray(arm.upper), np.ones(arm.dof, bool)).reshape(Q.shape)
            notes.append("workspace: a numeric IK (damped least squares, 8 starts) through the arm's FK")
        Q = np.where((code == int(Code.OK))[..., None], Q, np.nan)
    K, S, dof = Q.shape
    flat = Q.reshape(-1, dof)
    good = np.all(np.isfinite(flat), axis=1)
    valid = np.zeros(len(flat), bool)
    if good.any():
        codes = arm.check(flat[good], torque=False)
        valid[np.nonzero(good)[0]] = codes == int(Code.OK)
    valid = valid.reshape(K, S)
    return np.where(valid[..., None], Q, 0.0), valid, notes


# =========================================================================== occlusion
_SAMPLES: dict = {}


def surface_samples(arm, spacing: float = 0.004, cap: int = 60_000) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    """Points spread evenly over the arm's drawn meshes (area-weighted, ~`spacing` apart), per mesh in
    its own frame: what a camera sees (mesh vertices bunch at details and leave flat faces empty).
    Returns (mesh indices, [(mesh index, (P, 3))])."""
    key = (id(arm), spacing, cap)
    if key in _SAMPLES:
        return _SAMPLES[key]
    rng = np.random.default_rng(0)
    meshes = [(i, m) for i, m in enumerate(arm.meshes) if m.draw and m.faces is not None]
    areas = []
    for _, m in meshes:
        v, f = np.asarray(m.vertices, float), np.asarray(m.faces, int)
        areas.append(0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1))
    total = float(sum(a.sum() for a in areas))
    n_total = min(cap, int(total / spacing**2)) if total > 0 else 0
    out = []
    for (i, m), a in zip(meshes, areas):
        n = round(n_total * a.sum() / total) if total > 0 else 0
        if n == 0:
            continue
        v, f = np.asarray(m.vertices, float), np.asarray(m.faces, int)
        tri = rng.choice(len(f), size=n, p=a / a.sum())
        r1, r2 = rng.random(n), rng.random(n)
        s = np.sqrt(r1)
        p = (1 - s)[:, None] * v[f[tri, 0]] + (s * (1 - r2))[:, None] * v[f[tri, 1]] + (s * r2)[:, None] * v[f[tri, 2]]
        out.append((i, p))
    _SAMPLES[key] = ([i for i, _ in out], out)
    return _SAMPLES[key]


def arm_points_for_camera(arm, q: np.ndarray, max_per_mesh: int | None) -> np.ndarray:
    _, parts = surface_samples(arm)
    if parts and arm.mesh_poses is not None:
        T = np.asarray(arm.mesh_poses(q[None]), float)[0]
        pts = np.concatenate([p @ T[i, :3, :3].T + T[i, :3, 3] for i, p in parts])
    else:
        pts = arm.surface(q[None], max_per_mesh=max_per_mesh)[0]
    return pts[np.all(np.isfinite(pts), axis=1)]


def occlusion(camera: Camera, region: np.ndarray, arm_points: np.ndarray) -> float:
    """Fraction of `region` points the arm hides from `camera`: a z-buffer of the arm's points in the
    camera's image (bins of `resolution_deg`, the silhouette grown by `margin_bins`)."""
    region = np.asarray(region, float)
    c = np.asarray(camera.position, float)
    look = np.asarray(camera.look, float) if camera.look is not None else region.mean(0) - c
    look = look / np.linalg.norm(look)
    ref = np.array([1.0, 0.0, 0.0]) if abs(look[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(look, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(look, e1)

    def image(p):
        d = p - c
        z = d @ look
        return d @ e1 / z, d @ e2 / z, z

    ru, rv, rz = image(region)
    bin_ = math.tan(math.radians(camera.resolution_deg))
    pad = (camera.margin_bins + 2) * bin_
    u0, v0 = ru.min() - pad, rv.min() - pad
    nu, nv = int((ru.max() + pad - u0) / bin_) + 1, int((rv.max() + pad - v0) / bin_) + 1
    zbuf = np.full((nu, nv), np.inf)
    if len(arm_points):
        au, av, az = image(arm_points)
        iu, iv = np.floor((au - u0) / bin_).astype(int), np.floor((av - v0) / bin_).astype(int)
        inside = (az > 1e-6) & (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
        np.minimum.at(zbuf, (iu[inside], iv[inside]), az[inside])
    grown = zbuf.copy()
    for du in range(-camera.margin_bins, camera.margin_bins + 1):
        for dv in range(-camera.margin_bins, camera.margin_bins + 1):
            grown = np.minimum(grown, np.roll(np.roll(zbuf, du, axis=0), dv, axis=1))
    iu, iv = np.floor((ru - u0) / bin_).astype(int), np.floor((rv - v0) / bin_).astype(int)
    hidden = grown[iu, iv] < rz - 1e-3
    return float(np.mean(hidden))


# =========================================================================== limp: a passive rest
@dataclass(frozen=True)
class Settle:
    q_end: np.ndarray
    moved_deg: float  # the most any joint moved
    how: str  # what holds it: a potential minimum, stops, friction, a support
    on_support: str = ""  # the zone it came to rest on
    path: tuple = ()  # a few configurations along the way (for clearance along the settle)
    closes: bool = True


def potential(arm, q: np.ndarray) -> np.ndarray:
    """g · Σ m z of the arm's point masses, (N,), J. Needs mass_points + masses."""
    q = np.atleast_2d(q)
    return G * (arm.mass_points(q)[..., 2] @ np.asarray(arm.masses, float))


def settle(study: RestStudy, q0, *, friction: bool = True, supports: Sequence[str] = (), max_deg: float = 180.0,
           step_deg: float = 0.5, min_step_deg: float = 0.02, max_steps: int = 400) -> Settle:
    """Where the arm comes to rest with every torque off, starting still at `q0`.

    Gradient flow on the potential (the arm's point masses through its own FK, so a closed chain stays
    closed): joints move down the gravity torque; a joint at a limit (the hard stop) stays when gravity
    presses it into the stop; with `friction`, a joint whose gravity torque is under its static friction
    (Arm.friction) stays. A step that would break the closure (FK fails) or climb is halved. `supports`
    ("plane" / keep-out names) may be touched: the arm stops where it first touches one — sliding or
    folding about the contact is NOT modelled (stated assumption). Stops once a joint has moved
    `max_deg` (a caller's fail-fast)."""
    arm = study.arm
    q = np.asarray(q0, float).copy()
    lo, hi = np.asarray(arm.lower, float), np.asarray(arm.upper, float)
    fr = np.asarray(arm.friction, float) if (friction and arm.friction is not None) else np.zeros(arm.dof)
    step = math.radians(step_deg)
    U = float(potential(arm, q)[0])
    path = [q.copy()]
    reasons: set = set()
    on = ""
    for _ in range(max_steps):
        tau = np.asarray(arm.gravity_torque(q[None])[0], float)  # dU/dq: the arm accelerates along -tau
        free = np.abs(tau) > fr
        held_friction = ~free & (np.abs(tau) > 1e-9)
        push = -tau
        at_lo, at_hi = q <= lo + 1e-9, q >= hi - 1e-9
        on_stop = (at_lo & (push < 0)) | (at_hi & (push > 0))
        moving = free & ~on_stop
        if not moving.any():
            reasons |= {f"stop:{n}" for n in np.array(arm.joint_names)[on_stop & free]}
            reasons |= {f"friction:{n}" for n in np.array(arm.joint_names)[held_friction]}
            break
        d = np.where(moving, push, 0.0)
        accepted = False
        while step >= math.radians(min_step_deg):
            trial = np.clip(q + d / np.max(np.abs(d)) * step, lo, hi)
            _, ok = arm.forward(trial[None])
            if ok[0]:
                Ut = float(potential(arm, trial)[0])
                if Ut < U - 1e-12:
                    accepted = True
                    break
            step *= 0.5
        if not accepted:
            reasons.add("minimum")
            break
        if supports:
            touched = _touching(study, trial, supports)
            if touched:
                on = touched
                reasons.add(f"support:{touched}")
                break
        q, U = trial, Ut
        step = min(step * 1.5, math.radians(step_deg))
        path.append(q.copy())
        if np.max(np.abs(q - q0)) > math.radians(max_deg):
            reasons.add("still moving")
            break
    names = sorted(reasons)
    how = "; ".join(_describe(r) for r in names) or "minimum"
    moved = float(np.degrees(np.max(np.abs(q - np.asarray(q0, float)))))
    stride = max(1, len(path) // 8)
    return Settle(q_end=q, moved_deg=moved, how=how, on_support=on, path=tuple(path[::stride]) + (q,))


def _describe(reason: str) -> str:
    kind, _, what = reason.partition(":")
    return {"stop": f"{what} on its stop", "friction": f"{what} held by friction", "support": f"rests on the {what}",
            "minimum": "a potential minimum", "still moving": "still moving"}.get(kind, reason)


def _touching(study: RestStudy, q: np.ndarray, supports: Sequence[str]) -> str:
    scene = study.scene
    pts = study.arm.surface(q[None], max_per_mesh=60)[0]
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if not len(pts):
        return ""
    abh = scene.coords(pts)
    if "plane" in supports:
        over = scene.in_region(abh)
        if over.any() and float(np.min(scene.above_plane(pts[over]))) < 0.0:
            return scene.label("plane", "plane")
    for ko in scene.keepouts:
        if ko.name in supports:
            sel = _inside(abh[:, 0], ko.across) & _inside(abh[:, 1], ko.along)
            if sel.any() and float(np.min(abh[sel, 2] - ko.top)) < 0.0:
                return ko.name
    return ""


def _inside(x, iv):
    if iv is None:
        return np.ones(x.shape, bool)
    return (x >= iv[0]) & (x <= iv[1])


# =========================================================================== candidates and the solution
@dataclass(eq=False)
class Candidate:
    q: np.ndarray
    passed: bool
    failed: tuple[str, ...]
    objectives: dict  # name -> value (SI), None when not measured
    report: RestReport | None = None
    settle: Settle | None = None
    readiness_mean: float | None = None
    readiness_worst: float | None = None
    score: float = math.nan
    source: str = "sample"
    frictionless: Settle | None = None  # limp: where it would end with no friction at all (information)

    def to_dict(self, arm) -> dict:
        return {"q_rad": dict(zip(arm.joint_names, map(float, self.q))), "passed": self.passed, "failed": list(self.failed),
                "objectives": {k: (None if v is None or not np.isfinite(v) else float(v)) for k, v in self.objectives.items()},
                "readiness_mean_s": self.readiness_mean, "readiness_worst_s": self.readiness_worst, "score": _num(self.score),
                "source": self.source, "settle": None if self.settle is None else
                {"moved_deg": self.settle.moved_deg, "how": self.settle.how, "q_end_rad": list(map(float, self.settle.q_end))},
                "frictionless": None if self.frictionless is None else
                {"moved_deg": self.frictionless.moved_deg, "how": self.frictionless.how}}


def _num(x):
    return None if x is None or not np.isfinite(x) else float(x)


@dataclass(eq=False)
class Solution:
    goal: PoseGoal
    pick: Candidate | None
    pareto: list[Candidate]
    sampled: int
    passed: int
    seconds: float
    assumptions: list[str]
    gate_failures: dict  # gate -> how many sampled poses it refused
    record: dict = field(default_factory=dict)
    workspace_q: np.ndarray | None = None  # (K, S, dof) the workspace's configurations, valid (K, S) beside
    workspace_valid: np.ndarray | None = None
    norm: dict = field(default_factory=dict)  # objective -> (best, worst) the scores were normalised on

    def score_of(self, candidate: Candidate) -> float:
        """Another pose's score on this solution's scale (comparing a hand-picked pose with the pick)."""
        return _score(candidate, self.goal, self.norm) if candidate.passed else math.nan

    def to_dict(self, arm) -> dict:
        return {"goal": self.goal.name, "settings": self.goal.settings(), "sampled": self.sampled, "passed": self.passed,
                "norm": {k: list(v) for k, v in self.norm.items()},
                "seconds": self.seconds, "assumptions": self.assumptions, "gate_failures": self.gate_failures,
                "pick": None if self.pick is None else self.pick.to_dict(arm),
                "pareto": [c.to_dict(arm) for c in self.pareto], "record": self.record}


class _Evaluator:
    """Gates and objectives of one pose under one goal (shared normalisation set by `solve`)."""

    def __init__(self, study: RestStudy, goal: PoseGoal, ws_q, ws_valid, vmax, amax, per_mesh):
        self.study, self.goal = study, goal
        self.arm, self.scene = study.arm, study.scene
        self.ws_q, self.ws_valid, self.vmax, self.amax = ws_q, ws_valid, vmax, amax
        self.per_mesh = per_mesh
        # the stop margin binds the joints the search moves; a joint the goal fixes only stays inside its limits
        self.lo, self.hi, fixed = _search_bounds(self.arm, goal, margins_only=True)
        self.lo = np.where(fixed, self.arm.lower, self.lo)
        self.hi = np.where(fixed, self.arm.upper, self.hi)
        self.plane_keys = {"plane", self.scene.label("plane", "plane")}
        self.supports = [("plane" if k in self.plane_keys else k) for k, v in goal.clearances.items() if v is not None and v <= 0.0]
        self.region = None
        if goal.camera is not None:
            ws = goal.workspace
            if ws is not None and ws.region is not None:
                self.region = np.asarray(ws.region, float)
            elif ws is not None and ws.targets is not None:
                self.region = np.asarray(ws.targets, float)[:, :3, 3]
            else:
                raise ValueError(f"{goal.name}: a camera needs workspace.region or targets")
        self.failures: dict = {}

    def __call__(self, q: np.ndarray, *, full: bool = False, source: str = "sample") -> Candidate:
        goal, arm = self.goal, self.arm
        failed: list[str] = []
        if np.any(q < self.lo - 1e-4) or np.any(q > self.hi + 1e-4):  # 1e-4 rad: a pose written to rounding, not a hair outside
            failed.append("limits" + (f" (>= {goal.stop_margin_deg:g} deg inside)" if goal.stop_margin_deg else ""))
        rep = self.study.measure(q, hull=full, max_per_mesh=None if full else self.per_mesh)
        if not rep.ok:
            failed.append("closes")
            return self._done(Candidate(q, False, tuple(failed), {}, rep, source=source))
        hit = arm.collides(q[None])
        if hit is not None and bool(hit[0]):
            failed.append("self-collision")
        failed += self._clearances(rep)
        settle_ = None
        if goal.torque == "held":
            h = rep.holding
            if h.limit_frac is not None and np.any(h.limit_frac > 1.0):
                failed.append("torque limits")
            energy = h.total
        else:
            energy = 0.0
            if not failed or full:
                settle_ = settle(self.study, q, friction=goal.friction, supports=self.supports, max_deg=goal.settle_deg + 2.0)
                if settle_.moved_deg > goal.settle_deg or "still moving" in settle_.how:
                    failed.append(f"passive rest (settles <= {goal.settle_deg:g} deg)")
                else:
                    if not np.allclose(settle_.q_end, q):
                        end = self.study.measure(settle_.q_end, hull=False, max_per_mesh=self.per_mesh)
                        failed += [f + " (after settling)" for f in self._clearances(end)]
                    if not failed and not self._kicks_hold(q):
                        failed.append(f"passive rest (stays after a {goal.kick_deg:g} deg kick)")
        for g in goal.gates:
            if not g.passes(rep):
                failed.append(g.name)
        obj = {"energy": energy, "compactness": None if rep.footprint is None else rep.footprint.reach}
        r_mean = r_worst = None
        if self.ws_q is not None:
            _, r_mean, r_worst = readiness(q, self.ws_q, self.ws_valid, self.vmax, self.amax)
            obj["readiness"] = r_mean if goal.workspace.stat == "mean" else r_worst
        if goal.camera is not None and (full or not failed):
            obj["occlusion"] = occlusion(goal.camera, self.region, arm_points_for_camera(arm, q, self.per_mesh))
        c = Candidate(q, not failed, tuple(failed), obj, rep, settle_, r_mean, r_worst, source=source)
        return self._done(c)

    def _kicks_hold(self, q: np.ndarray) -> bool:
        """Every one-joint nudge of kick_deg either way settles back within settle_deg of q."""
        goal, arm = self.goal, self.arm
        kick = math.radians(goal.kick_deg)
        for i in range(arm.dof):
            if self.hi[i] - self.lo[i] < 1e-9 and np.isclose(arm.lower[i], arm.upper[i]):
                continue
            for sign in (+1.0, -1.0):
                start = q.copy()
                start[i] = np.clip(start[i] + sign * kick, arm.lower[i], arm.upper[i])
                if start[i] == q[i]:
                    continue
                s = settle(self.study, start, friction=goal.friction, supports=self.supports,
                           max_deg=max(goal.settle_deg, goal.kick_deg) + goal.kick_deg + 2.0)
                # WHY against q: a rest the nudge falls back into, or a joint friction keeps where the nudge left it, both
                # end within max(settle, kick) of q; a balance the nudge tips over ends far from it
                away = float(np.degrees(np.max(np.abs(s.q_end - q))))
                if not s.closes or away > max(goal.settle_deg, goal.kick_deg) + 0.5 or "still moving" in s.how:
                    return False
        return True

    def _done(self, c: Candidate) -> Candidate:
        for f in c.failed:
            key = f.split(" (")[0]
            self.failures[key] = self.failures.get(key, 0) + 1
        return c

    def _clearances(self, rep: RestReport) -> list[str]:
        out = []
        c, f = rep.clearance, rep.footprint
        for zone, need in self.goal.clearances.items():
            if need is None:
                continue
            if zone in self.plane_keys:
                have = math.inf if c.plane is None else c.plane
                label = self.scene.label("plane", "plane")
            elif zone == "lane":
                have = math.inf if f.past_shoulder is None else -f.past_shoulder
                label = "lane"
            else:
                if zone not in c.keepouts:
                    raise KeyError(f"{self.goal.name}: no zone {zone!r}; the scene has plane, lane and {list(c.keepouts)}")
                have = math.inf if c.keepouts[zone] is None else c.keepouts[zone]
                label = zone
            if have < need - 1e-9:
                out.append(f"clearance: {label} >= {1e3 * need:.0f} mm")
        return out


def _scores(cands: Sequence[Candidate], goal: PoseGoal) -> dict:
    """{objective: (best, worst)} — the goal's anchors, else the best and worst among `cands`."""
    out = {}
    for k, w in goal.weights.items():
        if not w:
            continue
        if k in goal.anchors:
            out[k] = tuple(map(float, goal.anchors[k]))
            continue
        vals = [c.objectives.get(k) for c in cands if c.objectives.get(k) is not None and np.isfinite(c.objectives.get(k))]
        out[k] = (min(vals), max(vals)) if vals else (0.0, 1.0)
    return out


def _score(c: Candidate, goal: PoseGoal, norm: dict) -> float:
    total = weight = 0.0
    for k, (best, worst) in norm.items():
        w = float(goal.weights[k])
        v = c.objectives.get(k)
        if v is None or not np.isfinite(v):
            s = 0.0
        elif best == worst:
            s = 1.0
        else:
            s = float(np.clip((v - worst) / (best - worst), 0.0, 1.0))
        total += w * s
        weight += w
    return total / weight if weight else 0.0


def pareto(cands: Sequence[Candidate], keys: Sequence[str]) -> list[Candidate]:
    """The non-dominated candidates over `keys` (all lower-is-better)."""
    vals = np.array([[c.objectives.get(k) if c.objectives.get(k) is not None else np.inf for k in keys] for c in cands], float)
    keep = []
    for i in range(len(cands)):
        dominated = np.any(np.all(vals <= vals[i], axis=1) & np.any(vals < vals[i], axis=1))
        if not dominated:
            keep.append(cands[i])
    return keep


def _search_bounds(arm, goal: PoseGoal, margins_only: bool = False):
    """(lo, hi, fixed): the limits less the stop margin, narrowed by the goal's search intervals (unless
    `margins_only`, the gate's bounds), a fixed joint's both bounds at its value."""
    margin = math.radians(goal.stop_margin_deg)
    lo = np.asarray(arm.lower, float) + margin
    hi = np.asarray(arm.upper, float) - margin
    fixed = np.zeros(arm.dof, bool)
    for name, spec in goal.search.items():
        i = arm.joint_names.index(name)
        if isinstance(spec, (tuple, list)):
            if not margins_only:
                lo[i], hi[i] = max(lo[i], float(spec[0])), min(hi[i], float(spec[1]))
        else:
            fixed[i] = True
            lo[i] = hi[i] = float(spec)
    return lo, hi, fixed


def solve(study: RestStudy, goal: PoseGoal, *, samples: int = 1500, refine: int = 3, seed: int = 0, screen_per_mesh: int = 120,
          seeds: Sequence[np.ndarray] = (), progress: Callable[[str], None] | None = None) -> Solution:
    """The best pose for `goal` (module docstring). `samples` uniform joint samples inside the search
    bounds (closed chains through their FK: a pose that does not close fails its gate), plus `seeds`
    (poses you already know, e.g. named poses); limp goals let each sample settle and take where it
    rests. The best `refine` passers are refined by a pattern search on the score (normalisation frozen
    from the samples). Screening measures `screen_per_mesh` vertices per mesh; the pick and the Pareto
    set are re-measured with the study's own."""
    t0 = time.perf_counter()
    arm = study.arm
    say = progress or (lambda _m: None)
    assumptions: list[str] = []
    rng = np.random.default_rng(seed)
    lo, hi, fixed = _search_bounds(arm, goal)
    ws_q = ws_valid = None
    vmax, amax, notes = drive_limits(arm)
    if goal.workspace is not None:
        assumptions += notes
        ws_q, ws_valid, wnotes = workspace_configurations(arm, goal.workspace, seeds=[*seeds, default_q(arm)])
        assumptions += wnotes
        assumptions.append(f"workspace: {int(ws_valid.any(axis=1).sum())} of {len(ws_valid)} poses reachable; readiness = per joint "
                           "trapezoid, the slowest joint, the fastest branch; the path between is not checked")
    if goal.torque == "limp":
        assumptions.append("limp: the joint limits are the hard stops; gravity flow on the arm's point masses (no inertia: no overshoot)")
        if goal.friction and arm.friction is not None:
            assumptions.append("limp: each joint holds while its gravity torque is under Arm.friction (static Coulomb)")
        else:
            assumptions.append("limp: frictionless (no Arm.friction, or the goal asks for the bound)")
        if any(v is not None and v <= 0 for v in goal.clearances.values()):
            assumptions.append("limp: a zone with clearance 0 may be touched and holds the arm where it first touches (no sliding)")
    ev = _Evaluator(study, goal, ws_q, ws_valid, vmax, amax, screen_per_mesh)

    # ---- sample: the arm's own sampler (study.sampler), else uniform inside the search bounds
    if study.sampler is not None:
        Q = np.clip(np.asarray(study.sampler(rng, samples, lo.copy(), hi.copy()), float).reshape(samples, arm.dof), lo, hi)
        assumptions.append("samples from the study's own sampler")
    else:
        Q = lo + rng.random((samples, arm.dof)) * (hi - lo)
    starts = [wrap_into_limits(arm, np.asarray(s, float)) for s in seeds] + list(Q)
    cands: list[Candidate] = []
    say(f"{goal.name}: sampling {len(starts)} poses")
    for k, q in enumerate(starts):
        q = np.clip(np.where(fixed, lo, q), lo, hi)  # a seed too stays inside the goal's search space
        src = "seed" if k < len(seeds) else "sample"
        if goal.torque == "limp":
            _, ok = arm.forward(q[None])
            if not ok[0]:
                ev._done(Candidate(q, False, ("closes",), {}))
                continue
            # WHY both: friction or a stop may already hold the sample where it is (a rest inside an envelope the
            # potential's own minimum lies outside of); otherwise follow gravity to where it does come to rest
            cands.append(ev(q, source=src))
            if cands[-1].passed:
                continue
            s = settle(study, q, friction=goal.friction, supports=ev.supports, max_deg=180.0, step_deg=2.0, max_steps=150)
            q = _back_off(s.q_end, lo, hi, arm)
            src += "→rest"
        cands.append(ev(q, source=src))
    passers = [c for c in cands if c.passed]
    say(f"{goal.name}: {len(passers)} of {len(cands)} pass")
    norm = _scores(passers, goal)
    for c in cands:
        c.score = _score(c, goal, norm) if c.passed else math.nan

    # ---- refine (held goals): pattern search from the best few, normalisation frozen
    if goal.torque == "held" and passers and refine:
        best = sorted(passers, key=lambda c: -c.score)
        starts_r, picked = [], []
        for c in best:
            if all(np.max(np.abs(c.q - p.q)) > math.radians(10) for p in picked):
                picked.append(c)
            if len(picked) >= refine:
                break
        for c in picked:
            r = _refine(ev, c, goal, norm, lo, hi, fixed)
            cands.append(r)
            starts_r.append(r)
        passers = [c for c in cands if c.passed]

    # ---- the pick and the Pareto set, re-measured with the study's own fidelity
    keys = [k for k, w in goal.weights.items() if w]
    front = pareto(passers, keys) if passers else []
    front = sorted(front, key=lambda c: (-c.score, -(c.report.clearance.plane or 0.0) if c.report and c.report.clearance else 0.0))
    front = _distinct(front, math.radians(1.0))[:8]
    final = []
    for c in front:
        f = ev(c.q, full=True, source=c.source)
        f.score = _score(f, goal, norm)
        if goal.torque == "limp":
            f.frictionless = settle(study, c.q, friction=False, supports=ev.supports, max_deg=180.0)
        final.append(f)
    final = [c for c in final if c.passed] or final
    final.sort(key=lambda c: -c.score)
    pick = final[0] if final else None
    sol = Solution(goal=goal, pick=pick, pareto=final, sampled=len(cands), passed=len(passers), seconds=time.perf_counter() - t0,
                   assumptions=assumptions, gate_failures=dict(sorted(ev.failures.items(), key=lambda kv: -kv[1])),
                   workspace_q=ws_q, workspace_valid=ws_valid, norm=norm)
    sol.record = record(study, sol)
    say(f"{goal.name}: done in {sol.seconds:.0f} s")
    return sol


def _back_off(q: np.ndarray, lo: np.ndarray, hi: np.ndarray, arm) -> np.ndarray:
    """A rest found against a hard stop, commanded `stop_margin` inside it (it settles back onto the stop)."""
    return np.clip(q, lo, hi)


def _distinct(cands: list[Candidate], tol: float) -> list[Candidate]:
    out: list[Candidate] = []
    for c in cands:
        if all(np.max(np.abs(c.q - o.q)) > tol for o in out):
            out.append(c)
    return out


def _refine(ev: _Evaluator, start: Candidate, goal: PoseGoal, norm: dict, lo, hi, fixed, step_deg: float = 4.0,
            min_step_deg: float = 0.25, budget: int = 160) -> Candidate:
    best = start
    step = math.radians(step_deg)
    free = np.nonzero(~fixed & (hi > lo))[0]
    evals = 0
    while step >= math.radians(min_step_deg) and evals < budget:
        improved = None
        for i in free:
            for sign in (+1.0, -1.0):
                q = best.q.copy()
                q[i] = np.clip(q[i] + sign * step, lo[i], hi[i])
                if q[i] == best.q[i]:
                    continue
                c = ev(q, source="refined")
                evals += 1
                if c.passed:
                    c.score = _score(c, goal, norm)
                    if c.score > (improved.score if improved is not None else best.score) + 1e-9:
                        improved = c
        if improved is None:
            step *= 0.5
        else:
            best = improved
    return best


# =========================================================================== the record
def record(study: RestStudy, sol: Solution, group: str = "arm") -> dict:
    """NamedPose-ready: joint names and positions, a generated one-line description (the use case and
    its headline numbers) and a why (the goal's settings and the result)."""
    goal, c, arm = sol.goal, sol.pick, study.arm
    if c is None:
        return {"name": goal.name, "group": group, "joint_names": list(arm.joint_names), "positions": None,
                "description": f"{goal.description or goal.name}: no pose passes the goal's gates",
                "why": f"{sol.sampled} poses sampled, none passed; refused most by: {sol.gate_failures}"}
    rep = c.report
    plane = study.scene.label("plane", "plane")
    clear = rep.clearance.plane if rep and rep.clearance else None
    parts = [f"{'nothing over the ' + plane if clear is None else f'{1e3 * clear:.0f} mm over the {plane}'}"]
    if goal.torque == "held":
        parts.append(f"{c.objectives.get('energy', math.nan):.2f} W to hold")
    else:
        how = c.settle.how if c.settle else "a passive rest"
        parts.append(f"torque off, 0 W ({how}; settles {c.settle.moved_deg:.1f} deg)" if c.settle else "torque off, 0 W")
    if rep and rep.footprint is not None:
        line = study.scene.label("line", "line").removeprefix("the ")
        parts.append(f"hand {1e3 * rep.footprint.reach:.0f} mm from the {line}")
    if c.readiness_mean is not None and np.isfinite(c.readiness_mean):
        parts.append(f"reaches the {goal.workspace.name} in {c.readiness_mean:.2f} s mean / {c.readiness_worst:.2f} s worst")
    if c.objectives.get("occlusion") is not None:
        parts.append(f"hides {100 * c.objectives['occlusion']:.1f} % of it from the {goal.camera.name}")
    description = f"{goal.description or goal.name}: " + ", ".join(parts) + "."
    s = goal.settings()
    runner = [o for o in sol.pareto if o is not c][:3]
    why = (f"Solved by robot_reach.goals for the goal '{goal.name}'. Settings: clearances {s['clearances_mm']} mm (0 = touching "
           f"allowed, None = ignored), torque {goal.torque}" + (f" (settles <= {goal.settle_deg:g} deg)" if goal.torque == "limp" else "")
           + f", workspace {s['workspace']}, weights {s['weights']}, stop margin {goal.stop_margin_deg:g} deg, search {s['search']}, "
           f"gates {s['gates'] or 'none extra'}. Result: {sol.sampled} poses evaluated, {sol.passed} passed every gate "
           f"(refused most by {dict(list(sol.gate_failures.items())[:4])}); the pick scores {c.score:.2f} over objectives "
           + ", ".join(f"{k} {_fmt(k, v)}" for k, v in c.objectives.items() if v is not None)
           + ("; Pareto runners-up: " + "; ".join(", ".join(f"{k} {_fmt(k, v)}" for k, v in o.objectives.items() if v is not None)
                                                 for o in runner) if runner else "")
           + (f". Torque off with no friction at all it would move {c.frictionless.moved_deg:.0f} deg ({c.frictionless.how})"
              if c.frictionless is not None else "")
           + ". Assumptions: " + "; ".join(sol.assumptions) + ".")
    return {"name": goal.name, "group": group, "joint_names": list(arm.joint_names), "positions": [float(v) for v in c.q],
            "degrees": {n: float(np.degrees(v)) for n, v in zip(arm.joint_names, c.q)}, "description": description, "why": why}


def _fmt(k: str, v) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    if k == "energy":
        return f"{v:.2f} W"
    if k == "readiness":
        return f"{v:.2f} s"
    if k == "compactness":
        return f"{1e3 * v:.0f} mm"
    if k == "occlusion":
        return f"{100 * v:.1f} %"
    return f"{v:.3g}"


def to_named_pose(rec: dict, joints: Sequence[str] | None = None):
    """The record as a robot_msgs NamedPose (`joints`: the subset the pose names, default all)."""
    from robot_msgs.manipulation import NamedPose

    names = list(joints or rec["joint_names"])
    pos = [rec["positions"][rec["joint_names"].index(n)] for n in names]
    kw = {"description": rec["description"]} if "description" in NamedPose.__dataclass_fields__ else {}
    return NamedPose(name=rec["name"], group=rec["group"], joint_names=tuple(names), positions=pos, why=rec["why"], **kw)


# =========================================================================== saved solutions (the explorer's cache)
SOLUTIONS_SCHEMA = "robot_reach/rest-solutions/1"


def goal_digest(goal: PoseGoal) -> str:
    """What a saved solution was solved for: the goal's settings (a changed setting means re-solve)."""
    import hashlib
    import json

    return hashlib.sha256(json.dumps(goal.settings(), sort_keys=True, default=str).encode()).hexdigest()[:16]


def save_solutions(study: RestStudy, solutions: Mapping[str, Solution], path, workspaces: Mapping[str, tuple] | None = None,
                   merge: bool = True) -> None:
    """Every solution as JSON (no pickle): settings digest, pick, Pareto set, record, assumptions; and
    each goal's workspace configurations, so a viewer computes readiness without the IK. With `merge`,
    goals already in the file and not solved now are kept (re-solving one goal keeps the others)."""
    import json
    from pathlib import Path

    doc = {"schema": SOLUTIONS_SCHEMA, "arm": study.arm.name, "joint_names": list(study.arm.joint_names), "goals": {}}
    if merge and Path(path).exists():
        old = json.loads(Path(path).read_text())
        if old.get("schema") == SOLUTIONS_SCHEMA and old.get("joint_names") == doc["joint_names"]:
            doc["goals"].update(old.get("goals", {}))
    for name, sol in solutions.items():
        entry = sol.to_dict(study.arm)
        entry["digest"] = goal_digest(sol.goal)
        if workspaces and name in workspaces:
            q, valid = workspaces[name]
            entry["workspace_q"] = np.asarray(q, float).tolist()
            entry["workspace_valid"] = np.asarray(valid, bool).tolist()
        doc["goals"][name] = entry
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=1, default=float))


def load_solutions(study: RestStudy, path) -> dict:
    """{goal name: saved entry} for the study's goals whose settings still match (stale ones dropped)."""
    import json
    from pathlib import Path

    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != SOLUTIONS_SCHEMA or doc.get("joint_names") != list(study.arm.joint_names):
        return {}
    out = {}
    for goal in study.goals:
        entry = doc["goals"].get(goal.name)
        if entry is not None and entry.get("digest") == goal_digest(goal):
            out[goal.name] = entry
    return out


def solve_all(study: RestStudy, *, path=None, seeds: Sequence[np.ndarray] = (), samples: Mapping[str, int] | None = None,
              progress: Callable[[str], None] | None = None, **kw) -> dict:
    """Solve every goal of the study (saving them to `path` when given): {name: Solution}."""
    sols, workspaces = {}, {}
    for goal in study.goals:
        n = (samples or {}).get(goal.name, 1500 if goal.torque == "held" else 200)
        sols[goal.name] = sol = solve(study, goal, samples=n, seeds=seeds, progress=progress, **kw)
        if sol.workspace_q is not None:
            workspaces[goal.name] = (sol.workspace_q, sol.workspace_valid)
    if path is not None:
        save_solutions(study, sols, path, workspaces)
    return sols
