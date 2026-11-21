"""Rest poses: where should an arm wait? Footprint, clearance and holding power of one pose over a
scene, and a gate-then-score ranking of candidate poses.

Zach, 2026-11-21: *"make a little widget to explore basically like these ready/resting positioning.
It should show you an image of the robot arm. and then also next to it kinda something that says
about its like foot print, clearance height above conveyor belt, and the energy to hold it. This is
a general kinda analysis and template I can do whenever I want to find like a good ready/resting
position for a robot arm."*

    from robot_reach.rest import Motor, RestStudy, Scene, rank

    study = RestStudy(arm, Scene(origin=(0.35, 0.0, 0.0), plane_point=(0, 0, 0), region_across=(-0.2, 0.2)),
                      motors=(Motor(kt=0.1, resistance=0.5, ratio=10),) * arm.dof)
    report = study.measure(q)          # RestReport: footprint, clearance, holding
    report.headline()                   # the three numbers, as text

The scene (`Scene`) lives in the arm's base frame (z up, gravity along -z, as `Arm.gravity_torque`):

- a **reference line** through `origin` along `along` (the belt's centre line under the shoulder);
  `across` is the other horizontal axis and heights run along ``across x along`` (up);
- the **support plane** (the belt's top) through `plane_point` with normal `plane_normal` (may tilt);
- the **region** on it (the belt between its edges): intervals of the across / along coordinates;
- **keep-outs** (rails): an across / along footprint and a top height each;
- the arm's **lane** on the axes where neighbouring arms sit (`neighbours`): fixed intervals, or the
  extent of the study's `shoulder` parts at the same pose.

Per pose (`RestStudy.measure` -> `RestReport`):

- **footprint**: the hand's and the whole arm's extent across / along from the reference line, the
  hull area of the arm projected onto the plane, and how far any non-shoulder point is past the lane
  (toward a neighbouring arm);
- **clearance**: the lowest arm point above the plane inside the region (distance along the plane's
  normal), and above each keep-out's top;
- **holding power**: the static gravity torque per joint — `Arm.gravity_torque` (virtual work,
  tau = dU/dq through the arm's own FK, so a closed chain such as a five-bar is handled by its FK) or
  the study's `torque(q)` (a measured model) — into copper watts per joint through a `Motor`, or the
  study's `power(q, tau)`.

Units SI (m, N·m, W, rad); the viewer and the cards show mm, degrees and W.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .arm import Arm
from .codes import CODE_LABELS, Code

Interval = tuple[float, float]
Vec3 = tuple[float, float, float]


# =========================================================================== the scene
@dataclass(frozen=True)
class KeepOut:
    """Something the arm must stay above (a rail, a camera bracket): an across / along footprint
    (None = unbounded) and the height of its top, all in the scene's coordinates."""

    name: str
    top: float
    across: Interval | None = None
    along: Interval | None = None


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError(f"zero-length axis {v}")
    return v / n


def _inside(x: np.ndarray, interval: Interval | None) -> np.ndarray:
    if interval is None:
        return np.ones(x.shape, bool)
    lo, hi = interval
    return (x >= lo) & (x <= hi)


@dataclass(frozen=True)
class Scene:
    """Where the arm rests: a support plane with a region on it, a reference line, keep-outs and the
    arm's lane. Everything in the arm's base frame (m); coordinates are measured from `origin`:
    ``a`` across, ``b`` along the reference line, ``h`` up (``across x along``)."""

    origin: Vec3
    across: Vec3 = (1.0, 0.0, 0.0)
    along: Vec3 = (0.0, 1.0, 0.0)
    plane_point: Vec3 | None = None  # default: through `origin`
    plane_normal: Vec3 | None = None  # default: up
    region_across: Interval | None = None  # the belt between its edges (None = unbounded)
    region_along: Interval | None = None
    keepouts: tuple[KeepOut, ...] = ()
    neighbours: tuple[str, ...] = ("along",)  # the axes on which a neighbouring arm sits
    lane_across: Interval | None = None  # the arm's own space (fixed); see RestStudy.shoulder
    lane_along: Interval | None = None
    min_clearance: float = 0.0  # m: the clearance gate (the viewer turns red below it)
    name: str = ""
    labels: Mapping[str, str] = field(default_factory=dict)  # "plane", "line": what to call them

    def __post_init__(self):
        u, v = _unit(self.across), _unit(self.along)
        if abs(float(u @ v)) > 1e-6:
            raise ValueError("Scene.across and Scene.along must be perpendicular")
        for name in self.neighbours:
            if name not in ("across", "along"):
                raise ValueError(f"Scene.neighbours: {name!r} is not 'across' or 'along'")

    # ------------------------------------------------------------------ frames
    @property
    def u(self) -> np.ndarray:
        return _unit(self.across)

    @property
    def v(self) -> np.ndarray:
        return _unit(self.along)

    @property
    def up(self) -> np.ndarray:
        return _unit(np.cross(self.u, self.v))

    @property
    def point(self) -> np.ndarray:
        return np.asarray(self.origin if self.plane_point is None else self.plane_point, float)

    @property
    def normal(self) -> np.ndarray:
        n = self.up if self.plane_normal is None else _unit(self.plane_normal)
        return n if n @ self.up >= 0 else -n

    def label(self, key: str, default: str) -> str:
        return str(self.labels.get(key, default))

    def coords(self, points) -> np.ndarray:
        """(..., 3) arm-frame points -> (..., 3) scene coordinates (a across, b along, h up)."""
        d = np.asarray(points, float) - np.asarray(self.origin, float)
        return np.stack([d @ self.u, d @ self.v, d @ self.up], axis=-1)

    def to_world(self, abh) -> np.ndarray:
        abh = np.asarray(abh, float)
        return np.asarray(self.origin, float) + abh[..., :1] * self.u + abh[..., 1:2] * self.v + abh[..., 2:3] * self.up

    def above_plane(self, points) -> np.ndarray:
        """Signed distance along the plane's normal (m), positive above."""
        return (np.asarray(points, float) - self.point) @ self.normal

    def plane_height(self, a, b) -> np.ndarray:
        """The plane's height h at scene coordinates (a, b)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        p0 = self.coords(self.point)
        n = np.array([self.normal @ self.u, self.normal @ self.v, self.normal @ self.up])
        return p0[2] - (n[0] * (a - p0[0]) + n[1] * (b - p0[1])) / n[2]

    def in_region(self, abh: np.ndarray) -> np.ndarray:
        return _inside(abh[..., 0], self.region_across) & _inside(abh[..., 1], self.region_along)

    def plane_basis(self) -> tuple[np.ndarray, np.ndarray]:
        """Two in-plane unit axes: `across` and `along` projected onto the support plane."""
        n = self.normal
        e1 = _unit(self.u - (self.u @ n) * n)
        return e1, np.cross(n, e1)


# =========================================================================== motors and the study
@dataclass(frozen=True)
class Motor:
    """A joint's drive for copper loss: P = 3/2 · R · i², i = |tau| / (kt · ratio).

    - `kt`: N·m per A of q-axis current (amplitude-invariant dq) at the motor shaft;
    - `resistance`: Ω, phase to neutral;
    - `ratio`: the gear reduction (the output turns `ratio` times slower).

    WHY copper only: a held pose does no mechanical work, so the heat in the windings is the energy
    the hold costs (driver and idle losses are the same in every pose and do not rank them)."""

    kt: float
    resistance: float
    ratio: float = 1.0
    name: str = ""

    def current(self, tau: float) -> float:
        return abs(float(tau)) / (self.kt * self.ratio)

    def copper_w(self, tau: float) -> float:
        return 1.5 * self.resistance * self.current(tau) ** 2

    @classmethod
    def from_kv(cls, kv_rpm_per_v: float, resistance: float, ratio: float = 1.0, *, factor: float = 1.5 / math.sqrt(3.0), name: str = "") -> Motor:
        """kt from a Kv (rpm/V): kt = factor · 60 / (2π · Kv). The default `factor` is moteus'
        kTorqueFactor (3/2 / √3), which matches its q-axis current."""
        return cls(kt=factor * 60.0 / (2.0 * math.pi) / float(kv_rpm_per_v), resistance=resistance, ratio=ratio, name=name)


@dataclass(frozen=True)
class PoseChoice:
    """One pose to look at: a named pose (a robot_msgs NamedPose) or a candidate (any JSON row)."""

    name: str
    q: np.ndarray
    description: str = ""
    digest: str = ""
    kind: str = "named"  # "named" | "candidate"
    source: str = ""

    @property
    def label(self) -> str:
        return self.name if self.kind == "named" else f"candidate · {self.name}"


@dataclass(frozen=True, eq=False)
class RestStudy:
    """An arm over a scene, with how to price a hold: everything `measure`, the viewer and the cards
    need. `hand` / `shoulder` name `ArmMesh.part`s (empty: the whole arm / the scene's fixed lane)."""

    arm: Arm
    scene: Scene
    name: str = ""
    motors: tuple[Motor | None, ...] | None = None
    # (q (dof,), tau (dof,)) -> W per joint (dof,) or a total: a measured power model; wins over motors
    power: Callable[[np.ndarray, np.ndarray], object] | None = None
    # q (dof,) -> tau (dof,) N·m: a measured gravity model; default Arm.gravity_torque (virtual work)
    torque: Callable[[np.ndarray], np.ndarray] | None = None
    torque_note: str = ""  # where the torque comes from, shown beside it
    hand: tuple[str, ...] = ()
    shoulder: tuple[str, ...] = ()
    max_per_mesh: int | None = None  # vertices per mesh that count (Arm.surface)
    poses: tuple[PoseChoice, ...] = ()
    gates: tuple[Gate, ...] = ()
    criteria: tuple[Criterion, ...] = ()
    aliases: Mapping[str, str] = field(default_factory=dict)  # other joint names (J1) -> the arm's, for pose files

    def __post_init__(self):
        if self.motors is not None and len(self.motors) != self.arm.dof:
            raise ValueError(f"RestStudy.motors: {len(self.motors)} given, the arm has {self.arm.dof} joints")

    @property
    def home(self) -> np.ndarray:
        return default_q(self.arm)

    def measure(self, q) -> RestReport:
        return measure(self, q)

    def with_poses(self, *groups: Sequence[PoseChoice]) -> RestStudy:
        import dataclasses

        poses = list(self.poses)
        for g in groups:
            poses += list(g)
        return dataclasses.replace(self, poses=tuple(poses))


def with_link_boxes(arm: Arm, width: float = 0.05, names: Sequence[str] | None = None) -> Arm:
    """`arm` with a square box of side `width` along every skeleton segment (`Arm.chain`), as
    `ArmMesh` parts ``link0``, ``link1``... (or `names`): thickness for an arm that has only a
    skeleton, so clearance and footprint mean something. Segment lengths come from the arm's home."""
    import dataclasses

    from .arm import ArmMesh

    chain = arm.chain(default_q(arm)[None])[0]
    lengths = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    names = list(names) if names is not None else [f"link{k}" for k in range(len(lengths))]
    hw = width / 2
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    meshes = []
    for name, length in zip(names, lengths):
        v = np.array([[x, y, z] for x in (0.0, length) for y in (-hw, hw) for z in (-hw, hw)], float)
        meshes.append(ArmMesh(name, v, faces, color=(150, 156, 168)))

    def mesh_poses(Q):
        c = arm.chain(np.atleast_2d(Q))
        d = np.diff(c, axis=1)  # (N, K, 3): each segment, base -> tip
        x = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
        ref = np.where(np.abs(x[..., 2:3]) > 0.9, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        y = np.cross(ref, x)
        y /= np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-12)
        T = np.zeros(d.shape[:2] + (4, 4))
        T[..., :3, 0], T[..., :3, 1], T[..., :3, 2] = x, y, np.cross(x, y)
        T[..., :3, 3], T[..., 3, 3] = c[:, :-1], 1.0
        return T

    return dataclasses.replace(arm, meshes=tuple(meshes), mesh_poses=mesh_poses)


# =========================================================================== the report
@dataclass(frozen=True)
class Footprint:
    across: Interval  # the whole arm, from the reference line (m)
    along: Interval
    hand_across: Interval  # the hand (RestStudy.hand; the whole arm without one)
    hand_along: Interval
    reach: float  # the hand's farthest point from the line, across or along: max |a|, |b|
    hull_area: float  # m², the whole arm projected onto the support plane
    hull: np.ndarray  # (K, 2) that hull, in scene coordinates (a, b)
    lane: dict  # axis -> (lo, hi): the arm's own space on each neighbour axis
    past_shoulder: float | None  # m past the lane toward a neighbour (<= 0: inside); None: no lane
    past_axis: str = ""


@dataclass(frozen=True)
class Clearance:
    plane: float | None  # m: the lowest point over the region, above the plane (None: nothing over it)
    lowest: np.ndarray | None  # (3,) that point, arm frame
    keepouts: dict  # name -> m above its top (None: nothing over it)
    keepout_points: dict  # name -> (3,) the lowest point over it
    threshold: float

    @property
    def ok(self) -> bool:
        return (self.plane is None or self.plane >= self.threshold) and all(v is None or v >= 0.0 for v in self.keepouts.values())


@dataclass(frozen=True)
class Holding:
    tau: np.ndarray | None  # (dof,) N·m
    watts: np.ndarray | None  # (dof,) W (None when there is no motor model)
    total: float | None  # W
    current: np.ndarray | None  # (dof,) A (motors only)
    limit_frac: np.ndarray | None  # (dof,) |tau| / Arm.torque_limits
    source: str


@dataclass(frozen=True, eq=False)
class RestReport:
    q: np.ndarray
    ok: bool  # FK closed and the geometry is finite
    code: int  # Arm.check: limits, FK, collision, torque
    footprint: Footprint | None
    clearance: Clearance | None
    holding: Holding
    seconds: float = 0.0

    @property
    def code_label(self) -> str:
        return CODE_LABELS[Code(self.code)]

    def metrics(self) -> dict:
        """Flat headline numbers (SI) for gates, criteria and JSON: see the keys below."""
        f, c, h = self.footprint, self.clearance, self.holding
        out = {"ok": bool(self.ok), "code": int(self.code), "code_ok": self.code == int(Code.OK)}
        if f is not None:
            out.update(
                reach_m=f.reach,
                hand_across_min_m=f.hand_across[0], hand_across_max_m=f.hand_across[1],
                hand_along_min_m=f.hand_along[0], hand_along_max_m=f.hand_along[1],
                across_min_m=f.across[0], across_max_m=f.across[1],
                along_min_m=f.along[0], along_max_m=f.along[1],
                hull_area_m2=f.hull_area, past_shoulder_m=f.past_shoulder,
            )
        if c is not None:
            # WHY inf: nothing over the region means nothing can meet an item there (compute.py's rule too)
            out["clearance_m"] = math.inf if c.plane is None else c.plane
            for name, value in c.keepouts.items():
                out[f"keepout_{name}_m"] = value
            out["clearance_ok"] = c.ok
        out["power_w"] = h.total
        out["max_limit_frac"] = None if h.limit_frac is None else float(np.max(h.limit_frac))
        return out

    def headline(self) -> dict:
        """The three numbers in words: {"footprint", "clearance", "holding"} -> str."""
        f, c, h = self.footprint, self.clearance, self.holding
        fp = "—" if f is None else f"{1e3 * f.reach:.0f} mm from the line · hull {f.hull_area:.3f} m²"
        cl = "—" if c is None or c.plane is None else f"{1e3 * c.plane:.0f} mm"
        hp = "—" if h.total is None else f"{h.total:.2f} W"
        return {"footprint": fp, "clearance": cl, "holding": hp}

    def to_dict(self, arm: Arm | None = None) -> dict:
        names = list(arm.joint_names) if arm is not None else [f"q{i}" for i in range(len(self.q))]
        f, c, h = self.footprint, self.clearance, self.holding

        def num(x):
            return None if x is None or not np.isfinite(float(x)) else float(x)

        out = {
            "q_deg": {n: float(np.degrees(v)) for n, v in zip(names, self.q)},
            "q_rad": {n: float(v) for n, v in zip(names, self.q)},
            "ok": bool(self.ok),
            "code": self.code_label,
            "metrics": {k: (num(v) if not isinstance(v, bool) else v) for k, v in self.metrics().items()},
            "holding": {
                "source": h.source,
                "tau_nm": None if h.tau is None else dict(zip(names, map(float, h.tau))),
                "watts": None if h.watts is None else dict(zip(names, map(float, h.watts))),
                "total_w": num(h.total),
            },
        }
        if f is not None:
            out["footprint"] = {"lane_m": {k: list(map(float, v)) for k, v in f.lane.items()}, "past_axis": f.past_axis,
                                "hull_ab_m": np.round(f.hull, 5).tolist()}
        if c is not None:
            out["clearance"] = {"lowest_point_m": None if c.lowest is None else c.lowest.tolist(), "threshold_m": c.threshold,
                                "keepouts_m": {k: num(v) for k, v in c.keepouts.items()}}
        return out


# =========================================================================== measuring
def default_q(arm: Arm) -> np.ndarray:
    """The arm's `meta["home"]` clipped into its limits (zeros without one)."""
    home = np.asarray(arm.meta.get("home", np.zeros(arm.dof)), float).reshape(-1)
    if home.shape != (arm.dof,):
        home = np.zeros(arm.dof)
    return np.clip(home, arm.lower, arm.upper)


def _surface_labels(arm: Arm, max_per_mesh: int | None) -> np.ndarray:
    """The part name of every point `arm.surface(q, max_per_mesh=...)` returns, in its order."""
    if not (arm.meshes and arm.mesh_poses is not None):
        return np.zeros(0, dtype=object)
    labels = []
    for i in arm.surface_parts():
        n = len(arm.meshes[i].vertices)
        n = min(n, max_per_mesh) if max_per_mesh else n  # Arm.surface's rule
        labels += [arm.meshes[i].part] * n
    return np.asarray(labels, dtype=object)


def convex_hull(points: np.ndarray) -> np.ndarray:
    """2-D convex hull (Andrew's monotone chain), counter-clockwise, (K, 2)."""
    pts = np.asarray(points, float).reshape(-1, 2)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(pts) > 1:
        pts = pts[np.r_[True, np.any(np.diff(pts, axis=0) != 0, axis=1)]]
    if len(pts) < 3:
        return pts
    seq = pts.tolist()  # WHY plain floats: the chain is a Python loop; numpy scalars are ~10x slower

    def half(points):
        out: list = []
        for px, py in points:
            while len(out) >= 2 and (out[-1][0] - out[-2][0]) * (py - out[-2][1]) - (out[-1][1] - out[-2][1]) * (px - out[-2][0]) <= 0:
                out.pop()
            out.append((px, py))
        return out

    lower, upper = half(seq), half(seq[::-1])
    return np.array(lower[:-1] + upper[:-1])


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _outside_polygon_of_extremes(ab: np.ndarray, directions: int) -> np.ndarray:
    """Mask of the points NOT strictly inside the convex polygon of the extremes in `directions`
    directions: only those can be hull vertices."""
    angles = np.linspace(0, 2 * np.pi, directions, endpoint=False)
    poly = convex_hull(ab[np.unique(np.argmax(ab @ np.stack([np.cos(angles), np.sin(angles)], 0), axis=0))])
    if len(poly) < 3:
        return np.ones(len(ab), bool)
    keep = np.zeros(len(ab), bool)
    for p0, p1 in zip(poly, np.roll(poly, -1, axis=0)):  # counter-clockwise: inside is left of every edge
        keep |= (p1[0] - p0[0]) * (ab[:, 1] - p0[1]) - (p1[1] - p0[1]) * (ab[:, 0] - p0[0]) <= 1e-12
    return keep


def _hull_points(ab: np.ndarray, cap: int = 2000) -> np.ndarray:
    """The points a convex hull can come from, exactly: two passes that drop points strictly inside an
    inner polygon (an octagon of extremes, then a 64-gon on what is left). WHY: the monotone chain walks
    points in Python and a CAD arm has ~10^5 mesh vertices; the inner polygons hold all but a thin rim."""
    for directions in (8, 64):
        if len(ab) <= cap:
            break
        ab = ab[_outside_polygon_of_extremes(ab, directions)]
    return ab


def _interval(x: np.ndarray) -> Interval:
    return (float(np.min(x)), float(np.max(x))) if len(x) else (math.nan, math.nan)


def holding(study: RestStudy, q: np.ndarray) -> Holding:
    """Static holding torque and copper watts per joint (see the module docstring)."""
    arm = study.arm
    if study.torque is not None:
        tau = np.asarray(study.torque(q), float).reshape(arm.dof)
        source = study.torque_note or "the study's torque model"
    else:
        t = arm.gravity_torque(q[None])
        tau = None if t is None else np.asarray(t[0], float)
        source = study.torque_note or ("virtual work on the arm's point masses (dU/dq through its FK)" if tau is not None else "no masses")
    limit = None
    if tau is not None and arm.torque_limits is not None:
        limit = np.abs(tau) / np.asarray(arm.torque_limits, float)
    watts = current = total = None
    if tau is not None and study.power is not None:
        w = np.asarray(study.power(q, tau), float)
        if w.ndim == 0:
            total = float(w)
        else:
            watts = w.reshape(arm.dof)
            total = float(np.sum(watts))
        source += " · power: the study's power model"
    elif tau is not None and study.motors is not None:
        watts = np.array([0.0 if m is None else m.copper_w(t) for m, t in zip(study.motors, tau)])
        current = np.array([0.0 if m is None else m.current(t) for m, t in zip(study.motors, tau)])
        total = float(np.sum(watts))
        source += " · power: copper loss 3/2·R·i² per motor"
    return Holding(tau=tau, watts=watts, total=total, current=current, limit_frac=limit, source=source)


def measure(study: RestStudy, q) -> RestReport:
    """Footprint, clearance and holding power of one pose (module docstring)."""
    import time

    t0 = time.perf_counter()
    arm, scene = study.arm, study.scene
    q = np.asarray(q, float).reshape(arm.dof)
    code = int(arm.check(q[None])[0])
    _, fk_ok = arm.forward(q[None])
    pts = arm.surface(q[None], max_per_mesh=study.max_per_mesh)[0]
    labels = _surface_labels(arm, study.max_per_mesh)
    if len(labels) != len(pts):
        labels = np.full(len(pts), "", dtype=object)
    finite = np.all(np.isfinite(pts), axis=1)
    ok = bool(fk_ok[0]) and bool(finite.any())
    hold = holding(study, q) if ok else Holding(None, None, None, None, None, "the pose does not close")
    if not ok:
        return RestReport(q=q, ok=False, code=code, footprint=None, clearance=None, holding=hold, seconds=time.perf_counter() - t0)
    pts, labels = pts[finite], labels[finite]
    abh = scene.coords(pts)

    # ---- footprint
    hand = np.isin(labels, list(study.hand)) if study.hand else np.ones(len(pts), bool)
    if not hand.any():
        hand = np.ones(len(pts), bool)
    ha, hb = abh[hand, 0], abh[hand, 1]
    e1, e2 = scene.plane_basis()
    rel = pts - scene.point
    in_plane = np.stack([rel @ e1, rel @ e2], 1)
    hull_src = _hull_points(in_plane)
    hull = convex_hull(hull_src)
    # the hull drawn in scene (a, b): the same points, back on the plane
    hull_ab = scene.coords(scene.point + hull[:, :1] * e1 + hull[:, 1:2] * e2)[:, :2] if len(hull) else np.zeros((0, 2))
    shoulder = np.isin(labels, list(study.shoulder)) if study.shoulder else np.zeros(len(pts), bool)
    lane: dict = {}
    past, past_axis = None, ""
    for axis in scene.neighbours:
        k = 0 if axis == "across" else 1
        fixed = scene.lane_across if axis == "across" else scene.lane_along
        if fixed is not None:
            lane[axis] = tuple(map(float, fixed))
        elif shoulder.any():
            lane[axis] = _interval(abh[shoulder, k])
        else:
            continue
        others = abh[~shoulder, k] if shoulder.any() else abh[:, k]
        if len(others):
            lo, hi = lane[axis]
            over = float(max(np.max(others) - hi, lo - np.min(others)))
            if past is None or over > past:
                past, past_axis = over, axis
    foot = Footprint(
        across=_interval(abh[:, 0]), along=_interval(abh[:, 1]),
        hand_across=_interval(ha), hand_along=_interval(hb),
        reach=float(max(np.max(np.abs(ha)), np.max(np.abs(hb)))),
        hull_area=polygon_area(hull), hull=hull_ab, lane=lane, past_shoulder=past, past_axis=past_axis,
    )

    # ---- clearance
    over = scene.in_region(abh)
    plane = lowest = None
    if over.any():
        d = scene.above_plane(pts[over])
        i = int(np.argmin(d))
        plane, lowest = float(d[i]), pts[over][i]
    kos, kopts = {}, {}
    for ko in scene.keepouts:
        sel = _inside(abh[:, 0], ko.across) & _inside(abh[:, 1], ko.along)
        if sel.any():
            dh = abh[sel, 2] - ko.top
            i = int(np.argmin(dh))
            kos[ko.name], kopts[ko.name] = float(dh[i]), pts[sel][i]
        else:
            kos[ko.name], kopts[ko.name] = None, None
    clear = Clearance(plane=plane, lowest=lowest, keepouts=kos, keepout_points=kopts, threshold=float(scene.min_clearance))
    return RestReport(q=q, ok=True, code=code, footprint=foot, clearance=clear, holding=hold, seconds=time.perf_counter() - t0)


# =========================================================================== ranking
Metric = str | Callable[["RestReport"], object]


def _value(report: RestReport, metric: Metric):
    return metric(report) if callable(metric) else report.metrics().get(metric)


@dataclass(frozen=True)
class Gate:
    """Pass/fail: `metric` (a `RestReport.metrics()` key or a callable) within [min, max]; a bare
    boolean metric passes when true. A missing value (None) fails."""

    name: str
    metric: Metric
    min: float | None = None
    max: float | None = None

    def passes(self, report: RestReport) -> bool:
        v = _value(report, self.metric)
        if v is None:
            return False
        if isinstance(v, (bool, np.bool_)) and self.min is None and self.max is None:
            return bool(v)
        v = float(v)
        return (self.min is None or v >= self.min) and (self.max is None or v <= self.max)


@dataclass(frozen=True)
class Criterion:
    """A weighted score in [0, 1]: 1 at `best`, 0 at `worst`, linear between and clipped — anchors
    you fix before you look (criteria before options). Without anchors, the best and worst of the
    poses ranked together (`higher_is_better` says which way)."""

    name: str
    metric: Metric
    weight: float
    best: float | None = None
    worst: float | None = None
    higher_is_better: bool = False


@dataclass(frozen=True, eq=False)
class Ranked:
    name: str
    passed: bool
    failed: tuple[str, ...]
    score: float  # weighted mean of the criteria scores, 0..1
    scores: dict  # criterion -> 0..1 (None: no value)
    values: dict  # criterion -> the metric's value
    report: RestReport


def rank(reports: Mapping[str, RestReport], gates: Sequence[Gate] = (), criteria: Sequence[Criterion] = ()) -> list[Ranked]:
    """Gates first (every gate must pass), then the weighted score; passing poses rank above failing
    ones, each group by score (highest first)."""
    values = {c.name: {k: _value(r, c.metric) for k, r in reports.items()} for c in criteria}
    out = []
    for key, rep in reports.items():
        failed = tuple(g.name for g in gates if not g.passes(rep))
        scores, total, weight = {}, 0.0, 0.0
        for c in criteria:
            v = values[c.name][key]
            if v is None or np.isnan(float(v)):
                scores[c.name] = None
                weight += c.weight
                continue
            v = float(v)
            if c.best is not None and c.worst is not None:
                best, worst = c.best, c.worst
            else:
                seen = [float(x) for x in values[c.name].values() if x is not None and np.isfinite(float(x))] or [v]
                lo, hi = min(seen), max(seen)
                best, worst = (hi, lo) if c.higher_is_better else (lo, hi)
            if best == worst:
                s = 1.0
            elif np.isinf(v):
                s = 1.0 if (v > 0) == (best > worst) else 0.0
            else:
                s = float(np.clip((v - worst) / (best - worst), 0.0, 1.0))
            scores[c.name] = s
            total += c.weight * s
            weight += c.weight
        score = total / weight if weight else 0.0
        out.append(Ranked(key, not failed, failed, score, scores, {c.name: values[c.name][key] for c in criteria}, rep))
    out.sort(key=lambda r: (not r.passed, -r.score))
    return out


# =========================================================================== reading poses
def _revolute(arm: Arm) -> np.ndarray:
    rev = arm.meta.get("revolute")
    return np.ones(arm.dof, bool) if rev is None else np.asarray(rev, bool)


def wrap_into_limits(arm: Arm, q) -> np.ndarray:
    """A revolute joint outside its limits tries its ±2π twin (a stored angle is often wrapped to ±π)."""
    from .urdf import into_limits

    return into_limits(np.asarray(q, float)[None], np.asarray(arm.lower), np.asarray(arm.upper), _revolute(arm))[0]


def _assign(arm: Arm, base: np.ndarray, values: Mapping[str, float], aliases: Mapping[str, str] | None) -> tuple[np.ndarray, list[str]]:
    q = np.array(base, float)
    unknown = []
    index = {n: i for i, n in enumerate(arm.joint_names)}
    for name, value in values.items():
        name = (aliases or {}).get(name, name)
        if name in index:
            q[index[name]] = float(value)
        else:
            unknown.append(name)
    return wrap_into_limits(arm, q), unknown


def named_poses(source, arm: Arm, base_q=None, aliases: Mapping[str, str] | None = None) -> list[PoseChoice]:
    """Every joint-space pose of a robot_msgs named-poses JSON (``named_poses_to_json``) the arm can
    take: its joints set, the arm's other joints at `base_q` (default `default_q`). A Cartesian pose
    (it needs an IK) and a pose naming a joint the arm lacks are skipped. `description` is read with
    getattr, so a robot_msgs without the field still works."""
    from robot_msgs.manipulation import named_poses_from_json

    text = Path(source).read_text() if isinstance(source, (str, Path)) and Path(str(source)).exists() else str(source)
    base = default_q(arm) if base_q is None else np.asarray(base_q, float)
    out = []
    for pose in named_poses_from_json(text):
        if pose.kind != "joint":
            continue
        q, unknown = _assign(arm, base, pose.as_dict(), aliases)
        if unknown:
            continue
        group = f" ({pose.group})" if getattr(pose, "group", "") else ""
        out.append(PoseChoice(name=pose.name + ("" if pose.group in ("arm", "") else group), q=q,
                              description=str(getattr(pose, "description", "") or ""), digest=pose.digest(), kind="named",
                              source=str(source) if isinstance(source, (str, Path)) else ""))
    return out


_ROW_KEYS = ("candidates", "variants", "poses", "rows", "results")
_Q_KEYS_RAD = ("q_rad", "joint_rad", "joints_rad", "q")
_Q_KEYS_DEG = ("q_deg", "joint_deg", "joints_deg", "degrees")


def candidate_poses(source, arm: Arm, base_q=None, aliases: Mapping[str, str] | None = None) -> list[PoseChoice]:
    """Candidate poses from a plain JSON file: a list of rows, ``{"candidates" | "variants" | "poses"
    | "rows" | "results": [...]}``, or ``{name: row}``. A row names itself (``name`` / ``key``, else
    its dict key) and its joints as ``{joint: value}`` under ``q_rad`` / ``joint_rad`` / ``q`` (rad)
    or ``q_deg`` / ``joint_deg`` / ``degrees`` (deg); `aliases` maps other joint names (``J1``) to the
    arm's. ``description`` / ``why`` / ``title`` describe it. A robot_msgs named-poses file is read
    as named poses."""
    path = Path(source)
    doc = json.loads(path.read_text())
    if isinstance(doc, dict) and str(doc.get("schema", "")).startswith("robot_msgs/named-poses"):
        return [PoseChoice(p.name, p.q, p.description, p.digest, "candidate", str(path)) for p in named_poses(path, arm, base_q, aliases)]
    rows: list = []
    if isinstance(doc, list):
        rows = [(None, r) for r in doc]
    elif isinstance(doc, dict):
        for k in _ROW_KEYS:
            if isinstance(doc.get(k), list):
                rows = [(None, r) for r in doc[k]]
                break
            if isinstance(doc.get(k), dict):
                rows = list(doc[k].items())
                break
        else:
            rows = [(k, r) for k, r in doc.items() if isinstance(r, dict)]
    base = default_q(arm) if base_q is None else np.asarray(base_q, float)
    out = []
    for key, row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("key") or key or f"#{len(out) + 1}")
        values = None
        for k in _Q_KEYS_RAD:
            if isinstance(row.get(k), dict):
                values = {j: float(v) for j, v in row[k].items()}
                break
        if values is None:
            for k in _Q_KEYS_DEG:
                if isinstance(row.get(k), dict):
                    values = {j: math.radians(float(v)) for j, v in row[k].items()}
                    break
        if values is None:
            continue
        q, unknown = _assign(arm, base, values, aliases)
        if unknown:
            continue
        text = " — ".join(str(row[k]) for k in ("title", "description", "why") if row.get(k))
        out.append(PoseChoice(name=name, q=q, description=text, kind="candidate", source=str(path)))
    return out


def reports_for(study: RestStudy, poses: Sequence[PoseChoice] | None = None) -> dict[str, RestReport]:
    """`measure` every pose (default: the study's), keyed by `PoseChoice.label`."""
    return {p.label: study.measure(p.q) for p in (study.poses if poses is None else poses)}
