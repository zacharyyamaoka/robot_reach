"""Where to ask: FK clouds, boxes and grids.

Two ways to find a workspace, used together:

- **Forward** (`fk_cloud`): sample joint space, run FK. Always works, for any DOF, including a
  1-DOF pendulum whose workspace is a curve no Cartesian grid would ever hit exactly.
- **Inverse** (a grid + `reach_map`): ask IK at chosen targets. Exact per target, counts every IK
  branch, and is what a task ("can I pick here, straight down, at any yaw?") actually needs.

WHY the FK cloud bounds the default grid (kept from bam_reach v1, "FK sampling creates a
conservative centered grid"): no one has to guess the arm's extent, and the grid never clips it.

WHY the tightest of four shapes (Zach, 2026-11-23: "improve the initial voxel algorithm ... perhaps do
like a convex hull, or sphere, or cylinder, or rectangle? what ever fits best ... and then add just a
bit of padding"): `fit_grid` keeps the lattice of the v1 bounding box and drops the lattice points
outside the best-fitting shape, padded by one grid step, so every FK point is still inside and the
reach map spends no IK on corners the arm cannot reach.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arm import Arm


def sample_joints(arm: Arm, n: int, seed: int = 0) -> np.ndarray:
    """(n, dof) uniform inside the joint limits, seeded."""
    rng = np.random.default_rng(seed)
    lim = arm.limits
    return lim[:, 0] + rng.random((n, arm.dof)) * (lim[:, 1] - lim[:, 0])


def joint_grid(arm: Arm, per_joint: int) -> np.ndarray:
    """(per_joint**dof, dof) full grid inside the limits (small dof only)."""
    axes = [np.linspace(lo, hi, per_joint) for lo, hi in arm.limits]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, arm.dof)


@dataclass(frozen=True)
class FKCloud:
    q: np.ndarray  # (N, dof)
    T: np.ndarray  # (N, 4, 4)
    ok: np.ndarray  # (N,) FK succeeded

    @property
    def points(self) -> np.ndarray:
        return self.T[self.ok, :3, 3]


def fk_cloud(arm: Arm, n: int = 20_000, seed: int = 0, grid: bool | None = None) -> FKCloud:
    """FK of joint samples. 1-2 DOF arms use a dense grid (so curves and edges are clean)."""
    if grid is None:
        grid = arm.dof <= 2
    q = joint_grid(arm, max(2, int(round(n ** (1 / arm.dof))))) if grid else sample_joints(arm, n, seed)
    T, ok = arm.forward(q)
    return FKCloud(q=q, T=T, ok=ok)


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in the arm's base frame: centre and full side lengths (m)."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]

    def grid(self, step: float | tuple[float, float, float]) -> np.ndarray:
        """Centred grid. WHY expand, never shrink (kept from bam_reach v1, "Grid sampling expands a
        centered box to exact spacing"): each side grows to the next whole number of steps around a
        fixed centre, so an authored box is never clipped. A zero-size side is one layer."""
        step = np.broadcast_to(np.asarray(step, float), (3,))
        axes = []
        for c, s, h in zip(self.center, self.size, step):
            n = int(np.ceil(s / h - 1e-9)) if s > 0 else 0
            half = n * h / 2
            axes.append(c + np.linspace(-half, half, n + 1) if n else np.array([c]))
        return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)

    def contains(self, p: np.ndarray, pad: float = 1e-9) -> np.ndarray:
        lo = np.asarray(self.center) - np.asarray(self.size) / 2 - pad
        hi = np.asarray(self.center) + np.asarray(self.size) / 2 + pad
        return np.all((p >= lo) & (p <= hi), axis=-1)

    @classmethod
    def around(cls, points: np.ndarray, margin: float = 0.0, flat_tol: float = 1e-6) -> Box:
        """Bounding box of points. A flat axis (a planar arm) stays flat: one grid layer."""
        lo, hi = points.min(0), points.max(0)
        size = hi - lo
        size = np.where(size < flat_tol, 0.0, size + 2 * margin)
        return cls(tuple((lo + hi) / 2), tuple(size))


# --------------------------------------------------------------------------- grid shapes
HULL_DIRECTIONS = 98  # planes of the convex-hull fit (a k-DOP: an outer approximation of the hull)


def _sphere_directions(n: int) -> np.ndarray:
    """(n, 3) roughly even unit vectors (Fibonacci sphere), deterministic."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(np.maximum(0.0, 1 - z * z))
    phi = np.pi * (3 - np.sqrt(5)) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


@dataclass(frozen=True)
class GridShape:
    """The volume the reach grid is kept inside: ``box`` (v1), ``sphere``, ``cylinder`` (vertical axis)
    or ``hull`` (the FK cloud's convex hull, approximated from outside by `HULL_DIRECTIONS` planes).
    Every number is in the arm's base frame, m; ``pad`` is already included."""

    kind: str
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0  # sphere, cylinder
    z_range: tuple[float, float] = (0.0, 0.0)  # cylinder
    box: Box | None = None  # box: the padded bounding box
    normals: np.ndarray | None = None  # hull: (K, 3) unit normals
    offsets: np.ndarray | None = None  # hull: (K,) support distances (pad included)

    def contains(self, p: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        p = np.asarray(p, float).reshape(-1, 3)
        if self.kind == "box":
            return self.box.contains(p, pad=tol)
        if self.kind == "sphere":
            return np.linalg.norm(p - np.asarray(self.center), axis=1) <= self.radius + tol
        if self.kind == "cylinder":
            rho = np.linalg.norm(p[:, :2] - np.asarray(self.center[:2]), axis=1)
            return (rho <= self.radius + tol) & (p[:, 2] >= self.z_range[0] - tol) & (p[:, 2] <= self.z_range[1] + tol)
        if self.kind == "hull":
            return np.all(p @ self.normals.T <= self.offsets + tol, axis=1)
        raise ValueError(f"unknown grid shape {self.kind!r}")

    def describe(self) -> str:
        if self.kind == "sphere":
            return f"sphere r {self.radius:.3f} m"
        if self.kind == "cylinder":
            return f"cylinder r {self.radius:.3f} m, z {self.z_range[0]:+.3f}…{self.z_range[1]:+.3f} m"
        if self.kind == "hull":
            return f"convex hull ({len(self.normals)} planes)"
        return "box " + " × ".join(f"{v:.3f}" for v in self.box.size) + " m"

    def to_json(self) -> dict:
        out = {"kind": self.kind, "center": list(self.center), "radius": self.radius, "z_range": list(self.z_range)}
        if self.box is not None:
            out["box"] = {"center": list(self.box.center), "size": list(self.box.size)}
        if self.normals is not None:
            out["normals"], out["offsets"] = self.normals.tolist(), self.offsets.tolist()
        return out


def fit_shapes(points: np.ndarray, pad: float) -> dict[str, GridShape]:
    """Every candidate shape around `points`, each padded by `pad` (m)."""
    points = np.asarray(points, float).reshape(-1, 3)
    lo, hi = points.min(0), points.max(0)
    box = Box.around(points)
    padded = Box(box.center, tuple(s + 2 * pad if s > 0 else 0.0 for s in box.size))
    out = {"box": GridShape("box", tuple(box.center), box=padded)}
    # sphere: the smallest of the bbox centre, the centroid and a Badoiu-Clarkson minimum-ball estimate
    centers = [(lo + hi) / 2, points.mean(0)]
    c = centers[0].copy()
    for k in range(1, 120):  # step toward the farthest point by 1/(k+1): converges to the minimum ball
        far = points[int(np.argmax(np.linalg.norm(points - c, axis=1)))]
        c = c + (far - c) / (k + 1)
    centers.append(c)
    best = min(centers, key=lambda cc: float(np.max(np.linalg.norm(points - cc, axis=1))))
    out["sphere"] = GridShape("sphere", tuple(float(v) for v in best), radius=float(np.max(np.linalg.norm(points - best, axis=1))) + pad)
    cxy = (lo[:2] + hi[:2]) / 2
    rho = float(np.max(np.linalg.norm(points[:, :2] - cxy, axis=1)))
    out["cylinder"] = GridShape("cylinder", (float(cxy[0]), float(cxy[1]), float((lo[2] + hi[2]) / 2)), radius=rho + pad,
                                z_range=(float(lo[2] - pad), float(hi[2] + pad)))
    normals = _sphere_directions(HULL_DIRECTIONS)
    offsets = np.max(points @ normals.T, axis=0) + pad
    out["hull"] = GridShape("hull", tuple(float(v) for v in (lo + hi) / 2), normals=normals, offsets=offsets)
    return out


def fit_grid(points: np.ndarray, step: float, pad: float | None = None, shapes=("box", "sphere", "cylinder", "hull"),
             box: Box | None = None) -> tuple[np.ndarray, GridShape, dict[str, int]]:
    """The reach grid: the v1 bounding-box lattice (`Box.around(points).grid(step)`), kept inside the
    candidate shape with the fewest lattice points that still holds every point, padded by one grid
    step (`pad`). Returns (positions (P, 3), the shape used, {shape: points it would keep}).

    A flat axis (a planar arm) stays one layer. `box` overrides the bounding box (the lattice is then
    exactly the given box's: an authored box is used as is, kind ``box``)."""
    points = np.asarray(points, float).reshape(-1, 3)
    pad = float(step) if pad is None else float(pad)
    if box is not None:
        grid = box.grid(step)
        return grid, GridShape("box", tuple(box.center), box=box), {"box": len(grid)}
    bbox = Box.around(points)
    lattice = bbox.grid(step)
    fits = fit_shapes(points, pad)
    counts, kept = {}, {}
    for name in shapes:
        shape = fits[name]
        if not np.all(shape.contains(points)):  # cannot happen by construction; never trade truth for points
            continue
        keep = lattice if name == "box" else lattice[shape.contains(lattice)]
        counts[name], kept[name] = len(keep), keep
    # WHY ties go to the earlier (simpler) shape: the box is v1's answer and is easiest to read.
    name = min(counts, key=lambda k: (counts[k], shapes.index(k)))
    return kept[name], fits[name], counts


def voxel_keys(points: np.ndarray, step: float, origin: np.ndarray | None = None) -> np.ndarray:
    """(N, 3) integer voxel indices."""
    origin = np.zeros(3) if origin is None else origin
    return np.floor((points - origin) / step + 0.5).astype(np.int64)
