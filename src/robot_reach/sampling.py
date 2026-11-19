"""Where to ask: FK clouds, boxes and grids.

Two ways to find a workspace, used together:

- **Forward** (`fk_cloud`): sample joint space, run FK. Always works, for any DOF, including a
  1-DOF pendulum whose workspace is a curve no Cartesian grid would ever hit exactly.
- **Inverse** (a grid + `reach_map`): ask IK at chosen targets. Exact per target, counts every IK
  branch, and is what a task ("can I pick here, straight down, at any yaw?") actually needs.

WHY the FK cloud bounds the default grid (kept from bam_reach v1, "FK sampling creates a
conservative centered grid"): no one has to guess the arm's extent, and the grid never clips it.
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


def voxel_keys(points: np.ndarray, step: float, origin: np.ndarray | None = None) -> np.ndarray:
    """(N, 3) integer voxel indices."""
    origin = np.zeros(3) if origin is None else origin
    return np.floor((points - origin) / step + 0.5).astype(np.int64)
