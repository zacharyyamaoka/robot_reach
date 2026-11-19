"""Draw a reach map INTO someone else's viser scene (a RobotDescription viewer, a facility layout).

    from robot_reach.overlay import add_reach_map

    overlay = add_reach_map(server, reach_map, parent_frame="/facility/robot_1")
    overlay.kind = "four_dof"        # reachable | four_dof | hemisphere | dexterous
    overlay.visible = False

WHY a separate seam from `display()` (Zach's 2026-11-19 pipeline, step 6: "place it next to a
conveyor belt or in a Customer Facility — being able to see the reachability map here is helpful
for knowing how to position the robot"): the reach map has to live inside the host's scene, under
the host's robot-base frame, so it moves when the robot is placed.

Version-neutral: this module imports nothing from viser and only calls APIs present in both viser
1.0.16 (bam_robot_descriptions' pin) and 1.1.1 (viser-3d-viewer's pin): `add_batched_meshes_simple`
(batched_colors, lod, opacity), `add_mesh_simple`, and `on_click` with `instance_index`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .blobs import count_bodies, voxel_surface
from .codes import LEVEL_COLORS, LEVEL_NAMES, level_rgb, ramp_rgb

CUBE_V = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)], float)
CUBE_F = np.array(
    [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
)
KINDS = ("reachable", "four_dof", "hemisphere", "dexterous")
COLOR_BY = ("pose DOF", "IK solutions", "fraction of orientations")


def infer_step(positions: np.ndarray) -> float:
    """Grid spacing of a lattice of points (smallest non-zero gap along any axis)."""
    gaps = []
    for a in range(3):
        u = np.unique(np.round(positions[:, a], 9))
        if len(u) > 1:
            gaps.append(np.min(np.diff(u)))
    return float(min(gaps)) if gaps else 0.05


class ReachOverlay:
    """Voxels (clickable) and an optional blob for one ReachMap, under `parent_frame`."""

    def __init__(
        self,
        server,
        reach,
        *,
        parent_frame: str = "/reach",
        kind: str = "reachable",
        color_by: str = "pose DOF",
        voxel_scale: float = 0.55,
        blob: bool = False,
        step: float | None = None,
        on_select: Callable[[int], None] | None = None,
        cut: tuple[int, float] | None = None,
        dim: bool = False,
    ):
        self.server = server
        self.reach = reach
        self.root = parent_frame.rstrip("/") or "/reach"
        self.step = float(step or reach.meta.get("step") or infer_step(reach.positions))
        self.on_select = on_select
        self._kind = kind if reach.task != "position" else "reachable"
        self._color_by = color_by if reach.task != "position" else "IK solutions"
        self._voxel_scale = voxel_scale
        self._blob = blob
        self._cut = cut
        self._dim = dim
        self._visible = True
        self.shown_index = np.zeros(0, int)
        self.legend = ""
        self.redraw()

    # -------------------------------------------------------------- toggles (each redraws)
    @property
    def kind(self) -> str:
        return self._kind

    @kind.setter
    def kind(self, value: str):
        if value not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self._kind = value if self.reach.task != "position" else "reachable"
        self.redraw()

    @property
    def color_by(self) -> str:
        return self._color_by

    @color_by.setter
    def color_by(self, value: str):
        self._color_by = value
        self.redraw()

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        self._visible = bool(value)
        self.redraw()

    def update(self, *, kind=None, color_by=None, voxel_scale=None, blob=None, cut="keep", dim=None):
        """Change several settings with one redraw. `cut` = (axis 0/1/2, value[, keep_below]) or None."""
        if kind is not None:
            self._kind = kind if self.reach.task != "position" else "reachable"
        if color_by is not None:
            self._color_by = color_by
        if voxel_scale is not None:
            self._voxel_scale = float(voxel_scale)
        if blob is not None:
            self._blob = bool(blob)
        if cut != "keep":
            self._cut = cut
        if dim is not None:
            self._dim = bool(dim)
        self.redraw()

    # -------------------------------------------------------------- drawing
    def redraw(self):
        r = self.reach
        mask = r.workspace(self._kind) if self._visible else np.zeros(r.n_positions, bool)
        if self._cut is not None:
            # WHY a cut: from outside you only ever see the shell; the dexterous core is inside.
            axis, value, *rest = self._cut
            keep_below = rest[0] if rest else True
            coord = r.positions[:, axis]
            mask = mask & ((coord <= value + 1e-9) if keep_below else (coord >= value - 1e-9))
        idx = np.nonzero(mask)[0]
        self.shown_index = idx
        pos = r.positions[idx]
        if self._color_by == "pose DOF":
            colors = level_rgb(r.levels()[idx])
            self.legend = " ".join(f"<span style='color:rgb{LEVEL_COLORS[k]}'>■</span> {k} {LEVEL_NAMES[k]}" for k in (3, 4, 5, 6))
        elif self._color_by == "fraction of orientations":
            colors = ramp_rgb(r.score("dexterous")[idx], 1.0)
            self.legend = "white → deep blue = 0 → 100 % of all sampled orientations reachable"
        else:
            colors = ramp_rgb(r.n_solutions.max(axis=1)[idx], max(1, r.n_branches))
            self.legend = f"white → deep blue = 0 → {r.n_branches} IK solutions (best orientation)"
        s = self.server.scene
        size = self.step * self._voxel_scale
        if len(pos):
            self.voxels = s.add_batched_meshes_simple(
                f"{self.root}/voxels",
                CUBE_V * size,
                CUBE_F,
                batched_wxyzs=np.tile([1.0, 0.0, 0.0, 0.0], (len(pos), 1)),
                batched_positions=pos.astype(np.float32),
                batched_colors=colors,
                lod="off",
                # WHY dim on selection: the arm sits inside the workspace, so opaque voxels hide it.
                opacity=0.16 if self._dim else None,
            )
            self.voxels.on_click(self._clicked)
        else:
            self._remove(f"{self.root}/voxels")
        if self._blob and len(pos):
            v, f = voxel_surface(pos, self.step)
            s.add_mesh_simple(f"{self.root}/blob", v.astype(np.float32), f, color=(34, 94, 168), opacity=0.35, side="double", flat_shading=True)
            n = count_bodies(pos, self.step)
            self.legend += f"<br>blob: {n} separate bod{'y' if n == 1 else 'ies'}"
        else:
            self._remove(f"{self.root}/blob")

    def _clicked(self, event):
        i = event.instance_index
        if self.on_select is not None and i is not None and i < len(self.shown_index):
            self.on_select(int(self.shown_index[i]))

    def _remove(self, name: str):
        try:
            self.server.scene.remove_by_name(name)
        except Exception:  # absent already
            pass

    def remove(self):
        self._remove(f"{self.root}/voxels")
        self._remove(f"{self.root}/blob")


def add_reach_map(server, reach_map, parent_frame: str = "/reach", **kwargs) -> ReachOverlay:
    """Draw `reach_map` under `parent_frame` in any viser server; returns the toggle handle.

    Put it under the host's robot-base frame name (e.g. "/facility/robot_1") and it follows that
    frame, because reach-map positions are expressed in the arm's base frame."""
    return ReachOverlay(server, reach_map, parent_frame=parent_frame, **kwargs)
