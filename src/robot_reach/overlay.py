"""Draw a reach map INTO someone else's viser scene (a RobotDescription viewer, a facility layout).

    from robot_reach.overlay import add_reach_map

    overlay = add_reach_map(server, reach_map, parent_frame="/facility/robot_1")
    overlay.kind = "four_dof"        # all | reachable | four_dof | hemisphere | dexterous (or a SHOW key)
    overlay.color_by = "consistency" # pose DOF | IK solutions | fraction of orientations | consistency
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
from .codes import CODE_COLORS, LEVEL_COLORS, LEVEL_NAMES, Code, level_rgb, ramp_rgb

CUBE_V = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)], float)
CUBE_F = np.array(
    [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
)
KINDS = ("all", "reachable", "four_dof", "hemisphere", "dexterous")
COLOR_BY = ("pose DOF", "IK solutions", "fraction of orientations", "consistency")

# WHY a sampled-but-empty point is light grey, not white (Zach, 2026-11-23: "add one that is All. that
# way you can see like all the voxels that were sampled in the first place"): the "no IK solution"
# code colour, so "All" shows the whole grid and empty still reads as empty.
EMPTY_RGB = CODE_COLORS[Code.NO_SOLUTION]

# Consistency (bam_reach v1's "colour by consistency"): did FK(IK) land back on the target at a point?
# A branch the library checked and passed is any of these codes (the round trip comes before them).
_ROUND_TRIP_PASSED = (Code.OK, Code.COLLISION, Code.TORQUE_LIMIT, Code.SINGULAR)
_ROUND_TRIP_FAILED = (Code.FK_MISMATCH, Code.FK_FAIL)
CONSISTENCY_NAMES = {0: "no IK answer to check", 1: "FK(IK) matched", 2: "some FK(IK) mismatched", 3: "every FK(IK) mismatched"}
CONSISTENCY_COLORS = {0: EMPTY_RGB, 1: CODE_COLORS[Code.OK], 2: (240, 160, 200), 3: CODE_COLORS[Code.FK_MISMATCH]}


def consistency(reach) -> np.ndarray:
    """(P,) per point: 0 IK gave no in-limit answer to check, 1 every checked answer's FK landed back on
    its target, 2 some did not, 3 none did. "Checked" = the answer reached the FK round trip (in limits)."""
    passed = np.isin(reach.code, [int(c) for c in _ROUND_TRIP_PASSED]).sum(axis=(1, 2))
    failed = np.isin(reach.code, [int(c) for c in _ROUND_TRIP_FAILED]).sum(axis=(1, 2))
    out = np.zeros(reach.n_positions, np.int8)
    out[(passed > 0) & (failed == 0)] = 1
    out[(passed > 0) & (failed > 0)] = 2
    out[(passed == 0) & (failed > 0)] = 3
    return out


def show_options(reach, color_by: str) -> dict[str, str]:
    """The Show filter's choices for what is being coloured by: {label: mask key}. "All" is always first.

    WHY per colour (Zach, 2026-11-23: "The show should also update to match what you are coloring by")."""
    out = {"All": "all"}
    if reach.task == "position" and color_by in ("pose DOF", "fraction of orientations"):
        color_by = "IK solutions"
    if color_by == "pose DOF":
        out.update({"reachable (3+ DOF)": "reachable", "4-DOF+": "four_dof", "hemisphere (5+ DOF)": "hemisphere", "dexterous (6 DOF)": "dexterous"})
    elif color_by == "IK solutions":
        top = int(reach.n_solutions.max()) if reach.n_positions else 0
        out.update({f"{k}+ IK solutions": f"solutions>={k}" for k in range(1, max(top, 1) + 1)})
    elif color_by == "fraction of orientations":
        out.update({"> 0 % of orientations": "fraction>0", "≥ 25 %": "fraction>=0.25", "≥ 50 %": "fraction>=0.5",
                    "≥ 75 %": "fraction>=0.75", "100 %": "fraction>=1"})
    elif color_by == "consistency":
        out.update({"IK answered (checked)": "consistency>=1", "FK(IK) matched": "consistency==1",
                    "any FK(IK) mismatch": "consistency>=2"})
    else:
        raise ValueError(f"color_by must be one of {COLOR_BY}")
    return out


def default_show(reach, color_by: str) -> str:
    """The first choice after "All": the points where something works."""
    return list(show_options(reach, color_by))[1]


def show_mask(reach, key: str) -> np.ndarray:
    """(P,) bool for a mask key (`KINDS` or a `show_options` value)."""
    if key == "all":
        return np.ones(reach.n_positions, bool)
    if key in KINDS:
        return reach.workspace(key)
    for name, values in (("solutions", lambda: reach.n_solutions.max(axis=1)), ("fraction", lambda: reach.score("dexterous")),
                         ("consistency", lambda: consistency(reach))):
        if key.startswith(name):
            rest = key[len(name):]
            for op in (">=", "==", ">"):
                if rest.startswith(op):
                    v, x = values(), float(rest[len(op):])
                    return {">=": v >= x - 1e-9, "==": v == x, ">": v > x}[op]
    raise ValueError(f"unknown Show key {key!r}: one of {KINDS} or a show_options() value")


def colors_and_legend(reach, color_by: str, idx: np.ndarray) -> tuple[np.ndarray, str]:
    """(len(idx), 3) colours for points `idx` and the legend line (HTML) for what they mean."""
    def sw(rgb):
        return f"<span style='color:rgb{tuple(int(c) for c in rgb)}'>■</span>"

    if reach.task == "position" and color_by in ("pose DOF", "fraction of orientations"):
        color_by = "IK solutions"
    if color_by == "pose DOF":
        lv = reach.levels()[idx]
        colors = level_rgb(lv)
        colors[lv == 0] = EMPTY_RGB
        legend = " ".join(f"{sw(LEVEL_COLORS[k])} {k} {LEVEL_NAMES[k]}" for k in (3, 4, 5, 6))
        legend = f"{sw(EMPTY_RGB)} 0 none · " + legend
    elif color_by == "fraction of orientations":
        v = reach.score("dexterous")[idx]
        colors = ramp_rgb(v, 1.0)
        colors[v <= 0] = EMPTY_RGB
        legend = f"{sw(EMPTY_RGB)} 0 % · white → deep blue = 0 → 100 % of all {reach.n_orientations} sampled orientations reachable"
    elif color_by == "consistency":
        c = consistency(reach)[idx]
        colors = np.array([CONSISTENCY_COLORS[int(k)] for k in c], np.uint8).reshape(-1, 3)
        legend = " ".join(f"{sw(CONSISTENCY_COLORS[k])} {CONSISTENCY_NAMES[k]}" for k in (1, 2, 3, 0))
    elif color_by == "IK solutions":
        v = reach.n_solutions.max(axis=1)[idx]
        colors = ramp_rgb(v, max(1, reach.n_branches))
        colors[v <= 0] = EMPTY_RGB
        legend = f"{sw(EMPTY_RGB)} 0 · white → deep blue = 1 → {reach.n_branches} IK solutions (best orientation)"
    else:
        raise ValueError(f"color_by must be one of {COLOR_BY}")
    return colors.astype(np.uint8), legend


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
        self._kind = self._checked(kind)
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
        self._kind = self._checked(value)
        self.redraw()

    def _checked(self, kind: str) -> str:
        """A valid mask key; a position-only map has no oriented kinds, so those read as reachable."""
        if self.reach.task == "position" and kind in ("four_dof", "hemisphere", "dexterous"):
            return "reachable"
        show_mask(self.reach, kind)  # raises on an unknown key
        return kind

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
            self._kind = self._checked(kind)
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
        mask = show_mask(r, self._kind) if self._visible else np.zeros(r.n_positions, bool)
        if self._cut is not None:
            # WHY a cut: from outside you only ever see the shell; the dexterous core is inside.
            axis, value, *rest = self._cut
            keep_below = rest[0] if rest else True
            coord = r.positions[:, axis]
            mask = mask & ((coord <= value + 1e-9) if keep_below else (coord >= value - 1e-9))
        idx = np.nonzero(mask)[0]
        self.shown_index = idx
        colors, self.legend = colors_and_legend(r, self._color_by, idx)
        # WHY empty points small and faint (Show "All"): full-size grey cubes on every sampled point
        # bury the workspace inside them; small ones still say "sampled here, nothing works".
        empty = np.all(colors == np.asarray(EMPTY_RGB, np.uint8), axis=1)
        self._full_index, self._empty_index = idx[~empty], idx[empty]
        s = self.server.scene
        size = self.step * self._voxel_scale
        # WHY dim on selection: the arm sits inside the workspace, so opaque voxels hide it.
        for name, sub, scale, opacity, handler in (
            ("voxels", ~empty, 1.0, 0.16 if self._dim else None, self._clicked),
            ("empty_voxels", empty, 0.35, 0.12 if self._dim else 0.45, self._clicked_empty),
        ):
            if sub.any():
                n = int(sub.sum())
                h = s.add_batched_meshes_simple(
                    f"{self.root}/{name}",
                    CUBE_V * size * scale,
                    CUBE_F,
                    batched_wxyzs=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
                    batched_positions=r.positions[idx[sub]].astype(np.float32),
                    batched_colors=colors[sub],
                    lod="off",
                    opacity=opacity,
                )
                h.on_click(handler)
                if name == "voxels":
                    self.voxels = h
            else:
                self._remove(f"{self.root}/{name}")
        pos = r.positions[self._full_index]
        if self._blob and len(pos):
            v, f = voxel_surface(pos, self.step)
            s.add_mesh_simple(f"{self.root}/blob", v.astype(np.float32), f, color=(34, 94, 168), opacity=0.35, side="double", flat_shading=True)
            n = count_bodies(pos, self.step)
            self.legend += f"<br>blob: {n} separate bod{'y' if n == 1 else 'ies'}"
        else:
            self._remove(f"{self.root}/blob")

    def _clicked(self, event):
        """A click on a (non-empty) voxel: `on_select` gets its point index."""
        i = event.instance_index
        if self.on_select is not None and i is not None and i < len(self._full_index):
            self.on_select(int(self._full_index[i]))

    def _clicked_empty(self, event):
        i = event.instance_index
        if self.on_select is not None and i is not None and i < len(self._empty_index):
            self.on_select(int(self._empty_index[i]))

    def _remove(self, name: str):
        try:
            self.server.scene.remove_by_name(name)
        except Exception:  # absent already
            pass

    def remove(self):
        self._remove(f"{self.root}/voxels")
        self._remove(f"{self.root}/empty_voxels")
        self._remove(f"{self.root}/blob")


def add_reach_map(server, reach_map, parent_frame: str = "/reach", **kwargs) -> ReachOverlay:
    """Draw `reach_map` under `parent_frame` in any viser server; returns the toggle handle.

    Put it under the host's robot-base frame name (e.g. "/facility/robot_1") and it follows that
    frame, because reach-map positions are expressed in the arm's base frame."""
    return ReachOverlay(server, reach_map, parent_frame=parent_frame, **kwargs)
