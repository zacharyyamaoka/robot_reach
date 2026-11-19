"""The viser viewer: look at a ReachResult, click a point, step through its IK solutions, and ask
new questions (a box, a point of interest) in place.

What Zach asked for (2026-11-19 daily note), and where it is:

- "see the entire xyz extent of the arm (we show those as points)" -> Workspace: voxels (clickable)
  plus the raw FK cloud.
- "the number of wrist DOFs you can achieve at each of those points you indicate with color" ->
  Colour by "pose DOF": 3 reachable, 4 top-down any yaw, 5 hemisphere, 6 dexterous.
- "put the arm at those points you can click through the different solutions" -> click a voxel,
  then the Solution slider / Prev / Next walk every (orientation, IK branch) that works there.
- "blobs ... may not account for holes" -> Blob: the voxel boundary surface (keeps holes).
- "a table-like workspace sampled at higher fidelity" + "draw the boxes and/or drop points of
  interest and it could calculate it there" -> Ask here: a box gizmo + Compute box, and Drop POI,
  with your own approach direction for 4-DOF / hemisphere.
- Joint space: the 2R pictures, the joint line and the joint-range bars in the side panel, plus Jog
  sliders that pose the arm and say which code that posture gets.

House style (viser-3d-viewer): loopback, no logo, Fusion ViewCube, panel docked right, READY line.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from .codes import CODE_COLORS, CODE_LABELS, LEVEL_NAMES, Code, code_rgb
from .orientations import OrientationSet
from .overlay import ReachOverlay
from .reach import reach_map
from .sampling import Box

APPROACHES = {
    "down (-z)": (0.0, 0.0, -1.0),
    "forward (+x)": (1.0, 0.0, 0.0),
    "up (+z)": (0.0, 0.0, 1.0),
    "left (+y)": (0.0, 1.0, 0.0),
}


def _wxyz(R: np.ndarray) -> np.ndarray:
    from robot_msgs.geometry.rotations import quat_from_matrix, xyzw_to_wxyz

    return xyzw_to_wxyz(quat_from_matrix(R))


def _dur(seconds: float) -> str:
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.1f} s"


def _deg(q) -> str:
    return ", ".join(f"{v:+.1f}" for v in np.rad2deg(np.asarray(q, float)))


class ReachViewer:
    """One viewer for one or many results (an "Arm" dropdown switches between them)."""

    ROOTS = ("/env", "/fk_cloud", "/arm", "/selection", "/workspace", "/ask")

    def __init__(self, results, *, host: str = "127.0.0.1", port: int = 0, title: str | None = None):
        if not isinstance(results, dict):
            results = {results.arm_info["name"]: results}
        self.results = results
        self.lock = threading.Lock()
        first = next(iter(results.values()))
        self._geometry(first)
        self.home_direction = np.array((0.15, -1.0, 0.35) if self.planar else (1.0, -1.0, 0.8))
        try:
            from viser_3d_viewer import Viewer, ViewerConfig

            cfg = ViewerConfig(
                host=host,
                port=port,
                title=title or ("Reach · " + " / ".join(results)),
                grid_size=4.0,
                home_target=tuple(float(c) for c in self.center),
                home_distance=max(0.6, 1.7 * self.extent),
                home_direction=(0.15, -1.0, 0.35) if self.planar else (1.0, -1.0, 0.8),
                dark_mode=False,
            )
            self.viewer = Viewer(cfg)
            self.server = self.viewer.server
        except ImportError:  # plain viser still works, without the house chrome
            import viser

            self.viewer = None
            self.server = viser.ViserServer(host=host, port=port)
        self.url = f"http://{host}:{self.server.get_port()}/"
        self._gui_handles: list = []
        if len(results) > 1:
            self.g_arm = self.server.gui.add_dropdown("Arm", tuple(results), initial_value=next(iter(results)), order=-100)
            self.g_arm.on_update(lambda _e: self.mount(self.g_arm.value, move_camera=True))
        self.mount(next(iter(results)))

    def _geometry(self, result):
        pts = result.cloud_p[np.all(np.isfinite(result.cloud_p), axis=1)]
        self.center = (pts.min(0) + pts.max(0)) / 2 if len(pts) else np.zeros(3)
        self.extent = float(np.max(np.ptp(pts, axis=0))) if len(pts) else 1.0
        self.planar = len(pts) > 0 and np.ptp(pts[:, 1]) < 1e-6
        self.step = float(result.config.get("step") or self.extent / 20)

    def mount(self, name: str, move_camera: bool = False):
        """Show one result: clear the previous arm's scene nodes and panel, build this one's."""
        with self.lock:
            for root in self.ROOTS:
                try:
                    self.server.scene.remove_by_name(root)
                except Exception:
                    pass
            for h in self._gui_handles:
                try:
                    h.remove()
                except Exception:
                    pass
            self._gui_handles = []
        self.name = name
        self.result = self.results[name]
        self.reach = self.result.reach
        self.arm = self.result.arm
        self._geometry(self.result)
        self.sel_point = None
        self.sel_solutions = []
        self.layer_reach = None
        self.overlay = None
        self.visible_index = np.zeros(0, int)
        self.pois = []
        self._build_scene()
        self._build_gui()
        self._redraw_workspace()
        if self.reach is None:
            self._jog_changed()
        self._frame_camera(move=move_camera)

    def _frame_camera(self, move: bool):
        """Home frames this arm's workspace (FOV-aware, via the house camera rig); `move` also
        moves cameras already connected (an arm switch)."""
        pts = self.result.cloud_p[np.all(np.isfinite(self.result.cloud_p), axis=1)]
        if not len(pts):
            return
        lo, hi = pts.min(0), pts.max(0)
        pad = 0.05 * self.extent
        lo, hi = lo - pad, hi + pad
        if self.planar:  # give a flat workspace some depth so the fit does not divide by zero
            lo[1], hi[1] = -0.05 * self.extent, 0.05 * self.extent
        direction = (0.15, -1.0, 0.35) if self.planar else (1.0, -1.0, 0.8)
        self.home_direction = np.asarray(direction, float)
        if self.viewer is not None:
            self.viewer.camera.fit_home(tuple(lo), tuple(hi), direction=direction, margin=1.1)
            if move:
                self.viewer.camera.home()

    # ================================================================== scene
    def _build_scene(self):
        s = self.server.scene
        res = self.result
        floor = res.config.get("floor_z")
        if floor is not None:
            size = self.extent * 1.25
            s.add_box("/env/floor", color=(214, 219, 224), dimensions=(size, size, 0.004),
                      position=(float(self.center[0]), float(self.center[1]), float(floor) - 0.002), opacity=0.45)
        ok = np.all(np.isfinite(res.cloud_p), axis=1)
        sub = np.nonzero(ok)[0]
        if len(sub) > 12_000:
            sub = sub[np.linspace(0, len(sub) - 1, 12_000).astype(int)]
        self.cloud_handle = s.add_point_cloud(
            "/fk_cloud", res.cloud_p[sub].astype(np.float32), code_rgb(res.cloud_code[sub]),
            point_size=max(0.003, self.extent / 250), point_shape="circle", visible=self.reach is None,
        )
        self.arm_lines = None
        self.arm_joints = None
        self.target = s.add_frame("/selection/target", axes_length=self.extent * 0.08, axes_radius=self.extent * 0.004, visible=False)
        self.target_dot = s.add_icosphere("/selection/dot", radius=self.step * 0.35, color=(255, 196, 0), visible=False)

    def _draw_arm(self, q, color=(40, 44, 52)):
        if self.arm is None or q is None:
            if self.arm_lines is not None:
                self.arm_lines.visible = False
                self.arm_joints.visible = False
            return
        chain = self.arm.chain(np.asarray(q, float)[None])[0]
        segs = np.stack([chain[:-1], chain[1:]], axis=1)
        segs = segs[np.all(np.isfinite(segs.reshape(len(segs), -1)), axis=1)]
        s = self.server.scene
        thick = max(0.008, self.extent * 0.022)
        self.arm_lines = s.add_line_segments("/arm/links", segs.astype(np.float32), np.array(color, np.uint8), thickness=thick)
        joints = chain[np.all(np.isfinite(chain), axis=1)]
        self.arm_joints = s.add_point_cloud("/arm/joints", joints.astype(np.float32), np.array((90, 96, 108), np.uint8),
                                            point_size=thick * 1.25, point_shape="circle")

    # ================================================================== GUI
    def _build_gui(self):
        g = self.server.gui
        res, info = self.result, self.result.arm_info
        keep = self._gui_handles.append
        keep(g.add_markdown(self._header_md(), order=-90))

        with self._folder("Workspace"):
            if self.reach is not None:
                self._build_workspace_controls(g)
            self.g_cloud = g.add_checkbox("FK cloud", initial_value=self.reach is None)
            self.g_legend = g.add_markdown("")
            self.g_cloud.on_update(lambda _e: setattr(self.cloud_handle, "visible", self.g_cloud.value))

        if self.reach is not None:
            self._build_selection_controls(g)

        with self._folder("Joint space", expand_by_default=True):
            self._figures = {}
            try:
                from .plots import figure_rgb, figures

                for name, fig in figures(res).items():
                    self._figures[name] = figure_rgb(fig)
            except ImportError:
                pass
            if self._figures:
                names = tuple(self._figures)
                self.g_fig = g.add_dropdown("Picture", names, initial_value=names[0])
                self.g_img = g.add_image(self._figures[names[0]], format="png")
                self.g_fig.on_update(lambda _e: setattr(self.g_img, "image", self._figures[self.g_fig.value]))
            else:
                g.add_markdown("_Install the `plot` extra for the joint-space pictures._")

        with self._folder("Jog", expand_by_default=self.reach is None):
            lim = np.rad2deg(np.asarray([info["lower"], info["upper"]], float).T)
            home = np.zeros(len(lim)) if self.arm is None else np.rad2deg(np.clip(np.asarray(self.arm.meta.get("home", np.zeros(len(lim))), float), np.deg2rad(lim[:, 0]), np.deg2rad(lim[:, 1])))
            self.g_jog = [
                g.add_slider(f"{n} (deg)", min=float(lo), max=float(hi), step=0.5, initial_value=float(h))
                for n, (lo, hi), h in zip(info["joint_names"], lim, home)
            ]
            self.g_jog_md = g.add_markdown("")
            for h in self.g_jog:
                h.on_update(lambda _e: self._jog_changed())

        if self.arm is not None and self.arm.ik is not None and self.reach is not None:
            self._build_tools()
        elif self.reach is not None:
            keep(g.add_markdown("_Ask here needs the arm's IK: this result was loaded without a rebuildable arm._"))

    def _build_workspace_controls(self, g):
        kinds = ["reachable"]
        oriented = self.reach.task != "position"
        if oriented:
            kinds += ["4-DOF", "hemisphere", "dexterous"]
        self.g_show = g.add_dropdown("Show", tuple(kinds), initial_value="reachable")
        colors = ["pose DOF", "IK solutions", "fraction of orientations"] if oriented else ["IK solutions"]
        self.g_color = g.add_dropdown("Colour by", tuple(colors), initial_value=colors[0])
        self.g_voxel = g.add_slider("Voxel size", min=0.1, max=1.0, step=0.05, initial_value=0.55)
        self.g_cut = g.add_dropdown("Cut", ("none", "x", "y", "z"), initial_value="none")
        self.g_cut_at = g.add_slider("Cut at (m)", min=-1.0, max=1.0, step=0.005, initial_value=0.0, disabled=True)
        self.g_blob = g.add_checkbox("Blob (surface)", initial_value=False)
        for h in (self.g_show, self.g_color, self.g_voxel, self.g_blob, self.g_cut_at):
            h.on_update(lambda _e: self._redraw_workspace())
        self.g_cut.on_update(lambda _e: self._cut_changed())

    def _build_selection_controls(self, g):
        with self._folder("Selected point"):
            self.g_sel_md = g.add_markdown("_Click a voxel to put the arm there._")
            self.g_sol = g.add_slider("Solution", min=1, max=2, step=1, initial_value=1, disabled=True)
            row = g.add_button_group("Step", ("◀ Prev", "Next ▶", "Clear"))
            self.g_sol_md = g.add_markdown("")
            self.g_sol.on_update(lambda _e: self._show_solution())
            row.on_click(lambda e: self._clear_selection() if e.target.value == "Clear" else self._step_solution(-1 if e.target.value.startswith("◀") else +1))

    def _folder(self, label: str, expand_by_default: bool = True):
        h = self.server.gui.add_folder(label, expand_by_default=expand_by_default)
        self._gui_handles.append(h)
        return h

    def _header_md(self) -> str:
        info = self.result.arm_info
        lines = [f"**{info['name']}** · {len(info['joint_names'])} DOF · task: {info['task']}", f"<small>{info['description']}</small>"]
        if self.reach is not None:
            r = self.reach
            lv = r.levels()
            parts = [f"{LEVEL_NAMES[k]} {int((lv >= k).sum())}" for k in ((3, 4, 5, 6) if r.task != "position" else (3,))]
            lines.append(f"{r.n_positions:,} grid points × {r.n_orientations} orientations × {r.n_branches} IK branches → {len(r.sol_index):,} valid solutions in {_dur(r.seconds)}")
            lines.append("points: " + " · ".join(parts))
        else:
            lines.append("Workspace is a curve: shown as the FK cloud; IK grid skipped.")
        return "\n\n".join(lines)

    # ================================================================== workspace layer
    def _current_layer(self):
        return self.layer_reach if self.layer_reach is not None else self.reach

    def _redraw_workspace(self):
        reach = self._current_layer()
        if reach is None:
            self.g_legend.content = self._code_legend(np.unique(self.result.cloud_code))
            return
        kind = {"reachable": "reachable", "4-DOF": "four_dof", "hemisphere": "hemisphere", "dexterous": "dexterous"}[self.g_show.value]
        cut = None
        if self.g_cut.value != "none":
            axis = "xyz".index(self.g_cut.value)
            # WHY keep the far half: the cut face then faces the camera, so you look INTO the map
            cut = (axis, float(self.g_cut_at.value), bool(self.home_direction[axis] >= 0))
        settings = dict(kind=kind, color_by=self.g_color.value, voxel_scale=float(self.g_voxel.value), blob=self.g_blob.value,
                        cut=cut, dim=self.sel_point is not None)
        with self.lock:
            # the overlay is the same seam a host scene uses (overlay.add_reach_map); one per layer
            if self.overlay is None or self.overlay.reach is not reach:
                if self.overlay is not None:
                    self.overlay.remove()
                self.overlay = ReachOverlay(self.server, reach, parent_frame="/workspace", step=self._layer_step(),
                                            on_select=self._select, **settings)
            else:
                self.overlay.update(**settings)
        self.visible_index = self.overlay.shown_index
        self.g_legend.content = f"<small>{len(self.visible_index):,} points · {self.overlay.legend}</small>"

    def _layer_step(self) -> float:
        reach = self._current_layer()
        if reach is self.reach:
            return self.step
        return float(reach.meta.get("step", self.step))

    def _code_legend(self, codes) -> str:
        return "<small>" + " ".join(
            f"<span style='color:rgb{CODE_COLORS[Code(int(c))]}'>●</span> {CODE_LABELS[Code(int(c))]}" for c in codes if c != Code.PAD
        ) + "</small>"

    # ================================================================== selection
    def _on_voxel_click(self, event):
        """What a voxel click does (the overlay calls `_select` with the point index)."""
        if self.overlay is not None:
            self.overlay._clicked(event)

    def _cut_changed(self):
        reach = self._current_layer()
        if self.g_cut.value == "none" or reach is None:
            self.g_cut_at.disabled = True
        else:
            axis = "xyz".index(self.g_cut.value)
            lo, hi = float(reach.positions[:, axis].min()), float(reach.positions[:, axis].max())
            self.g_cut_at.min, self.g_cut_at.max = round(lo, 3), round(hi, 3)
            self.g_cut_at.value = round((lo + hi) / 2, 3)
            self.g_cut_at.disabled = False
        self._redraw_workspace()

    def _clear_selection(self):
        self.sel_point = None
        self.sel_solutions = []
        self._draw_arm(None)
        self.target.visible = False
        self.target_dot.visible = False
        self.g_sel_md.content = "_Click a voxel to put the arm there._"
        self.g_sol_md.content = ""
        self._redraw_workspace()

    def _select(self, p: int):
        reach = self._current_layer()
        was_selected = self.sel_point is not None
        self.sel_point = p
        if not was_selected:
            self._redraw_workspace()
        sols = reach.solutions(p)
        tilt = reach.orientations.tilt
        sols.sort(key=lambda s: (float(tilt[s[0]]), s[0], s[1]))  # straight-down approach first
        self.sel_solutions = sols
        lv = int(reach.levels()[p])
        x, y, z = reach.positions[p]
        self.g_sel_md.content = (
            f"**({x:+.3f}, {y:+.3f}, {z:+.3f}) m** · {LEVEL_NAMES.get(lv, lv)} ({lv} DOF)\n\n"
            f"{len(sols)} working (orientation, IK branch) pairs · "
            f"{int(reach.valid_mask[p].sum())}/{reach.n_orientations} orientations"
        )
        n = max(1, len(sols))
        self.g_sol.max = max(2, n)
        self.g_sol.value = 1
        self.g_sol.disabled = len(sols) < 2
        self._show_solution()

    def _step_solution(self, d: int):
        if not self.sel_solutions:
            return
        self.g_sol.value = int((self.g_sol.value - 1 + d) % len(self.sel_solutions)) + 1

    def _show_solution(self):
        reach = self._current_layer()
        if self.sel_point is None or reach is None:
            return
        p = self.sel_point
        pos = reach.positions[p]
        if not self.sel_solutions:
            self._draw_arm(None)
            self.target_dot.position = tuple(pos)
            self.target_dot.visible = True
            self.target.visible = False
            self.g_sol_md.content = "_No valid solution here._"
            return
        k = min(int(self.g_sol.value), len(self.sel_solutions)) - 1
        o, b, q = self.sel_solutions[k]
        self._draw_arm(q)
        self.target_dot.position = tuple(pos)
        self.target_dot.visible = True
        if reach.task != "position":
            self.target.position = tuple(pos)
            self.target.wxyz = tuple(_wxyz(reach.orientations.R[o]))
            self.target.visible = True
        else:
            self.target.visible = False
        tilt, roll = np.rad2deg(reach.orientations.tilt[o]), np.rad2deg(reach.orientations.roll[o])
        orient = "" if reach.task == "position" else f"orientation #{o}: tilt {tilt:.0f}° from approach, roll {roll:.0f}° · "
        self.g_sol_md.content = f"**{k + 1} / {len(self.sel_solutions)}** · {orient}IK branch {b}\n\nq = [{_deg(q)}]°"
        for h, v in zip(self.g_jog, np.rad2deg(q)):
            h.value = float(np.clip(v, h.min, h.max))

    def select_nearest(self, xyz, solution: int = 1):
        """Select the shown voxel nearest to xyz (the scripted twin of a click)."""
        if len(self.visible_index) == 0:
            return None
        reach = self._current_layer()
        d = np.linalg.norm(reach.positions[self.visible_index] - np.asarray(xyz, float), axis=1)
        p = int(self.visible_index[int(np.argmin(d))])
        self._select(p)
        if self.sel_solutions:
            self.g_sol.value = int(np.clip(solution, 1, len(self.sel_solutions)))
        return p

    # ================================================================== jog
    def _jog_changed(self):
        if self.arm is None:
            return
        q = np.deg2rad([h.value for h in self.g_jog])
        cfg = self.result.config
        code = int(self.arm.check(q[None], floor_z=cfg.get("floor_z"), singular_tol=max(cfg.get("singular_tol", 0.0), 0.02), w_ref=cfg.get("w_ref"))[0])
        self._draw_arm(q, color=(40, 44, 52) if code == Code.OK else CODE_COLORS[Code(code)])
        T, ok = self.arm.forward(q[None])
        tip = T[0, :3, 3]
        tau = self.arm.gravity_torque(q[None])
        tau_txt = "" if tau is None else f"\n\ngravity torque [{', '.join(f'{t:+.2f}' for t in tau[0])}] N·m"
        self.g_jog_md.content = (
            f"<span style='color:rgb{CODE_COLORS[Code(code)]}'>●</span> **{CODE_LABELS[Code(code)]}**"
            + (f" · tool ({tip[0]:+.3f}, {tip[1]:+.3f}, {tip[2]:+.3f}) m" if ok[0] else "")
            + tau_txt
        )

    # ================================================================== ask here
    def _build_tools(self):
        g, s = self.server.gui, self.server.scene
        size0 = max(self.step * 4, self.extent * 0.25)
        start = self.center + np.array([self.extent * 0.25, 0.0, 0.0])
        if self.planar:
            start[1] = 0.0
        with self._folder("Ask here", expand_by_default=True):
            g.add_markdown("<small>Drag the gizmo. **Compute box** samples it finer; **Drop POI** checks every orientation at its centre.</small>")
            self.g_box_on = g.add_checkbox("Show box", initial_value=False)
            self.g_box_size = g.add_vector3("Box size (m)", initial_value=tuple(float(size0) for _ in range(3)) if not self.planar else (float(size0), 0.0, float(size0)), step=0.01)
            self.g_box_step = g.add_number("Box step (m)", initial_value=round(max(self.step / 2, 0.005), 3), step=0.005, min=0.002)
            self.g_approach = g.add_dropdown("Approach (4-DOF / hemisphere)", tuple(APPROACHES), initial_value="down (-z)")
            btns = g.add_button_group("Run", ("Compute box", "Drop POI", "Back to full map"))
            self.g_tool_md = g.add_markdown("")
        self.gizmo = s.add_transform_controls("/ask/gizmo", scale=self.extent * 0.25, position=tuple(start), disable_rotations=True, visible=False)
        self.box_handle = s.add_box("/ask/gizmo/box", color=(245, 159, 0), dimensions=tuple(self.g_box_size.value), wireframe=True, visible=False)
        self.g_box_on.on_update(lambda _e: self._toggle_box())
        self.g_box_size.on_update(lambda _e: setattr(self.box_handle, "dimensions", tuple(max(1e-4, v) for v in self.g_box_size.value)))
        btns.on_click(lambda e: self._run_tool(e.target.value))

    def _toggle_box(self):
        on = self.g_box_on.value
        self.gizmo.visible = on
        self.box_handle.visible = on

    def _orientations(self):
        if self.reach.task == "position":
            return None
        return OrientationSet.sample(approach=APPROACHES[self.g_approach.value])

    def _run_tool(self, which: str):
        if which == "Back to full map":
            self.layer_reach = None
            self.server.scene.remove_by_name("/ask/poi_axes") if self.pois else None
            self.g_tool_md.content = ""
            self._redraw_workspace()
            return
        self.gizmo.visible = True
        self.g_box_on.value = True
        self._toggle_box()
        c = np.asarray(self.gizmo.position, float)
        cfg = self.result.config
        t0 = time.perf_counter()
        if which == "Compute box":
            size = np.maximum(np.asarray(self.g_box_size.value, float), 0.0)
            step = float(self.g_box_step.value)
            positions = Box(tuple(c), tuple(size)).grid(step)
            if len(positions) > 60_000:
                self.g_tool_md.content = f"**{len(positions):,} points is too many** — raise the step."
                return
            self.g_tool_md.content = f"computing {len(positions):,} points…"
            rm = reach_map(self.arm, positions, self._orientations(), floor_z=cfg.get("floor_z"), singular_tol=cfg.get("singular_tol", 0.0), w_ref=cfg.get("w_ref"))
            rm.meta["step"] = step
            self.layer_reach = rm
            lv = rm.levels()
            kinds = (3, 4, 5, 6) if rm.task != "position" else (3,)
            pct = " · ".join(f"{LEVEL_NAMES[k]} {100 * np.mean(lv >= k):.0f}%" for k in kinds)
            self.g_tool_md.content = f"**Box: {len(positions):,} points** in {_dur(time.perf_counter() - t0)}\n\n{pct} of the box"
            self._redraw_workspace()
        else:  # Drop POI
            O = self._orientations()
            rm = reach_map(self.arm, c[None], O, floor_z=cfg.get("floor_z"), singular_tol=cfg.get("singular_tol", 0.0), w_ref=cfg.get("w_ref"))
            rm.meta["step"] = self.step
            self.layer_reach = rm
            self.pois.append(c)
            lv = int(rm.levels()[0])
            if rm.task != "position":
                # a hedgehog: one spike per sampled approach direction, green where some roll works
                dirs = rm.orientations.R[:, :, 2]
                ok = rm.valid_mask[0]
                L = self.extent * 0.12
                segs = np.stack([np.broadcast_to(c, dirs.shape), c - dirs * L], axis=1)
                cols = np.where(ok[:, None], np.array(CODE_COLORS[Code.OK]), np.array((200, 200, 200)))
                cols = np.repeat(cols[:, None, :], 2, axis=1)  # viser wants a colour per segment end
                self.server.scene.add_line_segments("/ask/poi_axes", segs.astype(np.float32), cols.astype(np.uint8), thickness=self.extent * 0.004)
            self.g_tool_md.content = (
                f"**POI ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f})** · {LEVEL_NAMES.get(lv, lv)}\n\n"
                f"{int(rm.valid_mask[0].sum())}/{rm.n_orientations} orientations · {len(rm.sol_index)} solutions"
                + ("\n\n<small>spikes point back along each approach: green = reachable</small>" if rm.task != "position" else "")
            )
            self._redraw_workspace()
            self._select(0)

    # ================================================================== lifecycle
    def run(self, block: bool = True):
        if self.viewer is not None:
            self.viewer.print_ready()
        else:
            print(f"READY {self.url}", flush=True)
        if not block:
            return self
        if self.viewer is not None:
            self.viewer.serve_forever()
        else:
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                self.server.stop()
        return self

    def stop(self):
        if self.viewer is not None:
            self.viewer.stop()
        else:
            self.server.stop()
