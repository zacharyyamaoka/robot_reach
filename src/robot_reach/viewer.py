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

2026-11-23 feedback (PROJECT - BAM Reach, "Feedback on this reach viewer"):

- Deselect: Esc, a click on empty space, or Clear; the arm then hides (unless "Arm when nothing is
  selected" is on).
- Every joint name carries its chain index, "shoulder_lift_joint (J2)".
- A sweep picture says how it was made, and its other joints' ranges can be narrowed and recomputed.
- Selected point / Ask here state their angular resolution (tilt rings, roll step, counts).
- Workspace: Show has "All" (every sampled grid point) and follows Colour by; colour by consistency.
- The FK cloud explains itself.
- Robot model: URDF visual / collision meshes, a Show switch, opacity, and the parts in contact red.

House style (viser-3d-viewer): loopback, no logo, Fusion ViewCube, panel docked right, READY line.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from .codes import CODE_COLORS, CODE_LABELS, LEVEL_NAMES, Code, code_rgb
from .joint_space import joint_label, joint_labels, joint_map
from .orientations import OrientationSet
from .overlay import ReachOverlay, default_show, show_options
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


def _deg_labelled(q) -> str:
    """``J1 +10.0° · J2 -45.0° · ...``"""
    return " · ".join(f"J{i + 1} {v:+.1f}°" for i, v in enumerate(np.rad2deg(np.asarray(q, float))))


CLEAR_LABEL = "Clear"
# WHY a key listener that clicks the Clear button: viser sends no key events to Python, and a click is
# the one client->server path every viser version has. Replayed to every client that joins.
ESCAPE_JS = """(() => {
  if (window.__robotReachEscape) return;
  window.__robotReachEscape = true;
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    const t = e.target;
    if (t && (t.tagName === 'TEXTAREA' || (t.tagName === 'INPUT' && t.type === 'text'))) return;
    const b = [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === '%s');
    if (b) b.click();
  });
})();""" % CLEAR_LABEL

MODEL_SHOWS = ("visual", "collision", "both", "none")
COLLISION_MESH_RGB = (240, 170, 70)  # a collision mesh that is not in contact: amber, see-through by default


# ================================================================== pure helpers (tested)
def ray_hits_boxes(origin, direction, centers: np.ndarray, half) -> np.ndarray:
    """(N,) bool: does the ray hit each axis-aligned box (centre, half size)? Slab test, t >= 0."""
    o, d = np.asarray(origin, float), np.asarray(direction, float)
    centers = np.asarray(centers, float).reshape(-1, 3)
    half = np.broadcast_to(np.asarray(half, float), centers.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
        t1, t2 = (centers - half - o) * inv, (centers + half - o) * inv
    t_near = np.max(np.minimum(t1, t2), axis=1)
    t_far = np.min(np.maximum(t1, t2), axis=1)
    return (t_far >= np.maximum(t_near, 0.0))


def ray_near_points(origin, direction, points: np.ndarray, radius: float) -> bool:
    """True when any point lies within `radius` of the ray (in front of its origin)."""
    points = np.asarray(points, float).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    if not len(points):
        return False
    o = np.asarray(origin, float)
    d = np.asarray(direction, float) / max(np.linalg.norm(direction), 1e-12)
    rel = points - o
    t = np.clip(rel @ d, 0.0, None)
    return bool(np.any(np.linalg.norm(rel - t[:, None] * d, axis=1) <= radius))


def floor_contacts(points: np.ndarray, floor_z: float | None) -> np.ndarray:
    """(K,) bool: the skeleton points the floor check flags, exactly as `Arm.collides` tests them
    (every point but the fixed base, strictly below the floor)."""
    points = np.asarray(points, float)
    out = np.zeros(len(points), bool)
    if floor_z is not None and len(points) > 1:
        out[1:] = points[1:, 2] < floor_z - 1e-9
    return out


def contact_parts(arm, q, floor_z: float | None) -> set[str]:
    """Link names in contact at q: for each skeleton point the floor check flags, the link at that point
    and the link whose segment ends there (the body that reached under the floor). Needs the skeleton
    points to be the arm's links in order (`arm.meta["links"]`, as a URDF arm records them); else empty."""
    pts = arm.chain(np.asarray(q, float)[None])[0]
    hit = floor_contacts(pts, floor_z)
    links = list(arm.meta.get("links") or [])
    if len(links) != len(pts):
        return set()
    return {links[j] for k in np.nonzero(hit)[0] for j in (k - 1, k) if j >= 0}


def mesh_role(mesh) -> str:
    """'visual' (drawn only), 'collision' (measured only) or 'both' (one mesh serves both)."""
    if mesh.draw and mesh.measure:
        return "both"
    return "visual" if mesh.draw else "collision"


def model_shows(meshes) -> tuple[str, ...]:
    """The robot-model Show choices this arm's meshes support."""
    roles = {mesh_role(m) for m in meshes if m.faces is not None}
    if not roles:
        return ("none",)
    if roles == {"both"}:
        return ("visual", "none")
    has_visual = bool(roles & {"visual", "both"})
    has_collision = bool(roles & {"collision", "both"})
    if has_visual and has_collision:
        return MODEL_SHOWS
    return ("visual" if has_visual else "collision", "none")


def model_styles(meshes, show: str, contact: set[str]) -> list[tuple[bool, tuple[int, int, int] | None]]:
    """Per mesh: (visible, colour override or None). A part in contact always shows its collision
    geometry, red (Zach, 2026-11-23: "turn on the collision mesh automatically ... colour the parts in
    contact")."""
    red = CODE_COLORS[Code.COLLISION]
    out = []
    for m in meshes:
        role = mesh_role(m)
        touching = m.part in contact
        if role == "visual":
            visible, color = show in ("visual", "both"), None
        elif role == "collision":
            visible, color = show in ("collision", "both") or touching, red if touching else COLLISION_MESH_RGB
        else:
            visible, color = show != "none" or touching, red if touching else None
        if touching and role != "collision" and visible:
            color = red
        out.append((visible, color))
    return out


def collision_note(arm, floor_z) -> str:
    """What the collision check actually uses, in words (it may not be the drawn meshes)."""
    parts = []
    if floor_z is not None:
        n = len(arm.chain(np.asarray(arm.meta.get("home", np.zeros(arm.dof)), float)[None])[0]) - 1
        parts.append(f"the floor at z = {floor_z:+.3f} m, tested on the skeleton's {n} moving joint points (a point below it = collision)")
    if arm.collision is not None:
        parts.append("the arm's own collision function (it does not say which parts touch)")
    if not parts:
        return "No collision is checked for this arm."
    note = "Collision is checked against " + " and ".join(parts) + "."
    if any(mesh_role(m) != "visual" for m in arm.meshes):
        note += " The URDF collision meshes are drawn, not checked; no self-collision."
    return note


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
        self._install_deselect()
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
        self.arm_q = None  # the posture drawn now (None = hidden)
        self.mesh_handles = []
        self.g_model = self.g_opacity = None
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
        self.contact = set()
        self._build_meshes()

    def _draw_arm(self, q, color=(40, 44, 52), code: int | None = None):
        """Pose the skeleton and the model meshes at q (None hides the arm). A colliding posture marks
        its contact points and parts red (`model_styles`)."""
        self.arm_q = None if q is None or self.arm is None else np.asarray(q, float)
        if self.arm_q is None:
            if self.arm_lines is not None:
                self.arm_lines.visible = False
                self.arm_joints.visible = False
            self._remove("/arm/contacts")
            for _i, h in self.mesh_handles:
                h.visible = False
            self._sync_dim()
            return
        q = self.arm_q
        chain = self.arm.chain(q[None])[0]
        segs = np.stack([chain[:-1], chain[1:]], axis=1)
        segs = segs[np.all(np.isfinite(segs.reshape(len(segs), -1)), axis=1)]
        s = self.server.scene
        thick = max(0.008, self.extent * 0.022) * (0.45 if self.mesh_handles and self._model_show() != "none" else 1.0)
        self.arm_lines = s.add_line_segments("/arm/links", segs.astype(np.float32), np.array(color, np.uint8), thickness=thick)
        joints = chain[np.all(np.isfinite(chain), axis=1)]
        self.arm_joints = s.add_point_cloud("/arm/joints", joints.astype(np.float32), np.array((90, 96, 108), np.uint8),
                                            point_size=thick * 1.25, point_shape="circle")
        floor = self.result.config.get("floor_z")
        hit = floor_contacts(chain, floor) if code == Code.COLLISION else np.zeros(len(chain), bool)
        if hit.any():
            s.add_point_cloud("/arm/contacts", chain[hit].astype(np.float32), np.array(CODE_COLORS[Code.COLLISION], np.uint8),
                              point_size=max(0.02, self.extent * 0.045), point_shape="circle")
        else:
            self._remove("/arm/contacts")
        self.contact = contact_parts(self.arm, q, floor) if code == Code.COLLISION else set()
        self._pose_meshes()
        self._sync_dim()

    def _sync_dim(self):
        """WHY: the arm sits inside the workspace, so whenever it is drawn (a selection or a Jog) the
        voxels go see-through, and back to solid when it hides."""
        if self.overlay is not None and self.overlay._dim != (self.sel_point is not None or self.arm_q is not None):
            self._redraw_workspace()

    def _remove(self, name: str):
        try:
            self.server.scene.remove_by_name(name)
        except Exception:
            pass

    def _build_meshes(self):
        """One scene mesh per `ArmMesh` with faces, created once per arm and re-posed per posture."""
        s = self.server.scene
        self.mesh_handles = []
        for i, m in enumerate(self.arm.meshes if self.arm is not None else ()):
            if m.faces is None:
                continue
            color = m.color or ((200, 203, 208) if mesh_role(m) != "collision" else COLLISION_MESH_RGB)
            h = s.add_mesh_simple(f"/arm/mesh/{i}", np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.uint32),
                                  color=tuple(int(c) for c in color), visible=False)
            self.mesh_handles.append((i, h))

    def _model_show(self) -> str:
        return self.g_model.value if getattr(self, "g_model", None) is not None else "visual"

    def _pose_meshes(self):
        if not self.mesh_handles:
            return
        if self.arm_q is None:
            for _i, h in self.mesh_handles:
                h.visible = False
            return
        T = np.asarray(self.arm.mesh_poses(self.arm_q[None]), float)[0]
        meshes = [self.arm.meshes[i] for i, _ in self.mesh_handles]
        styles = model_styles(meshes, self._model_show(), getattr(self, "contact", set()))
        opacity = float(self.g_opacity.value) if getattr(self, "g_opacity", None) is not None else 1.0
        for (i, h), m, (visible, color) in zip(self.mesh_handles, meshes, styles):
            if not np.all(np.isfinite(T[i])):
                h.visible = False
                continue
            h.position = tuple(float(v) for v in T[i, :3, 3])
            h.wxyz = tuple(float(v) for v in _wxyz(T[i, :3, :3]))
            base = m.color or ((200, 203, 208) if mesh_role(m) != "collision" else COLLISION_MESH_RGB)
            h.color = tuple(int(c) for c in (color or base))
            # WHY collision meshes stay see-through: they wrap the visual ones, so opaque they hide them
            h.opacity = opacity * (0.55 if mesh_role(m) == "collision" else 1.0)
            h.visible = visible

    # ================================================================== GUI
    def _build_gui(self):
        g = self.server.gui
        res, info = self.result, self.result.arm_info
        keep = self._gui_handles.append
        keep(g.add_markdown(self._header_md(), order=-90))

        with self._folder("Workspace"):
            if self.reach is not None:
                self._build_workspace_controls(g)
            self.g_legend = g.add_markdown("")
            self.g_cloud = g.add_checkbox("FK cloud", initial_value=self.reach is None, hint=self._cloud_help(plain=True))
            self.g_cloud_md = g.add_markdown(self._cloud_help())
            self.g_cloud.on_update(lambda _e: setattr(self.cloud_handle, "visible", self.g_cloud.value))

        if self.reach is not None:
            self._build_selection_controls(g)

        self._build_model_controls(g)

        with self._folder("Joint space", expand_by_default=True):
            self._figures = {}
            self._maps_by_name = {}
            try:
                from .plots import figure_name, figure_rgb, figures

                for name, fig in figures(res).items():
                    self._figures[name] = figure_rgb(fig)
                self._maps_by_name = {figure_name(jm): jm for jm in res.joint_maps}
            except ImportError:
                pass
            if self._figures:
                names = tuple(self._figures)
                self.g_fig = g.add_dropdown("Picture", names, initial_value=names[0])
                self.g_img = g.add_image(self._figures[names[0]], format="png")
                self.g_fig_md = g.add_markdown("")
                self.g_fig.on_update(lambda _e: self._picture_changed())
                self._build_sweep_controls(g)
                self._picture_changed()
            else:
                g.add_markdown("_Install the `plot` extra for the joint-space pictures._")

        with self._folder("Jog", expand_by_default=self.reach is None):
            lim = np.rad2deg(np.asarray([info["lower"], info["upper"]], float).T)
            home = np.zeros(len(lim)) if self.arm is None else np.rad2deg(np.clip(np.asarray(self.arm.meta.get("home", np.zeros(len(lim))), float), np.deg2rad(lim[:, 0]), np.deg2rad(lim[:, 1])))
            self.g_jog = [
                g.add_slider(f"{label} (deg)", min=float(lo), max=float(hi), step=0.5, initial_value=float(h))
                for label, (lo, hi), h in zip(joint_labels(info["joint_names"]), lim, home)
            ]
            self.g_jog_md = g.add_markdown("")
            for h in self.g_jog:
                h.on_update(lambda _e: self._jog_changed())

        if self.arm is not None and self.arm.ik is not None and self.reach is not None:
            self._build_tools()
        elif self.reach is not None:
            keep(g.add_markdown("_Ask here needs the arm's IK: this result was loaded without a rebuildable arm._"))

    def _build_workspace_controls(self, g):
        oriented = self.reach.task != "position"
        colors = ["pose DOF", "IK solutions", "fraction of orientations", "consistency"] if oriented else ["IK solutions", "consistency"]
        self.g_color = g.add_dropdown("Colour by", tuple(colors), initial_value=colors[0],
                                      hint="consistency: did FK of each IK answer land back on the target")
        self._show_keys = show_options(self.reach, colors[0])
        self.g_show = g.add_dropdown("Show", tuple(self._show_keys), initial_value=default_show(self.reach, colors[0]),
                                     hint="All = every grid point that was sampled, including those where nothing works")
        # WHY bigger for planar maps: one layer of voxels hides nothing, and gaps make clicks miss
        self.g_voxel = g.add_slider("Voxel size", min=0.1, max=1.0, step=0.05, initial_value=0.9 if self.planar else 0.55)
        self.g_cut = g.add_dropdown("Cut", ("none", "x", "y", "z"), initial_value="none")
        self.g_cut_at = g.add_slider("Cut at (m)", min=-1.0, max=1.0, step=0.005, initial_value=0.0, disabled=True)
        self.g_blob = g.add_checkbox("Blob (surface)", initial_value=False)
        for h in (self.g_show, self.g_voxel, self.g_blob, self.g_cut_at):
            h.on_update(lambda _e: self._redraw_workspace())
        self.g_color.on_update(lambda _e: self._color_changed())
        self.g_cut.on_update(lambda _e: self._cut_changed())

    def _color_changed(self):
        """The Show choices follow what is coloured; "All" stays All, anything else resets to the default."""
        keep_all = self.g_show.value == "All"
        self._show_keys = show_options(self.reach, self.g_color.value)
        target = "All" if keep_all else default_show(self.reach, self.g_color.value)
        self.g_show.options = tuple(self._show_keys)
        self.g_show.value = target
        self._redraw_workspace()

    def _build_selection_controls(self, g):
        with self._folder("Selected point"):
            self.g_sel_md = g.add_markdown("_Click a voxel to put the arm there._")
            self.g_sol = g.add_slider("Solution", min=1, max=2, step=1, initial_value=1, disabled=True)
            row = g.add_button_group("Step", ("◀ Prev", "Next ▶", CLEAR_LABEL))
            self.g_sol_md = g.add_markdown("")
            self.g_res_md = g.add_markdown("")
            g.add_markdown("<small>Esc, a click on empty space or **Clear** deselects.</small>")
            self.g_arm_idle = g.add_checkbox("Arm when nothing is selected", initial_value=False,
                                             hint="Off: the arm hides when nothing is selected. On: it stays at the Jog pose.")
            self.g_sol.on_update(lambda _e: self._show_solution())
            self.g_arm_idle.on_update(lambda _e: self._idle_arm())
            row.on_click(lambda e: self._clear_selection() if e.target.value == CLEAR_LABEL else self._step_solution(-1 if e.target.value.startswith("◀") else +1))

    def _cloud_help(self, plain: bool = False) -> str:
        """What the FK cloud is and what its colours say, from what the code did (`fk_cloud`, `analyze`)."""
        res = self.result
        n, dof = len(res.cloud_q), len(res.arm_info["joint_names"])
        how = (f"a {round(n ** (1 / dof))}-per-joint grid" if dof <= 2 else f"{n:,} uniform random samples") + " of the joint limits"
        grid = ""
        if self.reach is not None:
            grid = (f" Its bounding box is the grid the voxel workspace was computed on: {self.reach.n_positions:,} points, "
                    f"step {self.step * 1000:.0f} mm. The voxels themselves come from IK at each grid point, not from these dots.")
        text = (f"FK cloud: {how}, each pushed through FK; a dot is where the tool lands.{grid} "
                f"Dot colour = that joint posture's own check (limits, linkage, floor collision, torque, singular):")
        if plain:
            return text.rstrip(":") + "."
        return f"<small>{text} " + self._code_legend(np.unique(res.cloud_code))[len("<small>"):]

    def _resolution_md(self, reach) -> str:
        """The angular resolution behind "N / M orientations" (Zach, 2026-11-23)."""
        if reach.task == "position":
            return "<small>Position-only arm: one target per point, no orientations.</small>"
        return f"<small>Orientations tried here: {reach.orientations.describe()}.</small>"

    def _ask_resolution(self):
        O = self._orientations()
        self.g_ask_res_md.content = ("<small>Position only: no orientations.</small>" if O is None
                                     else f"<small>Compute box / Drop POI try: {O.describe()}.</small>")

    def _build_model_controls(self, g):
        """Robot model: which meshes to draw (visual / collision / both / none), opacity, and what the
        collision check really uses."""
        if self.arm is None:
            return
        with self._folder("Robot model", expand_by_default=bool(self.mesh_handles)):
            shows = model_shows([self.arm.meshes[i] for i, _ in self.mesh_handles])
            if self.mesh_handles:
                self.g_model = g.add_dropdown("Show mesh", shows, initial_value=shows[0])
                self.g_opacity = g.add_slider("Opacity", min=0.1, max=1.0, step=0.05, initial_value=1.0)
                self.g_model.on_update(lambda _e: self._model_changed())
                self.g_opacity.on_update(lambda _e: self._pose_meshes())
            else:
                self.g_model = None
                self.g_opacity = None
                g.add_markdown("<small>Skeleton only: this arm has no meshes.</small>")
            note = collision_note(self.arm, self.result.config.get("floor_z"))
            mesh_note = self.arm.meta.get("mesh_note")
            if mesh_note:
                note += f"<br>Meshes: {mesh_note}"
            g.add_markdown(f"<small>{note} A colliding pose shows its contact points and parts red.</small>")

    def _model_changed(self):
        if self.arm_q is not None:
            self._draw_arm(self.arm_q, code=Code.COLLISION if self.contact else None)

    # ------------------------------------------------------------------ joint-space pictures
    def _picture_changed(self):
        name = self.g_fig.value
        self.g_img.image = self._figures[name]
        jm = self._maps_by_name.get(name)
        info = self.result.arm_info
        limits = np.stack([info["lower"], info["upper"]], axis=1).astype(float)
        if jm is None:
            text = ("One bar per joint: light = its limits, dark = how often the working IK solutions use each angle, "
                    "ticks = the range they use.") if name == "joint ranges" else ""
        elif jm.mode == "sweep":
            from .plots import SWEEP_PRECEDENCE

            text = f"**How this is made:** {jm.sweep_text(info['joint_names'], limits)}.<br>{SWEEP_PRECEDENCE}"
        elif len(jm.axes) == 1:
            text = "The joint alone, every other joint held; the line is its static gravity torque."
        else:
            others = [j for j in range(len(jm.q0)) if j not in jm.axes]
            held = ", ".join(f"{joint_label(info['joint_names'], j)} {np.rad2deg(jm.q0[j]):.0f}°" for j in others)
            text = f"**How this is made:** one posture per cell; the other joints held at {held}." if others else "The whole joint space of this arm."
        self.g_fig_md.content = f"<small>{text}</small>" if text else ""
        if getattr(self, "g_sweep_folder", None) is not None:
            self.g_sweep_folder.visible = jm is not None and jm.mode == "sweep"

    def _build_sweep_controls(self, g):
        """Narrow the range each other joint is sampled over, then recompute the sweep (needs the arm)."""
        self.g_sweep_folder = None
        sweeps = [jm for jm in self._maps_by_name.values() if jm.mode == "sweep"]
        if self.arm is None or not sweeps:
            return
        jm = sweeps[0]
        info = self.result.arm_info
        self.g_sweep_folder = g.add_folder("Sweep: other joints' ranges", expand_by_default=False)
        with self.g_sweep_folder:
            self.g_sweep_ranges = {}
            for j in range(self.arm.dof):
                if j in jm.axes:
                    continue
                lo, hi = np.rad2deg(self.arm.limits[j])
                rng = np.rad2deg(jm.other_ranges[j]) if jm.other_ranges is not None else (lo, hi)
                self.g_sweep_ranges[j] = g.add_multi_slider(f"{joint_label(info['joint_names'], j)} (deg)", min=float(lo), max=float(hi),
                                                            step=1.0, initial_value=(float(rng[0]), float(rng[1])))
            self.g_sweep_n = g.add_number("Postures per cell", initial_value=int(jm.n_sweep or 48), min=4, max=400, step=4)
            btn = g.add_button("Recompute sweep")
            self.g_sweep_md = g.add_markdown("")
        btn.on_click(lambda _e: self._recompute_sweep())

    def _recompute_sweep(self):
        from .plots import figure_name, figure_rgb, joint_map_figure

        name = next((n for n, jm in self._maps_by_name.items() if jm.mode == "sweep"), None)
        if name is None:
            return
        old = self._maps_by_name[name]
        cfg = self.result.config
        ranges = {j: np.deg2rad(h.value) for j, h in self.g_sweep_ranges.items()}
        self.g_sweep_md.content = "computing…"
        t0 = time.perf_counter()
        jm = joint_map(self.arm, old.axes, mode="sweep", n=len(old.grids[0]), n_sweep=int(self.g_sweep_n.value),
                       floor_z=cfg.get("floor_z"), singular_tol=max(cfg.get("singular_tol", 0.0), 0.02), w_ref=cfg.get("w_ref"),
                       other_ranges=ranges)
        info = self.result.arm_info
        limits = np.stack([info["lower"], info["upper"]], axis=1).astype(float)
        self._figures[name] = figure_rgb(joint_map_figure(jm, joint_names=info["joint_names"], limits=limits))
        assert figure_name(jm) == name
        self._maps_by_name[name] = jm
        self.g_sweep_md.content = f"<small>recomputed in {_dur(time.perf_counter() - t0)}</small>"
        if self.g_fig.value == name:
            self._picture_changed()
        else:
            self.g_fig.value = name

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
        kind = self._show_keys.get(self.g_show.value, "reachable")
        cut = None
        if self.g_cut.value != "none":
            axis = "xyz".index(self.g_cut.value)
            # WHY keep the far half: the cut face then faces the camera, so you look INTO the map
            cut = (axis, float(self.g_cut_at.value), bool(self.home_direction[axis] >= 0))
        settings = dict(kind=kind, color_by=self.g_color.value, voxel_scale=float(self.g_voxel.value), blob=self.g_blob.value,
                        cut=cut, dim=self.sel_point is not None or self.arm_q is not None)
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
        self.g_legend.content = f"<small>{len(self.visible_index):,} of {reach.n_positions:,} grid points · coloured by {self.g_color.value}: {self.overlay.legend}</small>"

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
        """The deselected state: no point, voxels undimmed, the arm hidden (or at the Jog pose when
        "Arm when nothing is selected" is on)."""
        self.sel_point = None
        self.sel_solutions = []
        self.target.visible = False
        self.target_dot.visible = False
        self.g_sel_md.content = "_Click a voxel to put the arm there._"
        self.g_sol_md.content = ""
        self.g_res_md.content = ""
        self.g_sol.disabled = True
        self._idle_arm()
        self._redraw_workspace()

    def _idle_arm(self):
        if self.sel_point is not None:
            return
        if self.g_arm_idle.value:
            self._jog_changed()
        else:
            self._draw_arm(None)

    def _install_deselect(self):
        """Esc (a key listener in the page) and a click on empty space both clear the selection."""
        try:
            from viser_3d_viewer.shim import send_javascript

            send_javascript(self.server, ESCAPE_JS)
        except Exception:  # plain viser: the Clear button and empty clicks still work
            pass
        try:
            self.server.scene.on_click()(self._scene_clicked)
        except Exception:  # a viser without scene clicks
            pass

    def _scene_clicked(self, event):
        """A click anywhere in the 3-D view (viser calls it after a voxel's own click). Deselect only
        when the ray misses every voxel, the arm, the target and the Ask-here box."""
        if self.sel_point is None:
            return
        o, d = np.asarray(event.ray_origin, float), np.asarray(event.ray_direction, float)
        if not self._ray_hits_something(o, d):
            self._clear_selection()

    def _ray_hits_something(self, o, d) -> bool:
        reach = self._current_layer()
        if reach is not None and len(self.visible_index):
            half = 0.5 * self._layer_step() * max(float(self.g_voxel.value), 0.6)
            if ray_hits_boxes(o, d, reach.positions[self.visible_index], half).any():
                return True
        if self.arm_q is not None:
            chain = self.arm.chain(self.arm_q[None])[0]
            t = np.linspace(0, 1, 12)[:, None, None]
            dense = (chain[:-1][None] * (1 - t) + chain[1:][None] * t).reshape(-1, 3)
            if ray_near_points(o, d, dense, max(0.03, self.extent * 0.06)):
                return True
        if self.target_dot.visible and ray_near_points(o, d, np.asarray(self.target_dot.position)[None], self.step):
            return True
        box_on = getattr(self, "g_box_on", None) is not None and self.g_box_on.value
        if box_on:
            c = np.asarray(self.gizmo.position, float)
            if ray_hits_boxes(o, d, c[None], np.asarray(self.g_box_size.value, float) / 2 + 1e-3).any() or ray_near_points(o, d, c[None], self.extent * 0.3):
                return True
        return False

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
        self.g_res_md.content = self._resolution_md(reach)
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
        orient = "" if reach.task == "position" else f"orientation #{o + 1} of {reach.n_orientations}: tilt {tilt:.0f}° from approach, roll {roll:.0f}° · "
        self.g_sol_md.content = f"**{k + 1} / {len(self.sel_solutions)}** · {orient}IK branch {b}\n\n{_deg_labelled(q)}"
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
        self._draw_arm(q, color=(40, 44, 52) if code == Code.OK else CODE_COLORS[Code(code)], code=code)
        T, ok = self.arm.forward(q[None])
        tip = T[0, :3, 3]
        tau = self.arm.gravity_torque(q[None])
        tau_txt = "" if tau is None else f"\n\ngravity torque [{', '.join(f'{t:+.2f}' for t in tau[0])}] N·m"
        touching = ""
        if code == Code.COLLISION:
            parts = sorted(self.contact)
            touching = "\n\n<small>in contact (red): " + (", ".join(parts) if parts else "the red skeleton points") + "</small>"
        self.g_jog_md.content = (
            f"<span style='color:rgb{CODE_COLORS[Code(code)]}'>●</span> **{CODE_LABELS[Code(code)]}**"
            + (f" · tool ({tip[0]:+.3f}, {tip[1]:+.3f}, {tip[2]:+.3f}) m" if ok[0] else "")
            + tau_txt + touching
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
            self.g_ask_res_md = g.add_markdown("")
            btns = g.add_button_group("Run", ("Compute box", "Drop POI", "Back to full map"))
            self.g_tool_md = g.add_markdown("")
        self.g_approach.on_update(lambda _e: self._ask_resolution())
        self._ask_resolution()
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
