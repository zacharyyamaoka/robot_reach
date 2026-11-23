"""The viser viewer: look at a ReachResult, click a point, step through its IK solutions, and ask
new questions (a box, a point of interest) in place.

What Zach asked for (2026-11-19 daily note), and where it is:

- "see the entire xyz extent of the arm (we show those as points)" -> Workspace: voxels (clickable)
  plus the raw FK cloud.
- "the number of wrist DOFs you can achieve at each of those points you indicate with color" ->
  Colour by "workspace": reachable, top-down (any yaw), hemisphere, dexterous.
- "put the arm at those points you can click through the different solutions" -> click a voxel,
  then the Solution slider / Prev / Next walk every (orientation, IK solution) that works there.
- "blobs ... may not account for holes" -> Blob: the voxel boundary surface (keeps holes).
- "a table-like workspace sampled at higher fidelity" + "draw the boxes and/or drop points of
  interest and it could calculate it there" -> Ask here: a box gizmo + Compute box, and Drop POI,
  the box's own frame giving the approach for top-down / hemisphere.
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

2026-11-23 feedback, round 2 (Daily Note, "Feedback on Robot Reach"):

- Workspace levels by name (reachable, top-down, hemisphere, dexterous), never "3+ DOF".
- A settings card (F2): what the map was computed with; edit, Compute workspace, Reset.
- Colour by: heat maps through a perceptual colormap (viridis default; magma, plasma, inferno, cividis;
  reverse), good/bad views a traffic light; the legend is a list (E4): swatch, label, count, click to hide.
- IK configurations (A1): one row per question, toggles, at least one answer each; filters the stepper
  and the map counts. "IK solution", never "branch", in anything a person reads.
- Joint ranges: one histogram per joint (B2).
- Ask here (C2): the box moves AND rotates in the 3-D view (viser's transform gizmo); its Z is the approach
  axis; the angle range and steps are fields, with the Python line that runs.
- Robot model: show / hide the skeleton.

House style (viser-3d-viewer): loopback, no logo, Fusion ViewCube, panel docked right, READY line.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from .codes import CODE_COLORS, CODE_LABELS, COLORMAPS, DEFAULT_COLORMAP, LEVEL_MEANINGS, LEVEL_NAMES, Code, code_rgb, is_collision
from .joint_space import joint_label, joint_labels, joint_map, ranges_of
from .orientations import OrientationSet
from .overlay import COLOR_BY, ReachOverlay, bins, color_mode
from .reach import reach_map
from .sampling import Box, fit_grid

GRID_SHAPE_CHOICES = ("auto (fewest points)", "box", "sphere", "cylinder", "hull")


def _wxyz(R: np.ndarray) -> np.ndarray:
    from robot_msgs.geometry.rotations import quat_from_matrix, xyzw_to_wxyz

    return xyzw_to_wxyz(quat_from_matrix(R))


def _dur(seconds: float) -> str:
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.1f} s"


def _deg(q) -> str:
    return ", ".join(f"{v:+.1f}" for v in np.rad2deg(np.asarray(q, float)))


def rpy_matrix(rpy_deg) -> np.ndarray:
    """Roll, pitch, yaw (deg) -> 3x3, R = Rz(yaw) Ry(pitch) Rx(roll) (the URDF convention)."""
    r, p, y = np.deg2rad(np.asarray(rpy_deg, float))
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def matrix_rpy(R) -> np.ndarray:
    """3x3 -> roll, pitch, yaw (deg), the inverse of `rpy_matrix` (pitch in [-90, 90])."""
    R = np.asarray(R, float)
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-9:
        roll, yaw = np.arctan2(R[2, 1], R[2, 2]), np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock: put it all in yaw
        roll, yaw = 0.0, np.arctan2(-R[0, 1], R[1, 1])
    return np.rad2deg([roll, pitch, yaw])


def wxyz_matrix(wxyz) -> np.ndarray:
    w, x, y, z = np.asarray(wxyz, float) / max(np.linalg.norm(wxyz), 1e-12)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def box_orientations(R_box, max_tilt_deg: float, tilt_step_deg: float, roll_step_deg: float) -> OrientationSet:
    """The orientation set an Ask-here box asks: its approach is the box's -Z (straight down when the box
    is level: box +Z is the workspace's up), tilted out to `max_tilt_deg` in `tilt_step_deg` rings, rolled
    every `roll_step_deg`."""
    approach = -np.asarray(R_box, float)[:, 2]
    n_rolls = max(1, int(round(360.0 / max(roll_step_deg, 1e-6))))
    return OrientationSet.sample(approach=tuple(approach), tilt_step_deg=max(float(tilt_step_deg), 1.0), n_rolls=n_rolls,
                                 max_tilt_deg=float(np.clip(max_tilt_deg, 0.0, 180.0)))


def box_positions(center, size, rpy_deg, step) -> np.ndarray:
    """An Ask-here box's grid: `Box.grid` in the box's own frame, rotated by `rpy_deg` and moved to `center`."""
    local = Box((0.0, 0.0, 0.0), tuple(float(v) for v in size)).grid(step)
    return local @ rpy_matrix(rpy_deg).T + np.asarray(center, float)


def toggle_choice(selected: set[int], option: int, n_options: int) -> set[int]:
    """One press on an IK-configuration toggle. Each question keeps at least one answer (Zach, 2026-11-23:
    "please make the buttons toggable though if I press front then back should light up. so each
    configuration always has at least 1 and at most 2"): pressing an unlit answer lights it; pressing a lit
    one turns it off, and when it was the only one lit the other answer lights instead."""
    selected = set(selected)
    if option not in selected:
        return selected | {option}
    if len(selected) > 1:
        return selected - {option}
    if n_options == 2:
        return {1 - option}
    return selected


CHIP_ON, CHIP_OFF = "✓ ", "○ "


def chip_labels(options, selected) -> tuple[str, ...]:
    return tuple((CHIP_ON if i in selected else CHIP_OFF) + o for i, o in enumerate(options))


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

# WHY CSS from a script: viser buttons and checkbox labels have no colour per option. This lights the
# IK-configuration chips ("✓ front" filled, "○ back" outlined) and draws each legend row's swatch and count
# bar from `window.__rrLegend` ({label text: {rgb, frac}}), re-applied on every DOM change. Only inline
# styles are touched (no nodes added), so React's re-renders stay safe. Without it, the ✓/○ marks and
# the counts in the labels still say everything.
DECORATE_JS = """(() => {
  if (window.__rrDecor) return;
  window.__rrDecor = true;
  const apply = () => {
    document.querySelectorAll('button').forEach((b) => {
      const t = (b.textContent || '').trim();
      if (t.startsWith('\u2713 ')) {
        b.style.background = '#228be6'; b.style.color = '#fff'; b.style.border = '1px solid #228be6';
      } else if (t.startsWith('\u25cb ')) {
        b.style.background = '#fff'; b.style.color = '#495057'; b.style.border = '1px solid #ced4da';
      }
    });
    const legend = window.__rrLegend || {};
    document.querySelectorAll('label').forEach((el) => {
      const t = (el.textContent || '').trim();  // viser puts <wbr> after hyphens: read the text, not a leaf
      if (!t || !(t in legend)) return;
      const row = legend[t];
      const c = 'rgb(' + row.rgb.join(',') + ')';
      const pct = Math.round(100 * row.frac);
      el.style.borderLeft = '12px solid ' + c;
      el.style.paddingLeft = '6px';
      el.style.background = 'linear-gradient(to right, rgba(' + row.rgb.join(',') + ',0.22) ' + pct + '%, transparent ' + pct + '%)';
    });
  };
  window.__rrApply = apply;
  new MutationObserver(() => apply()).observe(document.body, {subtree: true, childList: true, characterData: true});
  apply();
})();"""

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


def mesh_opacity(mesh, show: str, touching: bool, slider: float) -> float | None:
    """The opacity to give a robot mesh; None = opaque.

    WHY None at full opacity, not 1.0: viser renders any mesh with an opacity set in the transparent
    pass, after the (see-through) voxels, which have already written depth, so every part inside the
    workspace cloud vanished (seen 2026-11-23: only the UR5e's wrist, outside the cloud, showed).
    WHY a collision mesh is see-through only under "both": there it wraps the visual mesh it would hide."""
    value = float(slider)
    if mesh_role(mesh) == "collision" and show == "both" and not touching:
        value *= 0.55
    return None if value >= 0.999 else value


def collision_note(arm, floor_z) -> str:
    """Where collision comes from for this arm, in words. WHY (Zach, 2026-11-23: "reach doesn't really know
    anything about collision"): collision is a code the arm's FK/IK return; the library only colours it."""
    parts = []
    if floor_z is not None:
        n = len(arm.chain(np.asarray(arm.meta.get("home", np.zeros(arm.dof)), float)[None])[0]) - 1
        parts.append(f"an optional floor at z = {floor_z:+.3f} m, tested on the skeleton's {n} moving joint points (environment collision)")
    if arm.collision is not None:
        parts.append("the arm's collision adapter (it does not say which parts touch)")
    note = ("Collision is whatever this arm's FK/IK report (self-collision, environment collision); "
            + ("these add " + " and ".join(parts) + "." if parts else "nothing else is checked here: the arm is in free space."))
    if any(mesh_role(m) != "visual" for m in arm.meshes):
        note += " The URDF collision meshes are drawn, not checked."
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
        skeleton_on = self._skeleton_on()
        self.arm_lines = s.add_line_segments("/arm/links", segs.astype(np.float32), np.array(color, np.uint8), thickness=thick,
                                             visible=skeleton_on)
        joints = chain[np.all(np.isfinite(chain), axis=1)]
        self.arm_joints = s.add_point_cloud("/arm/joints", joints.astype(np.float32), np.array((90, 96, 108), np.uint8),
                                            point_size=thick * 1.25, point_shape="circle", visible=skeleton_on)
        floor = self.result.config.get("floor_z")
        colliding = code is not None and bool(is_collision(code))
        hit = floor_contacts(chain, floor) if colliding else np.zeros(len(chain), bool)
        if hit.any():
            s.add_point_cloud("/arm/contacts", chain[hit].astype(np.float32), np.array(CODE_COLORS[Code.COLLISION], np.uint8),
                              point_size=max(0.02, self.extent * 0.045), point_shape="circle")
        else:
            self._remove("/arm/contacts")
        self.contact = contact_parts(self.arm, q, floor) if colliding else set()
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
                                  color=tuple(int(c) for c in color), visible=False,
                                  # WHY double-sided: CAD exports often wind faces inside out (UR's .dae
                                  # link tubes do), and a one-sided render culls those parts away.
                                  side="double")
            self.mesh_handles.append((i, h))

    def _skeleton_on(self) -> bool:
        """WHY a switch (Zach, 2026-11-23: "For the URDF you should have an option to show or hide the
        skeleton"): with meshes drawn the skeleton is a second picture of the same arm. An arm without
        meshes always shows it (it is the only drawing)."""
        h = getattr(self, "g_skeleton", None)
        return True if h is None or not self.mesh_handles else bool(h.value)

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
            h.opacity = mesh_opacity(m, self._model_show(), m.part in self.contact, opacity)
            h.visible = visible

    # ================================================================== GUI
    def _build_gui(self):
        g = self.server.gui
        res, info = self.result, self.result.arm_info
        keep = self._gui_handles.append
        self.g_header = g.add_markdown(self._header_md(), order=-90)
        keep(self.g_header)

        self.g_ws_folder = self._folder("Workspace")
        with self.g_ws_folder:
            if self.reach is not None:
                self._build_workspace_controls(g)
            self.g_legend = g.add_markdown("", order=60)
            if self.reach is not None:
                self._build_view_controls(g)
            self.g_cloud = g.add_checkbox("FK cloud", initial_value=self.reach is None, hint=self._cloud_help(plain=True), order=80)
            self.g_cloud_md = g.add_markdown(self._cloud_help(), order=81)
            self.g_cloud.on_update(lambda _e: setattr(self.cloud_handle, "visible", self.g_cloud.value))

        if self.reach is not None:
            self._build_config_controls(g)
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
                self._img_shown = self._figures[names[0]]
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

    # ------------------------------------------------------------------ workspace: settings card (F2)
    def _build_workspace_controls(self, g):
        """The settings card (Zach's pick F2, 2026-11-23: "I like F2 used setting cards"): the fields show
        what this map was computed with; editing one lights Compute workspace and hides the old map until
        it is recomputed or Reset."""
        from .analyze import settings_of

        st = settings_of(self.result)
        orient = st.get("orientation") or {}
        can = self.arm is not None and self.arm.ik is not None
        self._settings_base = {
            "step": round(float(st["step_m"]) * 1000, 1),
            "shape": GRID_SHAPE_CHOICES[0] if st.get("grid_shape_auto") else st.get("grid_shape", "box"),
            "tilt": float(orient.get("tilt_step_deg") or 45.0),
            "roll": float(orient.get("roll_step_deg") or 360.0),
            "max_tilt": float(orient.get("tilt_max_deg", 180.0)),
        }
        b = self._settings_base
        self.g_set_md = g.add_markdown("**Computed with** <small>— edit to change" + ("" if can else " (needs the live arm)") + "</small>", order=1)
        self.g_set_step = g.add_number("Voxel size (mm)", initial_value=b["step"], min=1.0, step=1.0, order=2, disabled=not can,
                                       hint="Grid step: the distance between neighbouring grid points")
        self.g_set_shape = g.add_dropdown("Grid shape", GRID_SHAPE_CHOICES if b["shape"] in GRID_SHAPE_CHOICES else (*GRID_SHAPE_CHOICES, b["shape"]),
                                          initial_value=b["shape"], order=3, disabled=not can,
                                          hint="The volume the grid is kept inside; auto = the shape with the fewest points that holds every FK point (+ one voxel)")
        oriented = self.reach.task != "position"
        self.g_set_tilt = g.add_number("Orientation step (°)", initial_value=b["tilt"], min=5.0, max=180.0, step=5.0, order=4,
                                       disabled=not can, visible=oriented, hint="Tilt rings: angle between approach directions")
        self.g_set_roll = g.add_number("Roll step (°)", initial_value=b["roll"], min=5.0, max=360.0, step=5.0, order=5,
                                       disabled=not can, visible=oriented, hint="Rolls about the tool axis at each direction")
        self.g_set_max = g.add_number("Angles up to (°)", initial_value=b["max_tilt"], min=0.0, max=180.0, step=5.0, order=6,
                                      disabled=not can, visible=oriented, hint="Tilt from the approach axis: 0 = straight down only, 180 = every direction")
        self.g_set_btns = g.add_button("Compute workspace", disabled=True, order=7)
        self.g_set_reset = g.add_button("Reset", disabled=True, order=8, color="gray")
        self.g_set_used = g.add_markdown(self._settings_used_md(), order=9)
        for h in (self.g_set_step, self.g_set_shape, self.g_set_tilt, self.g_set_roll, self.g_set_max):
            h.on_update(lambda _e: self._settings_changed())
        self.g_set_btns.on_click(lambda _e: self._compute_workspace())
        self.g_set_reset.on_click(lambda _e: self._settings_reset())
        self._settings_dirty = False

    def _settings_values(self) -> dict:
        return {"step": float(self.g_set_step.value), "shape": self.g_set_shape.value, "tilt": float(self.g_set_tilt.value),
                "roll": float(self.g_set_roll.value), "max_tilt": float(self.g_set_max.value)}

    def _settings_used_md(self) -> str:
        from .analyze import settings_of

        st = settings_of(self.result)
        r = self.reach
        lattice = " × ".join(str(n) for n in st.get("lattice", []))
        shape = st.get("grid_shape_text", st.get("grid_shape", "box"))
        counts = st.get("grid_shape_points") or {}
        alt = (" (box would be " + f"{counts['box']:,}" + ")") if counts.get("box") and st.get("grid_shape") != "box" else ""
        orient = f" × {r.n_orientations} orientations" if r.task != "position" else ""
        return (f"<small>used: {r.n_positions:,} points ({lattice} lattice, inside the {shape}{alt}){orient} "
                f"= {r.n_positions * r.n_orientations:,} targets in {_dur(r.seconds)}"
                + (f"<br>{r.orientations.describe()}" if r.task != "position" else "") + "</small>")

    def _settings_changed(self):
        if getattr(self, "_settings_quiet", False):
            return
        now, base = self._settings_values(), self._settings_base
        names = {"step": ("voxel size", "mm"), "shape": ("grid shape", ""), "tilt": ("orientation step", "°"),
                 "roll": ("roll step", "°"), "max_tilt": ("angles up to", "°")}
        changed = [k for k in base if (now[k] != base[k] if k == "shape" else abs(now[k] - base[k]) > 1e-9)]
        self._settings_dirty = bool(changed)
        self.g_set_btns.disabled = not changed
        self.g_set_reset.disabled = not changed
        self.g_set_btns.color = "orange" if changed else None
        if changed:
            was = " · ".join(f"{names[k][0]} was {base[k]:g}{names[k][1]}" if k != "shape" else f"grid shape was {base[k]}" for k in changed)
            est = self._estimate_targets(now)
            self.g_set_used.content = (f"<small><b>Edited</b> ({was}). The old map is hidden until you Compute workspace or Reset."
                                       + (f"<br>new: ≈ {est[0]:,} points × {est[1]} orientations ≈ {_dur(est[2])}" if est else "") + "</small>")
        else:
            self.g_set_used.content = self._settings_used_md()
        self._redraw_workspace()

    def _estimate_targets(self, v: dict):
        """(points, orientations, seconds) the edited settings would compute, from this map's own rate."""
        try:
            positions, _shape, _counts = self._grid_for(v)
            O = 1 if self.reach.task == "position" else len(self._orientations_for(v))
        except Exception:
            return None
        rate = self.reach.seconds / max(1, self.reach.n_positions * self.reach.n_orientations)
        return len(positions), O, rate * len(positions) * O

    def _grid_for(self, v: dict):
        pts = self.result.cloud_p[np.all(np.isfinite(self.result.cloud_p), axis=1)]
        shapes = ("box", "sphere", "cylinder", "hull") if v["shape"].startswith("auto") else (v["shape"],)
        return fit_grid(pts, max(v["step"], 0.5) / 1000.0, shapes=shapes)

    def _orientations_for(self, v: dict) -> OrientationSet:
        approach = tuple(self.reach.orientations.approach) if self.reach is not None else (0.0, 0.0, -1.0)
        n_rolls = max(1, int(round(360.0 / max(v["roll"], 1e-6))))
        return OrientationSet.sample(approach=approach, tilt_step_deg=v["tilt"], n_rolls=n_rolls, max_tilt_deg=v["max_tilt"])

    def _settings_reset(self):
        self._settings_quiet = True
        try:
            b = self._settings_base
            self.g_set_step.value, self.g_set_shape.value = b["step"], b["shape"]
            self.g_set_tilt.value, self.g_set_roll.value, self.g_set_max.value = b["tilt"], b["roll"], b["max_tilt"]
        finally:
            self._settings_quiet = False
        self._settings_changed()

    def _compute_workspace(self):
        """Recompute the reach map with the card's settings (in memory; the saved result is unchanged)."""
        from .analyze import settings_of
        from .joint_space import joint_ranges

        if self.arm is None or self.arm.ik is None:
            return
        v = self._settings_values()
        positions, shape, counts = self._grid_for(v)
        O = None if self.reach.task == "position" else self._orientations_for(v)
        n_targets = len(positions) * (1 if O is None else len(O))
        if n_targets > 4_000_000:
            self.g_set_used.content = f"<small><b>{n_targets:,} targets is too many</b> (limit 4,000,000): raise the voxel size or a step.</small>"
            return
        self.g_set_btns.disabled = True
        self.g_set_used.content = f"<small>computing {n_targets:,} targets…</small>"
        cfg = self.result.config
        rm = reach_map(self.arm, positions, O, floor_z=cfg.get("floor_z"), singular_tol=cfg.get("singular_tol", 0.0), w_ref=cfg.get("w_ref"))
        step = max(v["step"], 0.5) / 1000.0
        lattice = [int(len(np.unique(np.round(Box.around(self.result.cloud_p[np.all(np.isfinite(self.result.cloud_p), axis=1)]).grid(step)[:, a], 9))))
                   for a in range(3)]
        st = dict(settings_of(self.result))
        st.update({"step_m": step, "grid_shape": shape.kind, "grid_shape_auto": v["shape"].startswith("auto"),
                   "grid_shape_text": shape.describe(), "grid_shape_points": counts, "lattice": lattice, "grid_points": len(positions),
                   "orientation": None if O is None else O.resolution()})
        cfg.update({"step": step, "settings": st, "grid_shape": shape.to_json(), "orientations": rm.n_orientations})
        self.result.reach = rm
        self.result.ranges = joint_ranges(self.arm, rm)
        self.reach = rm
        self.step = step
        self.layer_reach = None
        self._filter_cache = {}
        self._settings_base = {**v, "step": round(step * 1000, 1)}
        self._settings_changed()
        self.g_set_used.content = self._settings_used_md() + "<small> · recomputed here, not saved</small>"
        self.g_header.content = self._header_md()
        self._refresh_ranges_figure()
        if self.sel_point is not None:
            self._clear_selection()

    # ------------------------------------------------------------------ workspace: colour, legend list, view
    def _build_view_controls(self, g):
        oriented = self.reach.task != "position"
        colors = list(COLOR_BY) if oriented else ["workspace", "IK solutions", "consistency"]
        self.g_color = g.add_dropdown("Colour by", tuple(colors), initial_value=colors[0], order=10,
                                      hint="workspace: the best level each point reaches · consistency: did FK of each IK answer land back on the target")
        self.g_cmap = g.add_dropdown("Colours", tuple(COLORMAPS), initial_value=DEFAULT_COLORMAP, order=11,
                                     hint="Perceptually uniform maps (matplotlib's); consistency is always a traffic light")
        self.g_reverse = g.add_checkbox("Reverse (high = dark)", initial_value=False, order=12)
        self._hidden = {"0"}  # the empty bin starts hidden: the map shows where something works
        self._legend_rows: list[tuple[str, object]] = []
        self._legend_mode = None
        self.g_legend_note = g.add_markdown("", order=50)
        # WHY "Cube size": "Voxel size" now names the grid step in the settings card above
        self.g_voxel = g.add_slider("Cube size (drawn)", min=0.1, max=1.0, step=0.05, initial_value=0.9 if self.planar else 0.55, order=61)
        self.g_cut = g.add_dropdown("Cut", ("none", "x", "y", "z"), initial_value="none", order=62)
        self.g_cut_at = g.add_slider("Cut at (m)", min=-1.0, max=1.0, step=0.005, initial_value=0.0, disabled=True, order=63)
        self.g_blob = g.add_checkbox("Blob (surface)", initial_value=False, order=64)
        for h in (self.g_voxel, self.g_blob, self.g_cut_at, self.g_reverse):
            h.on_update(lambda _e: self._redraw_workspace())
        self.g_cmap.on_update(lambda _e: self._redraw_workspace())
        self.g_color.on_update(lambda _e: self._color_changed())
        self.g_cut.on_update(lambda _e: self._cut_changed())

    def _color_changed(self):
        """The legend list follows what is coloured (Zach, 2026-11-23: "The show should also update to match
        what you are coloring by"); its empty bin starts hidden again."""
        self._hidden = {"0"}
        heat = color_mode(self.reach, self.g_color.value) != "consistency"
        self.g_cmap.disabled = not heat
        self.g_reverse.disabled = not heat
        self._redraw_workspace()

    def _sync_legend(self, rows: list[dict], mode: str, total: int):
        """One checkbox per legend bin: label + count, swatch and count bar drawn by DECORATE_JS."""
        labels = [f"{r['label']} · {r['count']:,}" for r in rows]
        keys = [r["key"] for r in rows]
        if self._legend_mode != mode or [k for k, _ in self._legend_rows] != keys:
            for _k, h in self._legend_rows:
                h.remove()
            self._legend_rows = []
            with self.g_ws_folder:
                for i, (row, label) in enumerate(zip(rows, labels)):
                    h = self.server.gui.add_checkbox(label, initial_value=row["key"] not in self._hidden, order=20 + i * 0.01,
                                                     hint=row["meaning"])
                    h.on_update(lambda e, key=row["key"]: self._legend_toggled(key, e.target.value))
                    self._legend_rows.append((row["key"], h))
            self._legend_mode = mode
        else:
            self._legend_quiet = True  # WHY: keep the ticks true to `_hidden` without re-entering the toggle
            try:
                for (key, h), label in zip(self._legend_rows, labels):
                    if h.label != label:
                        h.label = label
                    if bool(h.value) != (key not in self._hidden):
                        h.value = key not in self._hidden
            finally:
                self._legend_quiet = False
        top = max([r["count"] for r in rows] + [1])
        legend = {label: {"rgb": [int(c) for c in r["rgb"]], "frac": r["count"] / top} for label, r in zip(labels, rows)}
        try:
            import json

            from viser_3d_viewer.shim import send_replaceable_javascript

            send_replaceable_javascript(self.server, f"window.__rrLegend = {json.dumps(legend)}; window.__rrApply && window.__rrApply();",
                                        key="robot-reach-legend")
        except Exception:  # plain viser: the counts in the labels still say it
            pass

    def _legend_toggled(self, key: str, on: bool):
        if getattr(self, "_legend_quiet", False):
            return
        (self._hidden.discard if on else self._hidden.add)(key)
        self._redraw_workspace()

    # ------------------------------------------------------------------ IK configurations (A1)
    def _build_config_controls(self, g):
        """One row per question, toggles (Zach's pick A1, 2026-11-23); filters the solution stepper, the map
        and the joint-range histograms. Tags come from each solution's posture (`configurations`)."""
        self._filter_cache = {}
        r = self.reach
        cfgs = r.configurations
        self._cfg_sel = None
        if cfgs is None or r.sol_config is None or not cfgs.questions:
            return
        self._cfg_sel = [set(range(len(q.options))) for q in cfgs.questions]
        with self._folder("IK configurations"):
            g.add_markdown("<small>Which IK solutions count. Each question keeps at least one answer; "
                           "a solution is tagged from its own posture, never from the solver's slot.</small>")
            self.g_cfg_rows = []
            for i, question in enumerate(cfgs.questions):
                h = g.add_button_group(cfgs.label(question), chip_labels(question.options, self._cfg_sel[i]), hint=question.rule)
                h.on_click(lambda e, i=i: self._cfg_clicked(i, e.target.value))
                self.g_cfg_rows.append(h)
            self.g_cfg_md = g.add_markdown(self._cfg_summary())

    def _cfg_clicked(self, i: int, label: str):
        question = self.reach.configurations.questions[i]
        option = question.options.index(label[len(CHIP_ON):] if label.startswith((CHIP_ON, CHIP_OFF)) else label)
        self._cfg_sel[i] = toggle_choice(self._cfg_sel[i], option, len(question.options))
        self.g_cfg_rows[i].options = chip_labels(question.options, self._cfg_sel[i])
        self.g_cfg_md.content = self._cfg_summary()
        self._refresh_ranges_figure()
        self._redraw_workspace()
        if self.sel_point is not None:
            self._select(self.sel_point)

    def _allowed(self):
        cfgs = self.reach.configurations if self.reach is not None else None
        if not getattr(self, "_cfg_sel", None) or cfgs is None:
            return None
        allowed = cfgs.allowed_from(self._cfg_sel)
        return None if allowed.all() else allowed

    def _filtered(self, reach):
        """`reach` with only the allowed IK configurations (cached per layer and choice)."""
        allowed = self._allowed()
        if reach is None or allowed is None or reach.sol_config is None:
            return reach
        key = (id(reach), tuple(bool(v) for v in allowed))
        if key not in self._filter_cache:
            self._filter_cache = {k: v for k, v in self._filter_cache.items() if k[0] == id(reach)}
            self._filter_cache[key] = reach.filtered(allowed)
        return self._filter_cache[key]

    def _cfg_summary(self) -> str:
        cfgs = self.reach.configurations
        allowed = cfgs.allowed_from(self._cfg_sel)
        r = self._filtered(self.reach)
        lv = r.levels()
        kinds = (3, 4, 5, 6) if r.task != "position" else (3,)
        counts = " · ".join(f"{LEVEL_NAMES[k]} {int((lv >= k).sum()):,}" for k in kinds)
        return (f"<small><b>{int(allowed.sum())} of {cfgs.n_configurations} IK configurations allowed</b> → {counts} points; "
                f"{len(r.sol_index):,} working IK solutions</small>")

    def _refresh_ranges_figure(self):
        """Joint-range histograms of the working solutions the current filter allows."""
        if "joint ranges" not in getattr(self, "_figures", {}) or self.reach is None:
            return
        try:
            from .plots import figure_rgb, ranges_figure
        except ImportError:
            return
        r = self._filtered(self.reach)
        info = self.result.arm_info
        limits = np.stack([info["lower"], info["upper"]], axis=1).astype(float)
        ranges = ranges_of(info["joint_names"], limits, r.sol_q.astype(float))
        note = "" if r is self.reach else f"filtered: {self._cfg_summary_plain()}"
        self._figures["joint ranges"] = figure_rgb(ranges_figure(ranges, info["name"], note=note))
        if self.g_fig.value == "joint ranges":
            self._picture_changed()

    def _cfg_summary_plain(self) -> str:
        cfgs = self.reach.configurations
        return " · ".join(f"{q.name} {'/'.join(q.options[i] for i in sorted(sel))}" for q, sel in zip(cfgs.questions, self._cfg_sel))

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
            from .analyze import settings_of

            shape = settings_of(self.result).get("grid_shape_text", "box")
            grid = (f" The voxel grid is fitted around it ({shape}, one voxel of padding): {self.reach.n_positions:,} points, "
                    f"step {self.step * 1000:.0f} mm. The voxels themselves come from IK at each grid point, not from these dots.")
        text = (f"FK cloud: {how}, each pushed through FK; a dot is where the tool lands.{grid} "
                f"Dot colour = that joint posture's own check (limits, what the FK reports, torque, singular):")
        if plain:
            return text.rstrip(":") + "."
        return f"<small>{text} " + self._code_legend(np.unique(res.cloud_code))[len("<small>"):]

    def _resolution_md(self, reach) -> str:
        """The angular resolution behind "N / M orientations" (Zach, 2026-11-23)."""
        if reach.task == "position":
            return "<small>Position-only arm: one target per point, no orientations.</small>"
        return f"<small>Orientations tried here: {reach.orientations.describe()}.</small>"

    def _ask_resolution(self):
        """C2: the angle set and box frame as numbers, with the Python line that runs."""
        O = self._orientations()
        c = np.asarray(self.g_box_center.value, float)
        size = np.asarray(self.g_box_size.value, float)
        rpy = np.asarray(self.g_box_rpy.value, float)
        try:
            n_pts = len(box_positions(c, size, rpy, float(self.g_box_step.value)))
        except Exception:
            n_pts = 0
        rate = self.reach.seconds / max(1, self.reach.n_positions * self.reach.n_orientations)
        n_o = 1 if O is None else len(O)
        fmt = lambda v: ", ".join(f"{float(x) + 0.0:.3g}" for x in np.round(np.asarray(v, float), 4))  # noqa: E731  (+0.0: no "-0")
        if O is None:
            what = "<small>Position only: no orientations.</small>"
            code = f"box_positions(center=({fmt(c)}), size=({fmt(size)}), rpy_deg=({fmt(rpy)}), step={float(self.g_box_step.value):g})"
        else:
            r = O.resolution()
            what = (f"<small><b>{r['n_directions']} directions × {r['n_rolls']} rolls = {r['n_orientations']} orientations</b> per point · "
                    f"box of {n_pts:,} points ≈ {_dur(rate * n_pts * n_o)} · approach = the box's −Z "
                    f"({fmt(O.approach)}): straight down when the box is level<br>{O.describe()}</small>")
            code = (f"box_positions(center=({fmt(c)}), size=({fmt(size)}), rpy_deg=({fmt(rpy)}), step={float(self.g_box_step.value):g})\n"
                    f"OrientationSet.sample(approach=({fmt(O.approach)}), tilt_step_deg={float(self.g_ask_tilt.value):g}, "
                    f"n_rolls={r['n_rolls']}, max_tilt_deg={float(self.g_ask_max.value):g})")
        self.g_ask_res_md.content = what
        self.g_ask_code.content = f"```python\n{code}\n```"

    def _build_model_controls(self, g):
        """Robot model: which meshes to draw (visual / collision / both / none), opacity, and what the
        collision check really uses."""
        if self.arm is None:
            return
        with self._folder("Robot model", expand_by_default=bool(self.mesh_handles)):
            shows = model_shows([self.arm.meshes[i] for i, _ in self.mesh_handles])
            self.g_skeleton = None
            if self.mesh_handles:
                self.g_model = g.add_dropdown("Show mesh", shows, initial_value=shows[0])
                self.g_opacity = g.add_slider("Opacity", min=0.1, max=1.0, step=0.05, initial_value=1.0)
                self.g_skeleton = g.add_checkbox("Show skeleton", initial_value=True,
                                                 hint="The chain of joint points drawn over the meshes")
                self.g_model.on_update(lambda _e: self._model_changed())
                self.g_opacity.on_update(lambda _e: self._pose_meshes())
                self.g_skeleton.on_update(lambda _e: self._model_changed())
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
            self._draw_arm(self.arm_q, code=Code.ENVIRONMENT_COLLISION if self.contact else None)

    # ------------------------------------------------------------------ joint-space pictures
    def _picture_changed(self):
        name = self.g_fig.value
        if getattr(self, "_img_shown", None) is not self._figures[name]:  # WHY: re-sending the same image revokes its blob mid-load
            self.g_img.image = self._figures[name]
            self._img_shown = self._figures[name]
        jm = self._maps_by_name.get(name)
        info = self.result.arm_info
        limits = np.stack([info["lower"], info["upper"]], axis=1).astype(float)
        if jm is None:
            text = ("One histogram per joint: bar height = the share of the working IK solutions at that angle (10° bins); "
                    "the grey band is the joint's limits. Follows the IK configurations filter.") if name == "joint ranges" else ""
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
            parts = [f"{LEVEL_NAMES[k]} {int((lv >= k).sum()):,}" for k in ((3, 4, 5, 6) if r.task != "position" else (3,))]
            lines.append(f"{r.n_positions:,} grid points × {r.n_orientations} orientations × up to {r.n_branches} IK solutions → "
                         f"{len(r.sol_index):,} working IK solutions in {_dur(r.seconds)}")
            lines.append("points: " + " · ".join(parts))
        else:
            lines.append("Workspace is a curve: shown as the FK cloud; IK grid skipped.")
        return "\n\n".join(lines)

    # ================================================================== workspace layer
    def _current_layer(self):
        """The map on screen: the full map or an Ask-here layer, filtered to the allowed IK configurations."""
        return self._filtered(self.layer_reach if self.layer_reach is not None else self.reach)

    def _redraw_workspace(self):
        reach = self._current_layer()
        if reach is None:
            self.g_legend.content = self._code_legend(np.unique(self.result.cloud_code))
            return
        cut = None
        if self.g_cut.value != "none":
            axis = "xyz".index(self.g_cut.value)
            # WHY keep the far half: the cut face then faces the camera, so you look INTO the map
            cut = (axis, float(self.g_cut_at.value), bool(self.home_direction[axis] >= 0))
        # WHY hidden while edited: an edited settings card no longer describes the map on screen
        stale = getattr(self, "_settings_dirty", False) and self.layer_reach is None
        settings = dict(visible=not stale, kind="all", color_by=self.g_color.value, voxel_scale=float(self.g_voxel.value), blob=self.g_blob.value,
                        cut=cut, dim=self.sel_point is not None or self.arm_q is not None, colormap=self.g_cmap.value,
                        reverse=bool(self.g_reverse.value), hidden=set(self._hidden))
        with self.lock:
            # the overlay is the same seam a host scene uses (overlay.add_reach_map); one per layer
            if self.overlay is None or self.overlay.reach is not reach:
                if self.overlay is not None:
                    self.overlay.remove()
                self.overlay = ReachOverlay(self.server, reach, parent_frame="/workspace", step=self._layer_step(),
                                            on_select=self._select, **{k: v for k, v in settings.items() if k != "visible"})
                if stale:
                    self.overlay.visible = False
            else:
                self.overlay.update(**settings)
        self.visible_index = self.overlay.shown_index if not stale else np.zeros(0, int)
        mode = color_mode(reach, self.g_color.value)
        rows = [r for r in self.overlay.bins if r["count"] > 0 or r["key"] == "0"]
        self._sync_legend(rows, mode, reach.n_positions)
        note = ""
        if mode == "workspace" and reach.task != "position":
            note = "<br>" + "<br>".join(f"<b>{LEVEL_NAMES[k]}</b>: {LEVEL_MEANINGS[k]}" for k in (3, 4, 5, 6)) + \
                   "<br>Nested: each includes the next; a point is counted at the highest it reaches."
        elif mode == "fraction of orientations":
            note = f"<br>Colour = the share of the {reach.n_orientations} orientations reachable there."
        self.g_legend_note.content = f"<small>{note[4:]}</small>" if note else ""
        filt = "" if self._allowed() is None else " · IK configurations filtered"
        shown = "map hidden: settings edited" if stale else f"{len(self.visible_index):,} of {reach.n_positions:,} grid points shown"
        blob = self.overlay.legend.split("<br>", 1)[1] if "<br>blob" in self.overlay.legend else ""
        self.g_legend.content = f"<small>{shown}{filt}" + (f" · {blob}" if blob else "") + "</small>"

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
            send_javascript(self.server, DECORATE_JS)
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
        configs = reach.solution_configs(p)
        configs = [None] * len(sols) if configs is None else [int(c) for c in configs]
        tilt = reach.orientations.tilt
        order = sorted(range(len(sols)), key=lambda i: (float(tilt[sols[i][0]]), sols[i][0], sols[i][1]))  # straight down first
        self.sel_solutions = [sols[i] for i in order]
        self.sel_configs = [configs[i] for i in order]
        sols = self.sel_solutions
        lv = int(reach.levels()[p])
        x, y, z = reach.positions[p]
        self.g_sel_md.content = (
            f"**({x:+.3f}, {y:+.3f}, {z:+.3f}) m** · {LEVEL_NAMES.get(lv, lv)}\n\n"
            f"{len(sols)} working IK solutions (over every orientation) · "
            f"{int(reach.valid_mask[p].sum())}/{reach.n_orientations} orientations"
            + ("" if self._allowed() is None else " · IK configurations filtered")
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
        orient = "" if reach.task == "position" else f"orientation #{o + 1} of {reach.n_orientations}: tilt {tilt:.0f}° from approach, roll {roll:.0f}°"
        cfg = self.sel_configs[k] if k < len(getattr(self, "sel_configs", [])) else None
        name = f"IK configuration: {reach.configurations.name(cfg)}" if cfg is not None and reach.configurations is not None else f"IK solution slot {b + 1}"
        self.g_sol_md.content = f"**{k + 1} / {len(self.sel_solutions)}** · {orient}\n\n{name}\n\n{_deg_labelled(q)}"
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
        if is_collision(code):
            parts = sorted(self.contact)
            touching = "\n\n<small>in contact (red): " + (", ".join(parts) if parts else "the red skeleton points") + "</small>"
        self.g_jog_md.content = (
            f"<span style='color:rgb{CODE_COLORS[Code(code)]}'>●</span> **{CODE_LABELS[Code(code)]}**"
            + (f" · tool ({tip[0]:+.3f}, {tip[1]:+.3f}, {tip[2]:+.3f}) m" if ok[0] else "")
            + tau_txt + touching
        )

    # ================================================================== ask here
    def _build_tools(self):
        """Ask here (Zach's pick C2, 2026-11-23): the box moves AND tilts in the 3-D view with viser's
        transform gizmo ("check out viser it already has like the built in thing ... you can drag XYZ but
        also tilt!!"); the same frame and the angle set are typed fields, with the Python line that runs."""
        g, s = self.server.gui, self.server.scene
        size0 = max(self.step * 4, self.extent * 0.25)
        start = self.center + np.array([self.extent * 0.25, 0.0, 0.0])
        if self.planar:
            start[1] = 0.0
        res = self.reach.orientations.resolution() if self.reach.task != "position" else {}
        with self._folder("Ask here", expand_by_default=True):
            g.add_markdown("<small>Drag the gizmo's arrows to move the box and its rings to tilt it. "
                           "**Compute box** samples it finer; **Drop POI** checks every orientation at its centre.</small>")
            self.g_box_on = g.add_checkbox("Show box", initial_value=False)
            self.g_box_center = g.add_vector3("Box centre (m)", initial_value=tuple(float(v) for v in start), step=0.01)
            self.g_box_rpy = g.add_vector3("Roll / pitch / yaw (°)", initial_value=(0.0, 0.0, 0.0), step=1.0)
            self.g_box_size = g.add_vector3("Box size (m)", initial_value=tuple(float(size0) for _ in range(3)) if not self.planar else (float(size0), 0.0, float(size0)), step=0.01)
            self.g_box_step = g.add_number("Box step (m)", initial_value=round(max(self.step / 2, 0.005), 3), step=0.005, min=0.002)
            oriented = self.reach.task != "position"
            self.g_ask_max = g.add_number("Angles up to (°)", initial_value=float(res.get("tilt_max_deg", 180.0)), min=0.0, max=180.0, step=5.0,
                                          visible=oriented, hint="Tilt from the approach (the box's −Z): 0 = straight down only")
            self.g_ask_tilt = g.add_number("Tilt step (°)", initial_value=float(res.get("tilt_step_deg") or 45.0), min=5.0, max=180.0, step=5.0,
                                           visible=oriented)
            self.g_ask_roll = g.add_number("Roll step (°)", initial_value=float(res.get("roll_step_deg") or 60.0), min=5.0, max=360.0, step=5.0,
                                           visible=oriented)
            self.g_ask_res_md = g.add_markdown("")
            self.g_ask_code = g.add_markdown("")
            btns = g.add_button_group("Run", ("Compute box", "Drop POI", "Back to full map"))
            self.g_tool_md = g.add_markdown("")
        self.gizmo = s.add_transform_controls("/ask/gizmo", scale=self.extent * 0.25, position=tuple(start), visible=False)
        self.box_handle = s.add_box("/ask/gizmo/box", color=(245, 159, 0), dimensions=tuple(self.g_box_size.value), wireframe=True, visible=False)
        self.box_up = s.add_line_segments("/ask/gizmo/up", np.array([[[0, 0, 0], [0, 0, 1]]], np.float32) * float(size0) * 0.8,
                                          np.array((34, 139, 230), np.uint8), thickness=self.extent * 0.004, visible=False)
        self._box_quiet = False
        self.gizmo.on_update(lambda _e: self._gizmo_moved())
        for h in (self.g_box_center, self.g_box_rpy):
            h.on_update(lambda _e: self._box_fields_changed())
        for h in (self.g_box_step, self.g_ask_max, self.g_ask_tilt, self.g_ask_roll):
            h.on_update(lambda _e: self._ask_resolution())
        self.g_box_on.on_update(lambda _e: self._toggle_box())
        self.g_box_size.on_update(lambda _e: self._box_size_changed())
        btns.on_click(lambda e: self._run_tool(e.target.value))
        self._ask_resolution()

    def _box_size_changed(self):
        size = tuple(max(1e-4, v) for v in self.g_box_size.value)
        self.box_handle.dimensions = size
        self._ask_resolution()

    def _gizmo_moved(self):
        """The gizmo was dragged: write its frame into the fields (without echoing back)."""
        if self._box_quiet:
            return
        self._box_quiet = True
        try:
            self.g_box_center.value = tuple(round(float(v), 4) for v in self.gizmo.position)
            self.g_box_rpy.value = tuple(round(float(v), 2) for v in matrix_rpy(wxyz_matrix(self.gizmo.wxyz)))
        finally:
            self._box_quiet = False
        self._ask_resolution()

    def _box_fields_changed(self):
        """A frame was typed: move the gizmo (and the box under it) there."""
        if self._box_quiet:
            return
        self._box_quiet = True
        try:
            self.gizmo.position = tuple(float(v) for v in self.g_box_center.value)
            self.gizmo.wxyz = tuple(float(v) for v in _wxyz(rpy_matrix(self.g_box_rpy.value)))
        finally:
            self._box_quiet = False
        self._ask_resolution()

    def _toggle_box(self):
        on = self.g_box_on.value
        self.gizmo.visible = on
        self.box_handle.visible = on
        self.box_up.visible = on

    def _orientations(self):
        if self.reach.task == "position":
            return None
        return box_orientations(rpy_matrix(self.g_box_rpy.value), float(self.g_ask_max.value), float(self.g_ask_tilt.value),
                                float(self.g_ask_roll.value))

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
        c = np.asarray(self.g_box_center.value, float)
        cfg = self.result.config
        t0 = time.perf_counter()
        if which == "Compute box":
            size = np.maximum(np.asarray(self.g_box_size.value, float), 0.0)
            step = float(self.g_box_step.value)
            positions = box_positions(c, size, self.g_box_rpy.value, step)
            if len(positions) > 60_000:
                self.g_tool_md.content = f"**{len(positions):,} points is too many** — raise the step."
                return
            self.g_tool_md.content = f"computing {len(positions):,} points…"
            rm = reach_map(self.arm, positions, self._orientations(), floor_z=cfg.get("floor_z"), singular_tol=cfg.get("singular_tol", 0.0), w_ref=cfg.get("w_ref"))
            rm.meta["step"] = step
            self.layer_reach = rm
            lv = self._filtered(rm).levels()
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
            lv = int(self._filtered(rm).levels()[0])
            if rm.task != "position":
                # a hedgehog: one spike per sampled approach direction, green where some roll works
                dirs = rm.orientations.R[:, :, 2]
                ok = self._filtered(rm).valid_mask[0]
                L = self.extent * 0.12
                segs = np.stack([np.broadcast_to(c, dirs.shape), c - dirs * L], axis=1)
                cols = np.where(ok[:, None], np.array(CODE_COLORS[Code.OK]), np.array((200, 200, 200)))
                cols = np.repeat(cols[:, None, :], 2, axis=1)  # viser wants a colour per segment end
                self.server.scene.add_line_segments("/ask/poi_axes", segs.astype(np.float32), cols.astype(np.uint8), thickness=self.extent * 0.004)
            self.g_tool_md.content = (
                f"**POI ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f})** · {LEVEL_NAMES.get(lv, lv)}\n\n"
                f"{int(self._filtered(rm).valid_mask[0].sum())}/{rm.n_orientations} orientations · {len(self._filtered(rm).sol_index)} IK solutions"
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
