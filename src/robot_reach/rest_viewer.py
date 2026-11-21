"""The rest-pose explorer: the arm in 3D over its scene, and beside it the three numbers that decide a
rest pose — footprint, clearance over the belt, holding power — live as you pick a pose or jog.

    python -m robot_reach rest shoulder3                    # the template: shoulder3 beside a fake belt
    python -m robot_reach rest /abs/my_arm.py:study --poses arm.named_poses.json --candidates results.json

Calm UI (Zach's dashboards rule): the three headline numbers first, everything else folded; red only
for a problem (clearance under the scene's threshold, a pose that fails a check).

House style (viser-3d-viewer): loopback, no logo, Fusion ViewCube, panel docked right, READY line.
"""

from __future__ import annotations

import argparse
import html
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .codes import CODE_COLORS, Code
from .rest import PoseChoice, RestReport, RestStudy, candidate_poses, named_poses, rank

INK = (40, 44, 52)
MUTED = (120, 128, 140)
BELT = (196, 204, 214)
LINE_RED = (214, 40, 40)
PROBLEM = (214, 40, 40)
FINE = (46, 140, 84)


def _wxyz(R: np.ndarray) -> np.ndarray:
    from robot_msgs.geometry.rotations import quat_from_matrix, xyzw_to_wxyz

    return xyzw_to_wxyz(quat_from_matrix(np.asarray(R, float)))


def _mm(x) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{1e3 * x:.0f}"


def _range_mm(iv) -> str:
    return f"{1e3 * iv[0]:+.0f} … {1e3 * iv[1]:+.0f} mm"


class RestViewer:
    ROOTS = ("/scene", "/arm", "/marks")

    def __init__(self, study: RestStudy, *, host: str = "127.0.0.1", port: int = 0, out: Path | None = None, title: str | None = None):
        self.study = study
        self.arm, self.scene = study.arm, study.scene
        self.out = Path(out) if out is not None else Path.cwd() / "rest-cards"
        self.lock = threading.Lock()
        # WHY one lock around every use of the arm: an FK backend may keep state (bam_fb_v2's MuJoCo data); the
        # ranking thread and the jog worker measuring at once read each other's pose (seen: a READY "over torque limit")
        self.arm_lock = threading.RLock()
        self._pending: np.ndarray | None = None
        self._worker: threading.Thread | None = None
        self._muted = False
        self.report: RestReport | None = None
        self.pose_name = ""
        self.q = study.poses[0].q.copy() if study.poses else study.home.copy()
        # the scene's extent for the camera and the drawing: the arm at its first pose plus the region
        with self.arm_lock:
            pts = self.arm.surface(self.q[None], max_per_mesh=50)[0]
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        abh = self.scene.coords(pts) if len(pts) else np.zeros((1, 3))
        a_lo, a_hi = float(abh[:, 0].min()), float(abh[:, 0].max())
        r_a = self.scene.region_across or (a_lo - 0.1, a_hi + 0.1)
        r_b = self.scene.region_along or (float(abh[:, 1].min()) - 0.35, float(abh[:, 1].max()) + 0.35)
        self.span_a = (min(r_a[0], a_lo), max(r_a[1], a_hi))
        self.span_b = (float(r_b[0]), float(r_b[1]))
        self.region_a, self.region_b = tuple(map(float, r_a)), self.span_b
        self.top_h = float(abh[:, 2].max())
        self.extent = max(self.span_a[1] - self.span_a[0], 0.3)
        lo = self.scene.to_world(np.array([self.span_a[0], abh[:, 1].min(), self.scene.plane_height(0.0, 0.0)]))
        hi = self.scene.to_world(np.array([self.span_a[1], abh[:, 1].max(), self.top_h]))
        center = (lo + hi) / 2
        direction = tuple(float(v) for v in (-0.55 * self.scene.u - 1.0 * self.scene.v + 0.45 * self.scene.up))
        try:
            from viser_3d_viewer import Viewer, ViewerConfig

            cfg = ViewerConfig(host=host, port=port, title=title or ("Rest · " + (study.name or self.arm.name)), grid=False,
                               home_target=tuple(map(float, center)), home_distance=max(0.8, 1.9 * self.extent),
                               home_direction=direction, dark_mode=False)
            self.viewer = Viewer(cfg)
            self.server = self.viewer.server
            self.viewer.camera.fit_home(tuple(np.minimum(lo, hi)), tuple(np.maximum(lo, hi)), direction=direction, margin=1.15)
        except ImportError:  # plain viser still works, without the house chrome
            import viser

            self.viewer = None
            self.server = viser.ViserServer(host=host, port=port)
        self.url = f"http://{host}:{self.server.get_port()}/"
        self._build_scene()
        self._build_arm()
        self._build_gui()
        self._apply(self.q, name=study.poses[0].label if study.poses else "home")

    # ================================================================== scene
    def _belt_frame(self):
        """The support plane's rotation (x across, z its normal) and the point under (a, b)."""
        e1, e2 = self.scene.plane_basis()
        return np.column_stack([e1, e2, self.scene.normal])

    def _on_plane(self, a, b, lift: float = 0.0) -> np.ndarray:
        a, b = np.asarray(a, float), np.asarray(b, float)
        abh = np.stack([a, b, self.scene.plane_height(a, b) + lift], -1)
        return self.scene.to_world(abh)

    def _build_scene(self):
        s, sc = self.server.scene, self.scene
        R = self._belt_frame()
        (a0, a1), (b0, b1) = self.region_a, self.region_b
        centre = self._on_plane((a0 + a1) / 2, (b0 + b1) / 2, -0.003)
        s.add_box("/scene/plane", color=BELT, dimensions=(a1 - a0, b1 - b0, 0.006), wxyz=tuple(_wxyz(R)), position=tuple(centre), opacity=0.85)
        # region edges and the reference line on the plane, the line's vertical up to the arm
        bb = np.array([b0, b1])
        edges = np.stack([self._on_plane([a0, a0], bb, 0.001), self._on_plane([a1, a1], bb, 0.001)])
        s.add_line_segments("/scene/edges", edges.astype(np.float32), np.array(MUTED, np.uint8), thickness=0.003)
        line = np.stack([self._on_plane([0.0, 0.0], bb, 0.0015)])
        top = sc.to_world(np.array([0.0, 0.0, self.top_h]))
        foot = self._on_plane(0.0, 0.0, 0.0)
        vertical = np.stack([foot, top])[None]
        s.add_line_segments("/scene/line", np.concatenate([line, vertical]).astype(np.float32), np.array(LINE_RED, np.uint8), thickness=0.003)
        for k, ko in enumerate(sc.keepouts):
            ka = ko.across or self.span_a
            kb = ko.along or (b0, b1)
            base = float(np.min(sc.plane_height(np.array(ka), np.array([kb[0], kb[0]]))))
            h0, h1 = min(base, ko.top), max(base, ko.top)
            c = sc.to_world(np.array([(ka[0] + ka[1]) / 2, (kb[0] + kb[1]) / 2, (h0 + h1) / 2]))
            Rk = np.column_stack([sc.u, sc.v, sc.up])
            s.add_box(f"/scene/keepout_{k}", color=(150, 156, 166), dimensions=(ka[1] - ka[0], kb[1] - kb[0], max(h1 - h0, 0.004)),
                      wxyz=tuple(_wxyz(Rk)), position=tuple(c), opacity=0.55)
        self.h_lane = None
        self.h_hull = None
        self.h_low = None
        self.h_low_line = None

    def _build_arm(self):
        s = self.server.scene
        self.h_mesh = []
        drawn = [i for i, m in enumerate(self.arm.meshes) if m.draw and m.faces is not None]
        if drawn and self.arm.mesh_poses is not None:
            for i in drawn:
                m = self.arm.meshes[i]
                h = s.add_mesh_simple(f"/arm/mesh_{i}", np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.uint32),
                                      color=m.color or (170, 176, 186), flat_shading=False)
                self.h_mesh.append((i, h))
        self.h_links = None

    def _draw_arm(self, q: np.ndarray, ok: bool):
        s = self.server.scene
        if self.h_mesh:
            T = np.asarray(self.arm.mesh_poses(q[None]), float)[0]
            if not np.all(np.isfinite(T[[i for i, _ in self.h_mesh]])):
                return  # keep the last good drawing; the panel says why
            with self.server.atomic():
                for i, h in self.h_mesh:
                    h.position = tuple(map(float, T[i, :3, 3]))
                    h.wxyz = tuple(map(float, _wxyz(T[i, :3, :3])))
            return
        chain = self.arm.chain(q[None])[0]
        segs = np.stack([chain[:-1], chain[1:]], axis=1)
        segs = segs[np.all(np.isfinite(segs.reshape(len(segs), -1)), axis=1)]
        if not len(segs):
            return
        thick = max(0.008, self.extent * 0.02)
        self.h_links = s.add_line_segments("/arm/links", segs.astype(np.float32), np.array(INK if ok else PROBLEM, np.uint8), thickness=thick)
        joints = chain[np.all(np.isfinite(chain), axis=1)]
        s.add_point_cloud("/arm/joints", joints.astype(np.float32), np.array((90, 96, 108), np.uint8), point_size=thick * 1.3, point_shape="circle")

    def _draw_marks(self, r: RestReport):
        s, sc = self.server.scene, self.scene
        f, c = r.footprint, r.clearance
        if f is None:
            return
        # the arm's footprint on the plane (its hull) and the lane's edges toward the neighbours
        hull = f.hull
        if len(hull) >= 3:
            loop = np.stack([hull, np.roll(hull, -1, axis=0)], 1)
            pts = self._on_plane(loop[..., 0], loop[..., 1], 0.002)
            s.add_line_segments("/marks/hull", pts.astype(np.float32), np.array((70, 110, 170), np.uint8), thickness=0.0025)
        lanes = []
        for axis, (lo, hi) in f.lane.items():
            for v in (lo, hi):
                if axis == "along":
                    lanes.append(self._on_plane([self.region_a[0], self.region_a[1]], [v, v], 0.002))
                else:
                    lanes.append(self._on_plane([v, v], [self.region_b[0], self.region_b[1]], 0.002))
        if lanes:
            s.add_line_segments("/marks/lane", np.stack(lanes).astype(np.float32), np.array((70, 110, 170), np.uint8), thickness=0.0015)
        if c is not None and c.lowest is not None:
            bad = c.plane is not None and c.plane < c.threshold
            col = PROBLEM if bad else FINE
            low = np.asarray(c.lowest, float)
            d = sc.above_plane(low)
            foot = low - d * sc.normal
            s.add_icosphere("/marks/lowest", radius=0.006, color=col, position=tuple(map(float, low)))
            s.add_line_segments("/marks/clearance", np.stack([low, foot])[None].astype(np.float32), np.array(col, np.uint8), thickness=0.003)
            s.add_label("/marks/clearance_label", f"{_mm(c.plane)} mm", position=tuple(map(float, (low + foot) / 2)))

    # ================================================================== GUI
    def _build_gui(self):
        g = self.server.gui
        st = self.study
        g.add_markdown(f"**{html.escape(st.name or self.arm.name)}**  \n<small>{html.escape(self.arm.description)}</small>", order=-100)
        labels = [p.label for p in st.poses] or ["home"]
        self.g_pose = g.add_dropdown("Pose", tuple(labels), initial_value=labels[0], order=-90)
        self.g_pose_md = g.add_markdown("", order=-89)
        self.g_head = g.add_html("", order=-80)
        self.g_pose.on_update(lambda _e: self._pose_selected())

        with g.add_folder("Detail", expand_by_default=False, order=-70):
            self.g_detail = g.add_markdown("")
        with g.add_folder("Jog", expand_by_default=True, order=-60):
            roles = self.arm.meta.get("roles") or {}
            lim = np.rad2deg(self.arm.limits)
            self.g_jog = []
            for name, (lo, hi), v in zip(self.arm.joint_names, lim, np.rad2deg(self.q)):
                label = f"{roles[name]} · {name} (deg)" if name in roles else f"{name} (deg)"
                h = g.add_slider(label, min=float(np.floor(lo * 10) / 10), max=float(np.ceil(hi * 10) / 10), step=0.1,
                                 initial_value=float(np.clip(v, lo, hi)))
                h.on_update(lambda _e: self._jogged())
                self.g_jog.append(h)
            self.g_jog_md = g.add_markdown("")
        if st.poses and (st.gates or st.criteria):
            with g.add_folder("Rank", expand_by_default=False, order=-50):
                self.g_rank = g.add_markdown("_ranking…_")
            threading.Thread(target=self._rank, daemon=True).start()
        with g.add_folder("Export", expand_by_default=False, order=-40):
            g.add_markdown(f"<small>One PNG + metrics card (JSON) for the pose shown, into `{html.escape(str(self.out))}`.</small>")
            btn = g.add_button("Save PNG + card")
            self.g_export_md = g.add_markdown("")
            btn.on_click(lambda _e: self._export_current())

    def _pose_selected(self):
        p = next((p for p in self.study.poses if p.label == self.g_pose.value), None)
        if p is not None:
            self._apply(p.q, name=p.label)

    def _apply(self, q, name: str):
        """Show a pose: sliders follow without each firing a recompute."""
        self.pose_name = name
        q = np.asarray(q, float)
        self._muted = True
        try:
            for h, v in zip(self.g_jog, np.rad2deg(q)):
                h.value = float(np.clip(v, h.min, h.max))
        finally:
            self._muted = False
        p = next((p for p in self.study.poses if p.label == name), None)
        if p is not None:
            parts = [html.escape(p.description)] if p.description else []
            if p.digest:
                parts.append(f"digest `{p.digest}`")
            self.g_pose_md.content = "<small>" + " · ".join(parts) + "</small>" if parts else ""
        self._request(q)

    def _jogged(self):
        if self._muted:
            return
        q = np.deg2rad([h.value for h in self.g_jog])
        self._request(q, jogged=True)

    def _request(self, q: np.ndarray, jogged: bool = False):
        """Latest wins: a slider drag queues many values; the worker measures only the newest."""
        with self.lock:
            self._pending = np.asarray(q, float)
            self._jogged_flag = jogged
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._work, daemon=True)
            self._worker.start()

    def _work(self):
        while True:
            with self.lock:
                q, self._pending = self._pending, None
                jogged = getattr(self, "_jogged_flag", False)
            if q is None:
                return
            with self.arm_lock:
                r = self.study.measure(q)
                self.q, self.report = q, r
                self._draw_arm(q, r.ok)
            self._draw_marks(r)
            self._show(r, jogged)

    def _show(self, r: RestReport, jogged: bool):
        self.g_head.content = headline_html(r, self.study)
        self.g_detail.content = detail_markdown(r, self.study)
        code = Code(r.code)
        col = CODE_COLORS[code]
        where = f"jogged from **{html.escape(self.pose_name)}**" if jogged else f"**{html.escape(self.pose_name)}**"
        self.g_jog_md.content = f"<span style='color:rgb{col}'>●</span> {r.code_label} · {where} · measured in {1e3 * r.seconds:.0f} ms"

    def _rank(self):
        try:
            reports = {}
            for pose in self.study.poses:  # one pose per lock hold, so a jog never waits for the whole ranking
                with self.arm_lock:
                    reports[pose.label] = self.study.measure(pose.q)
            ranked = rank(reports, self.study.gates, self.study.criteria)
            self.g_rank.content = rank_markdown(ranked, self.study)
        except Exception as exc:  # noqa: BLE001 - the explorer stays usable without a ranking
            self.g_rank.content = f"_ranking failed: {html.escape(repr(exc))}_"

    def _export_current(self):
        from .rest_plots import save_card

        name = self.pose_name + (" jogged" if getattr(self, "_jogged_flag", False) else "")
        p = PoseChoice(name=name.replace("candidate · ", ""), q=self.q, kind="named")
        try:
            with self.arm_lock:
                png = save_card(self.study, p, self.out, report=self.report)
            self.g_export_md.content = f"<small>saved `{html.escape(str(png))}` + `.json`</small>"
        except Exception as exc:  # noqa: BLE001 - a failed export is reported in the panel, the viewer keeps running
            self.g_export_md.content = f"<small>export failed: {html.escape(repr(exc))}</small>"

    # ================================================================== scripted twins of clicks
    def select(self, name: str):
        """Choose a pose by its label or name (the scripted twin of the dropdown)."""
        p = next((p for p in self.study.poses if name in (p.label, p.name)), None)
        if p is None:
            raise KeyError(f"no pose {name!r}; there are {[p.label for p in self.study.poses]}")
        self.g_pose.value = p.label
        self._apply(p.q, name=p.label)

    def set_joint(self, joint: str, degrees: float):
        roles = {v: k for k, v in (self.arm.meta.get("roles") or {}).items()}
        name = roles.get(joint, joint)
        i = self.arm.joint_names.index(name)
        self.g_jog[i].value = float(degrees)
        self._jogged()

    def wait_idle(self, timeout: float = 30.0):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            with self.lock:
                busy = self._pending is not None or (self._worker is not None and self._worker.is_alive())
            if not busy:
                return
            time.sleep(0.02)

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


# ====================================================================== panel text
def headline_html(r: RestReport, study: RestStudy) -> str:
    """The three numbers, big; red only for a problem."""
    scene, arm = study.scene, study.arm
    roles = arm.meta.get("roles") or {}
    f, c, h = r.footprint, r.clearance, r.holding
    plane = scene.label("plane", "plane")

    def cell(title, value, unit, sub, bad=False):
        colour = f"rgb{PROBLEM}" if bad else "inherit"
        return (f"<div style='flex:1;min-width:0;padding:6px 8px;border-radius:6px;background:rgba(120,128,140,0.08)'>"
                f"<div style='font-size:11px;opacity:0.7'>{title}</div>"
                f"<div style='font-size:20px;font-weight:600;color:{colour};white-space:nowrap'>{value}<span style='font-size:12px;font-weight:400'> {unit}</span></div>"
                f"<div style='font-size:10.5px;line-height:1.25;opacity:0.75'>{sub}</div></div>")

    if not r.ok:
        return f"<div style='color:rgb{PROBLEM};font-weight:600'>{html.escape(r.code_label)}: no geometry at this pose</div>"
    past = "" if f.past_shoulder is None else (f" · {_mm(f.past_shoulder)} mm past the shoulder" if f.past_shoulder > 0 else " · inside its lane")
    fp = cell("Footprint", _mm(f.reach), "mm", f"hand from the line · hull {f.hull_area:.2f} m²{past}", bad=bool(f.past_shoulder and f.past_shoulder > 0))
    clear_bad = c.plane is not None and c.plane < c.threshold
    ko_bad = any(v is not None and v < 0 for v in c.keepouts.values())
    ko = "".join(f" · {html.escape(k)} {_mm(v)} mm" for k, v in c.keepouts.items() if v is not None)
    over = "nothing over the " + html.escape(plane) if c.plane is None else f"over the {html.escape(plane)} (gate {_mm(c.threshold)})"
    cl = cell("Clearance", "∞" if c.plane is None else _mm(c.plane), "mm", over + ko, bad=clear_bad or ko_bad)
    worst = "" if h.watts is None or h.total is None or h.total <= 0 else f" · most: {html.escape(str(roles.get(arm.joint_names[int(np.argmax(h.watts))], arm.joint_names[int(np.argmax(h.watts))])))}"
    hp = cell("Holding", "—" if h.total is None else f"{h.total:.2f}", "W", ("copper, all joints" + worst) if h.watts is not None else "no motor model")
    return f"<div style='display:flex;gap:6px'>{fp}{cl}{hp}</div>"


def detail_markdown(r: RestReport, study: RestStudy) -> str:
    arm, scene = study.arm, study.scene
    roles = arm.meta.get("roles") or {}
    h = r.holding
    lines = [f"**Check** {r.code_label}"]
    if r.footprint is not None:
        f, c = r.footprint, r.clearance
        lines += [
            "",
            "**Footprint** (from the " + html.escape(scene.label("line", "reference line")) + ")",
            f"- hand across {_range_mm(f.hand_across)}, along {_range_mm(f.hand_along)}",
            f"- whole arm across {_range_mm(f.across)}, along {_range_mm(f.along)}",
            f"- hull {f.hull_area:.4f} m² on the {html.escape(scene.label('plane', 'plane'))}",
        ]
        for axis, iv in f.lane.items():
            lines.append(f"- lane {axis} {_range_mm(iv)} (the shoulder's own width)")
        if f.past_shoulder is not None:
            lines.append(f"- past the shoulder: {1e3 * f.past_shoulder:+.0f} mm {f.past_axis} (≤ 0 = inside)")
        lines += ["", "**Clearance**", f"- over the {html.escape(scene.label('plane', 'plane'))}: {_mm(c.plane)} mm (gate {_mm(c.threshold)} mm)"]
        for name, v in c.keepouts.items():
            lines.append(f"- over the {html.escape(name)}: {_mm(v)} mm" + (" (nothing over it)" if v is None else ""))
        if c.lowest is not None:
            a, b, _ = scene.coords(c.lowest)
            lines.append(f"- lowest point at {1e3 * a:+.0f} mm across, {1e3 * b:+.0f} mm along")
    lines += ["", "**Holding**", f"<small>{html.escape(h.source)}</small>", "", "| joint | deg | N·m | W | % limit |", "|---|---:|---:|---:|---:|"]
    for i, name in enumerate(arm.joint_names):
        tau = "—" if h.tau is None else f"{h.tau[i]:+.2f}"
        w = "—" if h.watts is None else f"{h.watts[i]:.2f}"
        pct = "—" if h.limit_frac is None else f"{100 * h.limit_frac[i]:.0f}"
        lines.append(f"| {roles.get(name, name)} | {np.degrees(r.q[i]):+.1f} | {tau} | {w} | {pct} |")
    if h.total is not None:
        lines.append(f"| **total** | | | **{h.total:.2f}** | |")
    return "\n".join(lines)


def rank_markdown(ranked, study: RestStudy) -> str:
    out = ["<small>Gates first, then the weighted score (anchors fixed before looking).</small>", "",
           "| # | pose | score | reach | clear | W | gates |", "|---:|---|---:|---:|---:|---:|---|"]
    for k, r in enumerate(ranked, 1):
        m = r.report.metrics()
        gates = "pass" if r.passed else "fails " + ", ".join(g.split(" ")[0] for g in r.failed)
        w = "—" if m.get("power_w") is None else f"{m['power_w']:.1f}"
        out.append(f"| {k} | {html.escape(r.name.replace('candidate · ', ''))} | {r.score:.2f} | {_mm(m.get('reach_m'))} | {_mm(m.get('clearance_m'))} | {w} | {gates} |")
    if study.criteria:
        out += ["", "<small>" + " · ".join(f"{html.escape(c.name)} ×{c.weight:g}" for c in study.criteria) + "</small>"]
    return "\n".join(out)


# ====================================================================== command line
def with_pose_files(study: RestStudy, poses=(), candidates=()) -> RestStudy:
    """`study` plus the poses of robot_msgs named-poses files and candidates files (read with its aliases)."""
    extra: list = []
    for path in poses:
        extra += named_poses(path, study.arm, aliases=study.aliases)
    for path in candidates:
        extra += candidate_poses(path, study.arm, aliases=study.aliases)
    return study.with_poses(extra) if extra else study


def cli(factory: Callable[[], RestStudy], argv=None, prog: str = "robot_reach rest") -> int:
    """The explorer's command line around a study factory (`python -m robot_reach rest`, adapters): view
    it, script a start (``--pose``, ``--set``), export the cards."""
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--pose", help="start at this pose (its name or label)")
    ap.add_argument("--set", action="append", default=[], metavar="JOINT=DEG", help="then jog a joint (repeatable)")
    ap.add_argument("--export", type=Path, help="write one PNG + metrics card per pose here (and index.json with the ranking)")
    ap.add_argument("--out", type=Path, help="where the viewer's Save button writes (default ./rest-cards)")
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    a = ap.parse_args(argv)
    study = factory()
    if a.export is not None:
        from .rest_plots import export_cards

        paths = export_cards(study, a.export)
        print(f"wrote {len(paths)} cards -> {a.export}", flush=True)
    if a.no_view:
        return 0
    viewer = RestViewer(study, host=a.host, port=a.port, out=a.out)
    if a.pose:
        viewer.select(a.pose)
    for item in a.set:
        joint, value = item.split("=", 1)
        viewer.set_joint(joint.strip(), float(value))
    viewer.wait_idle()
    viewer.run()
    return 0
