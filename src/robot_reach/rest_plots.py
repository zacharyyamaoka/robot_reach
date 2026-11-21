"""Static rest-pose cards for reports: one PNG (side view, top view, the numbers) + one JSON per pose.

    from robot_reach.rest_plots import export_cards
    export_cards(study, "out/rest")        # <pose>.png, <pose>.json, index.json (the ranking)

Needs the ``plot`` extra (matplotlib), imported lazily.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .rest import PoseChoice, RestReport, RestStudy, rank

PROBLEM = "#d62828"
FINE = "#2e8c54"
LINE = "#d62828"
ARM = "#5a606c"
HAND = "#3a6ea5"
BELT = "#c4ccd6"


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "pose"


def _mm(x) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{1e3 * x:.0f}"


_FACE_PICK: dict = {}


def _big_faces(arm, budget: int) -> dict:
    """Per drawn mesh, the indices of its largest triangles, the whole arm within `budget` triangles.

    WHY by area: a CAD mesh spends most triangles on bolts and fillets; the large faces make the
    silhouette, so the card stays a solid picture at a size matplotlib draws in a second."""
    key = (id(arm), budget)
    if key in _FACE_PICK:
        return _FACE_PICK[key]
    drawn = [i for i, m in enumerate(arm.meshes) if m.draw and m.faces is not None]
    areas = {}
    for i in drawn:
        v, f = np.asarray(arm.meshes[i].vertices, float), np.asarray(arm.meshes[i].faces, int)
        areas[i] = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    allareas = np.concatenate(list(areas.values())) if areas else np.zeros(0)
    cut = np.sort(allareas)[::-1][min(budget, len(allareas)) - 1] if len(allareas) > budget else 0.0
    pick = {i: np.nonzero(a >= cut)[0] for i, a in areas.items()}
    _FACE_PICK[key] = pick
    return pick


def _triangles(study: RestStudy, q: np.ndarray, budget: int = 60_000):
    """[(part, (F, 3, 3) world triangles)] of the drawn meshes (the largest `budget` of them), or None
    when the arm has no drawn mesh with faces."""
    arm = study.arm
    drawn = [i for i, m in enumerate(arm.meshes) if m.draw and m.faces is not None]
    if not drawn or arm.mesh_poses is None:
        return None
    pick = _big_faces(arm, budget)
    T = np.asarray(arm.mesh_poses(q[None]), float)[0]
    out = []
    for i in drawn:
        m = arm.meshes[i]
        v = np.asarray(m.vertices, float) @ T[i, :3, :3].T + T[i, :3, 3]
        out.append((m.part, v[np.asarray(m.faces, int)[pick[i]]]))
    return out


def _drawn_points(study: RestStudy, q: np.ndarray, cap: int = 400_000):
    """(points (P, 3), is_hand (P,)) — every vertex of the drawn meshes, posed (a dense picture of a CAD
    arm whose triangles are too many to plot); None without drawn meshes."""
    arm = study.arm
    drawn = [i for i, m in enumerate(arm.meshes) if m.draw]
    if not drawn or arm.mesh_poses is None:
        return None
    T = np.asarray(arm.mesh_poses(q[None]), float)[0]
    total = sum(len(arm.meshes[i].vertices) for i in drawn)
    stride = max(1, int(np.ceil(total / cap)))
    pts, hand = [], []
    for i in drawn:
        m = arm.meshes[i]
        v = np.asarray(m.vertices, float)[::stride] @ T[i, :3, :3].T + T[i, :3, 3]
        pts.append(v)
        hand.append(np.full(len(v), m.part in study.hand))
    pts, hand = np.concatenate(pts), np.concatenate(hand)
    good = np.all(np.isfinite(pts), axis=1)
    return pts[good], hand[good]


def card_figure(study: RestStudy, pose: PoseChoice, report: RestReport | None = None):
    """A matplotlib Figure: side view (across vs height) over top view (across vs along), sharing the
    across axis, and the card beside them."""
    import textwrap

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Polygon, Rectangle

    from .rest import _surface_labels

    arm, sc = study.arm, study.scene
    r = report if report is not None else study.measure(pose.q)
    pts = arm.surface(pose.q[None], max_per_mesh=study.max_per_mesh)[0]
    labels = _surface_labels(arm, study.max_per_mesh)
    if len(labels) != len(pts):
        labels = np.full(len(pts), "", dtype=object)
    good = np.all(np.isfinite(pts), axis=1)
    pts, labels = pts[good], labels[good]
    abh = sc.coords(pts) if len(pts) else np.zeros((0, 3))
    hand = np.isin(labels, list(study.hand)) if study.hand else np.zeros(len(pts), bool)
    tris = _triangles(study, pose.q) if r.ok else None
    dense = _drawn_points(study, pose.q) if r.ok and tris is None else None

    if dense is not None and len(dense[0]):
        abh_all = np.concatenate([abh, sc.coords(dense[0])])
    else:
        abh_all = abh
    ra = sc.region_across or ((abh[:, 0].min(), abh[:, 0].max()) if len(abh) else (-0.3, 0.3))
    a_lo = min(ra[0], abh_all[:, 0].min() if len(abh_all) else ra[0]) - 0.06
    a_hi = max(ra[1], abh_all[:, 0].max() if len(abh_all) else ra[1]) + 0.06
    b_all = abh[:, 1] if len(abh) else np.zeros(1)
    f, c = r.footprint, r.clearance
    lanes = [v for iv in (f.lane.values() if f is not None else ()) for v in iv]
    rb = sc.region_along or (min([float(b_all.min())] + lanes) - 0.06, max([float(b_all.max())] + lanes) + 0.06)
    h_floor = float(np.min(sc.plane_height(np.array([a_lo, a_hi]), 0.0))) - 0.03
    h_top = (abh_all[:, 2].max() if len(abh_all) else 0.0) + 0.04
    plane = sc.label("plane", "plane")

    fig = plt.figure(figsize=(12.5, 7.6), dpi=110)
    gs = fig.add_gridspec(2, 2, width_ratios=(1.5, 1.0), height_ratios=(h_top - h_floor, rb[1] - rb[0]), wspace=0.12, hspace=0.18)
    ax_s, ax_t, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])

    def draw_arm(ax, cols):
        if tris is not None:
            for part, t in tris:
                colour = HAND if part in study.hand else ARM
                ax.add_collection(PolyCollection(sc.coords(t)[..., cols], facecolors=colour, edgecolors="none", alpha=0.55))
        elif dense is not None and len(dense[0]):
            ab, hd = sc.coords(dense[0]), dense[1]
            ax.scatter(ab[~hd, cols[0]], ab[~hd, cols[1]], s=0.25, c=ARM, alpha=0.25, lw=0, rasterized=True)
            ax.scatter(ab[hd, cols[0]], ab[hd, cols[1]], s=0.25, c=HAND, alpha=0.35, lw=0, rasterized=True)
        elif len(abh):
            sub = np.linspace(0, len(abh) - 1, min(len(abh), 60_000)).astype(int)  # plotting only; every point counts in the numbers
            ab, hd = abh[sub], hand[sub]
            ax.scatter(ab[~hd, cols[0]], ab[~hd, cols[1]], s=0.2, c=ARM, alpha=0.35, lw=0, rasterized=True)
            ax.scatter(ab[hd, cols[0]], ab[hd, cols[1]], s=0.2, c=HAND, alpha=0.55, lw=0, rasterized=True)

    # ---- side view: across vs height (along collapsed)
    aa = np.linspace(a_lo, a_hi, 50)
    ax_s.fill_between(aa, h_floor, sc.plane_height(aa, 0.0), color=BELT, lw=0)
    for edge in ra:
        ax_s.axvline(edge, color="#8a94a2", lw=0.8, ls=":")
    for ko in sc.keepouts:
        ka = ko.across or (a_lo, a_hi)
        base = float(np.min(sc.plane_height(np.array(ka), 0.0)))
        ax_s.add_patch(Rectangle((ka[0], min(base, ko.top)), ka[1] - ka[0], abs(ko.top - base), color="#969ca6", alpha=0.7, lw=0))
    draw_arm(ax_s, [0, 2])
    ax_s.plot([0, 0], [float(sc.plane_height(0.0, 0.0)), h_top], color=LINE, lw=1.2)
    if c is not None and c.lowest is not None:
        a, b, h = sc.coords(c.lowest)
        foot_h = float(sc.plane_height(a, b))
        col = PROBLEM if c.plane < c.threshold else FINE
        ax_s.annotate("", xy=(a, foot_h), xytext=(a, h), arrowprops={"arrowstyle": "<->", "color": col, "lw": 1.4})
        ax_s.text(a + 0.01, (h + foot_h) / 2, f"{_mm(c.plane)} mm", color=col, fontsize=9, va="center")
    ax_s.set_aspect("equal")
    ax_s.set_xlim(a_lo, a_hi)
    ax_s.set_ylim(h_floor, h_top)
    ax_s.set_title(f"side: across vs height (m) · {plane} grey · red = {sc.label('line', 'reference line')}", loc="left", fontsize=9)
    ax_s.tick_params(labelsize=7, labelbottom=False)

    # ---- top view: across vs along
    ax_t.add_patch(Rectangle((ra[0], rb[0]), ra[1] - ra[0], rb[1] - rb[0], color=BELT, alpha=0.8, lw=0))
    for ko in sc.keepouts:
        ka, kb = ko.across or (a_lo, a_hi), ko.along or rb
        ax_t.add_patch(Rectangle((ka[0], kb[0]), ka[1] - ka[0], kb[1] - kb[0], color="#969ca6", alpha=0.7, lw=0))
    draw_arm(ax_t, [0, 1])
    if f is not None:
        if len(f.hull) >= 3:
            ax_t.add_patch(Polygon(f.hull, closed=True, fill=False, ec=HAND, lw=1.0))
        for axis, (lo, hi) in f.lane.items():
            for v in (lo, hi):
                (ax_t.axhline if axis == "along" else ax_t.axvline)(v, color=HAND, lw=0.8, ls="--")
    ax_t.axvline(0.0, color=LINE, lw=1.2)
    ax_t.set_aspect("equal")
    ax_t.set_xlim(a_lo, a_hi)
    ax_t.set_ylim(*rb)
    ax_t.set_title("top: across vs along (m) · hull outline · dashed = the lane (next arm beyond)", loc="left", fontsize=9)
    ax_t.tick_params(labelsize=7)

    # ---- the card
    ax_c.axis("off")
    head = r.headline()
    bad_clear = c is not None and c.plane is not None and c.plane < c.threshold
    lines = [(pose.label, 13, "bold", "black", None)]
    for chunk in textwrap.wrap(pose.description, 52)[:5] if pose.description else []:
        lines.append((chunk, 8.5, "normal", "#555", None))
    if pose.digest:
        lines.append((f"digest {pose.digest}", 8.5, "normal", "#555", None))
    lines += [("", 6, "normal", "black", None),
              (f"Footprint  {head['footprint']}", 10, "bold", "black", None),
              (f"Clearance  {head['clearance']} over the {plane} (gate {_mm(sc.min_clearance)} mm)", 10, "bold", PROBLEM if bad_clear else "black", None),
              (f"Holding    {head['holding']}", 10, "bold", "black", None),
              ("", 6, "normal", "black", None)]
    if f is not None:
        lines.append((f"hand across {1e3 * f.hand_across[0]:+.0f}…{1e3 * f.hand_across[1]:+.0f} mm, along {1e3 * f.hand_along[0]:+.0f}…{1e3 * f.hand_along[1]:+.0f} mm", 8, "normal", "#333", "monospace"))
        if f.past_shoulder is not None:
            lines.append((f"past the shoulder {1e3 * f.past_shoulder:+.0f} mm {f.past_axis} (<= 0 inside)", 8, "normal", "#333", "monospace"))
    if c is not None:
        for name, v in c.keepouts.items():
            lines.append((f"over the {name}: " + ("nothing over it" if v is None else f"{_mm(v)} mm"), 8, "normal", "#333", "monospace"))
    lines.append((f"check: {r.code_label}", 8, "normal", "#333", "monospace"))
    h = r.holding
    roles = arm.meta.get("roles") or {}
    if h.tau is not None:
        lines.append(("", 5, "normal", "black", None))
        lines.append((f"{'joint':8s} {'deg':>7s} {'N·m':>8s} {'W':>6s}", 8, "bold", "#333", "monospace"))
        for i, name in enumerate(arm.joint_names):
            w = "—" if h.watts is None else f"{h.watts[i]:6.2f}"
            lines.append((f"{roles.get(name, name)[:8]:8s} {np.degrees(r.q[i]):+7.1f} {h.tau[i]:+8.2f} {w:>6}", 8, "normal", "#333", "monospace"))
        for chunk in textwrap.wrap(h.source, 60)[:3]:
            lines.append((chunk, 7, "normal", "#777", None))
    fig.canvas.draw()
    height_pt = ax_c.get_window_extent().height * 72.0 / fig.dpi
    y = 1.0
    for text, size, weight, color, family in lines:
        ax_c.text(0.0, y, text, fontsize=size, fontweight=weight, color=color, va="top", family=family, transform=ax_c.transAxes)
        y -= 1.45 * size / height_pt
    fig.suptitle(f"{study.name or arm.name}", x=0.01, ha="left", fontsize=10, color="#555")
    return fig, r


def save_card(study: RestStudy, pose: PoseChoice, out, report: RestReport | None = None) -> Path:
    """`<out>/<pose>.png` + `<pose>.json`; returns the PNG path."""
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig, r = card_figure(study, pose, report)
    stem = _slug(pose.name)
    png = out / f"{stem}.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    doc = {"pose": pose.name, "kind": pose.kind, "description": pose.description, "digest": pose.digest, "source": pose.source,
           "study": study.name, "arm": study.arm.name, **r.to_dict(study.arm)}
    (out / f"{stem}.json").write_text(json.dumps(doc, indent=1, default=float))
    return png


def export_cards(study: RestStudy, out, poses=None) -> list[Path]:
    """A card per pose (default: the study's) plus `index.json`: every pose's metrics and, when the
    study has gates or criteria, the ranking."""
    out = Path(out)
    poses = list(study.poses if poses is None else poses)
    reports = {p.label: study.measure(p.q) for p in poses}
    paths = [save_card(study, p, out, reports[p.label]) for p in poses]
    index = {"study": study.name, "arm": study.arm.name, "poses": {p.label: reports[p.label].metrics() for p in poses}}
    if study.gates or study.criteria:
        index["ranking"] = [{"pose": r.name, "passed": r.passed, "failed": list(r.failed), "score": r.score, "scores": r.scores}
                            for r in rank(reports, study.gates, study.criteria)]
    (out / "index.json").write_text(json.dumps(index, indent=1, default=float))
    return paths
