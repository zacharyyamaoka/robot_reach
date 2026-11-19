"""2-D pictures (matplotlib, optional extra): the joint-space map, the joint line, the joint-range
bars, and the planar Cartesian map. The viewer embeds the same figures in its side panel.

Angles are shown in degrees (Zach's standing rule); the data stays in radians.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .codes import CODE_COLORS, CODE_LABELS, Code, code_rgb, ramp_rgb

INK = "#1f2328"
MUTED = "#6e7781"
GRID = "#d0d7de"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _legend(ax, codes, extra=()):
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=np.array(CODE_COLORS[Code(c)]) / 255, edgecolor=GRID, label=CODE_LABELS[Code(c)])
        for c in sorted(set(int(c) for c in np.unique(codes)))
        if c != Code.PAD
    ]
    handles += list(extra)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)


def joint_map_figure(jm, title: str | None = None):
    plt = _plt()
    if len(jm.axes) == 1:
        return joint_line_figure(jm, title)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=110)
    gx, gy = np.rad2deg(jm.grids[0]), np.rad2deg(jm.grids[1])
    if jm.mode == "slice":
        img = code_rgb(jm.code)
    else:
        # sweep: how often the cell works across the other joints' postures; white = never
        img = np.where((jm.ok_fraction > 0)[..., None], _green_ramp(jm.ok_fraction), code_rgb(jm.code))
    ax.imshow(np.swapaxes(img, 0, 1), origin="lower", extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect="auto", interpolation="nearest")
    ax.set_xlabel(f"{jm.names[0]} (deg)")
    ax.set_ylabel(f"{jm.names[1]} (deg)")
    others = "" if jm.mode == "sweep" else _q0_text(jm)
    ax.set_title(title or f"{jm.arm_name}: {jm.names[0]} × {jm.names[1]} — {jm.mode}{others}", loc="left")
    extra = []
    if jm.mode == "sweep":
        from matplotlib.patches import Patch

        extra = [Patch(facecolor=np.array(_green_ramp(np.array([1.0]))[0]) / 255, label="OK for all sampled postures"),
                 Patch(facecolor=np.array(_green_ramp(np.array([0.3]))[0]) / 255, label="OK for some")]
    _legend(ax, jm.code[jm.ok_fraction == 0] if jm.mode == "sweep" else jm.code, extra)
    fig.tight_layout()
    return fig


def _q0_text(jm):
    return ""


def _green_ramp(f):
    f = np.clip(np.asarray(f, float), 0, 1)[..., None]
    light, solid = np.array([220, 240, 224.0]), np.array(CODE_COLORS[Code.OK], float)
    return (light * (1 - f) + solid * f).astype(np.uint8)


def joint_line_figure(jm, title: str | None = None, torque_limits=None):
    plt = _plt()
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.4, 3.6), dpi=110, sharex=True, gridspec_kw={"height_ratios": [1, 4]})
    g = np.rad2deg(jm.grids[0])
    ax0.imshow(code_rgb(jm.code)[None], extent=[g[0], g[-1], 0, 1], aspect="auto", interpolation="nearest")
    ax0.set_yticks([])
    ax0.set_title(title or f"{jm.arm_name}: {jm.names[0]} — what the joint can hold", loc="left")
    if jm.torque is not None:
        ax1.plot(g, jm.torque[:, jm.axes[0]], color=INK, lw=1.4, label="gravity torque")
        if torque_limits is not None:
            lim = torque_limits[jm.axes[0]]
            for s in (+1, -1):
                ax1.axhline(s * lim, color=np.array(CODE_COLORS[Code.TORQUE_LIMIT]) / 255, ls="--", lw=1)
            ax1.text(g[0], lim, "  motor limit", va="bottom", color=MUTED, fontsize=8)
        ax1.set_ylabel("N·m")
    ax1.set_xlabel(f"{jm.names[0]} (deg)")
    ax1.grid(color=GRID, lw=0.5)
    _legend(ax0, jm.code)
    fig.tight_layout()
    return fig


def ranges_figure(ranges, arm_name: str = ""):
    """One bar per joint: the limits (light), the range OK IK solutions use (solid), and how often
    each angle is used (the strip's darkness)."""
    plt = _plt()
    n = len(ranges.names)
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * n + 1.0), dpi=110)
    lim = np.rad2deg(ranges.limits)
    for i in range(n):
        y = n - 1 - i
        ax.barh(y, lim[i, 1] - lim[i, 0], left=lim[i, 0], height=0.62, color="#f0f2f4", edgecolor=GRID)
        if ranges.hist is not None and ranges.hist[i].sum() > 0:
            edges = np.rad2deg(ranges.hist_edges[i])
            # WHY smoothed: a lattice of targets aliases into a striped histogram that reads as
            # noise; a 5-tap blur keeps the shape (where solutions cluster) and drops the stripes.
            h = np.convolve(ranges.hist[i].astype(float), [1, 2, 3, 2, 1], mode="same")
            h = h / h.max()
            base, solid = np.array([0.94, 0.95, 0.96]), np.array([34, 94, 168]) / 255
            colors = base * (1 - h[:, None]) + solid * h[:, None]
            for k in range(len(h)):
                if h[k] > 0:
                    ax.barh(y, edges[k + 1] - edges[k], left=edges[k], height=0.62, color=colors[k], edgecolor="none")
            u = np.rad2deg(ranges.used[i])
            ax.plot(u, [y, y], "|", color=INK, ms=14, mew=1.6)
            ax.text(lim[i, 1] + 4, y, f"uses {u[0]:.0f}…{u[1]:.0f}°", va="center", fontsize=8, color=MUTED)
        else:
            ax.text(lim[i, 1] + 4, y, f"{lim[i, 0]:.0f}…{lim[i, 1]:.0f}°", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(ranges.names)))
    ax.set_xlabel("joint angle (deg)")
    ax.set_title(f"{arm_name}: joint limits (light) and what the reachable solutions use (dark = often)", loc="left")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_xlim(lim.min() - 10, lim.max() + 90)
    fig.tight_layout()
    return fig


def plane_figure(result):
    """Planar arms: the Cartesian picture in the arm's plane, coloured by how many IK branches work
    (the classic 2R view: 0 white, 1 light, 2 solid)."""
    plt = _plt()
    reach = result.reach
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=110)
    p = reach.positions
    n = reach.n_solutions[:, 0]
    worst = _dominant_failure(reach.code[:, 0, :])
    step = float(result.config["step"])
    colors = np.where((n > 0)[:, None], ramp_rgb(n, max(1, reach.n_branches)), code_rgb(worst))
    ix = np.round((p[:, 0] - p[:, 0].min()) / step).astype(int)
    iz = np.round((p[:, 2] - p[:, 2].min()) / step).astype(int)
    img = np.full((iz.max() + 1, ix.max() + 1, 3), 255, np.uint8)
    img[iz, ix] = colors
    half = step / 2
    ax.imshow(img, origin="lower", interpolation="nearest",
              extent=[p[:, 0].min() - half, p[:, 0].max() + half, p[:, 2].min() - half, p[:, 2].max() + half])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"{result.arm_info['name']}: reachable points in its plane — colour = number of IK branches", loc="left")
    from matplotlib.patches import Patch

    extra = [Patch(facecolor=ramp_rgb(np.array([k]), max(1, reach.n_branches))[0] / 255, label=f"{k} solution{'s' if k > 1 else ''}")
             for k in range(1, int(n.max()) + 1)]
    _legend(ax, worst[n == 0], extra)
    if result.config.get("floor_z") is not None:
        ax.axhline(result.config["floor_z"], color=np.array(CODE_COLORS[Code.COLLISION]) / 255, lw=1, ls="--")
    ax.grid(color=GRID, lw=0.5)
    fig.tight_layout()
    return fig


def _dominant_failure(codes: np.ndarray) -> np.ndarray:
    """(P, S) branch codes -> (P,) the 'closest to working' failure, for colouring empty cells."""
    order = [Code.SINGULAR, Code.TORQUE_LIMIT, Code.COLLISION, Code.JOINT_LIMIT, Code.FK_MISMATCH, Code.FK_FAIL, Code.NO_SOLUTION]
    out = np.full(len(codes), int(Code.NO_SOLUTION), np.int8)
    for c in reversed(order):
        out[np.any(codes == c, axis=1)] = c
    return out


def figures(result) -> dict[str, object]:
    """Every 2-D figure that applies to this result, by name."""
    out = {}
    tl = result.arm_info.get("torque_limits")
    for jm in result.joint_maps:
        if len(jm.axes) == 1:
            out[f"{jm.names[0]}: torque line"] = joint_line_figure(jm, torque_limits=tl)
        else:
            out[f"{jm.names[0]} × {jm.names[1]}: {jm.mode}"] = joint_map_figure(jm)
    out["joint ranges"] = ranges_figure(result.ranges, result.arm_info["name"])
    if result.reach is not None and result.reach.task == "position" and np.ptp(result.reach.positions[:, 1]) == 0:
        out["Cartesian plane: IK solutions"] = plane_figure(result)
    return out


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower().replace("×", "x")).strip("_")


def save_figures(result, folder: str | Path) -> list[Path]:
    plt = _plt()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in figures(result).items():
        path = folder / f"{slug(name)}.png"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(path)
    return paths


def figure_rgb(fig) -> np.ndarray:
    """A figure as an (H, W, 3) uint8 image, for the viewer's side panel."""
    plt = _plt()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img
