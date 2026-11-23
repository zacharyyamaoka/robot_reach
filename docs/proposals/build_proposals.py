"""Build the design-proposals report from the REAL saved UR5e result (read-only).

    uv run --project /home/bam/robot_reach --extra viz python /home/bam/robot_reach/docs/proposals/build_proposals.py

Writes /home/bam/robot_reach/reports/reach-proposals-2026-11-23.html: one self-contained page. The
point clouds are drawn live in the page from the embedded data (gzip + base64, decoded with the
browser's DecompressionStream), so every mock-up filters and colours the same numbers the viewer has.
"""

from __future__ import annotations

import base64
import dataclasses
import gzip
import io
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import robot_reach as rr  # noqa: E402
from ik_branches import UR5E_BRANCHES, tag_all  # noqa: E402
from robot_reach.codes import CODE_COLORS, LEVEL_COLORS, Code  # noqa: E402
from robot_reach.orientations import OrientationSet  # noqa: E402
from robot_reach.reach import reach_map  # noqa: E402

OUT = Path("/home/bam/robot_reach/reports/reach-proposals-2026-11-23.html")
RESULT = "/home/bam/robot_reach/out/ur5e"


def b64gz(raw: bytes) -> str:
    return base64.b64encode(gzip.compress(raw, 9)).decode()


def fig_webp(fig, dpi=110) -> str:
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    out = io.BytesIO()
    Image.open(buf).convert("RGB").save(out, format="WEBP", quality=82, method=6)
    return "data:image/webp;base64," + base64.b64encode(out.getvalue()).decode()


# ============================================================================== the real data
res = rr.load(RESULT)
arm, m = res.arm, res.reach
P, O, S = m.code.shape
q_all = m.sol_q.astype(float)
sol_type = UR5E_BRANCHES.type_of(tag_all(q_all, arm.meta["frames"])).astype(np.uint8)
po = m.sol_index // S

# per (position, orientation): which branch TYPES work (bit t = type t), from the posture tags
mask = np.zeros(P * O, np.uint8)
np.bitwise_or.at(mask, po, (1 << sol_type).astype(np.uint8))

step = float(res.config["step"])
origin = m.positions.min(0)
gidx = np.rint((m.positions - origin) / step).astype(np.int16)
levels_py = m.levels()
nsol_max = m.n_solutions.max(axis=1).astype(np.uint8)
tilt_deg = np.rint(np.degrees(m.orientations.tilt)).astype(int)


def levels_for(allowed: np.ndarray) -> np.ndarray:
    bits = int(sum(1 << i for i in np.flatnonzero(allowed)))
    ok = (mask.reshape(P, O) & bits) > 0
    lv = np.where(ok.any(1), 3, 0)
    lv = np.where(ok[:, tilt_deg == 0].all(1), 4, lv)
    lv = np.where(ok[:, tilt_deg <= 90].all(1), 5, lv)
    return np.where(ok.all(1), 6, lv)


# the page recomputes levels from the mask; these are the numbers it must reproduce
checks = {
    "all": {int(k): int((levels_for(np.ones(8, bool)) == k).sum()) for k in (0, 3, 4, 5, 6)},
    "saved": {int(k): int((levels_py == k).sum()) for k in (0, 3, 4, 5, 6)},
    "elbow_up": {int(k): int((levels_for(UR5E_BRANCHES.allowed(elbow="up")) == k).sum()) for k in (0, 3, 4, 5, 6)},
    "front_up": {int(k): int((levels_for(UR5E_BRANCHES.allowed(shoulder="front", elbow="up")) == k).sum()) for k in (0, 3, 4, 5, 6)},
    "one_type": {int(k): int((levels_for(np.arange(8) == 0) == k).sum()) for k in (0, 3, 4, 5, 6)},
}
assert checks["all"] == checks["saved"], checks  # tags lose nothing: same levels as the saved map

# ----- consistency demo: the same map with an IK whose shoulder_lift zero is 0.011 deg off its FK
EPS = 2e-4  # rad
base_ik = arm.ik


def ik_offset(T):
    q, code = base_ik(T)
    q = q.copy()
    q[:, :, 1] += EPS
    return q, code


arm_off = dataclasses.replace(arm, ik=ik_offset)
cfg = res.config
n_claim, n_mism = np.zeros(P, int), np.zeros(P, int)
for lo in range(0, P, 400):  # WHY chunks: the host runs near its memory cap; per-point counts are all we keep
    m_off = reach_map(arm_off, m.positions[lo : lo + 400], m.orientations, floor_z=cfg["floor_z"], singular_tol=cfg["singular_tol"], w_ref=cfg["w_ref"])
    n_claim[lo : lo + 400] = (~np.isin(m_off.code, [Code.NO_SOLUTION, Code.PAD, Code.JOINT_LIMIT])).sum((1, 2))
    n_mism[lo : lo + 400] = (m_off.code == Code.FK_MISMATCH).sum((1, 2))
del m_off
consistency = np.where(n_claim == 0, 0, np.where(n_mism == 0, 1, np.where(n_mism < n_claim, 2, 3))).astype(np.uint8)
saved_mismatch = {name: int((rr.load(f"/home/bam/robot_reach/out/{name}", rebuild_arm=False).reach.code == Code.FK_MISMATCH).sum())
                  for name in ("ur5e", "ur5e/tool0", "rr", "shoulder3", "five_bar")}

# ----- FK cloud (H): what it is, verified from the saved arrays
ok_cloud = np.all(np.isfinite(res.cloud_p), 1)
sub = np.flatnonzero(ok_cloud)[:: max(1, ok_cloud.sum() // 6000)][:6000]
cloud_mm = np.rint(res.cloud_p[sub] * 1000).astype(np.int16)
cloud_code = res.cloud_code[sub].astype(np.uint8)
cloud_counts = {Code(int(c)).name: int(n) for c, n in zip(*np.unique(res.cloud_code, return_counts=True))}

# ----- the selected point of Zach's screenshot (+0.469, +0.009, +0.483): every solution there
p_sel = int(np.argmin(np.linalg.norm(m.positions - np.array([0.469, 0.009, 0.483]), axis=1)))
sols = m.solutions(p_sel)
s_lo = np.searchsorted(m.sol_index, p_sel * O * S)
sel_types = sol_type[s_lo : s_lo + len(sols)]
sel_rows = [[o, int(t), *np.round(np.degrees(q), 1).tolist()] for (o, _b, q), t in zip(sols, sel_types)]
# one posture per type at that point, for the type glyphs (the first orientation that has all 8)
glyph_o = next(o for o in range(O) if len({int(t) for (oo, _b, _q), t in zip(sols, sel_types) if oo == o}) == 8)
glyphs = {}
for (o, _b, q), t in zip(sols, sel_types):
    if o == glyph_o and int(t) not in glyphs:
        glyphs[int(t)] = np.round(arm.skeleton(q[None])[0], 4).tolist()
home_skel = np.round(arm.skeleton(np.asarray(arm.meta["home"])[None])[0], 4).tolist()

# ----- orientation counts from the library itself, for the page's JS port to match (C, F)
orient_checks = []
for ts, nr, mt in ((45, 6, 180), (45, 6, 90), (45, 6, 0), (30, 6, 90), (15, 12, 90), (30, 8, 180)):
    orient_checks.append({"tilt_step": ts, "n_rolls": nr, "max_tilt": mt, "n": len(OrientationSet.sample(tilt_step_deg=ts, n_rolls=nr, max_tilt_deg=mt))})

# ============================================================================== B: joint ranges
JL = [f"{n} (J{i + 1})" for i, n in enumerate(m.joint_names)]
qd = np.degrees(q_all)
lim = np.degrees(arm.limits)
rng = np.random.default_rng(0)
qd_s = qd[rng.choice(len(qd), 120_000, replace=False)]
elbow_up = np.array([UR5E_BRANCHES.tags_of(int(t))[1] == 0 for t in range(8)])
OKG, LIM, INK = "#2ea043", "#e9ecef", "#343a40"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})


def frame_axes(ax, ylabels=True):
    ax.set_xlim(-380, 380)
    ax.set_xticks(range(-360, 361, 90))
    ax.set_xlabel("joint angle (deg)")
    ax.grid(axis="x", color="#f1f3f5", lw=0.8)
    ax.set_axisbelow(True)


B = {}
# B1 violin
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for j in range(6):
    y = 5 - j
    ax.barh(y, lim[j, 1] - lim[j, 0], left=lim[j, 0], height=0.8, color=LIM, zorder=0)
    vp = ax.violinplot(qd_s[:, j], positions=[y], orientation="horizontal", widths=0.8, showextrema=False, points=200)
    for b in vp["bodies"]:
        b.set_facecolor(OKG); b.set_edgecolor("none"); b.set_alpha(0.85)
    lo, hi = np.percentile(qd[:, j], [0.5, 99.5])
    ax.text(375, y, f"{lo:+.0f}…{hi:+.0f}°", va="center", ha="right", fontsize=7.5, color="#495057")
ax.set_yticks(range(6)); ax.set_yticklabels(JL[::-1]); frame_axes(ax)
ax.set_title("width = how many working solutions use that angle · grey = joint limit", fontsize=8.5, color="#495057", loc="left")
B["violin"] = fig_webp(fig)

# B2 small-multiple histograms
fig, axs = plt.subplots(6, 1, figsize=(6.2, 4.4), sharex=True)
edges = np.arange(-360, 361, 10)
for j, ax in enumerate(axs):
    h, _ = np.histogram(qd[:, j], edges)
    ax.axvspan(lim[j, 0], lim[j, 1], color=LIM, zorder=0)
    ax.bar(edges[:-1], h / h.max(), width=10, align="edge", color=OKG, lw=0)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.text(-372, 0.55, JL[j], ha="left", va="center", fontsize=8, color=INK)
    ax.set_ylim(0, 1.05)
frame_axes(axs[-1])
axs[0].set_title("bar height = share of working solutions at that angle (10° bins) · grey = limit", fontsize=8.5, color="#495057", loc="left")
B["hist"] = fig_webp(fig)

# B3 ridgeline
fig, ax = plt.subplots(figsize=(6.2, 3.8))
centers = (edges[:-1] + edges[1:]) / 2
for j in range(6):
    y0 = (5 - j) * 1.0
    h, _ = np.histogram(qd[:, j], edges)
    h = h / h.max() * 0.9
    ax.plot([lim[j, 0], lim[j, 1]], [y0, y0], color="#adb5bd", lw=3, solid_capstyle="butt", zorder=1)
    ax.fill_between(centers, y0, y0 + h, color=OKG, alpha=0.75, lw=0, zorder=2 + j)
    ax.plot(centers, y0 + h, color="#1a7f37", lw=0.8, zorder=2 + j)
ax.set_yticks([5 - j for j in range(6)]); ax.set_yticklabels(JL); frame_axes(ax); ax.set_ylim(-0.2, 6.0)
ax.set_title("each ridge = one joint's distribution · grey base line = its limit", fontsize=8.5, color="#495057", loc="left")
B["ridge"] = fig_webp(fig)

# B4 range bars (box style)
fig, ax = plt.subplots(figsize=(6.2, 3.3))
for j in range(6):
    y = 5 - j
    p = np.percentile(qd[:, j], [1, 25, 50, 75, 99])
    ax.barh(y, lim[j, 1] - lim[j, 0], left=lim[j, 0], height=0.62, color=LIM, zorder=0)
    ax.barh(y, p[4] - p[0], left=p[0], height=0.26, color="#9be9a8", zorder=1)
    ax.barh(y, p[3] - p[1], left=p[1], height=0.44, color=OKG, zorder=2)
    ax.plot([p[2], p[2]], [y - 0.25, y + 0.25], color=INK, lw=1.6, zorder=3)
    used = (p[4] - p[0]) / (lim[j, 1] - lim[j, 0]) * 100
    ax.text(375, y, f"uses {used:.0f} % of limit", va="center", ha="right", fontsize=7.5, color="#495057")
ax.set_yticks(range(6)); ax.set_yticklabels(JL[::-1]); frame_axes(ax)
ax.set_title("light = 1–99 % of working solutions · dark = middle half · line = median · grey = limit", fontsize=8.5, color="#495057", loc="left")
B["box"] = fig_webp(fig)

# B5 histograms split by IK branch (elbow up vs down)
fig, axs = plt.subplots(6, 1, figsize=(6.2, 4.4), sharex=True)
up_sol = elbow_up[sol_type]
for j, ax in enumerate(axs):
    hu, _ = np.histogram(qd[up_sol, j], edges)
    hd, _ = np.histogram(qd[~up_sol, j], edges)
    top = max(hu.max(), hd.max())
    ax.axvspan(lim[j, 0], lim[j, 1], color=LIM, zorder=0)
    ax.bar(edges[:-1], hu / top, width=10, align="edge", color="#0072B2", lw=0, alpha=0.85, label="elbow up")
    ax.bar(edges[:-1], -hd / top, width=10, align="edge", color="#E69F00", lw=0, alpha=0.9, label="elbow down")
    ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.axhline(0, color="#adb5bd", lw=0.6)
    ax.text(-372, 0.5, JL[j], ha="left", va="center", fontsize=8, color=INK)
    ax.set_ylim(-1.05, 1.05)
frame_axes(axs[-1])
axs[0].legend(loc="lower right", fontsize=7, frameon=False, ncol=2, bbox_to_anchor=(1.0, 1.0))
axs[0].set_title("up = elbow-up solutions, down = elbow-down (mirror) · grey = limit", fontsize=8.5, color="#495057", loc="left")
B["split"] = fig_webp(fig)

range_facts = {JL[j]: [round(float(np.percentile(qd[:, j], 0.5)), 1), round(float(np.percentile(qd[:, j], 99.5)), 1)] for j in range(6)}

# ============================================================================== D: colour schemes
cm = matplotlib.colormaps


def cmap_rgb(name, ts):
    return [tuple(int(round(c * 255)) for c in cm[name](t)[:3]) for t in ts]


def v1_traffic(t):  # bam_reach v1 colorize.scores_to_colors: red -> yellow -> green
    R, Y, G = np.array((220, 38, 38)), np.array((245, 190, 45)), np.array((30, 190, 85))
    c = R + t * 2 * (Y - R) if t <= 0.5 else Y + (t - 0.5) * 2 * (G - Y)
    return tuple(int(round(x)) for x in c)


EMPTY = (236, 238, 240)
SCHEMES = {
    "today": {"name": "Today (reference)", "levels": [EMPTY, *[LEVEL_COLORS[k] for k in (3, 4, 5, 6)]],
              "ramp": [(247, 251, 255), (161, 218, 180), (65, 182, 196), (34, 94, 168), (12, 44, 132)]},
    "traffic": {"name": "D1 Traffic light", "levels": [EMPTY, *[v1_traffic(t) for t in (0, 1 / 3, 2 / 3, 1)]],
                "ramp": [v1_traffic(t) for t in np.linspace(0, 1, 5)]},
    # WHY light = least: in 3D the lowest level is the outer shell, so its colour covers most of the view
    "viridis": {"name": "D2 Viridis, light = least", "levels": [EMPTY, *cmap_rgb("viridis", (0.93, 0.68, 0.42, 0.12))],
                "ramp": cmap_rgb("viridis", np.linspace(1, 0, 7))},
    "rainbow": {"name": "D3 Rainbow bands", "levels": [EMPTY, (255, 0, 0), (255, 255, 0), (0, 255, 255), (0, 0, 255)],
                "ramp": [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255)]},
    "thermal": {"name": "D4 Thermal", "levels": [EMPTY, *cmap_rgb("inferno", (0.82, 0.62, 0.42, 0.18))],
                "ramp": cmap_rgb("inferno", np.linspace(0.92, 0.05, 7))},
    "okabe": {"name": "D5 Named colours", "levels": [EMPTY, (86, 180, 233), (0, 158, 115), (230, 159, 0), (0, 114, 178)],
              "ramp": cmap_rgb("cividis", np.linspace(0.95, 0.0, 7))},
}
MACHADO = {  # Machado, Oliveira & Fernandes 2009, severity 1.0, applied in linear RGB
    "deuteranopia": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    "protanopia": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    "tritanopia": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
}


def lin(c):
    c = np.asarray(c, float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def unlin(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055) * 255


def lab(rgb):
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin(rgb) @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def sim(rgb, kind):
    return unlin(lin(rgb) @ np.array(MACHADO[kind]).T)


def min_de(cols):
    L = lab(np.asarray(cols, float))
    return float(min(np.linalg.norm(L[i] - L[j]) for i in range(len(L)) for j in range(i + 1, len(L))))


problems = [CODE_COLORS[Code.COLLISION], CODE_COLORS[Code.SINGULAR]]
metrics = {}
for key, sc in SCHEMES.items():
    lv = np.array(sc["levels"][1:], float)
    row = {"normal": min_de(lv)}
    for kind in MACHADO:
        row[kind] = min_de(sim(lv, kind))
    Ls = lab(lv)[:, 0]
    row["lightness_order"] = bool(np.all(np.diff(Ls) > 0) or np.all(np.diff(Ls) < 0))
    row["vs_white"] = float(min(np.linalg.norm(lab(c) - lab((255, 255, 255))) for c in lv))
    row["vs_problem"] = float(min(np.linalg.norm(lab(c) - lab(p)) for c in lv for p in problems))
    row["sim"] = {kind: [tuple(int(x) for x in np.clip(np.rint(c), 0, 255)) for c in sim(lv, kind)] for kind in MACHADO}
    metrics[key] = row

# ============================================================================== G: the API demo, run
demo = subprocess.run([sys.executable, str(HERE / "api_demo.py")], capture_output=True, text=True, check=True).stdout

# ============================================================================== page
data = {
    "P": P, "O": O, "step": step, "origin": origin.tolist(), "n": (gidx.max(0) + 1).tolist(),
    "gidx": gidx.reshape(-1).tolist(), "tilt": tilt_deg.tolist(), "nsol": nsol_max.tolist(),
    "frac": np.rint(m.score("dexterous") * 100).astype(int).tolist(), "cons": consistency.tolist(),
    "levels_py": levels_py.tolist(), "checks": checks, "spec": UR5E_BRANCHES.ui_spec(),
    "sel": {"p": p_sel, "pos": np.round(m.positions[p_sel], 3).tolist(), "rows": sel_rows, "level": int(levels_py[p_sel]),
            "glyph_o": glyph_o, "glyphs": glyphs},
    "home": home_skel, "cloud": cloud_mm.reshape(-1).tolist(), "cloud_code": cloud_code.tolist(),
    "code_colors": {int(k): v for k, v in CODE_COLORS.items()},
    "box": res.config["box"], "schemes": SCHEMES, "metrics": metrics, "orient_checks": orient_checks,
    "rate": P * O / float(res.timings.get("reach_map", m.seconds)),
}
facts = {
    "P": P, "O": O, "S": S, "n_sol": int(len(m.sol_q)), "step_mm": step * 1000, "seconds": float(res.timings.get("reach_map", m.seconds)),
    "cloud_counts": cloud_counts, "n_fk": int(res.config["n_fk"]), "box": res.config["box"], "w_ref": res.config["w_ref"],
    "checks": checks, "saved_mismatch": saved_mismatch, "eps_deg": float(np.degrees(EPS)),
    "cons_counts": {int(k): int((consistency == k).sum()) for k in range(4)}, "range_facts": range_facts,
    "sel_n": len(sel_rows), "sel_pos": np.round(m.positions[p_sel], 3).tolist(), "per_type": np.bincount(sol_type, minlength=8).tolist(),
    "config": res.config, "metrics": metrics, "demo": demo,
}
blob = b64gz(json.dumps(data, separators=(",", ":")).encode())
mask_blob = b64gz(mask.tobytes())

sys.path.insert(0, str(HERE))
import page  # noqa: E402

page._SCHEMES.update(SCHEMES)
html = page.render(facts, B, blob, mask_blob, (HERE / "page.css").read_text(), (HERE / "page.js").read_text())
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html)
print(f"{OUT} {OUT.stat().st_size / 1e6:.2f} MB  data {len(blob) / 1e3:.0f} kB  mask {len(mask_blob) / 1e3:.0f} kB")
print("checks", checks, "cons", facts["cons_counts"], "saved mismatch", saved_mismatch)
print({k: {kk: (round(v, 1) if isinstance(v, float) else v) for kk, v in r.items() if kk != "sim"} for k, r in metrics.items()})
print("orient", orient_checks)
