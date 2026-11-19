"""Build the BAM Reach v2 review report: one self-contained HTML page (media inlined).

    .venv/bin/python docs/build_report.py

Reads the saved example results (out/<arm>/result.json + figures/), the viewer stills and the hero
video under reports/media/, and writes reports/media/bam-reach-v2-2026-11-19.html (media pages
live in the gitignored reports/media/).
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
MEDIA = ROOT / "reports" / "media"
DEST = MEDIA / "bam-reach-v2-2026-11-19.html"


def data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def img(path: Path, alt: str, cls: str = "") -> str:
    mime = "image/jpeg" if path.suffix == ".jpg" else "image/png"
    return f'<img class="{cls}" src="{data_uri(path, mime)}" alt="{html.escape(alt)}" loading="lazy">'


def _dur(seconds: float) -> str:
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.1f} s"


def summary(arm: str) -> dict:
    return json.loads((OUT / arm / "result.json").read_text())["summary"]


def fig(arm: str, name: str, alt: str) -> str:
    return img(OUT / arm / "figures" / f"{name}.png", alt)


ARMS = [
    ("pendulum", "1-DOF pendulum", "Its workspace is a circle, so the useful picture is the joint line: the motor (1.0 N·m) holds the 0.5 kg tip only within about 43° of vertical.",
     [("q1_torque_line", "joint line with gravity torque vs motor limit")], "17-pendulum"),
    ("rr", "2-DOF RR (shoulder + elbow)", "The classic 2R pair: joint space coloured by why a posture fails (floor collision red, torque orange), and the Cartesian plane coloured by how many IK branches work (1 light, 2 dark).",
     [("shoulder_x_elbow_slice", "RR joint-space map"), ("cartesian_plane_ik_solutions", "RR Cartesian plane by IK branches")], "11-rr-plane"),
    ("five_bar", "2-DOF five-bar (BAM's shoulder shape)", "The one 2-DOF arm whose joint picture has a hole: the grey island is where the distal links cannot meet. Purple bands are both singularity types with a 15° margin.",
     [("motor_left_x_motor_right_slice", "five-bar joint-space map"), ("cartesian_plane_ik_solutions", "five-bar plane by IK branches")], "15-five-bar"),
    ("shoulder3", "3-DOF shoulder (yaw + pitch + elbow)", "Four IK branches (reach forward or over the back × elbow up/down). The sweep equals the slice because yaw changes neither gravity nor the floor — your 'independent' intuition, measured. The purple diagonal is the tip crossing the yaw axis.",
     [("shoulder_x_elbow_sweep", "shoulder3 sweep map"), ("joint_ranges", "shoulder3 joint ranges")], "13-shoulder3"),
    ("ur5e", "6-DOF UR5e", "Eight closed-form branches at 905k targets. 1,955 reachable points, 1,396 with top-down 4-DOF, 827 hemisphere, 699 dexterous. In the shoulder-lift × elbow slice most failures are the floor (red).",
     [("shoulder_lift_x_elbow_slice", "UR5e joint slice"), ("joint_ranges", "UR5e joint ranges")], "04-ur5e-cut"),
]

ASKS = [
    ("Reachability in joint space and Cartesian space, fast and easy to see", "<code>rr.analyze()</code> does both; 1–3 DOF arms under 2 s, UR5e ~19 s"),
    ("3D pictures in viser", "<code>result.display()</code> on the house viewer (no logo, ViewCube)"),
    ("FK and IK as pure functions; consistency", "<code>Arm(fk=, ik=)</code>; the library re-checks limits, FK(IK) round trip, duplicates, collision, torque, singularity"),
    ("Multiple solutions; click through them", "every IK branch kept; click a voxel → <b>Solution</b> / <b>Next ▶</b> (real click proven in the video)"),
    ("xyz extent as points, coloured by wrist DOF", "voxels by <b>pose DOF</b>: 3 reachable · 4 top-down any yaw · 5 hemisphere · 6 dexterous"),
    ("The four workspaces", "masks over one map; <b>Show</b> dropdown"),
    ("Voxels; blobs that keep holes", "<b>Blob</b> = voxel boundary surface: holes and separate bodies kept; plus a <b>Cut</b> plane to look inside"),
    ("Compute once, view offline, <code>.display()</code>", "<code>save()</code> → pickle-free npz + json → <code>rr.load(...).display()</code>; the arm rebuilds from its recorded factory"),
    ("Table workspace at higher fidelity; your own orientation", "<b>Ask here → Compute box</b> with a finer step and an <b>Approach</b> dropdown"),
    ("Draw boxes / drop POIs and compute there", "<b>Ask here</b>: gizmo + <b>Compute box</b> / <b>Drop POI</b> (green/grey spike per approach)"),
    ("2R joint picture: solution / none / singular / collision; slice vs sum", "joint maps, slice and sweep, one colour code across every view"),
    ("Joint limits; torque limits", "both are checks; gravity torque vs motor limits → <code>TORQUE_LIMIT</code>"),
    ("A bar chart of every DOF's range", "<b>joint ranges</b>: limits vs what the working solutions use"),
    ("Examples for 1, 2, 3 and 6 DOF", "pendulum, RR, shoulder3, UR5e — plus the five-bar"),
]

CSS = """
:root{--bg:#ffffff;--fg:#1f2328;--muted:#59636e;--line:#d0d7de;--card:#f6f8fa;--ok:#2ea043;--accent:#225ea8}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--line:#30363d;--card:#161b22;--accent:#58a6ff}}
:root[data-theme="dark"]{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--line:#30363d;--card:#161b22;--accent:#58a6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 16px 80px}h1{font-size:28px;margin:0 0 6px}h2{font-size:20px;margin:40px 0 10px;padding-top:12px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:22px 0 6px}.lede{font-size:17px}.muted{color:var(--muted)}code{background:var(--card);padding:1px 5px;border-radius:4px;font-size:13px}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto;font-size:13px}
video,img{max-width:100%;border:1px solid var(--line);border-radius:8px;display:block;background:#fff}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-top:1px solid var(--line);padding:7px 8px;vertical-align:top;text-align:left}th{color:var(--muted);font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:1px 9px;font-size:12px;color:var(--muted);margin-right:6px}
ul{padding-left:20px}li{margin:3px 0}.ok{color:var(--ok);font-weight:600}
"""


def build() -> Path:
    s = {a: summary(a) for a, *_ in ARMS}
    hero = MEDIA / "hero2" / "hero.mp4"
    poster = MEDIA / "hero2" / "poster.jpg"
    stills = MEDIA / "stills"
    parts = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>BAM Reach v2</title><style>{CSS}</style></head><body><main>"]
    ur = s["ur5e"]["reach"]
    parts.append(
        "<p class=muted>BAM Reach · 2026-11-19 · Claude Opus 5.5</p><h1>BAM Reach is first class again</h1>"
        "<p class=lede><b>Reachability of any arm in joint space and Cartesian space, from FK and IK passed in as pure functions — computed once, viewed in viser, every IK solution clickable.</b> "
        f"Lives in <code>/home/bam/robot_reach</code> (<code>main</code>, local repo). UR5e: {ur['targets']:,} targets × {ur['ik_branches']} branches → {ur['ok_solutions']:,} valid solutions in ~19 s.</p>"
        "<pre>uv run --project /home/bam/robot_reach python -m robot_reach examples</pre>"
    )
    parts.append(
        "<h2>Hero: the real viewer</h2><p class=muted>Recorded from the running viewer in headless Chrome (1.7× speed). A real mouse click in the 3-D canvas selects a UR5e point (1 / 444 working orientation × branch pairs); Next ▶ steps through them, the camera orbits, the cut plane opens the map, then the five-bar is clicked and posed. Log: <code>reports/media/hero2/record.log</code>.</p>"
        f'<video controls muted loop autoplay playsinline poster="{data_uri(poster, "image/jpeg")}" src="{data_uri(hero, "video/mp4")}"></video>'
    )
    rows = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in ASKS)
    parts.append(f"<h2>What you asked on Nov 19 → where it is</h2><table><tr><th>You asked</th><th>Now</th></tr>{rows}</table>")
    parts.append("<h2>The viewer, state by state</h2><p class=muted>The orange 'Software WebGL rendering' toast in these stills is the headless test browser's software renderer; it does not appear on a GPU machine.</p><div class=grid>")
    for name, cap in [("01-ur5e-selected", "Click a voxel: voxels dim, the arm takes solution 2 of 816 at a dexterous point"),
                      ("04-ur5e-cut", "Cut y: the far half stays, so you look into the dexterous core (deep blue)"),
                      ("07-ur5e-poi", "Drop POI: every orientation checked at one point; a green spike per approach that works"),
                      ("18-responsive-900x1000", "900 px wide: panel still readable")]:
        parts.append(f"<div class=card>{img(stills / f'{name}.jpg', cap)}<p class=muted>{cap}</p></div>")
    parts.append("</div>")
    parts.append("<h2>The five examples</h2>")
    for arm, title, text, figs, still in ARMS:
        sm = s[arm]
        pills = f"<span class=pill>{sm['dof']} DOF</span><span class=pill>{sm['task']}</span><span class=pill>{_dur(sum(sm['timings_s'].values()))}</span>"
        if "reach" in sm:
            pills += f"<span class=pill>{sm['reach']['ok_solutions']:,} solutions</span><span class=pill>max {sm['reach']['max_solutions_at_one_pose']} at one point</span>"
        parts.append(f"<h3>{title}</h3><p>{pills}</p><p>{text}</p><div class=grid>")
        for name, alt in figs:
            parts.append(f"<div class=card>{fig(arm, name, alt)}</div>")
        parts.append(f"<div class=card>{img(stills / f'{still}.jpg', title + ' in the viewer')}<p class=muted>in the viewer</p></div></div>")
    parts.append(
        "<h2>The API</h2><pre>import robot_reach as rr\nfrom robot_reach.robots import ur5e\n\nresult = rr.analyze(ur5e(), floor_z=0.0)   # FK cloud, reach map, joint maps, joint ranges\nresult.save(\"out/ur5e\")                     # the result log\nrr.load(\"out/ur5e\").display()               # viser, any time later\n\n"
        "arm = rr.Arm(name, joint_names, lower, upper,\n             fk=my_fk,   # (N, dof) -> (N, 4, 4)\n             ik=my_ik,   # (N, 4, 4) -> (q (N, S, dof), code (N, S)): every branch\n             task=\"pose\")  # or \"axis\" / \"position\"\n\n"
        "from robot_reach.overlay import add_reach_map    # draw into someone else's viser scene\noverlay = add_reach_map(server, result.reach, parent_frame=\"/facility/robot_1\")\noverlay.kind = \"four_dof\"</pre>"
    )
    parts.append(
        "<h2>Decision surface</h2>"
        "<h3>Done and proved</h3><ul>"
        "<li><span class=ok>FK/IK round trip</span> on all five arms: every OK branch lands back on its target (≤1e-6), and some branch recovers the joints that made the pose in 100 % of 500 random samples each.</li>"
        "<li><span class=ok>Independent oracle:</span> UR5e FK matches the stdlib ur_kin.cpp port in bam_ws on 50 random postures (1e-9).</li>"
        "<li><span class=ok>Tests:</span> 38 pass (core, viewer, URDF adapter); a mutation that drops the joint-limit re-check turns the suite red. The URDF tests were written by the builder subagent but check against independent closed-form UR5e and RR solvers.</li>"
        "<li><span class=ok>Any URDF / RobotDescription:</span> UR5e from a plain URDF with numeric IK — 100 % solved, 4.5 distinct branches per target, each matching a closed-form branch; quick map 131k targets in ~75 s. Ran on a real <code>BamFbArmCad</code> description (3 chains).</li>"
        "<li><span class=ok>Real browser:</span> headless-viser review PASS (run 04, 19 states, 1440 + 900 px), after fixing three defects found in runs 01–03 (arm hidden in opaque voxels; cut kept the wrong half; camera not reframed on arm switch). A real canvas click selects and poses the arm (hero video).</li></ul>"
        "<h3>Decided for you (each undoable)</h3><ul>"
        "<li><b>New repo <code>robot_reach</code></b> instead of editing bam_reach v1 in place: v2 changes the data model (all branches, batched) and v1 still runs the workcell studio and dog-leg campaign. Undo: fold it back into bam_ws; v1 is untouched.</li>"
        "<li><b>Name <code>robot_reach</code></b> (your note's alias 'Robot Reachability'; fits robot_msgs / robot_models). Undo: rename.</li>"
        "<li><b>Levels count pose DOF</b> (3/4/5/6) as the colour for your 'number of wrist DOFs'.</li>"
        "<li><b>Voxel boundary surface for blobs</b>, not convex hulls — keeps holes and separate bodies, zero dependencies.</li>"
        "<li><b>Five-bar added</b> as a fifth example; illustrative dimensions, not bam_fb_v2.</li>"
        "<li><b>Skeleton drawing</b>, not meshes, for the examples.</li></ul>"
        "<h3>Left</h3><ul>"
        "<li><b>Wire it into the one-click flow</b> (your full-pipeline steps 5–6): the seams exist — <code>analyze_chains(desc)</code> (any RobotDescription or URDF, every kinematic chain, numeric multi-seed IK), <code>add_reach_map(server, …)</code> for a facility scene, <code>python -m robot_reach urdf PATH</code>. The Robotics IDE workflow agent is building the composer that calls them.</li>"
        "<li>Point the five-bar / dog-leg examples at BAM's measured geometry; meshes in the viewer.</li>"
        "<li>Not built: swept-path and scene-mesh collision, torque with payload/dynamics, GPU batch IK.</li></ul>"
        "<h3>Needs you</h3><ul><li>Nothing blocking. Optional: a GitHub remote for <code>robot_reach</code> (I did not create one).</li></ul>"
        "<h3>Deliberately not done</h3><ul><li>No audit (you didn't ask). No changes to bam_reach v1. No public push.</li></ul>"
    )
    parts.append("</main></body></html>")
    DEST.write_text("".join(parts))
    return DEST


if __name__ == "__main__":
    path = build()
    print(path, f"{path.stat().st_size / 1e6:.2f} MB")
