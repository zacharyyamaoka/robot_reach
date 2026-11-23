"""Build the robot_reach landing page: site/index.html plus its media in site/media/.

    uv run --project /home/bam/robot_reach --with markdown --with pillow python /home/bam/robot_reach/site/build_site.py

Inputs, all real (nothing on the page is hand-typed output):

- `out/site_media_raw/` — stills + `hero.webm` + `steps.json` recorded from the running viser viewer by
  `site/record_media.py` (headless Chrome, real canvas clicks).
- `out/site/<arm>/result.json` + `figures/` — the saved example results
  (`python -m robot_reach run all --out out/site`).
- `notebooks/reach_worked_example.ipynb` — the executed worked example; its cells and outputs are
  rendered as they ran.
- `docs/api.md` — the API reference, rendered into the page (one source).

Publishing (GitHub Pages): the page is static; `site/` IS the Pages root (index.html, media/, the notebook
copies). With a GitHub repo, a workflow running actions/upload-pages-artifact on `path: site` +
actions/deploy-pages publishes it; nothing is built in CI.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import markdown
from PIL import Image

from robot_reach.codes import CODE_COLORS, CODE_LABELS, LEVEL_COLORS, LEVEL_NAMES, Code

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
MEDIA = SITE / "media"
RAW = ROOT / "out" / "site_media_raw"  # WHY under out/ (gitignored): the raw 1440p PNGs + webm are ~9 MB
RESULTS = ROOT / "out" / "site"
NOTEBOOK = ROOT / "notebooks" / "reach_worked_example.ipynb"
SCRIPT = ROOT / "notebooks" / "reach_worked_example.py"
API_MD = ROOT / "docs" / "api.md"
GITHUB = "https://github.com/zacharyyamaoka/robot_reach"  # WHY a placeholder link: no GitHub repo exists yet (Zach's call)
ARMS = ("pendulum", "rr", "five_bar", "shoulder3", "ur5e")
TITLES = {"pendulum": "1-DOF pendulum", "rr": "2-DOF RR", "five_bar": "2-DOF five-bar", "shoulder3": "3-DOF shoulder", "ur5e": "6-DOF UR5e"}


def sh(*cmd: str) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


# ---------------------------------------------------------------------------------------------- media
def build_media() -> dict:
    MEDIA.mkdir(parents=True, exist_ok=True)
    stills = {}
    for png in sorted(RAW.glob("*.png")):
        dest = MEDIA / (png.stem + ".jpg")
        Image.open(png).convert("RGB").save(dest, quality=84, optimize=True)
        stills[png.stem] = dest.name
    webm = RAW / "hero.webm"
    if webm.exists():
        steps = json.loads((RAW / "steps.json").read_text())
        times = {s["step"]: s["t"] for s in steps if "t" in s}
        # WHY trim: the first seconds are the page loading; start just before the workspace still.
        start = max(0.0, times.get("01-ur5e-workspace", 5.0) - 0.5)
        end = times.get("end", 60.0)
        vf = "setpts=PTS/2,scale=1280:-2"
        sh("ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start}", "-to", f"{end}", "-i", str(webm), "-vf", vf, "-an",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30", "-preset", "slow", "-movflags", "+faststart", str(MEDIA / "hero.mp4"))
        sh("ffmpeg", "-y", "-loglevel", "error", "-ss", "1.2", "-i", str(MEDIA / "hero.mp4"), "-frames:v", "1", "-q:v", "4", str(MEDIA / "hero-poster.jpg"))
        sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(MEDIA / "hero.mp4"), "-vf",
           "fps=7,scale=720:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
           str(MEDIA / "hero.gif"))
    figs = {}
    for arm in ARMS:
        for png in sorted((RESULTS / arm / "figures").glob("*.png")):
            dest = MEDIA / f"fig-{arm}-{png.stem}.png"
            img = Image.open(png)
            if img.width > 1100:
                img = img.resize((1100, round(img.height * 1100 / img.width)), Image.LANCZOS)
            img.save(dest, optimize=True)
            figs.setdefault(arm, []).append(dest.name)
    return {"stills": stills, "figs": figs}


# ---------------------------------------------------------------------------------------------- pieces
def esc(s: str) -> str:
    return html.escape(str(s))


def summaries() -> dict:
    out = {}
    for arm in ARMS:
        f = RESULTS / arm / "result.json"
        if f.exists():
            out[arm] = json.loads(f.read_text())["summary"]
    return out


def arms_table(sums: dict) -> str:
    rows = []
    for arm in ARMS:
        s = sums.get(arm)
        if not s:
            continue
        r = s.get("reach")
        t = sum(s["timings_s"].values())
        if r:
            lv = r["points_by_level"]
            grid = f'{r["positions"]:,}' + (f' × {r["orientations"]} orientations' if r["orientations"] > 1 else "")
            levels = " · ".join(f"{LEVEL_NAMES[int(k)]} {v:,}" for k, v in lv.items() if v)
            cells = [grid, str(r["ik_branches"]), f'{r["ok_solutions"]:,}', levels]
        else:
            cells = ["— (workspace is a curve: the FK cloud)", "1", "—", "—"]
        rows.append(f"<tr><td><b>{esc(TITLES[arm])}</b></td><td>{s['dof']}</td><td>{esc(s['task'])}</td>"
                    + "".join(f"<td>{esc(c)}</td>" for c in cells) + f"<td>{t:.1f} s</td></tr>")
    head = "<tr><th>Arm</th><th>DOF</th><th>Task</th><th>Grid</th><th>IK branches</th><th>OK solutions</th><th>Points by level</th><th>Compute</th></tr>"
    return f'<div class="scroll"><table class="doc">{head}{"".join(rows)}</table></div>'


def legend() -> str:
    codes = "".join(
        f'<span class="sw"><i style="background:rgb{CODE_COLORS[c]}"></i>{esc(CODE_LABELS[c])}</span>'
        for c in Code if c != Code.PAD
    )
    levels = "".join(
        f'<span class="sw"><i style="background:rgb{LEVEL_COLORS[k]}"></i>{k} · {esc(v)}</span>' for k, v in LEVEL_NAMES.items() if k
    )
    return f'<p class="legend"><b>Levels</b> {levels}</p><p class="legend"><b>Codes</b> {codes}</p>'


def notebook_html() -> str:
    nb = json.loads(NOTEBOOK.read_text())
    out = []
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            # the notebook's own title + run block are the page's quickstart; skip the first cell
            if src.startswith("# robot_reach, worked"):
                continue
            out.append(f'<div class="nb-md">{markdown.markdown(src, extensions=["tables", "fenced_code"])}</div>')
            continue
        texts = []
        for o in cell.get("outputs", []):
            t = o.get("text") or o.get("data", {}).get("text/plain") or ""
            texts.append("".join(t))
        output = re.sub(r"\x1b\[[0-9;]*m", "", "".join(texts)).rstrip()
        out.append(f'<pre class="nb-in"><code>{esc(src)}</code></pre>')
        if output:
            out.append(f'<pre class="nb-out"><code>{esc(output)}</code></pre>')
    return "\n".join(out)


def api_html() -> str:
    text = API_MD.read_text()
    text = text.split("\n", 1)[1]  # the page has its own section heading
    text = text.replace("](../notebooks/reach_worked_example.py)", "](reach_worked_example.py)")
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    body = body.replace("<h2>robot_msgs", '<h2 id="msgs">robot_msgs')
    return re.sub(r"<(/?)h2", r"<\1h3", body)


def fig(name: str, alt: str, cls: str = "") -> str:
    return f'<img class="{cls}" src="media/{name}" alt="{esc(alt)}" loading="lazy">'


def figure(name: str, caption: str) -> str:
    return f'<figure class="shot">{fig(name, caption)}<figcaption>{caption}</figcaption></figure>'


# ---------------------------------------------------------------------------------------------- page
def build() -> Path:
    media = build_media()
    st, figs = media["stills"], media["figs"]
    sums = summaries()
    for f in (NOTEBOOK, SCRIPT):
        shutil.copy2(f, SITE / f.name)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    steps = json.loads((RAW / "steps.json").read_text()) if (RAW / "steps.json").exists() else []
    counter = next((s.get("counter") for s in steps if s.get("step") == "click"), None)
    ur = sums.get("ur5e", {}).get("reach", {})

    def s(key, caption):
        return figure(st[key], caption) if key in st else ""

    def armfig(arm, pattern, caption):
        # WHY a pattern: figure names follow the viewer's joint labels ("shoulder_j1_x_elbow_j2_slice"), which change
        names = [n for n in figs.get(arm, []) if re.search(pattern, n)]
        if not names:
            raise SystemExit(f"no {arm} figure matches {pattern!r}: {figs.get(arm)}")
        return figure(names[0], caption)

    hero = ""
    if (MEDIA / "hero.mp4").exists():
        hero = f"""<figure class="hero"><video autoplay muted loop playsinline controls poster="media/hero-poster.jpg">
<source src="media/hero.mp4" type="video/mp4"><img src="media/hero.gif" alt="the viewer, recorded"></video>
<figcaption>Recorded from the running viewer (headless Chrome, real mouse clicks, 2× speed): the UR5e's workspace by pose DOF,
a voxel clicked{f' ({esc(counter)} working IK solutions there)' if counter else ''}, <b>Next ▶</b> through its IK branches, the dexterous and
4-DOF workspaces, a cut, then the five-bar and its joint-space map. <a href="media/hero.gif">GIF version</a>.</figcaption></figure>"""

    page = TEMPLATE
    repl = {
        "@@SUB@@": f"BAM Reach v2 · numpy + robot_msgs at runtime, viser to look · {esc(commit)} · built {dt.date.today().isoformat()} · <a href=\"{GITHUB}\">source</a>",
        "@@HEADLINE@@": (
            "<b>Give it a robot's kinematics, get its workspace.</b> FK and IK as two pure functions, or a URDF / RobotDescription. "
            "For any DOF, from a 1-joint pendulum to a 6-DOF UR5e, it maps where the tool can go (and with how much wrist freedom), "
            "which joint postures work, and why the rest fail. Every IK branch is kept and re-checked. It computes once and saves plain arrays; "
            "viser shows them and recomputes live from Python as you click."
            + (f" UR5e: {ur.get('targets', 0):,} target poses × {ur.get('ik_branches', 0)} branches → {ur.get('ok_solutions', 0):,} verified solutions." if ur else "")
        ),
        "@@HERO@@": hero,
        "@@ARMS_TABLE@@": arms_table(sums),
        "@@LEGEND@@": legend(),
        "@@SHOTS_WORKSPACE@@": "".join([
            s("01-ur5e-workspace", "UR5e, coloured by pose DOF: pale = position only, deep blue = every orientation."),
            s("02-clicked", "Click a voxel: the arm is posed at the first working IK solution there."),
            s("03-branches", "<b>Next ▶</b> walks every (orientation, IK branch) that works at that point."),
            s("04-dexterous", "<b>Show → dexterous</b>: only the points reachable in every sampled orientation."),
            s("05-four-dof", "<b>Show → 4-DOF</b>: straight-down approach at every yaw (top-down picking)."),
            s("06-cut", "<b>Cut → y</b>: look inside the shell."),
        ]),
        "@@SHOTS_DOF@@": "".join([
            armfig("pendulum", r"torque_line", "1-DOF: the workspace is a circle, so the picture is the joint line with gravity torque vs the motor limit."),
            armfig("rr", r"shoulder.*elbow.*_slice", "2-DOF RR: joint space coloured by why a posture fails."),
            armfig("five_bar", r"motor_left.*motor_right.*_slice", "2-DOF five-bar: the grey island is where the linkage cannot close."),
            s("08-five-bar-clicked", "The five-bar in the viewer, a point clicked."),
            armfig("shoulder3", r"shoulder.*elbow.*_sweep", "3-DOF: <b>sweep</b>, each cell the fraction of the other joint's postures that work."),
            s("10-shoulder3", "The 3-DOF shoulder's reach map."),
            armfig("ur5e", r"joint_ranges", "6-DOF UR5e: each joint's limits vs the range the working solutions actually use."),
            armfig("ur5e", r"shoulder_lift.*elbow.*_sweep", "UR5e: shoulder-lift × elbow, swept over the other joints."),
        ]),
        "@@API@@": api_html(),
        "@@NOTEBOOK@@": notebook_html(),
    }
    for k, v in repl.items():
        page = page.replace(k, v)
    leftover = re.findall(r"@@[A-Z_]+@@", page)
    if leftover:
        raise SystemExit(f"unfilled placeholders: {leftover}")
    dest = SITE / "index.html"
    dest.write_text(page)
    for f in MEDIA.iterdir():  # only what the page shows is published
        if f.is_file() and f"media/{f.name}" not in page:
            f.unlink()
    return dest


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>robot_reach</title>
<meta name="description" content="Kinematics in, workspace out: reachability of any arm in Cartesian and joint space from FK and IK (or a URDF / RobotDescription), computed once and explored live in viser.">
<style>
:root{--bg:#fbfbfa;--fg:#1f2328;--muted:#656d76;--line:#e4e4e2;--card:#fff;--accent:#2f6fde;--good:#3aa46b;--code:#f3f3f1;--hot-bg:#dcefe2;--hot-line:#3aa46b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#16181b;--fg:#e6e6e3;--muted:#9aa0a6;--line:#2c3036;--card:#1d2024;--accent:#7aa2f7;--good:#3f9e69;--code:#24272b;--hot-bg:#1e3a2a;--hot-line:#3f9e69}}
:root[data-theme="dark"]{--bg:#16181b;--fg:#e6e6e3;--muted:#9aa0a6;--line:#2c3036;--card:#1d2024;--accent:#7aa2f7;--good:#3f9e69;--code:#24272b;--hot-bg:#1e3a2a;--hot-line:#3f9e69}
*{box-sizing:border-box}
html{overflow-x:hidden}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;overflow-x:hidden}
main{max-width:1240px;margin:0 auto;padding:28px 16px 80px}
a{color:var(--accent)}
h1{font-size:28px;margin:0 0 4px} h2{font-size:21px;margin:44px 0 10px;padding-top:6px;border-top:1px solid var(--line)} h3{font-size:16px;margin:18px 0 6px}
.purpose{font-size:17px;margin:4px 0 6px;max-width:900px}
.sub{color:var(--muted);margin:0 0 14px;font-size:13px}
.toc{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13.5px;margin:0 0 16px}
.answer{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:8px;padding:14px 18px;font-size:15.5px}
.prose{max-width:960px} .prose p,.prose li{max-width:900px}
pre{margin:6px 0 12px;background:var(--code);border-radius:6px;padding:9px 11px;overflow-x:auto;font:12.5px/1.5 ui-monospace,Menlo,monospace;max-width:100%}
code{font:12.5px ui-monospace,Menlo,monospace;background:var(--code);padding:1px 4px;border-radius:4px;overflow-wrap:anywhere}
pre code{background:none;padding:0;overflow-wrap:normal}
.quick pre{font-size:13px}
.hero{margin:22px 0 0} .hero video{width:100%;border-radius:8px;border:1px solid var(--line);background:#000;display:block}
figcaption{color:var(--muted);font-size:13px;margin-top:6px}
.scroll{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;font-size:13px}
table.doc,.prose table{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:6px 0 12px}
table.doc th,table.doc td,.prose th,.prose td{padding:6px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
table.doc th,.prose th{color:var(--muted);font-weight:500;font-size:12.5px}
.prose table{display:block;overflow-x:auto;max-width:100%}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:18px;margin:12px 0}
.shot{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}
.shot img{width:100%;height:auto;display:block;border-radius:6px;background:#fff}
.legend{font-size:13px;display:flex;flex-wrap:wrap;gap:4px 14px;align-items:center;margin:4px 0}
.sw{display:inline-flex;align-items:center;gap:5px} .sw i{display:inline-block;width:13px;height:13px;border-radius:3px;border:1px solid var(--line)}
.cases{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px;margin:10px 0}
.case{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;font-size:14px}
.case h3{margin:0 0 6px;font-size:15px} .case p{margin:4px 0} .case .how{color:var(--muted);font-size:12.5px}
.loop{display:block;width:100%;max-width:1100px;height:auto;margin:6px 0 10px}
.loop .box{fill:var(--card);stroke:var(--line)} .loop .hot{fill:var(--hot-bg);stroke:var(--hot-line)}
.loop text{fill:var(--fg);font:12.5px -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif} .loop .small{fill:var(--muted);font-size:11px}
.loop .arrow{stroke:var(--muted);fill:none;marker-end:url(#arrowhead)} .loop #arrowhead path{fill:var(--muted)}
.nb-md{max-width:960px} .nb-md h2{font-size:18px;margin:30px 0 6px;border-top:none;padding-top:0}
pre.nb-in{border-left:3px solid var(--accent)} pre.nb-out{background:var(--card);border:1px solid var(--line);color:var(--fg)}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:10px 0}
details summary{cursor:pointer;font-weight:600}
.note{color:var(--muted);font-size:13px}
footer{color:var(--muted);font-size:12.5px;margin-top:40px}
@media (max-width:860px){h1{font-size:24px} .grid{grid-template-columns:1fr}}
</style></head><body><main>
<h1>robot_reach</h1>
<p class="purpose">Where can this arm go, and how well? Kinematics in (FK and IK, or a URDF / RobotDescription), workspace out: Cartesian and joint space, for any number of joints.</p>
<p class="sub">@@SUB@@</p>
<nav class="toc"><a href="#how">How it works</a><a href="#viewer">The viewer</a><a href="#dof">Any DOF</a><a href="#cases">Use cases</a><a href="#api">API</a><a href="#msgs">robot_msgs</a><a href="#example">Worked example</a><a href="#next">Next</a></nav>
<div class="answer">@@HEADLINE@@</div>
<div class="quick prose">
<pre><code>uv run --project /home/bam/robot_reach --extra viz python -m robot_reach examples</code></pre>
<pre><code>import robot_reach as rr
from robot_reach.robots import ur5e

result = rr.analyze(ur5e(), floor_z=0.0)   # FK cloud + reach map (every IK branch) + joint maps + joint ranges
result.save("out/ur5e")                     # the result log: npz + result.json, no pickle
rr.load("out/ur5e").display()               # viser; clicks recompute live in Python</code></pre>
</div>
@@HERO@@

<h2 id="how">How it works: kinematics is the centre</h2>
<svg class="loop" viewBox="0 0 1100 170" role="img" aria-label="RobotDescription, URDF or fk/ik functions become an Arm; analyze computes FK cloud, reach map, joint maps and ranges; the ReachResult is saved, viewed in viser, overlaid into a scene, or read downstream">
<defs><marker id="arrowhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z"/></marker></defs>
<rect class="box" x="4" y="8" width="170" height="40" rx="8"/><text x="89" y="32" text-anchor="middle">RobotDescription</text>
<rect class="box" x="4" y="58" width="170" height="40" rx="8"/><text x="89" y="82" text-anchor="middle">URDF file / XML</text>
<rect class="box" x="4" y="108" width="170" height="40" rx="8"/><text x="89" y="128" text-anchor="middle">fk(q), ik(T)</text><text class="small" x="89" y="142" text-anchor="middle">pure batched functions</text>
<rect class="hot" x="214" y="48" width="130" height="60" rx="8"/><text x="279" y="74" text-anchor="middle">Arm</text><text class="small" x="279" y="92" text-anchor="middle">FK · IK · limits · task</text>
<rect class="box" x="384" y="18" width="200" height="120" rx="8"/><text x="484" y="40" text-anchor="middle">analyze()</text>
<text class="small" x="398" y="62">1 FK cloud (any DOF)</text><text class="small" x="398" y="80">2 reach map: every IK branch</text><text class="small" x="398" y="98">3 joint maps (slice · sweep)</text><text class="small" x="398" y="116">4 joint ranges</text>
<rect class="hot" x="624" y="48" width="150" height="60" rx="8"/><text x="699" y="74" text-anchor="middle">ReachResult</text><text class="small" x="699" y="92" text-anchor="middle">save() / load(): npz + json</text>
<rect class="box" x="814" y="8" width="282" height="40" rx="8"/><text x="955" y="32" text-anchor="middle">display(): viser, live Python backend</text>
<rect class="box" x="814" y="58" width="282" height="40" rx="8"/><text x="955" y="82" text-anchor="middle">add_reach_map(): into your viser scene</text>
<rect class="box" x="814" y="108" width="282" height="40" rx="8"/><text x="955" y="128" text-anchor="middle">downstream: ReachMap arrays</text><text class="small" x="955" y="142" text-anchor="middle">levels · workspace(kind) · solutions(p)</text>
<path class="arrow" d="M174 28 C195 28 195 70 210 74"/><path class="arrow" d="M174 78 H210"/><path class="arrow" d="M174 128 C195 128 195 86 210 82"/>
<path class="arrow" d="M344 78 H380"/><path class="arrow" d="M584 78 H620"/>
<path class="arrow" d="M774 70 C795 60 795 28 810 28"/><path class="arrow" d="M774 78 H810"/><path class="arrow" d="M774 86 C795 96 795 128 810 128"/>
<text class="small" x="4" y="166">The library re-checks every IK claim (limits, FK round trip, duplicates, collision, torque, singularity): a lazy solver cannot overclaim.</text>
</svg>
<div class="prose">
<ul>
<li><b>Two directions, used together.</b> Forward: sample joints, run FK. That works for any DOF, even a pendulum whose workspace is a circle no grid would hit, and it bounds the grid so nobody guesses an extent. Inverse: a grid × one shared orientation set (22 approach directions × 6 rolls), IK at every target, <b>every branch kept</b>.</li>
<li><b>Levels count pose DOF.</b> 3 reachable · 4 straight down at any yaw · 5 any approach in the hemisphere · 6 every orientation. Four masks over one map.</li>
<li><b>One vocabulary of codes</b> for every picture, joint space and Cartesian alike: why a posture or a target fails.</li>
<li><b>Compute once, look any time.</b> The result log is plain arrays; viser, a notebook or another package reads the same numbers. In the viewer the Python backend recomputes on demand: <b>Compute box</b> maps a dragged box finer, <b>Drop POI</b> checks every orientation at one point, <b>Jog</b> shows the live code and torque.</li>
</ul>
</div>
@@LEGEND@@

<h2 id="viewer">The viewer</h2>
<div class="grid">@@SHOTS_WORKSPACE@@</div>

<h2 id="dof">Any DOF, one call</h2>
<p class="prose">The same <code>rr.analyze(arm)</code> on five arms. Numbers from the saved results that the viewer above loads.</p>
@@ARMS_TABLE@@
<div class="grid">@@SHOTS_DOF@@</div>

<h2 id="cases">Use cases</h2>
<div class="cases">
<div class="case"><h3>Pick from a conveyor belt</h3><p>Which points over the belt can be picked straight down at any yaw, and which from any approach? Intersect a <code>Box</code> with <code>workspace("four_dof")</code>, then map just that box finer.</p><p class="how">Worked example §5: UR5e, belt box 0.35–0.85 m out: 4-DOF on 98.5 % of it, hemisphere 66 %, dexterous 51 %.</p></div>
<div class="case"><h3>Sample valid targets</h3><p>Draw test or training targets only from a workspace you trust, with every IK branch that reaches them already solved and checked.</p><p class="how">§4: <code>workspace("hemisphere")</code>, <code>solutions(p)</code>, FK round trip 7e-8 m over 264 solutions.</p></div>
<div class="case"><h3>Any robot from its description</h3><p>A URDF or a RobotDescription in; every kinematic chain found and analysed, numeric multi-seed IK when there is no closed form.</p><p class="how">§6: a 3-DOF URDF, 4 distinct IK branches at one target; <code>python -m robot_reach urdf robot.urdf</code>.</p></div>
<div class="case"><h3>Place a robot in a cell</h3><p>Draw its reach map under the robot's frame in someone else's viser scene, so it moves when the robot is placed.</p><p class="how">§8: <code>add_reach_map(server, m, parent_frame="/cell/robot_1")</code>, viser 1.0.16 and 1.1.1.</p></div>
<div class="case"><h3>Choose a mechanism and its limits</h3><p>Joint maps say which postures fail and why (limit, singular, collision, torque, cannot close); joint ranges show what the work actually uses.</p><p class="how">The five-bar's cannot-close island; the pendulum's torque line.</p></div>
<div class="case"><h3>A ready / rest pose</h3><p>Footprint, clearance over the belt and holding power per pose; solve a use case to a Pareto set and a robot_msgs <code>NamedPose</code>.</p><p class="how"><code>python -m robot_reach rest shoulder3 --solve</code> (README).</p></div>
</div>

<h2 id="api">API (what exists today)</h2>
<div class="prose">@@API@@</div>

<h2 id="example">Worked example, executed</h2>
<p class="prose">Every cell below ran as shown; the outputs are the real ones. Download <a href="reach_worked_example.ipynb">the notebook</a> or <a href="reach_worked_example.py">the script</a> (the same cells). It runs against a local checkout (robot_reach and robot_msgs are not on PyPI yet), so it is a notebook, not a Colab link.</p>
<pre><code>uv run --project /home/bam/robot_reach --extra viz python /home/bam/robot_reach/notebooks/reach_worked_example.py</code></pre>
<div class="nb">@@NOTEBOOK@@</div>

<h2 id="next">Next</h2>
<div class="prose"><p>A typed downstream API is being proposed separately, not built: for example one frozen workspace value a trajectory generator could hold without importing robot_reach, with <code>contains</code> / <code>region</code> / <code>sample</code> verbs (a prototype outside <code>src/</code>, in <code>docs/proposals/workspace_api.py</code>; the write-up is section G of <code>reports/reach-proposals-2026-11-23.html</code> in the repo). Everything above this section is the code as it is.</p></div>

<footer>robot_reach (BAM Reach v2) · MIT · every number and picture on this page was produced by the code at the commit above.</footer>
</main></body></html>
"""


if __name__ == "__main__":
    print(build())
