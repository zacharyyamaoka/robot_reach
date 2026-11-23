"""Build the robot_reach landing page: site/index.html (media in site/media/).

    uv run --project /home/bam/robot_reach --with markdown --with pillow --with pygments python /home/bam/robot_reach/site/build_site.py

Flags: `--media` re-makes site/media/ from the raw recording (`out/site_media_raw/`) and the saved results' figures
(`out/site/<arm>/figures/`); without it the committed media are reused as they are, so a copy-only rebuild never
re-encodes the video. `--run` re-executes `site/quickstart.py` and keeps its printed output in `site/quickstart.out`.
`site/record_media.py` runs all of it after a fresh recording.

Inputs, all real (nothing on the page is hand-typed output):

- `site/media/` (or, with --media, `out/site_media_raw/` from `site/record_media.py`): stills + the hero video,
  recorded from the running viser viewer with real canvas clicks.
- `out/site/<arm>/result.json`: the saved example results (`python -m robot_reach run all --out out/site`).
- `site/quickstart.py` + `site/quickstart.out`: the happy path, run as shown.
- `notebooks/reach_worked_example.ipynb`: the executed worked example; cells and outputs rendered as they ran.
- `docs/api.md`: the API reference, rendered into the page (one source).

Look: pidtuna.com's "Yellowfin" landing (pidtuna-website `site/.vitepress/theme/style.css` tokens and
`components/Landing.vue` layout, at 28c1f9b). WHY a copy and not an import: that look lives inside a VitePress
Vue theme with no published package, and this page is plain static HTML served from `site/` (GitHub Pages), so
there is nothing it could import. The copied rules sit in one block (`PT_CSS`) under the same `--pt-*` names, so a
re-sync is a paste.

Publishing (GitHub Pages): the page is static; `site/` IS the Pages root (index.html, media/, the notebook
copies). With a GitHub repo, a workflow running actions/upload-pages-artifact on `path: site` +
actions/deploy-pages publishes it; nothing is built in CI.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import BashLexer, PythonLexer

from robot_reach.codes import LEVEL_COLORS, LEVEL_NAMES

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
MEDIA = SITE / "media"
RAW = ROOT / "out" / "site_media_raw"  # WHY under out/ (gitignored): the raw 1440p PNGs + webm are ~9 MB
RESULTS = ROOT / "out" / "site"
NOTEBOOK = ROOT / "notebooks" / "reach_worked_example.ipynb"
SCRIPT = ROOT / "notebooks" / "reach_worked_example.py"
QUICKSTART = SITE / "quickstart.py"
QUICKSTART_OUT = SITE / "quickstart.out"
API_MD = ROOT / "docs" / "api.md"
GITHUB = "https://github.com/zacharyyamaoka/robot_reach"  # WHY not linked yet: no GitHub repo exists (Zach's call)
ARMS = ("pendulum", "rr", "five_bar", "shoulder3", "ur5e")
TITLES = {"pendulum": "Pendulum", "rr": "RR arm", "five_bar": "Five-bar (closed chain)", "shoulder3": "3-joint shoulder", "ur5e": "UR5e"}
TASKS = {"position": "position", "axis": "position + approach axis", "pose": "full pose"}
# WHY plain-word level descriptions keyed by number, names from LEVEL_NAMES: the viewer's level names are being renamed;
# what each level MEANS (how much of the pose the arm controls there) is not.
LEVEL_MEANING = {3: "position only", 4: "straight down, any yaw", 5: "any approach in the lower hemisphere", 6: "every orientation"}


def sh(*cmd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


# ---------------------------------------------------------------------------------------------- media
def build_media() -> None:
    """Raw recording + saved figures -> site/media/ (only with --media)."""
    from PIL import Image

    MEDIA.mkdir(parents=True, exist_ok=True)
    pngs = sorted(RAW.glob("*.png"))
    if pngs:  # WHY clear first: a still the new recording skipped must not survive from the old one
        for old in MEDIA.glob("*.jpg"):
            old.unlink()
    for png in pngs:
        Image.open(png).convert("RGB").save(MEDIA / (png.stem + ".jpg"), quality=84, optimize=True)
    webm = RAW / "hero.webm"
    if webm.exists():
        steps = json.loads((RAW / "steps.json").read_text())
        times = {s["step"]: s["t"] for s in steps if "t" in s}
        # WHY trim: the first seconds are the page loading; start just before the workspace still.
        start = max(0.0, times.get("01-ur5e-workspace", 5.0) - 0.5)
        end = times.get("end", 60.0)
        sh("ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start}", "-to", f"{end}", "-i", str(webm), "-vf", "setpts=PTS/2,scale=1280:-2", "-an",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30", "-preset", "slow", "-movflags", "+faststart", str(MEDIA / "hero.mp4"))
        sh("ffmpeg", "-y", "-loglevel", "error", "-ss", "1.2", "-i", str(MEDIA / "hero.mp4"), "-frames:v", "1", "-q:v", "4", str(MEDIA / "hero-poster.jpg"))
        sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(MEDIA / "hero.mp4"), "-vf",
           "fps=7,scale=720:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
           str(MEDIA / "hero.gif"))
        shutil.copy2(RAW / "steps.json", MEDIA / "steps.json")
    for arm in ARMS:
        for png in sorted((RESULTS / arm / "figures").glob("*.png")):
            img = Image.open(png)
            if img.width > 1100:
                img = img.resize((1100, round(img.height * 1100 / img.width)), Image.LANCZOS)
            img.save(MEDIA / f"fig-{arm}-{png.stem}.png", optimize=True)


def media_index() -> tuple[dict, dict]:
    stills = {p.stem: p.name for p in MEDIA.glob("*.jpg") if not p.stem.startswith("hero")}
    figs: dict[str, list[str]] = {}
    for arm in ARMS:
        figs[arm] = sorted(p.name for p in MEDIA.glob(f"fig-{arm}-*.png"))
    return stills, figs


# ---------------------------------------------------------------------------------------------- pieces
def esc(s) -> str:
    return html.escape(str(s))


def code(src: str, lang: str = "python") -> str:
    lexer = PythonLexer() if lang == "python" else BashLexer()
    body = highlight(src.rstrip("\n"), lexer, HtmlFormatter(nowrap=True))
    return f'<div class="code"><span class="lang">{lang}</span><pre><code>{body}</code></pre></div>'


def summaries() -> dict:
    out = {}
    for arm in ARMS:
        f = RESULTS / arm / "result.json"
        if f.exists():
            out[arm] = json.loads(f.read_text())["summary"]
    return out


def rgb(level: int) -> str:
    return "rgb({},{},{})".format(*LEVEL_COLORS[level])


def arms_table(sums: dict) -> str:
    """pidtuna's proof table, re-cast: one row per arm, a stacked bar of its grid by level."""
    rows = []
    for arm in ARMS:
        s = sums.get(arm)
        if not s:
            continue
        r = s.get("reach")
        t = sum(s["timings_s"].values())
        name = (f'<span class="name" role="cell"><strong>{esc(TITLES.get(arm, arm))}</strong>'
                f'<small>{s["dof"]} {"joint" if s["dof"] == 1 else "joints"} · {esc(TASKS.get(s["task"], s["task"]))}</small></span>')
        if r:
            by = {int(k): v for k, v in r["points_by_level"].items()}
            segs = "".join(
                f'<i style="width:{100 * v / r["positions"]:.2f}%;background:{rgb(k)}" title="{esc(LEVEL_NAMES[k])}: {v:,} points"></i>'
                for k, v in sorted(by.items()) if v)
            reach_n = sum(by.values())
            grid = f'{r["positions"]:,}' + (f'<small>× {r["orientations"]}</small>' if r["orientations"] > 1 else "")
            cells = (f'<span class="num" role="cell">{grid}</span><span class="num" role="cell">{r["ik_branches"]}</span>'
                     f'<span class="num strong" role="cell">{r["ok_solutions"]:,}</span>'
                     f'<span class="bars" role="cell" title="{reach_n:,} of {r["positions"]:,} grid points reachable">{segs}</span>')
        else:
            cells = ('<span class="num" role="cell">–</span><span class="num" role="cell">1</span><span class="num" role="cell">–</span>'
                     '<span class="bars none" role="cell">a curve, not a volume: the FK cloud is the workspace</span>')
        rows.append(f'<div class="tr" role="row">{name}{cells}<span class="num dim" role="cell">{t:.1f} s</span></div>')
    head = ('<div class="tr head" role="row"><span role="columnheader">Robot</span><span role="columnheader" class="num">Grid</span>'
            '<span role="columnheader" class="num" title="IK configurations per target (elbow up / down, wrist flip …)">IK&nbsp;cfg</span>'
            '<span role="columnheader" class="num">Verified</span><span role="columnheader" class="bars">Grid points by level</span>'
            '<span role="columnheader" class="num">Compute</span></div>')
    return f'<div class="table" role="table" aria-label="The same analyze() on five arms">{head}{"".join(rows)}</div>'


def legend() -> str:
    return "".join(
        f'<span class="sw"><i style="background:{rgb(k)}"></i><b>{esc(LEVEL_NAMES[k])}</b>&nbsp;{esc(LEVEL_MEANING.get(k, ""))}</span>'
        for k in LEVEL_NAMES if k)


def notebook_html() -> tuple[str, int]:
    nb = json.loads(NOTEBOOK.read_text())
    out, n = [], 0
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            if src.startswith("# robot_reach, worked"):  # its title + run block are this section's own header
                continue
            out.append(f'<div class="nb-md">{markdown.markdown(src, extensions=["tables", "fenced_code"])}</div>')
            continue
        n += 1
        texts = []
        for o in cell.get("outputs", []):
            t = o.get("text") or o.get("data", {}).get("text/plain") or ""
            texts.append("".join(t))
        output = re.sub(r"\x1b\[[0-9;]*m", "", "".join(texts)).rstrip()
        out.append(code(src))
        if output:
            out.append(f'<pre class="out"><code>{esc(output)}</code></pre>')
    return "\n".join(out), n


def api_html() -> str:
    text = API_MD.read_text().split("\n", 1)[1]  # the page has its own heading
    text = text.replace("](../notebooks/reach_worked_example.py)", "](reach_worked_example.py)")
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    body = body.replace("<h2>robot_msgs", '<h2 id="msgs">robot_msgs')
    return re.sub(r"<(/?)h2", r"<\1h3", body)


def figure(name: str, caption: str) -> str:
    return (f'<figure class="shot"><a href="media/{name}"><img src="media/{name}" alt="{esc(re.sub("<[^>]+>", "", caption))}" loading="lazy"></a>'
            f"<figcaption>{caption}</figcaption></figure>")


MARK = """<svg class="mark" viewBox="0 0 64 48" role="img" aria-label="robot_reach">
<path d="M6 44 A38 38 0 0 1 44 6" fill="none" stroke="var(--pt-logo-fin)" stroke-width="2.4" stroke-dasharray="3.2 3.6" stroke-linecap="round"/>
<path d="M4 46 H24" stroke="var(--pt-logo-body)" stroke-width="4" stroke-linecap="round"/>
<path d="M14 44 L22 24 L40 16" fill="none" stroke="var(--pt-logo-body)" stroke-width="5.5" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="14" cy="44" r="4.2" fill="var(--pt-logo-body)"/><circle cx="22" cy="24" r="3.4" fill="var(--pt-logo-belly)" stroke="var(--pt-logo-body)" stroke-width="2"/>
<circle cx="44" cy="14.4" r="4.4" fill="var(--pt-logo-fin)"/></svg>"""


# ---------------------------------------------------------------------------------------------- page
def build(media: bool = False, run: bool = False) -> Path:
    if media:
        build_media()
    if run or not QUICKSTART_OUT.exists():
        res = subprocess.run(["taskset", "-c", "2-5,16,17", str(ROOT / ".venv/bin/python"), str(QUICKSTART)], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        QUICKSTART_OUT.write_text(res.stdout)
    st, figs = media_index()
    sums = summaries()
    for f in (NOTEBOOK, SCRIPT):
        shutil.copy2(f, SITE / f.name)
    commit = sh("git", "-C", str(ROOT), "rev-parse", "--short", "HEAD").stdout.strip()
    ur = sums.get("ur5e", {}).get("reach", {})

    def still(keys, caption):
        # WHY several keys: record_media.py's still names change with the viewer (03-branches -> 03-ik-configurations)
        for k in keys if isinstance(keys, tuple) else (keys,):
            if k in st:
                return figure(st[k], caption)
        return ""

    def armfig(arm, pattern, caption):
        # WHY a pattern: figure names follow the viewer's joint labels ("shoulder_j1_x_elbow_j2_slice"), which change
        names = [n for n in figs.get(arm, []) if re.search(pattern, n)]
        return figure(names[0], caption) if names else ""

    hero = ""
    if (MEDIA / "hero.mp4").exists():
        hero = """<video autoplay muted loop playsinline controls poster="media/hero-poster.jpg"><source src="media/hero.mp4" type="video/mp4">
<img src="media/hero.gif" alt="the viewer, recorded"></video>"""
    notebook, n_cells = notebook_html()
    nb_out = QUICKSTART_OUT.read_text().rstrip()
    stats = (f"{ur.get('targets', 0):,} target poses on the UR5e, {ur.get('ok_solutions', 0):,} verified joint solutions" if ur else "")

    repl = {
        "@@MARK@@": MARK,
        "@@COMMIT@@": esc(commit),
        "@@DATE@@": dt.date.today().isoformat(),
        "@@HERO@@": hero,
        "@@STATS@@": esc(stats),
        "@@LEGEND@@": legend(),
        "@@QUICK@@": code(QUICKSTART.read_text().split('"""', 2)[2].lstrip()),
        "@@QUICK_OUT@@": esc(nb_out),
        "@@INSTALL@@": code("uv run --project /home/bam/robot_reach --extra viz python -m robot_reach examples", "bash"),
        "@@ARMS_TABLE@@": arms_table(sums),
        "@@SHOTS_VIEWER@@": "".join([
            still("01-ur5e-workspace", "The UR5e's workspace, coloured by level: how much of the pose the arm controls at each point."),
            still("02-clicked", "Click a point: the arm is posed at a working IK solution there, re-checked by FK."),
            still(("03-ik-configurations", "03-branches"), "<b>Next ▶</b> steps through every working orientation and IK configuration at that point."),
            still("04-dexterous", f"<b>Show → {esc(LEVEL_NAMES[6])}</b>: only the points reachable in every sampled orientation."),
            still(("05-top-down", "05-four-dof"), f"<b>Show → {esc(LEVEL_NAMES[4])}</b>: straight down at any yaw, the top-down picking workspace."),
            still("06-cut", "<b>Cut → y</b>: look inside the shell."),
        ]),
        "@@SHOTS_JOINT@@": "".join([
            armfig("pendulum", r"torque_line", "One joint: the workspace is a circle, so the picture is the joint line, gravity torque against the motor limit."),
            armfig("rr", r"shoulder.*elbow.*_slice", "RR arm: joint space coloured by why a posture fails."),
            armfig("five_bar", r"motor_left.*motor_right.*_slice", "Five-bar: the grey island is where the linkage cannot close."),
            still("08-five-bar-clicked", "The five-bar in the viewer, a point clicked."),
            armfig("shoulder3", r"shoulder.*elbow.*_sweep", "3 joints: a <b>sweep</b>, each cell the fraction of the other joint's postures that work."),
            still("10-shoulder3", "The 3-joint shoulder in the viewer."),
            armfig("ur5e", r"joint_ranges", "UR5e: each joint's limits against the range the working solutions actually use."),
            armfig("ur5e", r"shoulder_lift.*elbow.*_sweep", "UR5e: shoulder lift × elbow, swept over the other joints."),
        ]),
        "@@API@@": api_html(),
        "@@NOTEBOOK@@": notebook,
        "@@NB_CELLS@@": str(n_cells),
    }
    page = TEMPLATE.replace("@@PT_CSS@@", PT_CSS)
    for k, v in repl.items():
        page = page.replace(k, v)
    leftover = re.findall(r"@@[A-Z_]+@@", page)
    if leftover:
        raise SystemExit(f"unfilled placeholders: {leftover}")
    dest = SITE / "index.html"
    dest.write_text(page)
    for f in MEDIA.iterdir():  # only what the page shows is published
        if f.is_file() and f"media/{f.name}" not in page and f.name != "steps.json":
            f.unlink()
    return dest


# ---------------------------------------------------------------------------------------------- look
# Copied from pidtuna-website @ 28c1f9b (site/.vitepress/theme/style.css "1 · Yellowfin" + Landing.vue's scoped rules),
# same --pt-* token names. Light and dark exactly as pidtuna's; the hero panel is Yellowfin's deep-ocean card.
PT_CSS = r"""
:root{--pt-font-base:'Inter',ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;--pt-font-display:'Space Grotesk','Inter',ui-sans-serif,system-ui,sans-serif;--pt-font-mono:'JetBrains Mono',ui-monospace,'SFMono-Regular',Menlo,monospace;
--pt-bg:#ffffff;--pt-bg-soft:#f2f6fa;--pt-bg-deep:#0a1f33;--pt-on-deep:#e8f0f8;--pt-text:#0b1b2b;--pt-text-2:#3d5166;--pt-text-3:#6b7d90;--pt-line:#d9e2ec;
--pt-brand:#0f4c81;--pt-brand-2:#1763a6;--pt-brand-soft:rgba(15,76,129,.1);--pt-accent:#c98a00;--pt-accent-soft:rgba(229,165,0,.16);--pt-warn:#d6402b;
--pt-radius:14px;--pt-radius-sm:8px;--pt-border-w:1px;--pt-shadow:0 1px 2px rgba(10,31,51,.06),0 8px 24px rgba(10,31,51,.06);
--pt-display-weight:700;--pt-display-tracking:-0.025em;--pt-logo-body:#0f4c81;--pt-logo-belly:#c9d6e3;--pt-logo-fin:#f0ad00;
--c-kw:#7c2fa8;--c-str:#2a7a2a;--c-num:#b35c00;--c-com:#6b7d90;--c-fn:#1763a6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--pt-bg:#07131f;--pt-bg-soft:#0c1d2e;--pt-bg-deep:#040c15;--pt-text:#e6edf5;--pt-text-2:#a9b8c8;--pt-text-3:#7a8b9c;--pt-line:#1b3046;
--pt-brand:#5fb0ff;--pt-brand-2:#8cc6ff;--pt-brand-soft:rgba(95,176,255,.14);--pt-accent:#ffc21a;--pt-accent-soft:rgba(255,194,26,.14);--pt-warn:#ff6b57;
--pt-shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.25);--pt-logo-body:#3d8fe0;--pt-logo-fin:#ffc21a;
--c-kw:#c792ea;--c-str:#9ece6a;--c-num:#ff9e64;--c-com:#7a8b9c;--c-fn:#8cc6ff}}
:root[data-theme="dark"]{--pt-bg:#07131f;--pt-bg-soft:#0c1d2e;--pt-bg-deep:#040c15;--pt-text:#e6edf5;--pt-text-2:#a9b8c8;--pt-text-3:#7a8b9c;--pt-line:#1b3046;
--pt-brand:#5fb0ff;--pt-brand-2:#8cc6ff;--pt-brand-soft:rgba(95,176,255,.14);--pt-accent:#ffc21a;--pt-accent-soft:rgba(255,194,26,.14);--pt-warn:#ff6b57;
--pt-shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.25);--pt-logo-body:#3d8fe0;--pt-logo-fin:#ffc21a;
--c-kw:#c792ea;--c-str:#9ece6a;--c-num:#ff9e64;--c-com:#7a8b9c;--c-fn:#8cc6ff}
*{box-sizing:border-box}
html{overflow-x:hidden;scroll-padding-top:72px}
body{margin:0;background:var(--pt-bg);color:var(--pt-text);font:16px/1.6 var(--pt-font-base);overflow-x:hidden;-webkit-font-smoothing:antialiased}
a{color:var(--pt-brand)}
h1,h2{font-family:var(--pt-font-display);font-weight:var(--pt-display-weight);letter-spacing:var(--pt-display-tracking);line-height:1.05;margin:0}
/* nav: VitePress' bar */
.nav{position:sticky;top:0;z-index:10;background:var(--pt-bg);border-bottom:1px solid var(--pt-line)}
.nav-in{max-width:1376px;margin:0 auto;height:64px;padding:0 32px;display:flex;align-items:center;gap:24px}
.brand{display:flex;align-items:center;gap:10px;color:var(--pt-text);text-decoration:none;font-weight:700;font-size:16px;letter-spacing:-.01em}
.brand .mark{width:34px;height:auto}
.nav-links{margin-left:auto;display:flex;gap:24px}
.nav-links a{color:var(--pt-text);text-decoration:none;font-size:14px;font-weight:500}
.nav-links a:hover{color:var(--pt-brand)}
.theme{margin-left:8px;width:40px;height:22px;border-radius:11px;border:1px solid var(--pt-line);background:var(--pt-bg-soft);cursor:pointer;position:relative;padding:0}
.theme::after{content:"";position:absolute;top:2px;left:2px;width:16px;height:16px;border-radius:50%;background:var(--pt-bg);box-shadow:0 1px 2px rgba(0,0,0,.2);transition:transform .2s}
.is-dark .theme::after{transform:translateX(18px)}
/* landing */
.pt-landing{--gap:clamp(48px,7vw,96px);max-width:1180px;margin:0 auto;padding:24px 24px 96px}
section{margin-top:var(--gap)}
section.hero{margin-top:clamp(16px,4vw,48px)}
.section-head{max-width:760px;margin-bottom:24px}
.section-head h2{font-size:clamp(26px,3.4vw,38px)}
.section-head p{margin:10px 0 0;color:var(--pt-text-2);font-size:16px;line-height:1.6}
.hero{display:grid;grid-template-columns:minmax(0,.9fr) minmax(0,1.1fr);gap:clamp(24px,4vw,56px);align-items:center}
.eyebrow{margin:0 0 14px;color:var(--pt-accent);font-size:13px;font-weight:700;letter-spacing:.08em;text-transform:uppercase}
.wordmark{display:flex;align-items:center;gap:12px;font-size:clamp(38px,4.6vw,66px);min-width:0;overflow-wrap:anywhere}
.wordmark .mark{width:1.05em;height:auto;flex:none}
.lead{margin:22px 0 0;font-size:clamp(19px,2vw,23px);line-height:1.45;font-weight:600}
.sub{margin:14px 0 0;color:var(--pt-text-2);font-size:16px;line-height:1.6}
.actions{display:flex;flex-wrap:wrap;gap:10px;margin-top:28px}
.btn{display:inline-flex;align-items:center;padding:10px 18px;border:var(--pt-border-w) solid var(--pt-line);border-radius:var(--pt-radius-sm);color:var(--pt-text);font-weight:600;font-size:15px;text-decoration:none;transition:background .15s,border-color .15s}
.btn:hover{border-color:var(--pt-brand)}
.btn.primary{background:var(--pt-brand);border-color:var(--pt-brand);color:var(--pt-bg)}
.btn.primary:hover{background:var(--pt-brand-2)}
.btn.ghost{border-color:transparent;color:var(--pt-text-2);cursor:default;font-weight:500}
.hero-stage{position:relative;min-width:0;padding:14px;border-radius:calc(var(--pt-radius) + 6px);background:radial-gradient(120% 90% at 85% 0%,rgba(255,194,26,.12),transparent 55%),linear-gradient(160deg,#0f4c81 0%,#0a1f33 70%);box-shadow:var(--pt-shadow)}
.hero-stage video{display:block;width:100%;border-radius:var(--pt-radius-sm);background:#000}
.hero-stage .legend{color:#e8f0f8;margin:12px 2px 0}
.hero-stage .caption{color:rgba(232,240,248,.72);margin:8px 2px 0;font-size:13px;line-height:1.5}
.hero-stage .caption a{color:#ffc21a}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:13px}
.sw{display:inline-flex;align-items:center;gap:6px}
.sw i{display:inline-block;width:12px;height:12px;border-radius:3px;border:1px solid rgba(255,255,255,.35)}
/* steps strip */
.stages{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:0;padding:0;list-style:none}
.stages li{border:var(--pt-border-w) solid var(--pt-line);border-radius:var(--pt-radius);background:var(--pt-bg);transition:border-color .15s,transform .15s}
.stages li:hover{border-color:var(--pt-brand);transform:translateY(-2px)}
.stages li a{display:block;height:100%;padding:16px;color:inherit;text-decoration:none}
.stages li.out{background:var(--pt-brand-soft);border-color:var(--pt-brand)}
.stages .n{display:inline-grid;place-items:center;width:28px;height:28px;border-radius:50%;background:var(--pt-accent-soft);color:var(--pt-accent);font-family:var(--pt-font-mono);font-weight:700;font-size:14px}
.stages strong{display:block;margin-top:10px;font-size:16px}
.stages p{margin:6px 0 0;color:var(--pt-text-2);font-size:14px;line-height:1.5}
.stages code{font-size:12.5px}
.more{margin-top:14px;font-size:14px}
/* code */
code{font-family:var(--pt-font-mono);font-size:.875em;background:var(--pt-bg-soft);padding:1px 5px;border-radius:4px;overflow-wrap:anywhere}
.code{position:relative;margin:0 0 16px;background:var(--pt-bg-soft);border-radius:var(--pt-radius-sm)}
.code .lang{position:absolute;top:6px;right:12px;font-size:12px;color:var(--pt-text-3);font-family:var(--pt-font-base)}
.code pre,pre.out{margin:0;padding:20px 24px;overflow-x:auto;font:13.5px/1.7 var(--pt-font-mono)}
.code pre code,pre.out code{background:none;padding:0;font-size:inherit;overflow-wrap:normal}
pre.out{margin:-8px 0 16px;background:var(--pt-bg);border:1px solid var(--pt-line);border-radius:var(--pt-radius-sm);color:var(--pt-text-2);font-size:12.5px;padding:12px 24px}
.code .k,.code .kn,.code .kd,.code .kc,.code .ow,.code .nb{color:var(--c-kw)}
.code .s,.code .s1,.code .s2,.code .sd,.code .si,.code .sa{color:var(--c-str)}
.code .m,.code .mi,.code .mf{color:var(--c-num)}
.code .c,.code .c1,.code .cm{color:var(--c-com);font-style:italic}
.code .nf,.code .fm{color:var(--c-fn)}
.quick{max-width:860px}
/* proof table */
.table{border:var(--pt-border-w) solid var(--pt-line);border-radius:var(--pt-radius);overflow:hidden}
.tr{display:grid;grid-template-columns:minmax(170px,1.4fr) 92px 64px 104px minmax(140px,2fr) 76px;gap:12px;align-items:center;padding:11px 16px;border-top:1px solid var(--pt-line)}
.tr.head{border-top:0;background:var(--pt-bg-soft);color:var(--pt-text-3);font-size:12px;font-weight:700;letter-spacing:.06em;text-transform:uppercase}
.name strong{display:block;font-size:15px}
.name small{color:var(--pt-text-3);font-size:12px}
.num{text-align:right;font-family:var(--pt-font-mono);font-variant-numeric:tabular-nums;font-size:14px}
.tr.head .num{font-family:var(--pt-font-base)}
.num small{color:var(--pt-text-3);margin-left:4px;font-size:12px}
.num.strong{font-weight:700}
.num.dim{color:var(--pt-text-3)}
.bars{display:flex;height:10px;border-radius:3px;overflow:hidden;background:var(--pt-bg-soft);border:1px solid var(--pt-line)}
.bars i{display:block;height:100%}
.tr.head .bars,.bars.none{height:auto;background:none;border:0;border-radius:0}
.bars.none{color:var(--pt-text-3);font-size:12.5px;line-height:1.4}
.footnote{margin-top:14px;max-width:820px;color:var(--pt-text-2);font-size:14px;line-height:1.6}
.footnote .legend .sw i{border-color:var(--pt-line)}
/* figures */
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
.shot{margin:0;border:var(--pt-border-w) solid var(--pt-line);border-radius:var(--pt-radius);overflow:hidden;background:var(--pt-bg);transition:border-color .15s}
.shot:hover{border-color:var(--pt-brand)}
.shot img{display:block;width:100%;height:auto;background:#fff;border-bottom:1px solid var(--pt-line)}
.shot figcaption{padding:10px 14px 12px;color:var(--pt-text-2);font-size:13.5px;line-height:1.5}
/* promises */
.promises{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
.promise{padding:20px;border-top:3px solid var(--pt-accent);background:var(--pt-bg-soft);border-radius:var(--pt-radius-sm)}
.promise strong{font-size:16px}
.promise p{margin:8px 0 0;color:var(--pt-text-2);font-size:14px;line-height:1.55}
.promise .how{color:var(--pt-text-3);font-size:13px}
/* long reference blocks */
details.fold{border:var(--pt-border-w) solid var(--pt-line);border-radius:var(--pt-radius);padding:0 20px}
details.fold>summary{cursor:pointer;padding:16px 0;font-weight:600;color:var(--pt-brand);list-style-position:inside}
details.fold[open]>summary{border-bottom:1px solid var(--pt-line);margin-bottom:16px}
.doc{max-width:900px;font-size:15px}
.doc h3{font-family:var(--pt-font-display);font-size:21px;margin:32px 0 8px;letter-spacing:-.01em}
.doc h2{font-family:var(--pt-font-display);font-size:22px;margin:36px 0 8px}
.doc table{display:block;overflow-x:auto;max-width:100%;border-collapse:collapse;font-size:13.5px;margin:8px 0 16px}
.doc th,.doc td{padding:7px 10px;border:1px solid var(--pt-line);text-align:left;vertical-align:top}
.doc th{background:var(--pt-bg-soft)}
.doc pre{background:var(--pt-bg-soft);border-radius:var(--pt-radius-sm);padding:16px 20px;overflow-x:auto;font:13px/1.6 var(--pt-font-mono)}
.doc pre code{background:none;padding:0;overflow-wrap:normal}
.nb-md{max-width:900px}
.cta{display:flex;align-items:center;gap:16px;padding-top:32px;border-top:1px solid var(--pt-line);color:var(--pt-text-2)}
.cta .mark{width:56px;flex:none}
.cta p{margin:0;font-size:16px}
footer{border-top:1px solid var(--pt-line);padding:24px;text-align:center;color:var(--pt-text-2);font-size:14px}
@media (max-width:960px){.hero{grid-template-columns:1fr}.stages{grid-template-columns:repeat(2,minmax(0,1fr))}.promises{grid-template-columns:1fr}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.nav-links{display:none}.theme{margin-left:auto}}
@media (max-width:640px){.nav-in{padding:0 16px}.pt-landing{padding:12px 16px 80px}
.tr{grid-template-columns:minmax(0,1fr) auto;gap:8px;padding:10px 12px}
.tr>:nth-child(2),.tr>:nth-child(3),.tr>:nth-child(6){display:none}.tr .bars{grid-column:1/-1}.tr.head .bars{display:none}
.stages,.grid{grid-template-columns:1fr}.code pre,pre.out{padding:16px}.cta{flex-direction:column;align-items:flex-start}details.fold{padding:0 14px}}
"""

TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>robot_reach</title>
<meta name="description" content="Robot kinematics for any number of joints: a RobotDescription in, FK and IK out, and the workspace they reach, sampled and explored live in viser.">
<meta name="theme-color" content="#0f3b5f">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@500;700&display=swap">
<script>try{var t=localStorage.getItem('rr-theme');if(t)document.documentElement.dataset.theme=t}catch(e){}</script>
<style>@@PT_CSS@@</style></head><body>
<header class="nav"><div class="nav-in">
<a class="brand" href="#top">@@MARK@@<span>robot_reach</span></a>
<nav class="nav-links"><a href="#happy-path">Quickstart</a><a href="#any-dof">Any DOF</a><a href="#viewer">Viewer</a><a href="#example">Worked example</a><a href="#api">API</a></nav>
<button class="theme" type="button" aria-label="Toggle dark mode" title="Toggle dark mode"></button>
</div></header>
<main class="pt-landing" id="top">

<section class="hero">
<div class="hero-copy">
<p class="eyebrow">Robot kinematics · open source, MIT</p>
<h1 class="wordmark">@@MARK@@<span>robot_reach</span></h1>
<p class="lead">A robot description in; FK, IK and the workspace they reach out. Any number of joints, live in viser.</p>
<p class="sub">Load a RobotDescription, a URDF or your own two functions. robot_reach maps where the tool can go and with how much
of the pose, keeps every IK solution it finds, re-checks each one, and lets you sample targets from the workspace you trust.
Compute once, save plain arrays, click around in viser while Python recomputes.</p>
<div class="actions">
<a class="btn primary" href="#happy-path">Quickstart</a>
<a class="btn" href="#viewer">See the viewer</a>
<a class="btn" href="#any-dof">Five robots, one call</a>
<a class="btn" href="#example">Worked example</a>
<span class="btn ghost" title="The repository is not public yet">Source: coming soon</span>
</div>
</div>
<div class="hero-stage">
@@HERO@@
<p class="legend">@@LEGEND@@</p>
<p class="caption">Recorded from the running viewer in headless Chrome, real mouse clicks, 2× speed: the UR5e's workspace, a point
clicked and its IK solutions stepped through, the workspace levels, a cut, then the five-bar and its joint space.
<a href="media/hero.gif">GIF</a></p>
</div>
</section>

<section id="happy-path">
<div class="section-head"><h2>Load a robot, get its workspace, sample it</h2>
<p>Kinematics is the centre: an arm is its forward and inverse kinematics. Everything else (the workspace, the
joint-space maps, the viewer) is computed from those two functions.</p></div>
<ol class="stages">
<li><a href="#quick"><span class="n">1</span><strong>Load a robot</strong><p>A <code>RobotDescription</code>, a URDF, or your own <code>fk(q)</code> and <code>ik(T)</code> as batched numpy. Numeric multi-seed IK when there is no closed form.</p></a></li>
<li><a href="#quick"><span class="n">2</span><strong>Workspace</strong><p><code>rr.analyze(arm)</code>: FK sizes the space, IK runs at every point × orientation and keeps every solution. Each point gets a level: how much of the pose the arm controls there.</p></a></li>
<li><a href="#quick"><span class="n">3</span><strong>Sample</strong><p>Draw targets from <code>workspace(kind)</code>, each with the joint angles that reach it, already round-tripped through FK: <code>solutions(p)</code>.</p></a></li>
<li class="out"><a href="#viewer"><span class="n">→</span><strong>See it live</strong><p><code>display()</code> opens viser: click a point, step through its solutions, cut the map, jog the arm. Python recomputes on demand.</p></a></li>
</ol>
<div class="quick" id="quick" style="margin-top:24px">
@@QUICK@@
<pre class="out"><code>@@QUICK_OUT@@</code></pre>
<p class="footnote">That output is <code>site/quickstart.py</code> run as shown. Open the five example robots in the viewer:</p>
@@INSTALL@@
</div>
</section>

<section id="any-dof">
<div class="section-head"><h2>Any DOF, one call</h2>
<p>The same <code>rr.analyze(arm)</code> on five arms, from a single pendulum to a six-joint UR5e. <strong>Verified</strong>
counts joint solutions that passed every check; the bar splits each grid by level. @@STATS@@.</p></div>
@@ARMS_TABLE@@
<div class="footnote"><p class="legend">@@LEGEND@@</p>
<p>Levels count how much of the pose the arm controls at a point; position-only arms stop at the first. Numbers are the saved
results the viewer below loads.</p></div>
</section>

<section id="viewer">
<div class="section-head"><h2>Live insight in viser</h2>
<p>The result is plain arrays on disk; the viewer draws them and its Python backend recomputes as you click:
map a dragged box finer, check every orientation at one point, jog the arm and watch its status and torque.</p></div>
<div class="grid">@@SHOTS_VIEWER@@</div>
</section>

<section id="joint-space">
<div class="section-head"><h2>Joint space, with the reason</h2>
<p>One vocabulary of status codes for every picture: a posture fails for a joint limit, a singularity, a collision, the
motor torque, or a linkage that cannot close.</p></div>
<div class="grid">@@SHOTS_JOINT@@</div>
</section>

<section class="promises">
<div class="promise"><strong>It never trusts a solver's OK</strong><p>Every IK answer is re-checked from the maths (joint limits, the FK round trip, duplicates, torque, singularity), and the collision reasons your FK or IK return are kept. A lazy solver cannot overclaim.</p></div>
<div class="promise"><strong>It computes once</strong><p>The result log is npz + JSON, no pickle. A viewer, a notebook or another package reads the same numbers; nothing recomputes to look.</p></div>
<div class="promise"><strong>It never guesses an extent</strong><p>The FK cloud bounds the grid, so even a pendulum whose workspace is a circle gets the right picture, and a planar arm's grid collapses onto its plane.</p></div>
</section>

<section id="example">
<div class="section-head"><h2>Worked example, executed</h2>
<p>Every cell ran as shown; the outputs are the real ones: an arm from two functions, any DOF, a saved UR5e map queried the
way a downstream package would, a conveyor belt, a URDF, robot_msgs, someone else's viser scene. Download
<a href="reach_worked_example.ipynb">the notebook</a> or <a href="reach_worked_example.py">the script</a>; it runs against a
local checkout (robot_reach and robot_msgs are not on PyPI yet).</p></div>
<details class="fold"><summary>Show all @@NB_CELLS@@ cells and their output</summary><div class="doc">@@NOTEBOOK@@</div></details>
</section>

<section id="api">
<div class="section-head"><h2>API</h2>
<p>What exists on <code>main</code> today, rendered from <code>docs/api.md</code>. A typed downstream API is being proposed
separately (<code>docs/proposals/</code>); it is not code yet.</p></div>
<details class="fold"><summary>Show the API reference</summary><div class="doc">@@API@@</div></details>
</section>

<section class="cta">@@MARK@@<p>Start with the <a href="#happy-path">quickstart</a>, watch the <a href="#viewer">viewer</a>, or read the <a href="#example">worked example</a>.</p></section>
</main>
<footer>robot_reach (BAM Reach v2) · MIT · every number and picture on this page was produced by the code at @@COMMIT@@, built @@DATE@@.</footer>
<script>
document.querySelector('.theme').addEventListener('click',function(){var r=document.documentElement,dark=r.classList.contains('is-dark');r.dataset.theme=dark?'light':'dark';try{localStorage.setItem('rr-theme',r.dataset.theme)}catch(e){}sync()});
function sync(){var r=document.documentElement,t=r.dataset.theme;r.classList.toggle('is-dark',t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches)}sync();
</script>
</body></html>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--media", action="store_true", help="re-make site/media/ from out/site_media_raw/ + out/site/*/figures/")
    ap.add_argument("--run", action="store_true", help="re-execute site/quickstart.py for its printed output")
    a = ap.parse_args()
    print(build(media=a.media, run=a.run))
