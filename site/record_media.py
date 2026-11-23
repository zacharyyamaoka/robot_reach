#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright==1.55.0"]
# ///
"""Re-record the landing page's media from the REAL viewer and rebuild the page. One command, re-runnable:

    uv run --script /home/bam/robot_reach/site/record_media.py

1. Recomputes the saved example results (`python -m robot_reach run all --out out/site --fresh --no-view`), so the
   figures and numbers match the viewer you are about to record (`--keep-results` skips this).
2. Launches `python -m robot_reach examples --arm ur5e` on port 8121 (owned by this script, stopped by PID) and drives
   it in headless Chrome with real mouse clicks: the UR5e workspace, a point clicked, its IK configurations stepped
   through, the workspace levels switched, the map cut open, then the five-bar and its joint space, then the 3-joint
   shoulder. Stills, the raw webm and `steps.json` go into `out/site_media_raw/`.
3. Runs `site/build_site.py --media --run` (stills -> jpg, hero mp4 + GIF + poster, figures, quickstart output).

WHY every GUI lookup has alternatives and a soft skip: the viewer's labels change under this script (level names,
"Clear" vs "Deselect" …). A step whose control is missing is logged in steps.json under "skipped" and its still is
left out; the page shows what was recorded instead of the whole run failing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "site_media_raw"  # gitignored: raw PNGs + webm (~9 MB); build_site.py makes the page media
PORT = 8121  # WHY fixed in 8120-8129: the lane's assigned range; 8097, 8110-8119 and 8130-8149 belong to others
CPUS = "2-5,16,17"  # WHY pinned: the host's agent CPU set (bam_ws memory/CPU policy)
PY = str(ROOT / ".venv/bin/python")

# Workspace levels by meaning, not by label: each is a list of patterns tried in order against the Show options.
LEVEL_TOP_DOWN = [r"^4-DOF", r"top.?down", r"straight down", r"any yaw", r"^four"]
LEVEL_DEXTEROUS = [r"^dexterous", r"every orientation", r"^6"]
LEVEL_REACHABLE = [r"^reachable", r"position", r"^3"]


def results() -> None:
    cmd = ["taskset", "-c", CPUS, PY, "-m", "robot_reach", "run", "all", "--out", str(ROOT / "out" / "site"), "--fresh", "--no-view"]
    print("results:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def start_viewer() -> tuple[subprocess.Popen, str]:
    cmd = ["taskset", "-c", CPUS, PY, "-m", "robot_reach", "examples", "--arm", "ur5e", "--out", str(ROOT / "out" / "site"), "--port", str(PORT)]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    deadline = time.time() + 300
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line:
            print("viewer:", line.rstrip(), flush=True)
        m = re.search(r"READY (http://127\.0\.0\.1:\d+/?)", line or "")
        if m:
            return proc, m.group(1)
        if proc.poll() is not None:
            break
    stop(proc)
    raise RuntimeError("viewer never printed READY")


def stop(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def record() -> list:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*"):
        old.unlink()
    proc, url = start_viewer()
    steps: list = []
    skipped: list = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True, args=["--enable-webgl", "--enable-unsafe-swiftshader"])
            ctx = browser.new_context(viewport={"width": 1440, "height": 900}, record_video_dir=str(OUT), record_video_size={"width": 1440, "height": 900})
            page = ctx.new_page()
            t0 = time.time()
            errors: list = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.goto(url)
            page.locator("canvas").first.wait_for(state="visible", timeout=60_000)
            page.wait_for_timeout(5000)
            # WHY hide toasts: headless Chrome renders WebGL in software and viser says so in a toast;
            # on a GPU machine it never appears, so it is not part of the app.
            page.add_style_tag(content="[class*='Notification'], [class*='notification'] { display: none !important; }")
            canvas = max(page.locator("canvas").all(), key=lambda c: (c.bounding_box() or {"width": 0})["width"])
            box = canvas.bounding_box()
            cx, cy = box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5

            def mark(name):
                steps.append({"step": name, "t": round(time.time() - t0, 2)})

            def shot(name):
                page.screenshot(path=str(OUT / f"{name}.png"))
                mark(name)

            def text():
                # WHY a long timeout: software WebGL can hold the page's main thread for a while after a click
                return page.locator("body").inner_text(timeout=120_000)

            def counter():
                m = re.search(r"(\d+) / (\d+)", text())
                return m.group(0) if m else None

            def click_canvas(dx=0.0, dy=0.0):
                before = text()
                page.mouse.move(cx + dx, cy + dy)
                page.mouse.down()
                page.mouse.up()
                page.wait_for_timeout(1500)
                return text() != before

            def button(names, wait=700) -> bool:
                for name in names:
                    loc = page.get_by_role("button", name=name, exact=True)
                    if loc.count():
                        loc.first.click()
                        page.wait_for_timeout(wait)
                        return True
                skipped.append({"button": names})
                return False

            def choose(labels, patterns, wait=1500) -> bool:
                """Open the first dropdown whose label exists, pick the first option matching a pattern (in order)."""
                for label in labels:
                    field = page.get_by_label(label, exact=True)
                    if not field.count():
                        continue
                    field.first.click()
                    page.wait_for_timeout(300)
                    options = page.get_by_role("option")
                    names = [options.nth(i).inner_text().strip() for i in range(options.count())]
                    for pat in patterns:
                        hit = next((i for i, n in enumerate(names) if re.search(pat, n, re.I)), None)
                        if hit is not None:
                            options.nth(hit).click()
                            page.wait_for_timeout(wait)
                            return True
                    page.keyboard.press("Escape")
                    skipped.append({"dropdown": label, "wanted": patterns, "options": names})
                    return False
                skipped.append({"dropdown": labels})
                return False

            def orbit(dx, dy):
                page.mouse.move(cx + 200, cy + 200)
                page.mouse.down()
                page.mouse.move(cx + 200 + dx, cy + 200 + dy, steps=30)
                page.mouse.up()
                page.wait_for_timeout(700)

            page.mouse.move(cx, cy)
            page.mouse.wheel(0, -350)  # closer: the default view fits the whole FK box
            page.wait_for_timeout(1500)
            shot("01-ur5e-workspace")
            clicked = any(click_canvas(dx, dy) for dx, dy in ((0, 0), (-40, 20), (30, -30), (-60, -10)))
            steps.append({"step": "click", "changed": clicked, "counter": counter()})
            shot("02-clicked")
            for _ in range(4):
                button(["Next ▶", "Next"])
            orbit(-140, -30)
            steps.append({"step": "after Next x4", "counter": counter()})
            shot("03-ik-configurations")
            button(["Clear", "Deselect"], 1000)
            if choose(["Show"], LEVEL_DEXTEROUS):
                shot("04-dexterous")
            if choose(["Show"], LEVEL_TOP_DOWN):
                shot("05-top-down")
            choose(["Show"], LEVEL_REACHABLE, 800)
            if choose(["Cut"], [r"^y$"]):
                shot("06-cut")
                choose(["Cut"], [r"^none$"], 800)
            if choose(["Arm", "Robot"], [r"^five_bar$", r"five"], 3000):
                shot("07-five-bar")
                click_canvas(0, -60) or click_canvas(-30, -40) or click_canvas(20, -90)
                button(["Next ▶", "Next"], 1200)
                shot("08-five-bar-clicked")
                if choose(["Picture"], [r"slice"], 1800):
                    shot("09-five-bar-joint-map")
            if choose(["Arm", "Robot"], [r"^shoulder3$", r"shoulder"], 3000):
                shot("10-shoulder3")
            page.wait_for_timeout(600)
            mark("end")
            video = page.video.path()
            ctx.close()
            browser.close()
            Path(video).rename(OUT / "hero.webm")
            steps.append({"errors": errors, "skipped": skipped})
    except PWTimeout as e:
        steps.append({"error": str(e), "skipped": skipped})
        raise
    finally:
        stop(proc)
        (OUT / "steps.json").write_text(json.dumps(steps, indent=1))
    return steps


def main() -> int:
    ap = argparse.ArgumentParser(description="re-record the landing page media and rebuild site/index.html")
    ap.add_argument("--keep-results", action="store_true", help="reuse out/site/ instead of recomputing the example results")
    ap.add_argument("--no-build", action="store_true", help="record only; do not run build_site.py")
    a = ap.parse_args()
    if not a.keep_results:
        results()
    steps = record()
    print(json.dumps(steps, indent=1))
    if not a.no_build:
        cmd = ["uv", "run", "--project", str(ROOT), "--with", "markdown", "--with", "pillow", "--with", "pygments",
               "python", str(ROOT / "site" / "build_site.py"), "--media", "--run"]
        subprocess.run(cmd, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
