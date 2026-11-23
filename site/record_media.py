#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright==1.55.0"]
# ///
"""Record the landing page's media from the REAL viewer in headless Chrome.

    uv run --script /home/bam/robot_reach/site/record_media.py

Launches `python -m robot_reach examples --arm ur5e` on port 8121 (this script owns that process and
stops it by PID) over the results in `out/site/` (`python -m robot_reach run all --out out/site` first), then drives it with real mouse clicks: the UR5e workspace, a voxel clicked, its IK
branches stepped through, the workspace levels switched, the map cut open, then the five-bar and its
joint-space picture. Writes stills, the raw webm and `steps.json` (when each step happened) into
`out/site_media_raw/`; `build_site.py` turns them into the hero mp4 + GIF.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "site_media_raw"  # gitignored: raw PNGs + webm (~9 MB); build_site.py makes the page media
PORT = 8121  # WHY fixed in 8120-8129: the lane's assigned range; 8097 and 8110-8119 belong to others


def start_viewer() -> tuple[subprocess.Popen, str]:
    cmd = ["taskset", "-c", "2-5,16,17", str(ROOT / ".venv/bin/python"), "-m", "robot_reach", "examples", "--arm", "ur5e", "--out", str(ROOT / "out" / "site"), "--port", str(PORT)]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    deadline = time.time() + 180
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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*"):
        old.unlink()
    proc, url = start_viewer()
    steps, t0 = [], None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True, args=["--enable-webgl", "--enable-unsafe-swiftshader"])
            ctx = browser.new_context(viewport={"width": 1440, "height": 900}, record_video_dir=str(OUT), record_video_size={"width": 1440, "height": 900})
            page = ctx.new_page()
            t0 = time.time()
            errors = []
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

            def click_canvas(dx=0.0, dy=0.0):
                before = text()
                page.mouse.move(cx + dx, cy + dy)
                page.mouse.down()
                page.mouse.up()
                page.wait_for_timeout(1500)
                return text() != before

            def button(name, wait=700):
                page.get_by_role("button", name=name, exact=True).click()
                page.wait_for_timeout(wait)

            def choose(label, option, wait=1500):
                page.get_by_label(label, exact=True).click()
                # WHY a pattern: option labels carry suffixes ("reachable (3+ DOF)") that change with the viewer
                page.get_by_role("option", name=re.compile(option)).first.click()
                page.wait_for_timeout(wait)

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
            m = re.search(r"(\d+) / (\d+)", text())
            steps.append({"step": "click", "changed": clicked, "counter": m.group(0) if m else None})
            shot("02-clicked")
            for _ in range(4):
                button("Next ▶")
            orbit(-140, -30)
            m = re.search(r"(\d+) / (\d+)", text())
            steps.append({"step": "after Next x4", "counter": m.group(0) if m else None})
            shot("03-branches")
            button("Clear", 1000)
            choose("Show", "^dexterous")
            shot("04-dexterous")
            choose("Show", "^4-DOF")
            shot("05-four-dof")
            choose("Show", "^reachable", 800)
            choose("Cut", "^y$")
            shot("06-cut")
            choose("Cut", "^none$", 800)
            choose("Arm", "^five_bar$", 3000)
            shot("07-five-bar")
            click_canvas(0, -60) or click_canvas(-30, -40) or click_canvas(20, -90)
            button("Next ▶", 1200)
            shot("08-five-bar-clicked")
            choose("Picture", "slice", 1800)
            shot("09-five-bar-joint-map")
            choose("Arm", "^shoulder3$", 3000)
            shot("10-shoulder3")
            page.wait_for_timeout(600)
            mark("end")
            video = page.video.path()
            ctx.close()
            browser.close()
            Path(video).rename(OUT / "hero.webm")
            steps.append({"errors": errors})
    finally:
        stop(proc)
    (OUT / "steps.json").write_text(json.dumps(steps, indent=1))
    print(json.dumps(steps, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
