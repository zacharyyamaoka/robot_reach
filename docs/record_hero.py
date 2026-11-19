#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright==1.55.0"]
# ///
"""Record the hero video from the REAL viewer: launch it, click a voxel in the 3D canvas, step through
its IK solutions, cut the map open, switch to the five-bar and click again. Also saves stills.

    uv run --script docs/record_hero.py --out reports/media/hero

The viewer is started and stopped by this script (it owns that one process). A real mouse click in
the canvas is used — not the --select seam — so the video proves clicking works end to end.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def start_viewer() -> tuple[subprocess.Popen, str]:
    proc = subprocess.Popen(
        [str(ROOT / ".venv/bin/python"), "-m", "robot_reach", "examples", "--arm", "ur5e", "--port", "0"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        line = proc.stdout.readline()
        m = re.search(r"READY (http://127\.0\.0\.1:\d+/)", line or "")
        if m:
            return proc, m.group(1)
    proc.terminate()
    raise RuntimeError("viewer never printed READY")


def panel_text(page) -> str:
    return page.locator("body").inner_text()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "reports/media/hero")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    proc, url = start_viewer()
    log = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True, args=["--enable-webgl", "--enable-unsafe-swiftshader"])
            ctx = browser.new_context(viewport={"width": 1440, "height": 900}, record_video_dir=str(a.out), record_video_size={"width": 1440, "height": 900})
            page = ctx.new_page()
            page.goto(url)
            page.locator("canvas").first.wait_for(state="visible", timeout=60_000)
            page.wait_for_timeout(4000)
            # WHY hide toasts: headless Chrome renders WebGL in software (SwiftShader) and viser
            # says so in a toast; on a GPU machine it never appears, so it is not part of the app.
            page.add_style_tag(content="[class*='Notification'], [class*='notification'] { display: none !important; }")
            canvas = max(page.locator("canvas").all(), key=lambda c: (c.bounding_box() or {"width": 0})["width"])
            box = canvas.bounding_box()
            cx, cy = box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5

            def shot(name):
                page.screenshot(path=str(a.out / f"{name}.png"))

            def click_canvas(dx=0.0, dy=0.0):
                before = panel_text(page)
                page.mouse.move(cx + dx, cy + dy)
                page.mouse.down()
                page.mouse.up()
                page.wait_for_timeout(1500)
                changed = panel_text(page) != before
                log.append(f"canvas click at ({cx + dx:.0f},{cy + dy:.0f}) changed panel: {changed}")
                return changed

            def button(name, wait=900):
                page.get_by_role("button", name=name, exact=True).click()
                page.wait_for_timeout(wait)

            def choose(label, option, wait=1500):
                page.get_by_label(label, exact=True).click()
                page.get_by_role("option", name=option, exact=True).click()
                page.wait_for_timeout(wait)

            shot("01-workspace")
            for dx, dy in ((0, 0), (-40, 20), (30, -30), (-60, -10)):
                if click_canvas(dx, dy):
                    break
            m = re.search(r"(\d+) / (\d+)", panel_text(page))
            log.append(f"after click: solution counter {m.group(0) if m else 'none'}")
            shot("02-clicked")
            for _ in range(5):
                button("Next ▶")
            # orbit a little so the posed arm reads in 3-D
            page.mouse.move(cx + 200, cy + 200)
            page.mouse.down()
            page.mouse.move(cx + 80, cy + 170, steps=25)
            page.mouse.up()
            page.wait_for_timeout(1000)
            shot("03-stepped")
            m = re.search(r"(\d+) / (\d+)", panel_text(page))
            log.append(f"after 5x Next: {m.group(0) if m else 'none'}")
            button("Clear", 1200)
            choose("Cut", "y")
            shot("04-cut")
            choose("Cut", "none", 800)
            choose("Arm", "five_bar", 3000)
            shot("05-five-bar")
            click_canvas(0, -60) or click_canvas(-30, -40) or click_canvas(20, -90)
            button("Next ▶", 1200)
            shot("06-five-bar-clicked")
            choose("Picture", "motor_left × motor_right: slice", 1500)
            shot("07-five-bar-joint-map")
            page.wait_for_timeout(800)
            video = page.video.path()
            ctx.close()
            browser.close()
            log.append(f"video: {video}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    (a.out / "record.log").write_text("\n".join(log) + "\n")
    print("\n".join(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
