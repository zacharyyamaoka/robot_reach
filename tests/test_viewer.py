"""Viewer and overlay behaviour without a browser (the browser pass is the headless-viser review)."""

from __future__ import annotations

import types

import numpy as np
import pytest

import robot_reach as rr
from robot_reach.overlay import add_reach_map
from robot_reach.robots import pendulum, rr as rr_arm, ur5e

viser = pytest.importorskip("viser")


@pytest.fixture(scope="module")
def results():
    from robot_reach import OrientationSet

    small = rr.analyze(ur5e(), floor_z=0.0, step=0.25, orientations=OrientationSet.sample(tilt_step_deg=90, n_rolls=4))
    return {"ur5e": small, "rr": rr.analyze(rr_arm(), floor_z=-0.15), "pendulum": rr.analyze(pendulum())}


def test_click_selects_and_steps_through_solutions(results):
    from robot_reach.viewer import ReachViewer

    v = ReachViewer(results)
    try:
        v.mount("ur5e")
        v._on_voxel_click(types.SimpleNamespace(instance_index=0))
        assert v.sel_point == int(v.visible_index[0])
        n = len(v.sel_solutions)
        assert n >= 1 and v.overlay._dim  # voxels dim so the arm shows
        v._step_solution(+1)
        assert v.g_sol.value == (2 if n > 1 else 1)
        o, b, q = v.sel_solutions[v.g_sol.value - 1]
        # the arm is drawn at exactly the selected solution and FK lands on the target
        T, _ = v.arm.forward(q[None])
        assert np.allclose(T[0, :3, 3], v.reach.positions[v.sel_point], atol=1e-4)
        v._clear_selection()
        assert v.sel_point is None and not v.overlay._dim
    finally:
        v.stop()


def test_cut_poi_box_and_arm_switching(results):
    from robot_reach.viewer import ReachViewer

    v = ReachViewer(results)
    try:
        v.mount("ur5e")
        full = len(v.visible_index)
        v.g_cut.value = "y"
        v._cut_changed()
        assert 0 < len(v.visible_index) < full
        v._run_tool("Drop POI")
        assert v.layer_reach.n_positions == 1 and "POI" in v.g_tool_md.content
        v.g_box_step.value = 0.1
        v._run_tool("Compute box")
        assert v.layer_reach.n_positions > 1 and "of the box" in v.g_tool_md.content
        v._run_tool("Back to full map")
        assert v.layer_reach is None
        v.mount("pendulum")
        v.g_jog[0].value = 90.0
        v._jog_changed()
        assert "OK" in v.g_jog_md.content  # straight up: no gravity torque
        v.g_jog[0].value = 0.0
        v._jog_changed()
        assert "torque" in v.g_jog_md.content
    finally:
        v.stop()


def test_overlay_draws_into_a_foreign_server(results):
    server = viser.ViserServer(host="127.0.0.1", port=0, verbose=False)
    try:
        server.scene.add_frame("/facility/robot_1", position=(2.0, 0.0, 0.0))
        picked = []
        ov = add_reach_map(server, results["ur5e"].reach, parent_frame="/facility/robot_1/reach", on_select=picked.append)
        n_all = len(ov.shown_index)
        ov.kind = "dexterous"
        assert len(ov.shown_index) <= n_all
        ov.update(blob=True, color_by="IK solutions")
        ov._clicked(types.SimpleNamespace(instance_index=0))
        assert picked == [int(ov.shown_index[0])]
        ov.visible = False
        assert len(ov.shown_index) == 0
        ov.remove()
    finally:
        server.stop()
