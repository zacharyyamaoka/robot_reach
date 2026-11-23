"""Zach's 2026-11-23 feedback, round 2 (Daily Note "Feedback on Robot Reach"): collision as FK/IK codes,
no floor on the UR, the tightest grid shape, workspace levels by name, IK configurations (A1), joint-range
histograms (B2), the tilting Ask-here box (C2), colormaps + traffic light, the legend list (E4), the
settings card (F2) and the skeleton switch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import robot_reach as rr
from robot_reach import Code, OrientationSet
from robot_reach.codes import COLORMAPS, LEVEL_NAMES, TRAFFIC, colormap_rgb, is_collision
from robot_reach.configurations import IKConfigurations
from robot_reach.examples import EXAMPLES, UR5E_URDF
from robot_reach.overlay import CONSISTENCY_COLORS, bins, show_options
from robot_reach.robots import rr as rr_arm
from robot_reach.robots import shoulder3, ur5e
from robot_reach.sampling import Box, fit_grid, fk_cloud, sample_joints


# ------------------------------------------------------------------ 2. collision comes from FK/IK codes
def _rr_with(fk_code=None, ik_code=None, collision=None):
    import dataclasses

    base = rr_arm()

    def fk(q):
        T = base.fk(q)
        if fk_code is None:
            return T
        return T, np.where(q[:, 1] > 1.0, int(fk_code), int(Code.OK)).astype(np.int8)  # elbow folded past 57° "touches"

    def ik(T):
        q, code = base.ik(T)
        if ik_code is not None:
            code = np.where((code == Code.OK) & (q[..., 1] < -1.0), int(ik_code), code).astype(np.int8)
        return q, code

    return dataclasses.replace(base, fk=fk, ik=ik, collision=collision)


def test_self_and_environment_collision_are_first_class_codes():
    assert {Code.SELF_COLLISION, Code.ENVIRONMENT_COLLISION} <= set(Code)
    assert is_collision([Code.COLLISION, Code.SELF_COLLISION, Code.ENVIRONMENT_COLLISION, Code.OK]).tolist() == [True, True, True, False]
    assert rr.CODE_LABELS[Code.SELF_COLLISION] == "self-collision"


def test_fk_codes_reach_the_joint_check_and_the_reach_map():
    arm = _rr_with(fk_code=Code.SELF_COLLISION)
    q = np.array([[0.5, 1.5], [0.5, 0.2]])
    assert arm.check(q, torque=False).tolist() == [Code.SELF_COLLISION, Code.OK]
    T, ok = arm.forward(q)
    assert ok.all()  # a self-colliding posture still has a pose
    m = rr.reach_map(arm, Box((0.3, 0.0, 0.2), (0.4, 0.0, 0.4)).grid(0.05))
    assert m.code_counts().get("SELF_COLLISION", 0) > 0
    assert np.all(m.sol_q[:, 1] <= 1.0 + 1e-6)  # no working solution has the flagged posture


def test_ik_codes_and_the_adapter_fold_into_the_same_codes():
    arm = _rr_with(ik_code=Code.ENVIRONMENT_COLLISION)
    m = rr.reach_map(arm, Box((0.3, 0.0, 0.2), (0.4, 0.0, 0.4)).grid(0.05))
    assert m.code_counts().get("ENVIRONMENT_COLLISION", 0) > 0
    adapter = _rr_with(collision=lambda q: np.where(q[:, 0] > 1.0, int(Code.SELF_COLLISION), int(Code.OK)))
    assert adapter.check(np.array([[1.2, 0.1], [0.2, 0.1]]), torque=False).tolist() == [Code.SELF_COLLISION, Code.OK]
    boolean = _rr_with(collision=lambda q: q[:, 0] > 1.0)
    assert boolean.check(np.array([[1.2, 0.1]]), torque=False).tolist() == [Code.COLLISION]
    # the optional floor is an environment collision
    assert rr_arm().check(np.array([[-0.4, 0.0]]), floor_z=0.0, torque=False).tolist() == [Code.ENVIRONMENT_COLLISION]


# ------------------------------------------------------------------ 5. no floor on the UR
def test_the_ur_examples_have_no_floor():
    assert "floor_z" not in EXAMPLES["ur5e"][1]
    import inspect

    from robot_reach import examples

    assert "floor_z" not in inspect.getsource(examples.analyze_urdf_example).split('"""')[2]


# ------------------------------------------------------------------ 3+4. the tightest grid shape
@pytest.mark.parametrize("factory", [ur5e, shoulder3, rr_arm])
def test_grid_shape_holds_every_fk_point_and_beats_the_box(factory):
    arm = factory()
    pts = fk_cloud(arm, 20_000).points
    step = float(max(Box.around(pts).size)) / 12
    positions, shape, counts = fit_grid(pts, step)
    assert shape.contains(pts).all()  # every FK point inside, padding included
    assert len(positions) == counts[shape.kind] == min(counts.values())
    assert set(counts) == {"box", "sphere", "cylinder", "hull"}
    if factory is ur5e:
        assert len(positions) < counts["box"] * 0.75 and shape.kind != "box"
    # every FK point has a kept grid point within one voxel (the padding's promise)
    near = np.min(np.linalg.norm(pts[::40, None, :] - positions[None], axis=2), axis=1)
    assert near.max() <= step * np.sqrt(3) / 2 + 1e-9


def test_analyze_records_every_setting_and_old_results_still_load(tmp_path):
    res = rr.analyze(ur5e(), steps_across=8, joint_axes=[])
    st = res.config["settings"]
    assert st["grid_shape"] in ("sphere", "cylinder", "hull") and st["grid_points"] == res.reach.n_positions
    assert st["orientation"]["tilt_step_deg"] == 45.0 and st["orientation"]["roll_step_deg"] == 60.0
    assert st["grid_shape_points"]["box"] > st["grid_points"] and len(st["lattice"]) == 3
    res.save(tmp_path / "ur")
    doc = json.loads((tmp_path / "ur" / "result.json").read_text())
    assert doc["schema"] == "robot_reach.result.v3" and doc["config"]["settings"]["grid_shape"] == st["grid_shape"]
    # a v2 result (no settings) still loads, and its settings are read back from what it stored
    del doc["config"]["settings"]
    doc["schema"] = "robot_reach.result.v2"
    (tmp_path / "ur" / "result.json").write_text(json.dumps(doc))
    from robot_reach.analyze import settings_of

    old = settings_of(rr.load(tmp_path / "ur"))
    assert old["grid_shape"] == "box" and old["orientation"]["n_orientations"] == 132


# ------------------------------------------------------------------ 6. workspace levels by name
def test_levels_are_named_without_dof_numbers_and_nest():
    assert [LEVEL_NAMES[k] for k in (3, 4, 5, 6)] == ["reachable", "top-down", "hemisphere", "dexterous"]
    assert not any("DOF" in v for v in LEVEL_NAMES.values())
    m = rr.reach_map(ur5e(), Box((0.4, 0.0, 0.3), (0.4, 0.4, 0.4)).grid(0.2), OrientationSet.sample())
    assert not any("DOF" in label for label in show_options(m, "workspace"))
    masks = [m.workspace(k) for k in ("reachable", "four_dof", "hemisphere", "dexterous")]
    for wide, narrow in zip(masks, masks[1:]):
        assert np.all(wide | ~narrow)  # nested: each includes the next


# ------------------------------------------------------------------ 7. IK configurations
def test_ur_configurations_name_all_eight_solutions_from_the_posture():
    arm = ur5e()
    cfg = arm.configurations
    assert cfg.n_configurations == 8 and [q.name for q in cfg.questions] == ["shoulder", "elbow", "wrist"]
    assert cfg.label(cfg.questions[1]) == "elbow (J3)"
    T, _ = arm.forward(sample_joints(arm, 1500, seed=4))
    q, code = arm.inverse(T)
    full = np.all(code == Code.OK, axis=1)
    assert full.sum() > 500
    for i in np.flatnonzero(full):
        assert len(set(cfg.configuration(q[i]).tolist())) == 8, i  # every closed-form solution is its own configuration


@pytest.mark.skipif(not UR5E_URDF.exists(), reason="no UR5e URDF on this machine")
def test_a_ur_urdf_gets_the_same_configurations_as_the_closed_form():
    from robot_reach.urdf import arm_from_urdf

    closed, urdf = ur5e(), arm_from_urdf(UR5E_URDF, "tool0")
    assert urdf.configurations is not None
    q = sample_joints(closed, 4000, seed=5)
    q[:, 2] = np.clip(q[:, 2], -np.pi, np.pi)
    assert np.mean(closed.configurations.configuration(q) == urdf.configurations.configuration(q)) == 1.0


def test_filter_and_toggles(tmp_path):
    from robot_reach.viewer import chip_labels, toggle_choice

    # A1 toggles: at least one answer, at most two; pressing the only lit one swaps to the other
    assert toggle_choice({0, 1}, 0, 2) == {1}
    assert toggle_choice({1}, 1, 2) == {0}
    assert toggle_choice({1}, 0, 2) == {0, 1}
    assert chip_labels(("up", "down"), {0}) == ("✓ up", "○ down")

    m = rr.reach_map(ur5e(), Box((0.4, 0.0, 0.3), (0.6, 0.6, 0.6)).grid(0.15), OrientationSet.sample())
    cfg = m.configurations
    assert m.sol_config is not None and len(m.sol_config) == len(m.sol_index)
    up = m.filtered(cfg.allowed(elbow="up"))
    assert np.all(cfg.index_of(np.array([cfg.tags_of(c) for c in up.sol_config]))[:, None] == up.sol_config[:, None])
    assert all(cfg.tags_of(c)[1] == 0 for c in np.unique(up.sol_config))
    assert len(up.sol_index) < len(m.sol_index) and (up.code == Code.EXCLUDED).sum() == len(m.sol_index) - len(up.sol_index)
    assert np.all(up.n_solutions <= m.n_solutions) and m.filtered(cfg.allowed()) is m
    # saved and loaded without the arm: tags and questions come back
    path = m.save(tmp_path / "m")
    back = rr.ReachMap.load(path)
    assert np.array_equal(back.sol_config, m.sol_config) and isinstance(back.configurations, IKConfigurations)
    assert back.configurations.tag is None and [q.options for q in back.configurations.questions] == [q.options for q in cfg.questions]


# ------------------------------------------------------------------ 8. one histogram per joint
def test_joint_ranges_are_ten_degree_histograms():
    pytest.importorskip("matplotlib")
    from robot_reach.plots import ranges_figure

    res = rr.analyze(ur5e(), steps_across=6, joint_axes=[])
    edges = np.rad2deg(res.ranges.hist_edges)
    assert np.allclose(np.diff(edges, axis=1), 10.0) and res.ranges.hist.sum() == len(res.reach.sol_q) * 6
    fig = ranges_figure(res.ranges, "ur5e")
    assert len(fig.axes) == 6


# ------------------------------------------------------------------ 9. the tilting Ask-here box
def test_box_frame_round_trips_and_its_z_is_the_approach():
    from robot_reach.viewer import box_orientations, box_positions, matrix_rpy, rpy_matrix, wxyz_matrix, _wxyz

    for rpy in ((0, 0, 0), (20, -35, 140), (-170, 80, -10)):
        R = rpy_matrix(rpy)
        assert np.allclose(rpy_matrix(matrix_rpy(R)), R)
        assert np.allclose(wxyz_matrix(_wxyz(R)), R, atol=1e-9)
    level = box_orientations(rpy_matrix((0, 0, 0)), 180, 45, 60)
    assert np.allclose(level.approach, (0, 0, -1)) and len(level) == 132
    tilted = box_orientations(rpy_matrix((0, 30, 0)), 90, 30, 90)
    assert np.allclose(tilted.approach, -rpy_matrix((0, 30, 0))[:, 2]) and tilted.resolution()["tilt_max_deg"] == 90
    pts = box_positions((1, 2, 3), (0.2, 0.2, 0.2), (0, 0, 45), 0.1)
    assert len(pts) == 27 and np.allclose(pts.mean(0), (1, 2, 3))
    assert np.isclose(np.max(np.abs(pts[:, 0] - 1)), 0.1 * np.sqrt(2))  # the corners turned 45°


# ------------------------------------------------------------------ 10. colours
def test_colormaps_and_traffic_light():
    assert set(COLORMAPS) == {"viridis", "magma", "plasma", "inferno", "cividis"}
    lo, hi = colormap_rgb(np.array([0.0, 1.0]))
    assert tuple(lo) == (68, 1, 84) and tuple(hi) == (253, 231, 37)  # matplotlib's viridis ends
    assert np.array_equal(colormap_rgb(np.array([0.0, 1.0]), reverse=True), colormap_rgb(np.array([1.0, 0.0])))
    assert CONSISTENCY_COLORS[1] == TRAFFIC["good"] and CONSISTENCY_COLORS[2] == TRAFFIC["partial"] and CONSISTENCY_COLORS[3] == TRAFFIC["bad"]


# ------------------------------------------------------------------ 11. the legend list
def test_legend_bins_count_every_point():
    m = rr.reach_map(ur5e(), Box((0.4, 0.0, 0.3), (0.8, 0.8, 0.8)).grid(0.2), OrientationSet.sample())
    for mode in ("workspace", "IK solutions", "fraction of orientations", "consistency"):
        key, rows = bins(m, mode)
        assert sum(r["count"] for r in rows) == m.n_positions and rows[0]["key"] == "0"
        assert set(key) <= {r["key"] for r in rows}


# ------------------------------------------------------------------ viewer: settings card, legend, filter, skeleton
viser = pytest.importorskip("viser")


@pytest.fixture(scope="module")
def viewer():
    from robot_reach.viewer import ReachViewer

    results = {"ur5e": rr.analyze(ur5e(), steps_across=7, joint_axes=[(1, 2)]), "shoulder3": rr.analyze(shoulder3(), floor_z=0.0)}
    v = ReachViewer(results, port=0)
    yield v
    v.stop()


def test_viewer_settings_card_edit_reset_compute(viewer):
    v = viewer
    v.mount("ur5e")
    shown = len(v.visible_index)
    assert shown > 0 and v.g_set_btns.disabled
    v.g_set_tilt.value = 60.0
    v._settings_changed()
    assert not v.g_set_btns.disabled and len(v.visible_index) == 0 and "was 45" in v.g_set_used.content
    v._settings_reset()
    assert v.g_set_btns.disabled and len(v.visible_index) == shown
    v.g_set_tilt.value = 90.0
    v._settings_changed()
    v._compute_workspace()
    assert v.reach.orientations.resolution()["tilt_step_deg"] == 90.0 and v.result.config["settings"]["orientation"]["tilt_step_deg"] == 90.0
    assert v.g_set_btns.disabled and len(v.visible_index) > 0


def test_viewer_legend_list_filter_and_skeleton(viewer):
    v = viewer
    v.mount("ur5e")
    labels = [h.label for _k, h in v._legend_rows]
    assert labels[0].startswith("unreachable · ") and not any("DOF" in label for label in labels)
    everything = sum(int(h.label.rsplit(" · ", 1)[1].replace(",", "")) for _k, h in v._legend_rows)
    assert everything == v.reach.n_positions
    before = len(v.visible_index)
    v._legend_toggled("6", False)  # hide dexterous
    assert len(v.visible_index) == before - int(np.sum(v.reach.levels() == 6))
    v._legend_toggled("6", True)
    # IK configurations: only elbow up -> fewer working solutions, stepper shows the configuration's name
    n_all = len(v._current_layer().sol_index)
    v._cfg_clicked(1, "✓ down")
    assert len(v._current_layer().sol_index) < n_all and "1 of" not in v.g_cfg_md.content and "4 of 8" in v.g_cfg_md.content
    v.select_nearest((0.4, 0.0, 0.3))
    assert "elbow up" in v.g_sol_md.content
    v._cfg_clicked(1, "○ down")
    assert "8 of 8" in v.g_cfg_md.content
    v.mount("shoulder3")
    assert v.g_skeleton is None  # no meshes: the skeleton is the drawing
