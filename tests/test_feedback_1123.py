"""Zach's 2026-11-23 viewer feedback: joint labels, sweep precedence and ranges, Show "All", colour by
consistency, angular resolution, deselect, URDF meshes and the robot-model controls (visual /
collision / opacity / parts in contact)."""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

import robot_reach as rr
from robot_reach import Code, OrientationSet
from robot_reach.joint_space import classify_sweep, joint_label, joint_labels, joint_map, sweep_ranges
from robot_reach.overlay import EMPTY_RGB, colors_and_legend, consistency, default_show, show_mask, show_options
from robot_reach.reach import ReachMap
from robot_reach.robots import rr as rr_arm
from robot_reach.robots import shoulder3, ur5e
from robot_reach.urdf import arm_from_urdf, load_urdf, resolve_mesh

viser = pytest.importorskip("viser")
pytest.importorskip("trimesh")

# A planar 2R (x-z plane) whose links carry box geometry, plus a finger hanging off the tool on a
# revolute joint that is NOT on the chain (it must ride along at its zero).
L1, L2 = 0.30, 0.25
BOXED_RR = f"""<?xml version="1.0"?>
<robot name="boxed_rr">
  <material name="blue"><color rgba="0 0 1 1"/></material>
  <link name="base_link"/>
  <link name="upper">
    <visual><origin xyz="{L1 / 2} 0 0"/><geometry><box size="{L1} 0.04 0.04"/></geometry><material name="blue"/></visual>
    <collision><origin xyz="{L1 / 2} 0 0"/><geometry><box size="{L1} 0.05 0.05"/></geometry></collision>
  </link>
  <link name="fore">
    <visual><origin xyz="{L2 / 2} 0 0"/><geometry><box size="{L2} 0.04 0.04"/></geometry></visual>
    <collision><origin xyz="{L2 / 2} 0 0"/><geometry><box size="{L2} 0.05 0.05"/></geometry></collision>
  </link>
  <link name="tool"/>
  <link name="finger"><visual><geometry><sphere radius="0.01"/></geometry></visual></link>
  <joint name="shoulder" type="revolute">
    <parent link="base_link"/><child link="upper"/><axis xyz="0 -1 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="1" velocity="1"/>
  </joint>
  <joint name="elbow" type="revolute">
    <parent link="upper"/><child link="fore"/><origin xyz="{L1} 0 0"/><axis xyz="0 -1 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="1" velocity="1"/>
  </joint>
  <joint name="tool_joint" type="fixed"><parent link="fore"/><child link="tool"/><origin xyz="{L2} 0 0"/></joint>
  <joint name="finger_joint" type="revolute">
    <parent link="tool"/><child link="finger"/><origin xyz="0.02 0 0"/><axis xyz="0 0 1"/>
    <limit lower="0.0" upper="0.8" effort="1" velocity="1"/>
  </joint>
</robot>
"""


@pytest.fixture(scope="module")
def boxed(tmp_path_factory):
    path = tmp_path_factory.mktemp("urdf") / "boxed_rr.urdf"
    path.write_text(BOXED_RR)
    return path


# ---------------------------------------------------------------- 3. joint labels
def test_joint_labels_append_the_chain_index():
    names = ur5e().joint_names
    assert joint_label(names, 1) == f"{names[1]} (J2)"
    assert joint_labels(("a", "b"))[1] == "b (J2)"
    jm = joint_map(ur5e(), (1, 2), n=5)
    assert jm.labels == (f"{names[1]} (J2)", f"{names[2]} (J3)")
    from robot_reach.plots import figure_name

    assert figure_name(jm) == f"{names[1]} (J2) × {names[2]} (J3): slice"


# ---------------------------------------------------------------- 4. sweep precedence + ranges
def test_sweep_cells_keep_always_singular_and_always_collision_apart():
    S, C, OK, LIM = int(Code.SINGULAR), int(Code.COLLISION), int(Code.OK), int(Code.JOINT_LIMIT)
    codes = np.array([
        [S, S, S],      # always singular
        [C, C, C],      # always collision
        [C, S, LIM],    # never OK, mixed: worst present is singular, not uniform
        [OK, C, S],     # OK for some
        [OK, OK, OK],   # OK for all
    ])
    cell, frac, uniform = classify_sweep(codes)
    assert cell.tolist() == [S, C, S, OK, OK]
    assert uniform.tolist() == [True, True, False, False, True]
    assert np.allclose(frac, [0, 0, 0, 1 / 3, 1])


def test_sweep_samples_only_the_chosen_ranges_and_says_so():
    arm = ur5e()
    full = sweep_ranges(arm, (1, 2))
    assert np.isnan(full[1]).all() and np.allclose(full[0], arm.limits[0])
    narrow = sweep_ranges(arm, (1, 2), {0: (-0.5, 0.5), 3: (9.0, -9.0)})  # reversed + past the limit: sorted, clipped
    assert np.allclose(narrow[0], (-0.5, 0.5)) and np.allclose(narrow[3], arm.limits[3])
    jm = joint_map(arm, (1, 2), mode="sweep", n=7, n_sweep=6, other_ranges={0: (-0.5, 0.5)})
    assert jm.n_sweep == 6 and np.allclose(jm.other_ranges[0], (-0.5, 0.5))
    text = jm.sweep_text(arm.joint_names, arm.limits)
    assert "6 postures" in text and "(J1) -29…29°" in text and "(J4) -360…360° (full limits)" in text


def test_sweep_fields_survive_save_and_load(tmp_path):
    res = rr.analyze(shoulder3(), floor_z=0.0, step=0.2)
    res.save(tmp_path / "s3")
    back = rr.load(tmp_path / "s3")
    sweep, sweep_back = (next(m for m in r.joint_maps if m.mode == "sweep") for r in (res, back))
    assert sweep_back.n_sweep == sweep.n_sweep > 0
    assert np.array_equal(sweep_back.uniform, sweep.uniform)
    assert np.allclose(sweep_back.other_ranges, sweep.other_ranges, equal_nan=True)


# ---------------------------------------------------------------- 6 + 7. Show "All", consistency
def _tiny_map() -> ReachMap:
    """4 points x 1 orientation x 2 branches: nothing / matched / half mismatched / all mismatched."""
    P, OK, MIS, NO, PAD, COL = int(Code.PAD), int(Code.OK), int(Code.FK_MISMATCH), int(Code.NO_SOLUTION), int(Code.PAD), int(Code.COLLISION)
    code = np.array([[[NO, PAD]], [[OK, COL]], [[OK, MIS]], [[MIS, MIS]]], np.int8)
    flat = np.nonzero(code.reshape(-1) == OK)[0]
    return ReachMap("t", ("a", "b"), "position", np.eye(4)[:, :3], OrientationSet.identity(), code, flat,
                    np.zeros((len(flat), 2), np.float32))


def test_consistency_says_whether_fk_of_ik_landed_back():
    assert consistency(_tiny_map()).tolist() == [0, 1, 2, 3]


def test_show_all_includes_points_with_zero_solutions_and_colours_them_empty():
    m = _tiny_map()
    assert show_mask(m, "all").all()
    assert show_mask(m, "reachable").tolist() == [False, True, True, False]
    colors, legend = colors_and_legend(m, "IK solutions", np.arange(4))
    assert tuple(colors[0]) == EMPTY_RGB and tuple(colors[3]) == EMPTY_RGB
    assert "0" in legend


def test_show_follows_colour_by():
    m = _tiny_map()
    for color_by in ("IK solutions", "consistency"):
        opts = show_options(m, color_by)
        assert list(opts)[0] == "All" and default_show(m, color_by) == list(opts)[1]
        for key in opts.values():
            show_mask(m, key)  # every offered key is a valid mask
    assert show_mask(m, show_options(m, "consistency")["any FK(IK) mismatch"]).tolist() == [False, False, True, True]
    colors, legend = colors_and_legend(m, "consistency", np.arange(4))
    assert "FK(IK) matched" in legend and len({tuple(c) for c in colors}) == 4
    with pytest.raises(ValueError):
        show_mask(m, "bogus")


# ---------------------------------------------------------------- 5. angular resolution
def test_orientation_set_states_its_resolution():
    r = OrientationSet.sample(tilt_step_deg=60.0, n_rolls=4).resolution()
    assert (r["tilt_step_deg"], r["roll_step_deg"], r["n_rolls"], r["n_directions"], r["n_orientations"]) == (60.0, 90.0, 4, 12, 48)
    assert OrientationSet.sample().describe().startswith("approach -z: tilt 0…180° from it in 45° rings (22 directions)")


# ---------------------------------------------------------------- 1. URDF meshes
def test_urdf_meshes_pose_with_the_chain_and_off_chain_links_ride_along(boxed):
    arm = arm_from_urdf(boxed, "tool", meshes=True)
    parts = [(m.part, m.draw, m.measure) for m in arm.meshes]
    assert ("upper", True, False) in parts and ("upper", False, True) in parts and ("finger", True, False) in parts
    assert arm.meshes[[m.part for m in arm.meshes].index("upper")].color == (0, 0, 255)
    q = np.array([[np.pi / 2, 0.0]])  # straight up
    T = arm.mesh_poses(q)[0]
    finger = [m.part for m in arm.meshes].index("finger")
    tool = arm.forward(q)[0][0]
    assert np.allclose(T[finger, :3, 3], tool[:3, 3] + tool[:3, :3] @ [0.02, 0, 0], atol=1e-9)
    # the posed upper-link box spans the upper link: 0 .. L1 up the z axis
    z = arm.surface(q, parts=["upper"])[0][:, 2]
    assert np.isclose(z.min(), 0.0, atol=0.03) and np.isclose(z.max(), L1, atol=0.03)


def test_mesh_paths_resolve_package_uris(tmp_path):
    (tmp_path / "assets" / "pkg" / "meshes").mkdir(parents=True)
    target = tmp_path / "assets" / "pkg" / "meshes" / "a.stl"
    target.write_text("solid a\nendsolid a\n")
    assert resolve_mesh("package://pkg/meshes/a.stl", tmp_path) == target
    assert resolve_mesh("file://" + str(target), None) == target
    assert resolve_mesh("package://pkg/missing.stl", tmp_path) is None


def test_unreadable_mesh_falls_back_and_says_why(tmp_path):
    bad = BOXED_RR.replace('<box size="0.3 0.04 0.04"/>', '<mesh filename="package://nowhere/x.dae"/>')
    path = tmp_path / "bad.urdf"
    path.write_text(bad)
    arm = arm_from_urdf(path, "tool", meshes=True)
    assert "upper visual" in arm.meta["mesh_note"] and "drawn with collision meshes: upper" in arm.meta["mesh_note"]
    assert any(m.part == "upper" and m.draw and m.measure for m in arm.meshes)


def test_composed_ur5e_2f85_reaches_with_the_hand(tmp_path):
    from robot_reach.examples import ROBOTIQ_2F85_TCP_Z, ROBOTIQ_2F85_URDF, UR5E_URDF, compose_ur5e_2f85

    if not (UR5E_URDF.exists() and ROBOTIQ_2F85_URDF.exists()):
        pytest.skip("stock URDFs not on this machine")
    path = compose_ur5e_2f85(tmp_path)
    hand = arm_from_urdf(path, "tcp")
    bare = arm_from_urdf(UR5E_URDF, "tool0")
    q = np.array([[0.3, -1.2, 1.1, -0.4, 1.3, 0.2]])
    T_hand, T_tool0 = hand.forward(q)[0][0], bare.forward(q)[0][0]
    assert hand.dof == 6 and np.allclose(T_hand[:3, 3], T_tool0[:3, 3] + T_tool0[:3, 2] * ROBOTIQ_2F85_TCP_Z, atol=1e-9)
    assert "robotiq_85_base_link" in load_urdf(path).links


# ---------------------------------------------------------------- 9. robot model: visual / collision / contact
def test_model_shows_styles_and_contacts(boxed):
    from robot_reach.viewer import COLLISION_MESH_RGB, collision_note, contact_parts, floor_contacts, model_shows, model_styles

    arm = arm_from_urdf(boxed, "tool", meshes=True)
    assert model_shows(arm.meshes) == ("visual", "collision", "both", "none")
    pts = np.array([[0, 0, 0], [0, 0, -0.1], [0.2, 0, 0.1]])
    assert floor_contacts(pts, 0.0).tolist() == [False, True, False]
    assert floor_contacts(pts, None).tolist() == [False, False, False]
    q_down = np.array([0.0, -np.pi / 2])  # forearm hangs down below z = 0
    assert np.asarray(arm.collides(q_down[None], floor_z=0.0))[0]
    assert contact_parts(arm, q_down, 0.0) == {"fore", "tool"}  # the tool point is under; the forearm took it there
    styles = dict(zip([(m.part, m.draw) for m in arm.meshes], model_styles(arm.meshes, "visual", {"fore"})))
    assert styles[("fore", False)] == (True, (220, 53, 69))  # the collision mesh in contact shows, red
    assert styles[("upper", False)] == (False, COLLISION_MESH_RGB)  # others follow Show
    assert styles[("fore", True)] == (True, (220, 53, 69))
    assert all(not v for v, _ in model_styles(arm.meshes, "none", set()))
    note = collision_note(arm, 0.0)
    assert "skeleton" in note and "collision meshes are drawn, not checked" in note
    assert collision_note(rr_arm(), None) == "No collision is checked for this arm."


def test_ray_helpers():
    from robot_reach.viewer import ray_hits_boxes, ray_near_points

    centers = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    assert ray_hits_boxes((-3, 0, 0), (1, 0, 0), centers, 0.1).tolist() == [True, False]
    assert not ray_hits_boxes((3, 0, 0), (1, 0, 0), centers, 0.1).any()  # box behind the ray
    assert ray_near_points((0, 0, 1), (0, 0, -1), [[0.01, 0, 0]], 0.02)
    assert not ray_near_points((0, 0, 1), (0, 0, -1), [[0.5, 0, 0]], 0.02)


# ---------------------------------------------------------------- 2. deselect, and the viewer wiring
@pytest.fixture(scope="module")
def viewer(boxed):
    from robot_reach.viewer import ReachViewer

    arm = arm_from_urdf(boxed, "tool", meshes=True)
    small = rr.analyze(arm, floor_z=-0.05, step=0.05)
    ur = rr.analyze(ur5e(), floor_z=0.0, step=0.25, orientations=OrientationSet.sample(tilt_step_deg=90, n_rolls=4))
    v = ReachViewer({"boxed": small, "ur5e": ur})
    yield v
    v.stop()


def test_escape_and_empty_click_deselect_and_hide_the_arm(viewer):
    from robot_reach.viewer import CLEAR_LABEL, ESCAPE_JS

    v = viewer
    v.mount("ur5e")
    assert f"'{CLEAR_LABEL}'" in ESCAPE_JS and "Escape" in ESCAPE_JS
    p = v.select_nearest((0.4, 0.0, 0.3))
    assert v.arm_q is not None and "orientations" in v.g_res_md.content.lower()
    assert "J1 " in v.g_sol_md.content
    # a click whose ray passes through the selected voxel keeps the selection
    pos = v.reach.positions[p]
    v._scene_clicked(types.SimpleNamespace(ray_origin=tuple(pos + [3.0, 0, 0]), ray_direction=(-1.0, 0.0, 0.0)))
    assert v.sel_point == p
    # a click into empty sky deselects: no point, arm hidden, voxels undimmed
    v._scene_clicked(types.SimpleNamespace(ray_origin=(5.0, 5.0, 5.0), ray_direction=(0.0, 0.0, 1.0)))
    assert v.sel_point is None and v.arm_q is None and not v.overlay._dim
    # "Arm when nothing is selected" keeps it at the Jog pose instead
    v.select_nearest((0.4, 0.0, 0.3))
    v.g_arm_idle.value = True
    v._clear_selection()
    assert v.sel_point is None and v.arm_q is not None
    v.g_arm_idle.value = False
    v._idle_arm()
    assert v.arm_q is None


def test_viewer_show_all_consistency_and_labels(viewer):
    v = viewer
    v.mount("ur5e")
    assert v.g_jog[1].label.endswith("(J2) (deg)")
    v.g_show.value = "All"
    v._redraw_workspace()
    assert len(v.visible_index) == v.reach.n_positions
    v.g_color.value = "consistency"
    v._color_changed()
    assert v.g_show.value == "All" and "IK answered (checked)" in v.g_show.options
    assert "coloured by consistency" in v.g_legend.content
    assert "FK cloud" in v.g_cloud_md.content and "grid the voxel workspace" in v.g_cloud_md.content
    sweep = next(n for n in v._figures if n.endswith(": sweep"))
    v.g_fig.value = sweep
    v._picture_changed()
    assert "postures of the other joints" in v.g_fig_md.content and v.g_sweep_folder.visible
    v.g_sweep_ranges[0].value = (-30.0, 30.0)
    v.g_sweep_n.value = 8
    v._recompute_sweep()
    assert "J1) -30…30°" in v.g_fig_md.content and "8 postures" in v.g_fig_md.content
    assert "Compute box / Drop POI try" in v.g_ask_res_md.content


def test_viewer_robot_model_controls(viewer):
    v = viewer
    v.mount("boxed")
    assert v.g_model.options == ("visual", "collision", "both", "none")
    # jog the forearm down through the floor: collision, the contact red and listed
    v.g_jog[0].value = 0.0
    v.g_jog[1].value = -90.0
    v._jog_changed()
    assert "collision" in v.g_jog_md.content and "in contact" in v.g_jog_md.content and "tool" in v.contact
    handles = {v.arm.meshes[i].part + ("/c" if not v.arm.meshes[i].draw else ""): h for i, h in v.mesh_handles}
    assert handles["upper"].visible and not handles["upper/c"].visible
    v.g_opacity.value = 0.4
    v._pose_meshes()
    assert np.isclose(handles["upper"].opacity, 0.4)
    v.g_model.value = "none"
    v._model_changed()
    assert not handles["upper"].visible
