"""URDF adapter: FK straight from URDF XML, generic multi-branch numeric IK, chains, front doors.

Oracles: the closed-form UR5e (`robot_reach.robots.ur5e`, DH + Hawkins IK) and the planar 2R
(`robot_reach.robots.rr`, `_rr_ik`) — written independently of the URDF path.
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path

import numpy as np
import pytest

import robot_reach as rr
from robot_reach import Code
from robot_reach.analyze import rebuild
from robot_reach.robots import ur5e
from robot_reach.robots.planar import _rr_ik
from robot_reach.robots.planar import rr as rr_arm
from robot_reach.urdf import arm_from_description, arm_from_urdf, chains, load_urdf, rpy_matrix

UR5E_URDF = Path(__file__).parent / "fixtures" / "ur5e.urdf"
L1, L2 = 0.30, 0.25

# A planar 2R in the x-z plane, angles from +x toward +z (axis -y), as robot_reach.robots.rr.
RR_URDF = f"""<?xml version="1.0"?>
<robot name="planar_rr">
  <link name="base_link"/>
  <link name="upper"/>
  <link name="fore"/>
  <link name="tool"/>
  <joint name="shoulder" type="revolute">
    <parent link="base_link"/>
    <child link="upper"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 -1 0"/>
    <limit lower="-3.141592653589793" upper="3.141592653589793" effort="10" velocity="3"/>
  </joint>
  <joint name="elbow" type="revolute">
    <parent link="upper"/>
    <child link="fore"/>
    <origin xyz="{L1} 0 0"/>
    <axis xyz="0 -1 0"/>
    <limit lower="-3.141592653589793" upper="3.141592653589793" effort="10" velocity="3"/>
  </joint>
  <joint name="tool_joint" type="fixed">
    <parent link="fore"/>
    <child link="tool"/>
    <origin xyz="{L2} 0 0"/>
  </joint>
</robot>
"""

# The 2R plus a camera bolted to the upper link: a second, 1-DOF chain.
RR_WITH_CAMERA = RR_URDF.replace(
    "</robot>",
    """  <link name="camera"/>
  <joint name="camera_mount" type="fixed">
    <parent link="upper"/>
    <child link="camera"/>
    <origin xyz="0.1 0 0.05"/>
  </joint>
</robot>""",
)

# Prismatic (non-unit axis), continuous, and a mimic joint driven by the continuous one.
MIXED_URDF = """<robot name="mixed">
  <link name="world"/><link name="carriage"/><link name="turret"/><link name="finger"/><link name="tip"/>
  <joint name="slide" type="prismatic">
    <parent link="world"/><child link="carriage"/>
    <origin xyz="0.1 0 0.2"/><axis xyz="0 0 2"/>
    <limit lower="-0.1" upper="0.4" effort="1" velocity="1"/>
  </joint>
  <joint name="spin" type="continuous">
    <parent link="carriage"/><child link="turret"/>
    <origin xyz="0 0 0.05"/><axis xyz="0 0 1"/>
  </joint>
  <joint name="follower" type="revolute">
    <parent link="turret"/><child link="finger"/>
    <origin xyz="0.2 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
    <mimic joint="spin" multiplier="-1" offset="0.3"/>
  </joint>
  <joint name="tip_joint" type="fixed">
    <parent link="finger"/><child link="tip"/><origin xyz="0.1 0 0"/>
  </joint>
</robot>
"""


def _inv(T: np.ndarray) -> np.ndarray:
    out = np.zeros_like(T)
    R = np.swapaxes(T[..., :3, :3], -1, -2)
    out[..., :3, :3] = R
    out[..., :3, 3] = -(R @ T[..., :3, 3, None])[..., 0]
    out[..., 3, 3] = 1.0
    return out


def _rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    cos = (np.trace(np.swapaxes(Ra, -1, -2) @ Rb, axis1=-2, axis2=-1) - 1) / 2
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _wrapped_gap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Largest per-joint angular difference, 2pi twins counted as equal (all-revolute arms)."""
    return np.max(np.abs((a - b + np.pi) % (2 * np.pi) - np.pi), axis=-1)


def _uniform_q(arm: rr.Arm, n: int, seed: int) -> np.ndarray:
    lim = arm.limits
    return lim[:, 0] + np.random.default_rng(seed).random((n, arm.dof)) * (lim[:, 1] - lim[:, 0])


# --------------------------------------------------------------------------- (a) FK vs the DH UR5e
def test_ur5e_urdf_fk_matches_closed_form_up_to_fixed_base_and_tool_frames():
    arm = arm_from_urdf(UR5E_URDF, "tool0")
    oracle = ur5e()
    assert arm.joint_names == (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    np.testing.assert_allclose(arm.limits, oracle.limits, atol=1e-12)
    assert arm.task == "pose"

    # The two constants, from one sample q0: the base constant is where the URDF puts the
    # controller's base frame (base_link_inertia); the tool constant then falls out of one pose.
    chain = load_urdf(UR5E_URDF).chain("tool0")
    q0 = _uniform_q(arm, 1, seed=11)
    frames = chain.link_frames(q0)[0]
    base = frames[chain.links.index("base_link_inertia")]
    tool = _inv(base @ oracle.fk(q0)[0]) @ arm.fk(q0)[0]

    q = _uniform_q(arm, 200, seed=12)
    predicted = base @ oracle.fk(q) @ tool
    np.testing.assert_allclose(arm.fk(q), predicted, atol=1e-9, rtol=0)
    # And they are the values UR documents: base_link is the controller base turned 180 deg about Z,
    # tool0 is the DH flange frame (to the 2e-10 rad the yaml's rounded pi/2 leaves).
    np.testing.assert_allclose(base, np.diag([-1.0, -1.0, 1.0, 1.0]), atol=1e-12)
    np.testing.assert_allclose(tool, np.eye(4), atol=1e-8)


def test_rpy_is_fixed_axis_roll_then_pitch_then_yaw():
    def rx(a):
        return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])

    def ry(a):
        return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])

    for roll, pitch, yaw in np.random.default_rng(13).uniform(-np.pi, np.pi, (20, 3)):
        np.testing.assert_allclose(rpy_matrix(roll, pitch, yaw), _rz(yaw) @ ry(pitch) @ rx(roll), atol=1e-15)


def test_skeleton_walks_every_link_frame_from_base_to_tip():
    arm = arm_from_urdf(UR5E_URDF, "tool0")
    chain = load_urdf(UR5E_URDF).chain("tool0")
    assert chain.links[0] == "base_link" and chain.links[-1] == "tool0"
    q = _uniform_q(arm, 50, seed=3)
    pts = arm.skeleton(q)
    assert pts.shape == (50, len(chain.links), 3)
    np.testing.assert_allclose(pts[:, 0], 0.0, atol=1e-15)
    np.testing.assert_allclose(pts[:, -1], arm.fk(q)[:, :3, 3], atol=1e-12)
    np.testing.assert_allclose(pts, chain.link_frames(q)[..., :3, 3], atol=1e-15)


def test_tool_offset_is_appended_at_the_tip():
    offset = np.eye(4)
    offset[:3, 3] = [0.0, 0.0, 0.12]
    offset[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    plain = arm_from_urdf(UR5E_URDF, "tool0")
    tooled = arm_from_urdf(UR5E_URDF, "tool0", tool_offset=offset)
    q = _uniform_q(plain, 40, seed=4)
    np.testing.assert_allclose(tooled.fk(q), plain.fk(q) @ offset, atol=1e-12)
    pts = tooled.skeleton(q)
    assert pts.shape[1] == plain.skeleton(q).shape[1] + 1
    np.testing.assert_allclose(pts[:, -1], tooled.fk(q)[:, :3, 3], atol=1e-12)
    assert tooled.meta["factory"][2]["tool_offset"] == offset.tolist()


@pytest.mark.parametrize("source,tip", [(UR5E_URDF, "tool0"), (MIXED_URDF, "tip")])
def test_analytic_jacobian_matches_finite_differences(source, tip):
    arm = arm_from_urdf(source, tip)
    chain = load_urdf(source).chain(tip)
    q = _uniform_q(arm, 30, seed=5)
    T, J = chain.fk_jacobian(q)
    np.testing.assert_allclose(T, arm.fk(q), atol=1e-14)
    np.testing.assert_allclose(J, arm.jacobian(q), atol=2e-5)


# --------------------------------------------------------------------------- (b) numeric IK, UR5e
def test_numeric_ik_ur5e_round_trip_finds_several_genuine_branches():
    arm = arm_from_urdf(UR5E_URDF, "tool0")  # no ik= -> the generic numeric solver
    assert arm.ik is not None
    q_true = _uniform_q(arm, 300, seed=21)
    T = arm.fk(q_true)

    t0 = time.perf_counter()
    q, code = arm.inverse(T)
    seconds = time.perf_counter() - t0
    n, S, dof = q.shape
    assert S == 8 and dof == 6

    # Slot layout: OK branches first, then PAD; NO_SOLUTION only in slot 0 of an empty row.
    ok = code == Code.OK
    assert set(np.unique(code)) <= {Code.OK, Code.NO_SOLUTION, Code.PAD}
    assert np.all(np.diff(ok.astype(int), axis=1) <= 0)
    assert np.all((code[:, 1:] == Code.OK) | (code[:, 1:] == Code.PAD))
    assert np.all((code[:, 0] == Code.NO_SOLUTION) == ~ok.any(axis=1))

    # Round trip: at least one OK branch reproduces the pose to 1e-6 m / 1e-5 rad.
    Tq = arm.fk(q.reshape(-1, dof)).reshape(n, S, 4, 4)
    perr = np.linalg.norm(Tq[..., :3, 3] - T[:, None, :3, 3], axis=-1)
    rerr = _rot_angle(Tq[..., :3, :3], T[:, None, :3, :3])
    exact = ok & (perr < 1e-6) & (rerr < 1e-5)
    success = exact.any(axis=1).mean()
    branches = ok.sum(axis=1).mean()
    assert np.all(exact == ok), "every branch reported OK must reproduce the pose"
    assert success >= 0.98, success
    assert branches >= 2.0, branches

    # Branches are distinct, inside the limits, and each is one of the closed-form UR5e's eight.
    for a in range(S):
        for b in range(a + 1, S):
            both = ok[:, a] & ok[:, b]
            assert np.all(_wrapped_gap(q[both, a], q[both, b]) > 1e-4)
    assert np.all(arm.within_limits(q[ok]))
    oracle = ur5e()
    chain = load_urdf(UR5E_URDF).chain("tool0")
    base = chain.link_frames(np.zeros((1, 6)))[0, chain.links.index("base_link_inertia")]
    tool = _inv(base @ oracle.fk(np.zeros((1, 6)))[0]) @ arm.fk(np.zeros((1, 6)))[0]
    q_cf, code_cf = oracle.ik(_inv(base) @ T @ _inv(tool))
    rows, cols = np.nonzero(ok)
    gaps = _wrapped_gap(q[rows, cols][:, None, :], q_cf[rows])  # (K, 8)
    gaps = np.where(code_cf[rows] == Code.OK, gaps, np.inf)
    genuine = (gaps.min(axis=1) < 1e-5).mean()
    assert genuine >= 0.99, genuine
    print(
        f"\nUR5e numeric IK: 300 poses x 8 seeds in {seconds:.2f} s; success {success:.1%}; "
        f"mean distinct branches {branches:.2f}; matched to a closed-form branch {genuine:.1%}"
    )


def test_numeric_ik_handles_5k_targets_in_reasonable_time():
    arm = arm_from_urdf(UR5E_URDF, "tool0")
    rng = np.random.default_rng(31)
    reachable = arm.fk(_uniform_q(arm, 2500, seed=32))
    # Half the targets are random poses in a 2 m box: mostly out of reach, as on a reach-map grid.
    wild = np.tile(np.eye(4), (2500, 1, 1))
    wild[:, :3, :3] = reachable[rng.permutation(2500), :3, :3]
    wild[:, :3, 3] = rng.uniform(-1.0, 1.0, (2500, 3))
    T = np.concatenate([reachable, wild])
    t0 = time.perf_counter()
    q, code = arm.inverse(T)
    seconds = time.perf_counter() - t0
    found = (code == Code.OK).any(axis=1)
    assert found[:2500].mean() >= 0.98
    assert seconds < 120.0
    print(
        f"\nUR5e numeric IK: 5000 targets x 8 seeds in {seconds:.2f} s "
        f"(reachable half solved {found[:2500].mean():.1%}, wild half {found[2500:].mean():.1%})"
    )


def test_numeric_ik_axis_task_on_a_5dof_arm_ignores_roll():
    # The UR5e with wrist 3 welded: 5 DOF, so the default task is position + tool +Z.
    xml = UR5E_URDF.read_text().replace('name="wrist_3_joint" type="revolute"', 'name="wrist_3_joint" type="fixed"')
    arm = arm_from_urdf(xml, "tool0")
    assert arm.dof == 5 and arm.task == "axis"
    rng = np.random.default_rng(51)
    T = arm.fk(_uniform_q(arm, 200, seed=52))
    roll = rng.uniform(-np.pi, np.pi, 200)  # spin each target about its own +Z: still the same axis task
    spin = np.zeros((200, 4, 4))
    spin[:, 0, 0], spin[:, 0, 1], spin[:, 1, 0], spin[:, 1, 1] = np.cos(roll), -np.sin(roll), np.sin(roll), np.cos(roll)
    spin[:, 2, 2] = spin[:, 3, 3] = 1.0
    T = T @ spin
    q, code = arm.inverse(T)
    ok = code == Code.OK
    assert ok.any(axis=1).mean() >= 0.98
    Tq = arm.fk(q[ok])
    target = np.repeat(T[:, None], q.shape[1], axis=1)[ok]
    np.testing.assert_allclose(Tq[:, :3, 3], target[:, :3, 3], atol=1e-6)
    np.testing.assert_allclose(Tq[:, :3, 2], target[:, :3, 2], atol=1e-6)


# --------------------------------------------------------------------------- (c) planar 2R
def test_planar_2r_urdf_matches_rr_and_finds_both_elbow_branches():
    arm = arm_from_urdf(RR_URDF, "tool", task="position")
    assert arm.dof == 2 and arm.task == "position"
    assert arm_from_urdf(RR_URDF, "tool").task == "position"  # the default for <= 3 DOF
    assert arm.plane is not None and abs(abs(arm.plane[1][1]) - 1.0) < 1e-12  # the x-z plane
    oracle = rr_arm(l1=L1, l2=L2)
    q = _uniform_q(arm, 100, seed=41)
    np.testing.assert_allclose(arm.fk(q), oracle.fk(q), atol=1e-12)

    rng = np.random.default_rng(42)
    q_true = np.stack([rng.uniform(-2.5, 2.5, 100), rng.choice([-1, 1], 100) * rng.uniform(0.3, 2.5, 100)], -1)
    T = arm.fk(q_true)
    q_num, code = arm.inverse(T)
    ok = code == Code.OK
    q_cf, reach_cf = _rr_ik(T[:, 0, 3], T[:, 2, 3], L1, L2)
    assert reach_cf.all()
    both = 0
    for i in range(len(T)):
        found = q_num[i, ok[i]]
        gaps = _wrapped_gap(found[:, None, :], q_cf[i][None])  # (k, 2)
        assert np.all(gaps.min(axis=1) < 1e-6), "every numeric branch is a closed-form branch"
        both += int(np.all(gaps.min(axis=0) < 1e-6))
    assert both / len(T) >= 0.95, both


# --------------------------------------------------------------------------- (d) analyze end to end
def test_analyze_runs_end_to_end_on_a_urdf_arm(tmp_path):
    arm = arm_from_urdf(RR_URDF, "tool")
    result = rr.analyze(arm, n_fk=4000)
    reach = result.reach
    assert reach is not None and reach.n_positions > 100
    assert reach.task == "position" and reach.n_orientations == 1
    r = np.hypot(reach.positions[:, 0], reach.positions[:, 2])
    np.testing.assert_allclose(reach.positions[:, 1], 0.0, atol=1e-12)  # the grid collapsed onto the plane
    reachable = reach.workspace("reachable")
    inner, outer = abs(L1 - L2), L1 + L2
    inside = (r > inner + 0.01) & (r < outer - 0.01)
    outside = (r < inner - 0.01) | (r > outer + 0.01)
    assert inside.sum() > 50 and outside.sum() > 50
    assert not reachable[outside].any()
    assert reachable[inside].mean() > 0.97
    assert result.summary()["reach"]["ok_solutions"] > 0
    folder = result.save(tmp_path / "rr_urdf")
    loaded = rr.load(folder)
    assert loaded.reach.n_positions == reach.n_positions
    assert loaded.arm is None  # an in-memory URDF string records no factory to rebuild from


def test_file_urdf_records_a_factory_that_rebuilds_the_arm():
    arm = arm_from_urdf(UR5E_URDF, "tool0")
    module, function, kwargs = arm.meta["factory"]
    assert (module, function) == ("robot_reach.urdf", "arm_from_urdf")
    assert kwargs["urdf"] == str(UR5E_URDF.resolve()) and kwargs["tip_link"] == "tool0"
    json.dumps(arm.meta["factory"])
    again = rebuild({"factory": arm.meta["factory"]})
    q = _uniform_q(arm, 10, seed=6)
    np.testing.assert_allclose(again.fk(q), arm.fk(q), atol=1e-15)
    assert arm.meta["home"] == [0.0] * 6
    assert arm.meta["revolute"] == [True] * 6
    assert "factory" not in arm_from_urdf(UR5E_URDF.read_text(), "tool0").meta


# --------------------------------------------------------------------------- (e) chains
def test_chains_lists_every_movable_leaf():
    assert chains(UR5E_URDF) == [("base_link", "tool0", 6)]  # 'base' hangs off a fixed joint: 0 DOF
    assert chains(RR_URDF) == [("base_link", "tool", 2)]
    assert sorted(chains(RR_WITH_CAMERA)) == [("base_link", "camera", 1), ("base_link", "tool", 2)]
    camera = arm_from_urdf(RR_WITH_CAMERA, "camera")
    assert camera.joint_names == ("shoulder",)


# --------------------------------------------------------------------------- joint kinds, errors, doors
def test_prismatic_continuous_and_mimic_joints():
    arm = arm_from_urdf(MIXED_URDF, "tip")
    assert arm.joint_names == ("slide", "spin")  # the mimic joint is driven, not a DOF
    np.testing.assert_allclose(arm.limits, [[-0.1, 0.4], [-np.pi, np.pi]])  # continuous -> +-pi
    assert arm.meta["revolute"] == [False, True]
    assert arm.task == "position"
    s, th = np.array([0.0, 0.25, -0.05]), np.array([0.0, 1.0, -2.0])
    T = arm.fk(np.stack([s, th], -1))
    expected = np.stack(
        [0.1 + 0.2 * np.cos(th) + 0.1 * np.cos(0.3), 0.2 * np.sin(th) + 0.1 * np.sin(0.3), 0.25 + s], -1
    )
    np.testing.assert_allclose(T[:, :3, 3], expected, atol=1e-12)
    np.testing.assert_allclose(T[:, :3, :3], np.broadcast_to(_rz(0.3), (3, 3, 3)), atol=1e-12)
    q, code = arm.inverse(T)
    assert np.all(code[:, 0] == Code.OK)
    Tq = arm.fk(q[:, 0])
    np.testing.assert_allclose(Tq[:, :3, 3], T[:, :3, 3], atol=1e-7)
    assert chains(MIXED_URDF) == [("world", "tip", 2)]


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_bad_chains_say_what_is_wrong():
    with pytest.raises(ValueError, match="tool0"):
        arm_from_urdf(UR5E_URDF, "no_such_link")  # the message lists the candidate tips
    with pytest.raises(ValueError, match="not an ancestor"):
        arm_from_urdf(UR5E_URDF, "tool0", base_link="base")
    with pytest.raises(ValueError, match="no movable joint"):
        arm_from_urdf(UR5E_URDF, "base")


def test_arm_from_description_is_duck_typed():
    class Description:  # stands in for bam_robot_descriptions.urdf.RobotDescription
        def to_urdf_string(self) -> str:
            return RR_URDF

    arm = arm_from_description(Description(), "tool")
    assert arm.dof == 2 and "factory" not in arm.meta
    assert chains(Description()) == [("base_link", "tool", 2)]
    q = _uniform_q(arm, 5, seed=7)
    np.testing.assert_allclose(arm.fk(q), arm_from_urdf(RR_URDF, "tool").fk(q))


def test_analytic_ik_can_be_passed_in():
    def ik(T):
        q, ok = _rr_ik(T[:, 0, 3], T[:, 2, 3], L1, L2)
        return q, np.where(ok[:, None], Code.OK, Code.NO_SOLUTION).repeat(2, axis=1).astype(np.int8)

    arm = arm_from_urdf(RR_URDF, "tool", ik=ik)
    assert arm.ik is ik


def test_urdf_module_imports_only_stdlib_numpy_and_robot_reach():
    source = (Path(rr.__file__).parent / "urdf.py").read_text()
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert roots <= {"__future__", "collections", "dataclasses", "pathlib", "typing", "xml", "math", "numpy"}, roots


def test_analyze_chains_gives_one_result_per_chain_and_the_cli_runs(tmp_path):
    from robot_reach.__main__ import main
    from robot_reach.urdf import analyze_chains

    results = analyze_chains(RR_WITH_CAMERA)
    assert sorted(results) == ["camera (1 DOF)", "tool (2 DOF)"]
    assert results["tool (2 DOF)"].reach is not None  # a 2-DOF planar chain gets a grid
    assert results["camera (1 DOF)"].reach is None  # a 1-DOF chain is a curve: FK cloud only
    with pytest.raises(ValueError):
        analyze_chains(RR_WITH_CAMERA, tips=["nope"])
    urdf = tmp_path / "rr.urdf"
    urdf.write_text(RR_URDF)
    assert main(["urdf", str(urdf), "--out", str(tmp_path / "out"), "--no-view"]) == 0
    assert (tmp_path / "out" / "urdf" / "rr" / "tool" / "result.json").exists()
