"""The contract, tested on the five example arms.

Independent oracles where they exist: FK of the IK answer (the round trip) and recovery of the
joint vector that produced the pose; for the UR5e, the stdlib ur_kin.cpp port in bam_ws when present.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import robot_reach as rr
from robot_reach import Box, Code, OrientationSet, joint_map, reach_map
from robot_reach.robots import five_bar, pendulum, rr as rr_arm, shoulder3, ur5e
from robot_reach.sampling import sample_joints

ARMS = [pendulum, rr_arm, five_bar, shoulder3, ur5e]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@pytest.mark.parametrize("factory", ARMS, ids=lambda f: f.__name__)
def test_every_ok_branch_round_trips_and_one_recovers_the_original(factory):
    arm = factory()
    q = sample_joints(arm, 500, seed=3)
    T, ok = arm.forward(q)
    q, T = q[ok], T[ok]
    Q, C = arm.inverse(T)
    S = Q.shape[1]
    T2, _ = arm.forward(Q.reshape(-1, arm.dof))
    T2 = T2.reshape(len(q), S, 4, 4)
    good = C == Code.OK
    perr = np.linalg.norm(T2[..., :3, 3] - T[:, None, :3, 3], axis=-1)
    assert np.all(perr[good] < 1e-6)
    if arm.task == "pose":
        R = np.swapaxes(T2[..., :3, :3], -1, -2) @ T[:, None, :3, :3]
        ang = np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
        assert np.all(ang[good] < 1e-6)
    recovered = np.any(good & (np.abs(_wrap(Q - q[:, None])).max(-1) < 1e-6), axis=1)
    assert recovered.all()


def test_branch_counts_match_the_mechanisms():
    T = rr_arm().forward(np.array([[0.5, 1.0]]))[0]
    assert (rr_arm().inverse(T)[1] == Code.OK).sum() == 2  # elbow up / down
    T = shoulder3().forward(np.array([[0.3, 0.6, -1.0]]))[0]
    assert (shoulder3().inverse(T)[1] == Code.OK).sum() == 4  # front/back x elbow
    T = ur5e().forward(np.array([[0.3, -1.0, 1.2, -0.5, 0.8, 0.4]]))[0]
    assert (ur5e().inverse(T)[1] == Code.OK).sum() == 8


def test_ur5e_fk_matches_the_independent_ur_kin_port():
    path = Path("/home/bam/bam_ws/src/core/mdp/agent/actor/trajectory_generation/ur_kinematics/src")
    if not path.exists():
        pytest.skip("bam_ws ur_kinematics oracle not on this machine")
    import sys

    sys.path.insert(0, str(path))
    from ur_kinematics.ur_kinematics import ur_forward_kinematics

    from robot_reach.robots.spatial import UR5E_DH

    params = [UR5E_DH[k] for k in ("d1", "a2", "a3", "d4", "d5", "d6")]
    q = sample_joints(ur5e(), 50, seed=9)
    mine = ur5e().forward(q)[0][:, :3, 3]
    theirs = np.array([np.array(ur_forward_kinematics(list(row), params))[:3, 3] for row in q])
    # ur_kin.cpp's base is the DH base turned 180 deg about z
    assert np.allclose(mine, theirs * [-1, -1, 1], atol=1e-9)


def test_reach_map_codes_are_honest():
    arm = rr_arm()
    positions = Box((0.0, 0.0, 0.1), (1.4, 0.0, 1.0)).grid(0.05)
    m = reach_map(arm, positions, floor_z=-0.15)
    codes = set(m.code_counts())
    # out of reach, joint limits, the floor and the motors all show up, each with its own code
    assert {"OK", "NO_SOLUTION", "JOINT_LIMIT", "ENVIRONMENT_COLLISION", "TORQUE_LIMIT"} <= codes  # the floor is the environment
    far = np.linalg.norm(positions[:, [0, 2]], axis=1) > 0.55 + 1e-6
    assert np.all(m.code[far][..., 0] == Code.NO_SOLUTION)
    # every stored solution is OK and inside the limits
    assert np.all(m.code.reshape(-1)[m.sol_index] == Code.OK)
    assert np.all(arm.within_limits(m.sol_q.astype(float), tol=1e-5))


def test_workspaces_nest_and_come_from_one_map():
    arm = ur5e()
    positions = Box((0.35, 0.0, 0.35), (0.6, 0.6, 0.4)).grid(0.2)
    m = reach_map(arm, positions, OrientationSet.sample(tilt_step_deg=60, n_rolls=4), floor_z=0.0)
    reach, four, hemi, dex = (m.workspace(k) for k in ("reachable", "four_dof", "hemisphere", "dexterous"))
    assert np.all(dex <= hemi) and np.all(hemi <= four) and np.all(four <= reach)
    lv = m.levels()
    assert set(np.unique(lv)) <= {0, 3, 4, 5, 6}
    assert np.array_equal(lv >= 6, dex)


def test_position_only_arms_claim_no_orientation_workspace():
    m = reach_map(shoulder3(), Box((0.3, 0.0, 0.3), (0.2, 0.2, 0.2)).grid(0.1))
    assert m.n_orientations == 1
    assert not m.workspace("dexterous").any()
    assert set(np.unique(m.levels())) <= {0, 3}


def test_duplicate_branches_count_once():
    # fully stretched RR: both elbow branches are the same joint vector
    arm = rr_arm(lower_deg=(-90, -150), upper_deg=(180, 150))
    m = reach_map(arm, np.array([[0.0, 0.0, 0.55]]), singular_tol=0.0)
    assert m.n_solutions[0, 0] == 1


def test_save_load_round_trip_rebuilds_the_arm(tmp_path):
    res = rr.analyze(rr_arm(), floor_z=-0.15)
    res.save(tmp_path / "rr")
    back = rr.load(tmp_path / "rr")
    assert back.arm is not None and back.arm.name == "rr"
    assert np.array_equal(back.reach.code, res.reach.code)
    assert np.allclose(back.reach.sol_q, res.reach.sol_q)
    assert back.summary()["reach"] == res.summary()["reach"]
    doc = json.loads((tmp_path / "rr" / "result.json").read_text())
    assert doc["schema"] == "robot_reach.result.v3"


def test_v1_compatible_fields():
    """robot_trajectories duck-types bam_reach v1: joint_names, joint_solutions (P,O,J), valid_mask (P,O)."""
    m = reach_map(rr_arm(), Box((0.2, 0.0, 0.2), (0.4, 0.0, 0.4)).grid(0.1))
    assert m.joint_solutions.shape == (m.n_positions, m.n_orientations, 2)
    assert np.array_equal(np.isfinite(m.joint_solutions[..., 0]), m.valid_mask)


def test_joint_maps_show_the_mechanism():
    fb = joint_map(five_bar(), (0, 1), n=91)
    assert "FK_FAIL" in fb.code_counts()  # the five-bar's "cannot close" island
    assert "SINGULAR" in fb.code_counts()
    line = joint_map(pendulum(), (0,), n=181)
    assert line.torque is not None
    assert {"OK", "TORQUE_LIMIT"} == set(line.code_counts())
    sweep = joint_map(shoulder3(), (1, 2), n=21, mode="sweep", n_sweep=8, floor_z=0.0)
    assert sweep.ok_fraction.shape == (21, 21) and 0 < sweep.ok_fraction.mean() < 1


def test_from_scalar_wraps_one_pose_functions():
    L1, L2 = 0.3, 0.25

    def fk(q):
        x = L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1])
        z = L1 * np.sin(q[0]) + L2 * np.sin(q[0] + q[1])
        T = np.eye(4)
        T[0, 3], T[2, 3] = x, z
        return T

    def ik(T):
        x, z = T[0, 3], T[2, 3]
        c2 = (x * x + z * z - L1 * L1 - L2 * L2) / (2 * L1 * L2)
        if abs(c2) > 1:
            return []
        out = []
        for q2 in (np.arccos(c2), -np.arccos(c2)):
            q1 = np.arctan2(z, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
            out.append(np.array([q1, q2]))
        return out

    arm = rr.Arm.from_scalar("scalar_rr", ("a", "b"), (-np.pi, -np.pi), (np.pi, np.pi), fk, ik, task="position")
    m = reach_map(arm, np.array([[0.3, 0.0, 0.2], [2.0, 0.0, 0.0]]))
    assert m.n_solutions[0, 0] == 2 and m.n_solutions[1, 0] == 0


def test_analyze_skips_the_grid_for_a_curve_workspace():
    res = rr.analyze(pendulum())
    assert res.reach is None and len(res.joint_maps) == 1


# --------------------------------------------------------------------------- optional geometry hooks
def test_surface_poses_meshes_and_falls_back_to_the_skeleton():
    from robot_reach import ArmMesh
    from robot_reach.robots import rr as rr_arm

    base = rr_arm()
    q = np.array([[0.3, -0.6]])
    # no meshes: points every <= 5 mm along the skeleton, ends included
    pts = base.surface(q)[0]
    chain = base.chain(q)[0]
    assert np.allclose(pts[0], chain[0]) and np.allclose(pts[-1], chain[-1])
    assert np.max(np.linalg.norm(np.diff(pts, axis=0), axis=1)) <= 0.005 + 1e-12

    # one cube per link, posed at the link's start: surface = the cube's vertices moved by that pose
    cube = np.array([[x, y, z] for x in (0, 0.02) for y in (0, 0.02) for z in (0, 0.02)], float)

    def mesh_poses(Q):
        c = base.chain(Q)
        T = np.tile(np.eye(4), (len(Q), 2, 1, 1))
        T[:, 0, :3, 3], T[:, 1, :3, 3] = c[:, 0], c[:, 1]
        return T

    import dataclasses

    arm = dataclasses.replace(base, meshes=(ArmMesh("upper", cube), ArmMesh("fore", cube, draw=False), ArmMesh("cad", cube, measure=False)),
                              mesh_poses=lambda Q: np.concatenate([mesh_poses(Q), mesh_poses(Q)[:, :1]], axis=1))
    got = arm.surface(q)[0]
    assert got.shape == (16, 3)  # the unmeasured third mesh never counts
    assert np.allclose(got[:8], cube + chain[0]) and np.allclose(got[8:], cube + chain[1])
    assert np.allclose(arm.surface(q, parts=("fore",))[0], cube + chain[1])
    assert arm.surface(q, max_per_mesh=2).shape == (1, 4, 3)
    assert arm.surface(q, parts=("nothing",)).shape == (1, 0, 3)


def test_drive_limits_and_friction_are_optional_per_joint_fields():
    import dataclasses

    from robot_reach.robots import rr as rr_arm

    arm = rr_arm()
    assert arm.velocity_limits is None and arm.acceleration_limits is None and arm.friction is None
    fast = dataclasses.replace(arm, velocity_limits=(2.0, 3.0), acceleration_limits=(8.0, 9.0), friction=(0.1, 0.05))
    assert fast.velocity_limits == (2.0, 3.0) and fast.friction[1] == 0.05 and fast.dof == 2
