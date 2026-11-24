"""The downstream workspace API (src/robot_reach/workspace.py) on a real UR5e result.

Every pose any happy path hands out is FK-checked against the arm (< 1e-3 mm, < 1e-3 deg)."""

from __future__ import annotations

import json

import numpy as np
import pytest

import robot_reach as rr
from robot_reach.robots import ur5e

BELT = rr.Box(center=(0.5, 0.0, 0.2), size=(0.4, 0.6, 0.2))


@pytest.fixture(scope="module")
def result():
    return rr.analyze(ur5e(), steps_across=8, joint_axes=[])


@pytest.fixture()
def ws(result):
    return result.workspace()


def fk_check(arm, s: rr.PoseSamples) -> None:
    T, ok = arm.forward(s.q.reshape(-1, s.q.shape[-1]))
    want = s.pose.reshape(-1, 4, 4)
    assert np.all(ok)
    pos_mm = np.linalg.norm(T[:, :3, 3] - want[:, :3, 3], axis=1) * 1000
    assert pos_mm.max() < 1e-3, pos_mm.max()
    cos = (np.einsum("kij,kij->k", T[:, :3, :3], want[:, :3, :3]) - 1) / 2
    assert np.degrees(np.arccos(np.clip(cos, -1, 1))).max() < 1e-3


def test_regions_nest_and_match_reach_map(ws, result):
    m = result.reach
    assert list(ws.regions) == ["full", "reachable", "top_down", "hemisphere", "dexterous"]
    for name, kind in [("reachable", "reachable"), ("top_down", "four_dof"), ("hemisphere", "hemisphere"), ("dexterous", "dexterous")]:
        assert np.array_equal(ws.region(name).voxel_mask, m.workspace(kind)), name
    sizes = [len(ws.region(n)) for n in ws.regions]
    assert sizes == sorted(sizes, reverse=True) and sizes[-1] > 0
    assert list(ws.reachable.regions) == ["top_down"] and ws.dexterous.parent == "hemisphere"


def test_alias_is_the_registry_lookup(ws):
    for name in ws.regions:
        assert getattr(ws, name) == ws.regions[name] == ws.region(name)
    assert ws.hemisphere.dexterous == ws.hemisphere.region("dexterous") == ws.dexterous  # a sub-region, by attribute


def test_dexterous_sample(ws, result):
    s = ws.dexterous.sample(50, np.random.default_rng(0))
    assert s.shape == (50,) and s.pose.shape == (50, 4, 4) and s.q.shape == (50, 6)
    assert ws.dexterous.contains(s.pose[:, :3, 3]).all()
    fk_check(result.arm, s)


def test_belt_pairs_elbow_up(ws, result):
    region = ws.hemisphere.inside(BELT).only(elbow="up")
    s = region.sample_pairs(40, np.random.default_rng(1))
    assert s.shape == (40, 2)
    assert BELT.contains(s.pose[..., :3, 3]).all()
    assert all("elbow up" in s.configuration_name(i, j) for i in range(40) for j in range(2))
    assert np.all(s.configuration[:, 0] == s.configuration[:, 1])  # same IK configuration A -> B
    fk_check(result.arm, s)
    w = region.sample_waypoints(10, 5, np.random.default_rng(2))
    assert w.shape == (10, 5)
    fk_check(result.arm, w)


def test_only_rejudges_membership_and_commutes(ws, result):
    up = ws.hemisphere.only(elbow="up")
    assert up == ws.only(elbow="up").hemisphere
    assert np.array_equal(up.voxel_mask, result.reach.filtered(ws.configurations.allowed(elbow="up")).workspace("hemisphere"))
    one = ws.hemisphere.only(shoulder="front", elbow="up", wrist="J5+")
    assert len(one) < len(up) <= len(ws.hemisphere)  # one IK configuration cannot cover every orientation everywhere
    assert set(up.count_by_configuration()) <= {n for n in ws.only(elbow="up").regions["full"].count_by_configuration()}
    with pytest.raises(ValueError):
        ws.hemisphere.only(elbw="up")  # a typo never silently allows everything


def test_registered_custom_regions(ws, result):
    cone = ws.regions.register("steep", ws.hemisphere.within(30))
    assert ws.steep == cone == ws.regions["steep"] and ws.regions.parent("steep") == "hemisphere"
    assert "steep" in ws.hemisphere.regions and ws.hemisphere.steep == cone
    assert len(ws.dexterous) <= len(cone) <= len(ws.top_down)
    s = ws.steep.sample(30, np.random.default_rng(3))
    tilt = np.degrees(np.arccos(np.clip(-s.pose[:, 2, 2], -1, 1)))
    assert tilt.max() <= 30 + 1e-6
    fk_check(result.arm, s)
    zone = ws.regions.register("belt_pick", ws.top_down.inside(BELT), parent="top_down")
    assert ws.belt_pick == zone and len(zone) == int((BELT.contains(ws.positions) & ws.top_down.voxel_mask).sum())
    fk_check(result.arm, ws.belt_pick.only(elbow="up").sample(10, np.random.default_rng(4)))
    with pytest.raises(ValueError):
        ws.regions.register("sample", cone)  # a method name is never shadowed


def test_cylinder_zone_contains_and_nearest(ws, result):
    cyl = rr.Cylinder(center=(0.0, 0.4, 0.3), radius=0.25, height=0.3)
    r = ws.reachable.inside(cyl)
    assert cyl.contains(r.positions).all() and len(r) > 0
    assert r.contains(r.positions).all() and not r.contains([[5.0, 5.0, 5.0]]).any()
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])  # straight down
    T[:3, 3] = (0.62, 0.05, 0.47)
    n = ws.top_down.nearest_ok_pose(T)
    assert n.angle_deg < 1e-6 and n.distance_m < ws.voxel_size
    fk_check(result.arm, rr.PoseSamples("ur5e", "base_link", ws.joint_names, n.pose[None], n.q[None], np.zeros(1, np.uint8), ("",)))


def test_spec_json_round_trip(ws):
    region = ws.hemisphere.inside(BELT).only(elbow="up").within(60)
    spec = json.loads(json.dumps(region.spec()))
    assert spec["region"] == "hemisphere" and spec["within_deg"] == 60 and len(spec["configurations"]) == 4
    assert ws.from_spec(spec) == region


def test_to_sampler_save_load(ws, result, tmp_path):
    region = ws.hemisphere.inside(BELT).only(elbow="up")
    s = rr.PoseSampler.load(region.to_sampler().save(tmp_path / "belt.poses.npz"))
    assert len(s) == len(np.unique(ws.solution_pose[region._rows]))
    fk_check(result.arm, s.sample(20, np.random.default_rng(5)))
    fk_check(result.arm, s.sample_pairs(20, np.random.default_rng(6)))


def test_as_reach_map_like(ws, result):
    region = ws.hemisphere.only(elbow="up")
    m = region.as_reach_map_like()
    P, O = len(ws.positions), len(ws.rotations)
    assert m.joint_solutions.shape == (P, O, 6) and m.valid_mask.shape == (P, O) and m.reach_scores.shape == (P,)
    assert not m.valid_mask[~region.voxel_mask].any()
    assert np.isnan(m.joint_solutions[~m.valid_mask]).all() and not np.isnan(m.joint_solutions[m.valid_mask]).any()
    p, o = np.argwhere(m.valid_mask)[:20].T
    T, _ = result.arm.forward(m.joint_solutions[p, o])
    assert np.abs(T[:, :3, 3] - ws.positions[p]).max() * 1000 < 1e-3


def test_robot_msgs_at_the_edge(ws):
    s = ws.dexterous.sample(2, np.random.default_rng(7))
    poses, joints = s.poses(), s.joint_states()
    assert np.allclose(poses[0].to_matrix(), s.pose[0], atol=1e-9)
    assert joints[1].name == ws.joint_names and np.allclose(joints[1].position, s.q[1])
